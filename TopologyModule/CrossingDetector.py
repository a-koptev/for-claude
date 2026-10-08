"""Детектор переходов между камерами по движению и двум границам bbox."""
from dataclasses import dataclass
from typing import Iterable, Optional

from Config import Config
from TopologyModule.CameraNetwork import Camera
from TopologyModule.TransitionLine import MIN_MOTION_ALIGNMENT, Role, TransitionLine


@dataclass(frozen=True)
class TransitionEvent:
    camera_id: int
    line_id: int
    track_id: int
    role: Role
    peer_camera: int
    frame_id: int
    motion: tuple[float, float]

    @property
    def from_camera(self) -> int:
        return self.camera_id if self.role is Role.EXIT else self.peer_camera

    @property
    def to_camera(self) -> int:
        return self.peer_camera if self.role is Role.EXIT else self.camera_id


@dataclass
class _CrossingState:
    """Состояние одного track ID относительно одной линии."""
    fired: bool = False
    missed: int = 0
    far_frames: int = 0


def _center(bbox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


class CrossingDetector:
    def __init__(
            self,
            camera: Camera,
            min_move_px: Optional[float] = None,
            release_frames: Optional[int] = None,
            release_margin_px: Optional[float] = None,
            forget_frames: Optional[int] = None,
    ):
        self.camera = camera
        # Параметр оставлен для совместимости конструктора старого кода.
        # Факт движения определяется по разнице центров двух последовательных bbox.
        self.min_move_px = (
            Config.TRANSITION_MIN_MOVE_PX if min_move_px is None else min_move_px
        )
        self.release_frames = (
            Config.TRANSITION_RELEASE_FRAMES if release_frames is None else release_frames
        )
        self.release_margin_px = (
            Config.TRANSITION_RELEASE_MARGIN_PX if release_margin_px is None else release_margin_px
        )
        self.forget_frames = Config.TRANSITION_FORGET_FRAMES if forget_frames is None else forget_frames

        self._prev_center: dict[int, tuple[float, float]] = {}
        self._prev_bbox: dict[int, object] = {}
        self._last_seen: dict[int, int] = {}
        self._states: dict[tuple[int, int], _CrossingState] = {}
        self.dropped_tracks: list[int] = []
        self._motion_by_track: dict[int, tuple[float, float]] = {}

    def update(self, frame_id: int, tracks: Iterable[tuple[int, object]]) -> list[TransitionEvent]:
        """tracks - (track_id, bbox xyxy) всех треков кадра."""
        events: list[TransitionEvent] = []
        present = set()

        for track_id, bbox in tracks:
            track_id = int(track_id)
            present.add(track_id)
            center = _center(bbox)
            prev = self._prev_center.get(track_id)
            self._motion_by_track[track_id] = (0.0, 0.0)
            prev_bbox = self._prev_bbox.get(track_id)

            if prev is not None and prev_bbox is not None:
                motion = (center[0] - prev[0], center[1] - prev[1])
                self._motion_by_track[track_id] = motion
                for line in self.camera.lines:
                    event = self._update_crossing(line, track_id, bbox, center, prev, prev_bbox, frame_id)
                    if event is not None:
                        events.append(event)

            self._prev_center[track_id] = center
            self._prev_bbox[track_id] = bbox
            self._last_seen[track_id] = frame_id

        self._release_missing(present)
        self._forget_old(frame_id)
        return events

    def is_moving(self, track_id: int) -> bool:
        """True when the latest bbox-center displacement passes the movement filter."""
        motion = self._motion_by_track.get(int(track_id), (0.0, 0.0))
        return (motion[0] ** 2 + motion[1] ** 2) ** 0.5 >= self.min_move_px

    def _update_crossing(
            self,
            line: TransitionLine,
            track_id: int,
            bbox,
            center: tuple[float, float],
            prev: tuple[float, float],
            prev_bbox,
            frame_id: int,
    ):
        key = (track_id, line.line_id)
        state = self._states.setdefault(key, _CrossingState())

        # После события ждём, пока центр автомобиля действительно отойдёт от
        # линии на заданное расстояние. Это не условие самого перехода, а только
        # защита от повторных событий из-за дрожания трека около линии.
        current_distance = abs(line.signed_distance(center))
        if state.fired:
            if current_distance >= self.release_margin_px:
                state.far_frames += 1
                if state.far_frames >= self.release_frames:
                    state.fired = False
                    state.far_frames = 0
            else:
                state.far_frames = 0
            return None

        motion = (center[0] - prev[0], center[1] - prev[1])

        # Сначала проверяем геометрию текущего bbox. Важен сам факт того, что
        # линия находится внутри bbox: для горизонтальной линии это означает,
        # что верхняя и нижняя границы bbox оказались по разные стороны линии.
        #
        # Нельзя сначала требовать большой сдвиг центра или высокий alignment:
        # при медленном движении центр может почти не измениться, хотя bbox уже
        # физически пересёк линию. Именно это приводило к пропуску перехода.
        if not line.bbox_straddles_line(bbox, motion):
            return None

        # Направление всё равно обязательно. Берём именно компонент движения
        # поперёк линии: так ENTER/EXIT определяется по фактическому движению,
        # а движение вдоль линии не считается переходом.
        nx, ny = line.normal
        line_length = line.length
        if line_length == 0:
            return None

        normal_projection = (
            motion[0] * nx + motion[1] * ny
        ) / line_length

        # Нужен ненулевой знак движения через линию, но не большой по модулю
        # сдвиг. Это позволяет корректно поймать медленный автомобиль.
        if abs(normal_projection) < 1e-6:
            return None

        state.fired = True
        state.far_frames = 0

        return TransitionEvent(
            camera_id=line.camera_id,
            line_id=line.line_id,
            track_id=track_id,
            role=line.role_of_motion(motion),
            peer_camera=line.peer_camera,
            frame_id=frame_id,
            motion=motion,
        )

    def _release_missing(self, present: set) -> None:
        for (track_id, line_id), state in list(self._states.items()):
            if track_id not in present:
                state.missed += 1
                if state.missed >= self.release_frames:
                    del self._states[(track_id, line_id)]
            else:
                state.missed = 0

    def _forget_old(self, frame_id: int) -> None:
        self.dropped_tracks = [
            track_id for track_id, seen in self._last_seen.items()
            if frame_id - seen >= self.forget_frames
        ]
        for track_id in self.dropped_tracks:
            self._last_seen.pop(track_id, None)
            self._prev_center.pop(track_id, None)
            self._prev_bbox.pop(track_id, None)
            self._motion_by_track.pop(track_id, None)
            for key in [k for k in self._states if k[0] == track_id]:
                del self._states[key]

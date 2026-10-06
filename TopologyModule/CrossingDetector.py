"""
Детектор переходов между камерами по движению bbox относительно линии.

В обычном случае событие возникает, когда центр bbox пересекает линию.
Если YOLO начал детектировать автомобиль поздно и центр уже находится за
линией, используется положение задней относительно движения грани bbox с
допуском TRANSITION_LATE_EDGE_TOLERANCE_PX.

Само касание линии bbox не является достаточным условием перехода.
"""
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
        self.min_move_px = min_move_px
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

    def update(self, frame_id: int, tracks: Iterable[tuple[int, object]]) -> list[TransitionEvent]:
        """tracks - (track_id, bbox xyxy) всех треков кадра."""
        events: list[TransitionEvent] = []
        present = set()

        for track_id, bbox in tracks:
            track_id = int(track_id)
            present.add(track_id)
            center = _center(bbox)
            prev = self._prev_center.get(track_id)
            prev_bbox = self._prev_bbox.get(track_id)

            if prev is not None and prev_bbox is not None:
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
        if line.alignment(motion) < MIN_MOTION_ALIGNMENT:
            return None

        center_crossed = line.center_crosses(prev, center)

        # YOLO мог создать bbox уже после линии. В этом случае смотрим на
        # заднюю относительно движения грань bbox.
        late_detection = False
        if not center_crossed:
            prev_side = line.signed_distance(prev)
            current_side = line.signed_distance(center)
            if prev_side * current_side > 0 and current_side != 0:
                rear_edge_distance = line.signed_bbox_rear_edge_distance(bbox, motion)
                motion_side = line._dot_normal(motion)
                late_detection = (
                    current_side * motion_side > 0
                    and rear_edge_distance <= Config.TRANSITION_LATE_EDGE_TOLERANCE_PX
                )

        if not center_crossed and not late_detection:
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
            for key in [k for k in self._states if k[0] == track_id]:
                del self._states[key]

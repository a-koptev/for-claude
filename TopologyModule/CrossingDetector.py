"""
Детектор пересечения линий перехода на одной камере.

Событие возникает, когда bbox трека коснулся линии (не его край, а сам bbox). Направление
берётся из смещения центра bbox относительно положения ДО касания (предыдущий bbox),
поэтому два авто, едущих в разные стороны через одну линию, получают разные события.

Ни куда ехать, ни что делать дальше детектор не решает: он только говорит
"трек T пересёк линию L в сторону EXIT/ENTER (соседняя камера N)".
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
    role: Role                      # EXIT: уезжает с camera_id в peer_camera; ENTER: приехал из peer_camera
    peer_camera: int
    frame_id: int
    motion: tuple[float, float]     # смещение центра bbox, по которому определено направление

    @property
    def from_camera(self) -> int:
        return self.camera_id if self.role is Role.EXIT else self.peer_camera

    @property
    def to_camera(self) -> int:
        return self.peer_camera if self.role is Role.EXIT else self.camera_id


@dataclass
class _Contact:
    """Одно касание линии одним треком: от первого кадра касания до его завершения."""
    start_center: tuple[float, float]   # положение центра перед касанием (или в первом кадре, если авто появилось на линии)
    missed: int = 0                     # кадров подряд без касания
    fired: bool = False                 # событие по этому касанию уже выдано


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
        self.min_move_px = Config.TRANSITION_MIN_MOVE_PX if min_move_px is None else min_move_px
        self.release_frames = Config.TRANSITION_RELEASE_FRAMES if release_frames is None else release_frames
        self.release_margin_px = (
            Config.TRANSITION_RELEASE_MARGIN_PX if release_margin_px is None else release_margin_px
        )
        self.forget_frames = Config.TRANSITION_FORGET_FRAMES if forget_frames is None else forget_frames

        self._prev_center: dict[int, tuple[float, float]] = {}   # предыдущий bbox трека (его центр)
        self._last_seen: dict[int, int] = {}
        self._contacts: dict[tuple[int, int], _Contact] = {}     # (track_id, line_id) -> касание
        # Треки, забытые при последнем update (по ним можно чистить внешнее состояние)
        self.dropped_tracks: list[int] = []

    def update(self, frame_id: int, tracks: Iterable[tuple[int, object]]) -> list[TransitionEvent]:
        """tracks - (track_id, bbox xyxy) всех треков кадра. Возвращает новые события пересечения."""
        events: list[TransitionEvent] = []
        present = set()

        for track_id, bbox in tracks:
            track_id = int(track_id)
            present.add(track_id)
            center = _center(bbox)
            prev = self._prev_center.get(track_id)

            for line in self.camera.lines:
                event = self._update_contact(line, track_id, bbox, center, prev, frame_id)
                if event is not None:
                    events.append(event)

            self._prev_center[track_id] = center
            self._last_seen[track_id] = frame_id

        self._release_missing(present)
        self._forget_old(frame_id)
        return events

    def _update_contact(self, line: TransitionLine, track_id, bbox, center, prev, frame_id):
        key = (track_id, line.line_id)
        contact = self._contacts.get(key)

        touches = line.touches(bbox)
        if not touches:
            if contact is not None:
                # Касание заканчивается, только когда bbox ушёл от линии с запасом
                # и так продержался release_frames кадров подряд
                if self._near(line, bbox):
                    contact.missed = 0
                else:
                    contact.missed += 1
                    if contact.missed >= self.release_frames:
                        del self._contacts[key]
            return None

        if contact is None:
            # Первый кадр касания. Если трек появился сразу на линии, предыдущего bbox нет:
            # направление определится по накопленному смещению в следующих кадрах
            contact = _Contact(start_center=prev if prev is not None else center)
            self._contacts[key] = contact
        contact.missed = 0
        if contact.fired:
            return None

        motion = (center[0] - contact.start_center[0], center[1] - contact.start_center[1])
        if (motion[0] ** 2 + motion[1] ** 2) ** 0.5 < self.min_move_px:
            return None
        if line.alignment(motion) < MIN_MOTION_ALIGNMENT:
            return None   # едет вдоль линии: ждём, пока направление станет ясным

        contact.fired = True
        return TransitionEvent(
            camera_id=line.camera_id,
            line_id=line.line_id,
            track_id=track_id,
            role=line.role_of_motion(motion),
            peer_camera=line.peer_camera,
            frame_id=frame_id,
            motion=motion,
        )

    def _near(self, line: TransitionLine, bbox) -> bool:
        m = self.release_margin_px
        x1, y1, x2, y2 = bbox
        return line.touches((x1 - m, y1 - m, x2 + m, y2 + m))

    def _release_missing(self, present: set) -> None:
        """Треки, которых нет в кадре, тоже 'не касаются' линии."""
        for (track_id, line_id), contact in list(self._contacts.items()):
            if track_id in present:
                continue
            contact.missed += 1
            if contact.missed >= self.release_frames:
                del self._contacts[(track_id, line_id)]

    def _forget_old(self, frame_id: int) -> None:
        self.dropped_tracks = [
            track_id for track_id, seen in self._last_seen.items()
            if frame_id - seen >= self.forget_frames
        ]
        for track_id in self.dropped_tracks:
            self._last_seen.pop(track_id, None)
            self._prev_center.pop(track_id, None)
            for key in [k for k in self._contacts if k[0] == track_id]:
                del self._contacts[key]

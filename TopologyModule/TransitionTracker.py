"""
Превращает пересечения линий в команды для главного трекера.

Два сценария:

1. EXIT. Авто на камере A пересёк линию к камере B в сторону выезда:
   команда SEARCH_ON_CAMERA - "переключись на камеру B и ищи там автомобиль трека T (камера A)".

2. ENTER. Авто на камере B пересёк линию, ведущую к камере A, в сторону въезда:
   команда MARK_ARRIVED - "трек U на камере B приехал с камеры A". Это пропуск к сравнению:
   когда ищут машину, уехавшую с A, сравнивать нужно только треки, приехавшие с A
   (is_eligible / arrived_from). Не двигавшиеся и пересёкшие другую линию отсеиваются.

Главный трекер пока не написан: команды просто возвращаются из update() и отдаются
в необязательный колбэк on_command. Дальше они подключаются к TrackletManager.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Optional

from TopologyModule.CameraNetwork import CameraNetwork
from TopologyModule.CrossingDetector import CrossingDetector, TransitionEvent
from TopologyModule.TransitionLine import Role


class CommandType(Enum):
    SEARCH_ON_CAMERA = "SEARCH_ON_CAMERA"
    MARK_ARRIVED = "MARK_ARRIVED"


@dataclass(frozen=True)
class TransitionCommand:
    """
    SEARCH_ON_CAMERA: на камере camera_id искать авто, уехавшее с source_camera (его трек там - track_id).
    MARK_ARRIVED:     трек track_id на камере camera_id приехал с камеры source_camera.
    """
    type: CommandType
    camera_id: int
    source_camera: int
    track_id: int
    frame_id: int
    event: TransitionEvent

    def __str__(self) -> str:
        if self.type is CommandType.SEARCH_ON_CAMERA:
            return (f"SEARCH_ON_CAMERA: искать на камере {self.camera_id} авто с камеры "
                    f"{self.source_camera} (трек {self.track_id})")
        return (f"MARK_ARRIVED: трек {self.track_id} на камере {self.camera_id} "
                f"приехал с камеры {self.source_camera}")


class TransitionTracker:
    """Одна камера: детектор пересечений + память, какой трек откуда приехал."""

    def __init__(
            self,
            network: CameraNetwork,
            camera_id: int,
            on_command: Optional[Callable[[TransitionCommand], None]] = None,
            detector: Optional[CrossingDetector] = None,
    ):
        self.camera_id = camera_id
        self.detector = detector or CrossingDetector(network.camera(camera_id))
        self.on_command = on_command
        self._arrived_from: dict[int, int] = {}

    def update(self, frame_id: int, tracks: Iterable[tuple[int, object]]) -> list[TransitionCommand]:
        """tracks - (track_id, bbox xyxy) всех треков камеры в этом кадре."""
        commands = []
        for event in self.detector.update(frame_id, tracks):
            command = self._command_for(event)
            if event.role is Role.ENTER:
                self._arrived_from[event.track_id] = event.peer_camera
            commands.append(command)
            if self.on_command is not None:
                self.on_command(command)

        for track_id in self.detector.dropped_tracks:
            self._arrived_from.pop(track_id, None)
        return commands

    @staticmethod
    def _command_for(event: TransitionEvent) -> TransitionCommand:
        if event.role is Role.EXIT:
            return TransitionCommand(
                CommandType.SEARCH_ON_CAMERA, camera_id=event.peer_camera,
                source_camera=event.camera_id, track_id=event.track_id,
                frame_id=event.frame_id, event=event,
            )
        return TransitionCommand(
            CommandType.MARK_ARRIVED, camera_id=event.camera_id,
            source_camera=event.peer_camera, track_id=event.track_id,
            frame_id=event.frame_id, event=event,
        )

    # ------------------------------------------------------------ фильтр кандидатов

    def arrived_from(self, track_id: int) -> Optional[int]:
        """С какой камеры приехал трек (None, если не пересекал входную линию)."""
        return self._arrived_from.get(track_id)

    def is_eligible(self, track_id: int, from_camera: int) -> bool:
        """Можно ли сравнивать трек с авто, уехавшим с from_camera: он должен приехать оттуда."""
        return self._arrived_from.get(track_id) == from_camera

    def tracks_arrived_from(self, from_camera: int) -> list[int]:
        return sorted(t for t, cam in self._arrived_from.items() if cam == from_camera)


def create_trackers(
        network: CameraNetwork,
        on_command: Optional[Callable[[TransitionCommand], None]] = None,
) -> dict[int, TransitionTracker]:
    """По одному TransitionTracker на каждую камеру сети."""
    return {cid: TransitionTracker(network, cid, on_command) for cid in network.camera_ids}

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from TopologyModule.TransitionLine import (
    MIN_MOTION_ALIGNMENT,
    Direction,
    TransitionLine,
)

# Система координат линий. Tracking.py приводит кадр к 1280x720 до трекинга.
DEFAULT_FRAME_SIZE = (1280, 720)


class TopologyError(ValueError):
    """Ошибки конфигурации топологии. Собираются все сразу, а не только первая."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        details = "\n".join(f"  - {problem}" for problem in self.problems)
        super().__init__(f"Ошибки в конфигурации топологии:\n{details}")


@dataclass(frozen=True)
class Camera:
    id: int
    name: str
    frame_size: tuple[int, int]
    lines: tuple[TransitionLine, ...] = ()


@dataclass(frozen=True)
class RouteSegment:
    """Участок маршрута между камерами. preserve_order: автомобили здесь не меняются местами."""
    from_camera: int
    to_camera: int
    preserve_order: bool = False


class CameraNetwork:
    """
    Сеть камер: камеры, их линии перехода и допустимые переходы между камерами.

    Линия на камере A с соседом B двусторонняя и задаёт сразу два перехода:
    A -> B (EXIT, авто уезжает с A) и B -> A (ENTER, авто приехал на A из B).
    Именно по этим переходам потом строится candidate generation.
    Описывать линию на обеих камерах не обязательно, но желательно: так видно
    пересечение с обеих сторон; если линия есть только с одной, будет предупреждение.
    """

    def __init__(
            self,
            cameras: dict[int, Camera],
            segments: Optional[dict[tuple[int, int], RouteSegment]] = None
    ):
        self._cameras = dict(cameras)
        self._segments = dict(segments or {})
        self._exit_pairs, self._enter_pairs = self._collect_pairs(self._cameras)
        self._pairs = frozenset(self._exit_pairs | self._enter_pairs)
        self.warnings: list[str] = self._build_warnings()

    # ------------------------------------------------------------------ загрузка

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "CameraNetwork":
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = yaml.safe_load(file)
        except yaml.YAMLError as error:
            raise TopologyError([f"Файл {path} не читается как YAML: {error}"]) from error
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Any) -> "CameraNetwork":
        problems: list[str] = []

        if not isinstance(data, dict):
            raise TopologyError(["Корень конфигурации должен быть словарём (frame_size, cameras, segments)"])

        default_size = _parse_size(data.get("frame_size", DEFAULT_FRAME_SIZE), "frame_size", problems)
        default_size = default_size or DEFAULT_FRAME_SIZE

        raw_cameras = data.get("cameras")
        if not isinstance(raw_cameras, list) or not raw_cameras:
            problems.append("cameras: нужен непустой список камер")
            raw_cameras = []

        # Номера камер нужны заранее: правила линий ссылаются на камеры, описанные ниже по файлу
        known_ids = {
            raw["id"] for raw in raw_cameras
            if isinstance(raw, dict) and _is_int(raw.get("id"))
        }

        cameras: dict[int, Camera] = {}
        for index, raw in enumerate(raw_cameras):
            camera = _parse_camera(raw, index, default_size, known_ids, problems)
            if camera is None:
                continue
            if camera.id in cameras:
                problems.append(f"Камера {camera.id}: id встречается больше одного раза")
                continue
            cameras[camera.id] = camera

        segments = _parse_segments(data.get("segments"), cameras, problems)

        if problems:
            raise TopologyError(problems)
        return cls(cameras, segments)

    # --------------------------------------------------------------------- запросы

    @property
    def camera_ids(self) -> list[int]:
        return sorted(self._cameras)

    def camera(self, camera_id: int) -> Camera:
        if camera_id not in self._cameras:
            raise KeyError(f"Камеры {camera_id} нет в топологии")
        return self._cameras[camera_id]

    def lines(self, camera_id: int) -> tuple[TransitionLine, ...]:
        return self.camera(camera_id).lines

    def line(self, camera_id: int, line_id: int) -> TransitionLine:
        for line in self.lines(camera_id):
            if line.line_id == line_id:
                return line
        raise KeyError(f"Линии {line_id} нет на камере {camera_id}")

    @property
    def transitions(self) -> frozenset[tuple[int, int]]:
        """Все допустимые направленные переходы (from_camera, to_camera)."""
        return self._pairs

    def has_transition(self, from_camera: int, to_camera: int) -> bool:
        return (from_camera, to_camera) in self._pairs

    def next_cameras(self, camera_id: int) -> list[int]:
        """Куда можно попасть с камеры."""
        return sorted(to for frm, to in self._pairs if frm == camera_id)

    def previous_cameras(self, camera_id: int) -> list[int]:
        """Откуда можно попасть на камеру."""
        return sorted(frm for frm, to in self._pairs if to == camera_id)

    def lines_to(self, camera_id: int, peer_camera: int) -> tuple[TransitionLine, ...]:
        """Линии камеры camera_id, которые ведут к камере peer_camera."""
        return tuple(line for line in self.lines(camera_id) if line.peer_camera == peer_camera)

    def segment(self, from_camera: int, to_camera: int) -> Optional[RouteSegment]:
        """Участок маршрута; None, если такого перехода нет. Без явной настройки preserve_order=False."""
        if not self.has_transition(from_camera, to_camera):
            return None
        return self._segments.get((from_camera, to_camera), RouteSegment(from_camera, to_camera))

    # ------------------------------------------------------------------- служебное

    @staticmethod
    def _collect_pairs(cameras: dict[int, Camera]) -> tuple[set, set]:
        exits: set[tuple[int, int]] = set()
        enters: set[tuple[int, int]] = set()
        for camera in cameras.values():
            for line in camera.lines:
                exits.add((camera.id, line.peer_camera))
                enters.add((line.peer_camera, camera.id))
        return exits, enters

    def _build_warnings(self) -> list[str]:
        warnings = []
        for camera_id in self.camera_ids:
            if not self._cameras[camera_id].lines:
                warnings.append(f"Камера {camera_id}: нет ни одной линии перехода")

        # Линия A -> B есть, а на B нет ни одной линии к A: переход виден только с одной стороны
        linked = {(camera.id, line.peer_camera) for camera in self._cameras.values() for line in camera.lines}
        for camera_id, peer in sorted(linked):
            if (peer, camera_id) not in linked:
                warnings.append(
                    f"Камеры {camera_id} и {peer}: линия есть на камере {camera_id}, "
                    f"а на камере {peer} нет линии к камере {camera_id}"
                )

        for from_camera, to_camera in sorted(self._segments):
            if (from_camera, to_camera) not in self._pairs:
                warnings.append(
                    f"Участок {from_camera} -> {to_camera} задан в segments, но линий перехода между "
                    f"этими камерами нет"
                )
        return warnings


# ---------------------------------------------------------------------- разбор YAML

def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _parse_size(raw: Any, label: str, problems: list[str]) -> Optional[tuple[int, int]]:
    if (isinstance(raw, (list, tuple)) and len(raw) == 2
            and all(_is_int(v) and v > 0 for v in raw)):
        return int(raw[0]), int(raw[1])
    problems.append(f"{label}: ожидается [ширина, высота] целыми положительными числами, получено {raw!r}")
    return None


def _parse_point(raw: Any, label: str, size: tuple[int, int], problems: list[str]) -> Optional[tuple[float, float]]:
    if not (isinstance(raw, (list, tuple)) and len(raw) == 2 and all(_is_number(v) for v in raw)):
        problems.append(f"{label}: ожидается [x, y] числами, получено {raw!r}")
        return None
    x, y = float(raw[0]), float(raw[1])
    width, height = size
    if not (0 <= x <= width and 0 <= y <= height):
        problems.append(f"{label}: точка ({x:g}, {y:g}) вне кадра {width}x{height}")
        return None
    return x, y


def _parse_enum(enum_class, raw: Any, label: str, problems: list[str], required: bool = True):
    if raw is None and not required:
        return None
    if isinstance(raw, str):
        try:
            return enum_class(raw.strip().upper())
        except ValueError:
            pass
    allowed = " | ".join(member.value for member in enum_class)
    problems.append(f"{label}: ожидается {allowed}, получено {raw!r}")
    return None


def _parse_line(
        raw: Any,
        camera_id: int,
        size: tuple[int, int],
        known_ids: set[int],
        problems: list[str]
) -> Optional[TransitionLine]:
    if not isinstance(raw, dict):
        problems.append(f"Камера {camera_id}: линия должна быть словарём (id, p1, p2, peer, exit)")
        return None
    line_id = raw.get("id")
    if not _is_int(line_id):
        problems.append(f"Камера {camera_id}: у линии нет целочисленного id (получено {line_id!r})")
        return None

    where = f"Камера {camera_id}, линия {line_id}"
    before = len(problems)
    p1 = _parse_point(raw.get("p1"), f"{where}, p1", size, problems)
    p2 = _parse_point(raw.get("p2"), f"{where}, p2", size, problems)
    if p1 is not None and p1 == p2:
        problems.append(f"{where}: p1 и p2 совпадают, у линии нулевая длина")

    peer = raw.get("peer")
    if not _is_int(peer):
        problems.append(f"{where}, peer: ожидается номер соседней камеры (целое число), получено {peer!r}")
    elif peer == camera_id:
        problems.append(f"{where}: линия ведёт в саму камеру {camera_id}")
    elif peer not in known_ids:
        problems.append(f"{where}: камеры {peer} нет в списке cameras")

    exit_direction = _parse_enum(Direction, raw.get("exit"), f"{where}, exit", problems)

    if len(problems) > before:
        return None
    line = TransitionLine(
        camera_id=camera_id, line_id=line_id, p1=p1, p2=p2,
        peer_camera=peer, exit_direction=exit_direction,
    )

    # Направление выезда должно идти поперёк линии, иначе "туда" и "обратно" не различить
    alignment = line.alignment(exit_direction.vector)
    if alignment < MIN_MOTION_ALIGNMENT:
        problems.append(
            f"{where}: направление выезда {exit_direction.value} почти параллельно линии, "
            f"туда и обратно не различить. Выберите направление поперёк линии"
        )
        return None
    return line


def _parse_camera(
        raw: Any,
        index: int,
        default_size: tuple[int, int],
        known_ids: set[int],
        problems: list[str]
) -> Optional[Camera]:
    if not isinstance(raw, dict):
        problems.append(f"cameras[{index}]: камера должна быть словарём (id, name, lines)")
        return None
    camera_id = raw.get("id")
    if not _is_int(camera_id):
        problems.append(f"cameras[{index}]: у камеры нет целочисленного id (получено {camera_id!r})")
        return None

    size = default_size
    if "frame_size" in raw:
        size = _parse_size(raw["frame_size"], f"Камера {camera_id}, frame_size", problems) or default_size

    raw_lines = raw.get("lines") or []
    if not isinstance(raw_lines, list):
        problems.append(f"Камера {camera_id}: lines должен быть списком")
        raw_lines = []

    lines: list[TransitionLine] = []
    line_ids: set[int] = set()
    for raw_line in raw_lines:
        line = _parse_line(raw_line, camera_id, size, known_ids, problems)
        if line is None:
            continue
        if line.line_id in line_ids:
            problems.append(f"Камера {camera_id}: id линии {line.line_id} встречается больше одного раза")
            continue
        line_ids.add(line.line_id)
        lines.append(line)

    return Camera(
        id=camera_id,
        name=str(raw.get("name") or f"Камера {camera_id}"),
        frame_size=size,
        lines=tuple(lines),
    )


def _parse_segments(raw: Any, cameras: dict[int, Camera], problems: list[str]) -> dict[tuple[int, int], RouteSegment]:
    segments: dict[tuple[int, int], RouteSegment] = {}
    if raw is None:
        return segments
    if not isinstance(raw, list):
        problems.append("segments: ожидается список участков (from, to, preserve_order)")
        return segments

    for index, item in enumerate(raw):
        label = f"segments[{index}]"
        if not isinstance(item, dict):
            problems.append(f"{label}: ожидается словарь с ключами from, to, preserve_order")
            continue
        from_camera, to_camera = item.get("from"), item.get("to")
        preserve_order = item.get("preserve_order", False)
        if not (_is_int(from_camera) and _is_int(to_camera)):
            problems.append(f"{label}: from и to должны быть номерами камер")
            continue
        if not isinstance(preserve_order, bool):
            problems.append(f"{label}: preserve_order должен быть true или false")
            continue
        unknown = [c for c in (from_camera, to_camera) if c not in cameras]
        if unknown:
            problems.append(f"{label}: камеры {unknown} нет в списке cameras")
            continue
        if (from_camera, to_camera) in segments:
            problems.append(f"{label}: участок {from_camera} -> {to_camera} задан больше одного раза")
            continue
        segments[(from_camera, to_camera)] = RouteSegment(from_camera, to_camera, preserve_order)
    return segments

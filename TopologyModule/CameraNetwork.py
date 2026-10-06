import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from TopologyModule.TransitionLine import (
    MIN_MOTION_ALIGNMENT,
    CrossingRule,
    Edge,
    Role,
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

    Переход A -> B существует, если на камере A есть линия с правилом EXIT в B
    или на камере B есть линия с правилом ENTER из A. Именно по этим переходам
    потом строится candidate generation.
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
                for rule in line.rules:
                    if rule.role is Role.EXIT:
                        exits.add((camera.id, rule.camera))
                    else:
                        enters.add((rule.camera, camera.id))
        return exits, enters

    def _build_warnings(self) -> list[str]:
        warnings = []
        for camera_id in self.camera_ids:
            if not self._cameras[camera_id].lines:
                warnings.append(f"Камера {camera_id}: нет ни одной линии перехода")

        for from_camera, to_camera in sorted(self._exit_pairs - self._enter_pairs):
            warnings.append(
                f"Переход {from_camera} -> {to_camera}: есть выходная линия (EXIT) на камере {from_camera}, "
                f"но на камере {to_camera} нет входной линии (ENTER) из камеры {from_camera}"
            )
        for from_camera, to_camera in sorted(self._enter_pairs - self._exit_pairs):
            warnings.append(
                f"Переход {from_camera} -> {to_camera}: есть входная линия (ENTER) на камере {to_camera}, "
                f"но на камере {from_camera} нет выходной линии (EXIT) в камеру {to_camera}"
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


def _parse_rule(raw: Any, index: int, where: str, problems: list[str]) -> Optional[CrossingRule]:
    label = f"{where}, crossings[{index}]"
    if not isinstance(raw, dict):
        problems.append(f"{label}: ожидается словарь с ключами edge, role, camera")
        return None

    before = len(problems)
    edge = _parse_enum(Edge, raw.get("edge"), f"{label}, edge", problems)
    role = _parse_enum(Role, raw.get("role"), f"{label}, role", problems)
    motion = _parse_enum(Edge, raw.get("motion"), f"{label}, motion", problems, required=False)
    camera = raw.get("camera")
    if not _is_int(camera):
        problems.append(f"{label}, camera: ожидается номер камеры (целое число), получено {camera!r}")

    if len(problems) > before:
        return None
    return CrossingRule(edge=edge, role=role, camera=camera, motion=motion)


def _parse_line(
        raw: Any,
        camera_id: int,
        size: tuple[int, int],
        known_ids: set[int],
        problems: list[str]
) -> Optional[TransitionLine]:
    if not isinstance(raw, dict):
        problems.append(f"Камера {camera_id}: линия должна быть словарём (id, p1, p2, crossings)")
        return None
    line_id = raw.get("id")
    if not _is_int(line_id):
        problems.append(f"Камера {camera_id}: у линии нет целочисленного id (получено {line_id!r})")
        return None

    where = f"Камера {camera_id}, линия {line_id}"
    p1 = _parse_point(raw.get("p1"), f"{where}, p1", size, problems)
    p2 = _parse_point(raw.get("p2"), f"{where}, p2", size, problems)
    degenerate = p1 is not None and p1 == p2
    if degenerate:
        problems.append(f"{where}: p1 и p2 совпадают, у линии нулевая длина")

    raw_rules = raw.get("crossings")
    rules: list[CrossingRule] = []
    if not isinstance(raw_rules, list) or not raw_rules:
        problems.append(f"{where}: нужен непустой список crossings")
    else:
        for index, raw_rule in enumerate(raw_rules):
            rule = _parse_rule(raw_rule, index, where, problems)
            if rule is None:
                continue
            if rule.camera == camera_id:
                problems.append(f"{where}: переход камеры {camera_id} в саму себя")
            elif rule.camera not in known_ids:
                problems.append(f"{where}: камеры {rule.camera} нет в списке cameras")
            rules.append(rule)

    if p1 is None or p2 is None or degenerate:
        return None
    line = TransitionLine(camera_id=camera_id, line_id=line_id, p1=p1, p2=p2, rules=tuple(rules))

    seen: dict[tuple[Edge, int], CrossingRule] = {}
    for rule in rules:
        motion = rule.motion_direction
        if line.motion_alignment(motion) < MIN_MOTION_ALIGNMENT:
            problems.append(
                f"{where}: край {rule.edge.value} при движении {motion.value} почти параллелен линии, "
                f"сторону пересечения не определить. Выберите край/motion поперёк линии"
            )
            continue
        key = (rule.edge, line.rule_sign(rule))
        if key in seen:
            problems.append(
                f"{where}: два правила для одного и того же пересечения "
                f"(край {rule.edge.value}, движение {motion.value})"
            )
            continue
        seen[key] = rule
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

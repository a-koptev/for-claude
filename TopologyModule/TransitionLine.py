from dataclasses import dataclass
from enum import Enum
from typing import Optional

# Минимальный |cos| угла между направлением движения и нормалью к линии.
# Если движение почти параллельно линии, сторону пересечения определить нельзя.
MIN_MOTION_ALIGNMENT = 0.5


class Edge(Enum):
    """Край bbox (он же направление движения по изображению)."""
    TOP = "TOP"
    BOTTOM = "BOTTOM"
    LEFT = "LEFT"
    RIGHT = "RIGHT"

    @property
    def vector(self) -> tuple[int, int]:
        """Направление наружу от центра bbox. Оси как в OpenCV: x вправо, y вниз."""
        return _EDGE_VECTORS[self]


_EDGE_VECTORS = {
    Edge.TOP: (0, -1),
    Edge.BOTTOM: (0, 1),
    Edge.LEFT: (-1, 0),
    Edge.RIGHT: (1, 0),
}


class Role(Enum):
    EXIT = "EXIT"    # пересечение означает, что автомобиль покидает эту камеру
    ENTER = "ENTER"  # пересечение означает, что автомобиль входит в эту камеру


@dataclass(frozen=True)
class CrossingRule:
    """
    Что значит пересечение линии краем bbox.

    edge   - каким краем bbox линия пересечена
    role   - EXIT (уезжает с этой камеры) или ENTER (приезжает на эту камеру)
    camera - для EXIT: в какую камеру уезжает; для ENTER: из какой камеры приехал
    motion - в какую сторону при этом движется bbox. По умолчанию совпадает с edge,
             то есть событие даёт ВЕДУЩИЙ край (BOTTOM при движении вниз).
             Нужен, если событие должно давать задний край: TOP при движении вниз
             означает "автомобиль полностью прошёл линию".
    """
    edge: Edge
    role: Role
    camera: int
    motion: Optional[Edge] = None

    @property
    def motion_direction(self) -> Edge:
        return self.motion or self.edge


@dataclass(frozen=True)
class TransitionLine:
    """
    Линия перехода на кадре одной камеры.

    Стороны линии: side(point) > 0 - положительная, < 0 - отрицательная.
    crossing_sign = +1, если при пересечении side растёт (отрицательная -> положительная),
    -1, если убывает. Это сравнивается со знаком, который посчитает детектор
    пересечений по двум последовательным положениям края bbox.
    """
    camera_id: int
    line_id: int
    p1: tuple[float, float]
    p2: tuple[float, float]
    rules: tuple[CrossingRule, ...] = ()

    @property
    def key(self) -> tuple[int, int]:
        return self.camera_id, self.line_id

    @property
    def normal(self) -> tuple[float, float]:
        """Градиент side(): направление в сторону положительной полуплоскости."""
        x1, y1 = self.p1
        x2, y2 = self.p2
        return -(y2 - y1), x2 - x1

    def side(self, point: tuple[float, float]) -> float:
        x1, y1 = self.p1
        x2, y2 = self.p2
        px, py = point
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    def motion_alignment(self, motion: Edge) -> float:
        """|cos| между движением и нормалью линии: 1 - строго поперёк линии, 0 - вдоль неё."""
        nx, ny = self.normal
        mx, my = motion.vector
        length = (nx * nx + ny * ny) ** 0.5
        if length == 0:
            return 0.0
        return abs(mx * nx + my * ny) / length

    def crossing_sign(self, motion: Edge) -> int:
        """+1, если движение в сторону `motion` переводит через линию в положительную сторону, иначе -1."""
        nx, ny = self.normal
        mx, my = motion.vector
        return 1 if mx * nx + my * ny > 0 else -1

    def rule_sign(self, rule: CrossingRule) -> int:
        return self.crossing_sign(rule.motion_direction)

    def match(self, edge: Edge, sign: int) -> Optional[CrossingRule]:
        """Правило для пересечения линии краем `edge` в направлении `sign` (или None)."""
        for rule in self.rules:
            if rule.edge is edge and self.rule_sign(rule) == sign:
                return rule
        return None

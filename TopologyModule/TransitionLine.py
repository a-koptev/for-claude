from dataclasses import dataclass
from enum import Enum

# Минимальный |cos| угла между движением и нормалью к линии.
# Если авто едет почти вдоль линии, сказать, пересекает он её туда или обратно, нельзя.
MIN_MOTION_ALIGNMENT = 0.5


class Direction(Enum):
    """Направление движения по изображению. Оси как в OpenCV: x вправо, y вниз."""
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"

    @property
    def vector(self) -> tuple[int, int]:
        return _DIRECTION_VECTORS[self]


_DIRECTION_VECTORS = {
    Direction.UP: (0, -1),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
    Direction.RIGHT: (1, 0),
}


class Role(Enum):
    EXIT = "EXIT"    # авто пересекает линию в сторону соседней камеры: уезжает с этой камеры
    ENTER = "ENTER"  # авто пересекает линию в обратную сторону: приехал с соседней камеры


@dataclass(frozen=True)
class TransitionLine:
    """
    Линия перехода на кадре одной камеры. Связывает эту камеру ровно с одной соседней.

    Линия двусторонняя: одно движение через неё значит "уезжаю в камеру peer_camera" (EXIT),
    противоположное - "приехал из камеры peer_camera" (ENTER).

    exit_direction - в какую сторону по кадру едет авто, когда уезжает в peer_camera.
                     Это единственное, что нужно знать о направлении: обратное движение
                     автоматически означает ENTER. Грани bbox не используются, линию
                     пересекает сам bbox (см. touches).
    """
    camera_id: int
    line_id: int
    p1: tuple[float, float]
    p2: tuple[float, float]
    peer_camera: int
    exit_direction: Direction

    @property
    def key(self) -> tuple[int, int]:
        return self.camera_id, self.line_id

    @property
    def normal(self) -> tuple[float, float]:
        x1, y1 = self.p1
        x2, y2 = self.p2
        return -(y2 - y1), x2 - x1

    @property
    def length(self) -> float:
        return ((self.p2[0] - self.p1[0]) ** 2 + (self.p2[1] - self.p1[1]) ** 2) ** 0.5

    @property
    def exit_sign(self) -> int:
        """+1 или -1: знак проекции движения на нормаль линии при выезде в peer_camera."""
        return 1 if self._dot_normal(self.exit_direction.vector) > 0 else -1

    def _dot_normal(self, vector: tuple[float, float]) -> float:
        nx, ny = self.normal
        return vector[0] * nx + vector[1] * ny

    def alignment(self, vector: tuple[float, float]) -> float:
        """|cos| между вектором движения и нормалью: 1 - строго поперёк линии, 0 - вдоль неё."""
        length = (vector[0] ** 2 + vector[1] ** 2) ** 0.5
        if length == 0 or self.length == 0:
            return 0.0
        return abs(self._dot_normal(vector)) / (length * self.length)

    def role_of_motion(self, vector: tuple[float, float]) -> Role:
        """EXIT, если вектор движения направлен в сторону выезда, иначе ENTER."""
        return Role.EXIT if self._dot_normal(vector) * self.exit_sign > 0 else Role.ENTER

    def touches(self, bbox) -> bool:
        """Пересекает ли bbox (x1, y1, x2, y2) отрезок линии: касание или пересечение границы или линия внутри."""
        return segment_intersects_box(self.p1, self.p2, bbox)


def segment_intersects_box(p1, p2, bbox) -> bool:
    """Отрезок p1-p2 имеет общие точки с прямоугольником bbox (отсечение Лианга - Барски)."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    px, py = p1
    dx, dy = p2[0] - px, p2[1] - py
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, px - x1), (dx, x2 - px), (-dy, py - y1), (dy, y2 - py)):
        if p == 0:
            if q < 0:
                return False
            continue
        t = q / p
        if p < 0:
            if t > t1:
                return False
            t0 = max(t0, t)
        else:
            if t < t0:
                return False
            t1 = min(t1, t)
    return t0 <= t1

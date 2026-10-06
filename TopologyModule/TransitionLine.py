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

    Переход фиксируется, когда центр bbox пересекает отрезок линии. Границы bbox
    сами по себе не являются условием перехода.

    exit_direction - в какую сторону по кадру едет центр автомобиля, когда уезжает
    в peer_camera. Обратное движение автоматически означает ENTER.
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

    def signed_distance(self, point: tuple[float, float]) -> float:
        """
        Знаковая величина положения точки относительно бесконечной линии.

        Знак определяет сторону линии. Масштаб пропорционален расстоянию до линии
        и дополнительно учитывает длину линии; для сравнения со стороной/пересечением
        важен только знак.
        """
        x1, y1 = self.p1
        x2, y2 = self.p2
        px, py = point
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    def center_crosses(
            self,
            previous_center: tuple[float, float],
            current_center: tuple[float, float],
    ) -> bool:
        """
        Проверяет, прошёл ли центр bbox через сам отрезок линии между двумя кадрами.

        Простого смены стороны относительно бесконечной линии недостаточно: центр
        мог пересечь продолжение линии за пределами её p1-p2. Поэтому дополнительно
        проверяется пересечение двух отрезков:
          previous_center -> current_center
          p1 -> p2.
        """
        previous_side = self.signed_distance(previous_center)
        current_side = self.signed_distance(current_center)

        # Центр должен оказаться по разные стороны. Если он только коснулся линии
        # и остался на той же стороне, это не считается переходом.
        if previous_side == 0 or current_side == 0:
            return False
        if previous_side * current_side >= 0:
            return False

        return _segments_intersect(previous_center, current_center, self.p1, self.p2)

    def touches(self, bbox) -> bool:
        """
        Совместимость со старой геометрией: пересекает ли bbox отрезок линии.

        Этот метод НЕ используется CrossingDetector для фиксации перехода.
        """
        return segment_intersects_box(self.p1, self.p2, bbox)


def _cross(a, b) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _segments_intersect(a, b, c, d, eps: float = 1e-9) -> bool:
    """Пересечение двух отрезков, включая касание концами."""
    r = (b[0] - a[0], b[1] - a[1])
    s = (d[0] - c[0], d[1] - c[1])
    denominator = _cross(r, s)
    c_minus_a = (c[0] - a[0], c[1] - a[1])

    if abs(denominator) <= eps:
        # Параллельные/коллинеарные отрезки. Для center_crosses это почти не
        # встречается, но обработаем корректно на случай численных погрешностей.
        if abs(_cross(c_minus_a, r)) > eps:
            return False

        def within(value, left, right):
            return min(left, right) - eps <= value <= max(left, right) + eps

        return (
            within(c[0], a[0], b[0]) or within(d[0], a[0], b[0])
            or within(a[0], c[0], d[0]) or within(b[0], c[0], d[0])
        ) and (
            within(c[1], a[1], b[1]) or within(d[1], a[1], b[1])
            or within(a[1], c[1], d[1]) or within(b[1], c[1], d[1])
        )

    t = _cross(c_minus_a, s) / denominator
    u = _cross(c_minus_a, r) / denominator
    return -eps <= t <= 1 + eps and -eps <= u <= 1 + eps


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

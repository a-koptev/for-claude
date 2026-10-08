from dataclasses import dataclass
from enum import Enum

# Минимальный |cos| угла между движением и нормалью к линии.
MIN_MOTION_ALIGNMENT = 0.5
# Допуск за концами отрезка линии: bbox может немного выступить за p1/p2,
# но автомобиль далеко за пределами линии не должен считаться пересёкшим её.
LINE_ENDPOINT_TOLERANCE_PX = 30.0


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
    EXIT = "EXIT"
    ENTER = "ENTER"


@dataclass(frozen=True)
class TransitionLine:
    """Линия перехода между двумя соседними камерами."""
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
        return 1 if self._dot_normal(self.exit_direction.vector) > 0 else -1

    def _dot_normal(self, vector: tuple[float, float]) -> float:
        nx, ny = self.normal
        return vector[0] * nx + vector[1] * ny

    def alignment(self, vector: tuple[float, float]) -> float:
        length = (vector[0] ** 2 + vector[1] ** 2) ** 0.5
        if length == 0 or self.length == 0:
            return 0.0
        return abs(self._dot_normal(vector)) / (length * self.length)

    def role_of_motion(self, vector: tuple[float, float]) -> Role:
        return Role.EXIT if self._dot_normal(vector) * self.exit_sign > 0 else Role.ENTER

    def signed_distance(self, point: tuple[float, float]) -> float:
        x1, y1 = self.p1
        x2, y2 = self.p2
        px, py = point
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        line_length = self.length
        return cross / line_length if line_length else 0.0

    @staticmethod
    def _bbox_corners(bbox):
        x1, y1, x2, y2 = (float(v) for v in bbox)
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return (
            (x1, y1),
            (x2, y1),
            (x2, y2),
            (x1, y2),
        )

    def _bbox_intersects_line_segment(self, bbox) -> bool:
        """Проверяет, находится ли bbox на уровне конечного отрезка линии."""
        if self.length == 0:
            return False

        nx, ny = self.normal
        normal_length = self.length
        nx /= normal_length
        ny /= normal_length
        tx, ty = -ny, nx

        along = [
            (px - self.p1[0]) * tx + (py - self.p1[1]) * ty
            for px, py in self._bbox_corners(bbox)
        ]
        tolerance = LINE_ENDPOINT_TOLERANCE_PX
        return not (max(along) < -tolerance or min(along) > self.length + tolerance)

    def bbox_straddles_line(self, bbox, motion: tuple[float, float]) -> bool:
        """True, если текущий bbox пересекает линию и движение идёт через неё."""
        if self.length == 0:
            return False

        nx, ny = self.normal
        normal_length = self.length
        nx /= normal_length
        ny /= normal_length

        motion_projection = motion[0] * nx + motion[1] * ny
        if abs(motion_projection) < 1e-9:
            return False

        if not self._bbox_intersects_line_segment(bbox):
            return False

        signed = [self.signed_distance(corner) for corner in self._bbox_corners(bbox)]
        return min(signed) <= 0.0 <= max(signed)

    def bbox_crosses_between(self, previous_bbox, current_bbox, motion: tuple[float, float]) -> bool:
        """
        Проверяет переход bbox через линию между двумя последовательными кадрами.

        Важный случай: маленький/далёкий автомобиль может за один кадр оказаться
        целиком с одной стороны линии, а на следующем — целиком с другой. Тогда
        ни previous_bbox, ни current_bbox по отдельности не straddle-ят линию,
        поэтому проверяем смену сторон всего bbox.
        """
        if self.length == 0:
            return False

        nx, ny = self.normal
        normal_length = self.length
        nx /= normal_length
        ny /= normal_length

        motion_projection = motion[0] * nx + motion[1] * ny
        if abs(motion_projection) < 1e-9:
            return False

        previous_corners = self._bbox_corners(previous_bbox)
        current_corners = self._bbox_corners(current_bbox)

        previous_signed = [self.signed_distance(corner) for corner in previous_corners]
        current_signed = [self.signed_distance(corner) for corner in current_corners]

        previous_min = min(previous_signed)
        previous_max = max(previous_signed)
        current_min = min(current_signed)
        current_max = max(current_signed)

        # Обычный случай: bbox уже пересекает линию в текущем кадре.
        if self._bbox_intersects_line_segment(current_bbox):
            if current_min <= 0.0 <= current_max:
                return True

        # Ключевой случай: весь bbox был по одну сторону, затем целиком
        # оказался по другую. Это означает, что bbox пересёк линию между кадрами.
        crossed_sides = (
            previous_max < 0.0 and current_min > 0.0
        ) or (
            previous_min > 0.0 and current_max < 0.0
        )
        if not crossed_sides:
            return False

        # Проверяем, что область bbox в этих двух кадрах находится напротив
        # конечного отрезка линии, а не только напротив его продолжения.
        return (
            self._bbox_intersects_line_segment(previous_bbox)
            or self._bbox_intersects_line_segment(current_bbox)
            or self._swept_along_line_overlaps(previous_corners, current_corners)
        )

    def _swept_along_line_overlaps(self, previous_corners, current_corners) -> bool:
        """Проверяет overlap проекций двух bbox вдоль линии с самим отрезком."""
        nx, ny = self.normal
        tx, ty = -ny, nx

        along = [
            (px - self.p1[0]) * tx + (py - self.p1[1]) * ty
            for px, py in (*previous_corners, *current_corners)
        ]
        tolerance = LINE_ENDPOINT_TOLERANCE_PX
        return max(along) >= -tolerance and min(along) <= self.length + tolerance

    def center_crosses(self, previous_center, current_center) -> bool:
        previous_side = self.signed_distance(previous_center)
        current_side = self.signed_distance(current_center)
        if previous_side == 0 or current_side == 0:
            return False
        if previous_side * current_side >= 0:
            return False
        return _segments_intersect(previous_center, current_center, self.p1, self.p2)

    def signed_bbox_rear_edge_distance(self, bbox, motion: tuple[float, float]) -> float:
        x1, y1, x2, y2 = (float(v) for v in bbox)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        nx, ny = self.normal
        normal_length = self.length
        if normal_length == 0:
            return 0.0

        nx /= normal_length
        ny /= normal_length

        half_projection = (
            abs(nx) * abs(x2 - x1) + abs(ny) * abs(y2 - y1)
        ) / 2.0

        center_distance = self.signed_distance((cx, cy))
        motion_projection = motion[0] * nx + motion[1] * ny
        if abs(motion_projection) < 1e-9:
            return center_distance

        motion_sign = 1.0 if motion_projection > 0 else -1.0
        return center_distance - motion_sign * half_projection

    def touches(self, bbox) -> bool:
        return segment_intersects_box(self.p1, self.p2, bbox)


def _cross(a, b) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _segments_intersect(a, b, c, d, eps: float = 1e-9) -> bool:
    r = (b[0] - a[0], b[1] - a[1])
    s = (d[0] - c[0], d[1] - c[1])
    denominator = _cross(r, s)
    c_minus_a = (c[0] - a[0], c[1] - a[1])

    if abs(denominator) <= eps:
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
    """Отрезок p1-p2 имеет общие точки с прямоугольником bbox."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if x2 < x1:
        x1, x2 = x2, x2
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

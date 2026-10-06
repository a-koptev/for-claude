"""Отрисовка линий перехода на кадре камеры, чтобы проверить topology.yaml глазами."""
from pathlib import Path
from typing import Union

import cv2
import numpy as np

from TopologyModule.CameraNetwork import Camera
from TopologyModule.TransitionLine import TransitionLine

# Цвета в BGR
LINE_COLOR = (0, 220, 255)     # линия и её концы
EXIT_COLOR = (60, 60, 255)     # стрелка правила EXIT
ENTER_COLOR = (80, 200, 80)    # стрелка правила ENTER
TEXT_COLOR = (255, 255, 255)
BACKGROUND_DIM = 0.35          # во сколько раз затемняется фон под текстом

ARROW_HALF_LENGTH = 35         # половина длины стрелки направления, px
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55


def read_frame(video_path: Union[str, Path], frame_size: tuple[int, int]) -> np.ndarray:
    """Первый кадр видео, приведённый к системе координат камеры (frame_size = ширина, высота)."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()

    if not ok or frame is None:
        raise RuntimeError(f"Не удалось открыть или прочитать видео: {video_path}")
    return cv2.resize(frame, frame_size)


def draw_topology(frame: np.ndarray, camera: Camera) -> np.ndarray:
    """
    Копия кадра с нарисованными линиями камеры.

    Линия: жёлтая, с номером и соседней камерой. Две стрелки поперёк линии показывают
    движение: красная - EXIT (уезжает в соседнюю камеру), зелёная - ENTER (приехал из неё).
    Подписи только латиницей: OpenCV не умеет рисовать кириллицу.
    """
    image = frame.copy()
    for line in camera.lines:
        _draw_line(image, line)
    _draw_legend(image, camera)
    return image


def _point(point: tuple[float, float]) -> tuple[int, int]:
    return round(point[0]), round(point[1])


def _put_text(image: np.ndarray, text: str, x: int, y: int) -> None:
    """Белый текст на затемнённой подложке (читается на любом фоне), не выходящий за границы кадра."""
    height, width = image.shape[:2]
    (text_width, text_height), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, 1)
    x = max(6, min(x, width - text_width - 6))
    y = max(text_height + 6, min(y, height - baseline - 6))

    # Подложка: затемняем область под текстом, сам кадр остаётся виден
    x0, y0 = x - 4, y - text_height - 4
    x1, y1 = x + text_width + 4, y + baseline + 3
    region = image[y0:y1, x0:x1]
    image[y0:y1, x0:x1] = (region * BACKGROUND_DIM).astype(np.uint8)

    cv2.putText(image, text, (x, y), FONT, FONT_SCALE, TEXT_COLOR, 1, cv2.LINE_AA)


def _draw_line(image: np.ndarray, line: TransitionLine) -> None:
    p1, p2 = _point(line.p1), _point(line.p2)
    cv2.line(image, p1, p2, LINE_COLOR, 2, cv2.LINE_AA)
    cv2.circle(image, p1, 5, LINE_COLOR, -1, cv2.LINE_AA)
    cv2.circle(image, p2, 5, LINE_COLOR, -1, cv2.LINE_AA)
    _put_text(image, f"L{line.line_id} <-> cam {line.peer_camera}", p1[0] + 8, p1[1] - 8)

    # Две стрелки поперёк линии, разнесённые вдоль неё: красная - выезд в соседнюю камеру,
    # зелёная - въезд из неё (движение в обратную сторону)
    vx, vy = line.exit_direction.vector
    arrows = ((0.33, 1, EXIT_COLOR, f"EXIT -> cam {line.peer_camera}"),
              (0.67, -1, ENTER_COLOR, f"ENTER <- cam {line.peer_camera}"))
    for t, sign, color, label in arrows:
        cx = p1[0] + (p2[0] - p1[0]) * t
        cy = p1[1] + (p2[1] - p1[1]) * t
        dx, dy = sign * vx * ARROW_HALF_LENGTH, sign * vy * ARROW_HALF_LENGTH
        start, end = _point((cx - dx, cy - dy)), _point((cx + dx, cy + dy))
        cv2.arrowedLine(image, start, end, color, 3, cv2.LINE_AA, tipLength=0.4)
        if vx != 0:   # движение по горизонтали: подпись над стрелкой
            (text_width, _), _ = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
            _put_text(image, label, round(cx - text_width / 2), round(cy) - 18)
        else:         # движение по вертикали: подпись справа от наконечника
            _put_text(image, label, end[0] + 20, end[1] + 5)


def _draw_legend(image: np.ndarray, camera: Camera) -> None:
    _put_text(image, f"CAM {camera.id}  |  lines: {len(camera.lines)}  |  frame {camera.frame_size[0]}x{camera.frame_size[1]}", 10, 22)
    if camera.lines:
        _put_text(image, "arrows = motion across the line: red = EXIT to peer camera, green = ENTER from it", 10, 44)
    else:
        _put_text(image, "no lines defined for this camera", 10, 44)

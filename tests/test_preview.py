import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from TopologyModule.CameraNetwork import CameraNetwork  # noqa: E402
from TopologyModule.Preview import (  # noqa: E402
    ENTER_COLOR,
    EXIT_COLOR,
    LINE_COLOR,
    draw_topology,
    read_frame,
)


def network(lines):
    return CameraNetwork.from_dict({
        "frame_size": [1280, 720],
        "cameras": [{"id": 1, "lines": lines}, {"id": 2}, {"id": 5}],
    })


def has_color(image, color, tolerance=12):
    return bool(np.all(np.abs(image.astype(int) - np.array(color)) <= tolerance, axis=2).any())


HORIZONTAL = {"id": 1, "p1": [100, 400], "p2": [1180, 400], "peer": 5, "exit": "DOWN"}


def test_line_and_role_colors_are_drawn_without_touching_the_input():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    camera = network([HORIZONTAL]).camera(1)

    image = draw_topology(frame, camera)

    assert image.shape == frame.shape
    assert not frame.any()                       # исходный кадр не изменён
    assert has_color(image[399:402, 600:610], LINE_COLOR)   # сама линия
    assert has_color(image, EXIT_COLOR)          # стрелка EXIT
    assert has_color(image, ENTER_COLOR)         # стрелка ENTER


def test_arrows_go_across_the_line():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    camera = network([HORIZONTAL]).camera(1)

    image = draw_topology(frame, camera)

    # выезд (красная) и въезд (зелёная) нарисованы стрелками, пересекающими линию y = 400
    for color in (EXIT_COLOR, ENTER_COLOR):
        mask = np.all(np.abs(image.astype(int) - np.array(color)) <= 12, axis=2)
        rows = np.where(mask.any(axis=1))[0]
        assert rows.min() < 400 < rows.max()


def test_camera_without_lines_still_gets_legend():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    image = draw_topology(frame, network([]).camera(1))

    assert image.any()                           # подпись "no lines defined" нарисована


def test_labels_near_frame_border_do_not_crash():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    camera = network([{"id": 7, "p1": [1, 1], "p2": [1279, 1], "peer": 5, "exit": "UP"}]).camera(1)

    assert draw_topology(frame, camera).shape == frame.shape


def test_read_frame_resizes_to_camera_coordinates(tmp_path):
    path = tmp_path / "video.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 360))
    for _ in range(3):
        writer.write(np.full((360, 640, 3), 128, dtype=np.uint8))
    writer.release()

    frame = read_frame(path, (1280, 720))

    assert frame.shape == (720, 1280, 3)


def test_read_frame_reports_unreadable_video(tmp_path):
    with pytest.raises(RuntimeError, match="Не удалось открыть"):
        read_frame(tmp_path / "missing.mp4", (1280, 720))

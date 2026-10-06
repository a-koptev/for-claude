"""
Проверка конфигурации топологии.

    python check_topology.py [путь/к/topology.yaml]

Без аргумента берётся Config.TOPOLOGY_CONFIG_PATH. Код возврата 1, если в файле есть ошибки.

Чтобы увидеть линии на кадре камеры, заполните две переменные ниже и запустите скрипт.
"""
import os
import sys
from pathlib import Path

from Config import Config
from TopologyModule.CameraNetwork import CameraNetwork, TopologyError

# --- Предпросмотр линий на кадре --------------------------------------------------
VIDEO_PATH = r"D:\video\3\5009.ts"   # путь до видео камеры, например "../videos/cam1.mp4" (пусто - без предпросмотра)
CAMERA_ID = 9     # номер камеры из topology.yaml, линии которой нужно показать


def describe(network: CameraNetwork) -> None:
    for camera_id in network.camera_ids:
        camera = network.camera(camera_id)
        width, height = camera.frame_size
        print(f"Камера {camera.id} ({camera.name}), кадр {width}x{height}, линий: {len(camera.lines)}")
        for line in camera.lines:
            print(
                f"  линия {line.line_id}: {line.p1} -> {line.p2}, соседняя камера {line.peer_camera}, "
                f"выезд = движение {line.exit_direction.value}"
            )

    print()
    print("Переходы между камерами:")
    if not network.transitions:
        print("  пока нет")
    for from_camera, to_camera in sorted(network.transitions):
        segment = network.segment(from_camera, to_camera)
        order = "порядок сохраняется" if segment.preserve_order else "порядок не фиксирован"
        print(f"  {from_camera} -> {to_camera} ({order})")


def can_show_window() -> bool:
    """На Linux без дисплея cv2.imshow не бросает исключение, а аварийно завершает процесс."""
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def show_preview(network: CameraNetwork) -> int:
    """Рисует линии камеры CAMERA_ID на первом кадре видео VIDEO_PATH и показывает окно."""
    import cv2
    from TopologyModule.Preview import draw_topology, read_frame

    try:
        camera = network.camera(CAMERA_ID)
    except KeyError as error:
        print(f"\nПредпросмотр: {error.args[0]}. Камеры в конфиге: {network.camera_ids}")
        return 1

    try:
        frame = read_frame(VIDEO_PATH, camera.frame_size)
    except RuntimeError as error:
        print(f"\nПредпросмотр: {error}")
        return 1

    image = draw_topology(frame, camera)
    title = f"topology preview, camera {camera.id}"

    if can_show_window():
        try:
            cv2.imshow(title, image)
            print(f"\nПредпросмотр камеры {camera.id}: закройте окно любой клавишей")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return 0
        except cv2.error:
            pass  # сборка OpenCV без окон (opencv-python-headless)

    # Окна нет (сервер, SSH без X11, headless-сборка): сохраняем картинку в файл
    path = Path(Config.BASE_DIR) / f"topology_preview_cam{camera.id}.png"
    cv2.imwrite(str(path), image)
    print(f"\nОкно показать нельзя, картинка сохранена: {path}")
    return 0


def main(argv: list[str]) -> int:
    path = argv[1] if len(argv) > 1 else Config.TOPOLOGY_CONFIG_PATH
    try:
        network = CameraNetwork.from_yaml(path)
    except FileNotFoundError:
        print(f"Файл не найден: {path}")
        return 1
    except TopologyError as error:
        print(error)
        return 1

    print(f"Файл: {path}")
    print(f"Камер: {len(network.camera_ids)}, переходов: {len(network.transitions)}\n")
    describe(network)

    if network.warnings:
        print(f"\nПредупреждения ({len(network.warnings)}):")
        for warning in network.warnings:
            print(f"  - {warning}")
    else:
        print("\nОшибок и предупреждений нет")

    if VIDEO_PATH:
        return show_preview(network)
    print("\nЧтобы увидеть линии на кадре, задайте VIDEO_PATH и CAMERA_ID в check_topology.py")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

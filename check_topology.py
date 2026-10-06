"""
Проверка конфигурации топологии.

    python check_topology.py [путь/к/topology.yaml]

Без аргумента берётся Config.TOPOLOGY_CONFIG_PATH. Код возврата 1, если в файле есть ошибки.
"""
import sys

from Config import Config
from TopologyModule.CameraNetwork import CameraNetwork, TopologyError


def describe(network: CameraNetwork) -> None:
    for camera_id in network.camera_ids:
        camera = network.camera(camera_id)
        width, height = camera.frame_size
        print(f"Камера {camera.id} ({camera.name}), кадр {width}x{height}, линий: {len(camera.lines)}")
        for line in camera.lines:
            print(f"  линия {line.line_id}: {line.p1} -> {line.p2}")
            for rule in line.rules:
                motion = f", движение {rule.motion_direction.value}" if rule.motion else ""
                direction = "в камеру" if rule.role.value == "EXIT" else "из камеры"
                print(
                    f"    край {rule.edge.value}{motion}: {rule.role.value} {direction} {rule.camera} "
                    f"(знак пересечения {line.rule_sign(rule):+d})"
                )

    print()
    print("Переходы между камерами:")
    if not network.transitions:
        print("  пока нет")
    for from_camera, to_camera in sorted(network.transitions):
        segment = network.segment(from_camera, to_camera)
        order = "порядок сохраняется" if segment.preserve_order else "порядок не фиксирован"
        print(f"  {from_camera} -> {to_camera} ({order})")


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
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

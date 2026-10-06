"""Топология парковки: сеть камер и двусторонние линии перехода."""
import pytest

from Config import Config
from TopologyModule.CameraNetwork import CameraNetwork, RouteSegment, TopologyError
from TopologyModule.CrossingDetector import CrossingDetector
from TopologyModule.TransitionLine import Direction, Role, TransitionLine, segment_intersects_box


def horizontal_line(line_id=1, y=600, peer=5, exit_direction="DOWN"):
    """Горизонтальная линия поперёк кадра: вниз - уезжаем в peer, вверх - приезжаем из peer."""
    return {"id": line_id, "p1": [100, y], "p2": [1180, y], "peer": peer, "exit": exit_direction}


def make_config(**overrides):
    """Три камеры: 1 <-> 5 и 2 <-> 1, у каждой пары линия описана с обеих сторон."""
    config = {
        "frame_size": [1280, 720],
        "cameras": [
            {"id": 1, "lines": [horizontal_line(1, peer=5), horizontal_line(2, y=100, peer=2, exit_direction="UP")]},
            {"id": 2, "lines": [horizontal_line(1, peer=1)]},
            {"id": 5, "lines": [horizontal_line(1, peer=1, exit_direction="UP")]},
        ],
        "segments": [{"from": 1, "to": 5, "preserve_order": True}],
    }
    config.update(overrides)
    return config


def problems_of(config):
    with pytest.raises(TopologyError) as error:
        CameraNetwork.from_dict(config)
    return error.value.problems


def has_problem(problems, text):
    return any(text in problem for problem in problems)


# --- шаблон и базовая загрузка --------------------------------------------------

def test_shipped_template_has_nine_cameras():
    network = CameraNetwork.from_yaml(Config.TOPOLOGY_CONFIG_PATH)

    assert network.camera_ids == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert network.transitions == frozenset({
        (1, 4), (4, 1), (1, 9), (9, 1),
        (2, 4), (4, 2), (2, 9), (9, 2),
        (3, 4), (4, 3), (3, 9), (9, 3),
        (4, 5), (5, 4), (5, 6), (6, 5),
        (6, 7), (7, 6), (7, 8), (8, 7),
        (8, 9), (9, 8),
    })
    assert all(network.camera(i).frame_size == (1280, 720) for i in network.camera_ids)


def test_one_line_gives_both_directions():
    network = CameraNetwork.from_dict(make_config())

    # линия 1 на камере 1 к камере 5: 1 -> 5 (выезд) и 5 -> 1 (въезд); линия к камере 2 аналогично
    assert network.transitions == frozenset({(1, 5), (5, 1), (1, 2), (2, 1)})
    assert network.warnings == []


def test_neighbours():
    network = CameraNetwork.from_dict(make_config())

    assert network.next_cameras(1) == [2, 5]
    assert network.previous_cameras(1) == [2, 5]
    assert network.has_transition(5, 1) and network.has_transition(1, 5)
    assert not network.has_transition(2, 5)


def test_segments():
    network = CameraNetwork.from_dict(make_config())

    assert network.segment(1, 5) == RouteSegment(1, 5, preserve_order=True)
    assert network.segment(2, 1) == RouteSegment(2, 1, preserve_order=False)  # по умолчанию
    assert network.segment(2, 5) is None                                       # перехода нет


def test_line_lookup():
    network = CameraNetwork.from_dict(make_config())

    assert network.line(1, 1).p1 == (100.0, 600.0)
    assert [line.line_id for line in network.lines_to(1, 2)] == [2]
    assert network.lines_to(1, 9) == ()
    with pytest.raises(KeyError):
        network.line(1, 99)
    with pytest.raises(KeyError):
        network.camera(42)


# --- геометрия линии ------------------------------------------------------------

def line(p1=(100, 600), p2=(1180, 600), exit_direction=Direction.DOWN):
    return TransitionLine(1, 1, p1, p2, peer_camera=5, exit_direction=exit_direction)


@pytest.mark.parametrize("vector, role", [
    ((0, 30), Role.EXIT),      # вниз - выезд
    ((5, 30), Role.EXIT),      # вниз с небольшим уходом вбок
    ((0, -30), Role.ENTER),    # вверх - въезд
    ((-5, -30), Role.ENTER),
])
def test_role_of_motion_for_horizontal_line_exit_down(vector, role):
    assert line().role_of_motion(vector) is role


@pytest.mark.parametrize("p1, p2", [((100, 600), (1180, 600)), ((1180, 600), (100, 600))])
def test_role_does_not_depend_on_point_order(p1, p2):
    assert line(p1, p2).role_of_motion((0, 20)) is Role.EXIT
    assert line(p1, p2).role_of_motion((0, -20)) is Role.ENTER


def test_vertical_line_with_left_right():
    vertical = line((600, 50), (600, 650), Direction.RIGHT)

    assert vertical.role_of_motion((25, 0)) is Role.EXIT
    assert vertical.role_of_motion((-25, 0)) is Role.ENTER
    assert vertical.alignment((25, 0)) == pytest.approx(1.0)
    assert vertical.alignment((0, 25)) == pytest.approx(0.0)


@pytest.mark.parametrize("bbox, expected", [
    ((500, 550, 700, 650), True),    # линия проходит через bbox
    ((500, 400, 700, 600), True),    # линия ровно по нижней границе
    ((500, 400, 700, 599), False),   # чуть не дотянулся
    ((500, 601, 700, 700), False),   # целиком ниже линии
    ((10, 550, 90, 650), False),     # на уровне линии, но левее её конца
    ((1170, 550, 1270, 650), True),  # касается правого конца
    ((700, 650, 500, 550), True),    # координаты в обратном порядке
])
def test_touches(bbox, expected):
    assert line().touches(bbox) is expected


def test_center_crosses_only_when_center_changes_side():
    transition_line = line()

    assert transition_line.center_crosses((500, 590), (500, 610))
    assert transition_line.center_crosses((500, 610), (500, 590))
    assert not transition_line.center_crosses((500, 590), (500, 595))


def test_center_crossing_does_not_depend_on_bbox_size():
    transition_line = line()
    detector = CrossingDetector(
        CameraNetwork.from_dict(make_config()).camera(1),
        release_frames=1,
        release_margin_px=10,
    )

    # Большой bbox уже касается линии верхней границей, но его центр ещё выше.
    # Переход не должен фиксироваться.
    assert detector.update(0, [(10, (400, 500, 800, 590))]) == []

    # Теперь центр действительно оказался ниже линии -> фиксируем переход.
    events = detector.update(1, [(10, (400, 610, 800, 700))])
    assert len(events) == 1
    assert events[0].from_camera == 1
    assert events[0].to_camera == 5
    assert events[0].role is Role.EXIT


def test_late_detection_accepts_bbox_not_fully_over_line():
    detector = CrossingDetector(CameraNetwork.from_dict(make_config()).camera(1), release_frames=1, release_margin_px=10)
    # Первый bbox уже после линии, но задняя грань ещё не прошла её.
    assert detector.update(0, [(10, (400, 610, 800, 710))]) == []
    events = detector.update(1, [(10, (400, 620, 800, 720))])
    assert len(events) == 1
    assert events[0].from_camera == 1
    assert events[0].to_camera == 5

def test_late_detection_accepts_edge_within_ten_pixels():
    detector = CrossingDetector(CameraNetwork.from_dict(make_config()).camera(1), release_frames=1, release_margin_px=10)
    # Центр далеко за линией, но задняя грань выступает всего на 5 px.
    assert detector.update(0, [(10, (400, 602, 800, 702))]) == []
    events = detector.update(1, [(10, (400, 605, 800, 705))])
    assert len(events) == 1

def test_late_detection_rejects_fully_crossed_vehicle():
    detector = CrossingDetector(CameraNetwork.from_dict(make_config()).camera(1), release_frames=1, release_margin_px=10)
    # Вся машина уже за линией более чем на 10 px.
    assert detector.update(0, [(10, (400, 700, 800, 800))]) == []
    assert detector.update(1, [(10, (400, 710, 800, 810))]) == []

def test_late_detection_uses_motion_direction():
    detector = CrossingDetector(CameraNetwork.from_dict(make_config()).camera(1), release_frames=1, release_margin_px=10)
    # Центр за линией, но движется обратно к ней: это не поздний EXIT.
    assert detector.update(0, [(10, (400, 610, 800, 710))]) == []
    assert detector.update(1, [(10, (400, 605, 800, 705))]) == []

def test_segment_intersects_box_diagonal():
    assert segment_intersects_box((0, 0), (100, 100), (40, 40, 60, 60))
    assert not segment_intersects_box((0, 0), (100, 100), (60, 10, 90, 40))   # рядом с диагональю, но не на ней


# --- валидация: ошибки собираются все сразу --------------------------------------

def test_all_problems_are_reported_together():
    config = make_config()
    config["cameras"][0]["lines"] = [
        horizontal_line(1, peer=77),
        {"id": 2, "p1": [10, 10], "p2": [10, 10], "peer": 5, "exit": "DOWN"},
    ]
    config["segments"] = [{"from": 1, "to": 88}]

    problems = problems_of(config)

    assert has_problem(problems, "камеры 77 нет в списке")
    assert has_problem(problems, "нулевая длина")
    assert has_problem(problems, "камеры [88] нет в списке")


def test_unknown_peer_is_reported_even_if_line_geometry_is_broken():
    config = make_config()
    config["cameras"][0]["lines"] = [
        {"id": 1, "p1": [100, 600], "p2": [1180, 9999], "peer": 7, "exit": "DOWN"}
    ]

    problems = problems_of(config)

    assert has_problem(problems, "вне кадра")
    assert has_problem(problems, "камеры 7 нет в списке")


@pytest.mark.parametrize("mutate, expected", [
    (lambda c: c["cameras"].append({"id": 1, "lines": []}), "id встречается больше одного раза"),
    (lambda c: c["cameras"][0]["lines"].append(horizontal_line(1)), "id линии 1 встречается больше одного раза"),
    (lambda c: c["cameras"][0]["lines"][0].__setitem__("peer", 1), "ведёт в саму камеру"),
    (lambda c: c["cameras"][0]["lines"][0].__setitem__("peer", "пять"), "ожидается номер соседней камеры"),
    (lambda c: c["cameras"][0]["lines"][0].pop("peer"), "ожидается номер соседней камеры"),
    (lambda c: c["cameras"][0]["lines"][0].__setitem__("p2", [1180, 5000]), "вне кадра 1280x720"),
    (lambda c: c["cameras"][0]["lines"][0].__setitem__("p1", [100]), "ожидается [x, y]"),
    (lambda c: c["cameras"][0]["lines"][0].__setitem__("exit", "DIAGONAL"), "ожидается UP | DOWN | LEFT | RIGHT"),
    (lambda c: c["cameras"][0]["lines"][0].pop("exit"), "ожидается UP | DOWN | LEFT | RIGHT"),
    (lambda c: c["cameras"][0]["lines"][0].__setitem__("exit", "LEFT"), "почти параллельно линии"),
    (lambda c: c["segments"].append({"from": 1, "to": 5}), "задан больше одного раза"),
    (lambda c: c["segments"].__setitem__(0, {"from": 1, "to": 5, "preserve_order": "да"}), "true или false"),
])
def test_invalid_config_is_rejected(mutate, expected):
    config = make_config()
    mutate(config)

    assert has_problem(problems_of(config), expected)


def test_direction_names_are_case_insensitive():
    config = make_config()
    config["cameras"][0]["lines"][0]["exit"] = "down"

    assert CameraNetwork.from_dict(config).line(1, 1).exit_direction is Direction.DOWN


def test_frame_size_is_per_camera():
    config = make_config()
    config["cameras"][0]["frame_size"] = [1920, 1080]
    config["cameras"][0]["lines"][0]["p2"] = [1800, 600]  # вне 1280, но внутри 1920

    assert CameraNetwork.from_dict(config).camera(1).frame_size == (1920, 1080)


@pytest.mark.parametrize("config", [None, [], "text", {}, {"cameras": []}])
def test_garbage_root_is_rejected(config):
    assert problems_of(config)


# --- предупреждения -----------------------------------------------------------------

def test_line_on_one_side_only_warns():
    config = make_config()
    config["cameras"][2]["lines"] = []  # на камере 5 нет линии к камере 1

    warnings = CameraNetwork.from_dict(config).warnings

    assert any("Камеры 1 и 5" in w and "нет линии к камере 1" in w for w in warnings)
    assert any("Камера 5" in w and "нет ни одной линии" in w for w in warnings)


def test_segment_without_transition_warns():
    config = make_config()
    config["segments"] = [{"from": 5, "to": 2, "preserve_order": True}]

    warnings = CameraNetwork.from_dict(config).warnings

    assert any("5 -> 2" in w and "линий перехода между этими камерами нет" in w for w in warnings)


# --- файл --------------------------------------------------------------------------

def test_from_yaml_roundtrip(tmp_path):
    import yaml
    path = tmp_path / "topology.yaml"
    path.write_text(yaml.safe_dump(make_config(), allow_unicode=True), encoding="utf-8")

    assert CameraNetwork.from_yaml(path).transitions == frozenset({(1, 5), (5, 1), (1, 2), (2, 1)})


@pytest.mark.parametrize("text", ["", "cameras: [unclosed", "- just\n- a list"])
def test_broken_yaml_is_reported_as_topology_error(tmp_path, text):
    path = tmp_path / "topology.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(TopologyError):
        CameraNetwork.from_yaml(path)

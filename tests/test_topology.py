import pytest

from Config import Config
from TopologyModule.CameraNetwork import CameraNetwork, RouteSegment, TopologyError
from TopologyModule.TransitionLine import CrossingRule, Edge, Role


def horizontal_line(line_id=1, y=600, crossings=None):
    """Горизонтальная линия поперёк кадра; по умолчанию как пример в topology.yaml."""
    return {
        "id": line_id,
        "p1": [100, y],
        "p2": [1180, y],
        "crossings": crossings or [
            {"edge": "BOTTOM", "role": "EXIT", "camera": 5},
            {"edge": "TOP", "role": "ENTER", "camera": 2},
        ],
    }


def make_config(**overrides):
    """Три камеры с симметричными линиями: 1 -> 5 и 2 -> 1 описаны с обеих сторон."""
    config = {
        "frame_size": [1280, 720],
        "cameras": [
            {"id": 1, "lines": [horizontal_line()]},
            {"id": 2, "lines": [horizontal_line(crossings=[{"edge": "BOTTOM", "role": "EXIT", "camera": 1}])]},
            {"id": 5, "lines": [horizontal_line(crossings=[{"edge": "TOP", "role": "ENTER", "camera": 1}])]},
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
    assert network.transitions == frozenset()
    assert all(network.camera(i).frame_size == (1280, 720) for i in network.camera_ids)


def test_valid_network_has_no_warnings():
    network = CameraNetwork.from_dict(make_config())

    assert network.warnings == []
    assert network.transitions == frozenset({(1, 5), (2, 1)})


def test_neighbours():
    network = CameraNetwork.from_dict(make_config())

    assert network.next_cameras(1) == [5]
    assert network.previous_cameras(1) == [2]
    assert network.next_cameras(5) == []
    assert network.previous_cameras(5) == [1]
    assert network.has_transition(1, 5) and not network.has_transition(5, 1)


def test_segments():
    network = CameraNetwork.from_dict(make_config())

    assert network.segment(1, 5) == RouteSegment(1, 5, preserve_order=True)
    assert network.segment(2, 1) == RouteSegment(2, 1, preserve_order=False)  # по умолчанию
    assert network.segment(5, 1) is None                                       # перехода нет


def test_line_lookup():
    network = CameraNetwork.from_dict(make_config())

    assert network.line(1, 1).p1 == (100.0, 600.0)
    with pytest.raises(KeyError):
        network.line(1, 99)
    with pytest.raises(KeyError):
        network.camera(42)


# --- правила пересечения: ведущий край, направление, motion ----------------------

def test_leading_edge_matches_only_in_its_own_direction():
    line = CameraNetwork.from_dict(make_config()).line(1, 1)

    # движение вниз: ведущим пересекает BOTTOM, потом задним TOP
    assert line.match(Edge.BOTTOM, +1) == CrossingRule(Edge.BOTTOM, Role.EXIT, 5)
    assert line.match(Edge.TOP, +1) is None
    # движение вверх: ведущим TOP, задним BOTTOM
    assert line.match(Edge.TOP, -1) == CrossingRule(Edge.TOP, Role.ENTER, 2)
    assert line.match(Edge.BOTTOM, -1) is None


def test_sign_agrees_with_side_function():
    line = CameraNetwork.from_dict(make_config()).line(1, 1)

    before, after = (640, 590), (640, 610)  # едет вниз через y=600
    sign = 1 if line.side(after) > line.side(before) else -1

    assert sign == line.rule_sign(line.rules[0])  # правило BOTTOM
    assert line.match(Edge.BOTTOM, sign).role is Role.EXIT


def test_motion_override_selects_trailing_edge():
    config = make_config()
    config["cameras"][0]["lines"] = [horizontal_line(crossings=[
        {"edge": "TOP", "role": "ENTER", "camera": 2, "motion": "BOTTOM"},  # TOP при движении вниз
    ])]
    line = CameraNetwork.from_dict(config).line(1, 1)

    assert line.match(Edge.TOP, +1).camera == 2
    assert line.match(Edge.TOP, -1) is None


def test_vertical_line_with_left_right_edges():
    config = make_config()
    config["cameras"][0]["lines"] = [{
        "id": 1, "p1": [600, 50], "p2": [600, 650],
        "crossings": [
            {"edge": "RIGHT", "role": "EXIT", "camera": 5},
            {"edge": "LEFT", "role": "ENTER", "camera": 2},
        ],
    }]
    line = CameraNetwork.from_dict(config).line(1, 1)

    right_rule, left_rule = line.rules
    assert line.rule_sign(right_rule) == -line.rule_sign(left_rule)
    assert line.match(Edge.RIGHT, line.rule_sign(right_rule)) == right_rule


def test_names_are_case_insensitive():
    config = make_config()
    config["cameras"][0]["lines"][0]["crossings"][0]["edge"] = "bottom"
    config["cameras"][0]["lines"][0]["crossings"][0]["role"] = "exit"

    assert CameraNetwork.from_dict(config).line(1, 1).rules[0].edge is Edge.BOTTOM


# --- валидация: ошибки собираются все сразу --------------------------------------

def test_all_problems_are_reported_together():
    config = make_config()
    config["cameras"][0]["lines"] = [
        horizontal_line(1, crossings=[{"edge": "BOTTOM", "role": "EXIT", "camera": 77}]),
        {"id": 2, "p1": [10, 10], "p2": [10, 10], "crossings": []},
    ]
    config["segments"] = [{"from": 1, "to": 88}]

    problems = problems_of(config)

    assert has_problem(problems, "камеры 77 нет в списке")
    assert has_problem(problems, "нулевая длина")
    assert has_problem(problems, "нужен непустой список crossings")
    assert has_problem(problems, "камеры [88] нет в списке")


def test_unknown_target_is_reported_even_if_line_geometry_is_broken():
    config = make_config()
    config["cameras"][0]["lines"] = [{
        "id": 1, "p1": [100, 600], "p2": [1180, 9999],
        "crossings": [{"edge": "BOTTOM", "role": "EXIT", "camera": 7}],
    }]

    problems = problems_of(config)

    assert has_problem(problems, "вне кадра")
    assert has_problem(problems, "камеры 7 нет в списке")


@pytest.mark.parametrize("mutate, expected", [
    (lambda c: c["cameras"].append({"id": 1, "lines": []}), "id встречается больше одного раза"),
    (lambda c: c["cameras"][0]["lines"].append(horizontal_line(1)), "id линии 1 встречается больше одного раза"),
    (lambda c: c["cameras"][0]["lines"][0]["crossings"].__setitem__(
        0, {"edge": "BOTTOM", "role": "EXIT", "camera": 1}), "в саму себя"),
    (lambda c: c["cameras"][0]["lines"][0].__setitem__("p2", [1180, 5000]), "вне кадра 1280x720"),
    (lambda c: c["cameras"][0]["lines"][0].__setitem__("p1", [100]), "ожидается [x, y]"),
    (lambda c: c["cameras"][0]["lines"][0]["crossings"][0].__setitem__("edge", "DIAGONAL"),
     "ожидается TOP | BOTTOM | LEFT | RIGHT"),
    (lambda c: c["cameras"][0]["lines"][0]["crossings"][0].__setitem__("role", "STAY"),
     "ожидается EXIT | ENTER"),
    (lambda c: c["cameras"][0]["lines"][0]["crossings"][0].__setitem__("camera", "пять"),
     "ожидается номер камеры"),
    (lambda c: c["cameras"][0]["lines"][0]["crossings"][0].__setitem__("edge", "LEFT"),
     "почти параллелен линии"),
    (lambda c: c["cameras"][0]["lines"][0]["crossings"].append(
        {"edge": "BOTTOM", "role": "EXIT", "camera": 2}), "два правила для одного и того же пересечения"),
    (lambda c: c["segments"].append({"from": 1, "to": 5}), "задан больше одного раза"),
    (lambda c: c["segments"].__setitem__(0, {"from": 1, "to": 5, "preserve_order": "да"}),
     "true или false"),
])
def test_invalid_config_is_rejected(mutate, expected):
    config = make_config()
    mutate(config)

    assert has_problem(problems_of(config), expected)


def test_frame_size_is_per_camera():
    config = make_config()
    config["cameras"][0]["frame_size"] = [1920, 1080]
    config["cameras"][0]["lines"][0]["p2"] = [1800, 600]  # вне 1280, но внутри 1920

    assert CameraNetwork.from_dict(config).camera(1).frame_size == (1920, 1080)


@pytest.mark.parametrize("config", [None, [], "text", {}, {"cameras": []}])
def test_garbage_root_is_rejected(config):
    assert problems_of(config)


# --- предупреждения -----------------------------------------------------------------

def test_one_sided_transition_warns():
    config = make_config()
    config["cameras"][2]["lines"] = []  # на камере 5 нет входной линии из 1

    warnings = CameraNetwork.from_dict(config).warnings

    assert any("1 -> 5" in w and "нет входной линии" in w for w in warnings)
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

    assert CameraNetwork.from_yaml(path).transitions == frozenset({(1, 5), (2, 1)})


@pytest.mark.parametrize("text", ["", "cameras: [unclosed", "- just\n- a list"])
def test_broken_yaml_is_reported_as_topology_error(tmp_path, text):
    path = tmp_path / "topology.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(TopologyError):
        CameraNetwork.from_yaml(path)

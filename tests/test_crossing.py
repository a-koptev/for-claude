"""Детектор пересечений линий и команды главному трекеру (синтетические траектории)."""
import pytest

from TopologyModule.CameraNetwork import CameraNetwork
from TopologyModule.CrossingDetector import CrossingDetector
from TopologyModule.TransitionLine import Role
from TopologyModule.TransitionTracker import CommandType, TransitionTracker, create_trackers


def make_network():
    """
    Камера 4 соединена с камерой 1 линией на y=360 (вниз = уезжаем в камеру 1, вверх = приехали из 1).
    Камера 1 соединена с камерой 4 вертикальной линией на x=900 (вправо = уезжаем в камеру 4).
    """
    return CameraNetwork.from_dict({
        "frame_size": [1280, 720],
        "cameras": [
            {"id": 1, "lines": [{"id": 1, "p1": [900, 50], "p2": [900, 650], "peer": 4, "exit": "RIGHT"}]},
            {"id": 4, "lines": [{"id": 1, "p1": [100, 360], "p2": [1180, 360], "peer": 1, "exit": "DOWN"}]},
        ],
    })


def box(cx, cy, w=100, h=100):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def detector(camera_id=4, **kwargs):
    kwargs.setdefault("min_move_px", 15)
    kwargs.setdefault("release_frames", 3)
    kwargs.setdefault("release_margin_px", 30)
    kwargs.setdefault("forget_frames", 50)
    return CrossingDetector(make_network().camera(camera_id), **kwargs)


def drive(det, track_id, points, start_frame=0):
    """Прогоняет трек по центрам; возвращает все события."""
    events = []
    for i, (cx, cy) in enumerate(points):
        events += det.update(start_frame + i, [(track_id, box(cx, cy))])
    return events


def vertical_path(x, y_from, y_to, step=20):
    sign = 1 if y_to > y_from else -1
    return [(x, y) for y in range(y_from, y_to + sign, sign * step)]


def test_driving_down_through_line_is_exit_to_peer_camera():
    events = drive(detector(), 7, vertical_path(640, 100, 600))

    assert len(events) == 1
    event = events[0]
    assert (event.camera_id, event.line_id, event.track_id) == (4, 1, 7)
    assert event.role is Role.EXIT
    assert (event.from_camera, event.to_camera) == (4, 1)


def test_driving_up_through_line_is_enter_from_peer_camera():
    events = drive(detector(), 7, vertical_path(640, 600, 100))

    assert [e.role for e in events] == [Role.ENTER]
    assert (events[0].from_camera, events[0].to_camera) == (1, 4)


def test_event_is_given_once_per_crossing():
    det = detector()
    events = drive(det, 7, vertical_path(640, 100, 600))
    assert len(events) == 1
    # дальше машина стоит далеко от линии - новых событий нет
    assert drive(det, 7, [(640, 600)] * 10, start_frame=100) == []


def test_two_cars_in_opposite_directions_get_different_events():
    det = detector()
    down = vertical_path(300, 100, 600)
    up = vertical_path(900, 600, 100)
    events = []
    for i in range(len(down)):
        events += det.update(i, [(1, box(*down[i])), (2, box(*up[i]))])

    by_track = {e.track_id: e.role for e in events}
    assert by_track == {1: Role.EXIT, 2: Role.ENTER}


def test_two_cars_same_direction_each_get_own_event():
    det = detector()
    first = vertical_path(300, 100, 600)
    second = vertical_path(900, 20, 520)   # едет следом, пересекает позже
    events = []
    for i in range(len(first)):
        events += det.update(i, [(1, box(*first[i])), (2, box(*second[i]))])

    assert sorted(e.track_id for e in events) == [1, 2]
    assert {e.role for e in events} == {Role.EXIT}
    first_frame = {e.track_id: e.frame_id for e in events}
    assert first_frame[1] < first_frame[2]   # порядок пересечения сохраняется в frame_id


def test_car_that_does_not_reach_the_line_gives_no_event():
    assert drive(detector(), 7, vertical_path(640, 100, 250)) == []


def test_car_driving_along_the_line_gives_no_event():
    # горизонтальная линия, машина едет по горизонтали на её уровне
    assert drive(detector(), 7, [(x, 360) for x in range(100, 1100, 20)]) == []


def test_stationary_car_on_the_line_gives_no_event():
    assert drive(detector(), 7, [(640, 360)] * 30) == []


def test_jitter_around_the_line_does_not_repeat_the_event():
    det = detector(release_frames=3)
    # дошёл до линии (нижний край bbox на y=370 > 360), дальше дрожит: то касается, то нет
    path = vertical_path(640, 100, 320) + [(640, 300), (640, 305), (640, 320), (640, 300)] * 5
    events = drive(det, 7, path)

    assert [e.role for e in events] == [Role.EXIT]


def test_track_that_appears_on_the_line_gets_direction_after_moving():
    det = detector()
    # сразу в кадре на линии, едет вверх
    events = drive(det, 7, vertical_path(640, 400, 250, step=10))

    assert [e.role for e in events] == [Role.ENTER]


def test_crossing_again_after_leaving_the_line_gives_new_event():
    det = detector(release_frames=2)
    down = vertical_path(640, 100, 600)
    up = vertical_path(640, 600, 100)
    events = drive(det, 7, down + up)

    assert [e.role for e in events] == [Role.EXIT, Role.ENTER]


def test_vertical_line_on_camera_1_exit_to_the_right():
    det = detector(camera_id=1)
    events = drive(det, 3, [(x, 300) for x in range(500, 1200, 25)])

    assert [e.role for e in events] == [Role.EXIT]
    assert events[0].to_camera == 4


def test_old_tracks_are_forgotten():
    det = detector(forget_frames=5)
    drive(det, 7, vertical_path(640, 100, 200))
    det.update(100, [])

    assert det.dropped_tracks == [7]
    det.update(101, [])
    assert det.dropped_tracks == []


# --- команды главному трекеру -------------------------------------------------------

def test_exit_gives_search_command_for_peer_camera():
    network = make_network()
    got = []
    tracker = TransitionTracker(network, 4, on_command=got.append, detector=detector())

    commands = []
    for i, point in enumerate(vertical_path(640, 100, 600)):
        commands += tracker.update(i, [(5, box(*point))])

    assert len(commands) == 1 and got == commands
    command = commands[0]
    assert command.type is CommandType.SEARCH_ON_CAMERA
    # авто уехало с камеры 4 в камеру 1: на камере 1 нужно искать трек 5 с камеры 4
    assert (command.camera_id, command.source_camera, command.track_id) == (1, 4, 5)
    assert "камере 1" in str(command)


def test_enter_gives_mark_arrived_and_makes_track_eligible():
    tracker = TransitionTracker(make_network(), 4, detector=detector())

    commands = []
    for i, point in enumerate(vertical_path(640, 600, 100)):
        commands += tracker.update(i, [(9, box(*point))])

    assert [c.type for c in commands] == [CommandType.MARK_ARRIVED]
    assert (commands[0].camera_id, commands[0].source_camera, commands[0].track_id) == (4, 1, 9)
    assert tracker.arrived_from(9) == 1
    assert tracker.is_eligible(9, from_camera=1)
    assert not tracker.is_eligible(9, from_camera=2)
    assert tracker.tracks_arrived_from(1) == [9]


def test_filter_lets_through_only_cars_that_came_from_the_searched_camera():
    tracker = TransitionTracker(make_network(), 4, detector=detector())
    came_from_1 = vertical_path(300, 600, 100)
    drove_away = vertical_path(900, 100, 600)
    standing = [(1100, 200)] * len(came_from_1)

    for i in range(len(came_from_1)):
        tracker.update(i, [(1, box(*came_from_1[i])), (2, box(*drove_away[i])), (3, box(*standing[i]))])

    candidates = [t for t in (1, 2, 3) if tracker.is_eligible(t, from_camera=1)]
    assert candidates == [1]


def test_arrivals_are_forgotten_with_the_track():
    detector_ = detector(forget_frames=5)
    tracker = TransitionTracker(make_network(), 4, detector=detector_)
    for i, point in enumerate(vertical_path(640, 600, 100)):
        tracker.update(i, [(9, box(*point))])
    assert tracker.arrived_from(9) == 1

    tracker.update(500, [])

    assert tracker.arrived_from(9) is None


def test_create_trackers_for_every_camera():
    trackers = create_trackers(make_network())

    assert sorted(trackers) == [1, 4]

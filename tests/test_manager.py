import numpy as np
import pytest

import TrackletModule.TrackletManager as manager_module
from Config import Config
from TrackletModule.Tracklet import Tracklet, TrackletState
from tests.conftest import unit

FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)


class FakeFeaturesManager:
    def __init__(self, *args, **kwargs):
        pass

    def extract_features(self, images):
        return [unit(0) for _ in images]


class FakeDatabaseManager:
    def __init__(self, **kwargs):
        pass

    def connect(self):
        return True


class FakeRepository:
    gallery = {}

    def __init__(self, db_manager):
        pass

    def get_features_by_car_ids(self, ids):
        return {car_id: self.gallery[car_id] for car_id in ids if car_id in self.gallery}


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setattr(manager_module, "FeaturesManager", FakeFeaturesManager)
    monkeypatch.setattr(manager_module, "DatabaseManager", FakeDatabaseManager)
    monkeypatch.setattr(manager_module, "CarFeatureRepository", FakeRepository)
    monkeypatch.setattr(Tracklet, "_tracklet", -1)
    return manager_module.TrackletManager(
        cam_id=0, similarity_threshold=0.63, frames_for_confirm=3, frames_for_lost=4
    )


def box(x, y=100, size=100):
    return np.array([[x, y, x + size, y + size]], dtype=float)


def ids(*values):
    return np.array(values, dtype=int)


def empty_update(manager, frame_id):
    manager.update(np.empty((0, 4)), np.empty((0,), dtype=int), frame_id, FRAME)


# --- B1: движение обновляется ровно один раз за кадр ---------------------------

def test_movement_updated_once_per_frame(manager, monkeypatch):
    calls = []
    original = Tracklet.update_track_move

    def counting(self, new_bbox):
        calls.append(self.track_id)
        return original(self, new_bbox)

    monkeypatch.setattr(Tracklet, "update_track_move", counting)

    manager.update(box(100), ids(1), 0, FRAME)    # создание
    manager.update(box(100), ids(1), 1, FRAME)    # инициализация EMA
    calls.clear()
    manager.update(box(102.2), ids(1), 2, FRAME)  # сдвиг на границе порога движения

    assert calls == [1]
    tracklet = manager.t_storage.storage[TrackletState.MOVE_UNMATCHED][1]
    assert tracklet.is_moving


# --- B4: update_searched_vehicle не возвращает уже найденные машины ----------------

def test_update_searched_vehicle_keeps_found_out_of_searched(manager):
    FakeRepository.gallery = {22: np.array([unit(0)]), 23: np.array([unit(1)])}
    manager.f_storage.found = {22: np.array([unit(0)])}

    manager.update_searched_vehicle(["22", "23"])

    assert set(manager.f_storage.searched) == {23}
    assert set(manager.f_storage.found) == {22}


def test_losing_a_track_does_not_reload_searched(manager):
    FakeRepository.gallery = {22: np.array([unit(0)])}
    manager.update_searched_vehicle(["22"])
    manager.f_storage.vehicle_found(22)

    manager.update(box(100), ids(1), 0, FRAME)
    empty_update(manager, 1)  # трек пропал -> LOST

    assert 22 in manager.f_storage.found
    assert 22 not in manager.f_storage.searched


# --- B5: frames_for_lost доходит до Matcher, LOST-tracklet'ы закрываются ---------

def test_frames_for_lost_reaches_matcher(manager):
    assert manager.matcher.frames_for_lost == 4


def test_lost_tracklet_is_closed_and_confirmed_vehicle_released(manager, monkeypatch):
    monkeypatch.setattr(Config, "TRACKLET_LOST_TTL_FRAMES", 10)
    manager.update(box(100), ids(1), 0, FRAME)
    tracklet = manager.t_storage.storage[TrackletState.STAY_UNMATCHED][1]
    manager.t_storage.change_state(TrackletState.CONFIRMED, tracklet)
    tracklet.vehicle_id = 5
    manager.f_storage.found = {5: np.array([unit(0)])}

    empty_update(manager, 1)
    assert 1 in manager.t_storage.storage[TrackletState.LOST]
    assert 5 in manager.f_storage.found                      # пока ждём возвращения трека

    empty_update(manager, 11)                                # TTL истёк
    assert manager.t_storage.storage[TrackletState.LOST] == {}
    assert 1 not in manager.known_track_ids
    assert 5 in manager.f_storage.searched                   # машина снова ищется
    assert 5 not in manager.f_storage.found


def test_returned_track_goes_back_to_its_state(manager):
    manager.update(box(100), ids(1), 0, FRAME)
    empty_update(manager, 1)
    assert 1 in manager.t_storage.storage[TrackletState.LOST]

    manager.update(box(100), ids(1), 2, FRAME)

    assert manager.t_storage.storage[TrackletState.LOST] == {}
    assert 1 in manager.t_storage.storage[TrackletState.STAY_UNMATCHED]


# --- B9: плохие кропы не ломают извлечение признаков -------------------------------

def test_tiny_and_empty_crops_are_skipped(manager):
    tracks = [(1, np.array([10, 10, 13, 13.])), (2, np.array([-50, -50, -10, -10.])), (3, np.array([100, 100, 220, 220.]))]

    valid_tracks, features = manager.extract_features(tracks, FRAME)

    assert [track_id for track_id, _ in valid_tracks] == [3]
    assert len(features) == 1


def test_bbox_partially_outside_frame_is_clamped(manager):
    crop = manager.crop_bbox(FRAME, np.array([-30, -30, 100, 100.]))
    assert crop.shape[:2] == (100, 100)

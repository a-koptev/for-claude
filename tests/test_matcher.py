import numpy as np
import pytest

from Config import Config
from FeaturesModule.FeaturesManager import FeaturesManager
from TrackletModule.FindStorage import FindStorage
from TrackletModule.Matcher import Matcher
from TrackletModule.Tracklet import Tracklet, TrackletState
from TrackletModule.TrackletStorage import TrackletStorage
from tests.conftest import unit


def make_matcher(threshold=0.63, confirm=3, lost=5):
    t_storage = TrackletStorage()
    f_storage = FindStorage()
    # FeaturesManager без загрузки модели: нужен только статический calculate_similarity
    feature_manager = FeaturesManager.__new__(FeaturesManager)
    matcher = Matcher(feature_manager, t_storage, f_storage, threshold, confirm, lost)
    return matcher, t_storage, f_storage


def add_tracklet(t_storage, track_id, features, state=TrackletState.MOVE_UNMATCHED, vehicle_id=None):
    tracklet = Tracklet(cam_id=0, track_id=track_id, bbox=None)
    tracklet.features = list(features)
    t_storage.add_tracklet(tracklet)
    if state is not TrackletState.STAY_UNMATCHED:
        t_storage.change_state(new_state=state, tracklet=tracklet)
    tracklet.vehicle_id = vehicle_id
    return tracklet


# --- B2: оценка считается отдельно для каждой пары (машина, tracklet) ---------

def test_similarity_does_not_leak_between_tracklets():
    matcher, t_storage, f_storage = make_matcher()
    track_a = add_tracklet(t_storage, 10, [unit(0)])
    track_b = add_tracklet(t_storage, 11, [unit(1)])  # совсем другая машина

    near_zero = unit(0) * 0.9 + unit(2) * 0.1  # cos с A ~ 0.994
    f_storage.searched = {1: np.array([unit(0)]), 2: np.array([near_zero])}

    pairs = matcher._match(f_storage.searched, t_storage.storage[TrackletState.MOVE_UNMATCHED])

    # Машина 1 забирает A, машина 2 НЕ должна достаться чужому B
    assert [(v, t.track_id) for v, t in pairs] == [(1, 10)]


def test_tracklet_without_features_is_skipped():
    matcher, t_storage, f_storage = make_matcher()
    add_tracklet(t_storage, 10, [])  # раньше IndexError на similarities[0]
    f_storage.searched = {1: np.array([unit(0)])}

    assert matcher._match(f_storage.searched, t_storage.storage[TrackletState.MOVE_UNMATCHED]) == []


# --- B3: CONFIRMED сравниваем с CONFIRMED, а не с CANDIDATE -------------------

def test_confirmed_tracklet_stays_confirmed_while_it_matches():
    matcher, t_storage, f_storage = make_matcher(lost=5)
    f_storage.found = {7: np.array([unit(0)])}
    tracklet = add_tracklet(t_storage, 10, [unit(0)], TrackletState.CONFIRMED, vehicle_id=7)

    for _ in range(10):  # сильно больше frames_for_lost
        matcher.matching_confirmed_tracklets()

    assert tracklet.state is TrackletState.CONFIRMED
    assert tracklet.vehicle_id == 7
    assert 7 in f_storage.found


def test_confirmed_tracklet_is_lost_when_it_stops_matching():
    matcher, t_storage, f_storage = make_matcher(lost=3)
    f_storage.found = {7: np.array([unit(0)])}
    tracklet = add_tracklet(t_storage, 10, [unit(1)], TrackletState.CONFIRMED, vehicle_id=7)

    for _ in range(3):
        matcher.matching_confirmed_tracklets()

    assert tracklet.state is TrackletState.CANDIDATE
    assert tracklet.vehicle_id is None
    assert 7 in f_storage.searched and 7 not in f_storage.found


# --- B7: подтверждение только по идущим подряд совпадениям --------------------

def test_candidate_needs_consecutive_matches():
    matcher, t_storage, f_storage = make_matcher(confirm=3)
    f_storage.searched = {1: np.array([unit(0)])}
    tracklet = add_tracklet(t_storage, 10, [unit(0)])
    t_storage.move_2_candidate(tracklet=tracklet, vehicle_id=1)  # счётчик = 1

    matcher.matching_candidate_tracklets()          # совпадение -> 2
    tracklet.features = [unit(1)]
    matcher.matching_candidate_tracklets()          # промах -> 1
    tracklet.features = [unit(0)]
    matcher.matching_candidate_tracklets()          # совпадение -> 2

    # Раньше счётчик не убывал и здесь уже происходило подтверждение
    assert tracklet.state is TrackletState.CANDIDATE
    assert tracklet.match_count_by_vehicle_id == {1: 2}

    matcher.matching_candidate_tracklets()          # -> 3
    assert tracklet.state is TrackletState.CONFIRMED
    assert tracklet.vehicle_id == 1
    assert 1 in f_storage.found


def test_candidate_falls_back_to_unmatched_when_matches_vanish():
    matcher, t_storage, f_storage = make_matcher()
    f_storage.searched = {1: np.array([unit(0)])}
    tracklet = add_tracklet(t_storage, 10, [unit(1)])
    t_storage.move_2_candidate(tracklet=tracklet, vehicle_id=1)

    matcher.matching_candidate_tracklets()

    assert tracklet.state is TrackletState.MOVE_UNMATCHED
    assert tracklet.match_count_by_vehicle_id == {}


def test_candidate_outside_scope_is_not_touched():
    """Tracklet, чьи признаки в этом кадре не обновлялись, не получает ни +1, ни штраф."""
    matcher, t_storage, f_storage = make_matcher()
    f_storage.searched = {1: np.array([unit(0)])}
    tracklet = add_tracklet(t_storage, 10, [unit(0)])
    t_storage.move_2_candidate(tracklet=tracklet, vehicle_id=1)

    matcher.matching_candidate_tracklets(track_ids=[])

    assert tracklet.match_count_by_vehicle_id == {1: 1}


# --- B8: настраиваемая агрегация сходства ------------------------------------

@pytest.mark.parametrize("mode, expected_pairs", [("max", 1), ("topk_mean", 0)])
def test_similarity_aggregation_mode(monkeypatch, mode, expected_pairs):
    monkeypatch.setattr(Config, "MATCH_SIM_AGGREGATION", mode)
    monkeypatch.setattr(Config, "MATCH_SIM_TOPK", 2)
    matcher, t_storage, f_storage = make_matcher(threshold=0.63)
    # одна пара идеальна (1.0), другая нулевая: max = 1.0, mean(top-2) = 0.5
    add_tracklet(t_storage, 10, [unit(0), unit(1)])
    f_storage.searched = {1: np.array([unit(0)])}

    pairs = matcher._match(f_storage.searched, t_storage.storage[TrackletState.MOVE_UNMATCHED])

    assert len(pairs) == expected_pairs


# --- FindStorage -------------------------------------------------------------

def test_find_storage_is_safe_for_unknown_ids():
    storage = FindStorage()
    storage.vehicle_found(99)  # раньше KeyError
    storage.vehicle_lost(99)
    assert storage.searched == {} and storage.found == {}

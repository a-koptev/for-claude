from typing import Dict

from TrackletModule.Tracklet import Tracklet, TrackletState


class TrackletStorage:
    def __init__(self):
        self.storage: Dict[TrackletState, Dict[int, Tracklet]] = {
            TrackletState.STAY_UNMATCHED: {},
            TrackletState.MOVE_UNMATCHED: {},
            TrackletState.CANDIDATE: {},
            TrackletState.CONFIRMED: {},
            TrackletState.LOST: {},
        }

    def add_tracklet(self, tracklet: Tracklet) -> None:
        self.storage[TrackletState.STAY_UNMATCHED][tracklet.track_id] = tracklet

    def change_state(self, new_state: TrackletState, tracklet: Tracklet):
        old_state = tracklet.state
        self.storage[new_state][tracklet.track_id] = tracklet
        del self.storage[old_state][tracklet.track_id]
        tracklet.state = new_state

    def stay_to_move(self, tracklet: Tracklet) -> None:
        self.change_state(
            new_state=TrackletState.MOVE_UNMATCHED,
            tracklet=tracklet
        )

    def move_to_stay(self, tracklet: Tracklet) -> None:
        self.change_state(
            new_state=TrackletState.STAY_UNMATCHED,
            tracklet=tracklet
        )

    def move_2_candidate(self, tracklet: Tracklet, vehicle_id: int) -> None:
        tracklet.match_count_by_vehicle_id[vehicle_id] = 1

        self.change_state(
            new_state=TrackletState.CANDIDATE,
            tracklet=tracklet
        )

    def candidate_2_confirmed(self, tracklet: Tracklet, vehicle_id: int) -> None:
        tracklet.vehicle_id = vehicle_id
        tracklet.match_count_by_vehicle_id = {}

        # Удаляем vehicle_id из мэтчинга у остальных
        for candidate_tracklet in self.storage[TrackletState.CANDIDATE].values():
            if vehicle_id in candidate_tracklet.match_count_by_vehicle_id:
                candidate_tracklet.match_count_by_vehicle_id.pop(vehicle_id, None)

        self.change_state(
            new_state=TrackletState.CONFIRMED,
            tracklet=tracklet
        )

    def confirmed_2_candidate(self, tracklet: Tracklet) -> None:
        tracklet.match_count_by_vehicle_id[tracklet.vehicle_id] = 1

        tracklet.vehicle_id = None
        tracklet.lost_counter = 0

        self.change_state(
            new_state=TrackletState.CANDIDATE,
            tracklet=tracklet
        )

    def candidate_2_unmatched(self, tracklet: Tracklet) -> None:
        """Кандидат потерял все совпадения -> возвращается к обычному поиску."""
        tracklet.match_count_by_vehicle_id = {}

        self.change_state(
            new_state=TrackletState.MOVE_UNMATCHED,
            tracklet=tracklet
        )

    def close_lost(self, tracklet: Tracklet) -> None:
        """Окончательно удаляет LOST-tracklet из хранилища."""
        self.storage[TrackletState.LOST].pop(tracklet.track_id, None)

    def lost(self, tracklet: Tracklet):
        self.storage[TrackletState.LOST][tracklet.track_id] = tracklet
        del self.storage[tracklet.state][tracklet.track_id]

    def unlost(self, tracklet: Tracklet):
        self.storage[tracklet.state][tracklet.track_id] = tracklet
        del self.storage[TrackletState.LOST][tracklet.track_id]


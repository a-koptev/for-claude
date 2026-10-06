import logging
from typing import Dict

from TrackletModule.Tracklet import Tracklet, TrackletState
from TrackletModule.TrackletStorage import TrackletStorage
from TrackletModule.Matcher import Matcher
from FeaturesModule.FeaturesManager import FeaturesManager
from TrackletModule.FindStorage import FindStorage
from DB.database_manager import DatabaseManager
from DB.features.feature_repository import CarFeatureRepository
from Config import Config

logger = logging.getLogger(__name__)


class TrackletManager:
    def __init__(self, cam_id: int, similarity_threshold, frames_for_confirm, frames_for_lost):
        self.cam_id: int = cam_id
        self.similarity_threshold = similarity_threshold
        self.frames_for_confirm = frames_for_confirm
        self.frames_for_lost = frames_for_lost

        self.features_manager = FeaturesManager(Config.OSNET_MODEL_PATH)
        self.t_storage = TrackletStorage()
        self.f_storage = FindStorage()

        self.matcher: Matcher = Matcher(
            feature_manager=self.features_manager,
            tracklet_storage=self.t_storage,
            find_storage=self.f_storage,
            similarity_threshold=self.similarity_threshold,
            frames_for_confirm=self.frames_for_confirm,
            frames_for_lost=self.frames_for_lost
        )

        self.matchers = {
            TrackletState.STAY_UNMATCHED: None,
            TrackletState.MOVE_UNMATCHED: self.matcher.matching_unmatched_tracklets,
            TrackletState.CANDIDATE: self.matcher.matching_candidate_tracklets,
            TrackletState.CONFIRMED: self.matcher.matching_confirmed_tracklets,
            TrackletState.LOST: None,
        }
        self.buckets = {state: [] for state in TrackletState}

        db_manager = DatabaseManager(**Config.DB_PARAMS)
        db_manager.connect()
        self.car_feature_repository = CarFeatureRepository(db_manager=db_manager)

        self.known_track_ids: set[int] = set()

    def update(self, byte_track_bboxes, byte_track_ids, frame_id, frame):
        self.buckets = {state: [] for state in TrackletState}

        # Вернувшиеся треки сначала возвращаем из LOST в их рабочее состояние,
        # чтобы в этом же кадре они обрабатывались как обычные
        self.restore_returned_tracks(byte_track_ids=byte_track_ids)

        for track_id, bbox in zip(byte_track_ids, byte_track_bboxes):
            if track_id not in self.known_track_ids:
                self.known_track_ids.add(track_id)
                self.create_tracklet(track_id=track_id, bbox=bbox, frame_id=frame_id)
                continue

            for state, storage in self.t_storage.storage.items():
                if track_id in storage:
                    tracklet = storage[track_id]
                    tracklet.last_seen_frame = frame_id
                    if tracklet.should_reid(frame_id=frame_id):
                        self.buckets[state].append((track_id, bbox))
                    break

        # Движение считаем один раз за кадр для всех STAY/MOVE
        self.update_moving()

        for state in TrackletState:
            if self.buckets[state]:
                self.process_tracklets(state=state, frame_id=frame_id, frame=frame)

        self.move_missing_tracks_to_lost(byte_track_ids=byte_track_ids, frame_id=frame_id)

    def restore_returned_tracks(self, byte_track_ids):
        lost_storage = self.t_storage.storage[TrackletState.LOST]
        for track_id in byte_track_ids:
            if track_id in lost_storage:
                self.t_storage.unlost(tracklet=lost_storage[track_id])

    def move_missing_tracks_to_lost(self, byte_track_ids, frame_id):
        tracks = set(byte_track_ids)
        for state in TrackletState:
            if state is TrackletState.LOST:
                continue
            for tracklet in list(self.t_storage.storage[state].values()):
                if tracklet.track_id not in tracks:
                    logger.debug(f"tracklet {tracklet.tracklet_id} LOST")
                    self.t_storage.lost(tracklet=tracklet)

        self.close_expired_lost(frame_id=frame_id)

    def close_expired_lost(self, frame_id):
        """Закрывает tracklet'ы, которых нет дольше TRACKLET_LOST_TTL_FRAMES кадров."""
        ttl = Config.TRACKLET_LOST_TTL_FRAMES
        for tracklet in list(self.t_storage.storage[TrackletState.LOST].values()):
            if frame_id - tracklet.last_seen_frame < ttl:
                continue

            # Для LOST состояние tracklet'а - то, в котором он был до потери
            if tracklet.state is TrackletState.CONFIRMED and tracklet.vehicle_id is not None:
                self.f_storage.vehicle_lost(vehicle_id=tracklet.vehicle_id)

            logger.debug(f"tracklet {tracklet.tracklet_id} CLOSED")
            self.t_storage.close_lost(tracklet=tracklet)
            self.known_track_ids.discard(tracklet.track_id)

    def create_tracklet(self, track_id, bbox, frame_id=-1):
        tracklet = Tracklet(
            cam_id=self.cam_id,
            track_id=track_id,
            bbox=bbox,
        )
        tracklet.last_seen_frame = frame_id
        self.t_storage.add_tracklet(tracklet=tracklet)

    def update_moving(self):
        # TODO (подумать над unconfirmed и confirmed)
        # ВРЕМЕННО ТОЛЬКО UNMATCHED
        moving_states = (TrackletState.STAY_UNMATCHED, TrackletState.MOVE_UNMATCHED)

        # Снимок корзин до любых перекладываний: tracklet, перешедший из одной
        # корзины в другую, не должен обновляться во второй раз за тот же кадр
        snapshot = {state: list(self.buckets[state]) for state in moving_states}
        new_buckets = {state: [] for state in moving_states}

        for state in moving_states:
            storage = self.t_storage.storage[state]
            for track_id, bbox in snapshot[state]:
                tracklet = storage[track_id]
                tracklet.update_track_move(new_bbox=bbox)

                new_state = TrackletState.MOVE_UNMATCHED if tracklet.is_moving else TrackletState.STAY_UNMATCHED
                if new_state != state:
                    self.t_storage.change_state(
                        new_state=new_state,
                        tracklet=tracklet
                    )
                new_buckets[new_state].append((track_id, bbox))

        for state in moving_states:
            self.buckets[state] = new_buckets[state]

    def process_tracklets(self, state, frame_id, frame):
        if state is TrackletState.STAY_UNMATCHED:
            return

        tracks = self.buckets[state]
        tracklets = self.t_storage.storage[state]
        match_fun = self.matchers[state]

        valid_tracks, features = self.extract_features(tracks=tracks, frame=frame)
        if not valid_tracks:
            return

        self.update_features(
            tracks=valid_tracks,
            tracklets=tracklets,
            features=features,
            frame_id=frame_id
        )
        if match_fun:
            # Матчим только те tracklet'ы, у которых признаки обновлены в этом кадре
            match_fun(track_ids=[track_id for track_id, _ in valid_tracks])

    def extract_features(self, tracks, frame):
        """Возвращает (tracks с пригодным кропом, их признаки). Пустые/крошечные кропы пропускаются."""
        valid_tracks = []
        track_images = []
        for track_id, bbox in tracks:
            crop = self.crop_bbox(frame, bbox)
            if crop is None:
                continue
            valid_tracks.append((track_id, bbox))
            track_images.append(crop)

        if not track_images:
            return [], []
        return valid_tracks, self.features_manager.extract_features(track_images)

    @staticmethod
    def update_features(tracks, tracklets, features, frame_id):
        for (track_id, _), feature in zip(tracks, features):
            tracklets[track_id].update_feature(feature)
            tracklets[track_id].mark_reid(frame_id=frame_id)

    @staticmethod
    def crop_bbox(frame, bbox):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        min_size = Config.FEATURE_MIN_CROP_SIZE
        if x2 - x1 < min_size or y2 - y1 < min_size:
            return None
        return frame[y1:y2, x1:x2]

    def update_searched_vehicle(self, searched_ids):
        """
        Задаёт, каких машин ищем. Уже найденные (found) в searched не возвращаются,
        поэтому метод безопасно вызывать в любой момент.
        """
        ids = [int(car_id) for car_id in searched_ids]
        features_by_ids = self.car_feature_repository.get_features_by_car_ids(ids)
        self.f_storage.searched = {
            vehicle_id: features
            for vehicle_id, features in features_by_ids.items()
            if vehicle_id not in self.f_storage.found
        }

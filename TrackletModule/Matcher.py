import logging
from typing import Dict, Iterable, Optional

import numpy as np

from Config import Config
from FeaturesModule.FeaturesManager import FeaturesManager
from TrackletModule.FindStorage import FindStorage
from TrackletModule.Tracklet import Tracklet, TrackletState
from TrackletModule.TrackletStorage import TrackletStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Matcher:
    def __init__(
            self,
            feature_manager: FeaturesManager,
            tracklet_storage: TrackletStorage,
            find_storage: FindStorage,
            similarity_threshold: float = 0.7,
            frames_for_confirm: int = 3,
            frames_for_lost: int = 5
    ):
        self.fm = feature_manager
        self.t_storage = tracklet_storage
        self.f_storage = find_storage
        self.similarity_threshold = similarity_threshold
        self.frames_for_confirm = frames_for_confirm
        self.frames_for_lost = frames_for_lost

    def _scope(
            self,
            state: TrackletState,
            track_ids: Optional[Iterable]
    ) -> Dict[int, Tracklet]:
        """
        Tracklet'ы состояния `state`, участвующие в этом цикле матчинга.
        track_ids - только те, у кого признаки обновлены в текущем кадре
        (иначе один и тот же признак учитывался бы повторно, а tracklet,
        только что повышенный в статусе, получал бы совпадение в том же кадре).
        """
        storage = self.t_storage.storage[state]
        if track_ids is None:
            return dict(storage)
        return {track_id: storage[track_id] for track_id in track_ids if track_id in storage}

    def matching_unmatched_tracklets(self, track_ids: Optional[Iterable] = None) -> None:
        tracklets = self._scope(TrackletState.MOVE_UNMATCHED, track_ids)

        result_pairs = self._match(
            vehicles=self.f_storage.searched,
            tracklets=tracklets
        )

        for vehicle_id, tracklet in result_pairs:
            logger.info(f"ДЕЛАЕМ ТРЕКЛЕТ CANDIDATE veh_{vehicle_id}, track_{tracklet.track_id}")
            self.t_storage.move_2_candidate(vehicle_id=vehicle_id, tracklet=tracklet)

    def matching_candidate_tracklets(self, track_ids: Optional[Iterable] = None) -> None:
        tracklets = self._scope(TrackletState.CANDIDATE, track_ids)

        result_pairs = self._match(
            vehicles=self.f_storage.searched,
            tracklets=tracklets
        )
        matched_vehicle_by_tracklet = {
            tracklet.tracklet_id: vehicle_id for vehicle_id, tracklet in result_pairs
        }

        penalty = Config.CANDIDATE_MISS_PENALTY
        for tracklet in list(tracklets.values()):
            counts = tracklet.match_count_by_vehicle_id
            matched_vehicle_id = matched_vehicle_by_tracklet.get(tracklet.tracklet_id)

            # Совпадения должны идти подряд: по остальным машинам счётчик затухает
            for vehicle_id in list(counts):
                if vehicle_id == matched_vehicle_id:
                    continue
                counts[vehicle_id] -= penalty
                if counts[vehicle_id] <= 0:
                    del counts[vehicle_id]

            if matched_vehicle_id is not None:
                counts[matched_vehicle_id] = counts.get(matched_vehicle_id, 0) + 1
                if counts[matched_vehicle_id] >= self.frames_for_confirm:
                    logger.info(
                        f"!!!!!!!!!!!!!! CONFIRMED !!!!!!!!!!!  "
                        f"veh_{matched_vehicle_id}, track_{tracklet.track_id}"
                    )
                    self.t_storage.candidate_2_confirmed(tracklet=tracklet, vehicle_id=matched_vehicle_id)
                    self.f_storage.vehicle_found(vehicle_id=matched_vehicle_id)
                    continue

            # Все совпадения потеряны -> обратно в обычный поиск
            if tracklet.state is TrackletState.CANDIDATE and not tracklet.match_count_by_vehicle_id:
                self.t_storage.candidate_2_unmatched(tracklet=tracklet)

    def matching_confirmed_tracklets(self, track_ids: Optional[Iterable] = None) -> None:
        # Раньше здесь сравнивались CANDIDATE-tracklet'ы, а не CONFIRMED
        tracklets = self._scope(TrackletState.CONFIRMED, track_ids)

        result_pairs = self._match(
            vehicles=self.f_storage.found,
            tracklets=tracklets
        )

        matched_tracklet_by_vehicle = {
            vehicle_id: tracklet
            for vehicle_id, tracklet in result_pairs
        }

        # lost_counter считается в циклах Re-ID подтверждённых tracklet'ов
        # (интервал TrackletState.CONFIRMED.reid_interval кадров), а не в кадрах
        for tracklet in list(tracklets.values()):
            vehicle_id = tracklet.vehicle_id
            if matched_tracklet_by_vehicle.get(vehicle_id) is not tracklet:
                tracklet.lost_counter += 1
                if tracklet.lost_counter >= self.frames_for_lost:
                    logger.info(
                        f"!!!!!!!!!!!!!!! LOST !!!!!!!!!!!  veh_{vehicle_id}, track_{tracklet.track_id}"
                    )
                    self.t_storage.confirmed_2_candidate(tracklet=tracklet)
                    self.f_storage.vehicle_lost(vehicle_id=vehicle_id)
            else:
                tracklet.lost_counter = max(0, tracklet.lost_counter - 1)

    @staticmethod
    def _normalize(vectors) -> np.ndarray:
        matrix = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / (norms + 1e-12)

    @staticmethod
    def _aggregate(similarities: np.ndarray) -> float:
        """Сворачивает матрицу сходств в одно число (см. Config.MATCH_SIM_AGGREGATION)."""
        flat = similarities.ravel()
        if Config.MATCH_SIM_AGGREGATION == "topk_mean":
            k = min(Config.MATCH_SIM_TOPK, flat.size)
            return float(np.sort(flat)[-k:].mean())
        return float(flat.max())

    def _match(
            self,
            vehicles: Dict[int, list[np.ndarray]],
            tracklets: Dict[int, Tracklet]
    ) -> list[tuple[int, Tracklet]]:
        candidates = []
        for vehicle_id, vehicle_features in vehicles.items():
            if len(vehicle_features) == 0:
                continue
            vehicle_matrix = self._normalize(vehicle_features)

            for tracklet in tracklets.values():
                # Нет признаков (ещё не извлекались / плохой кроп) -> сравнивать нечего
                if not tracklet.features:
                    continue

                # Оценка считается отдельно для каждой пары (машина, tracklet)
                similarities = self._normalize(tracklet.features) @ vehicle_matrix.T
                similarity = self._aggregate(similarities)

                logger.debug(f"veh_{vehicle_id} track_{tracklet.track_id} sim={similarity:.4f}")

                if similarity >= self.similarity_threshold:
                    candidates.append((vehicle_id, tracklet, similarity))

        candidates.sort(key=lambda x: x[2], reverse=True)

        used_vehicle_ids = set()
        used_tracklet_ids = set()
        pairs = []
        for vehicle_id, tracklet, similarity in candidates:
            if vehicle_id in used_vehicle_ids:
                continue
            if tracklet.tracklet_id in used_tracklet_ids:
                continue

            pairs.append((vehicle_id, tracklet))
            used_vehicle_ids.add(vehicle_id)
            used_tracklet_ids.add(tracklet.tracklet_id)

            logger.debug(f"PAIR veh_{vehicle_id} track_{tracklet.track_id} sim={similarity:.4f}")

        return pairs

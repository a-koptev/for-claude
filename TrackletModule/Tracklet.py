from enum import Enum

import numpy as np
from typing import Optional, Dict

from Config import Config


class Tracklet:
    _tracklet = -1

    def __init__(
            self,
            cam_id: int,
            track_id,
            bbox
    ):
        self.tracklet_id: int = self._get_tracklet_id()
        self.cam_id: int = cam_id
        self.bbox = bbox
        self.track_id = track_id
        self.vehicle_id: Optional[int] = None
        self.state: TrackletState = TrackletState.STAY_UNMATCHED
        self.last_reid_frame: int = -1
        # Последний кадр, на котором трек присутствовал (для закрытия LOST-tracklet'ов)
        self.last_seen_frame: int = -1

        self.features: list[np.ndarray] = []
        self.mean_feature: Optional[np.ndarray] = None

        self.match_count_by_vehicle_id: Dict[int, int] = {}
        self.lost_counter = 0

        self.ema_x = None
        self.ema_y = None
        self.is_moving = False

    @classmethod
    def _get_tracklet_id(cls) -> int:
        cls._tracklet += 1
        return cls._tracklet

    def update_track_move(self, new_bbox):
        x1, y1, x2, y2 = new_bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        object_size = (width + height) / 2
        alpha = Config.TRACKER_MOVEMENT_ALFA

        self.bbox = new_bbox

        if object_size < Config.TRACKER_MOVEMENT_MIN_SIZE:
            self.is_moving = False
            return 0.0

        if self.ema_x is None:
            self.ema_x = center_x
            self.ema_y = center_y
            self.is_moving = False
            return 0.0

        old_ema_x = self.ema_x
        old_ema_y = self.ema_y

        self.ema_x = alpha * center_x + (1 - alpha) * old_ema_x
        self.ema_y = alpha * center_y + (1 - alpha) * old_ema_y

        displacement_px = ((self.ema_x - old_ema_x) ** 2 + (self.ema_y - old_ema_y) ** 2) ** 0.5
        normalized_displacement = displacement_px / object_size

        self.is_moving = normalized_displacement >= Config.TRACKER_MOVEMENT_THRESHOLD
        return normalized_displacement

    def update_feature(self, feature):
        self.features.append(feature)
        self.features = self.features[-5:]
        self.mean_feature = np.mean(self.features, axis=0)

    def should_reid(self, frame_id: int) -> bool:
        return frame_id - self.last_reid_frame >= self.state.reid_interval

    def mark_reid(self, frame_id: int):
        self.last_reid_frame = frame_id


class TrackletState(Enum):
    STAY_UNMATCHED = 0
    MOVE_UNMATCHED = 1
    CANDIDATE = 2
    CONFIRMED = 3
    LOST = 4

    @property
    def reid_interval(self) -> int:
        return {
            TrackletState.STAY_UNMATCHED: 1,
            TrackletState.MOVE_UNMATCHED: 1,
            TrackletState.CANDIDATE: 1,
            TrackletState.CONFIRMED: 10,
            TrackletState.LOST: 5,
        }[self]

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class FindStorage:
    def __init__(self):
        self.searched: Dict[int, list] = {}
        self.found: Dict[int, list] = {}

    def vehicle_found(self, vehicle_id: int):
        features = self.searched.pop(vehicle_id, None)
        if features is not None:
            self.found[vehicle_id] = features

    def vehicle_lost(self, vehicle_id: int):
        features = self.found.pop(vehicle_id, None)
        if features is not None:
            logger.info(f"ДОБАВИЛ {vehicle_id} обратно в searched")
            self.searched[vehicle_id] = features

"""Zone-based Re-ID feature sampling with original/flip galleries."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np
import yaml


@dataclass
class TrackGallery:
    """Embeddings for one local (camera_id, track_id) observation."""
    camera_id: int
    track_id: int
    original: list[np.ndarray] = field(default_factory=list)
    flipped: list[np.ndarray] = field(default_factory=list)
    last_sample_frame: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "original": list(self.original),
            "flipped": list(self.flipped),
            "sample_count": len(self.original),
        }


class FeatureZoneManager:
    """Collect diverse embeddings while a local track is inside its capture zone.

    features_manager must implement extract_features(image_list) and
    calculate_similarity(feature_vector_1, feature_vector_2).
    Empty/unconfigured polygons safely disable sampling for that profile.
    """

    def __init__(
        self,
        features_manager,
        config_path: str | Path = "feature_zones.yaml",
        similarity_threshold: float | None = None,
        sample_every_n_frames: int | None = None,
        min_crop_size: int | None = None,
    ):
        self.features_manager = features_manager
        with open(config_path, "r", encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream) or {}
        threshold = self.config.get("similarity_threshold", 0.90) if similarity_threshold is None else similarity_threshold
        interval = self.config.get("sample_every_n_frames", 3) if sample_every_n_frames is None else sample_every_n_frames
        crop_size = self.config.get("min_crop_size", 8) if min_crop_size is None else min_crop_size
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if interval < 1:
            raise ValueError("sample_every_n_frames must be >= 1")
        if crop_size < 1:
            raise ValueError("min_crop_size must be >= 1")
        self.similarity_threshold = float(threshold)
        self.sample_every_n_frames = int(interval)
        self.min_crop_size = int(crop_size)
        self._galleries: dict[tuple[int, int], TrackGallery] = {}
        self._polygons = self._parse_polygons(self.config.get("cameras", {}))

    @staticmethod
    def _parse_polygons(cameras: Mapping) -> dict[int, dict[str, np.ndarray]]:
        parsed: dict[int, dict[str, np.ndarray]] = {}
        for camera_key, camera_cfg in cameras.items():
            try:
                camera_id = int(camera_key)
            except (TypeError, ValueError):
                continue
            profiles: dict[str, np.ndarray] = {}
            for profile_name, points in (camera_cfg or {}).items():
                if not points:
                    continue
                polygon = np.asarray(points, dtype=np.int32)
                if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
                    raise ValueError(
                        f"Camera {camera_id} profile {profile_name!r} needs >= 3 [x, y] points"
                    )
                profiles[str(profile_name).upper()] = polygon
            parsed[camera_id] = profiles
        return parsed

    def _profile_polygon(self, camera_id: int, entry_gate: str | None) -> np.ndarray | None:
        profiles = self._polygons.get(int(camera_id), {})
        if int(camera_id) in (2, 3):
            if not entry_gate:
                return None
            return profiles.get(str(entry_gate).upper())
        return profiles.get("DEFAULT")

    @staticmethod
    def _anchor(bbox) -> tuple[float, float]:
        x1, y1, x2, y2 = map(float, bbox)
        return (x1 + x2) / 2.0, y2

    @staticmethod
    def _inside_polygon(bbox, polygon: np.ndarray) -> bool:
        x, y = FeatureZoneManager._anchor(bbox)
        return cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0

    def process_frame(
        self,
        camera_id: int,
        frame_id: int,
        frame: np.ndarray,
        tracks: Iterable[tuple[int, Iterable[float]]],
        entry_gate: str | None = None,
    ) -> dict[int, bool]:
        """Sample eligible tracks in one batch; return {track_id: accepted}."""
        polygon = self._profile_polygon(camera_id, entry_gate)
        if polygon is None or frame_id % self.sample_every_n_frames != 0:
            return {}

        h, w = frame.shape[:2]
        pending: list[tuple[int, np.ndarray]] = []
        for track_id, bbox in tracks:
            key = (int(camera_id), int(track_id))
            gallery = self._galleries.get(key)
            if gallery is not None and gallery.last_sample_frame == frame_id:
                continue
            if not self._inside_polygon(bbox, polygon):
                continue
            x1, y1, x2, y2 = map(int, bbox)
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            if x2 - x1 < self.min_crop_size or y2 - y1 < self.min_crop_size:
                continue
            crop = frame[y1:y2, x1:x2].copy()
            if crop.size:
                pending.append((int(track_id), crop))
        if not pending:
            return {}

        # Originals and horizontal flips share one extractor batch per frame.
        batch: list[np.ndarray] = []
        for _, crop in pending:
            batch.extend((crop, cv2.flip(crop, 1)))
        features = self.features_manager.extract_features(batch)
        if len(features) != len(batch):
            raise RuntimeError(f"Feature extractor returned {len(features)} vectors for {len(batch)} crops")

        decisions: dict[int, bool] = {}
        for index, (track_id, _) in enumerate(pending):
            original = self._normalize(features[2 * index])
            flipped = self._normalize(features[2 * index + 1])
            key = (int(camera_id), track_id)
            gallery = self._galleries.setdefault(
                key, TrackGallery(camera_id=int(camera_id), track_id=track_id)
            )
            gallery.last_sample_frame = int(frame_id)

            # Compare only original-to-last-accepted-original; flip never affects filtering.
            if gallery.original:
                similarity = self.features_manager.calculate_similarity(
                    original, gallery.original[-1]
                )
                if similarity >= self.similarity_threshold:
                    decisions[track_id] = False
                    continue
            gallery.original.append(original)
            gallery.flipped.append(flipped)
            decisions[track_id] = True
        return decisions

    @staticmethod
    def _normalize(feature) -> np.ndarray:
        vector = np.asarray(feature, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError("Feature extractor returned a zero or non-finite embedding")
        return vector / norm

    def end_track(self, camera_id: int, track_id: int) -> dict[str, Any] | None:
        """Close a local observation and return its gallery, or None if empty."""
        gallery = self._galleries.pop((int(camera_id), int(track_id)), None)
        return gallery.as_dict() if gallery is not None else None

    def end_tracks_on_exit(self, camera_id: int, transition_events: Iterable[Any]) -> dict[int, dict[str, Any]]:
        """Close galleries for EXIT events emitted by this camera's CrossingDetector."""
        completed: dict[int, dict[str, Any]] = {}
        for event in transition_events:
            event_camera = getattr(event, "camera_id", None)
            role = getattr(getattr(event, "role", None), "value", getattr(event, "role", None))
            track_id = getattr(event, "track_id", None)
            if event_camera != int(camera_id) or str(role).upper() != "EXIT" or track_id is None:
                continue
            gallery = self.end_track(camera_id, int(track_id))
            if gallery is not None:
                completed[int(track_id)] = gallery
        return completed

    def active_track_keys(self) -> tuple[tuple[int, int], ...]:
        return tuple(self._galleries.keys())

    def close_all(self) -> dict[tuple[int, int], dict[str, Any]]:
        """Close all active observations, e.g. at end-of-stream."""
        completed = {key: gallery.as_dict() for key, gallery in self._galleries.items()}
        self._galleries.clear()
        return completed

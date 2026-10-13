"""Zone-based Re-ID feature sampling with camera-specific anchor points."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping
import cv2
import numpy as np
import yaml

@dataclass
class TrackGallery:
    """Embeddings for one local track observation."""
    camera_id: int
    track_id: int
    zone_profile: str
    original: list[np.ndarray] = field(default_factory=list)
    flipped: list[np.ndarray] = field(default_factory=list)
    last_sample_frame: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"camera_id": self.camera_id, "track_id": self.track_id,
                "zone_profile": self.zone_profile, "original": list(self.original),
                "flipped": list(self.flipped), "sample_count": len(self.original)}

class FeatureZoneManager:
    """Capture only on cameras 1-3: cam1 bottom-center, cam2 right-center, cam3 top-center.
    Cameras 4-9 are intentionally ignored by this module.
    """
    def __init__(self, features_manager, config_path: str | Path = "feature_zones.yaml",
                 similarity_threshold: float | None = None,
                 sample_every_n_frames: int | None = None, min_crop_size: int | None = None):
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
        self.similarity_threshold, self.sample_every_n_frames, self.min_crop_size = float(threshold), int(interval), int(crop_size)
        self._galleries: dict[tuple[int, int], TrackGallery] = {}
        self._polygons = self._parse_polygons(self.config.get("cameras", {}))

    @staticmethod
    def _parse_polygons(cameras: Mapping) -> dict[int, dict[str, np.ndarray]]:
        parsed = {}
        for camera_key, camera_cfg in cameras.items():
            try:
                camera_id = int(camera_key)
            except (TypeError, ValueError):
                continue
            profiles = {}
            for profile_name, points in (camera_cfg or {}).items():
                if not points:
                    continue
                polygon = np.asarray(points, dtype=np.int32)
                if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
                    raise ValueError(f"Camera {camera_id} profile {profile_name!r} needs >= 3 [x, y] points")
                profiles[str(profile_name).upper()] = polygon
            parsed[camera_id] = profiles
        return parsed

    @staticmethod
    def anchor(camera_id: int, bbox) -> tuple[float, float]:
        camera_id = int(camera_id)
        if camera_id not in (1, 2, 3):
            raise ValueError("Feature capture anchors are defined only for cameras 1, 2, and 3")
        x1, y1, x2, y2 = map(float, bbox)
        if camera_id == 2:
            return x2, (y1 + y2) / 2.0
        if int(camera_id) == 3:
            return (x1 + x2) / 2.0, y1
        return (x1 + x2) / 2.0, y2

    @staticmethod
    def _inside_polygon(point: tuple[float, float], polygon: np.ndarray) -> bool:
        return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0

    def _matching_profile(self, camera_id: int, bbox) -> str | None:
        profiles = self._polygons.get(int(camera_id), {})
        candidates = ("LEFT", "RIGHT") if int(camera_id) in (1, 2) else ("DEFAULT",)
        point = self.anchor(camera_id, bbox)
        # Deterministic priority if polygons overlap; ideally the two zones should not overlap.
        for name in candidates:
            if name in profiles and self._inside_polygon(point, profiles[name]):
                return name
        return None

    def process_frame(self, camera_id: int, frame_id: int, frame: np.ndarray,
                      tracks: Iterable[tuple[int, Iterable[float]]],
                      entry_gate: str | None = None) -> dict[int, bool]:
        """Activate tracks when their anchor enters a zone, then sample them outside it too.

        Zone membership is an activation trigger, not a spatial restriction on every
        feature sample. All currently visible activated tracks on this camera are
        considered on sampling frames, so multiple vehicles can be processed in one
        batch. A new local track ID entering a zone gets its own gallery.
        """
        camera_id = int(camera_id)
        if camera_id not in (1, 2, 3):
            return {}

        # Trigger evaluation happens on every call, even when this frame is not a
        # feature-sampling frame, so frame skipping cannot silently miss activation.
        visible: dict[int, tuple[float, float, float, float]] = {}
        for track_id, bbox in tracks:
            track_id = int(track_id)
            box = tuple(map(float, bbox))
            visible[track_id] = box
            key = (camera_id, track_id)
            if key not in self._galleries:
                profile = self._matching_profile(camera_id, box)
                if profile is not None:
                    self._galleries[key] = TrackGallery(camera_id, track_id, profile)

        if frame_id % self.sample_every_n_frames != 0:
            return {}

        h, w = frame.shape[:2]
        pending: list[tuple[int, str, np.ndarray]] = []
        for (gallery_camera, track_id), gallery in list(self._galleries.items()):
            if gallery_camera != camera_id or track_id not in visible:
                continue
            if gallery.last_sample_frame == frame_id:
                continue
            x1, y1, x2, y2 = map(int, visible[track_id])
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            if x2 - x1 < self.min_crop_size or y2 - y1 < self.min_crop_size:
                continue
            crop = frame[y1:y2, x1:x2].copy()
            if crop.size:
                pending.append((track_id, gallery.zone_profile, crop))

        if not pending:
            return {}

        # Originals and horizontal flips are passed in a single extractor call.
        batch = []
        for _, _, crop in pending:
            batch.extend((crop, cv2.flip(crop, 1)))
        features = self.features_manager.extract_features(batch)
        if len(features) != len(batch):
            raise RuntimeError(f"Feature extractor returned {len(features)} vectors for {len(batch)} crops")

        decisions = {}
        for index, (track_id, profile, _) in enumerate(pending):
            original = self._normalize(features[2 * index])
            flipped = self._normalize(features[2 * index + 1])
            gallery = self._galleries[(camera_id, track_id)]
            gallery.last_sample_frame = int(frame_id)
            if gallery.original:
                similarity = self.features_manager.calculate_similarity(original, gallery.original[-1])
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
        gallery = self._galleries.pop((int(camera_id), int(track_id)), None)
        return gallery.as_dict() if gallery is not None else None

    def end_tracks_on_exit(self, camera_id: int, transition_events: Iterable[Any]) -> dict[int, dict[str, Any]]:
        completed = {}
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
        completed = {key: gallery.as_dict() for key, gallery in self._galleries.items()}
        self._galleries.clear()
        return completed

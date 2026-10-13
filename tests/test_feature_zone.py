import tempfile
import unittest
from pathlib import Path

import numpy as np

from FeatureZoneModule import FeatureZoneManager


class FakeFeaturesManager:
    def __init__(self, vectors):
        self.vectors = list(vectors)
        self.batches = []

    def extract_features(self, image_list):
        self.batches.append(list(image_list))
        result, self.vectors = self.vectors[:len(image_list)], self.vectors[len(image_list):]
        return result

    @staticmethod
    def calculate_similarity(feature_vector_1, feature_vector_2):
        a, b = feature_vector_1, feature_vector_2
        return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


class FeatureZoneManagerTests(unittest.TestCase):
    def make_manager(self, feature_manager, polygon=None, interval=1, threshold=0.90):
        import yaml
        points = [[0, 0], [99, 0], [99, 99], [0, 99]] if polygon is None else polygon
        config = {
            "sample_every_n_frames": interval,
            "similarity_threshold": threshold,
            "min_crop_size": 8,
            "cameras": {"1": {"LEFT": points}},
        }
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8", delete=False)
        try:
            yaml.safe_dump(config, tmp)
            tmp.close()
            self.addCleanup(lambda: Path(tmp.name).unlink(missing_ok=True))
            return FeatureZoneManager(feature_manager, tmp.name, threshold, interval, 8)
        except Exception:
            tmp.close()
            raise

    def test_batches_original_and_flip_together_and_accepts_first_pair(self):
        fm = FakeFeaturesManager([np.array([1., 0.]), np.array([0., 1.])])
        manager = self.make_manager(fm)
        frame = np.ones((100, 100, 3), dtype=np.uint8)
        result = manager.process_frame(1, 0, frame, [(7, [10, 10, 50, 50])])
        self.assertEqual(result, {7: True})
        self.assertEqual(len(fm.batches), 1)
        self.assertEqual(len(fm.batches[0]), 2)
        gallery = manager.end_track(1, 7)
        self.assertEqual(gallery["sample_count"], 1)
        self.assertEqual(len(gallery["original"]), 1)
        self.assertEqual(len(gallery["flipped"]), 1)

    def test_filter_uses_originals_only_and_rejects_similar_original(self):
        fm = FakeFeaturesManager([
            np.array([1., 0.]), np.array([0., 1.]),
            np.array([1., 0.]), np.array([0., -1.]),
        ])
        manager = self.make_manager(fm)
        frame = np.ones((100, 100, 3), dtype=np.uint8)
        manager.process_frame(1, 0, frame, [(7, [10, 10, 50, 50])])
        result = manager.process_frame(1, 1, frame, [(7, [10, 10, 50, 50])])
        self.assertEqual(result, {7: False})
        gallery = manager.end_track(1, 7)
        self.assertEqual(gallery["sample_count"], 1)

    def test_outside_zone_is_ignored(self):
        fm = FakeFeaturesManager([])
        manager = self.make_manager(fm, polygon=[[0, 0], [20, 0], [20, 20], [0, 20]])
        frame = np.ones((100, 100, 3), dtype=np.uint8)
        result = manager.process_frame(1, 0, frame, [(7, [40, 40, 70, 70])])
        self.assertEqual(result, {})
        self.assertEqual(fm.batches, [])

    def test_sample_interval(self):
        fm = FakeFeaturesManager([])
        manager = self.make_manager(fm, interval=3)
        frame = np.ones((100, 100, 3), dtype=np.uint8)
        self.assertEqual(manager.process_frame(1, 1, frame, [(7, [10, 10, 50, 50])]), {})
        self.assertEqual(fm.batches, [])

    def test_empty_polygon_disables_profile(self):
        fm = FakeFeaturesManager([])
        manager = self.make_manager(fm, polygon=[])
        frame = np.ones((100, 100, 3), dtype=np.uint8)
        self.assertEqual(manager.process_frame(1, 0, frame, [(7, [10, 10, 50, 50])]), {})


    def test_camera_2_selects_profile_from_zone_hit(self):
        import yaml
        config = {
            "cameras": {
                "2": {
                    "LEFT": [[0, 0], [99, 0], [99, 99], [0, 99]],
                    "RIGHT": [[100, 100], [199, 100], [199, 199], [100, 199]],
                }
            }
        }
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8", delete=False)
        yaml.safe_dump(config, tmp)
        tmp.close()
        self.addCleanup(lambda: Path(tmp.name).unlink(missing_ok=True))
        fm = FakeFeaturesManager([np.array([1., 0.]), np.array([0., 1.])])
        manager = FeatureZoneManager(fm, tmp.name, sample_every_n_frames=1)
        frame = np.ones((220, 220, 3), dtype=np.uint8)
        # The right-edge center lands in LEFT, so the zone identifies the side.
        self.assertEqual(manager.process_frame(2, 0, frame, [(7, [10, 10, 50, 50])]), {7: True})
        self.assertEqual(len(fm.batches), 1)
        gallery = manager.end_track(2, 7)
        self.assertEqual(gallery["zone_profile"], "LEFT")

    def test_exit_event_closes_only_matching_camera_track(self):
        from types import SimpleNamespace
        from enum import Enum

        class Role(Enum):
            EXIT = "EXIT"
            ENTER = "ENTER"

        fm = FakeFeaturesManager([np.array([1., 0.]), np.array([0., 1.])])
        manager = self.make_manager(fm)
        frame = np.ones((100, 100, 3), dtype=np.uint8)
        manager.process_frame(1, 0, frame, [(7, [10, 10, 50, 50])])
        events = [
            SimpleNamespace(camera_id=2, role=Role.EXIT, track_id=7),
            SimpleNamespace(camera_id=1, role=Role.ENTER, track_id=7),
            SimpleNamespace(camera_id=1, role=Role.EXIT, track_id=7),
        ]
        completed = manager.end_tracks_on_exit(1, events)
        self.assertEqual(list(completed), [7])
        self.assertEqual(completed[7]["sample_count"], 1)
        self.assertEqual(manager.active_track_keys(), ())

    def test_end_track_is_scoped_to_camera_and_track(self):
        fm = FakeFeaturesManager([np.array([1., 0.]), np.array([0., 1.])])
        manager = self.make_manager(fm)
        frame = np.ones((100, 100, 3), dtype=np.uint8)
        manager.process_frame(1, 0, frame, [(7, [10, 10, 50, 50])])
        self.assertIsNone(manager.end_track(2, 7))
        self.assertIsNotNone(manager.end_track(1, 7))
        self.assertEqual(manager.active_track_keys(), ())

    def test_camera_specific_anchor_points(self):
        self.assertEqual(FeatureZoneManager.anchor(1, [10, 20, 50, 80]), (30.0, 80.0))
        self.assertEqual(FeatureZoneManager.anchor(2, [10, 20, 50, 80]), (50.0, 50.0))
        self.assertEqual(FeatureZoneManager.anchor(3, [10, 20, 50, 80]), (30.0, 20.0))
        with self.assertRaises(ValueError):
            FeatureZoneManager.anchor(8, [10, 20, 50, 80])


    def test_cameras_4_to_9_never_extract_features(self):
        fm = FakeFeaturesManager([])
        manager = self.make_manager(fm)
        frame = np.ones((100, 100, 3), dtype=np.uint8)
        for camera_id in range(4, 10):
            self.assertEqual(manager.process_frame(camera_id, 0, frame, [(7, [10, 10, 50, 50])]), {})
        self.assertEqual(fm.batches, [])

    def test_track_id_switch_inside_zone_starts_new_gallery_and_is_sampled(self):
        fm = FakeFeaturesManager([
            np.array([1., 0.]), np.array([0., 1.]),
            np.array([0., 1.]), np.array([1., 0.]),
        ])
        manager = self.make_manager(fm)
        frame = np.ones((100, 100, 3), dtype=np.uint8)
        self.assertEqual(manager.process_frame(1, 0, frame, [(7, [10, 10, 50, 50])]), {7: True})
        self.assertEqual(manager.process_frame(1, 1, frame, [(19, [10, 10, 50, 50])]), {19: True})
        self.assertIsNotNone(manager.end_track(1, 7))
        self.assertIsNotNone(manager.end_track(1, 19))


if __name__ == "__main__":
    unittest.main()

import unittest

from FeatureZoneModule.EntrySessionManager import EntrySessionManager


class EntrySessionManagerTests(unittest.TestCase):
    def setUp(self):
        self.manager = EntrySessionManager()

    def test_two_triggers_create_ordered_sessions_without_blocking(self):
        first = self.manager.register_entry(1, 11, 100)
        second = self.manager.register_entry(1, 22, 104)
        self.assertEqual((first.entry_order, second.entry_order), (1, 2))
        self.assertEqual(len(self.manager.active_sessions()), 2)

    def test_repeated_trigger_for_same_track_is_idempotent(self):
        first = self.manager.register_entry(1, 11, 100)
        again = self.manager.register_entry(1, 11, 101)
        self.assertEqual(first.session_id, again.session_id)
        self.assertEqual(len(self.manager.active_sessions()), 1)

    def test_interloper_does_not_consume_fifo_slot(self):
        first = self.manager.register_entry(1, 11, 100)
        self.manager.register_entry(1, 22, 104)
        self.assertIsNone(
            self.manager.bind_next_in_order(2, 99, confirmed_registered_entry=False)
        )
        bound = self.manager.bind_next_in_order(
            2, 31, confirmed_registered_entry=True
        )
        self.assertEqual(bound.session_id, first.session_id)
        self.assertEqual(bound.camera_tracks[2], 31)

    def test_camera2_track_loss_finishes_only_its_session(self):
        first = self.manager.register_entry(1, 11, 100)
        second = self.manager.register_entry(1, 22, 104)
        self.manager.bind_track(first.session_id, 2, 31)
        self.manager.bind_track(second.session_id, 2, 32)
        completed = self.manager.finish_on_camera2_track_lost(31)
        self.assertEqual(completed["session_id"], first.session_id)
        self.assertEqual(completed["finish_reason"], "camera_2_track_left_frame")
        self.assertEqual([s["session_id"] for s in self.manager.active_sessions()],
                         [second.session_id])
        self.assertIsNone(self.manager.finish_on_camera2_track_lost(999))

    def test_track_cannot_be_assigned_to_two_sessions(self):
        first = self.manager.register_entry(1, 11, 100)
        second = self.manager.register_entry(1, 22, 104)
        self.manager.bind_track(first.session_id, 2, 31)
        with self.assertRaises(ValueError):
            self.manager.bind_track(second.session_id, 2, 31)

    def test_cameras_4_to_9_are_rejected(self):
        session = self.manager.register_entry(1, 11, 100)
        for camera_id in range(4, 10):
            with self.assertRaises(ValueError):
                self.manager.bind_track(session.session_id, camera_id, 100 + camera_id)


if __name__ == "__main__":
    unittest.main()

import unittest

from utility.tracker import Tracker


class TrackerSignalContextTests(unittest.TestCase):
    def test_tracker_stores_latest_relevant_signal_context(self):
        tracker = Tracker()

        tracker.update(
            obstacle_snapshots=[],
            timestamp_s=1.0,
            next_signal_context={
                "signal_found": True,
                "signal_state": "red",
                "signal_distance_m": 18.0,
                "signal_actor_id": 12,
                "signal_actor_name": "signal_12",
                "signal_source": "stop_waypoint_match",
            },
            next_stop_target={
                "x_m": 0.0,
                "y_m": 12.0,
                "lane_id": 1,
                "distance_m": 12.0,
            },
        )

        signal_context = tracker.get_next_signal_context()

        self.assertTrue(bool(signal_context["signal_found"]))
        self.assertEqual(str(signal_context["phase"]), "red")
        self.assertEqual(int(signal_context["signal_actor_id"]), 12)
        self.assertAlmostEqual(float(signal_context["signal_distance_m"]), 12.0, places=3)
        self.assertEqual(int(signal_context["stop_target"]["lane_id"]), 1)

    def test_tracker_clears_latest_signal_context_when_none_is_provided(self):
        tracker = Tracker()

        tracker.update(
            obstacle_snapshots=[],
            timestamp_s=1.0,
            next_signal_context={
                "signal_found": True,
                "signal_state": "green",
                "signal_distance_m": 15.0,
                "signal_actor_id": 7,
                "signal_actor_name": "signal_7",
                "signal_source": "stop_waypoint_match",
            },
            next_stop_target={
                "distance_m": 15.0,
                "lane_id": 1,
            },
        )
        tracker.update(
            obstacle_snapshots=[],
            timestamp_s=2.0,
            next_signal_context=None,
            next_stop_target=None,
        )

        signal_context = tracker.get_next_signal_context()

        self.assertFalse(bool(signal_context["signal_found"]))
        self.assertEqual(str(signal_context["signal_state"]), "unknown")
        self.assertIsNone(signal_context["stop_target"])


if __name__ == "__main__":
    unittest.main()

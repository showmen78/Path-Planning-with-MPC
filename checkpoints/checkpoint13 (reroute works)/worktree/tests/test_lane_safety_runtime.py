import unittest

from behavior_planner.lane_safety import LaneSafetyScorer
from carla_scenario.runner import _same_lane_safety_corridor


class LaneSafetyRuntimeTests(unittest.TestCase):
    def test_empty_lane_scores_default_to_safe(self):
        scorer = LaneSafetyScorer()

        scores = scorer.compute_lane_scores(
            ego_snapshot={"x": 0.0, "y": 0.0, "v": 5.0, "psi": 0.0},
            obstacle_snapshots=[],
            lane_assignments={},
            ego_lane_id=1,
            available_lane_ids=[1, 2],
            timestamp_s=0.0,
        )

        self.assertEqual(scores, {1: 1.0, 2: 1.0})

    def test_lane_safety_corridor_filter_rejects_different_road(self):
        ego_context = {
            "road_id": "12:0",
            "road_numeric_id": 12,
            "direction": "positive",
            "lane_id": 1,
        }
        obstacle_context = {
            "road_id": "34:0",
            "road_numeric_id": 34,
            "direction": "positive",
            "lane_id": 1,
        }

        self.assertFalse(_same_lane_safety_corridor(ego_context, obstacle_context))

    def test_lane_safety_corridor_filter_accepts_same_road_and_direction(self):
        ego_context = {
            "road_id": "12:0",
            "road_numeric_id": 12,
            "direction": "positive",
            "lane_id": 1,
        }
        obstacle_context = {
            "road_id": "12:1",
            "road_numeric_id": 12,
            "direction": "positive",
            "lane_id": 2,
        }

        self.assertTrue(_same_lane_safety_corridor(ego_context, obstacle_context))


if __name__ == "__main__":
    unittest.main()

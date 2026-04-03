import unittest

from carla_scenario.runner import _allowed_lane_ids_from_context


class RunnerLaneModelTests(unittest.TestCase):
    def test_allowed_lane_ids_prefer_same_direction_context_over_fallback_count(self):
        lane_ids = _allowed_lane_ids_from_context(
            local_context={"lane_ids": [1, 2]},
            fallback_lane_count=4,
        )

        self.assertEqual(lane_ids, [1, 2])


if __name__ == "__main__":
    unittest.main()

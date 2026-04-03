import unittest

from carla_scenario.runner import (
    _filter_obstacle_snapshots_by_vertical_overlap,
    _nearest_front_obstacle_by_lane,
    _should_force_intersection_reroute,
    _static_obstacle_replan_candidate_lane_ids,
)


class RunnerObstacleFilterTests(unittest.TestCase):
    def test_filters_out_obstacles_on_different_vertical_level(self):
        filtered = _filter_obstacle_snapshots_by_vertical_overlap(
            ego_z_m=0.5,
            ego_height_m=1.8,
            object_snapshots=[
                {
                    "vehicle_id": "same_level",
                    "x": 1.0,
                    "y": 2.0,
                    "z": 0.8,
                    "height_m": 1.8,
                },
                {
                    "vehicle_id": "bridge_above",
                    "x": 1.0,
                    "y": 2.0,
                    "z": 7.5,
                    "height_m": 1.8,
                },
            ],
            vertical_clearance_margin_m=1.0,
            default_obstacle_height_m=2.0,
        )

        self.assertEqual([snapshot["vehicle_id"] for snapshot in filtered], ["same_level"])

    def test_keeps_snapshots_without_height_or_z_metadata(self):
        filtered = _filter_obstacle_snapshots_by_vertical_overlap(
            ego_z_m=0.5,
            ego_height_m=1.8,
            object_snapshots=[
                {
                    "vehicle_id": "unknown_height",
                    "x": 1.0,
                    "y": 2.0,
                },
            ],
            vertical_clearance_margin_m=1.0,
            default_obstacle_height_m=2.0,
        )

        self.assertEqual([snapshot["vehicle_id"] for snapshot in filtered], ["unknown_height"])

    def test_nearest_front_obstacle_by_lane_ignores_rear_and_keeps_nearest_front(self):
        nearest = _nearest_front_obstacle_by_lane(
            ego_snapshot={"x": 0.0, "y": 0.0, "psi": 0.0},
            obstacle_snapshots=[
                {"vehicle_id": "rear", "x": -3.0, "y": 0.0, "v": 1.0},
                {"vehicle_id": "front_far", "x": 10.0, "y": 0.0, "v": 2.0},
                {"vehicle_id": "front_near", "x": 4.0, "y": 0.0, "v": 3.0},
            ],
            lane_assignments={
                "rear": 1,
                "front_far": 1,
                "front_near": 1,
            },
            available_lane_ids=[1],
        )

        self.assertEqual(str(nearest[1]["vehicle_id"]), "front_near")
        self.assertAlmostEqual(float(nearest[1]["front_distance_m"]), 4.0)

    def test_static_obstacle_replan_candidate_lane_ids_prioritize_safer_alternative_lane(self):
        candidate_lane_ids = _static_obstacle_replan_candidate_lane_ids(
            current_lane_id=2,
            available_lane_ids=[1, 2, 3],
            lane_safety_scores={1: 0.9, 2: 0.1, 3: 0.6},
        )

        self.assertEqual(candidate_lane_ids, [1, 3, 2])

    def test_force_intersection_reroute_requires_low_ego_lane_safety_and_front_obstacle(self):
        should_reroute = _should_force_intersection_reroute(
            mode="INTERSECTION",
            ego_lane_id=1,
            lane_safety_scores={1: 0.0, 2: 1.0},
            nearest_front_obstacles_by_lane={1: {"vehicle_id": "front_blocker"}},
            safety_threshold=0.5,
        )

        self.assertTrue(bool(should_reroute))

    def test_force_intersection_reroute_stays_false_without_front_obstacle_or_in_normal_mode(self):
        self.assertFalse(
            bool(
                _should_force_intersection_reroute(
                    mode="INTERSECTION",
                    ego_lane_id=1,
                    lane_safety_scores={1: 0.0},
                    nearest_front_obstacles_by_lane={},
                    safety_threshold=0.5,
                )
            )
        )
        self.assertFalse(
            bool(
                _should_force_intersection_reroute(
                    mode="NORMAL",
                    ego_lane_id=1,
                    lane_safety_scores={1: 0.0},
                    nearest_front_obstacles_by_lane={1: {"vehicle_id": "front_blocker"}},
                    safety_threshold=0.5,
                )
            )
        )


if __name__ == "__main__":
    unittest.main()

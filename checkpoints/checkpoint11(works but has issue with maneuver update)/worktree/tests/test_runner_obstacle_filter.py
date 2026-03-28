import unittest

from carla_scenario.runner import _filter_obstacle_snapshots_by_vertical_overlap


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


if __name__ == "__main__":
    unittest.main()

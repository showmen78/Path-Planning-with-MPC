import unittest

from carla_scenario import list_available_scenarios, load_carla_scenario
from carla_scenario.roadway_hazard.scenario import (
    filter_dynamic_obstacle_snapshots,
    initialize_runtime,
)


class RoadwayHazardScenarioTests(unittest.TestCase):
    def test_scenario_is_available_and_uses_expected_runtime_module(self):
        self.assertIn("roadway_hazard", list_available_scenarios())

        scenario_cfg = load_carla_scenario("roadway_hazard")

        self.assertEqual(
            str(scenario_cfg.get("obstacles", {}).get("spawner_module", "")),
            "carla_scenario.roadway_hazard.scenario",
        )
        self.assertEqual(
            str(scenario_cfg.get("runtime", {}).get("module", "")),
            "carla_scenario.roadway_hazard.scenario",
        )

    def test_hidden_obstacle_stays_filtered_before_cooperative_trigger(self):
        runtime_state = initialize_runtime(
            scenario_cfg={
                "runtime": {
                    "hidden_obstacle_id": "obstacle4",
                    "relay_obstacle_id": "obstacle6",
                    "reveal_distance_m": 20.0,
                }
            }
        )

        filtered_snapshots, next_runtime_state = filter_dynamic_obstacle_snapshots(
            runtime_state=runtime_state,
            object_snapshots=[
                {"vehicle_id": "obstacle4", "x": 0.0, "y": 0.0},
                {"vehicle_id": "obstacle6", "x": 25.0, "y": 0.0},
                {"vehicle_id": "obstacle2", "x": 10.0, "y": 0.0},
            ],
        )

        filtered_ids = [str(snapshot.get("vehicle_id", "")) for snapshot in filtered_snapshots]
        self.assertNotIn("obstacle4", filtered_ids)
        self.assertIn("obstacle6", filtered_ids)
        self.assertFalse(bool(next_runtime_state.get("hidden_obstacle_revealed", False)))

    def test_hidden_obstacle_is_revealed_when_obstacle6_is_within_20m(self):
        runtime_state = initialize_runtime(
            scenario_cfg={
                "runtime": {
                    "hidden_obstacle_id": "obstacle4",
                    "relay_obstacle_id": "obstacle6",
                    "reveal_distance_m": 20.0,
                }
            }
        )

        filtered_snapshots, next_runtime_state = filter_dynamic_obstacle_snapshots(
            runtime_state=runtime_state,
            object_snapshots=[
                {"vehicle_id": "obstacle4", "x": 0.0, "y": 0.0},
                {"vehicle_id": "obstacle6", "x": 19.5, "y": 0.0},
            ],
        )

        filtered_ids = [str(snapshot.get("vehicle_id", "")) for snapshot in filtered_snapshots]
        self.assertIn("obstacle4", filtered_ids)
        self.assertTrue(bool(next_runtime_state.get("hidden_obstacle_revealed", False)))

    def test_hidden_obstacle_stays_visible_after_trigger_latches(self):
        filtered_snapshots, next_runtime_state = filter_dynamic_obstacle_snapshots(
            runtime_state={
                "hidden_obstacle_id": "obstacle4",
                "relay_obstacle_id": "obstacle6",
                "reveal_distance_m": 20.0,
                "hidden_obstacle_revealed": True,
            },
            object_snapshots=[
                {"vehicle_id": "obstacle4", "x": 0.0, "y": 0.0},
                {"vehicle_id": "obstacle6", "x": 40.0, "y": 0.0},
            ],
        )

        filtered_ids = [str(snapshot.get("vehicle_id", "")) for snapshot in filtered_snapshots]
        self.assertIn("obstacle4", filtered_ids)
        self.assertTrue(bool(next_runtime_state.get("hidden_obstacle_revealed", False)))


if __name__ == "__main__":
    unittest.main()

import unittest
from types import SimpleNamespace

from carla_scenario import list_available_scenarios, load_carla_scenario
from carla_scenario.high_level_route_planning.scenario import (
    initialize_runtime,
    maybe_replan_global_route,
)


class _FakePlanner:
    def __init__(self):
        self.calls = []

    def plan_route_astar_avoiding_points(
        self,
        start_xy,
        goal_xy,
        *,
        blocked_points_xy,
        block_radius_m=8.0,
        replace_stored_route=False,
        **_,
    ):
        self.calls.append(
            {
                "start_xy": list(start_xy),
                "goal_xy": list(goal_xy),
                "blocked_points_xy": [list(point) for point in list(blocked_points_xy or [])],
                "block_radius_m": float(block_radius_m),
                "replace_stored_route": bool(replace_stored_route),
            }
        )
        return SimpleNamespace(
            route_found=True,
            route_waypoints=[
                [float(start_xy[0]), float(start_xy[1])],
                [5.0, 5.0],
                [float(goal_xy[0]), float(goal_xy[1])],
            ],
        )


class HighLevelRoutePlanningScenarioTests(unittest.TestCase):
    def test_scenario_is_available_and_uses_expected_runtime_module(self):
        self.assertIn("high_level_route_planning", list_available_scenarios())

        scenario_cfg = load_carla_scenario("high_level_route_planning")

        self.assertEqual(
            str(scenario_cfg.get("obstacles", {}).get("spawner_module", "")),
            "carla_scenario.high_level_route_planning.scenario",
        )
        self.assertEqual(
            str(scenario_cfg.get("runtime", {}).get("module", "")),
            "carla_scenario.high_level_route_planning.scenario",
        )

    def test_route_is_not_replanned_before_delay(self):
        runtime_state = initialize_runtime(
            scenario_cfg={
                "runtime": {
                    "obstacle_aware_replan_delay_s": 5.0,
                    "blocked_obstacle_ids": ["obstacle1"],
                    "block_radius_m": 9.0,
                }
            }
        )
        planner = _FakePlanner()

        route_summary, route_points, next_runtime_state = maybe_replan_global_route(
            runtime_state=runtime_state,
            global_planner=planner,
            ego_transform=SimpleNamespace(location=SimpleNamespace(x=1.0, y=2.0)),
            goal_location=SimpleNamespace(x=10.0, y=20.0),
            object_snapshots=[{"vehicle_id": "obstacle1", "x": 4.0, "y": 6.0}],
            sim_time_s=4.9,
        )

        self.assertIsNone(route_summary)
        self.assertIsNone(route_points)
        self.assertFalse(bool(next_runtime_state.get("route_replan_applied", False)))
        self.assertEqual(planner.calls, [])

    def test_route_is_replanned_once_after_delay_using_obstacle_positions(self):
        runtime_state = initialize_runtime(
            scenario_cfg={
                "runtime": {
                    "obstacle_aware_replan_delay_s": 5.0,
                    "blocked_obstacle_ids": ["obstacle1"],
                    "block_radius_m": 9.0,
                }
            }
        )
        planner = _FakePlanner()

        route_summary, route_points, next_runtime_state = maybe_replan_global_route(
            runtime_state=runtime_state,
            global_planner=planner,
            ego_transform=SimpleNamespace(location=SimpleNamespace(x=1.0, y=2.0)),
            goal_location=SimpleNamespace(x=10.0, y=20.0),
            object_snapshots=[
                {"vehicle_id": "obstacle1", "x": 4.0, "y": 6.0},
                {"vehicle_id": "obstacle2", "x": 100.0, "y": 200.0},
            ],
            sim_time_s=5.0,
        )

        self.assertTrue(bool(getattr(route_summary, "route_found", False)))
        self.assertEqual(route_points, [[1.0, 2.0], [5.0, 5.0], [10.0, 20.0]])
        self.assertTrue(bool(next_runtime_state.get("route_replan_applied", False)))
        self.assertEqual(len(planner.calls), 1)
        self.assertEqual(planner.calls[0]["blocked_points_xy"], [[4.0, 6.0]])
        self.assertAlmostEqual(float(planner.calls[0]["block_radius_m"]), 9.0)
        self.assertTrue(bool(planner.calls[0]["replace_stored_route"]))

        route_summary_2, route_points_2, next_runtime_state_2 = maybe_replan_global_route(
            runtime_state=next_runtime_state,
            global_planner=planner,
            ego_transform=SimpleNamespace(location=SimpleNamespace(x=1.0, y=2.0)),
            goal_location=SimpleNamespace(x=10.0, y=20.0),
            object_snapshots=[{"vehicle_id": "obstacle1", "x": 4.0, "y": 6.0}],
            sim_time_s=6.0,
        )

        self.assertIsNone(route_summary_2)
        self.assertIsNone(route_points_2)
        self.assertTrue(bool(next_runtime_state_2.get("route_replan_applied", False)))
        self.assertEqual(len(planner.calls), 1)


if __name__ == "__main__":
    unittest.main()

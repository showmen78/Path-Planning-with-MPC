import math
import types
import unittest

from behavior_planner.planner import RuleBasedBehaviorPlanner
from behavior_planner.traffic_light_stop import (
    find_relevant_signal_context,
    find_stop_target_from_ego,
    normalize_signal_state,
    should_stop_for_signal,
)


class _DummyWaypoint:
    def __init__(
        self,
        *,
        x_m: float,
        y_m: float,
        road_id: int,
        section_id: int,
        lane_id: int,
        is_junction: bool,
        yaw_deg: float = 90.0,
    ):
        self.road_id = int(road_id)
        self.section_id = int(section_id)
        self.lane_id = int(lane_id)
        self.is_junction = bool(is_junction)
        self.transform = types.SimpleNamespace(
            location=types.SimpleNamespace(x=float(x_m), y=float(y_m), z=0.0),
            rotation=types.SimpleNamespace(yaw=float(yaw_deg)),
        )


class _NearestWaypointMap:
    def __init__(self, *waypoints):
        self._waypoints = list(waypoints)

    def get_waypoint(self, location, project_to_road=True, lane_type=None):
        del project_to_road
        del lane_type
        if len(self._waypoints) == 0:
            return None
        return min(
            self._waypoints,
            key=lambda waypoint: (
                (float(waypoint.transform.location.x) - float(location.x)) ** 2
                + (float(waypoint.transform.location.y) - float(location.y)) ** 2
            ),
        )


class _DummyCarla:
    class Location:
        def __init__(self, x, y, z):
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)

    class LaneType:
        Driving = "Driving"


class _FakeTrafficLightActor:
    def __init__(
        self,
        *,
        actor_id: int,
        name: str,
        state: str,
        x_m: float,
        y_m: float,
        stop_waypoints=None,
    ):
        self.id = int(actor_id)
        self.type_id = "traffic.traffic_light"
        self.attributes = {"name": str(name)}
        self._state = str(state)
        self._transform = types.SimpleNamespace(
            location=types.SimpleNamespace(x=float(x_m), y=float(y_m), z=0.0),
        )
        self._stop_waypoints = list(stop_waypoints or [])

    def get_transform(self):
        return self._transform

    def get_state(self):
        return str(self._state)

    def get_stop_waypoints(self):
        return list(self._stop_waypoints)


class _FakeWorld:
    def __init__(self, actors):
        self._actors = list(actors)

    def get_actors(self):
        return list(self._actors)


class _FakeEgoVehicle:
    def __init__(self, traffic_light_actor=None):
        self._traffic_light_actor = traffic_light_actor

    def get_traffic_light(self):
        return self._traffic_light_actor


class TrafficLightStopTests(unittest.TestCase):
    def test_find_stop_target_returns_last_non_junction_waypoint_before_junction(self):
        wp0 = _DummyWaypoint(x_m=0.0, y_m=0.0, road_id=10, section_id=0, lane_id=-1, is_junction=False)
        wp1 = _DummyWaypoint(x_m=0.0, y_m=2.0, road_id=10, section_id=0, lane_id=-1, is_junction=False)
        wp2 = _DummyWaypoint(x_m=0.0, y_m=4.0, road_id=10, section_id=0, lane_id=-1, is_junction=False)
        wp3 = _DummyWaypoint(x_m=0.0, y_m=6.0, road_id=10, section_id=0, lane_id=-1, is_junction=False)
        wp4 = _DummyWaypoint(x_m=0.0, y_m=8.0, road_id=11, section_id=0, lane_id=-1, is_junction=True)
        world_map = _NearestWaypointMap(wp0, wp1, wp2, wp3, wp4)
        ego_transform = types.SimpleNamespace(location=wp0.transform.location)

        stop_target = find_stop_target_from_ego(
            world_map=world_map,
            carla=_DummyCarla,
            ego_transform=ego_transform,
            global_route_points=[[0.0, 0.0], [0.0, 2.0], [0.0, 4.0], [0.0, 6.0], [0.0, 8.0]],
            search_distance_m=20.0,
        )

        self.assertIsNotNone(stop_target)
        self.assertAlmostEqual(float(stop_target["x_m"]), 0.0, places=3)
        self.assertAlmostEqual(float(stop_target["y_m"]), 6.0, places=3)
        self.assertEqual(int(stop_target["road_id"]), 10)
        self.assertEqual(int(stop_target["section_id"]), 0)
        self.assertEqual(int(stop_target["lane_id"]), 1)
        self.assertAlmostEqual(float(stop_target["distance_m"]), 6.0, places=3)

    def test_find_stop_target_returns_none_when_no_junction_is_within_search_distance(self):
        wp0 = _DummyWaypoint(x_m=0.0, y_m=0.0, road_id=10, section_id=0, lane_id=-1, is_junction=False)
        wp1 = _DummyWaypoint(x_m=0.0, y_m=20.0, road_id=10, section_id=0, lane_id=-1, is_junction=False)
        wp2 = _DummyWaypoint(x_m=0.0, y_m=40.0, road_id=10, section_id=0, lane_id=-1, is_junction=True)
        world_map = _NearestWaypointMap(wp0, wp1, wp2)
        ego_transform = types.SimpleNamespace(location=wp0.transform.location)

        stop_target = find_stop_target_from_ego(
            world_map=world_map,
            carla=_DummyCarla,
            ego_transform=ego_transform,
            global_route_points=[[0.0, 0.0], [0.0, 20.0], [0.0, 40.0]],
            search_distance_m=10.0,
        )

        self.assertIsNone(stop_target)

    def test_find_stop_target_advances_to_next_junction_across_close_parallel_route_segments(self):
        wp0 = _DummyWaypoint(x_m=0.0, y_m=0.0, road_id=30, section_id=0, lane_id=-1, is_junction=False)
        wp1 = _DummyWaypoint(x_m=0.0, y_m=5.0, road_id=30, section_id=0, lane_id=-1, is_junction=False)
        wp2 = _DummyWaypoint(x_m=0.0, y_m=10.0, road_id=31, section_id=0, lane_id=-1, is_junction=True)
        wp3 = _DummyWaypoint(x_m=2.0, y_m=10.0, road_id=32, section_id=0, lane_id=-1, is_junction=False)
        wp4 = _DummyWaypoint(x_m=2.0, y_m=5.0, road_id=32, section_id=0, lane_id=-1, is_junction=False)
        wp5 = _DummyWaypoint(x_m=2.0, y_m=0.0, road_id=32, section_id=0, lane_id=-1, is_junction=False)
        wp6 = _DummyWaypoint(x_m=2.0, y_m=-5.0, road_id=33, section_id=0, lane_id=-1, is_junction=True)
        world_map = _NearestWaypointMap(wp0, wp1, wp2, wp3, wp4, wp5, wp6)
        route_points = [
            [0.0, 0.0],
            [0.0, 5.0],
            [0.0, 10.0],
            [2.0, 10.0],
            [2.0, 5.0],
            [2.0, 0.0],
            [2.0, -5.0],
        ]

        for ego_x_m, ego_y_m in (
            (0.0, 0.0),
            (0.0, 4.0),
            (1.8, 9.0),
        ):
            _ = find_stop_target_from_ego(
                world_map=world_map,
                carla=_DummyCarla,
                ego_transform=types.SimpleNamespace(
                    location=types.SimpleNamespace(x=float(ego_x_m), y=float(ego_y_m), z=0.0)
                ),
                global_route_points=route_points,
                search_distance_m=40.0,
                query_key="test_parallel_progression",
            )

        stop_target = find_stop_target_from_ego(
            world_map=world_map,
            carla=_DummyCarla,
            ego_transform=types.SimpleNamespace(
                location=types.SimpleNamespace(x=1.0, y=4.0, z=0.0)
            ),
            global_route_points=route_points,
            search_distance_m=40.0,
            query_key="test_parallel_progression",
        )

        self.assertIsNotNone(stop_target)
        self.assertAlmostEqual(float(stop_target["x_m"]), 2.0, places=3)
        self.assertAlmostEqual(float(stop_target["y_m"]), 0.0, places=3)
        self.assertEqual(int(stop_target["road_id"]), 32)
        self.assertAlmostEqual(float(stop_target["distance_m"]), 5.0, places=3)

    def test_should_stop_for_signal_stops_on_yellow_only_when_feasible(self):
        stop_target = {"distance_m": 15.0}
        self.assertTrue(
            should_stop_for_signal(
                signal_state="yellow",
                stop_target=stop_target,
                ego_velocity_mps=4.0,
                ego_max_deceleration_mps2=2.0,
                ego_in_junction=False,
                stop_buffer_m=2.0,
            )
        )
        self.assertFalse(
            should_stop_for_signal(
                signal_state="yellow",
                stop_target={"distance_m": 3.0},
                ego_velocity_mps=6.0,
                ego_max_deceleration_mps2=2.0,
                ego_in_junction=False,
                stop_buffer_m=2.0,
            )
        )

    def test_should_stop_for_signal_ignores_green_and_in_junction(self):
        self.assertEqual(normalize_signal_state("Green"), "green")
        self.assertFalse(
            should_stop_for_signal(
                signal_state="green",
                stop_target={"distance_m": 10.0},
                ego_velocity_mps=5.0,
                ego_max_deceleration_mps2=2.0,
                ego_in_junction=False,
            )
        )
        self.assertFalse(
            should_stop_for_signal(
                signal_state="red",
                stop_target={"distance_m": 20.0},
                ego_velocity_mps=5.0,
                ego_max_deceleration_mps2=2.0,
                ego_in_junction=True,
            )
        )

    def test_behavior_planner_emits_stop_and_holds_commitment_until_green(self):
        planner = RuleBasedBehaviorPlanner()
        stop_target = {
            "x_m": 0.0,
            "y_m": 12.0,
            "heading_rad": math.pi / 2.0,
            "lane_id": 1,
            "road_id": 10,
            "distance_m": 12.0,
        }

        first_result = planner.update(
            lane_safety_scores={1: 1.0},
            ego_lane_id=1,
            mode="NORMAL",
            traffic_signal_state="red",
            traffic_stop_target=stop_target,
            ego_speed_mps=4.0,
            ego_max_deceleration_mps2=2.0,
            ego_in_junction=False,
        )
        self.assertEqual(first_result["decision"], "stop")
        self.assertEqual(int(first_result["target_lane_id"]), 1)
        self.assertFalse(bool(first_result["blue_dot_rolling"]))
        self.assertEqual(str(first_result["mode_override"]), "INTERSECTION")
        self.assertAlmostEqual(float(first_result["stop_target"]["y_m"]), 12.0, places=3)

        second_result = planner.update(
            lane_safety_scores={1: 1.0},
            ego_lane_id=1,
            mode="NORMAL",
            traffic_signal_state="unknown",
            traffic_stop_target=None,
            ego_speed_mps=0.5,
            ego_max_deceleration_mps2=2.0,
            ego_in_junction=False,
        )
        self.assertEqual(second_result["decision"], "stop")

        third_result = planner.update(
            lane_safety_scores={1: 1.0},
            ego_lane_id=1,
            mode="NORMAL",
            traffic_signal_state="green",
            traffic_stop_target=stop_target,
            ego_speed_mps=0.0,
            ego_max_deceleration_mps2=2.0,
            ego_in_junction=False,
        )
        self.assertEqual(third_result["decision"], "lane_follow")

    def test_behavior_planner_does_not_release_red_latch_for_different_green_signal(self):
        planner = RuleBasedBehaviorPlanner()
        stop_target = {
            "x_m": 0.0,
            "y_m": 12.0,
            "heading_rad": math.pi / 2.0,
            "lane_id": 1,
            "road_id": 10,
            "distance_m": 12.0,
        }

        first_result = planner.update(
            lane_safety_scores={1: 1.0},
            ego_lane_id=1,
            mode="NORMAL",
            traffic_signal_state="red",
            traffic_stop_target=stop_target,
            traffic_signal_context={
                "signal_found": True,
                "signal_state": "red",
                "signal_distance_m": 12.0,
                "signal_actor_id": 101,
                "signal_actor_name": "controlled_signal",
                "signal_source": "stop_waypoint_match",
            },
            ego_speed_mps=4.0,
            ego_max_deceleration_mps2=2.0,
            ego_in_junction=False,
        )
        self.assertEqual(first_result["decision"], "stop")

        second_result = planner.update(
            lane_safety_scores={1: 1.0},
            ego_lane_id=1,
            mode="NORMAL",
            traffic_signal_state="green",
            traffic_stop_target=stop_target,
            traffic_signal_context={
                "signal_found": True,
                "signal_state": "green",
                "signal_distance_m": 12.0,
                "signal_actor_id": 202,
                "signal_actor_name": "different_signal",
                "signal_source": "ego_vehicle_association",
            },
            ego_speed_mps=0.0,
            ego_max_deceleration_mps2=2.0,
            ego_in_junction=False,
        )
        self.assertEqual(second_result["decision"], "stop")
        self.assertTrue(bool(second_result["traffic_light_debug"]["stop_latched"]))
        self.assertFalse(bool(second_result["blue_dot_rolling"]))

    def test_find_relevant_signal_context_selects_signal_by_stop_waypoint_match(self):
        stop_target = {"x_m": 0.0, "y_m": 12.0, "distance_m": 12.0}
        matching_wp = _DummyWaypoint(
            x_m=0.0,
            y_m=12.5,
            road_id=10,
            section_id=0,
            lane_id=-1,
            is_junction=False,
        )
        wrong_wp = _DummyWaypoint(
            x_m=20.0,
            y_m=40.0,
            road_id=11,
            section_id=0,
            lane_id=-1,
            is_junction=False,
        )
        correct_signal = _FakeTrafficLightActor(
            actor_id=101,
            name="signal_correct",
            state="Red",
            x_m=5.0,
            y_m=12.0,
            stop_waypoints=[matching_wp],
        )
        wrong_signal = _FakeTrafficLightActor(
            actor_id=102,
            name="signal_wrong",
            state="Green",
            x_m=30.0,
            y_m=30.0,
            stop_waypoints=[wrong_wp],
        )
        signal_context = find_relevant_signal_context(
            world=_FakeWorld([wrong_signal, correct_signal]),
            ego_vehicle=_FakeEgoVehicle(),
            ego_transform=types.SimpleNamespace(
                location=types.SimpleNamespace(x=0.0, y=0.0, z=0.0)
            ),
            stop_target=stop_target,
        )

        self.assertTrue(bool(signal_context["signal_found"]))
        self.assertEqual(str(signal_context["signal_state"]), "red")
        self.assertEqual(int(signal_context["signal_actor_id"]), 101)
        self.assertEqual(str(signal_context["signal_actor_name"]), "signal_correct")
        self.assertEqual(str(signal_context["signal_source"]), "stop_waypoint_match")
        self.assertAlmostEqual(float(signal_context["signal_distance_m"]), 12.0, places=3)

    def test_find_relevant_signal_context_prefers_stop_waypoint_match_over_wrong_associated_signal(self):
        stop_target = {"x_m": 0.0, "y_m": 12.0, "distance_m": 12.0}
        matching_wp = _DummyWaypoint(
            x_m=0.0,
            y_m=12.5,
            road_id=10,
            section_id=0,
            lane_id=-1,
            is_junction=False,
        )
        correct_signal = _FakeTrafficLightActor(
            actor_id=101,
            name="signal_correct",
            state="Red",
            x_m=5.0,
            y_m=12.0,
            stop_waypoints=[matching_wp],
        )
        wrong_associated_signal = _FakeTrafficLightActor(
            actor_id=202,
            name="signal_wrong",
            state="Green",
            x_m=50.0,
            y_m=50.0,
            stop_waypoints=[],
        )
        signal_context = find_relevant_signal_context(
            world=_FakeWorld([wrong_associated_signal, correct_signal]),
            ego_vehicle=_FakeEgoVehicle(traffic_light_actor=wrong_associated_signal),
            ego_transform=types.SimpleNamespace(
                location=types.SimpleNamespace(x=0.0, y=0.0, z=0.0)
            ),
            stop_target=stop_target,
        )

        self.assertTrue(bool(signal_context["signal_found"]))
        self.assertEqual(int(signal_context["signal_actor_id"]), 101)
        self.assertEqual(str(signal_context["signal_state"]), "red")
        self.assertEqual(str(signal_context["signal_source"]), "stop_waypoint_match")

    def test_find_relevant_signal_context_uses_lane_aware_stop_waypoint_match_for_later_signal(self):
        stop_target = {
            "x_m": 0.0,
            "y_m": 12.0,
            "distance_m": 25.0,
            "lane_id": 1,
            "road_id": 20,
            "section_id": 1,
        }
        correct_far_wp = _DummyWaypoint(
            x_m=0.0,
            y_m=30.0,
            road_id=20,
            section_id=1,
            lane_id=-1,
            is_junction=False,
        )
        wrong_near_wp = _DummyWaypoint(
            x_m=0.0,
            y_m=12.5,
            road_id=21,
            section_id=0,
            lane_id=-1,
            is_junction=False,
        )
        correct_signal = _FakeTrafficLightActor(
            actor_id=301,
            name="signal_later_correct",
            state="Red",
            x_m=5.0,
            y_m=30.0,
            stop_waypoints=[correct_far_wp],
        )
        wrong_signal = _FakeTrafficLightActor(
            actor_id=302,
            name="signal_wrong_nearby",
            state="Green",
            x_m=2.0,
            y_m=12.0,
            stop_waypoints=[wrong_near_wp],
        )

        signal_context = find_relevant_signal_context(
            world=_FakeWorld([wrong_signal, correct_signal]),
            ego_vehicle=_FakeEgoVehicle(),
            ego_transform=types.SimpleNamespace(
                location=types.SimpleNamespace(x=0.0, y=0.0, z=0.0)
            ),
            stop_target=stop_target,
        )

        self.assertTrue(bool(signal_context["signal_found"]))
        self.assertEqual(int(signal_context["signal_actor_id"]), 301)
        self.assertEqual(str(signal_context["signal_state"]), "red")
        self.assertEqual(str(signal_context["signal_source"]), "stop_waypoint_match")
        self.assertAlmostEqual(float(signal_context["signal_distance_m"]), 25.0, places=3)

    def test_behavior_planner_outputs_signal_debug_when_not_stopping(self):
        planner = RuleBasedBehaviorPlanner()
        result = planner.update(
            lane_safety_scores={1: 1.0},
            ego_lane_id=1,
            mode="NORMAL",
            traffic_signal_state="green",
            traffic_stop_target={"distance_m": 20.0, "lane_id": 1},
            traffic_signal_context={
                "signal_found": True,
                "signal_state": "green",
                "signal_distance_m": 20.0,
                "signal_actor_id": 7,
                "signal_actor_name": "signal_debug",
                "signal_source": "stop_waypoint_match",
            },
            ego_speed_mps=4.0,
            ego_max_deceleration_mps2=2.0,
            ego_in_junction=False,
        )

        self.assertEqual(result["decision"], "lane_follow")
        self.assertIn("traffic_light_debug", result)
        self.assertEqual(str(result["traffic_light_debug"]["signal_actor_name"]), "signal_debug")
        self.assertFalse(bool(result["traffic_light_debug"]["should_stop_now"]))
        self.assertFalse(bool(result["traffic_light_debug"]["stop_decision_active"]))

    def test_behavior_planner_debug_keeps_final_stop_state_while_red_latch_holds(self):
        planner = RuleBasedBehaviorPlanner()
        first = planner.update(
            lane_safety_scores={1: 1.0},
            ego_lane_id=1,
            mode="NORMAL",
            traffic_signal_state="red",
            traffic_stop_target={"distance_m": 20.0, "lane_id": 1},
            traffic_signal_context={
                "signal_found": True,
                "signal_state": "red",
                "signal_distance_m": 20.0,
                "signal_actor_id": 7,
                "signal_actor_name": "signal_debug",
                "signal_source": "stop_waypoint_match",
            },
            ego_speed_mps=4.0,
            ego_max_deceleration_mps2=2.0,
            ego_in_junction=False,
        )
        self.assertEqual(first["decision"], "stop")
        self.assertTrue(bool(first["traffic_light_debug"]["stop_decision_active"]))

        second = planner.update(
            lane_safety_scores={1: 1.0},
            ego_lane_id=1,
            mode="NORMAL",
            traffic_signal_state="red",
            traffic_stop_target={"distance_m": 0.5, "lane_id": 1},
            traffic_signal_context={
                "signal_found": True,
                "signal_state": "red",
                "signal_distance_m": 0.5,
                "signal_actor_id": 7,
                "signal_actor_name": "signal_debug",
                "signal_source": "stop_waypoint_match",
            },
            ego_speed_mps=4.0,
            ego_max_deceleration_mps2=2.0,
            ego_in_junction=False,
        )

        self.assertEqual(second["decision"], "stop")
        self.assertFalse(bool(second["traffic_light_debug"]["should_stop_now"]))
        self.assertTrue(bool(second["traffic_light_debug"]["stop_decision_active"]))


if __name__ == "__main__":
    unittest.main()

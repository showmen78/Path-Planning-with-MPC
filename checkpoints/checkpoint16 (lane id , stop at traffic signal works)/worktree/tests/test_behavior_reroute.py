import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from behavior_planner.planner import RuleBasedBehaviorPlanner
from behavior_planner.reroute import (
    lane_closure_messages,
    remove_cp_messages_by_id,
    reroute_from_lane_closure_messages,
)


class _DummyLaneType:
    Driving = "Driving"


class _DummyCarla:
    LaneType = _DummyLaneType

    class Location:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)


class _DummyWaypoint:
    def __init__(self, road_id, section_id, lane_id, x=0.0, y=0.0):
        self.road_id = int(road_id)
        self.section_id = int(section_id)
        self.lane_id = int(lane_id)
        self.transform = SimpleNamespace(
            location=SimpleNamespace(x=float(x), y=float(y), z=0.0)
        )
        self._left_lane = None
        self._right_lane = None
        self._next_lane = None

    def get_left_lane(self):
        return self._left_lane

    def get_right_lane(self):
        return self._right_lane

    def next(self, distance):
        del distance
        if self._next_lane is None:
            return []
        return [self._next_lane]

    def set_neighbors(self, *, left=None, right=None, next_lane=None):
        self._left_lane = left
        self._right_lane = right
        self._next_lane = next_lane


class _DummyWorldMap:
    def __init__(self, waypoint, waypoint_by_xy=None):
        self._waypoint = waypoint
        self._waypoint_by_xy = dict(waypoint_by_xy or {})

    def get_waypoint(self, location, project_to_road=True, lane_type=None):
        del project_to_road
        del lane_type
        key = (
            round(float(getattr(location, "x", 0.0)), 3),
            round(float(getattr(location, "y", 0.0)), 3),
        )
        if key in self._waypoint_by_xy:
            return self._waypoint_by_xy[key]
        return self._waypoint


class _DummyReroutePlanner:
    def __init__(self):
        self.carla_blocked_route_calls = []
        self.blocking_route_calls = []
        self.segment_calls = []
        self.route_calls = []
        self.avoid_point_calls = []
        self.location_penalty_calls = []
        self.via_location_calls = []
        self.fail_carla_blocked_route = False
        self.fail_blocked_segment_route = False
        self.fail_segment_route = False
        self.carla_blocked_route_waypoints = None
        self.blocking_route_waypoints = None
        self.segment_route_waypoints = None
        self.avoid_route_waypoints = None
        self.location_penalty_route_waypoints = None
        self.via_location_route_waypoints = None
        self.nearest_query_road_id = "12:0"
        self.nearest_query_lane_id = 1
        self.nearest_carla_key = (12, 0, 1)

    def segment_keys_for_road_and_lane(self, *, road_id, lane_id, direction=None):
        self.segment_calls.append(("road_lane", road_id, int(lane_id), direction))
        return [("12:0", int(lane_id))]

    def segment_keys_for_road(self, *, road_id, direction=None):
        self.segment_calls.append(("road", road_id, direction))
        return [("12:0", 1), ("12:0", 2)]

    def nearest_waypoint_query(self, *, x_m, y_m):
        del x_m, y_m
        return SimpleNamespace(
            road_id=str(self.nearest_query_road_id),
            lane_id=int(self.nearest_query_lane_id),
        )

    def nearest_waypoint_carla_key(self, *, x_m, y_m):
        del x_m, y_m
        return tuple(self.nearest_carla_key)

    def plan_route_carla_with_blocked_lanes(
        self,
        *,
        start_location,
        goal_location,
        blocked_raw_lanes,
        replace_stored_route=False,
        fallback_start_xy=None,
        fallback_goal_xy=None,
    ):
        self.carla_blocked_route_calls.append(
            {
                "start_location": start_location,
                "goal_location": goal_location,
                "blocked_raw_lanes": list(blocked_raw_lanes or []),
                "replace_stored_route": bool(replace_stored_route),
                "fallback_start_xy": None if fallback_start_xy is None else list(fallback_start_xy),
                "fallback_goal_xy": None if fallback_goal_xy is None else list(fallback_goal_xy),
            }
        )
        if self.fail_carla_blocked_route:
            return SimpleNamespace(
                route_found=False,
                route_waypoints=[],
                debug_reason="blocked-CARLA-lane route failed",
            )
        return SimpleNamespace(
            route_found=True,
            route_waypoints=list(
                self.carla_blocked_route_waypoints
                or [
                    [float(getattr(start_location, "x", 0.0)), float(getattr(start_location, "y", 0.0))],
                    [5.0, 2.0],
                    [float(getattr(goal_location, "x", 0.0)), float(getattr(goal_location, "y", 0.0))],
                ]
            ),
        )

    def plan_route_astar_blocking_segments(
        self,
        start_xy,
        goal_xy,
        *,
        blocked_segments,
        replace_stored_route=False,
        start_waypoint=None,
        goal_waypoint=None,
    ):
        self.blocking_route_calls.append(
            {
                "start_xy": list(start_xy),
                "goal_xy": list(goal_xy),
                "blocked_segments": list(blocked_segments),
                "replace_stored_route": bool(replace_stored_route),
                "start_waypoint": start_waypoint,
                "goal_waypoint": goal_waypoint,
            }
        )
        if self.fail_blocked_segment_route:
            return SimpleNamespace(
                route_found=False,
                route_waypoints=[],
                debug_reason="blocked-segment route failed",
            )
        return SimpleNamespace(
            route_found=True,
            route_waypoints=list(
                self.blocking_route_waypoints
                or [
                    [float(start_xy[0]), float(start_xy[1])],
                    [5.0, 2.0],
                    [float(goal_xy[0]), float(goal_xy[1])],
                ]
            ),
        )

    def plan_route_astar_with_segment_penalties(
        self,
        start_xy,
        goal_xy,
        *,
        segment_penalties,
        replace_stored_route=False,
        start_waypoint=None,
        goal_waypoint=None,
    ):
        self.route_calls.append(
            {
                "start_xy": list(start_xy),
                "goal_xy": list(goal_xy),
                "segment_penalties": dict(segment_penalties),
                "replace_stored_route": bool(replace_stored_route),
                "start_waypoint": start_waypoint,
                "goal_waypoint": goal_waypoint,
            }
        )
        if self.fail_segment_route:
            return SimpleNamespace(
                route_found=False,
                route_waypoints=[],
                debug_reason="segment-penalty route failed",
            )
        return SimpleNamespace(
            route_found=True,
            route_waypoints=list(
                self.segment_route_waypoints
                or [
                    [float(start_xy[0]), float(start_xy[1])],
                    [5.0, 2.0],
                    [float(goal_xy[0]), float(goal_xy[1])],
                ]
            ),
        )

    def plan_route_astar_avoiding_points(
        self,
        start_xy,
        goal_xy,
        *,
        blocked_points_xy,
        blocked_lane_ids=None,
        block_radius_m=8.0,
        replace_stored_route=False,
        start_waypoint=None,
        goal_waypoint=None,
    ):
        self.avoid_point_calls.append(
            {
                "start_xy": list(start_xy),
                "goal_xy": list(goal_xy),
                "blocked_points_xy": [list(point) for point in list(blocked_points_xy or [])],
                "blocked_lane_ids": None if blocked_lane_ids is None else list(blocked_lane_ids),
                "block_radius_m": float(block_radius_m),
                "replace_stored_route": bool(replace_stored_route),
                "start_waypoint": start_waypoint,
                "goal_waypoint": goal_waypoint,
            }
        )
        return SimpleNamespace(
            route_found=True,
            route_waypoints=list(
                self.avoid_route_waypoints
                or [
                    [float(start_xy[0]), float(start_xy[1])],
                    [6.0, 8.0],
                    [float(goal_xy[0]), float(goal_xy[1])],
                ]
            ),
        )

    def plan_route_from_locations_with_segment_penalties(
        self,
        *,
        start_location,
        goal_location,
        segment_penalties,
        replace_stored_route=False,
        start_waypoint=None,
    ):
        self.location_penalty_calls.append(
            {
                "start_location": start_location,
                "goal_location": goal_location,
                "segment_penalties": dict(segment_penalties),
                "replace_stored_route": bool(replace_stored_route),
                "start_waypoint": start_waypoint,
            }
        )
        return SimpleNamespace(
            route_found=True,
            route_waypoints=list(
                self.location_penalty_route_waypoints
                or [
                    [float(getattr(start_location, "x", 0.0)), float(getattr(start_location, "y", 0.0))],
                    [6.0, 8.0],
                    [float(getattr(goal_location, "x", 0.0)), float(getattr(goal_location, "y", 0.0))],
                ]
            ),
        )

    def plan_route_from_locations_via_locations(
        self,
        *,
        start_location,
        goal_location,
        intermediate_locations=None,
        replace_stored_route=False,
    ):
        self.via_location_calls.append(
            {
                "start_location": start_location,
                "goal_location": goal_location,
                "intermediate_locations": list(intermediate_locations or []),
                "replace_stored_route": bool(replace_stored_route),
            }
        )
        return SimpleNamespace(
            route_found=True,
            route_waypoints=list(
                self.via_location_route_waypoints
                or [
                    [float(getattr(start_location, "x", 0.0)), float(getattr(start_location, "y", 0.0))],
                    [8.0, 4.0],
                    [float(getattr(goal_location, "x", 0.0)), float(getattr(goal_location, "y", 0.0))],
                ]
            ),
        )


class BehaviorRerouteTests(unittest.TestCase):
    def test_lane_closure_messages_are_batched_and_removed_from_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            message_path = os.path.join(tmp_dir, "cp_message.json")
            with open(message_path, "w", encoding="utf-8") as message_file:
                json.dump(
                    [
                        {
                            "id": "closure_1",
                            "type": "lane_closure",
                            "position": [10.0, 20.0],
                        },
                        {
                            "id": "closure_2",
                            "type": "lane_closure",
                            "position": [11.0, 21.0],
                        },
                        {
                            "id": "note_1",
                            "type": "speed_limit",
                            "position": [12.0, 22.0],
                        },
                    ],
                    message_file,
                )

            planner = RuleBasedBehaviorPlanner(
                cp_message_path=message_path,
                cooperative_message_check_frequency_hz=1.0,
            )
            result = planner.update(
                lane_safety_scores={1: 1.0},
                ego_lane_id=1,
                mode="NORMAL",
                current_time_s=0.0,
            )

            self.assertEqual(result["decision"], "reroute")
            self.assertEqual(int(result["target_lane_id"]), 1)
            self.assertEqual(
                [message["id"] for message in result["reroute_messages"]],
                ["closure_1", "closure_2"],
            )

            with open(message_path, "r", encoding="utf-8") as message_file:
                remaining_messages = json.load(message_file)
            self.assertEqual(
                [message["id"] for message in remaining_messages],
                ["note_1"],
            )

    def test_same_lane_closure_message_is_not_reprocessed_while_it_remains_in_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            message_path = os.path.join(tmp_dir, "cp_message.json")
            with open(message_path, "w", encoding="utf-8") as message_file:
                json.dump(
                    [
                        {
                            "id": "closure_once",
                            "type": "lane_closure",
                            "position": [10.0, 20.0],
                        }
                    ],
                    message_file,
                )

            planner = RuleBasedBehaviorPlanner(
                cp_message_path=message_path,
                cooperative_message_check_frequency_hz=2.0,
            )
            first_result = planner.update(
                lane_safety_scores={1: 1.0},
                ego_lane_id=1,
                mode="NORMAL",
                current_time_s=0.0,
            )
            self.assertEqual(first_result["decision"], "reroute")

            second_result = planner.update(
                lane_safety_scores={1: 1.0},
                ego_lane_id=1,
                mode="NORMAL",
                current_time_s=0.5,
            )
            self.assertEqual(second_result["decision"], "lane_follow")

    def test_lane_closure_message_triggers_reroute_in_intersection_mode(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            message_path = os.path.join(tmp_dir, "cp_message.json")
            with open(message_path, "w", encoding="utf-8") as message_file:
                json.dump(
                    [
                        {
                            "id": "closure_intersection",
                            "type": "lane_closure",
                            "position": [4.0, 5.0],
                        }
                    ],
                    message_file,
                )

            planner = RuleBasedBehaviorPlanner(cp_message_path=message_path)
            result = planner.update(
                lane_safety_scores={1: 0.0},
                ego_lane_id=1,
                mode="INTERSECTION",
                current_time_s=0.0,
            )

            self.assertEqual(result["decision"], "reroute")
            self.assertEqual(
                [message["id"] for message in result["reroute_messages"]],
                ["closure_intersection"],
            )

    def test_lane_closure_messages_follow_poll_frequency(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            message_path = os.path.join(tmp_dir, "cp_message.json")
            with open(message_path, "w", encoding="utf-8") as message_file:
                json.dump([], message_file)

            planner = RuleBasedBehaviorPlanner(
                cp_message_path=message_path,
                cooperative_message_check_frequency_hz=1.0,
            )

            first_result = planner.update(
                lane_safety_scores={1: 1.0},
                ego_lane_id=1,
                mode="NORMAL",
                current_time_s=0.0,
            )
            self.assertEqual(first_result["decision"], "lane_follow")

            with open(message_path, "w", encoding="utf-8") as message_file:
                json.dump(
                    [{"id": "closure_late", "type": "lane_closure", "position": [3.0, 4.0]}],
                    message_file,
                )

            second_result = planner.update(
                lane_safety_scores={1: 1.0},
                ego_lane_id=1,
                mode="NORMAL",
                current_time_s=0.5,
            )
            self.assertEqual(second_result["decision"], "lane_follow")

            third_result = planner.update(
                lane_safety_scores={1: 1.0},
                ego_lane_id=1,
                mode="NORMAL",
                current_time_s=1.0,
            )
            self.assertEqual(third_result["decision"], "reroute")
            self.assertEqual(
                [message["id"] for message in third_result["reroute_messages"]],
                ["closure_late"],
            )

    def test_remove_cp_messages_by_id_keeps_unhandled_messages(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            message_path = os.path.join(tmp_dir, "cp_message.json")
            with open(message_path, "w", encoding="utf-8") as message_file:
                json.dump(
                    [
                        {"id": "m1", "type": "lane_closure", "position": [1.0, 2.0]},
                        {"id": "m2", "type": "lane_closure", "position": [3.0, 4.0]},
                    ],
                    message_file,
                )

            remaining = remove_cp_messages_by_id(["m1"], message_path=message_path)

            self.assertEqual([message["id"] for message in remaining], ["m2"])

    def test_lane_closure_messages_filters_non_closure_and_missing_id(self):
        filtered = lane_closure_messages(
            [
                {"id": "ok", "type": "lane_closure", "position": [1.0, 2.0]},
                {"id": "skip", "type": "speed_limit", "position": [1.0, 2.0]},
                {"type": "lane_closure", "position": [1.0, 2.0]},
            ]
        )

        self.assertEqual([message["id"] for message in filtered], ["ok"])

    def test_reroute_retries_via_bypass_when_first_route_is_unchanged(self):
        planner = _DummyReroutePlanner()
        planner.carla_blocked_route_waypoints = [
            [0.0, 0.0],
            [5.0, 0.0],
            [10.0, 0.0],
        ]
        planner.blocking_route_waypoints = [
            [0.0, 0.0],
            [5.0, 0.0],
            [10.0, 0.0],
        ]
        planner.avoid_route_waypoints = [
            [0.0, 0.0],
            [5.0, 0.0],
            [10.0, 0.0],
        ]
        planner.location_penalty_route_waypoints = [
            [0.0, 0.0],
            [5.0, 0.0],
            [10.0, 0.0],
        ]
        planner.via_location_route_waypoints = [
            [0.0, 3.5],
            [5.0, 3.5],
            [10.0, 0.0],
        ]

        ego_waypoint = _DummyWaypoint(road_id=12, section_id=0, lane_id=1, x=0.0, y=0.0)
        ego_adjacent_waypoint = _DummyWaypoint(road_id=12, section_id=0, lane_id=2, x=0.0, y=3.5)
        blocked_waypoint = _DummyWaypoint(road_id=12, section_id=0, lane_id=1, x=5.0, y=0.0)
        adjacent_waypoint = _DummyWaypoint(road_id=12, section_id=0, lane_id=2, x=5.0, y=3.5)
        adjacent_waypoint_ahead = _DummyWaypoint(road_id=12, section_id=0, lane_id=2, x=9.0, y=3.5)
        ego_waypoint.set_neighbors(left=ego_adjacent_waypoint)
        ego_adjacent_waypoint.set_neighbors(right=ego_waypoint, next_lane=adjacent_waypoint)
        blocked_waypoint.set_neighbors(left=adjacent_waypoint)
        adjacent_waypoint.set_neighbors(right=blocked_waypoint, next_lane=adjacent_waypoint_ahead)
        adjacent_waypoint_ahead.set_neighbors(right=blocked_waypoint)
        world_map = _DummyWorldMap(
            ego_waypoint,
            waypoint_by_xy={
                (0.0, 0.0): ego_waypoint,
                (5.0, 0.0): blocked_waypoint,
            },
        )

        result = reroute_from_lane_closure_messages(
            messages=[
                {
                    "id": "closure_same_route",
                    "type": "lane_closure",
                    "position": [5.0, 0.0],
                }
            ],
            world_map=world_map,
            carla=_DummyCarla,
            global_planner=planner,
            ego_transform=SimpleNamespace(
                location=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            ),
            goal_location=SimpleNamespace(x=10.0, y=0.0, z=0.0),
            current_route_points=[
                [0.0, 0.0],
                [5.0, 0.0],
                [10.0, 0.0],
            ],
        )

        self.assertTrue(bool(result["route_points"]))
        self.assertEqual(result["route_points"][0], [0.0, 3.5])
        self.assertEqual(len(planner.carla_blocked_route_calls), 1)
        self.assertEqual(len(planner.via_location_calls), 1)

    def test_reroute_uses_position_when_road_and_lane_are_missing(self):
        planner = _DummyReroutePlanner()
        world_map = _DummyWorldMap(_DummyWaypoint(road_id=12, section_id=0, lane_id=1))
        result = reroute_from_lane_closure_messages(
            messages=[
                {
                    "id": "closure_infer",
                    "type": "lane_closure",
                    "position": [8.0, 4.0],
                }
            ],
            world_map=world_map,
            carla=_DummyCarla,
            global_planner=planner,
            ego_transform=SimpleNamespace(location=SimpleNamespace(x=0.0, y=0.0, z=0.0)),
            goal_location=SimpleNamespace(x=20.0, y=0.0),
        )

        self.assertTrue(bool(result["route_summary"].route_found))
        self.assertEqual(result["handled_message_ids"], ["closure_infer"])
        self.assertEqual(
            planner.carla_blocked_route_calls[0]["blocked_raw_lanes"],
            [(12, 0, 1)],
        )

    def test_reroute_expands_whole_road_when_only_road_id_is_given(self):
        planner = _DummyReroutePlanner()
        world_map = _DummyWorldMap(_DummyWaypoint(road_id=7, section_id=0, lane_id=1))
        result = reroute_from_lane_closure_messages(
            messages=[
                {
                    "id": "closure_road",
                    "type": "lane_closure",
                    "position": [8.0, 4.0],
                    "road_id": 12,
                }
            ],
            world_map=world_map,
            carla=_DummyCarla,
            global_planner=planner,
            ego_transform=SimpleNamespace(location=SimpleNamespace(x=0.0, y=0.0, z=0.0)),
            goal_location=SimpleNamespace(x=20.0, y=0.0),
        )

        self.assertEqual(
            planner.blocking_route_calls[0]["blocked_segments"],
            [("12:0", 1), ("12:0", 2)],
        )
        self.assertEqual(len(planner.carla_blocked_route_calls), 0)
        self.assertEqual(result["handled_message_ids"], ["closure_road"])

    def test_reroute_prefers_position_snap_over_conflicting_message_road_and_lane_ids(self):
        planner = _DummyReroutePlanner()
        planner.nearest_query_road_id = "20:0"
        planner.nearest_query_lane_id = 2
        planner.nearest_carla_key = (20, 0, -2)
        world_map = _DummyWorldMap(_DummyWaypoint(road_id=20, section_id=0, lane_id=-2))

        result = reroute_from_lane_closure_messages(
            messages=[
                {
                    "id": "closure_conflict",
                    "type": "lane_closure",
                    "position": [8.0, 4.0],
                    "road_id": 1,
                    "section_id": 0,
                    "lane_id": 1,
                }
            ],
            world_map=world_map,
            carla=_DummyCarla,
            global_planner=planner,
            ego_transform=SimpleNamespace(location=SimpleNamespace(x=0.0, y=0.0, z=0.0)),
            goal_location=SimpleNamespace(x=20.0, y=0.0),
        )

        self.assertTrue(bool(result["route_summary"].route_found))
        self.assertEqual(
            planner.carla_blocked_route_calls[0]["blocked_raw_lanes"],
            [(20, 0, -2)],
        )
        self.assertEqual(
            result["resolved_messages"][0]["expanded_segments"],
            [["20:0", 2]],
        )
        self.assertEqual(result["resolved_messages"][0]["road_id"], 20)
        self.assertEqual(result["resolved_messages"][0]["lane_id"], 2)
        self.assertEqual(result["resolved_messages"][0]["carla_lane_id"], -2)

    def test_reroute_uses_adjacent_unblocked_start_lane_when_current_lane_is_blocked(self):
        planner = _DummyReroutePlanner()
        blocked_waypoint = _DummyWaypoint(road_id=12, section_id=0, lane_id=1, x=0.0, y=0.0)
        adjacent_waypoint = _DummyWaypoint(road_id=12, section_id=0, lane_id=2, x=0.0, y=3.5)
        blocked_waypoint.set_neighbors(left=adjacent_waypoint)
        adjacent_waypoint.set_neighbors(right=blocked_waypoint)
        world_map = _DummyWorldMap(blocked_waypoint)

        result = reroute_from_lane_closure_messages(
            messages=[
                {
                    "id": "closure_adjacent_start",
                    "type": "lane_closure",
                    "position": [8.0, 4.0],
                }
            ],
            world_map=world_map,
            carla=_DummyCarla,
            global_planner=planner,
            ego_transform=SimpleNamespace(location=SimpleNamespace(x=0.0, y=0.0, z=0.0)),
            goal_location=SimpleNamespace(x=20.0, y=0.0, z=0.0),
        )

        self.assertTrue(bool(result["route_summary"].route_found))
        self.assertEqual(len(planner.carla_blocked_route_calls), 1)
        self.assertEqual(
            [
                float(getattr(planner.carla_blocked_route_calls[0]["start_location"], "x", 0.0)),
                float(getattr(planner.carla_blocked_route_calls[0]["start_location"], "y", 0.0)),
            ],
            [0.0, 3.5],
        )

    def test_reroute_falls_back_to_blocked_point_avoidance_when_penalty_route_fails(self):
        planner = _DummyReroutePlanner()
        planner.fail_carla_blocked_route = True
        planner.fail_blocked_segment_route = True
        world_map = _DummyWorldMap(_DummyWaypoint(road_id=12, section_id=0, lane_id=1))

        result = reroute_from_lane_closure_messages(
            messages=[
                {
                    "id": "closure_fallback",
                    "type": "lane_closure",
                    "position": [8.0, 4.0],
                }
            ],
            world_map=world_map,
            carla=_DummyCarla,
            global_planner=planner,
            ego_transform=SimpleNamespace(location=SimpleNamespace(x=0.0, y=0.0, z=0.0)),
            goal_location=SimpleNamespace(x=20.0, y=0.0, z=0.0),
        )

        self.assertTrue(bool(result["route_summary"].route_found))
        self.assertEqual(len(planner.carla_blocked_route_calls), 1)
        self.assertEqual(len(planner.blocking_route_calls), 1)
        self.assertEqual(len(planner.avoid_point_calls), 1)
        self.assertEqual(planner.avoid_point_calls[0]["blocked_points_xy"], [[8.0, 4.0]])
        self.assertEqual(planner.avoid_point_calls[0]["blocked_lane_ids"], [1])

    def test_reroute_retries_with_point_avoidance_when_penalty_route_still_hits_blocked_point(self):
        planner = _DummyReroutePlanner()
        planner.carla_blocked_route_waypoints = [
            [0.0, 0.0],
            [8.0, 4.0],
            [20.0, 0.0],
        ]
        planner.blocking_route_waypoints = [
            [0.0, 0.0],
            [8.0, 4.0],
            [20.0, 0.0],
        ]
        planner.avoid_route_waypoints = [
            [0.0, 0.0],
            [6.0, 8.0],
            [20.0, 0.0],
        ]
        world_map = _DummyWorldMap(_DummyWaypoint(road_id=12, section_id=0, lane_id=1))

        result = reroute_from_lane_closure_messages(
            messages=[
                {
                    "id": "closure_overlap",
                    "type": "lane_closure",
                    "position": [8.0, 4.0],
                }
            ],
            world_map=world_map,
            carla=_DummyCarla,
            global_planner=planner,
            ego_transform=SimpleNamespace(location=SimpleNamespace(x=0.0, y=0.0, z=0.0)),
            goal_location=SimpleNamespace(x=20.0, y=0.0, z=0.0),
        )

        self.assertTrue(bool(result["route_summary"].route_found))
        self.assertEqual(len(planner.carla_blocked_route_calls), 1)
        self.assertEqual(len(planner.blocking_route_calls), 1)
        self.assertEqual(len(planner.avoid_point_calls), 1)
        self.assertEqual(result["route_points"], [[0.0, 0.0], [6.0, 8.0], [20.0, 0.0]])


if __name__ == "__main__":
    unittest.main()

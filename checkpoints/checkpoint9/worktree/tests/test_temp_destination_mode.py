import math
import types
import unittest

from behavior_planner.temp_destination import (
    _build_route_reference_samples_from_anchor,
    _determine_mode,
    _route_waypoint_from_anchor,
    _should_follow_turn_branch_from_route,
    _walk_forward,
)


class _DummyWaypoint:
    def __init__(
        self,
        road_id: int,
        is_junction: bool,
        x_m: float = 0.0,
        y_m: float = 0.0,
        yaw_deg: float = 0.0,
        lane_id: int = 1,
    ):
        self.road_id = int(road_id)
        self.is_junction = bool(is_junction)
        self.lane_id = int(lane_id)
        self.transform = types.SimpleNamespace(
            location=types.SimpleNamespace(x=float(x_m), y=float(y_m)),
            rotation=types.SimpleNamespace(yaw=float(yaw_deg)),
        )
        self._next_waypoints = []

    def set_next(self, *waypoints):
        self._next_waypoints = list(waypoints)

    def next(self, step_m):
        del step_m
        return list(self._next_waypoints)

    def get_right_lane(self):
        return None

    def get_left_lane(self):
        return None


class _DummyMap:
    def __init__(self, left_wp, straight_wp, ego_wp):
        self._left_wp = left_wp
        self._straight_wp = straight_wp
        self._ego_wp = ego_wp

    def get_waypoint(self, location, project_to_road=True, lane_type=None):
        del project_to_road
        del lane_type
        if abs(float(location.x) - float(self._ego_wp.transform.location.x)) < 1e-6 and abs(
            float(location.y) - float(self._ego_wp.transform.location.y)
        ) < 1e-6:
            return self._ego_wp
        if float(location.x) < -0.5:
            return self._left_wp
        return self._straight_wp


class _DummyCarla:
    class Location:
        def __init__(self, x, y, z):
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)

    class LaneType:
        Driving = "Driving"


class TempDestinationModeTests(unittest.TestCase):
    def test_turn_maneuver_enters_intersection_mode_when_blue_dot_is_near_junction(self):
        start_wp = _DummyWaypoint(road_id=10, is_junction=False)
        junction_wp = _DummyWaypoint(road_id=10, is_junction=True)
        start_wp.set_next(junction_wp)

        is_intersection, road_id = _determine_mode(
            ref_wp=start_wp,
            step_m=5.0,
            intersection_threshold_m=30.0,
            prev_mode=0.0,
            prev_road_id=10,
            next_macro_maneuver="Left Turn",
        )

        self.assertTrue(bool(is_intersection))
        self.assertEqual(int(road_id), 10)

    def test_straight_maneuver_keeps_mode_normal_even_when_blue_dot_is_near_junction(self):
        start_wp = _DummyWaypoint(road_id=12, is_junction=False)
        junction_wp = _DummyWaypoint(road_id=12, is_junction=True)
        start_wp.set_next(junction_wp)

        is_intersection, road_id = _determine_mode(
            ref_wp=start_wp,
            step_m=5.0,
            intersection_threshold_m=30.0,
            prev_mode=1.0,
            prev_road_id=12,
            next_macro_maneuver="Continue Straight",
        )

        self.assertFalse(bool(is_intersection))
        self.assertEqual(int(road_id), 12)

    def test_walk_forward_uses_remaining_route_to_choose_left_branch(self):
        start_wp = _DummyWaypoint(road_id=20, is_junction=True, x_m=0.0, y_m=0.0, yaw_deg=90.0)
        straight_wp = _DummyWaypoint(road_id=21, is_junction=False, x_m=0.0, y_m=2.0, yaw_deg=90.0)
        left_wp = _DummyWaypoint(
            road_id=22,
            is_junction=False,
            x_m=-math.sqrt(2.0),
            y_m=math.sqrt(2.0),
            yaw_deg=135.0,
        )
        start_wp.set_next(straight_wp, left_wp)

        chosen_wp = _walk_forward(
            wp=start_wp,
            distance_m=2.0,
            step_m=2.0,
            route_points=[[0.0, 0.0], [-1.0, 1.0], [-2.0, 2.0]],
            cum_dists=[0.0, math.sqrt(2.0), 2.0 * math.sqrt(2.0)],
        )

        self.assertIs(chosen_wp, left_wp)

    def test_route_waypoint_from_anchor_stays_on_left_branch(self):
        ego_wp = _DummyWaypoint(road_id=30, is_junction=False, x_m=0.0, y_m=0.0, yaw_deg=90.0)
        left_wp = _DummyWaypoint(road_id=31, is_junction=False, x_m=-2.0, y_m=2.0, yaw_deg=135.0)
        straight_wp = _DummyWaypoint(road_id=32, is_junction=False, x_m=0.0, y_m=2.0, yaw_deg=90.0)
        world_map = _DummyMap(left_wp=left_wp, straight_wp=straight_wp, ego_wp=ego_wp)

        route_wp = _route_waypoint_from_anchor(
            world_map=world_map,
            carla=_DummyCarla,
            anchor_wp=ego_wp,
            route_points=[[0.0, 0.0], [-1.0, 1.0], [-2.0, 2.0]],
            lookahead_m=2.0,
            fallback_wp=straight_wp,
        )

        self.assertIs(route_wp, left_wp)

    def test_route_reference_samples_from_anchor_follow_left_branch(self):
        ego_wp = _DummyWaypoint(road_id=40, is_junction=False, x_m=0.0, y_m=0.0, yaw_deg=90.0)
        left_wp = _DummyWaypoint(road_id=41, is_junction=False, x_m=-2.0, y_m=2.0, yaw_deg=135.0)
        straight_wp = _DummyWaypoint(road_id=42, is_junction=False, x_m=0.0, y_m=2.0, yaw_deg=90.0)
        world_map = _DummyMap(left_wp=left_wp, straight_wp=straight_wp, ego_wp=ego_wp)

        samples = _build_route_reference_samples_from_anchor(
            world_map=world_map,
            carla=_DummyCarla,
            anchor_wp=ego_wp,
            route_points=[[0.0, 0.0], [-1.0, 1.0], [-2.0, 2.0]],
            horizon_steps=3,
            step_distance_m=1.0,
            fallback_lane_id=2,
        )

        self.assertGreaterEqual(len(samples), 1)
        self.assertLess(float(samples[-1]["x_ref_m"]), 0.0)

    def test_lane_follow_maneuver_keeps_route_branch_follow_enabled(self):
        self.assertTrue(
            _should_follow_turn_branch_from_route(
                is_intersection=True,
                next_macro_maneuver="Lane Follow",
                decision="LANE_KEEP",
            )
        )


if __name__ == "__main__":
    unittest.main()

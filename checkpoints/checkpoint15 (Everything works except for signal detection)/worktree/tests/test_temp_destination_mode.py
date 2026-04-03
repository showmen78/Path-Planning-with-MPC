import math
import types
import unittest

from behavior_planner.temp_destination import (
    _build_route_reference_samples_from_anchor,
    _determine_mode,
    _route_waypoint_from_anchor,
    _should_follow_turn_branch_from_route,
    _walk_forward,
    build_reference_samples,
    compute_temp_destination,
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
        section_id: int = 0,
    ):
        self.road_id = int(road_id)
        self.section_id = int(section_id)
        self.is_junction = bool(is_junction)
        self.lane_id = int(lane_id)
        self.lane_type = types.SimpleNamespace(name="Driving")
        self.transform = types.SimpleNamespace(
            location=types.SimpleNamespace(x=float(x_m), y=float(y_m), z=0.0),
            rotation=types.SimpleNamespace(yaw=float(yaw_deg)),
        )
        self._next_waypoints = []
        self._left_lane = None
        self._right_lane = None

    def set_next(self, *waypoints):
        self._next_waypoints = list(waypoints)

    def set_lateral(self, *, left=None, right=None):
        self._left_lane = left
        self._right_lane = right

    def next(self, step_m):
        del step_m
        return list(self._next_waypoints)

    def get_right_lane(self):
        return self._right_lane

    def get_left_lane(self):
        return self._left_lane


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


class TempDestinationModeTests(unittest.TestCase):
    def test_turn_maneuver_enters_intersection_mode_when_blue_dot_is_near_junction(self):
        start_wp = _DummyWaypoint(road_id=10, is_junction=False)
        ego_wp = _DummyWaypoint(road_id=10, is_junction=False)
        junction_wp = _DummyWaypoint(road_id=10, is_junction=True)
        start_wp.set_next(junction_wp)

        is_intersection, road_id, entered_intersection = _determine_mode(
            ref_wp=start_wp,
            ego_wp=ego_wp,
            step_m=5.0,
            intersection_threshold_m=30.0,
            prev_mode=0.0,
            prev_road_id=10,
            next_macro_maneuver="Left Turn",
        )

        self.assertTrue(bool(is_intersection))
        self.assertEqual(int(road_id), 10)
        self.assertFalse(bool(entered_intersection))

    def test_straight_maneuver_keeps_mode_normal_even_when_blue_dot_is_near_junction(self):
        start_wp = _DummyWaypoint(road_id=12, is_junction=False)
        ego_wp = _DummyWaypoint(road_id=12, is_junction=False)
        junction_wp = _DummyWaypoint(road_id=12, is_junction=True)
        start_wp.set_next(junction_wp)

        is_intersection, road_id, entered_intersection = _determine_mode(
            ref_wp=start_wp,
            ego_wp=ego_wp,
            step_m=5.0,
            intersection_threshold_m=30.0,
            prev_mode=0.0,
            prev_road_id=12,
            next_macro_maneuver="Continue Straight",
        )

        self.assertFalse(bool(is_intersection))
        self.assertEqual(int(road_id), 12)
        self.assertFalse(bool(entered_intersection))

    def test_intersection_mode_stays_latched_before_ego_enters_junction(self):
        ref_wp = _DummyWaypoint(road_id=15, is_junction=False)
        ego_wp = _DummyWaypoint(road_id=15, is_junction=False)

        is_intersection, road_id, entered_intersection = _determine_mode(
            ref_wp=ref_wp,
            ego_wp=ego_wp,
            step_m=5.0,
            intersection_threshold_m=30.0,
            prev_mode=1.0,
            prev_road_id=15,
            next_macro_maneuver="Continue Straight",
            prev_entered_intersection=False,
        )

        self.assertTrue(bool(is_intersection))
        self.assertEqual(int(road_id), 15)
        self.assertFalse(bool(entered_intersection))

    def test_intersection_mode_stays_active_while_ego_is_in_junction(self):
        ref_wp = _DummyWaypoint(road_id=16, is_junction=False)
        ego_wp = _DummyWaypoint(road_id=16, is_junction=True)

        is_intersection, road_id, entered_intersection = _determine_mode(
            ref_wp=ref_wp,
            ego_wp=ego_wp,
            step_m=5.0,
            intersection_threshold_m=30.0,
            prev_mode=1.0,
            prev_road_id=16,
            next_macro_maneuver="Continue Straight",
            prev_entered_intersection=True,
        )

        self.assertTrue(bool(is_intersection))
        self.assertEqual(int(road_id), 16)
        self.assertTrue(bool(entered_intersection))

    def test_intersection_mode_clears_only_after_ego_exits_junction(self):
        ref_wp = _DummyWaypoint(road_id=17, is_junction=False)
        ego_wp = _DummyWaypoint(road_id=17, is_junction=False)

        is_intersection, road_id, entered_intersection = _determine_mode(
            ref_wp=ref_wp,
            ego_wp=ego_wp,
            step_m=5.0,
            intersection_threshold_m=30.0,
            prev_mode=1.0,
            prev_road_id=17,
            next_macro_maneuver="Continue Straight",
            prev_entered_intersection=True,
        )

        self.assertFalse(bool(is_intersection))
        self.assertEqual(int(road_id), 17)
        self.assertFalse(bool(entered_intersection))

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
                next_macro_maneuver="left",
                decision="lane_follow",
            )
        )

    def test_normal_mode_lane_follow_keeps_blue_dot_on_selected_lane(self):
        right_wp_0 = _DummyWaypoint(road_id=60, section_id=0, is_junction=False, x_m=0.0, y_m=0.0, yaw_deg=90.0, lane_id=1)
        right_wp_1 = _DummyWaypoint(road_id=60, section_id=0, is_junction=False, x_m=0.0, y_m=2.0, yaw_deg=90.0, lane_id=1)
        right_wp_2 = _DummyWaypoint(road_id=60, section_id=0, is_junction=False, x_m=0.0, y_m=4.0, yaw_deg=90.0, lane_id=1)
        right_wp_0.set_next(right_wp_1)
        right_wp_1.set_next(right_wp_2)
        right_wp_2.set_next()

        left_wp_0 = _DummyWaypoint(road_id=60, section_id=0, is_junction=False, x_m=-3.5, y_m=0.0, yaw_deg=90.0, lane_id=2)
        left_wp_1 = _DummyWaypoint(road_id=60, section_id=0, is_junction=False, x_m=-3.5, y_m=2.0, yaw_deg=90.0, lane_id=2)
        left_wp_2 = _DummyWaypoint(road_id=60, section_id=0, is_junction=False, x_m=-3.5, y_m=4.0, yaw_deg=90.0, lane_id=2)
        left_wp_0.set_next(left_wp_1)
        left_wp_1.set_next(left_wp_2)
        left_wp_2.set_next()

        for right_lane_wp, left_lane_wp in (
            (right_wp_0, left_wp_0),
            (right_wp_1, left_wp_1),
            (right_wp_2, left_wp_2),
        ):
            right_lane_wp.set_lateral(left=left_lane_wp)
            left_lane_wp.set_lateral(right=right_lane_wp)

        world_map = _NearestWaypointMap(
            right_wp_0,
            right_wp_1,
            right_wp_2,
            left_wp_0,
            left_wp_1,
            left_wp_2,
        )
        ego_transform = types.SimpleNamespace(location=right_wp_0.transform.location, rotation=right_wp_0.transform.rotation)

        destination = compute_temp_destination(
            world_map=world_map,
            carla=_DummyCarla,
            ego_transform=ego_transform,
            target_lane_id=2,
            decision="lane_follow",
            lookahead_m=4.0,
            target_v_mps=5.0,
            global_route_points=[[0.0, 0.0], [0.0, 2.0], [0.0, 4.0]],
            next_macro_maneuver="straight",
        )

        self.assertAlmostEqual(float(destination[0]), -3.5, places=3)
        self.assertAlmostEqual(float(destination[1]), 4.0, places=3)
        self.assertEqual(int(destination[4]), 2)

    def test_repeated_calls_keep_blue_dot_lookahead_from_ego_instead_of_accumulating(self):
        wp0 = _DummyWaypoint(road_id=60, section_id=0, is_junction=False, x_m=0.0, y_m=0.0, yaw_deg=90.0, lane_id=1)
        wp1 = _DummyWaypoint(road_id=60, section_id=0, is_junction=False, x_m=0.0, y_m=2.0, yaw_deg=90.0, lane_id=1)
        wp2 = _DummyWaypoint(road_id=60, section_id=0, is_junction=False, x_m=0.0, y_m=4.0, yaw_deg=90.0, lane_id=1)
        wp3 = _DummyWaypoint(road_id=60, section_id=0, is_junction=False, x_m=0.0, y_m=6.0, yaw_deg=90.0, lane_id=1)
        wp4 = _DummyWaypoint(road_id=60, section_id=0, is_junction=False, x_m=0.0, y_m=8.0, yaw_deg=90.0, lane_id=1)
        world_map = _NearestWaypointMap(wp0, wp1, wp2, wp3, wp4)
        ego_transform = types.SimpleNamespace(location=wp0.transform.location, rotation=wp0.transform.rotation)

        first_destination = compute_temp_destination(
            world_map=world_map,
            carla=_DummyCarla,
            ego_transform=ego_transform,
            target_lane_id=1,
            decision="lane_follow",
            lookahead_m=4.0,
            target_v_mps=5.0,
            global_route_points=[[0.0, 0.0], [0.0, 2.0], [0.0, 4.0], [0.0, 6.0], [0.0, 8.0]],
            next_macro_maneuver="straight",
        )
        second_destination = compute_temp_destination(
            world_map=world_map,
            carla=_DummyCarla,
            ego_transform=ego_transform,
            target_lane_id=1,
            decision="lane_follow",
            lookahead_m=4.0,
            target_v_mps=5.0,
            global_route_points=[[0.0, 0.0], [0.0, 2.0], [0.0, 4.0], [0.0, 6.0], [0.0, 8.0]],
            mode_reference_xy=(float(first_destination[0]), float(first_destination[1])),
            prev_mode=float(first_destination[5]),
            prev_road_id=int(first_destination[6]),
            next_macro_maneuver="straight",
        )

        self.assertAlmostEqual(float(first_destination[1]), 4.0, places=3)
        self.assertAlmostEqual(float(second_destination[1]), 4.0, places=3)

    def test_intersection_mode_blue_dot_keeps_selected_lane_without_lane_change_decision(self):
        right_wp_0 = _DummyWaypoint(road_id=61, section_id=0, is_junction=False, x_m=0.0, y_m=0.0, yaw_deg=90.0, lane_id=1)
        right_wp_1 = _DummyWaypoint(road_id=61, section_id=0, is_junction=False, x_m=0.0, y_m=2.0, yaw_deg=90.0, lane_id=1)
        right_wp_2 = _DummyWaypoint(road_id=61, section_id=0, is_junction=False, x_m=0.0, y_m=4.0, yaw_deg=90.0, lane_id=1)
        right_wp_0.set_next(right_wp_1)
        right_wp_1.set_next(right_wp_2)
        right_wp_2.set_next()

        left_wp_0 = _DummyWaypoint(road_id=61, section_id=0, is_junction=False, x_m=-3.5, y_m=0.0, yaw_deg=90.0, lane_id=2)
        left_wp_1 = _DummyWaypoint(road_id=61, section_id=0, is_junction=False, x_m=-3.5, y_m=2.0, yaw_deg=90.0, lane_id=2)
        left_wp_2 = _DummyWaypoint(road_id=61, section_id=0, is_junction=False, x_m=-3.5, y_m=4.0, yaw_deg=90.0, lane_id=2)
        left_wp_0.set_next(left_wp_1)
        left_wp_1.set_next(left_wp_2)
        left_wp_2.set_next()

        for right_lane_wp, left_lane_wp in (
            (right_wp_0, left_wp_0),
            (right_wp_1, left_wp_1),
            (right_wp_2, left_wp_2),
        ):
            right_lane_wp.set_lateral(left=left_lane_wp)
            left_lane_wp.set_lateral(right=right_lane_wp)

        world_map = _NearestWaypointMap(
            right_wp_0,
            right_wp_1,
            right_wp_2,
            left_wp_0,
            left_wp_1,
            left_wp_2,
        )
        ego_transform = types.SimpleNamespace(location=right_wp_0.transform.location, rotation=right_wp_0.transform.rotation)

        destination = compute_temp_destination(
            world_map=world_map,
            carla=_DummyCarla,
            ego_transform=ego_transform,
            target_lane_id=2,
            decision="lane_follow",
            lookahead_m=4.0,
            target_v_mps=5.0,
            global_route_points=[[-3.5, 0.0], [-3.5, 2.0], [-3.5, 4.0]],
            next_macro_maneuver="left",
            mode_override="INTERSECTION",
            follow_global_route_lane=True,
        )

        self.assertAlmostEqual(float(destination[0]), -3.5, places=3)
        self.assertAlmostEqual(float(destination[1]), 4.0, places=3)
        self.assertEqual(int(destination[4]), 2)
        self.assertGreater(float(destination[5]), 0.5)

    def test_intersection_mode_blue_dot_changes_lane_after_lane_change_decision(self):
        right_wp_0 = _DummyWaypoint(road_id=62, section_id=0, is_junction=False, x_m=0.0, y_m=0.0, yaw_deg=90.0, lane_id=1)
        right_wp_1 = _DummyWaypoint(road_id=62, section_id=0, is_junction=False, x_m=0.0, y_m=2.0, yaw_deg=90.0, lane_id=1)
        right_wp_2 = _DummyWaypoint(road_id=62, section_id=0, is_junction=False, x_m=0.0, y_m=4.0, yaw_deg=90.0, lane_id=1)
        right_wp_0.set_next(right_wp_1)
        right_wp_1.set_next(right_wp_2)
        right_wp_2.set_next()

        left_wp_0 = _DummyWaypoint(road_id=62, section_id=0, is_junction=False, x_m=-3.5, y_m=0.0, yaw_deg=90.0, lane_id=2)
        left_wp_1 = _DummyWaypoint(road_id=62, section_id=0, is_junction=False, x_m=-3.5, y_m=2.0, yaw_deg=90.0, lane_id=2)
        left_wp_2 = _DummyWaypoint(road_id=62, section_id=0, is_junction=False, x_m=-3.5, y_m=4.0, yaw_deg=90.0, lane_id=2)
        left_wp_0.set_next(left_wp_1)
        left_wp_1.set_next(left_wp_2)
        left_wp_2.set_next()

        for right_lane_wp, left_lane_wp in (
            (right_wp_0, left_wp_0),
            (right_wp_1, left_wp_1),
            (right_wp_2, left_wp_2),
        ):
            right_lane_wp.set_lateral(left=left_lane_wp)
            left_lane_wp.set_lateral(right=right_lane_wp)

        world_map = _NearestWaypointMap(
            right_wp_0,
            right_wp_1,
            right_wp_2,
            left_wp_0,
            left_wp_1,
            left_wp_2,
        )
        ego_transform = types.SimpleNamespace(location=right_wp_0.transform.location, rotation=right_wp_0.transform.rotation)

        destination = compute_temp_destination(
            world_map=world_map,
            carla=_DummyCarla,
            ego_transform=ego_transform,
            target_lane_id=2,
            decision="lane_change_left",
            lookahead_m=4.0,
            target_v_mps=5.0,
            global_route_points=[[-3.5, 0.0], [-3.5, 2.0], [-3.5, 4.0]],
            next_macro_maneuver="left",
            mode_override="INTERSECTION",
        )

        self.assertAlmostEqual(float(destination[0]), -3.5, places=3)
        self.assertAlmostEqual(float(destination[1]), 4.0, places=3)
        self.assertEqual(int(destination[4]), 2)

    def test_intersection_mode_reference_samples_keep_selected_lane_without_lane_change_decision(self):
        right_wp_0 = _DummyWaypoint(road_id=63, section_id=0, is_junction=False, x_m=0.0, y_m=0.0, yaw_deg=90.0, lane_id=1)
        right_wp_1 = _DummyWaypoint(road_id=63, section_id=0, is_junction=False, x_m=0.0, y_m=2.0, yaw_deg=90.0, lane_id=1)
        right_wp_2 = _DummyWaypoint(road_id=63, section_id=0, is_junction=False, x_m=0.0, y_m=4.0, yaw_deg=90.0, lane_id=1)
        right_wp_0.set_next(right_wp_1)
        right_wp_1.set_next(right_wp_2)
        right_wp_2.set_next()

        left_wp_0 = _DummyWaypoint(road_id=63, section_id=0, is_junction=False, x_m=-3.5, y_m=0.0, yaw_deg=90.0, lane_id=2)
        left_wp_1 = _DummyWaypoint(road_id=63, section_id=0, is_junction=False, x_m=-3.5, y_m=2.0, yaw_deg=90.0, lane_id=2)
        left_wp_2 = _DummyWaypoint(road_id=63, section_id=0, is_junction=False, x_m=-3.5, y_m=4.0, yaw_deg=90.0, lane_id=2)
        left_wp_0.set_next(left_wp_1)
        left_wp_1.set_next(left_wp_2)
        left_wp_2.set_next()

        for right_lane_wp, left_lane_wp in (
            (right_wp_0, left_wp_0),
            (right_wp_1, left_wp_1),
            (right_wp_2, left_wp_2),
        ):
            right_lane_wp.set_lateral(left=left_lane_wp)
            left_lane_wp.set_lateral(right=right_lane_wp)

        world_map = _NearestWaypointMap(
            right_wp_0,
            right_wp_1,
            right_wp_2,
            left_wp_0,
            left_wp_1,
            left_wp_2,
        )
        ego_transform = types.SimpleNamespace(location=right_wp_0.transform.location, rotation=right_wp_0.transform.rotation)

        samples = build_reference_samples(
            world_map=world_map,
            carla=_DummyCarla,
            ego_transform=ego_transform,
            target_lane_id=2,
            decision="lane_follow",
            horizon_steps=3,
            step_distance_m=2.0,
            global_route_points=[[-3.5, 0.0], [-3.5, 2.0], [-3.5, 4.0]],
            next_macro_maneuver="left",
            mode_override="INTERSECTION",
            follow_global_route_lane=True,
        )

        self.assertEqual(len(samples), 3)
        self.assertTrue(all(abs(float(sample["x_ref_m"]) + 3.5) <= 1e-6 for sample in samples))
        self.assertTrue(all(int(sample["lane_id"]) == 2 for sample in samples))

    def test_lane_change_decision_moves_blue_dot_to_target_lane_even_when_route_lane_override_is_requested(self):
        right_wp_0 = _DummyWaypoint(road_id=64, section_id=0, is_junction=False, x_m=0.0, y_m=0.0, yaw_deg=90.0, lane_id=1)
        right_wp_1 = _DummyWaypoint(road_id=64, section_id=0, is_junction=False, x_m=0.0, y_m=2.0, yaw_deg=90.0, lane_id=1)
        right_wp_2 = _DummyWaypoint(road_id=64, section_id=0, is_junction=False, x_m=0.0, y_m=4.0, yaw_deg=90.0, lane_id=1)
        right_wp_0.set_next(right_wp_1)
        right_wp_1.set_next(right_wp_2)
        right_wp_2.set_next()

        left_wp_0 = _DummyWaypoint(road_id=64, section_id=0, is_junction=False, x_m=-3.5, y_m=0.0, yaw_deg=90.0, lane_id=2)
        left_wp_1 = _DummyWaypoint(road_id=64, section_id=0, is_junction=False, x_m=-3.5, y_m=2.0, yaw_deg=90.0, lane_id=2)
        left_wp_2 = _DummyWaypoint(road_id=64, section_id=0, is_junction=False, x_m=-3.5, y_m=4.0, yaw_deg=90.0, lane_id=2)
        left_wp_0.set_next(left_wp_1)
        left_wp_1.set_next(left_wp_2)
        left_wp_2.set_next()

        for right_lane_wp, left_lane_wp in (
            (right_wp_0, left_wp_0),
            (right_wp_1, left_wp_1),
            (right_wp_2, left_wp_2),
        ):
            right_lane_wp.set_lateral(left=left_lane_wp)
            left_lane_wp.set_lateral(right=right_lane_wp)

        world_map = _NearestWaypointMap(
            right_wp_0,
            right_wp_1,
            right_wp_2,
            left_wp_0,
            left_wp_1,
            left_wp_2,
        )
        ego_transform = types.SimpleNamespace(location=right_wp_0.transform.location, rotation=right_wp_0.transform.rotation)

        destination = compute_temp_destination(
            world_map=world_map,
            carla=_DummyCarla,
            ego_transform=ego_transform,
            target_lane_id=2,
            decision="lane_change_left",
            lookahead_m=4.0,
            target_v_mps=5.0,
            global_route_points=[[0.0, 0.0], [0.0, 2.0], [0.0, 4.0]],
            next_macro_maneuver="straight",
            follow_global_route_lane=True,
        )

        self.assertAlmostEqual(float(destination[0]), -3.5, places=3)
        self.assertAlmostEqual(float(destination[1]), 4.0, places=3)
        self.assertEqual(int(destination[4]), 2)

    def test_lane_change_reference_samples_move_to_target_lane_even_when_route_lane_override_is_requested(self):
        right_wp_0 = _DummyWaypoint(road_id=65, section_id=0, is_junction=False, x_m=0.0, y_m=0.0, yaw_deg=90.0, lane_id=1)
        right_wp_1 = _DummyWaypoint(road_id=65, section_id=0, is_junction=False, x_m=0.0, y_m=2.0, yaw_deg=90.0, lane_id=1)
        right_wp_2 = _DummyWaypoint(road_id=65, section_id=0, is_junction=False, x_m=0.0, y_m=4.0, yaw_deg=90.0, lane_id=1)
        right_wp_0.set_next(right_wp_1)
        right_wp_1.set_next(right_wp_2)
        right_wp_2.set_next()

        left_wp_0 = _DummyWaypoint(road_id=65, section_id=0, is_junction=False, x_m=-3.5, y_m=0.0, yaw_deg=90.0, lane_id=2)
        left_wp_1 = _DummyWaypoint(road_id=65, section_id=0, is_junction=False, x_m=-3.5, y_m=2.0, yaw_deg=90.0, lane_id=2)
        left_wp_2 = _DummyWaypoint(road_id=65, section_id=0, is_junction=False, x_m=-3.5, y_m=4.0, yaw_deg=90.0, lane_id=2)
        left_wp_0.set_next(left_wp_1)
        left_wp_1.set_next(left_wp_2)
        left_wp_2.set_next()

        for right_lane_wp, left_lane_wp in (
            (right_wp_0, left_wp_0),
            (right_wp_1, left_wp_1),
            (right_wp_2, left_wp_2),
        ):
            right_lane_wp.set_lateral(left=left_lane_wp)
            left_lane_wp.set_lateral(right=right_lane_wp)

        world_map = _NearestWaypointMap(
            right_wp_0,
            right_wp_1,
            right_wp_2,
            left_wp_0,
            left_wp_1,
            left_wp_2,
        )
        ego_transform = types.SimpleNamespace(location=right_wp_0.transform.location, rotation=right_wp_0.transform.rotation)

        samples = build_reference_samples(
            world_map=world_map,
            carla=_DummyCarla,
            ego_transform=ego_transform,
            target_lane_id=2,
            decision="lane_change_left",
            horizon_steps=3,
            step_distance_m=2.0,
            global_route_points=[[0.0, 0.0], [0.0, 2.0], [0.0, 4.0]],
            next_macro_maneuver="straight",
            follow_global_route_lane=True,
        )

        self.assertEqual(len(samples), 3)
        self.assertEqual(int(samples[0]["lane_id"]), 1)
        self.assertAlmostEqual(float(samples[0]["x_ref_m"]), 0.0, places=3)
        self.assertLess(float(samples[1]["x_ref_m"]), 0.0)
        self.assertGreater(float(samples[1]["x_ref_m"]), -3.5)
        self.assertEqual(int(samples[-1]["lane_id"]), 2)
        self.assertAlmostEqual(float(samples[-1]["x_ref_m"]), -3.5, places=3)

    def test_lane_change_reference_samples_transition_smoothly_into_target_lane(self):
        ego_wp = _DummyWaypoint(road_id=50, section_id=0, is_junction=False, x_m=0.0, y_m=0.0, yaw_deg=90.0, lane_id=1)
        ego_wp_next_1 = _DummyWaypoint(road_id=50, section_id=0, is_junction=False, x_m=0.0, y_m=2.0, yaw_deg=90.0, lane_id=1)
        ego_wp_next_2 = _DummyWaypoint(road_id=50, section_id=0, is_junction=False, x_m=0.0, y_m=4.0, yaw_deg=90.0, lane_id=1)
        ego_wp_next_3 = _DummyWaypoint(road_id=50, section_id=0, is_junction=False, x_m=0.0, y_m=6.0, yaw_deg=90.0, lane_id=1)
        ego_wp_next_4 = _DummyWaypoint(road_id=50, section_id=0, is_junction=False, x_m=0.0, y_m=8.0, yaw_deg=90.0, lane_id=1)
        ego_wp_next_5 = _DummyWaypoint(road_id=50, section_id=0, is_junction=False, x_m=0.0, y_m=10.0, yaw_deg=90.0, lane_id=1)
        ego_wp.set_next(ego_wp_next_1)
        ego_wp_next_1.set_next(ego_wp_next_2)
        ego_wp_next_2.set_next(ego_wp_next_3)
        ego_wp_next_3.set_next(ego_wp_next_4)
        ego_wp_next_4.set_next(ego_wp_next_5)
        ego_wp_next_5.set_next()

        left_wp = _DummyWaypoint(road_id=50, section_id=0, is_junction=False, x_m=-3.5, y_m=0.0, yaw_deg=90.0, lane_id=2)
        left_wp_next_1 = _DummyWaypoint(road_id=50, section_id=0, is_junction=False, x_m=-3.5, y_m=2.0, yaw_deg=90.0, lane_id=2)
        left_wp_next_2 = _DummyWaypoint(road_id=50, section_id=0, is_junction=False, x_m=-3.5, y_m=4.0, yaw_deg=90.0, lane_id=2)
        left_wp_next_3 = _DummyWaypoint(road_id=50, section_id=0, is_junction=False, x_m=-3.5, y_m=6.0, yaw_deg=90.0, lane_id=2)
        left_wp_next_4 = _DummyWaypoint(road_id=50, section_id=0, is_junction=False, x_m=-3.5, y_m=8.0, yaw_deg=90.0, lane_id=2)
        left_wp_next_5 = _DummyWaypoint(road_id=50, section_id=0, is_junction=False, x_m=-3.5, y_m=10.0, yaw_deg=90.0, lane_id=2)
        left_wp.set_next(left_wp_next_1)
        left_wp_next_1.set_next(left_wp_next_2)
        left_wp_next_2.set_next(left_wp_next_3)
        left_wp_next_3.set_next(left_wp_next_4)
        left_wp_next_4.set_next(left_wp_next_5)
        left_wp_next_5.set_next()

        for right_lane_wp, left_lane_wp in (
            (ego_wp, left_wp),
            (ego_wp_next_1, left_wp_next_1),
            (ego_wp_next_2, left_wp_next_2),
            (ego_wp_next_3, left_wp_next_3),
            (ego_wp_next_4, left_wp_next_4),
            (ego_wp_next_5, left_wp_next_5),
        ):
            right_lane_wp.set_lateral(left=left_lane_wp)
            left_lane_wp.set_lateral(right=right_lane_wp)

        world_map = _DummyMap(left_wp=left_wp, straight_wp=ego_wp, ego_wp=ego_wp)
        ego_transform = types.SimpleNamespace(location=ego_wp.transform.location, rotation=ego_wp.transform.rotation)

        samples = build_reference_samples(
            world_map=world_map,
            carla=_DummyCarla,
            ego_transform=ego_transform,
            target_lane_id=2,
            decision="lane_change_left",
            horizon_steps=6,
            step_distance_m=2.0,
            global_route_points=[],
            next_macro_maneuver="straight",
        )

        self.assertEqual(len(samples), 6)
        self.assertEqual(int(samples[0]["lane_id"]), 1)
        self.assertAlmostEqual(float(samples[0]["x_ref_m"]), 0.0, places=3)
        self.assertLess(float(samples[2]["x_ref_m"]), 0.0)
        self.assertGreater(float(samples[2]["x_ref_m"]), -3.5)
        self.assertEqual(int(samples[-1]["lane_id"]), 2)
        self.assertAlmostEqual(float(samples[-1]["x_ref_m"]), -3.5, places=3)

    def test_stop_decision_freezes_temp_destination_at_stop_target(self):
        ego_wp = _DummyWaypoint(road_id=70, section_id=0, is_junction=False, x_m=0.0, y_m=0.0, yaw_deg=90.0, lane_id=1)
        stop_wp = _DummyWaypoint(road_id=70, section_id=0, is_junction=False, x_m=0.0, y_m=6.0, yaw_deg=90.0, lane_id=1)
        world_map = _NearestWaypointMap(ego_wp, stop_wp)
        ego_transform = types.SimpleNamespace(location=ego_wp.transform.location, rotation=ego_wp.transform.rotation)

        destination = compute_temp_destination(
            world_map=world_map,
            carla=_DummyCarla,
            ego_transform=ego_transform,
            target_lane_id=1,
            decision="stop",
            lookahead_m=10.0,
            target_v_mps=5.0,
            global_route_points=[[0.0, 0.0], [0.0, 2.0], [0.0, 4.0], [0.0, 6.0], [0.0, 8.0]],
            next_macro_maneuver="straight",
            stop_target_state=[0.0, 6.0, 0.0, math.pi / 2.0, 1, 0.0, 70.0, 0.0],
        )

        self.assertAlmostEqual(float(destination[0]), 0.0, places=3)
        self.assertAlmostEqual(float(destination[1]), 6.0, places=3)
        self.assertAlmostEqual(float(destination[2]), 0.0, places=3)
        self.assertEqual(int(destination[4]), 1)
        self.assertGreater(float(destination[5]), 0.5)

    def test_stop_reference_samples_reach_stop_target_and_hold(self):
        ego_wp = _DummyWaypoint(road_id=71, section_id=0, is_junction=False, x_m=0.0, y_m=0.0, yaw_deg=90.0, lane_id=1)
        wp1 = _DummyWaypoint(road_id=71, section_id=0, is_junction=False, x_m=0.0, y_m=2.0, yaw_deg=90.0, lane_id=1)
        wp2 = _DummyWaypoint(road_id=71, section_id=0, is_junction=False, x_m=0.0, y_m=4.0, yaw_deg=90.0, lane_id=1)
        stop_wp = _DummyWaypoint(road_id=71, section_id=0, is_junction=False, x_m=0.0, y_m=6.0, yaw_deg=90.0, lane_id=1)
        world_map = _NearestWaypointMap(ego_wp, wp1, wp2, stop_wp)
        ego_transform = types.SimpleNamespace(location=ego_wp.transform.location, rotation=ego_wp.transform.rotation)

        samples = build_reference_samples(
            world_map=world_map,
            carla=_DummyCarla,
            ego_transform=ego_transform,
            target_lane_id=1,
            decision="stop",
            horizon_steps=5,
            step_distance_m=2.0,
            global_route_points=[[0.0, 0.0], [0.0, 2.0], [0.0, 4.0], [0.0, 6.0], [0.0, 8.0]],
            next_macro_maneuver="straight",
            stop_target_state=[0.0, 6.0, 0.0, math.pi / 2.0, 1, 0.0, 71.0, 0.0],
        )

        self.assertEqual(len(samples), 5)
        self.assertAlmostEqual(float(samples[-1]["x_ref_m"]), 0.0, places=3)
        self.assertAlmostEqual(float(samples[-1]["y_ref_m"]), 6.0, places=3)
        self.assertAlmostEqual(float(samples[-2]["y_ref_m"]), 6.0, places=3)

    def test_lane_follow_keeps_blue_dot_on_selected_lane_even_when_route_is_elsewhere(self):
        right_wp_0 = _DummyWaypoint(road_id=72, section_id=0, is_junction=False, x_m=0.0, y_m=0.0, yaw_deg=90.0, lane_id=1)
        right_wp_1 = _DummyWaypoint(road_id=72, section_id=0, is_junction=False, x_m=0.0, y_m=2.0, yaw_deg=90.0, lane_id=1)
        right_wp_2 = _DummyWaypoint(road_id=72, section_id=0, is_junction=False, x_m=0.0, y_m=4.0, yaw_deg=90.0, lane_id=1)
        right_wp_0.set_next(right_wp_1)
        right_wp_1.set_next(right_wp_2)
        right_wp_2.set_next()

        left_wp_0 = _DummyWaypoint(road_id=72, section_id=0, is_junction=False, x_m=-3.5, y_m=0.0, yaw_deg=90.0, lane_id=2)
        left_wp_1 = _DummyWaypoint(road_id=72, section_id=0, is_junction=False, x_m=-3.5, y_m=2.0, yaw_deg=90.0, lane_id=2)
        left_wp_2 = _DummyWaypoint(road_id=72, section_id=0, is_junction=False, x_m=-3.5, y_m=4.0, yaw_deg=90.0, lane_id=2)
        left_wp_0.set_next(left_wp_1)
        left_wp_1.set_next(left_wp_2)
        left_wp_2.set_next()

        for right_lane_wp, left_lane_wp in (
            (right_wp_0, left_wp_0),
            (right_wp_1, left_wp_1),
            (right_wp_2, left_wp_2),
        ):
            right_lane_wp.set_lateral(left=left_lane_wp)
            left_lane_wp.set_lateral(right=right_lane_wp)

        world_map = _NearestWaypointMap(
            right_wp_0,
            right_wp_1,
            right_wp_2,
            left_wp_0,
            left_wp_1,
            left_wp_2,
        )
        ego_transform = types.SimpleNamespace(location=right_wp_0.transform.location, rotation=right_wp_0.transform.rotation)

        destination = compute_temp_destination(
            world_map=world_map,
            carla=_DummyCarla,
            ego_transform=ego_transform,
            target_lane_id=2,
            decision="lane_follow",
            lookahead_m=4.0,
            target_v_mps=5.0,
            global_route_points=[[0.0, 0.0], [0.0, 2.0], [0.0, 4.0]],
            next_macro_maneuver="straight",
        )

        self.assertAlmostEqual(float(destination[0]), -3.5, places=3)
        self.assertAlmostEqual(float(destination[1]), 4.0, places=3)
        self.assertEqual(int(destination[4]), 2)

    def test_first_blue_dot_is_projected_onto_global_route_before_lane_change(self):
        right_wp_0 = _DummyWaypoint(road_id=74, section_id=0, is_junction=False, x_m=0.0, y_m=0.0, yaw_deg=90.0, lane_id=1)
        right_wp_1 = _DummyWaypoint(road_id=74, section_id=0, is_junction=False, x_m=0.0, y_m=2.0, yaw_deg=90.0, lane_id=1)
        right_wp_2 = _DummyWaypoint(road_id=74, section_id=0, is_junction=False, x_m=0.0, y_m=4.0, yaw_deg=90.0, lane_id=1)
        right_wp_0.set_next(right_wp_1)
        right_wp_1.set_next(right_wp_2)
        right_wp_2.set_next()

        left_wp_0 = _DummyWaypoint(road_id=74, section_id=0, is_junction=False, x_m=-3.5, y_m=0.0, yaw_deg=90.0, lane_id=2)
        left_wp_1 = _DummyWaypoint(road_id=74, section_id=0, is_junction=False, x_m=-3.5, y_m=2.0, yaw_deg=90.0, lane_id=2)
        left_wp_2 = _DummyWaypoint(road_id=74, section_id=0, is_junction=False, x_m=-3.5, y_m=4.0, yaw_deg=90.0, lane_id=2)
        left_wp_0.set_next(left_wp_1)
        left_wp_1.set_next(left_wp_2)
        left_wp_2.set_next()

        for right_lane_wp, left_lane_wp in (
            (right_wp_0, left_wp_0),
            (right_wp_1, left_wp_1),
            (right_wp_2, left_wp_2),
        ):
            right_lane_wp.set_lateral(left=left_lane_wp)
            left_lane_wp.set_lateral(right=right_lane_wp)

        world_map = _NearestWaypointMap(
            right_wp_0,
            right_wp_1,
            right_wp_2,
            left_wp_0,
            left_wp_1,
            left_wp_2,
        )
        ego_transform = types.SimpleNamespace(location=right_wp_0.transform.location, rotation=right_wp_0.transform.rotation)

        destination = compute_temp_destination(
            world_map=world_map,
            carla=_DummyCarla,
            ego_transform=ego_transform,
            target_lane_id=2,
            decision="lane_change_left",
            lookahead_m=4.0,
            target_v_mps=5.0,
            global_route_points=[[-3.5, 0.0], [-3.5, 2.0], [-3.5, 4.0]],
            next_macro_maneuver="straight",
        )

        self.assertAlmostEqual(float(destination[0]), -3.5, places=3)
        self.assertAlmostEqual(float(destination[1]), 4.0, places=3)
        self.assertEqual(int(destination[4]), 2)


if __name__ == "__main__":
    unittest.main()

import unittest

from behavior_planner.planner import (
    RuleBasedBehaviorPlanner,
    evaluate_intersection_obstacle_response,
    intersection_route_follow_maneuver,
)


class RuleBasedBehaviorPlannerIntersectionTests(unittest.TestCase):
    def test_intersection_lane_follow_maneuver_after_reaching_leftmost_turn_lane(self):
        maneuver = intersection_route_follow_maneuver(
            mode="INTERSECTION",
            next_macro_maneuver="left",
            decision="lane_follow",
            target_lane_id=4,
            available_lane_ids=[1, 2, 3, 4],
            current_road_option="LaneFollow",
        )

        self.assertEqual(maneuver, "left")

    def test_intersection_lane_follow_maneuver_after_reaching_rightmost_turn_lane(self):
        maneuver = intersection_route_follow_maneuver(
            mode="INTERSECTION",
            next_macro_maneuver="right",
            decision="lane_follow",
            target_lane_id=1,
            available_lane_ids=[1, 2, 3, 4],
            current_road_option="LaneFollow",
        )

        self.assertEqual(maneuver, "right")

    def test_intersection_lane_follow_maneuver_does_not_activate_during_lane_change(self):
        maneuver = intersection_route_follow_maneuver(
            mode="INTERSECTION",
            next_macro_maneuver="left",
            decision="lane_change_left",
            target_lane_id=3,
            available_lane_ids=[1, 2, 3, 4],
            current_road_option="LaneFollow",
        )

        self.assertEqual(maneuver, "left")

    def test_intersection_lane_follow_maneuver_activates_when_blue_dot_is_on_turn_block(self):
        maneuver = intersection_route_follow_maneuver(
            mode="INTERSECTION",
            next_macro_maneuver="left",
            decision="lane_change_left",
            target_lane_id=2,
            available_lane_ids=[1, 2],
            current_road_option="LEFT",
        )

        self.assertEqual(maneuver, "lane_follow")

    def test_normal_mode_keeps_route_lane_when_it_is_still_safe(self):
        planner = RuleBasedBehaviorPlanner(hysteresis_delta=0.05)

        result = planner.update(
            lane_safety_scores={1: 0.6, 2: 0.9},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="NORMAL",
            route_optimal_lane_id=1,
            next_macro_maneuver="straight",
        )

        self.assertEqual(result["decision"], "lane_follow")
        self.assertEqual(int(result["target_lane_id"]), 1)

    def test_normal_mode_leaves_route_lane_when_front_obstacle_blocks_it(self):
        planner = RuleBasedBehaviorPlanner(hysteresis_delta=0.05)

        result = planner.update(
            lane_safety_scores={1: 0.0, 2: 0.9},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="NORMAL",
            route_optimal_lane_id=1,
            next_macro_maneuver="straight",
            front_obstacle_distance_by_lane={1: 4.0},
        )

        self.assertEqual(result["decision"], "lane_change_left")
        self.assertEqual(int(result["target_lane_id"]), 2)

    def test_normal_mode_waits_for_safe_target_lane_before_detour(self):
        planner = RuleBasedBehaviorPlanner(
            hysteresis_delta=0.05,
            lane_change_target_safety_threshold=0.10,
        )

        result = planner.update(
            lane_safety_scores={1: 0.0, 2: 0.10},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="NORMAL",
            route_optimal_lane_id=1,
            next_macro_maneuver="straight",
            front_obstacle_distance_by_lane={1: 4.0},
        )

        self.assertEqual(result["decision"], "lane_follow")
        self.assertEqual(int(result["target_lane_id"]), 1)

    def test_intersection_mode_moves_one_lane_at_a_time_toward_route_lane(self):
        planner = RuleBasedBehaviorPlanner(hysteresis_delta=0.05)

        first = planner.update(
            lane_safety_scores={1: 0.2, 2: 0.3, 3: 0.9, 4: 1.0},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=4,
            next_macro_maneuver="left",
        )
        self.assertEqual(first["decision"], "lane_change_left")
        self.assertEqual(int(first["target_lane_id"]), 2)

        second = planner.update(
            lane_safety_scores={1: 0.2, 2: 0.3, 3: 0.9, 4: 1.0},
            ego_lane_id=2,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=4,
            next_macro_maneuver="left",
        )
        self.assertEqual(second["decision"], "lane_change_left")
        self.assertEqual(int(second["target_lane_id"]), 3)

        third = planner.update(
            lane_safety_scores={1: 0.2, 2: 0.3, 3: 0.9, 4: 1.0},
            ego_lane_id=3,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=4,
            next_macro_maneuver="left",
        )
        self.assertEqual(third["decision"], "lane_change_left")
        self.assertEqual(int(third["target_lane_id"]), 4)

    def test_intersection_mode_moves_one_lane_at_a_time_toward_rightmost_lane_for_right_turn(self):
        planner = RuleBasedBehaviorPlanner(hysteresis_delta=0.05)

        first = planner.update(
            lane_safety_scores={1: 0.2, 2: 0.3, 3: 0.9, 4: 1.0},
            ego_lane_id=4,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=1,
            next_macro_maneuver="right",
        )
        self.assertEqual(first["decision"], "lane_change_right")
        self.assertEqual(int(first["target_lane_id"]), 3)

        second = planner.update(
            lane_safety_scores={1: 0.2, 2: 0.3, 3: 0.9, 4: 1.0},
            ego_lane_id=3,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=1,
            next_macro_maneuver="right",
        )
        self.assertEqual(second["decision"], "lane_change_right")
        self.assertEqual(int(second["target_lane_id"]), 2)

        third = planner.update(
            lane_safety_scores={1: 0.2, 2: 0.3, 3: 0.9, 4: 1.0},
            ego_lane_id=2,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=1,
            next_macro_maneuver="right",
        )
        self.assertEqual(third["decision"], "lane_change_right")
        self.assertEqual(int(third["target_lane_id"]), 1)

    def test_intersection_mode_keeps_lane_after_reaching_route_lane(self):
        planner = RuleBasedBehaviorPlanner(hysteresis_delta=0.05)

        left_result = planner.update(
            lane_safety_scores={1: 0.1, 2: 0.5, 3: 0.6},
            ego_lane_id=3,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=3,
            next_macro_maneuver="left",
        )
        self.assertEqual(left_result["decision"], "lane_follow")
        self.assertEqual(int(left_result["target_lane_id"]), 3)

        planner.reset()
        right_result = planner.update(
            lane_safety_scores={1: 0.1, 2: 0.5, 3: 0.6},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=1,
            next_macro_maneuver="right",
        )
        self.assertEqual(right_result["decision"], "lane_follow")
        self.assertEqual(int(right_result["target_lane_id"]), 1)

    def test_intersection_mode_still_prioritizes_route_lane_for_straight_maneuver(self):
        planner = RuleBasedBehaviorPlanner(hysteresis_delta=0.05)

        result = planner.update(
            lane_safety_scores={1: 0.1, 2: 0.9, 3: 0.2},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=3,
            next_macro_maneuver="straight",
        )

        self.assertEqual(result["decision"], "lane_change_left")
        self.assertEqual(int(result["target_lane_id"]), 2)

    def test_intersection_mode_waits_for_safe_left_target_lane(self):
        planner = RuleBasedBehaviorPlanner(
            hysteresis_delta=0.05,
            lane_change_target_safety_threshold=0.10,
        )

        result = planner.update(
            lane_safety_scores={1: 1.0, 2: 0.10, 3: 1.0},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=3,
            next_macro_maneuver="left",
        )

        self.assertEqual(result["decision"], "lane_follow")
        self.assertEqual(int(result["target_lane_id"]), 1)

    def test_intersection_mode_allows_left_lane_change_when_target_lane_is_safe(self):
        planner = RuleBasedBehaviorPlanner(
            hysteresis_delta=0.05,
            lane_change_target_safety_threshold=0.10,
        )

        result = planner.update(
            lane_safety_scores={1: 1.0, 2: 0.11, 3: 1.0},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=3,
            next_macro_maneuver="left",
        )

        self.assertEqual(result["decision"], "lane_change_left")
        self.assertEqual(int(result["target_lane_id"]), 2)

    def test_lane_change_completes_to_lane_follow_on_detour_lane(self):
        planner = RuleBasedBehaviorPlanner(hysteresis_delta=0.05)

        first = planner.update(
            lane_safety_scores={1: 0.0, 2: 0.9},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="NORMAL",
            route_optimal_lane_id=1,
            next_macro_maneuver="straight",
            front_obstacle_distance_by_lane={1: 4.0},
        )
        self.assertEqual(first["decision"], "lane_change_left")

        second = planner.update(
            lane_safety_scores={1: 0.0, 2: 0.9},
            ego_lane_id=2,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="NORMAL",
            route_optimal_lane_id=1,
            next_macro_maneuver="straight",
            front_obstacle_distance_by_lane={1: 4.0},
        )
        self.assertEqual(second["decision"], "lane_follow")

    def test_normal_mode_returns_to_route_lane_when_it_becomes_safe_again(self):
        planner = RuleBasedBehaviorPlanner(hysteresis_delta=0.05)

        first = planner.update(
            lane_safety_scores={1: 0.0, 2: 0.9},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="NORMAL",
            route_optimal_lane_id=1,
            next_macro_maneuver="straight",
            front_obstacle_distance_by_lane={1: 4.0},
        )
        self.assertEqual(first["decision"], "lane_change_left")

        second = planner.update(
            lane_safety_scores={1: 0.9, 2: 0.8},
            ego_lane_id=2,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="NORMAL",
            route_optimal_lane_id=1,
            next_macro_maneuver="straight",
            front_obstacle_distance_by_lane={},
        )
        self.assertEqual(second["decision"], "lane_change_right")
        self.assertEqual(int(second["target_lane_id"]), 1)

    def test_mode_change_resets_lane_change_loop(self):
        planner = RuleBasedBehaviorPlanner(hysteresis_delta=0.05)

        first = planner.update(
            lane_safety_scores={1: 0.0, 2: 0.9},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="NORMAL",
            route_optimal_lane_id=1,
            next_macro_maneuver="straight",
            front_obstacle_distance_by_lane={1: 4.0},
        )
        self.assertEqual(first["decision"], "lane_change_left")

        second = planner.update(
            lane_safety_scores={1: 0.9, 2: 0.1},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=1,
            next_macro_maneuver="right",
        )
        self.assertEqual(second["decision"], "lane_follow")

    def test_intersection_obstacle_response_follows_moving_obstacle_speed(self):
        response = evaluate_intersection_obstacle_response(
            mode="INTERSECTION",
            front_obstacle_speed_mps=3.5,
            original_max_velocity_mps=8.0,
            moving_obstacle_speed_threshold_mps=0.5,
            route_lane_safety_score=0.2,
            static_obstacle_replan_lane_safety_threshold=0.5,
        )

        self.assertTrue(bool(response["follow_moving_obstacle"]))
        self.assertFalse(bool(response["request_static_obstacle_replan"]))
        self.assertAlmostEqual(float(response["speed_cap_mps"]), 3.5)

    def test_intersection_obstacle_response_requests_replan_for_static_obstacle(self):
        response = evaluate_intersection_obstacle_response(
            mode="INTERSECTION",
            front_obstacle_speed_mps=0.0,
            original_max_velocity_mps=8.0,
            moving_obstacle_speed_threshold_mps=0.5,
            route_lane_safety_score=0.4,
            static_obstacle_replan_lane_safety_threshold=0.5,
        )

        self.assertFalse(bool(response["follow_moving_obstacle"]))
        self.assertTrue(bool(response["request_static_obstacle_replan"]))
        self.assertAlmostEqual(float(response["speed_cap_mps"]), 8.0)

    def test_intersection_obstacle_response_does_not_request_replan_when_lane_is_still_safe(self):
        response = evaluate_intersection_obstacle_response(
            mode="INTERSECTION",
            front_obstacle_speed_mps=0.0,
            original_max_velocity_mps=8.0,
            moving_obstacle_speed_threshold_mps=0.5,
            route_lane_safety_score=0.5,
            static_obstacle_replan_lane_safety_threshold=0.5,
        )

        self.assertFalse(bool(response["follow_moving_obstacle"]))
        self.assertFalse(bool(response["request_static_obstacle_replan"]))
        self.assertAlmostEqual(float(response["speed_cap_mps"]), 8.0)


if __name__ == "__main__":
    unittest.main()

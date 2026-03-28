import unittest

from behavior_planner.planner import (
    RuleBasedBehaviorPlanner,
    intersection_route_follow_maneuver,
)


class RuleBasedBehaviorPlannerIntersectionTests(unittest.TestCase):
    def test_intersection_lane_follow_maneuver_after_reaching_leftmost_turn_lane(self):
        maneuver = intersection_route_follow_maneuver(
            mode="INTERSECTION",
            next_macro_maneuver="Left Turn",
            decision="LANE_KEEP",
            target_lane_id=4,
            available_lane_ids=[1, 2, 3, 4],
            current_road_option="LaneFollow",
        )

        self.assertEqual(maneuver, "Lane Follow")

    def test_intersection_lane_follow_maneuver_after_reaching_rightmost_turn_lane(self):
        maneuver = intersection_route_follow_maneuver(
            mode="INTERSECTION",
            next_macro_maneuver="Right Turn",
            decision="LANE_KEEP",
            target_lane_id=1,
            available_lane_ids=[1, 2, 3, 4],
            current_road_option="LaneFollow",
        )

        self.assertEqual(maneuver, "Lane Follow")

    def test_intersection_lane_follow_maneuver_does_not_activate_during_lane_change(self):
        maneuver = intersection_route_follow_maneuver(
            mode="INTERSECTION",
            next_macro_maneuver="Left Turn",
            decision="LANE_CHANGE_LEFT",
            target_lane_id=3,
            available_lane_ids=[1, 2, 3, 4],
            current_road_option="LaneFollow",
        )

        self.assertEqual(maneuver, "Left Turn")

    def test_intersection_lane_follow_maneuver_activates_when_blue_dot_is_on_turn_block(self):
        maneuver = intersection_route_follow_maneuver(
            mode="INTERSECTION",
            next_macro_maneuver="Left Turn",
            decision="LANE_CHANGE_LEFT",
            target_lane_id=2,
            available_lane_ids=[1, 2],
            current_road_option="LEFT",
        )

        self.assertEqual(maneuver, "Lane Follow")

    def test_normal_mode_still_uses_safest_lane(self):
        planner = RuleBasedBehaviorPlanner(hysteresis_delta=0.05)

        result = planner.update(
            lane_safety_scores={1: 0.2, 2: 0.9},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="NORMAL",
            route_optimal_lane_id=1,
            next_macro_maneuver="Continue Straight",
        )

        self.assertEqual(result["decision"], "LANE_CHANGE_LEFT")
        self.assertEqual(int(result["target_lane_id"]), 2)

    def test_intersection_mode_moves_one_lane_at_a_time_toward_leftmost_lane_for_left_turn(self):
        planner = RuleBasedBehaviorPlanner(hysteresis_delta=0.05)

        first = planner.update(
            lane_safety_scores={1: 0.2, 2: 0.3, 3: 0.9, 4: 1.0},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=1,
            next_macro_maneuver="Left Turn",
        )
        self.assertEqual(first["decision"], "LANE_CHANGE_LEFT")
        self.assertEqual(int(first["target_lane_id"]), 2)

        second = planner.update(
            lane_safety_scores={1: 0.2, 2: 0.3, 3: 0.9, 4: 1.0},
            ego_lane_id=2,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=1,
            next_macro_maneuver="Left Turn",
        )
        self.assertEqual(second["decision"], "LANE_CHANGE_LEFT")
        self.assertEqual(int(second["target_lane_id"]), 3)

        third = planner.update(
            lane_safety_scores={1: 0.2, 2: 0.3, 3: 0.9, 4: 1.0},
            ego_lane_id=3,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=1,
            next_macro_maneuver="Left Turn",
        )
        self.assertEqual(third["decision"], "LANE_CHANGE_LEFT")
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
            next_macro_maneuver="Right Turn",
        )
        self.assertEqual(first["decision"], "LANE_CHANGE_RIGHT")
        self.assertEqual(int(first["target_lane_id"]), 3)

        second = planner.update(
            lane_safety_scores={1: 0.2, 2: 0.3, 3: 0.9, 4: 1.0},
            ego_lane_id=3,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=1,
            next_macro_maneuver="Right Turn",
        )
        self.assertEqual(second["decision"], "LANE_CHANGE_RIGHT")
        self.assertEqual(int(second["target_lane_id"]), 2)

        third = planner.update(
            lane_safety_scores={1: 0.2, 2: 0.3, 3: 0.9, 4: 1.0},
            ego_lane_id=2,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=1,
            next_macro_maneuver="Right Turn",
        )
        self.assertEqual(third["decision"], "LANE_CHANGE_RIGHT")
        self.assertEqual(int(third["target_lane_id"]), 1)

    def test_intersection_mode_keeps_lane_after_reaching_extreme_turn_lane(self):
        planner = RuleBasedBehaviorPlanner(hysteresis_delta=0.05)

        left_result = planner.update(
            lane_safety_scores={1: 0.1, 2: 0.5, 3: 0.6},
            ego_lane_id=3,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=1,
            next_macro_maneuver="Left Turn",
        )
        self.assertEqual(left_result["decision"], "LANE_KEEP")
        self.assertEqual(int(left_result["target_lane_id"]), 3)

        planner.reset()
        right_result = planner.update(
            lane_safety_scores={1: 0.1, 2: 0.5, 3: 0.6},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=3,
            next_macro_maneuver="Right Turn",
        )
        self.assertEqual(right_result["decision"], "LANE_KEEP")
        self.assertEqual(int(right_result["target_lane_id"]), 1)

    def test_intersection_mode_with_straight_maneuver_falls_back_to_normal_safety_logic(self):
        planner = RuleBasedBehaviorPlanner(hysteresis_delta=0.05)

        result = planner.update(
            lane_safety_scores={1: 0.1, 2: 0.9, 3: 0.2},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=3,
            next_macro_maneuver="Continue Straight",
        )

        self.assertEqual(result["decision"], "LANE_CHANGE_LEFT")
        self.assertEqual(int(result["target_lane_id"]), 2)


if __name__ == "__main__":
    unittest.main()

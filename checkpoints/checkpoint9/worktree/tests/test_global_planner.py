import unittest

import numpy as np

from utility.global_planner import AStarGlobalPlanner, RoutePlanSummary


class ActiveRouteLaneTests(unittest.TestCase):
    def test_returns_current_route_lane_before_turn(self):
        route_opt_lane = AStarGlobalPlanner._active_lane_id_from_remaining_sequence(
            remaining_lane_ids=[2, 2, 1, 1],
            fallback_lane_id=1,
        )

        self.assertEqual(int(route_opt_lane), 2)

    def test_updates_when_remaining_route_lane_changes(self):
        route_opt_lane = AStarGlobalPlanner._active_lane_id_from_remaining_sequence(
            remaining_lane_ids=[1, 1, 1],
            fallback_lane_id=2,
        )

        self.assertEqual(int(route_opt_lane), 1)

    def test_skips_invalid_entries_and_uses_first_valid_remaining_lane(self):
        route_opt_lane = AStarGlobalPlanner._active_lane_id_from_remaining_sequence(
            remaining_lane_ids=[-1, -1, 2, 2],
            fallback_lane_id=1,
        )

        self.assertEqual(int(route_opt_lane), 2)


class BlueDotRouteProgressTests(unittest.TestCase):
    def _make_planner(self) -> AStarGlobalPlanner:
        planner = object.__new__(AStarGlobalPlanner)
        planner._route_sample_distance_m = 2.0
        planner._stored_route_xy = np.asarray(
            [
                [0.0, 0.0],
                [0.0, 4.0],
                [0.0, 8.0],
                [0.0, 10.0],
                [2.0, 10.0],
                [4.0, 10.0],
                [6.0, 10.0],
                [8.0, 10.0],
            ],
            dtype=float,
        )
        planner._stored_route_cum_dists = AStarGlobalPlanner._route_cumulative_distances(
            planner._stored_route_xy,
        )
        planner._stored_route_options = [
            "LANEFOLLOW",
            "LANEFOLLOW",
            "LANEFOLLOW",
            "LEFT",
            "LEFT",
            "LEFT",
            "LANEFOLLOW",
            "LANEFOLLOW",
        ]
        planner._stored_route_lane_ids = [2, 2, 2, 2, 1, 1, 1, 1]
        planner._stored_route_summary = RoutePlanSummary(
            route_found=True,
            start_road_id="start",
            start_lane_id=2,
            goal_road_id="goal",
            goal_lane_id=1,
            optimal_lane_id=2,
            distance_to_destination_m=18.0,
            next_macro_maneuver="Left Turn",
            route_waypoints=planner._stored_route_xy.tolist(),
            road_options=["LANEFOLLOW", "LEFT"],
        )
        planner._route_info_query_state = {}
        return planner

    def test_blue_dot_lookup_stays_on_current_lane_until_route_progress_reaches_change(self):
        planner = self._make_planner()

        first_summary = planner.get_current_route_info(
            x_m=0.1,
            y_m=8.6,
            query_key="blue_dot",
        )
        second_summary = planner.get_current_route_info(
            x_m=1.7,
            y_m=9.4,
            query_key="blue_dot",
        )

        self.assertEqual(int(first_summary.optimal_lane_id), 2)
        self.assertEqual(int(second_summary.optimal_lane_id), 2)

    def test_blue_dot_lookup_keeps_preparatory_turn_lane_through_intersection(self):
        planner = self._make_planner()

        planner.get_current_route_info(
            x_m=0.1,
            y_m=8.6,
            query_key="blue_dot",
        )
        planner.get_current_route_info(
            x_m=1.7,
            y_m=9.4,
            query_key="blue_dot",
        )
        planner.get_current_route_info(
            x_m=2.4,
            y_m=10.0,
            query_key="blue_dot",
        )
        final_summary = planner.get_current_route_info(
            x_m=3.2,
            y_m=10.0,
            query_key="blue_dot",
        )

        self.assertEqual(int(final_summary.optimal_lane_id), 2)
        self.assertEqual(str(final_summary.current_road_option).upper(), "LEFT")

    def test_blue_dot_lookup_updates_after_turn_block_is_passed(self):
        planner = self._make_planner()

        planner.get_current_route_info(
            x_m=0.1,
            y_m=8.6,
            query_key="blue_dot",
        )
        planner.get_current_route_info(
            x_m=1.7,
            y_m=9.4,
            query_key="blue_dot",
        )
        planner.get_current_route_info(
            x_m=3.2,
            y_m=10.0,
            query_key="blue_dot",
        )
        final_summary = planner.get_current_route_info(
            x_m=7.6,
            y_m=10.0,
            query_key="blue_dot",
        )

        self.assertEqual(int(final_summary.optimal_lane_id), 1)
        self.assertEqual(str(final_summary.current_road_option).upper(), "LANEFOLLOW")


if __name__ == "__main__":
    unittest.main()

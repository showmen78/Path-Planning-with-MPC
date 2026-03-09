"""
Workzone-with-bp scenario on a curved 4-lane road.

This scenario reuses the curved-road rolling-goal behavior from scenario4 and
adds a behavior-planner trigger that shifts the route onto the adjacent lane in
the workzone region.
"""

from __future__ import annotations

import os
import math
from typing import List, Mapping, Sequence

from scenarios.scenario4.scenario import Scenario4 as BaseScenario4
from .behavior_planner import WorkzoneBehaviorPlanner


class WorkzoneScenario(BaseScenario4):
    """Curved-road workzone scenario with lane-change behavior planner."""

    def __init__(self, scenario_yaml_path: str | None = None) -> None:
        default_yaml_path = os.path.join(os.path.dirname(__file__), "scenario.yaml")
        super().__init__(scenario_yaml_path=scenario_yaml_path or default_yaml_path)
        cfg = self.get_config()
        behavior_cfg = dict(cfg.get("behavior_planner", {}))
        self._behavior_planner = WorkzoneBehaviorPlanner(cfg=behavior_cfg)

    def _closest_waypoint_to_anchor(
        self,
        lane_center_waypoints: Sequence[Mapping[str, object]],
    ) -> Mapping[str, object] | None:
        anchor_x_m, anchor_y_m = self._behavior_planner.lane_change_anchor_xy
        valid_waypoints: List[Mapping[str, object]] = []
        for waypoint in lane_center_waypoints:
            if self._position_of_waypoint(waypoint) is None:
                continue
            valid_waypoints.append(waypoint)
        if len(valid_waypoints) == 0:
            return None

        return min(
            valid_waypoints,
            key=lambda waypoint: math.hypot(
                float(self._position_of_waypoint(waypoint)[0]) - float(anchor_x_m),
                float(self._position_of_waypoint(waypoint)[1]) - float(anchor_y_m),
            ),
        )

    def get_step_destination_state(
        self,
        ego_snapshot: Mapping[str, object],
        lane_center_waypoints: List[Mapping[str, object]],
        obstacle_snapshots: List[Mapping[str, object]],
        final_destination_state: Sequence[float],
        simulation_time_s: float,
    ) -> List[float]:
        if not self._behavior_planner.is_triggered:
            return list(
                super().get_step_destination_state(
                    ego_snapshot=ego_snapshot,
                    lane_center_waypoints=lane_center_waypoints,
                    obstacle_snapshots=obstacle_snapshots,
                    final_destination_state=final_destination_state,
                    simulation_time_s=simulation_time_s,
                )
            )

        anchor_waypoint = self._closest_waypoint_to_anchor(lane_center_waypoints=lane_center_waypoints)
        if anchor_waypoint is not None:
            anchor_position = self._position_of_waypoint(anchor_waypoint)
            if anchor_position is not None:
                cfg = self.get_config()
                mpc_goal_cfg = dict(cfg.get("mpc", {}).get("local_goal", {}))
                anchor_speed_mps = max(0.0, float(mpc_goal_cfg.get("v_ref_mps", 5.0)))
                lock_to_final_distance_m = max(
                    0.0,
                    float(mpc_goal_cfg.get("lock_to_final_distance_m", 2.0)),
                )
                ego_x_m = float(ego_snapshot.get("x", 0.0))
                ego_y_m = float(ego_snapshot.get("y", 0.0))
                anchor_x_m = float(anchor_position[0])
                anchor_y_m = float(anchor_position[1])
                anchor_heading_rad = float(anchor_waypoint.get("heading_rad", 0.0))
                dist_ego_to_anchor_m = math.hypot(anchor_x_m - ego_x_m, anchor_y_m - ego_y_m)
                if float(ego_x_m) < float(anchor_x_m) and dist_ego_to_anchor_m > lock_to_final_distance_m:
                    return [anchor_x_m, anchor_y_m, anchor_speed_mps, anchor_heading_rad]

        return list(
            super().get_step_destination_state(
                ego_snapshot=ego_snapshot,
                lane_center_waypoints=lane_center_waypoints,
                obstacle_snapshots=obstacle_snapshots,
                final_destination_state=final_destination_state,
                simulation_time_s=simulation_time_s,
            )
        )

    def get_final_destination_state(
        self,
        ego_snapshot: Mapping[str, object],
        object_snapshots: Sequence[Mapping[str, object]],
        current_final_destination_state: Sequence[float],
        simulation_time_s: float,
    ) -> List[float]:
        return self._behavior_planner.update_final_destination_state(
            ego_snapshot=ego_snapshot,
            object_snapshots=object_snapshots,
            current_final_destination_state=current_final_destination_state,
            simulation_time_s=simulation_time_s,
            lane_center_waypoints=self.get_latest_lane_waypoints(),
        )


def create_scenario() -> WorkzoneScenario:
    return WorkzoneScenario()

"""
VRU scenario on a curved 4-lane road.

This scenario keeps a fixed final destination on the ego lane, but temporary-
destination updates are handled only by the shared rolling-goal base.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Mapping, Sequence, Tuple

from scenarios.rolling_goal_base import RollingGoalScenario


class Scenario4(RollingGoalScenario):
    """VRU scenario with a fixed final destination and shared rolling goals."""

    def __init__(self, scenario_yaml_path: str | None = None) -> None:
        default_yaml_path = os.path.join(os.path.dirname(__file__), "scenario.yaml")
        super().__init__(scenario_yaml_path=scenario_yaml_path or default_yaml_path)
        self._fixed_final_destination_state = self._resolve_fixed_final_destination_state()
        self._config["destination"] = list(self._fixed_final_destination_state)

    @staticmethod
    def _position_of_waypoint(waypoint: Mapping[str, object]) -> Tuple[float, float] | None:
        position_raw = waypoint.get("position")
        if not isinstance(position_raw, (list, tuple)) or len(position_raw) < 2:
            return None
        return float(position_raw[0]), float(position_raw[1])

    def _build_projection_waypoints(
        self,
        road_cfg: Mapping[str, object],
        x_start_m: float,
        x_end_m: float,
    ) -> List[Dict[str, object]]:
        road_type = str(road_cfg.get("type", "straight")).strip().lower()
        if road_type == "curved":
            return self._road_model.build_curved_lane_center_waypoints(
                road_cfg=road_cfg,
                x_start_m=float(x_start_m),
                x_end_m=float(x_end_m),
            )
        return self._road_model.build_lane_center_waypoints(
            road_cfg=road_cfg,
            x_start_m=float(x_start_m),
            x_end_m=float(x_end_m),
        )

    def _resolve_fixed_final_destination_state(self) -> List[float]:
        """Project the configured final destination onto the ego vehicle lane."""

        cfg = self._config
        destination_state = list(cfg.get("destination", [0.0, 0.0, 0.0, 0.0]))
        while len(destination_state) < 4:
            destination_state.append(0.0)

        ego_vehicle_cfg = next(
            (
                vehicle_cfg
                for vehicle_cfg in list(cfg.get("vehicles", []))
                if str(vehicle_cfg.get("type", "")).lower() == "ego"
            ),
            None,
        )
        if not isinstance(ego_vehicle_cfg, Mapping):
            return [float(v) for v in destination_state[:4]]

        ego_state = list(ego_vehicle_cfg.get("initial_state", [0.0, 0.0, 0.0, 0.0]))
        while len(ego_state) < 4:
            ego_state.append(0.0)

        road_cfg = dict(cfg.get("road", {}))
        x_margin_m = max(20.0, float(road_cfg.get("visible_x_margin_m", 20.0)))
        x_start_m = min(float(ego_state[0]), float(destination_state[0])) - x_margin_m
        x_end_m = max(float(ego_state[0]), float(destination_state[0])) + x_margin_m
        lane_center_waypoints = self._build_projection_waypoints(
            road_cfg=road_cfg,
            x_start_m=x_start_m,
            x_end_m=x_end_m,
        )

        valid_waypoints = [
            waypoint
            for waypoint in lane_center_waypoints
            if self._position_of_waypoint(waypoint) is not None
        ]
        if len(valid_waypoints) == 0:
            return [float(v) for v in destination_state[:4]]

        ego_lane_waypoint = min(
            valid_waypoints,
            key=lambda waypoint: math.hypot(
                float(self._position_of_waypoint(waypoint)[0]) - float(ego_state[0]),
                float(self._position_of_waypoint(waypoint)[1]) - float(ego_state[1]),
            ),
        )
        ego_lane_id = int(ego_lane_waypoint.get("lane_id", -1))

        same_lane_waypoints = [
            waypoint
            for waypoint in valid_waypoints
            if int(waypoint.get("lane_id", -1)) == ego_lane_id
        ]
        if len(same_lane_waypoints) == 0:
            same_lane_waypoints = valid_waypoints

        projected_destination_waypoint = min(
            same_lane_waypoints,
            key=lambda waypoint: abs(
                float(self._position_of_waypoint(waypoint)[0]) - float(destination_state[0])
            ),
        )
        projected_position = self._position_of_waypoint(projected_destination_waypoint)
        if projected_position is None:
            return [float(v) for v in destination_state[:4]]

        return [
            float(projected_position[0]),
            float(projected_position[1]),
            float(destination_state[2]),
            float(projected_destination_waypoint.get("heading_rad", destination_state[3])),
        ]

    def get_final_destination_state(
        self,
        ego_snapshot: Mapping[str, object],
        object_snapshots: Sequence[Mapping[str, object]],
        current_final_destination_state: Sequence[float],
        simulation_time_s: float,
    ) -> List[float]:
        """Keep the projected final destination fixed for the whole run."""

        _ = ego_snapshot
        _ = object_snapshots
        _ = current_final_destination_state
        _ = simulation_time_s
        return list(self._fixed_final_destination_state)


def create_scenario() -> Scenario4:
    return Scenario4()

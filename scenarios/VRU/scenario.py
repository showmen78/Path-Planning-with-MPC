"""
Scenario 4: curved 4-lane road scenario.

This scenario reuses shared rolling-goal behavior and adds curved-road
lane-center-follow references for VRU tracking.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Mapping, Sequence, Tuple

from scenarios.rolling_goal_base import RollingGoalScenario


class Scenario4(RollingGoalScenario):
    """
    Scenario4 extends the shared rolling-goal base with curved-road lane-center
    follow support.

    New API exposed for main/MPC (scenario4-only):
    """

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
        """
        Keep the VRU final destination fixed on the ego vehicle's lane.

        The configured destination acts only as a rough longitudinal anchor. On a
        curved road, the actual lane-center point on the ego lane is computed
        once at scenario construction and then held fixed for the whole run.
        """

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
            key=lambda waypoint: abs(float(self._position_of_waypoint(waypoint)[0]) - float(destination_state[0])),
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

    def _destination_heading_from_waypoints(
        self,
        destination_xy: Tuple[float, float],
        lane_center_waypoints: Sequence[Mapping[str, object]],
        final_destination_state: Sequence[float],
    ) -> float | None:
        """
        Infer temporary-destination heading from the closest waypoint on the
        final-destination lane.

        Why scenario4 needs this:
            On curved roads, keeping psi_ref=0 for rolling destinations makes
            the planner pull toward an inconsistent heading and can make the ego
            appear to ignore local temporary goals.
        """

        if len(lane_center_waypoints) == 0 or len(final_destination_state) < 2:
            return None

        y_final_m = float(final_destination_state[1])
        target_x_m, target_y_m = float(destination_xy[0]), float(destination_xy[1])

        valid_waypoints: List[Mapping[str, object]] = []
        for waypoint in lane_center_waypoints:
            if self._position_of_waypoint(waypoint) is None:
                continue
            valid_waypoints.append(waypoint)
        if len(valid_waypoints) == 0:
            return None

        # On curved roads, use the waypoint nearest to the final destination in
        # full 2D to identify the destination lane. Matching only final y can
        # select the wrong lane when the curve folds lanes across similar y.
        target_lane_waypoint = min(
            valid_waypoints,
            key=lambda waypoint: math.hypot(
                float(self._position_of_waypoint(waypoint)[0]) - float(final_destination_state[0]),
                float(self._position_of_waypoint(waypoint)[1]) - y_final_m,
            ),
        )
        target_lane_id = int(target_lane_waypoint.get("lane_id", -1))

        same_lane_waypoints = [
            waypoint
            for waypoint in valid_waypoints
            if int(waypoint.get("lane_id", -1)) == target_lane_id
        ]
        if len(same_lane_waypoints) == 0:
            same_lane_waypoints = valid_waypoints

        nearest_waypoint = min(
            same_lane_waypoints,
            key=lambda waypoint: math.hypot(
                float(self._position_of_waypoint(waypoint)[0]) - target_x_m,
                float(self._position_of_waypoint(waypoint)[1]) - target_y_m,
            ),
        )
        return float(nearest_waypoint.get("heading_rad", 0.0))

    def get_step_destination_state(
        self,
        ego_snapshot: Mapping[str, object],
        lane_center_waypoints: List[Dict[str, object]],
        obstacle_snapshots: List[Mapping[str, object]],
        final_destination_state: Sequence[float],
        simulation_time_s: float,
    ) -> List[float]:
        """
        Scenario4 rolling-goal update:
            Always use the freshly computed temporary destination from current
            lane-center waypoints at each replan tick.
        """

        _ = simulation_time_s

        destination_state = list(
            super().get_step_destination_state(
                ego_snapshot=ego_snapshot,
                lane_center_waypoints=lane_center_waypoints,
                obstacle_snapshots=obstacle_snapshots,
                final_destination_state=list(final_destination_state),
                simulation_time_s=simulation_time_s,
            )
        )

        if len(destination_state) < 2 or len(final_destination_state) < 2:
            return destination_state

        cfg = self.get_config()
        mpc_goal_cfg = dict(cfg.get("mpc", {}).get("local_goal", {}))
        lock_to_final_distance_m = max(
            0.0,
            float(mpc_goal_cfg.get("lock_to_final_distance_m", 2.0)),
        )

        x_ego_m = float(ego_snapshot.get("x", 0.0))
        y_ego_m = float(ego_snapshot.get("y", 0.0))
        x_dest_m = float(destination_state[0])
        y_dest_m = float(destination_state[1])
        x_final_m = float(final_destination_state[0])
        y_final_m = float(final_destination_state[1])

        dist_ego_to_final_m = math.hypot(x_final_m - x_ego_m, y_final_m - y_ego_m)
        dist_dest_to_final_m = math.hypot(x_final_m - x_dest_m, y_final_m - y_dest_m)

        # If destination has already locked to final, clear local memory.
        if dist_dest_to_final_m <= lock_to_final_distance_m + 1e-6:
            return destination_state

        # For curved roads, align temporary-goal heading with the same lane as
        # the final destination, not simply psi=0.
        destination_heading_rad = self._destination_heading_from_waypoints(
            destination_xy=(x_dest_m, y_dest_m),
            lane_center_waypoints=lane_center_waypoints,
            final_destination_state=final_destination_state,
        )
        if destination_heading_rad is not None:
            while len(destination_state) < 4:
                destination_state.append(0.0)
            destination_state[3] = float(destination_heading_rad)

        # Always use the freshly computed temporary destination at replan time.
        return destination_state

    def get_final_destination_state(
        self,
        ego_snapshot: Mapping[str, object],
        object_snapshots: Sequence[Mapping[str, object]],
        current_final_destination_state: Sequence[float],
        simulation_time_s: float,
    ) -> List[float]:
        """
        Keep the VRU final destination fixed for the whole run.
        """

        _ = ego_snapshot
        _ = object_snapshots
        _ = current_final_destination_state
        _ = simulation_time_s
        return list(self._fixed_final_destination_state)


def create_scenario() -> Scenario4:
    return Scenario4()

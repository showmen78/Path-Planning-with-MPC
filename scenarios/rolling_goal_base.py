"""
Shared rolling-goal scenario base.

This replaces the old Scenario1 dependency for scenarios that need:
- YAML-backed scenario config
- road drawing + lane-center waypoint cache
- rolling temporary-destination generation along the destination lane
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Dict, List, Mapping, Sequence, Tuple

from road import RoadModel
from utility import load_yaml_file


class RollingGoalScenario:
    """Common rolling-goal scenario behavior used by curved-road scenarios."""

    def __init__(self, scenario_yaml_path: str) -> None:
        self._scenario_yaml_path = str(scenario_yaml_path)
        self._config: Dict[str, object] = load_yaml_file(self._scenario_yaml_path)
        self._runtime_lookahead_waypoint_count: int | None = None
        self._road_model = RoadModel()
        self._lock_destination_to_final = False
        self._last_final_destination_xy: Tuple[float, float] | None = None

    def get_config(self) -> Dict[str, object]:
        return deepcopy(self._config)

    def set_runtime_lookahead_waypoint_count(self, lookahead_waypoint_count: int | None) -> None:
        """
        Allow main.py to inject the shared lookahead setting from `MPC/mpc.yaml`
        without changing any other scenario-local configuration behavior.
        """

        if lookahead_waypoint_count is None:
            self._runtime_lookahead_waypoint_count = None
            return
        self._runtime_lookahead_waypoint_count = max(1, int(lookahead_waypoint_count))

    def draw_road(
        self,
        surface,
        road_cfg: Mapping[str, object],
        camera_center_world: Tuple[float, float],
        pixels_per_meter: float,
        world_to_screen_fn=None,
    ) -> None:
        _ = world_to_screen_fn
        self._road_model.draw(
            surface=surface,
            road_cfg=road_cfg,
            camera_center_world=camera_center_world,
            pixels_per_meter=pixels_per_meter,
        )

    def get_latest_lane_waypoints(self) -> List[Dict[str, object]]:
        return self._road_model.get_latest_lane_waypoints()

    @staticmethod
    def _position_of_waypoint(waypoint: Mapping[str, object]) -> Tuple[float, float] | None:
        position_raw = waypoint.get("position")
        if not isinstance(position_raw, (list, tuple)) or len(position_raw) < 2:
            return None
        return float(position_raw[0]), float(position_raw[1])

    @staticmethod
    def _waypoint_key(x_m: float, y_m: float) -> Tuple[float, float]:
        return (round(float(x_m), 3), round(float(y_m), 3))

    def _infer_heading_from_waypoints(
        self,
        lane_center_waypoints: Sequence[Mapping[str, object]],
        target_x_m: float,
        target_y_m: float,
    ) -> float | None:
        valid_waypoints = [
            waypoint
            for waypoint in lane_center_waypoints
            if self._position_of_waypoint(waypoint) is not None
        ]
        if len(valid_waypoints) == 0:
            return None

        nearest_waypoint = min(
            valid_waypoints,
            key=lambda waypoint: math.hypot(
                float(self._position_of_waypoint(waypoint)[0]) - float(target_x_m),
                float(self._position_of_waypoint(waypoint)[1]) - float(target_y_m),
            ),
        )
        return float(nearest_waypoint.get("heading_rad", 0.0))

    def _final_destination_lane_waypoints(
        self,
        lane_center_waypoints: Sequence[Mapping[str, object]],
        final_destination_state: Sequence[float],
    ) -> Tuple[List[Mapping[str, object]], Dict[Tuple[float, float], Mapping[str, object]]]:
        valid_waypoints = [
            waypoint
            for waypoint in lane_center_waypoints
            if self._position_of_waypoint(waypoint) is not None
        ]
        if len(valid_waypoints) == 0:
            return [], {}

        target_lane_waypoint = min(
            valid_waypoints,
            key=lambda waypoint: math.hypot(
                float(self._position_of_waypoint(waypoint)[0]) - float(final_destination_state[0]),
                float(self._position_of_waypoint(waypoint)[1]) - float(final_destination_state[1]),
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

        waypoint_by_xy = {
            self._waypoint_key(*self._position_of_waypoint(waypoint)): waypoint
            for waypoint in same_lane_waypoints
            if self._position_of_waypoint(waypoint) is not None
        }
        return same_lane_waypoints, waypoint_by_xy

    def _is_waypoint_blocked(
        self,
        waypoint_position: Tuple[float, float],
        obstacle_snapshots: Sequence[Mapping[str, object]],
    ) -> bool:
        waypoint_x_m = float(waypoint_position[0])
        waypoint_y_m = float(waypoint_position[1])
        for snapshot in obstacle_snapshots:
            obs_x_m = float(snapshot.get("x", 0.0))
            obs_y_m = float(snapshot.get("y", 0.0))
            obs_length_m = max(0.0, float(snapshot.get("length_m", 4.5)))
            obs_width_m = max(0.0, float(snapshot.get("width_m", 2.0)))
            block_radius_m = max(1.5, 0.5 * max(obs_length_m, obs_width_m) + 0.5)
            if math.hypot(waypoint_x_m - obs_x_m, waypoint_y_m - obs_y_m) <= block_radius_m:
                return True
        return False

    def _select_local_waypoint_target(
        self,
        lane_center_waypoints: Sequence[Mapping[str, object]],
        ego_x_m: float,
        ego_y_m: float,
        final_destination_state: Sequence[float],
        lookahead_waypoint_count: int,
        obstacle_snapshots: Sequence[Mapping[str, object]],
    ) -> Tuple[float, float, float] | None:
        same_lane_waypoints, waypoint_by_xy = self._final_destination_lane_waypoints(
            lane_center_waypoints=lane_center_waypoints,
            final_destination_state=final_destination_state,
        )
        if len(same_lane_waypoints) == 0:
            return None

        # Never let the temporary waypoint target advance beyond the final
        # destination along the destination-lane heading. Without this clamp,
        # a lookahead target can reappear past the final stop point and pull the
        # ego forward again after it has already reached the goal.
        final_lane_waypoint = min(
            same_lane_waypoints,
            key=lambda waypoint: math.hypot(
                float(self._position_of_waypoint(waypoint)[0]) - float(final_destination_state[0]),
                float(self._position_of_waypoint(waypoint)[1]) - float(final_destination_state[1]),
            ),
        )
        final_heading_rad = float(final_lane_waypoint.get("heading_rad", 0.0))
        final_dir_x = math.cos(final_heading_rad)
        final_dir_y = math.sin(final_heading_rad)

        def _is_at_or_before_final(waypoint: Mapping[str, object]) -> bool:
            waypoint_position = self._position_of_waypoint(waypoint)
            if waypoint_position is None:
                return False
            along_final_m = (
                final_dir_x * (float(waypoint_position[0]) - float(final_destination_state[0]))
                + final_dir_y * (float(waypoint_position[1]) - float(final_destination_state[1]))
            )
            return along_final_m <= 1e-6

        same_lane_waypoints = [
            waypoint
            for waypoint in same_lane_waypoints
            if _is_at_or_before_final(waypoint)
        ]
        if len(same_lane_waypoints) == 0:
            same_lane_waypoints = [final_lane_waypoint]
        waypoint_by_xy = {
            self._waypoint_key(*self._position_of_waypoint(waypoint)): waypoint
            for waypoint in same_lane_waypoints
            if self._position_of_waypoint(waypoint) is not None
        }

        forward_waypoints = [
            waypoint
            for waypoint in same_lane_waypoints
            if (
                math.cos(float(waypoint.get("heading_rad", 0.0))) * (float(self._position_of_waypoint(waypoint)[0]) - float(ego_x_m))
                + math.sin(float(waypoint.get("heading_rad", 0.0))) * (float(self._position_of_waypoint(waypoint)[1]) - float(ego_y_m))
            ) >= -1e-6
        ]
        seed_candidates = forward_waypoints if len(forward_waypoints) > 0 else same_lane_waypoints

        current_waypoint = min(
            seed_candidates,
            key=lambda waypoint: math.hypot(
                float(self._position_of_waypoint(waypoint)[0]) - float(ego_x_m),
                float(self._position_of_waypoint(waypoint)[1]) - float(ego_y_m),
            ),
        )

        traversed_waypoints: List[Mapping[str, object]] = [current_waypoint]
        visited_keys = {
            self._waypoint_key(
                float(self._position_of_waypoint(current_waypoint)[0]),
                float(self._position_of_waypoint(current_waypoint)[1]),
            )
        }

        for _ in range(max(0, int(lookahead_waypoint_count))):
            next_position_raw = current_waypoint.get("next", None)
            if not isinstance(next_position_raw, (list, tuple)) or len(next_position_raw) < 2:
                break
            next_key = self._waypoint_key(float(next_position_raw[0]), float(next_position_raw[1]))
            if next_key in visited_keys:
                break
            next_waypoint = waypoint_by_xy.get(next_key)
            if next_waypoint is None:
                break
            traversed_waypoints.append(next_waypoint)
            visited_keys.add(next_key)
            current_waypoint = next_waypoint

        for waypoint in reversed(traversed_waypoints):
            current_position = self._position_of_waypoint(waypoint)
            if current_position is None:
                continue
            if self._is_waypoint_blocked(current_position, obstacle_snapshots=obstacle_snapshots):
                continue
            local_heading_rad = float(waypoint.get("heading_rad", 0.0))
            return (float(current_position[0]), float(current_position[1]), local_heading_rad)

        return None

    def get_step_destination_state(
        self,
        ego_snapshot: Mapping[str, object],
        lane_center_waypoints: List[Mapping[str, object]],
        obstacle_snapshots: List[Mapping[str, object]],
        final_destination_state: List[float] | Tuple[float, ...],
        simulation_time_s: float,
    ) -> List[float]:
        """
        Provide a rolling temporary destination until the final destination
        is close enough to lock.
        """

        _ = float(simulation_time_s)
        cfg = self.get_config()
        mpc_goal_cfg = dict(cfg.get("mpc", {}).get("local_goal", {}))
        legacy_local_goal_cfg = dict(cfg.get("local_goal", {}))
        shared_lookahead_waypoint_count = (
            int(self._runtime_lookahead_waypoint_count)
            if self._runtime_lookahead_waypoint_count is not None
            else 20
        )
        lookahead_waypoint_count = max(
            1,
            int(mpc_goal_cfg.get("lookahead_waypoint_count", shared_lookahead_waypoint_count)),
        )
        lock_to_final_distance_m = max(
            0.0,
            float(mpc_goal_cfg.get("lock_to_final_distance_m", 2.0)),
        )
        final_reached_threshold_m = max(
            0.05,
            float(mpc_goal_cfg.get("final_reached_threshold_m", cfg.get("mpc", {}).get("destination_reached_threshold_m", 0.5))),
        )
        temp_destination_v_ref_mps = max(
            0.0,
            float(mpc_goal_cfg.get("v_ref_mps", legacy_local_goal_cfg.get("v_ref_mps", 10.0))),
        )

        x_ego_m = float(ego_snapshot.get("x", 0.0))
        y_ego_m = float(ego_snapshot.get("y", 0.0))
        ego_psi_rad = float(ego_snapshot.get("psi", 0.0))
        x_final_m = float(final_destination_state[0])
        y_final_m = float(final_destination_state[1])
        final_xy = (float(x_final_m), float(y_final_m))

        final_heading_rad = self._infer_heading_from_waypoints(
            lane_center_waypoints=lane_center_waypoints,
            target_x_m=float(x_final_m),
            target_y_m=float(y_final_m),
        )
        if final_heading_rad is None:
            if len(final_destination_state) >= 4:
                final_heading_rad = float(final_destination_state[3])
            else:
                final_heading_rad = float(ego_psi_rad)

        if self._last_final_destination_xy is None:
            self._last_final_destination_xy = final_xy
        elif math.hypot(
            float(self._last_final_destination_xy[0]) - float(final_xy[0]),
            float(self._last_final_destination_xy[1]) - float(final_xy[1]),
        ) > 1e-6:
            self._lock_destination_to_final = False
            self._last_final_destination_xy = final_xy

        if math.hypot(x_final_m - x_ego_m, y_final_m - y_ego_m) <= final_reached_threshold_m:
            self._lock_destination_to_final = True
            return [x_final_m, y_final_m, 0.0, float(final_heading_rad)]

        if self._lock_destination_to_final:
            return [x_final_m, y_final_m, 0.0, float(final_heading_rad)]

        waypoint_target = self._select_local_waypoint_target(
            lane_center_waypoints=lane_center_waypoints,
            ego_x_m=x_ego_m,
            ego_y_m=y_ego_m,
            final_destination_state=list(final_destination_state),
            lookahead_waypoint_count=lookahead_waypoint_count,
            obstacle_snapshots=list(obstacle_snapshots),
        )
        if waypoint_target is None:
            return [x_ego_m, y_ego_m, 0.0, float(ego_psi_rad)]

        x_local_m = float(waypoint_target[0])
        y_local_m = float(waypoint_target[1])
        local_heading_rad = float(waypoint_target[2])

        if math.hypot(x_final_m - x_local_m, y_final_m - y_local_m) <= lock_to_final_distance_m:
            self._lock_destination_to_final = True
            return [x_final_m, y_final_m, 0.0, float(final_heading_rad)]

        return [x_local_m, y_local_m, temp_destination_v_ref_mps, local_heading_rad]

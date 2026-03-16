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
        self._runtime_local_goal_cfg: Dict[str, object] = {}
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
            self._runtime_local_goal_cfg.pop("lookahead_waypoint_count", None)
            return
        self._runtime_lookahead_waypoint_count = max(1, int(lookahead_waypoint_count))
        self._runtime_local_goal_cfg["lookahead_waypoint_count"] = self._runtime_lookahead_waypoint_count

    def set_runtime_local_goal_config(self, local_goal_cfg: Mapping[str, object] | None) -> None:
        """
        Allow main.py to inject the shared rolling-goal configuration from
        `MPC/mpc.yaml` while preserving the scenario-local override behavior.
        """

        self._runtime_local_goal_cfg = dict(local_goal_cfg or {})
        lookahead_waypoint_count = self._runtime_local_goal_cfg.get("lookahead_waypoint_count")
        if lookahead_waypoint_count is None:
            self._runtime_lookahead_waypoint_count = None
        else:
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

    @staticmethod
    def _distance_between_points(
        start_xy: Tuple[float, float],
        end_xy: Tuple[float, float],
    ) -> float:
        return math.hypot(float(end_xy[0]) - float(start_xy[0]), float(end_xy[1]) - float(start_xy[1]))

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

    def _follow_waypoint_chain(
        self,
        start_waypoint: Mapping[str, object],
        waypoint_by_xy: Mapping[Tuple[float, float], Mapping[str, object]],
        max_distance_m: float | None,
        max_waypoint_steps: int | None,
        origin_xy: Tuple[float, float],
    ) -> List[Mapping[str, object]]:
        traversed_waypoints: List[Mapping[str, object]] = [start_waypoint]
        start_position = self._position_of_waypoint(start_waypoint)
        if start_position is None:
            return traversed_waypoints

        cumulative_distance_m = self._distance_between_points(origin_xy, start_position)
        current_waypoint = start_waypoint
        current_position = start_position
        visited_keys = {self._waypoint_key(float(current_position[0]), float(current_position[1]))}

        while True:
            if max_distance_m is not None and cumulative_distance_m >= max_distance_m:
                break
            if max_waypoint_steps is not None and (len(traversed_waypoints) - 1) >= max_waypoint_steps:
                break

            next_position_raw = current_waypoint.get("next", None)
            if not isinstance(next_position_raw, (list, tuple)) or len(next_position_raw) < 2:
                break
            next_key = self._waypoint_key(float(next_position_raw[0]), float(next_position_raw[1]))
            if next_key in visited_keys:
                break
            next_waypoint = waypoint_by_xy.get(next_key)
            if next_waypoint is None:
                break

            next_position = self._position_of_waypoint(next_waypoint)
            if next_position is None:
                break

            cumulative_distance_m += self._distance_between_points(current_position, next_position)
            traversed_waypoints.append(next_waypoint)
            visited_keys.add(next_key)
            current_waypoint = next_waypoint
            current_position = next_position

        return traversed_waypoints

    def _collect_forward_sample_positions(
        self,
        start_waypoint: Mapping[str, object],
        waypoint_by_xy: Mapping[Tuple[float, float], Mapping[str, object]],
        origin_xy: Tuple[float, float],
        target_distances_m: Sequence[float],
    ) -> List[Tuple[float, float]]:
        if len(target_distances_m) == 0:
            return []

        start_position = self._position_of_waypoint(start_waypoint)
        if start_position is None:
            return []

        sample_positions: List[Tuple[float, float]] = []
        cumulative_distance_m = self._distance_between_points(origin_xy, start_position)
        current_waypoint = start_waypoint
        current_position = start_position
        visited_keys = {self._waypoint_key(float(current_position[0]), float(current_position[1]))}

        while len(sample_positions) < len(target_distances_m) and cumulative_distance_m >= target_distances_m[len(sample_positions)]:
            sample_positions.append((float(current_position[0]), float(current_position[1])))

        while len(sample_positions) < len(target_distances_m):
            next_position_raw = current_waypoint.get("next", None)
            if not isinstance(next_position_raw, (list, tuple)) or len(next_position_raw) < 2:
                break
            next_key = self._waypoint_key(float(next_position_raw[0]), float(next_position_raw[1]))
            if next_key in visited_keys:
                break
            next_waypoint = waypoint_by_xy.get(next_key)
            if next_waypoint is None:
                break

            next_position = self._position_of_waypoint(next_waypoint)
            if next_position is None:
                break

            cumulative_distance_m += self._distance_between_points(current_position, next_position)
            current_waypoint = next_waypoint
            current_position = next_position
            visited_keys.add(next_key)

            while len(sample_positions) < len(target_distances_m) and cumulative_distance_m >= target_distances_m[len(sample_positions)]:
                sample_positions.append((float(current_position[0]), float(current_position[1])))

        return sample_positions

    def _estimate_path_curvature(
        self,
        start_waypoint: Mapping[str, object],
        waypoint_by_xy: Mapping[Tuple[float, float], Mapping[str, object]],
        origin_xy: Tuple[float, float],
        sample_spacing_m: float,
    ) -> float:
        if sample_spacing_m <= 0.0:
            return 0.0

        sample_positions = self._collect_forward_sample_positions(
            start_waypoint=start_waypoint,
            waypoint_by_xy=waypoint_by_xy,
            origin_xy=origin_xy,
            target_distances_m=[sample_spacing_m, 2.0 * sample_spacing_m, 3.0 * sample_spacing_m],
        )
        if len(sample_positions) < 3:
            return 0.0

        p1, p2, p3 = sample_positions[0], sample_positions[1], sample_positions[2]
        side_a = self._distance_between_points(p1, p2)
        side_b = self._distance_between_points(p2, p3)
        side_c = self._distance_between_points(p1, p3)
        if min(side_a, side_b, side_c) <= 1e-9:
            return 0.0

        triangle_area = 0.5 * abs(
            (float(p2[0]) - float(p1[0])) * (float(p3[1]) - float(p1[1]))
            - (float(p3[0]) - float(p1[0])) * (float(p2[1]) - float(p1[1]))
        )
        if triangle_area <= 1e-12:
            return 0.0

        return float((4.0 * triangle_area) / (side_a * side_b * side_c))

    def _select_local_waypoint_target(
        self,
        lane_center_waypoints: Sequence[Mapping[str, object]],
        ego_x_m: float,
        ego_y_m: float,
        final_destination_state: Sequence[float],
        lookahead_waypoint_count: int,
        obstacle_snapshots: Sequence[Mapping[str, object]],
        ego_speed_mps: float,
        dynamic_lookahead_cfg: Mapping[str, object] | None = None,
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

        lookahead_distance_m: float | None = None
        if bool((dynamic_lookahead_cfg or {}).get("enabled", False)):
            min_distance_m = max(
                0.0,
                float((dynamic_lookahead_cfg or {}).get("min_distance_m", 0.0)),
            )
            max_distance_m = max(
                min_distance_m,
                float((dynamic_lookahead_cfg or {}).get("max_distance_m", min_distance_m)),
            )
            speed_gain = float((dynamic_lookahead_cfg or {}).get("speed_gain", 0.0))
            curvature_gain = float((dynamic_lookahead_cfg or {}).get("curvature_gain", 0.0))
            sample_spacing_m = max(
                0.1,
                float((dynamic_lookahead_cfg or {}).get("curvature_sample_spacing_m", 5.0)),
            )

            local_curvature = self._estimate_path_curvature(
                start_waypoint=current_waypoint,
                waypoint_by_xy=waypoint_by_xy,
                origin_xy=(float(ego_x_m), float(ego_y_m)),
                sample_spacing_m=sample_spacing_m,
            )
            raw_lookahead_distance_m = (
                min_distance_m
                + speed_gain * max(0.0, float(ego_speed_mps))
                - curvature_gain * abs(local_curvature)
            )
            lookahead_distance_m = min(
                max_distance_m,
                max(min_distance_m, raw_lookahead_distance_m),
            )

        traversed_waypoints = self._follow_waypoint_chain(
            start_waypoint=current_waypoint,
            waypoint_by_xy=waypoint_by_xy,
            max_distance_m=lookahead_distance_m,
            max_waypoint_steps=None if lookahead_distance_m is not None else max(0, int(lookahead_waypoint_count)),
            origin_xy=(float(ego_x_m), float(ego_y_m)),
        )

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
        runtime_local_goal_cfg = dict(self._runtime_local_goal_cfg)
        scenario_local_goal_cfg = dict(cfg.get("mpc", {}).get("local_goal", {}))
        mpc_goal_cfg = {**runtime_local_goal_cfg, **scenario_local_goal_cfg}
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
        dynamic_lookahead_cfg = {
            "enabled": bool(mpc_goal_cfg.get("dynamic_lookahead_enabled", False)),
            "min_distance_m": max(0.0, float(mpc_goal_cfg.get("dynamic_lookahead_min_distance_m", 0.0))),
            "max_distance_m": max(0.0, float(mpc_goal_cfg.get("dynamic_lookahead_max_distance_m", 0.0))),
            "speed_gain": float(mpc_goal_cfg.get("dynamic_lookahead_speed_gain", 0.0)),
            "curvature_gain": float(mpc_goal_cfg.get("dynamic_lookahead_curvature_gain", 0.0)),
            "curvature_sample_spacing_m": max(
                0.1,
                float(mpc_goal_cfg.get("dynamic_lookahead_curvature_sample_spacing_m", 5.0)),
            ),
        }

        x_ego_m = float(ego_snapshot.get("x", 0.0))
        y_ego_m = float(ego_snapshot.get("y", 0.0))
        ego_speed_mps = float(
            ego_snapshot.get(
                "v",
                (ego_snapshot.get("current_state", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0])[2],
            )
        )
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
            ego_speed_mps=ego_speed_mps,
            dynamic_lookahead_cfg=dynamic_lookahead_cfg,
        )
        if waypoint_target is None:
            # Keep the temporary destination ahead of the ego even when the
            # obstacle-aware target selection cannot find an unblocked forward
            # waypoint. In that case, fall back to the same lookahead search
            # without waypoint blocking so the local goal does not collapse
            # onto the ego position at low speed.
            waypoint_target = self._select_local_waypoint_target(
                lane_center_waypoints=lane_center_waypoints,
                ego_x_m=x_ego_m,
                ego_y_m=y_ego_m,
                final_destination_state=list(final_destination_state),
                lookahead_waypoint_count=lookahead_waypoint_count,
                obstacle_snapshots=[],
                ego_speed_mps=ego_speed_mps,
                dynamic_lookahead_cfg=dynamic_lookahead_cfg,
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

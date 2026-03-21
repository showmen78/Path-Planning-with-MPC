"""
LTV-MPC (QP + OSQP) for MPC_custom.

Model and optimization summary:
1. State:  X_k = [x_k, y_k, v_k, psi_k]
2. Input:  U_k = [a_k, delta_k]
3. Nonlinear kinematic bicycle model (CG-reference with slip angle beta) is
   used for reference rollout.
4. Dynamics are linearized around the reference rollout (LTV form).
5. The resulting convex QP is solved with OSQP.
6. Output is the future state sequence only (no control sequence returned).

Cost function:
    J_total = sum_{k=1..N} (Cost_ref + Cost_LaneCenter + Cost_Repulsive + Cost_Control)

    Cost_ref:
      quadratic pull toward destination reference state.

    Cost_LaneCenter:
      soft lane-center tracking term when enabled.

    Cost_Repulsive:
      obstacle repulsive potential field, approximated in QP form.

    Cost_Control:
      control smoothness cost using acceleration-rate and steering-rate penalties.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

try:
    import osqp

    _OSQP_AVAILABLE = True
except Exception:  # pragma: no cover - import error path depends on environment
    osqp = None  # type: ignore[assignment]
    _OSQP_AVAILABLE = False



@dataclass
class MPCConstraintSpec:
    """Hard bounds and enabled safety constraints used by the QP."""

    min_velocity_mps: float
    max_velocity_mps: float
    min_acceleration_mps2: float
    max_acceleration_mps2: float
    max_jerk_mps3: float
    min_steer_rad: float
    max_steer_rad: float
    min_steer_rate_rps: float
    max_steer_rate_rps: float
    enforce_terminal_velocity_constraint: bool
    terminal_velocity_mps: float


@dataclass
class MPCComfortCostSpec:
    """Comfort cost weights for J_ctrl (reference tracking is in J_safe by user request)."""

    w_comf: float
    qx: float
    qy: float
    qv: float
    qpsi: float
    qa: float
    qdelta: float


@dataclass
class MPCSafetyCostSpec:
    """Safety cost top-level weight for reference-state tracking."""

    w_safe: float


@dataclass
class MPCRepulsivePotentialSpec:
    """
    Super-ellipsoid repulsive potential configuration.

    For each obstacle and stage:
        J_obs = w_s * exp(-k_s * (r_s - s_s))
              + w_c * exp(-k_c * (r_c - s_c))

    where:
        r_s : normalized distance to the larger safe zone
        r_c : normalized distance to the tighter collision zone
        w_s : safe-zone weight
        w_c : collision-zone weight
        k_s : safe-zone exponential gain
        k_c : collision-zone exponential gain
        s_s : safe-zone distance shift
        s_c : collision-zone distance shift
    """

    enabled: bool
    w_safe_zone: float
    w_collision_zone: float
    safe_exponential_gain: float
    safe_distance_shift: float
    collision_exponential_gain: float
    collision_distance_shift: float
    max_braking_deceleration_mps2: float
    comfort_deceleration_mps2: float
    reaction_time_s: float
    static_longitudinal_buffer_m: float
    static_lateral_buffer_m: float
    shape_exponent: float
    min_lateral_approach_speed_mps: float
    max_longitudinal_zone_length_m: float
    limit_lateral_zone_to_lane_width: bool
    max_lateral_zone_lane_fraction: float
    project_hessian_psd: bool
    min_hessian_eig: float


@dataclass
class QPIndex:
    """
    Decision-variable indexing helper.

    Variable layout:
        z = [X(0..N), U(0..N-1)]
    """

    nx: int
    nu: int
    horizon_steps: int

    @property
    def state_offset(self) -> int:
        return 0

    @property
    def control_offset(self) -> int:
        return (self.horizon_steps + 1) * self.nx

    @property
    def total_variables(self) -> int:
        return self.control_offset + self.horizon_steps * self.nu

    def state_index(self, k: int, i: int) -> int:
        return self.state_offset + k * self.nx + i

    def control_index(self, k: int, i: int) -> int:
        return self.control_offset + k * self.nu + i


class MPC:

    """
    Intent:
        Compute an optimal future trajectory of states [x,y,v,psi] over a finite
        horizon using an LTV-MPC QP solved by OSQP.

    Inputs to `plan_trajectory`:
        current_state:
            sequence[float], shape (4), [x, y, v, psi]
        destination_state:
            sequence[float], shape (2) or (4), [x, y, v, psi]
        object_snapshots:
            sequence of non-ego object snapshots. Each snapshot may include
            tracker predictions under `predicted_trajectory`.
        current_acceleration_mps2:
            float, previous applied acceleration command for jerk penalty/constraint.
        current_steering_rad:
            float, previous applied steering command for smoothness.

    Output:
        list[list[float]], shape (M, 4), future states for k=1..M where
        M<=N if the destination is reached within the horizon.
    """

    def __init__(
        self,
        mpc_cfg: Mapping[str, object],
        road_cfg: Mapping[str, object],
    ) -> None:

        self.horizon_s = float(mpc_cfg.get("horizon_s", 5.0))
        self.dt_s = float(mpc_cfg.get("plan_dt_s", 0.05))
        if self.horizon_s <= 0.0 or self.dt_s <= 0.0:
            raise ValueError("mpc.horizon_s and mpc.plan_dt_s must be > 0.")
        self.horizon_steps = max(1, int(round(self.horizon_s / self.dt_s)))
        self.horizon_s = float(self.horizon_steps * self.dt_s)

        self.trajectory_generation_frequency_hz = max(
            1e-3,
            float(mpc_cfg.get("trajectory_generation_frequency_hz", 2.0)),
        )
        self.trajectory_generation_period_s = 1.0 / self.trajectory_generation_frequency_hz

        self.destination_reached_threshold_m = max(
            0.05,
            float(mpc_cfg.get("destination_reached_threshold_m", 0.5)),
        )

        self.wheelbase_m = float(mpc_cfg.get("wheelbase_m", 2.7))
        if self.wheelbase_m <= 0.0:
            raise ValueError("mpc.wheelbase_m must be > 0.")
        # CG-reference model parameters. In this project we assume the CG is
        # centered between axles unless a scenario-specific axle split is added.
        self.l_r_m = max(1e-9, 0.5 * float(self.wheelbase_m))
        self.ego_length_m = max(0.0, float(mpc_cfg.get("ego_length_m", 0.0)))
        self.ego_width_m = max(0.0, float(mpc_cfg.get("ego_width_m", 0.0)))

        constraints_cfg = dict(mpc_cfg.get("constraints", {}))
        lane_count = max(1, int(road_cfg.get("lane_count", 3)))
        self.lane_count = int(lane_count)
        lane_width_m = float(road_cfg.get("lane_width_m", 4.0))
        self.lane_width_m = float(lane_width_m)
        self.constraints = MPCConstraintSpec(
            min_velocity_mps=float(constraints_cfg.get("min_velocity_mps", 0.0)),
            max_velocity_mps=float(constraints_cfg.get("max_velocity_mps", 15.0)),
            min_acceleration_mps2=float(constraints_cfg.get("min_acceleration_mps2", -3.0)),
            max_acceleration_mps2=float(constraints_cfg.get("max_acceleration_mps2", 3.0)),
            max_jerk_mps3=abs(float(constraints_cfg.get("max_jerk_mps3", 10.0))),
            min_steer_rad=float(constraints_cfg.get("min_steer_rad", -0.3)),
            max_steer_rad=float(constraints_cfg.get("max_steer_rad", 0.3)),
            min_steer_rate_rps=min(
                float(constraints_cfg.get("min_steer_rate_rps", -0.02)),
                float(constraints_cfg.get("max_steer_rate_rps", 0.02)),
            ),
            max_steer_rate_rps=max(
                float(constraints_cfg.get("min_steer_rate_rps", -0.02)),
                float(constraints_cfg.get("max_steer_rate_rps", 0.02)),
            ),
            enforce_terminal_velocity_constraint=bool(constraints_cfg.get("enforce_terminal_velocity_constraint", True)),
            terminal_velocity_mps=float(constraints_cfg.get("terminal_velocity_mps", 0.0)),
        )
        final_stop_speed_cap_cfg = dict(mpc_cfg.get("final_stop_speed_cap", {}))
        self.final_stop_speed_cap_enabled = bool(final_stop_speed_cap_cfg.get("enabled", True))
        self.final_stop_speed_cap_activation_threshold_mps = max(
            0.0,
            float(final_stop_speed_cap_cfg.get("destination_speed_activation_threshold_mps", 0.05)),
        )
        self.final_stop_speed_cap_stop_buffer_m = max(
            0.0,
            float(final_stop_speed_cap_cfg.get("stop_buffer_m", 2.0)),
        )

        cost_cfg = dict(mpc_cfg.get("cost", {}))
        attractive_cfg = dict(cost_cfg.get("attractive", {}))
        control_cfg = dict(cost_cfg.get("control", {}))
        self.comfort_cost = MPCComfortCostSpec(
            # Control weight (legacy fallback: cost.w_comf).
            w_comf=max(0.0, float(control_cfg.get("w_control", cost_cfg.get("w_comf", 0.3)))),
            # Attractive term weights (legacy fallback: cost.q_*).
            qx=max(0.0, float(attractive_cfg.get("q_x", cost_cfg.get("q_x", 5.0)))),
            qy=max(0.0, float(attractive_cfg.get("q_y", cost_cfg.get("q_y", 8.0)))),
            qv=max(0.0, float(attractive_cfg.get("q_v", cost_cfg.get("q_v", 2.0)))),
            qpsi=max(0.0, float(attractive_cfg.get("q_psi", cost_cfg.get("q_psi", 4.0)))),
            # Control-rate weights (legacy fallback: cost.q_a, cost.q_delta).
            qa=max(0.0, float(control_cfg.get("q_a", cost_cfg.get("q_a", 2.0)))),
            qdelta=max(0.0, float(control_cfg.get("q_delta", cost_cfg.get("q_delta", 4.0)))),
        )
        self.safety_cost = MPCSafetyCostSpec(
            # Attractive weight (legacy fallback: cost.w_safe).
            w_safe=max(0.0, float(attractive_cfg.get("w_attractive", cost_cfg.get("w_safe", 0.7)))),
        )
        repulsive_cfg = dict(cost_cfg.get("repulsive_potential", {}))
        legacy_static_buffer_m = max(0.0, float(repulsive_cfg.get("static_buffer_m", 0.5)))
        self.repulsive_cost = MPCRepulsivePotentialSpec(
            enabled=bool(repulsive_cfg.get("enabled", True)),
            w_safe_zone=max(0.0, float(repulsive_cfg.get("w_safe_zone", 10.0))),
            w_collision_zone=max(0.0, float(repulsive_cfg.get("w_collision_zone", 100.0))),
            safe_exponential_gain=max(0.0, float(repulsive_cfg.get("safe_exponential_gain", 10.0))),
            safe_distance_shift=float(repulsive_cfg.get("safe_distance_shift", 1.5)),
            collision_exponential_gain=max(0.0, float(repulsive_cfg.get("collision_exponential_gain", 6.0))),
            collision_distance_shift=float(repulsive_cfg.get("collision_distance_shift", 1.5)),
            max_braking_deceleration_mps2=max(
                1e-6,
                float(
                    repulsive_cfg.get(
                        "max_braking_deceleration_mps2",
                        max(1e-6, abs(float(self.constraints.min_acceleration_mps2))),
                    )
                ),
            ),
            comfort_deceleration_mps2=max(
                1e-6,
                float(repulsive_cfg.get("comfort_deceleration_mps2", 2.0)),
            ),
            reaction_time_s=max(
                0.0,
                float(repulsive_cfg.get("reaction_time_s", 1.0)),
            ),
            static_longitudinal_buffer_m=max(
                0.0,
                float(repulsive_cfg.get("static_longitudinal_buffer_m", legacy_static_buffer_m)),
            ),
            static_lateral_buffer_m=max(
                0.0,
                float(repulsive_cfg.get("static_lateral_buffer_m", legacy_static_buffer_m)),
            ),
            shape_exponent=max(
                2.0,
                float(repulsive_cfg.get("shape_exponent", 4.0)),
            ),
            min_lateral_approach_speed_mps=max(
                1e-6,
                float(repulsive_cfg.get("min_lateral_approach_speed_mps", 0.1)),
            ),
            max_longitudinal_zone_length_m=max(
                1e-6,
                float(repulsive_cfg.get("max_longitudinal_zone_length_m", 10.0)),
            ),
            limit_lateral_zone_to_lane_width=bool(
                repulsive_cfg.get("limit_lateral_zone_to_lane_width", True)
            ),
            max_lateral_zone_lane_fraction=max(
                1e-3,
                float(repulsive_cfg.get("max_lateral_zone_lane_fraction", 1.0)),
            ),
            project_hessian_psd=bool(repulsive_cfg.get("project_hessian_psd", repulsive_cfg.get("taylor_project_hessian_psd", True))),
            min_hessian_eig=max(
                0.0,
                float(repulsive_cfg.get("min_hessian_eig", repulsive_cfg.get("taylor_min_hessian_eig", 1e-9))),
            ),
        )

        lane_center_cfg = dict(cost_cfg.get("lane_center_follow", {}))
        self.lane_center_follow_enabled = bool(lane_center_cfg.get("enabled", False))
        self.lane_center_follow_weight = max(0.0, float(lane_center_cfg.get("w0", lane_center_cfg.get("w_lane_center", 0.0))))
        self.lane_center_follow_qpsi = max(
            0.0,
            float(lane_center_cfg.get("q_psi", lane_center_cfg.get("heading_weight", 0.0))),
        )

        self.reference_cfg = dict(mpc_cfg.get("reference_rollout", {}))
        self.reference_heading_gain = float(self.reference_cfg.get("heading_gain", 1.6))
        self.reference_speed_gain = float(self.reference_cfg.get("speed_gain", 1.2))
        self.reference_prefer_lane_center_path = bool(self.reference_cfg.get("prefer_lane_center_path", True))
        self.reference_path_los_heading_blend = min(
            1.0,
            max(0.0, float(self.reference_cfg.get("path_los_heading_blend", 0.35))),
        )
        self.reference_use_previous_solution_seed = bool(
            self.reference_cfg.get("use_previous_solution_seed", True)
        )
        self.reference_previous_solution_search_steps = max(
            0,
            int(self.reference_cfg.get("previous_solution_search_steps", 15)),
        )
        self.reference_previous_solution_max_position_error_m = max(
            0.0,
            float(self.reference_cfg.get("previous_solution_max_position_error_m", 3.0)),
        )
        self.reference_previous_solution_max_heading_error_rad = max(
            0.0,
            float(self.reference_cfg.get("previous_solution_max_heading_error_rad", 0.75)),
        )
        self.reference_previous_solution_max_speed_error_mps = max(
            0.0,
            float(self.reference_cfg.get("previous_solution_max_speed_error_mps", 4.0)),
        )
        self.reference_sequential_iterations = max(
            1,
            int(self.reference_cfg.get("sequential_linearization_iterations", 2)),
        )
        self.reference_obstacle_aware_speed_enabled = bool(
            self.reference_cfg.get("obstacle_aware_speed_enabled", True)
        )
        self.reference_obstacle_check_horizon_s = max(
            float(self.dt_s),
            float(self.reference_cfg.get("obstacle_check_horizon_s", 3.0)),
        )
        self.reference_lead_obstacle_trigger_distance_m = max(
            0.0,
            float(self.reference_cfg.get("lead_obstacle_trigger_distance_m", 18.0)),
        )
        self.reference_lead_obstacle_lateral_margin_m = max(
            0.0,
            float(self.reference_cfg.get("lead_obstacle_lateral_margin_m", 1.2)),
        )
        self.reference_lead_obstacle_stop_buffer_m = max(
            0.0,
            float(self.reference_cfg.get("lead_obstacle_stop_buffer_m", 6.0)),
        )
        self.reference_lead_obstacle_braking_decel_mps2 = max(
            1e-6,
            float(
                self.reference_cfg.get(
                    "lead_obstacle_braking_deceleration_mps2",
                    max(1e-6, abs(float(self.constraints.min_acceleration_mps2))),
                )
            ),
        )

        self.solver_cfg = dict(mpc_cfg.get("solver", {}))
        self.qp_max_iter = int(self.solver_cfg.get("max_iter", 4000))
        self.qp_eps_abs = float(self.solver_cfg.get("eps_abs", 1e-3))
        self.qp_eps_rel = float(self.solver_cfg.get("eps_rel", 1e-3))
        self.qp_polish = bool(self.solver_cfg.get("polish", True))
        if not _OSQP_AVAILABLE:
            raise ImportError("OSQP is required for the MPC QP solver. Install `osqp` in the environment.")

        self.nx = 4
        self.nu = 2
        self._last_status = "not_solved"
        self._last_solve_time_ms = 0.0
        self._last_active_max_velocity_mps = float(self.constraints.max_velocity_mps)
        self._last_cost_terms: Dict[str, float] = {
            "Cost_ref": 0.0,
            "Cost_LaneCenter": 0.0,
            "Cost_Repulsive_Safe": 0.0,
            "Cost_Repulsive_Collision": 0.0,
            "Cost_Repulsive": 0.0,
            "Cost_Control": 0.0,
        }
        self._last_x_solution: np.ndarray | None = None
        self._last_u_solution: np.ndarray | None = None
        self._previous_x_solution: np.ndarray | None = None
        self._previous_u_solution: np.ndarray | None = None

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    @staticmethod
    def _wrap_angle(angle_rad: float) -> float:
        return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi

    def _align_angle_near(self, angle_rad: float, around_rad: float) -> float:
        """
        Return the angle equivalent to `angle_rad` that is closest to `around_rad`.

        This keeps quadratic heading tracking terms consistent near the wrap
        boundary at +/-pi.
        """

        return float(around_rad) + float(self._wrap_angle(float(angle_rad) - float(around_rad)))

    @staticmethod
    def _blend_heading_angles(path_heading_rad: float, los_heading_rad: float, los_weight: float) -> float:
        los_weight = min(1.0, max(0.0, float(los_weight)))
        path_weight = 1.0 - los_weight
        blended_x = path_weight * math.cos(float(path_heading_rad)) + los_weight * math.cos(float(los_heading_rad))
        blended_y = path_weight * math.sin(float(path_heading_rad)) + los_weight * math.sin(float(los_heading_rad))
        if math.hypot(blended_x, blended_y) <= 1e-12:
            return float(path_heading_rad)
        return float(math.atan2(blended_y, blended_x))

    def _get_lane_center_stage_ref(
        self,
        lane_center_reference: Sequence[Mapping[str, object]] | None,
        stage_index: int,
        query_x_m: float | None = None,
        query_y_m: float | None = None,
    ) -> Tuple[float, float, float] | None:
        """
        Fetch lane-center reference tuple (x_ref, y_ref, heading_ref) for stage k.

        Behavior:
            - Default: returns stage-indexed lane-center sample.
            - If `query_x_m` and `query_y_m` are provided, returns the closest
              available lane-center waypoint to that query point. This is used
              in scenario4 so each MPC stage aligns with the nearest lane-center
              waypoint to the stage reference trajectory point.
        """

        if lane_center_reference is None or len(lane_center_reference) == 0:
            return None

        valid_samples: List[Mapping[str, object]] = []
        for sample in lane_center_reference:
            if not isinstance(sample, Mapping):
                continue
            if {"x_ref_m", "y_ref_m", "heading_rad"}.issubset(sample.keys()):
                valid_samples.append(sample)

        if len(valid_samples) == 0:
            return None

        # Closest-waypoint mode for stage k (scenario4):
        if query_x_m is not None and query_y_m is not None:
            qx = float(query_x_m)
            qy = float(query_y_m)
            best = min(
                valid_samples,
                key=lambda sample: math.hypot(
                    float(sample.get("x_ref_m", 0.0)) - qx,
                    float(sample.get("y_ref_m", 0.0)) - qy,
                ),
            )
            return (
                float(best.get("x_ref_m", 0.0)),
                float(best.get("y_ref_m", 0.0)),
                float(best.get("heading_rad", 0.0)),
            )

        idx = max(0, min(int(stage_index), len(valid_samples) - 1))
        sample = valid_samples[idx]
        return (
            float(sample.get("x_ref_m", 0.0)),
            float(sample.get("y_ref_m", 0.0)),
            float(sample.get("heading_rad", 0.0)),
        )
    @staticmethod
    def _lane_center_waypoint_position(waypoint: Mapping[str, object]) -> Tuple[float, float] | None:
        position_raw = waypoint.get("position")
        if not isinstance(position_raw, (list, tuple)) or len(position_raw) < 2:
            return None
        return float(position_raw[0]), float(position_raw[1])

    @staticmethod
    def _lane_center_waypoint_key(x_m: float, y_m: float) -> Tuple[float, float]:
        return (round(float(x_m), 3), round(float(y_m), 3))

    def _build_lane_center_reference(
        self,
        current_state: np.ndarray,
        destination_state: np.ndarray,
        lane_center_waypoints: Sequence[Mapping[str, object]] | None,
        destination_lane_id: int | None = None,
    ) -> List[Dict[str, float]]:
        """
        Build per-stage lane-center reference inside MPC.

        The integration layer provides road/lane waypoints. MPC owns the
        lane-keeping cost and the reference chain used by that cost.
        """

        if lane_center_waypoints is None or len(lane_center_waypoints) == 0:
            return []

        x_ego_m = float(current_state[0])
        y_ego_m = float(current_state[1])
        x_target_m = float(destination_state[0])
        y_target_m = float(destination_state[1])

        valid_waypoints: List[Dict[str, object]] = []
        waypoint_by_xy: Dict[Tuple[float, float], Dict[str, object]] = {}
        for waypoint in lane_center_waypoints:
            if not isinstance(waypoint, Mapping):
                continue
            position = self._lane_center_waypoint_position(waypoint)
            if position is None:
                continue
            waypoint_copy = dict(waypoint)
            waypoint_copy["heading_rad"] = float(waypoint_copy.get("heading_rad", 0.0))
            waypoint_key = self._lane_center_waypoint_key(position[0], position[1])
            waypoint_by_xy[waypoint_key] = waypoint_copy
            valid_waypoints.append(waypoint_copy)

        if len(valid_waypoints) == 0:
            return []

        normalized_destination_lane_id = (
            None
            if destination_lane_id is None
            else int(destination_lane_id)
        )
        if normalized_destination_lane_id is not None:
            target_lane_waypoints = [
                waypoint
                for waypoint in valid_waypoints
                if int(waypoint.get("lane_id", -1)) == int(normalized_destination_lane_id)
                and self._lane_center_waypoint_position(waypoint) is not None
            ]
        else:
            target_lane_waypoints = []

        if len(target_lane_waypoints) == 0:
            target_lane_waypoint = min(
                valid_waypoints,
                key=lambda waypoint: math.hypot(
                    float(self._lane_center_waypoint_position(waypoint)[0]) - x_target_m,
                    float(self._lane_center_waypoint_position(waypoint)[1]) - y_target_m,
                ) if self._lane_center_waypoint_position(waypoint) is not None else 1.0e9,
            )
            target_lane_id = int(target_lane_waypoint.get("lane_id", -1))
            target_lane_waypoints = [
                waypoint
                for waypoint in valid_waypoints
                if int(waypoint.get("lane_id", -1)) == target_lane_id
                and self._lane_center_waypoint_position(waypoint) is not None
            ]
            if len(target_lane_waypoints) == 0:
                target_lane_waypoints = valid_waypoints
        else:
            target_lane_id = int(normalized_destination_lane_id)

        current_waypoint = min(
            target_lane_waypoints,
            key=lambda waypoint: math.hypot(
                float(self._lane_center_waypoint_position(waypoint)[0]) - x_ego_m,
                float(self._lane_center_waypoint_position(waypoint)[1]) - y_ego_m,
            ) if self._lane_center_waypoint_position(waypoint) is not None else 1.0e9,
        )

        stage_reference: List[Dict[str, float]] = []
        visited_keys: set[Tuple[float, float]] = set()
        for _k in range(self.horizon_steps + 1):
            current_position = self._lane_center_waypoint_position(current_waypoint)
            if current_position is None:
                break
            stage_reference.append(
                {
                    "x_ref_m": float(current_position[0]),
                    "y_ref_m": float(current_position[1]),
                    "heading_rad": float(current_waypoint.get("heading_rad", 0.0)),
                    "lane_id": int(target_lane_id),
                }
            )

            next_position_raw = current_waypoint.get("next", None)
            if not isinstance(next_position_raw, (list, tuple)) or len(next_position_raw) < 2:
                break
            next_key = self._lane_center_waypoint_key(float(next_position_raw[0]), float(next_position_raw[1]))
            if next_key in visited_keys:
                break
            visited_keys.add(next_key)
            next_waypoint = waypoint_by_xy.get(next_key)
            if next_waypoint is None:
                break
            current_waypoint = next_waypoint

        if len(stage_reference) == 0:
            return []
        while len(stage_reference) < self.horizon_steps + 1:
            stage_reference.append(dict(stage_reference[-1]))
        return stage_reference

    def _get_object_state_at_stage(
        self,
        object_snapshot: Mapping[str, object],
        stage_index: int,
        dt_s: float,
    ) -> List[float]:
        """
        Return obstacle state [x,y,v,psi] at prediction stage using tracker
        prediction if available, otherwise constant-velocity propagation.
        """

        predicted = object_snapshot.get("predicted_trajectory", object_snapshot.get("future_trajectory", []))
        if isinstance(predicted, Sequence) and 0 <= int(stage_index) < len(predicted):
            state = predicted[int(stage_index)]
            if isinstance(state, Sequence) and len(state) >= 4:
                return [float(state[0]), float(state[1]), float(state[2]), float(state[3])]

        x = float(object_snapshot.get("x", 0.0))
        y = float(object_snapshot.get("y", 0.0))
        v = float(object_snapshot.get("v", 0.0))
        psi = float(object_snapshot.get("psi", 0.0))
        t = float(max(0, int(stage_index) + 1)) * float(dt_s)
        return [
            float(x + v * math.cos(psi) * t),
            float(y + v * math.sin(psi) * t),
            float(v),
            float(psi),
        ]

    def _build_shifted_previous_solution_seed(self, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray] | None:
        """
        Reuse the previous solved MPC trajectory as the next linearization seed.

        The current ego state is matched to a nearby stage of the previous
        solution, then the remainder of that solution is shifted forward.
        """

        if not bool(self.reference_use_previous_solution_seed):
            return None
        if self._previous_x_solution is None or self._previous_u_solution is None:
            return None

        prev_x = self._previous_x_solution
        prev_u = self._previous_u_solution
        if prev_x.shape != (self.horizon_steps + 1, self.nx):
            return None
        if prev_u.shape != (self.horizon_steps, self.nu):
            return None

        search_limit = min(
            int(self.reference_previous_solution_search_steps),
            prev_x.shape[0] - 1,
        )
        best_idx: int | None = None
        best_score = float("inf")

        for idx in range(search_limit + 1):
            position_error_m = math.hypot(
                float(prev_x[idx, 0]) - float(x0[0]),
                float(prev_x[idx, 1]) - float(x0[1]),
            )
            heading_error_rad = abs(self._wrap_angle(float(prev_x[idx, 3]) - float(x0[3])))
            speed_error_mps = abs(float(prev_x[idx, 2]) - float(x0[2]))
            if position_error_m > float(self.reference_previous_solution_max_position_error_m):
                continue
            if heading_error_rad > float(self.reference_previous_solution_max_heading_error_rad):
                continue
            if speed_error_mps > float(self.reference_previous_solution_max_speed_error_mps):
                continue

            score = position_error_m + 0.5 * heading_error_rad + 0.25 * speed_error_mps
            if score < best_score:
                best_score = float(score)
                best_idx = int(idx)

        if best_idx is None:
            return None

        x_seed = np.zeros_like(prev_x)
        u_seed = np.zeros_like(prev_u)
        x_seed[0] = np.asarray(x0, dtype=float)

        for k in range(1, self.horizon_steps + 1):
            src_idx = min(best_idx + k, prev_x.shape[0] - 1)
            x_seed[k] = np.asarray(prev_x[src_idx], dtype=float)
            x_seed[k, 3] = self._wrap_angle(float(x_seed[k, 3]))

        for k in range(self.horizon_steps):
            src_idx = min(best_idx + k, prev_u.shape[0] - 1)
            u_seed[k] = np.asarray(prev_u[src_idx], dtype=float)

        return x_seed, u_seed

    def _compute_reference_rollout_speed_limit(
        self,
        stage_x_m: float,
        stage_y_m: float,
        stage_heading_rad: float,
        stage_index: int,
        base_speed_mps: float,
        object_snapshots: Sequence[Mapping[str, object]],
    ) -> float:
        """
        Obstacle-aware speed heuristic for rollout generation.

        The rollout stays generic: if a lead obstacle is predicted ahead in the
        same corridor, cap the rollout speed to a simple stopping-speed bound.
        """

        base_speed_mps = max(0.0, float(base_speed_mps))
        if not bool(self.reference_obstacle_aware_speed_enabled):
            return float(base_speed_mps)
        if base_speed_mps <= 0.0:
            return 0.0
        if len(object_snapshots) == 0:
            return float(base_speed_mps)
        if float(stage_index) * float(self.dt_s) > float(self.reference_obstacle_check_horizon_s):
            return float(base_speed_mps)

        cos_heading = math.cos(float(stage_heading_rad))
        sin_heading = math.sin(float(stage_heading_rad))
        best_gap_m = float("inf")

        for object_snapshot in object_snapshots:
            obj_state = self._get_object_state_at_stage(
                object_snapshot=object_snapshot,
                stage_index=int(stage_index),
                dt_s=float(self.dt_s),
            )
            obj_x_m = float(obj_state[0])
            obj_y_m = float(obj_state[1])
            dx_m = obj_x_m - float(stage_x_m)
            dy_m = obj_y_m - float(stage_y_m)
            along_track_m = dx_m * cos_heading + dy_m * sin_heading
            cross_track_m = -dx_m * sin_heading + dy_m * cos_heading

            if along_track_m < 0.0:
                continue
            if along_track_m > float(self.reference_lead_obstacle_trigger_distance_m):
                continue

            obj_half_width_m = 0.5 * float(object_snapshot.get("width_m", 2.0))
            lateral_limit_m = obj_half_width_m + float(self.reference_lead_obstacle_lateral_margin_m)
            if abs(cross_track_m) > lateral_limit_m:
                continue

            best_gap_m = min(float(best_gap_m), float(along_track_m))

        if not math.isfinite(best_gap_m):
            return float(base_speed_mps)

        remaining_gap_m = max(0.0, float(best_gap_m) - float(self.reference_lead_obstacle_stop_buffer_m))
        stop_speed_limit_mps = math.sqrt(
            max(
                0.0,
                2.0 * float(self.reference_lead_obstacle_braking_decel_mps2) * remaining_gap_m,
            )
        )
        return float(min(base_speed_mps, stop_speed_limit_mps))

    def _superellipsoid_obstacle_cost_components(
        self,
        ego_state: Sequence[float],
        obstacle_state: Sequence[float],
        obstacle_length_m: float,
        obstacle_width_m: float,
    ) -> Tuple[float, float]:
        """
        Super-ellipsoid obstacle cost components from `super_ellipsoid.py`.

        Cost:
            J_obs = w_s * exp(-k_s * (r_s - s_s))
                  + w_c * exp(-k_c * (r_c - s_c))

        where:
            r_s = normalized distance to the safe zone
            r_c = normalized distance to the collision zone
        """

        geometry = self._superellipsoid_zone_geometry(
            ego_state=ego_state,
            obstacle_state=obstacle_state,
            obstacle_length_m=obstacle_length_m,
            obstacle_width_m=obstacle_width_m,
        )
        rc = float(geometry["rc"])
        rs = float(geometry["rs"])
        ego_x_m = float(ego_state[0]) if len(ego_state) >= 1 else 0.0
        ego_y_m = float(ego_state[1]) if len(ego_state) >= 2 else 0.0
        ego_v_mps = max(0.0, float(ego_state[2]) if len(ego_state) >= 3 else 0.0)
        ego_psi_rad = float(ego_state[3]) if len(ego_state) >= 4 else 0.0

        obs_x_m = float(obstacle_state[0]) if len(obstacle_state) >= 1 else 0.0
        obs_y_m = float(obstacle_state[1]) if len(obstacle_state) >= 2 else 0.0
        obs_v_mps = max(0.0, float(obstacle_state[2]) if len(obstacle_state) >= 3 else 0.0)
        obs_psi_rad = float(obstacle_state[3]) if len(obstacle_state) >= 4 else 0.0

        cost_safe = float(self.repulsive_cost.w_safe_zone) * math.exp(
            -float(self.repulsive_cost.safe_exponential_gain)
            * (float(rs) - float(self.repulsive_cost.safe_distance_shift))
        )
        cost_collision = float(self.repulsive_cost.w_collision_zone) * math.exp(
            -float(self.repulsive_cost.collision_exponential_gain)
            * (float(rc) - float(self.repulsive_cost.collision_distance_shift))
        )
        return float(cost_safe), float(cost_collision)

    def _superellipsoid_zone_geometry(
        self,
        ego_state: Sequence[float],
        obstacle_state: Sequence[float],
        obstacle_length_m: float,
        obstacle_width_m: float,
    ) -> Dict[str, float]:
        """
        Return the active super-ellipsoid zone geometry used by the live cost.

        The returned values are aligned to the obstacle frame.
        """

        ego_x_m = float(ego_state[0]) if len(ego_state) >= 1 else 0.0
        ego_y_m = float(ego_state[1]) if len(ego_state) >= 2 else 0.0
        ego_v_mps = max(0.0, float(ego_state[2]) if len(ego_state) >= 3 else 0.0)
        ego_psi_rad = float(ego_state[3]) if len(ego_state) >= 4 else 0.0

        obs_x_m = float(obstacle_state[0]) if len(obstacle_state) >= 1 else 0.0
        obs_y_m = float(obstacle_state[1]) if len(obstacle_state) >= 2 else 0.0
        obs_v_mps = max(0.0, float(obstacle_state[2]) if len(obstacle_state) >= 3 else 0.0)
        obs_psi_rad = float(obstacle_state[3]) if len(obstacle_state) >= 4 else 0.0

        heading_diff_rad = self._wrap_angle(float(ego_psi_rad) - float(obs_psi_rad))
        cos_obs = math.cos(float(obs_psi_rad))
        sin_obs = math.sin(float(obs_psi_rad))
        dx_m = float(ego_x_m) - float(obs_x_m)
        dy_m = float(ego_y_m) - float(obs_y_m)
        x_local_m = dx_m * cos_obs + dy_m * sin_obs
        y_local_m = -dx_m * sin_obs + dy_m * cos_obs

        v_approach_longitudinal_mps = float(ego_v_mps) * math.cos(float(heading_diff_rad)) - float(obs_v_mps)
        v_approach_lateral_mps = float(ego_v_mps) * math.sin(float(heading_diff_rad))
        delta_u_mps = max(0.0, float(v_approach_longitudinal_mps))
        delta_v_mps = max(
            float(self.repulsive_cost.min_lateral_approach_speed_mps),
            abs(float(v_approach_lateral_mps)),
        )

        projected_ego_length_m = abs(float(self.ego_length_m) * math.cos(float(heading_diff_rad)))
        projected_ego_length_m += abs(float(self.ego_width_m) * math.sin(float(heading_diff_rad)))
        projected_ego_width_m = abs(float(self.ego_length_m) * math.sin(float(heading_diff_rad)))
        projected_ego_width_m += abs(float(self.ego_width_m) * math.cos(float(heading_diff_rad)))

        obstacle_length_m = max(1e-6, float(obstacle_length_m))
        obstacle_width_m = max(1e-6, float(obstacle_width_m))
        x0_m = 0.5 * (projected_ego_length_m + obstacle_length_m)
        x0_m += float(self.repulsive_cost.static_longitudinal_buffer_m)
        y0_m = 0.5 * (projected_ego_width_m + obstacle_width_m)
        y0_m += float(self.repulsive_cost.static_lateral_buffer_m)

        a_max_mps2 = max(1e-6, float(self.repulsive_cost.max_braking_deceleration_mps2))
        a_comfort_mps2 = max(1e-6, float(self.repulsive_cost.comfort_deceleration_mps2))
        reaction_time_s = max(0.0, float(self.repulsive_cost.reaction_time_s))

        xc_m = x0_m + (delta_u_mps * delta_u_mps) / (2.0 * a_max_mps2)
        yc_m = y0_m + (delta_v_mps * delta_v_mps) / (2.0 * a_max_mps2)
        xs_m = x0_m + delta_u_mps * reaction_time_s + (delta_u_mps * delta_u_mps) / (2.0 * a_comfort_mps2)
        ys_m = y0_m + delta_v_mps * reaction_time_s + (delta_v_mps * delta_v_mps) / (2.0 * a_comfort_mps2)

        longitudinal_half_limit_m = 0.5 * float(self.repulsive_cost.max_longitudinal_zone_length_m)
        longitudinal_half_limit_m = max(1e-6, float(longitudinal_half_limit_m))
        xc_m = min(float(xc_m), longitudinal_half_limit_m)
        xs_m = min(float(xs_m), longitudinal_half_limit_m)

        if bool(self.repulsive_cost.limit_lateral_zone_to_lane_width):
            lane_width_limit_m = float(self.lane_width_m) * float(self.repulsive_cost.max_lateral_zone_lane_fraction)
            full_width_limit_m = max(float(lane_width_limit_m), float(obstacle_width_m))
            lateral_half_limit_m = max(1e-6, 0.5 * float(full_width_limit_m))
            yc_m = min(float(yc_m), lateral_half_limit_m)
            ys_m = min(float(ys_m), lateral_half_limit_m)

        n = max(2.0, float(self.repulsive_cost.shape_exponent))
        rc = (abs(float(x_local_m) / max(1e-6, float(xc_m))) ** n + abs(float(y_local_m) / max(1e-6, float(yc_m))) ** n) ** (1.0 / n)
        rs = (abs(float(x_local_m) / max(1e-6, float(xs_m))) ** n + abs(float(y_local_m) / max(1e-6, float(ys_m))) ** n) ** (1.0 / n)

        return {
            "x_local_m": float(x_local_m),
            "y_local_m": float(y_local_m),
            "x0_m": float(x0_m),
            "y0_m": float(y0_m),
            "xc_m": float(xc_m),
            "yc_m": float(yc_m),
            "xs_m": float(xs_m),
            "ys_m": float(ys_m),
            "shape_exponent": float(n),
            "obstacle_x_m": float(obs_x_m),
            "obstacle_y_m": float(obs_y_m),
            "obstacle_psi_rad": float(obs_psi_rad),
            "rc": float(rc),
            "rs": float(rs),
        }

    def _superellipsoid_obstacle_cost(
        self,
        ego_state: Sequence[float],
        obstacle_state: Sequence[float],
        obstacle_length_m: float,
        obstacle_width_m: float,
    ) -> float:
        cost_safe, cost_collision = self._superellipsoid_obstacle_cost_components(
            ego_state=ego_state,
            obstacle_state=obstacle_state,
            obstacle_length_m=obstacle_length_m,
            obstacle_width_m=obstacle_width_m,
        )
        return float(cost_safe + cost_collision)

    def _superellipsoid_cost_taylor_terms(
        self,
        ego_state_ref: Sequence[float],
        obstacle_state: Sequence[float],
        obstacle_length_m: float,
        obstacle_width_m: float,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Numerical Taylor ingredients of the super-ellipsoid obstacle cost with
        respect to ego state [x, y, v, psi] at one stage reference point.
        """

        state_ref = np.array(
            [
                float(ego_state_ref[0]) if len(ego_state_ref) >= 1 else 0.0,
                float(ego_state_ref[1]) if len(ego_state_ref) >= 2 else 0.0,
                float(ego_state_ref[2]) if len(ego_state_ref) >= 3 else 0.0,
                self._wrap_angle(float(ego_state_ref[3]) if len(ego_state_ref) >= 4 else 0.0),
            ],
            dtype=float,
        )
        step_sizes = np.array([0.05, 0.05, 0.05, 0.01], dtype=float)

        def evaluate(query_state: np.ndarray) -> float:
            state_eval = np.asarray(query_state, dtype=float).copy()
            state_eval[3] = self._wrap_angle(float(state_eval[3]))
            return self._superellipsoid_obstacle_cost(
                ego_state=state_eval,
                obstacle_state=obstacle_state,
                obstacle_length_m=float(obstacle_length_m),
                obstacle_width_m=float(obstacle_width_m),
            )

        p0 = float(evaluate(state_ref))
        gradient = np.zeros(4, dtype=float)
        hessian = np.zeros((4, 4), dtype=float)

        for idx in range(4):
            delta = np.zeros(4, dtype=float)
            delta[idx] = float(step_sizes[idx])
            f_plus = float(evaluate(state_ref + delta))
            f_minus = float(evaluate(state_ref - delta))
            gradient[idx] = (f_plus - f_minus) / (2.0 * float(step_sizes[idx]))
            hessian[idx, idx] = (f_plus - 2.0 * p0 + f_minus) / (float(step_sizes[idx]) ** 2)

        for row in range(4):
            for col in range(row + 1, 4):
                delta_row = np.zeros(4, dtype=float)
                delta_col = np.zeros(4, dtype=float)
                delta_row[row] = float(step_sizes[row])
                delta_col[col] = float(step_sizes[col])
                f_pp = float(evaluate(state_ref + delta_row + delta_col))
                f_pm = float(evaluate(state_ref + delta_row - delta_col))
                f_mp = float(evaluate(state_ref - delta_row + delta_col))
                f_mm = float(evaluate(state_ref - delta_row - delta_col))
                mixed = (f_pp - f_pm - f_mp + f_mm) / (4.0 * float(step_sizes[row]) * float(step_sizes[col]))
                hessian[row, col] = mixed
                hessian[col, row] = mixed

        hessian = 0.5 * (hessian + hessian.T)
        return float(p0), np.asarray(gradient, dtype=float), np.asarray(hessian, dtype=float)

    def _project_symmetric_hessian_to_psd(self, hessian: np.ndarray) -> np.ndarray:
        """
        Project a symmetric Hessian to PSD by clamping eigenvalues.

        This keeps the local quadratic obstacle approximation convex for OSQP.
        """

        H = np.asarray(hessian, dtype=float)
        H = 0.5 * (H + H.T)
        eigvals, eigvecs = np.linalg.eigh(H)
        eig_floor = float(self.repulsive_cost.min_hessian_eig)
        eigvals_clamped = np.maximum(eigvals, eig_floor)
        return np.asarray(eigvecs @ np.diag(eigvals_clamped) @ eigvecs.T, dtype=float)

    def get_runtime_status(self) -> Dict[str, object]:
        return {
            "solver_status": str(self._last_status),
            "solve_time_ms": float(self._last_solve_time_ms),
            "horizon_steps": int(self.horizon_steps),
            "plan_dt_s": float(self.dt_s),
            "horizon_s": float(self.horizon_s),
            "trajectory_generation_frequency_hz": float(self.trajectory_generation_frequency_hz),
            "active_max_velocity_mps": float(self._last_active_max_velocity_mps),
            "compute_backend": "cpu_osqp_qp",
        }

    def get_last_cost_terms(self) -> Dict[str, float]:
        """Return the most recently evaluated MPC cost terms."""

        return dict(self._last_cost_terms)

    def get_last_control_sequence(self, max_steps: int | None = None) -> List[Dict[str, float]]:
        """
        Return the most recent MPC-planned control sequence.

        This exposes the optimizer/fallback plan controls used for the latest
        candidate trajectory generation. It does not return the applied ego
        controls from the downstream PID tracker.
        """

        if self._last_u_solution is None:
            return []

        controls = np.asarray(self._last_u_solution, dtype=float)
        step_limit = int(controls.shape[0])
        if max_steps is not None:
            step_limit = max(0, min(step_limit, int(max_steps)))

        output: List[Dict[str, float]] = []
        for step_idx in range(step_limit):
            output.append(
                {
                    "step_index": int(step_idx),
                    "time_from_plan_start_s": float(step_idx) * float(self.dt_s),
                    "acceleration_mps2": float(controls[step_idx, 0]),
                    "steering_angle_rad": float(controls[step_idx, 1]),
                }
            )
        return output

    def _normalize_destination_state(self, destination_state: Sequence[float]) -> np.ndarray:
        """
        Intent:
            Normalize destination input to shape (4,) = [x,y,v,psi].

        PDF rule implemented:
            If destination provides only [x,y], default to v_ref = 0 and
            psi_ref = 0.
        """

        if len(destination_state) >= 4:
            x_ref = float(destination_state[0])
            y_ref = float(destination_state[1])
            v_ref = float(destination_state[2])
            psi_ref = float(destination_state[3])
            return np.array([x_ref, y_ref, v_ref, self._wrap_angle(psi_ref)], dtype=float)

        if len(destination_state) >= 2:
            x_ref = float(destination_state[0])
            y_ref = float(destination_state[1])
            v_ref = 0.0
            psi_ref = 0.0
            return np.array([x_ref, y_ref, v_ref, psi_ref], dtype=float)

        raise ValueError("destination_state must have at least [x, y].")

    def _compute_active_speed_upper_bound_mps(
        self,
        current_state: Sequence[float],
        destination_state: Sequence[float],
    ) -> float:
        """
        Compute the active speed upper bound for this MPC replan.

        When the active destination is a stop goal (destination speed near zero),
        use a braking-distance cap:
            v_cap(d) = min(v_max, sqrt(2 * a_brake * max(d - stop_buffer, 0)))

        where a_brake is taken from abs(min_acceleration_mps2).
        """

        base_max_velocity_mps = float(self.constraints.max_velocity_mps)
        if not bool(self.final_stop_speed_cap_enabled):
            return float(base_max_velocity_mps)
        if len(destination_state) < 3:
            return float(base_max_velocity_mps)

        destination_speed_mps = abs(float(destination_state[2]))
        if destination_speed_mps > float(self.final_stop_speed_cap_activation_threshold_mps):
            return float(base_max_velocity_mps)

        current_x_m = float(current_state[0]) if len(current_state) >= 1 else 0.0
        current_y_m = float(current_state[1]) if len(current_state) >= 2 else 0.0
        destination_x_m = float(destination_state[0]) if len(destination_state) >= 1 else current_x_m
        destination_y_m = float(destination_state[1]) if len(destination_state) >= 2 else current_y_m
        distance_to_destination_m = math.hypot(destination_x_m - current_x_m, destination_y_m - current_y_m)
        remaining_stop_distance_m = max(
            0.0,
            float(distance_to_destination_m) - float(self.final_stop_speed_cap_stop_buffer_m),
        )
        braking_deceleration_mps2 = max(1e-6, abs(float(self.constraints.min_acceleration_mps2)))
        speed_cap_mps = math.sqrt(2.0 * braking_deceleration_mps2 * remaining_stop_distance_m)
        return float(min(base_max_velocity_mps, speed_cap_mps))

    def _minimum_reachable_speed_profile_mps(
        self,
        current_speed_mps: float,
        current_acceleration_mps2: float,
    ) -> List[float]:
        """
        Compute the minimum reachable speed profile under bounded jerk and
        acceleration when applying the strongest allowed braking sequence.
        """

        profile = [max(float(self.constraints.min_velocity_mps), float(current_speed_mps))]
        min_velocity_mps = float(self.constraints.min_velocity_mps)
        min_acceleration_mps2 = float(self.constraints.min_acceleration_mps2)
        jerk_delta_limit = float(self.constraints.max_jerk_mps3) * float(self.dt_s)

        v_k_mps = float(profile[0])
        a_prev_mps2 = float(current_acceleration_mps2)
        for _ in range(self.horizon_steps):
            a_k_mps2 = max(min_acceleration_mps2, a_prev_mps2 - jerk_delta_limit)
            v_k_mps = max(min_velocity_mps, float(v_k_mps) + float(self.dt_s) * float(a_k_mps2))
            profile.append(float(v_k_mps))
            a_prev_mps2 = float(a_k_mps2)
        return profile

    def _future_speed_upper_bound_mps(
        self,
        active_speed_upper_bound_mps: float,
        future_state_index: int,
        reachable_speed_floor_profile_mps: Sequence[float] | None = None,
    ) -> float:
        """
        Compute a feasible per-stage future speed upper bound.

        If the current speed is already above the active cap, the QP must still
        remain feasible under bounded braking. This upper bound therefore follows
        the fastest physically achievable deceleration envelope down toward the
        active cap.
        """

        future_state_index = max(1, int(future_state_index))
        base_max_velocity_mps = float(self.constraints.max_velocity_mps)
        min_velocity_mps = float(self.constraints.min_velocity_mps)
        active_speed_upper_bound_mps = min(float(base_max_velocity_mps), max(min_velocity_mps, float(active_speed_upper_bound_mps)))
        if reachable_speed_floor_profile_mps is None or len(reachable_speed_floor_profile_mps) == 0:
            reachable_speed_floor_mps = float(min_velocity_mps)
        else:
            profile_idx = min(int(future_state_index), len(reachable_speed_floor_profile_mps) - 1)
            reachable_speed_floor_mps = max(
                min_velocity_mps,
                float(reachable_speed_floor_profile_mps[profile_idx]),
            )
        return float(min(base_max_velocity_mps, max(active_speed_upper_bound_mps, reachable_speed_floor_mps)))

    def _cg_slip_angle_beta(self, delta_rad: float) -> float:
        """
        Intent:
            Compute the CG-reference kinematic bicycle slip angle beta.

        Equation:
            beta = atan((l_r / L) * tan(delta))
        """

        k_ratio = float(self.l_r_m / max(1e-9, self.wheelbase_m))
        return float(math.atan(k_ratio * math.tan(float(delta_rad))))

    def _cg_slip_angle_beta_derivative(self, delta_rad: float) -> float:
        """
        Intent:
            Compute d(beta)/d(delta) for linearizing the CG-reference model.
        """

        delta_rad = float(delta_rad)
        k_ratio = float(self.l_r_m / max(1e-9, self.wheelbase_m))
        cos_delta = math.cos(delta_rad)
        sec_delta_sq = 1.0 / max(1e-9, cos_delta * cos_delta)
        tan_delta = math.tan(delta_rad)
        return float((k_ratio * sec_delta_sq) / (1.0 + (k_ratio * tan_delta) ** 2))

    def _reference_rollout(
        self,
        x0: np.ndarray,
        x_ref_target: np.ndarray,
        lane_center_reference: Sequence[Mapping[str, object]] | None,
        object_snapshots: Sequence[Mapping[str, object]],
        speed_upper_bound_mps: float | None = None,
        reachable_speed_floor_profile_mps: Sequence[float] | None = None,
        seed_state_traj: np.ndarray | None = None,
        seed_control_traj: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Intent:
            Build a deterministic nonlinear rollout used as the linearization
            reference for the LTV-MPC QP.

        Logic:
            - Prefer stage-wise lane-center targets when available.
            - Blend path heading with line-of-sight heading to the next path point.
            - Reduce rollout speed when a lead obstacle is predicted ahead.
            - Propagate with the nonlinear CG-reference kinematic bicycle model.

        Output:
            x_ref_traj:
                np.ndarray, shape (N+1,4)
            u_ref_traj:
                np.ndarray, shape (N,2)
        """

        effective_speed_upper_bound_mps = float(self.constraints.max_velocity_mps)
        if speed_upper_bound_mps is not None:
            effective_speed_upper_bound_mps = min(
                float(effective_speed_upper_bound_mps),
                max(float(self.constraints.min_velocity_mps), float(speed_upper_bound_mps)),
            )
        if (
            seed_state_traj is not None
            and seed_control_traj is not None
            and seed_state_traj.shape == (self.horizon_steps + 1, self.nx)
            and seed_control_traj.shape == (self.horizon_steps, self.nu)
        ):
            x_seed = np.asarray(seed_state_traj, dtype=float).copy()
            u_seed = np.asarray(seed_control_traj, dtype=float).copy()
            x_seed[0] = np.asarray(x0, dtype=float)
            for k in range(1, self.horizon_steps + 1):
                stage_speed_upper_bound_mps = self._future_speed_upper_bound_mps(
                    active_speed_upper_bound_mps=float(effective_speed_upper_bound_mps),
                    future_state_index=int(k),
                    reachable_speed_floor_profile_mps=reachable_speed_floor_profile_mps,
                )
                x_seed[k, 2] = self._clamp(
                    float(x_seed[k, 2]),
                    float(self.constraints.min_velocity_mps),
                    float(stage_speed_upper_bound_mps),
                )
            x_seed[:, 3] = np.asarray([self._wrap_angle(float(angle)) for angle in x_seed[:, 3]], dtype=float)
            return x_seed, u_seed

        x_ref_traj = np.zeros((self.horizon_steps + 1, self.nx), dtype=float)
        u_ref_traj = np.zeros((self.horizon_steps, self.nu), dtype=float)
        x_ref_traj[0] = x0

        x_goal, y_goal, v_goal, psi_goal = [float(v) for v in x_ref_target]
        v_goal = self._clamp(v_goal, self.constraints.min_velocity_mps, effective_speed_upper_bound_mps)
        psi_goal = self._wrap_angle(psi_goal)

        for k in range(self.horizon_steps):
            x_m, y_m, v_mps, psi_rad = [float(v) for v in x_ref_traj[k]]
            target_x_m = float(x_goal)
            target_y_m = float(y_goal)
            path_heading_rad = float(psi_goal)

            if bool(self.reference_prefer_lane_center_path):
                stage_ref = self._get_lane_center_stage_ref(
                    lane_center_reference=lane_center_reference,
                    stage_index=int(k + 1),
                )
                if stage_ref is not None:
                    target_x_m = float(stage_ref[0])
                    target_y_m = float(stage_ref[1])
                    path_heading_rad = float(stage_ref[2])

            dx_target = target_x_m - x_m
            dy_target = target_y_m - y_m
            los_heading_rad = (
                math.atan2(dy_target, dx_target)
                if (abs(dx_target) + abs(dy_target)) > 1e-9
                else float(path_heading_rad)
            )
            desired_heading = self._blend_heading_angles(
                path_heading_rad=float(path_heading_rad),
                los_heading_rad=float(los_heading_rad),
                los_weight=float(self.reference_path_los_heading_blend),
            )
            heading_error = self._wrap_angle(desired_heading - psi_rad)
            delta_des = self._clamp(
                self.reference_heading_gain * heading_error,
                self.constraints.min_steer_rad,
                self.constraints.max_steer_rad,
            )

            stage_speed_target_mps = self._compute_reference_rollout_speed_limit(
                stage_x_m=x_m,
                stage_y_m=y_m,
                stage_heading_rad=float(path_heading_rad),
                stage_index=int(k),
                base_speed_mps=float(v_goal),
                object_snapshots=object_snapshots,
            )
            next_stage_speed_upper_bound_mps = self._future_speed_upper_bound_mps(
                active_speed_upper_bound_mps=float(effective_speed_upper_bound_mps),
                future_state_index=int(k + 1),
                reachable_speed_floor_profile_mps=reachable_speed_floor_profile_mps,
            )
            stage_speed_target_mps = min(float(stage_speed_target_mps), float(next_stage_speed_upper_bound_mps))
            accel_des = self.reference_speed_gain * (stage_speed_target_mps - v_mps)
            accel_des = self._clamp(
                accel_des,
                self.constraints.min_acceleration_mps2,
                self.constraints.max_acceleration_mps2,
            )

            u_ref_traj[k] = np.array([accel_des, delta_des], dtype=float)

            beta_rad = self._cg_slip_angle_beta(delta_des)
            x_next = x_m + self.dt_s * v_mps * math.cos(psi_rad + beta_rad)
            y_next = y_m + self.dt_s * v_mps * math.sin(psi_rad + beta_rad)
            v_next = self._clamp(
                v_mps + self.dt_s * accel_des,
                self.constraints.min_velocity_mps,
                float(next_stage_speed_upper_bound_mps),
            )
            psi_next = self._wrap_angle(psi_rad + self.dt_s * (v_mps / self.l_r_m) * math.sin(beta_rad))
            x_ref_traj[k + 1] = np.array([x_next, y_next, v_next, psi_next], dtype=float)

        return x_ref_traj, u_ref_traj

    def _linearize_dynamics(self, x_bar: np.ndarray, u_bar: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Intent:
            Linearize Euler-discretized CG-reference kinematic bicycle dynamics
            around one reference point (x_bar, u_bar).

        QP form used:
            X_{k+1} = A_k X_k + B_k U_k + c_k
        """

        x_m, y_m, v_mps, psi_rad = [float(v) for v in x_bar]
        a_mps2, delta_rad = [float(v) for v in u_bar]
        _ = x_m, y_m, a_mps2

        dt_s = float(self.dt_s)
        l_r_m = float(self.l_r_m)
        beta_rad = self._cg_slip_angle_beta(delta_rad)
        beta_delta = self._cg_slip_angle_beta_derivative(delta_rad)
        cos_sum = math.cos(psi_rad + beta_rad)
        sin_sum = math.sin(psi_rad + beta_rad)
        sin_beta = math.sin(beta_rad)
        cos_beta = math.cos(beta_rad)

        A_k = np.array(
            [
                [1.0, 0.0, dt_s * cos_sum, -dt_s * v_mps * sin_sum],
                [0.0, 1.0, dt_s * sin_sum, dt_s * v_mps * cos_sum],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, dt_s * sin_beta / l_r_m, 1.0],
            ],
            dtype=float,
        )
        B_k = np.array(
            [
                [0.0, -dt_s * v_mps * sin_sum * beta_delta],
                [0.0, dt_s * v_mps * cos_sum * beta_delta],
                [dt_s, 0.0],
                [0.0, dt_s * (v_mps / l_r_m) * cos_beta * beta_delta],
            ],
            dtype=float,
        )

        f_bar = np.array(
            [
                float(x_bar[0] + dt_s * v_mps * cos_sum),
                float(x_bar[1] + dt_s * v_mps * sin_sum),
                float(x_bar[2] + dt_s * u_bar[0]),
                float(self._wrap_angle(x_bar[3] + dt_s * (v_mps / l_r_m) * sin_beta)),
            ],
            dtype=float,
        )
        c_k = f_bar - A_k @ x_bar - B_k @ u_bar
        return A_k, B_k, c_k

    def _build_qp(
        self,
        x0: np.ndarray,
        x_ref_target: np.ndarray,
        object_snapshots: Sequence[Mapping[str, object]],
        current_acceleration_mps2: float,
        current_steering_rad: float,
        x_ref_rollout: np.ndarray,
        u_ref_rollout: np.ndarray,
        lane_center_reference: Sequence[Mapping[str, object]] | None,
        speed_upper_bound_mps: float | None,
        reachable_speed_floor_profile_mps: Sequence[float] | None,
    ) -> Tuple[sp.csc_matrix, np.ndarray, sp.csc_matrix, np.ndarray, np.ndarray, QPIndex]:
        """
        Intent:
            Build the full convex QP matrices for OSQP.

        Important implementation note:
            Collision-checker auxiliary constraints are disabled in this mode.
            Obstacle handling is done through repulsive-potential cost shaping.
        """

        object_count = len(object_snapshots)
        index = QPIndex(nx=self.nx, nu=self.nu, horizon_steps=self.horizon_steps)
        n_var = index.total_variables
        effective_speed_upper_bound_mps = float(self.constraints.max_velocity_mps)
        if speed_upper_bound_mps is not None:
            effective_speed_upper_bound_mps = min(
                float(effective_speed_upper_bound_mps),
                max(float(self.constraints.min_velocity_mps), float(speed_upper_bound_mps)),
            )

        # --- Helpers for sparse QP assembly ---
        q = np.zeros(n_var, dtype=float)
        p_entries: Dict[Tuple[int, int], float] = {}
        a_row: List[int] = []
        a_col: List[int] = []
        a_data: List[float] = []
        lower_bounds: List[float] = []
        upper_bounds: List[float] = []

        def add_p_entry(i: int, j: int, value: float) -> None:
            if abs(value) < 1e-12:
                return
            ii, jj = (i, j) if i <= j else (j, i)
            p_entries[(ii, jj)] = p_entries.get((ii, jj), 0.0) + float(value)

        def add_quadratic(var_idx: int, weight: float) -> None:
            if weight <= 0.0:
                return
            add_p_entry(var_idx, var_idx, 2.0 * float(weight))

        def add_tracking(var_idx: int, weight: float, ref_value: float) -> None:
            if weight <= 0.0:
                return
            add_quadratic(var_idx, weight)
            q[var_idx] += -2.0 * float(weight) * float(ref_value)

        def add_constraint(coeffs: Mapping[int, float], lower: float, upper: float) -> None:
            row_idx = len(lower_bounds)
            for col_idx, coeff in coeffs.items():
                if abs(float(coeff)) < 1e-12:
                    continue
                a_row.append(row_idx)
                a_col.append(int(col_idx))
                a_data.append(float(coeff))
            lower_bounds.append(float(lower))
            upper_bounds.append(float(upper))

        # Reference state is the destination state.
        # The image shows the state-tracking sum from k=0..N. Here the k=0 term
        # is omitted from optimization assembly because X_0 is fixed by an
        # equality constraint, so that term is a constant offset and does not
        # change the optimizer solution.
        x_ref_value = float(x_ref_target[0])
        y_ref_value = float(x_ref_target[1])
        v_ref_value = float(x_ref_target[2])
        psi_ref_value = float(x_ref_target[3])

        # --- Objective: control term Cost_Control ---
        # Penalizes rapid changes in acceleration and steering across horizon.
        comfort_scale = float(self.comfort_cost.w_comf)

        # J_ctrl rate terms: ((a_k-a_{k-1})/dt)^2 + ((delta_k-delta_{k-1})/dt)^2
        qa_eff = comfort_scale * float(self.comfort_cost.qa) / max(1e-9, self.dt_s * self.dt_s)
        qd_eff = comfort_scale * float(self.comfort_cost.qdelta) / max(1e-9, self.dt_s * self.dt_s)

        def add_rate_penalty(var_idx: int, prev_idx: int | None, prev_value: float, weight: float) -> None:
            if weight <= 0.0:
                return
            if prev_idx is None:
                # (u - u_prev_const)^2
                add_quadratic(var_idx, weight)
                q[var_idx] += -2.0 * float(weight) * float(prev_value)
                return
            # (u_k - u_{k-1})^2 = u_k^2 + u_{k-1}^2 - 2 u_k u_{k-1}
            add_quadratic(var_idx, weight)
            add_quadratic(prev_idx, weight)
            add_p_entry(var_idx, prev_idx, -2.0 * float(weight))

        for k in range(self.horizon_steps):
            a_idx = index.control_index(k, 0)
            d_idx = index.control_index(k, 1)
            if k == 0:
                add_rate_penalty(a_idx, None, float(current_acceleration_mps2), qa_eff)
                add_rate_penalty(d_idx, None, float(current_steering_rad), qd_eff)
            else:
                add_rate_penalty(a_idx, index.control_index(k - 1, 0), 0.0, qa_eff)
                add_rate_penalty(d_idx, index.control_index(k - 1, 1), 0.0, qd_eff)

        # --- Objective: attractive term Cost_ref ---
        # Quadratic pull to destination reference state.
        attractive_scale = float(self.safety_cost.w_safe)
        w_qx_safe = attractive_scale * float(self.comfort_cost.qx)
        w_qy_safe = attractive_scale * float(self.comfort_cost.qy)
        w_qv_safe = attractive_scale * float(self.comfort_cost.qv)
        w_qpsi_safe = attractive_scale * float(self.comfort_cost.qpsi)
        for k in range(1, self.horizon_steps + 1):
            x_k_idx = index.state_index(k, 0)
            y_k_idx = index.state_index(k, 1)
            add_tracking(x_k_idx, w_qx_safe, x_ref_value)
            add_tracking(y_k_idx, w_qy_safe, y_ref_value)
            add_tracking(index.state_index(k, 2), w_qv_safe, v_ref_value)
            add_tracking(index.state_index(k, 3), w_qpsi_safe, psi_ref_value)

            # Lane-center-follow term:
            #   P_att_lane = w_lane * e_y^2 + w_lane * q_psi_lane * e_psi^2,
            # where
            #   e_y   = -(x-x_ref)sin(theta_ref) + (y-y_ref)cos(theta_ref)
            #   e_psi = wrap(psi - theta_ref)
            if bool(self.lane_center_follow_enabled) and float(self.lane_center_follow_weight) > 0.0:
                lane_ref = self._get_lane_center_stage_ref(
                    lane_center_reference=lane_center_reference,
                    stage_index=int(k),
                    query_x_m=float(x_ref_rollout[k, 0]),
                    query_y_m=float(x_ref_rollout[k, 1]),
                )
                if lane_ref is not None:
                    lane_x_ref, lane_y_ref, lane_heading_ref = lane_ref
                    a_coef = -math.sin(float(lane_heading_ref))
                    b_coef = math.cos(float(lane_heading_ref))
                    c_coef = math.sin(float(lane_heading_ref)) * float(lane_x_ref) - math.cos(float(lane_heading_ref)) * float(lane_y_ref)
                    lane_weight = float(self.lane_center_follow_weight)
                    lane_heading_weight = lane_weight * float(self.lane_center_follow_qpsi)
                    lane_heading_ref_aligned = self._align_angle_near(
                        angle_rad=float(lane_heading_ref),
                        around_rad=float(x_ref_rollout[k, 3]),
                    )

                    add_quadratic(x_k_idx, lane_weight * a_coef * a_coef)
                    add_quadratic(y_k_idx, lane_weight * b_coef * b_coef)
                    add_p_entry(x_k_idx, y_k_idx, 2.0 * lane_weight * a_coef * b_coef)
                    q[x_k_idx] += 2.0 * lane_weight * a_coef * c_coef
                    q[y_k_idx] += 2.0 * lane_weight * b_coef * c_coef
                    if lane_heading_weight > 0.0:
                        add_tracking(index.state_index(k, 3), lane_heading_weight, lane_heading_ref_aligned)

        # --- Objective: repulsive potential field Cost_Repulsive ---
        # Super-ellipsoid obstacle cost from `super_ellipsoid.py`, approximated
        # by a local quadratic Taylor model in [x, y, v, psi] for each stage.
        if bool(self.repulsive_cost.enabled) and object_count > 0:
            for k in range(1, self.horizon_steps + 1):
                stage_idx = k - 1
                x_idx = index.state_index(k, 0)
                y_idx = index.state_index(k, 1)
                v_idx = index.state_index(k, 2)
                psi_idx = index.state_index(k, 3)

                ego_state_ref = np.array(
                    [
                        float(x_ref_rollout[k, 0]),
                        float(x_ref_rollout[k, 1]),
                        float(x_ref_rollout[k, 2]),
                        float(self._wrap_angle(float(x_ref_rollout[k, 3]))),
                    ],
                    dtype=float,
                )
                state_indices = [x_idx, y_idx, v_idx, psi_idx]

                for object_snapshot in object_snapshots:
                    obj_state = self._get_object_state_at_stage(
                        object_snapshot=object_snapshot,
                        stage_index=stage_idx,
                        dt_s=float(self.dt_s),
                    )

                    repulsive_weight = float(object_snapshot.get("repulsive_class_weight", 1.0))
                    if repulsive_weight <= 0.0:
                        continue

                    obstacle_length_m = float(object_snapshot.get("length_m", 4.5))
                    obstacle_width_m = float(object_snapshot.get("width_m", 2.0))
                    _p0, gradient, hessian = self._superellipsoid_cost_taylor_terms(
                        ego_state_ref=ego_state_ref,
                        obstacle_state=obj_state,
                        obstacle_length_m=obstacle_length_m,
                        obstacle_width_m=obstacle_width_m,
                    )

                    gradient = float(repulsive_weight) * np.asarray(gradient, dtype=float)
                    hessian = float(repulsive_weight) * np.asarray(hessian, dtype=float)

                    if bool(self.repulsive_cost.project_hessian_psd):
                        hessian = self._project_symmetric_hessian_to_psd(hessian=hessian)

                    linear_term = np.asarray(gradient - hessian @ ego_state_ref, dtype=float)

                    for row_local, row_idx in enumerate(state_indices):
                        q[row_idx] += float(linear_term[row_local])
                        for col_local in range(row_local, len(state_indices)):
                            add_p_entry(
                                row_idx,
                                state_indices[col_local],
                                float(hessian[row_local, col_local]),
                            )

        # Optional tiny regularization on controls to improve numerical conditioning.
        # This does not change the problem meaningfully but stabilizes OSQP.
        tiny_reg = 1e-6
        for k in range(self.horizon_steps):
            add_quadratic(index.control_index(k, 0), tiny_reg)
            add_quadratic(index.control_index(k, 1), tiny_reg)

        # --- Constraints ---
        # Initial state equality X_0 = current state.
        for i in range(self.nx):
            add_constraint({index.state_index(0, i): 1.0}, float(x0[i]), float(x0[i]))

        # LTV dynamics equality constraints.
        for k in range(self.horizon_steps):
            A_k, B_k, c_k = self._linearize_dynamics(x_ref_rollout[k], u_ref_rollout[k])
            for i in range(self.nx):
                coeffs: Dict[int, float] = {index.state_index(k + 1, i): 1.0}
                for j in range(self.nx):
                    coeffs[index.state_index(k, j)] = coeffs.get(index.state_index(k, j), 0.0) - float(A_k[i, j])
                for j in range(self.nu):
                    coeffs[index.control_index(k, j)] = coeffs.get(index.control_index(k, j), 0.0) - float(B_k[i, j])
                add_constraint(coeffs, float(c_k[i]), float(c_k[i]))

        # Speed constraints for future states.
        for k in range(1, self.horizon_steps + 1):
            stage_speed_upper_bound_mps = self._future_speed_upper_bound_mps(
                active_speed_upper_bound_mps=float(effective_speed_upper_bound_mps),
                future_state_index=int(k),
                reachable_speed_floor_profile_mps=reachable_speed_floor_profile_mps,
            )
            add_constraint(
                {index.state_index(k, 2): 1.0},
                self.constraints.min_velocity_mps,
                float(stage_speed_upper_bound_mps),
            )
        # Optional hard terminal-speed constraint. Apply it only for stop-like
        # destinations (destination speed near zero), otherwise every rolling
        # temporary goal would incorrectly force the horizon-end speed to zero.
        terminal_speed_constraint_active = bool(self.constraints.enforce_terminal_velocity_constraint) and (
            abs(float(x_ref_target[2])) <= float(self.final_stop_speed_cap_activation_threshold_mps)
        )
        if terminal_speed_constraint_active:
            add_constraint(
                {index.state_index(self.horizon_steps, 2): 1.0},
                float(self.constraints.terminal_velocity_mps),
                float(self.constraints.terminal_velocity_mps),
            )

        # Acceleration and steering bounds.
        for k in range(self.horizon_steps):
            add_constraint(
                {index.control_index(k, 0): 1.0},
                self.constraints.min_acceleration_mps2,
                self.constraints.max_acceleration_mps2,
            )
            add_constraint(
                {index.control_index(k, 1): 1.0},
                self.constraints.min_steer_rad,
                self.constraints.max_steer_rad,
            )

        # Jerk bounds |a_k - a_{k-1}| <= j_max * dt.
        jerk_delta_limit = float(self.constraints.max_jerk_mps3) * float(self.dt_s)
        for k in range(self.horizon_steps):
            a_k_idx = index.control_index(k, 0)
            if k == 0:
                add_constraint(
                    {a_k_idx: 1.0},
                    float(current_acceleration_mps2) - jerk_delta_limit,
                    float(current_acceleration_mps2) + jerk_delta_limit,
                )
            else:
                a_km1_idx = index.control_index(k - 1, 0)
                add_constraint({a_k_idx: 1.0, a_km1_idx: -1.0}, -jerk_delta_limit, jerk_delta_limit)

        # Steering-rate bounds:
        #   min_rate <= (delta_k - delta_{k-1}) / dt <= max_rate
        # where delta_{-1} is the currently applied steering angle.
        steer_delta_min = float(self.constraints.min_steer_rate_rps) * float(self.dt_s)
        steer_delta_max = float(self.constraints.max_steer_rate_rps) * float(self.dt_s)
        for k in range(self.horizon_steps):
            d_k_idx = index.control_index(k, 1)
            if k == 0:
                add_constraint(
                    {d_k_idx: 1.0},
                    float(current_steering_rad) + steer_delta_min,
                    float(current_steering_rad) + steer_delta_max,
                )
            else:
                d_km1_idx = index.control_index(k - 1, 1)
                add_constraint(
                    {d_k_idx: 1.0, d_km1_idx: -1.0},
                    steer_delta_min,
                    steer_delta_max,
                )
        # Collision-checker constraints removed per configuration.

        # Assemble sparse matrices.
        if len(p_entries) == 0:
            P = sp.csc_matrix((n_var, n_var), dtype=float)
        else:
            p_rows = [idx_pair[0] for idx_pair in p_entries.keys()]
            p_cols = [idx_pair[1] for idx_pair in p_entries.keys()]
            p_vals = [val for val in p_entries.values()]
            P = sp.csc_matrix((p_vals, (p_rows, p_cols)), shape=(n_var, n_var), dtype=float)

        A = sp.csc_matrix((a_data, (a_row, a_col)), shape=(len(lower_bounds), n_var), dtype=float)
        l = np.asarray(lower_bounds, dtype=float)
        u = np.asarray(upper_bounds, dtype=float)
        return P, q, A, l, u, index

    def _solve_qp(
        self,
        P: sp.csc_matrix,
        q: np.ndarray,
        A: sp.csc_matrix,
        l: np.ndarray,
        u: np.ndarray,
    ) -> Tuple[np.ndarray | None, str, float]:
        """Solve the QP with OSQP and return solution/status/time."""

        solver = osqp.OSQP()  # type: ignore[union-attr]
        t0 = time.perf_counter()
        solver.setup(
            P=P,
            q=q,
            A=A,
            l=l,
            u=u,
            verbose=False,
            warm_start= True,
            polish=self.qp_polish,
            max_iter=self.qp_max_iter,
            eps_abs=self.qp_eps_abs,
            eps_rel=self.qp_eps_rel,
            adaptive_rho=True,
        )
        result = solver.solve()
        solve_time_ms = (time.perf_counter() - t0) * 1000.0
        status = str(result.info.status).lower()
        if result.x is None or "solved" not in status:
            return None, status, float(solve_time_ms)
        return np.asarray(result.x, dtype=float), status, float(solve_time_ms)

    def _extract_solution(self, solution: np.ndarray, index: QPIndex) -> Tuple[np.ndarray, np.ndarray]:
        x_traj = np.zeros((self.horizon_steps + 1, self.nx), dtype=float)
        u_traj = np.zeros((self.horizon_steps, self.nu), dtype=float)
        for k in range(self.horizon_steps + 1):
            for i in range(self.nx):
                x_traj[k, i] = float(solution[index.state_index(k, i)])
            x_traj[k, 3] = self._wrap_angle(float(x_traj[k, 3]))
        for k in range(self.horizon_steps):
            for i in range(self.nu):
                u_traj[k, i] = float(solution[index.control_index(k, i)])
        return x_traj, u_traj

    def _evaluate_plan_cost_terms(
        self,
        x_traj: np.ndarray,
        u_traj: np.ndarray,
        x_ref_target: np.ndarray,
        object_snapshots: Sequence[Mapping[str, object]],
        current_acceleration_mps2: float,
        current_steering_rad: float,
        lane_center_reference: Sequence[Mapping[str, object]] | None,
    ) -> Dict[str, float]:
        """
        Evaluate per-term objective values for the most recent planned trajectory.

        These values are for runtime diagnostics/plotting and match the active
        cost terms used by this MPC implementation.
        """

        attractive_scale = float(self.safety_cost.w_safe)
        qx = float(self.comfort_cost.qx)
        qy = float(self.comfort_cost.qy)
        qv = float(self.comfort_cost.qv)
        qpsi = float(self.comfort_cost.qpsi)

        x_ref = float(x_ref_target[0])
        y_ref = float(x_ref_target[1])
        v_ref = float(x_ref_target[2])
        psi_ref = float(x_ref_target[3])

        cost_attractive_ref = 0.0
        cost_lane_center = 0.0
        for k in range(1, self.horizon_steps + 1):
            dx = float(x_traj[k, 0]) - x_ref
            dy = float(x_traj[k, 1]) - y_ref
            dv = float(x_traj[k, 2]) - v_ref
            dpsi = self._wrap_angle(float(x_traj[k, 3]) - psi_ref)
            cost_attractive_ref += qx * dx * dx + qy * dy * dy + qv * dv * dv + qpsi * dpsi * dpsi

            if bool(self.lane_center_follow_enabled) and float(self.lane_center_follow_weight) > 0.0:
                lane_ref = self._get_lane_center_stage_ref(
                    lane_center_reference=lane_center_reference,
                    stage_index=int(k),
                    query_x_m=float(x_traj[k, 0]),
                    query_y_m=float(x_traj[k, 1]),
                )
                if lane_ref is not None:
                    lane_x_ref, lane_y_ref, lane_heading_ref = lane_ref
                    e_y = -(float(x_traj[k, 0]) - float(lane_x_ref)) * math.sin(float(lane_heading_ref))
                    e_y += (float(x_traj[k, 1]) - float(lane_y_ref)) * math.cos(float(lane_heading_ref))
                    e_psi_lane = self._wrap_angle(float(x_traj[k, 3]) - float(lane_heading_ref))
                    lane_weight_k = float(self.lane_center_follow_weight)
                    cost_lane_center += lane_weight_k * e_y * e_y
                    cost_lane_center += lane_weight_k * float(self.lane_center_follow_qpsi) * e_psi_lane * e_psi_lane

        cost_attractive = attractive_scale * cost_attractive_ref
        cost_lane_center = float(cost_lane_center)
        j_ctrl = 0.0
        a_prev = float(current_acceleration_mps2)
        d_prev = float(current_steering_rad)
        inv_dt = 1.0 / max(1e-9, float(self.dt_s))
        qa = float(self.comfort_cost.qa)
        qd = float(self.comfort_cost.qdelta)
        for k in range(self.horizon_steps):
            a_k = float(u_traj[k, 0])
            d_k = float(u_traj[k, 1])
            da = (a_k - a_prev) * inv_dt
            dd = (d_k - d_prev) * inv_dt
            j_ctrl += qa * da * da + qd * dd * dd
            a_prev = a_k
            d_prev = d_k
        cost_control = float(self.comfort_cost.w_comf) * j_ctrl

        cost_repulsive_safe = 0.0
        cost_repulsive_collision = 0.0
        if bool(self.repulsive_cost.enabled) and len(object_snapshots) > 0:
            for k in range(1, self.horizon_steps + 1):
                stage_idx = k - 1
                ego_state = [
                    float(x_traj[k, 0]),
                    float(x_traj[k, 1]),
                    float(x_traj[k, 2]),
                    float(self._wrap_angle(float(x_traj[k, 3]))),
                ]

                for object_snapshot in object_snapshots:
                    obj_state = self._get_object_state_at_stage(
                        object_snapshot=object_snapshot,
                        stage_index=stage_idx,
                        dt_s=float(self.dt_s),
                    )
                    repulsive_weight = float(object_snapshot.get("repulsive_class_weight", 1.0))
                    if repulsive_weight <= 0.0:
                        continue

                    obstacle_length_m = float(object_snapshot.get("length_m", 4.5))
                    obstacle_width_m = float(object_snapshot.get("width_m", 2.0))
                    obstacle_cost_safe, obstacle_cost_collision = self._superellipsoid_obstacle_cost_components(
                        ego_state=ego_state,
                        obstacle_state=obj_state,
                        obstacle_length_m=obstacle_length_m,
                        obstacle_width_m=obstacle_width_m,
                    )
                    cost_repulsive_safe += float(repulsive_weight) * float(obstacle_cost_safe)
                    cost_repulsive_collision += float(repulsive_weight) * float(obstacle_cost_collision)
        cost_repulsive = float(cost_repulsive_safe + cost_repulsive_collision)
        return {
            "Cost_ref": float(cost_attractive),
            "Cost_LaneCenter": float(cost_lane_center),
            "Cost_Repulsive_Safe": float(cost_repulsive_safe),
            "Cost_Repulsive_Collision": float(cost_repulsive_collision),
            "Cost_Repulsive": float(cost_repulsive),
            "Cost_Control": float(cost_control),
        }

    def plan_trajectory(
        self,
        current_state: Sequence[float],
        destination_state: Sequence[float],
        object_snapshots: Sequence[Mapping[str, object]],
        current_acceleration_mps2: float,
        current_steering_rad: float,
        lane_center_waypoints: Sequence[Mapping[str, object]] | None = None,
    ) -> List[List[float]]:
        """
        Intent:
            Solve one MPC optimization and return future states [x,y,v,psi].
        """

        if len(current_state) != 4:
            raise ValueError("current_state must be [x, y, v, psi].")

        x0 = np.array(
            [
                float(current_state[0]),
                float(current_state[1]),
                self._clamp(float(current_state[2]), self.constraints.min_velocity_mps, self.constraints.max_velocity_mps),
                self._wrap_angle(float(current_state[3])),
            ],
            dtype=float,
        )
        destination = self._normalize_destination_state(destination_state)
        destination_lane_id = (
            int(destination_state[4])
            if len(destination_state) >= 5
            else None
        )
        active_speed_upper_bound_mps = self._compute_active_speed_upper_bound_mps(
            current_state=x0,
            destination_state=destination,
        )
        reachable_speed_floor_profile_mps = self._minimum_reachable_speed_profile_mps(
            current_speed_mps=float(x0[2]),
            current_acceleration_mps2=float(current_acceleration_mps2),
        )
        self._last_active_max_velocity_mps = float(active_speed_upper_bound_mps)
        destination[2] = self._clamp(
            float(destination[2]),
            self.constraints.min_velocity_mps,
            float(active_speed_upper_bound_mps),
        )
        destination[3] = self._wrap_angle(float(destination[3]))

        lane_center_reference = self._build_lane_center_reference(
            current_state=x0,
            destination_state=destination,
            lane_center_waypoints=lane_center_waypoints,
            destination_lane_id=destination_lane_id,
        )

        shifted_seed = self._build_shifted_previous_solution_seed(x0=x0)
        x_ref_rollout, u_ref_rollout = self._reference_rollout(
            x0=x0,
            x_ref_target=destination,
            lane_center_reference=lane_center_reference,
            object_snapshots=object_snapshots,
            speed_upper_bound_mps=float(active_speed_upper_bound_mps),
            reachable_speed_floor_profile_mps=reachable_speed_floor_profile_mps,
            seed_state_traj=shifted_seed[0] if shifted_seed is not None else None,
            seed_control_traj=shifted_seed[1] if shifted_seed is not None else None,
        )

        total_solve_time_ms = 0.0
        current_x_ref_rollout = np.asarray(x_ref_rollout, dtype=float)
        current_u_ref_rollout = np.asarray(u_ref_rollout, dtype=float)
        best_x_solution: np.ndarray | None = None
        best_u_solution: np.ndarray | None = None
        best_status = "not_solved"

        for iteration_idx in range(int(self.reference_sequential_iterations)):
            P, q, A, l, u, index = self._build_qp(
                x0=x0,
                x_ref_target=destination,
                object_snapshots=object_snapshots,
                current_acceleration_mps2=float(current_acceleration_mps2),
                current_steering_rad=float(current_steering_rad),
                x_ref_rollout=current_x_ref_rollout,
                u_ref_rollout=current_u_ref_rollout,
                lane_center_reference=lane_center_reference,
                speed_upper_bound_mps=float(active_speed_upper_bound_mps),
                reachable_speed_floor_profile_mps=reachable_speed_floor_profile_mps,
            )
            solution, status, solve_time_ms = self._solve_qp(P=P, q=q, A=A, l=l, u=u)
            total_solve_time_ms += float(solve_time_ms)

            if solution is None:
                if best_x_solution is None:
                    best_status = str(status)
                break

            best_status = str(status)
            best_x_solution, best_u_solution = self._extract_solution(solution=solution, index=index)
            if int(iteration_idx) + 1 >= int(self.reference_sequential_iterations):
                break

            current_x_ref_rollout = np.asarray(best_x_solution, dtype=float)
            current_u_ref_rollout = np.asarray(best_u_solution, dtype=float)

        self._last_status = str(best_status)
        self._last_solve_time_ms = float(total_solve_time_ms)

        if best_x_solution is None or best_u_solution is None:
            # Deterministic fallback: return the current rollout if all QP solves fail.
            x_solution = current_x_ref_rollout
            u_solution = current_u_ref_rollout
        else:
            x_solution = np.asarray(best_x_solution, dtype=float)
            u_solution = np.asarray(best_u_solution, dtype=float)
        x_solution = np.asarray(x_solution, dtype=float)
        u_solution = np.asarray(u_solution, dtype=float)
        for k in range(1, self.horizon_steps + 1):
            stage_speed_upper_bound_mps = self._future_speed_upper_bound_mps(
                active_speed_upper_bound_mps=float(active_speed_upper_bound_mps),
                future_state_index=int(k),
                reachable_speed_floor_profile_mps=reachable_speed_floor_profile_mps,
            )
            x_solution[k, 2] = self._clamp(
                float(x_solution[k, 2]),
                float(self.constraints.min_velocity_mps),
                float(stage_speed_upper_bound_mps),
            )
            x_solution[k, 3] = self._wrap_angle(float(x_solution[k, 3]))

        self._last_x_solution = np.asarray(x_solution, dtype=float)
        self._last_u_solution = np.asarray(u_solution, dtype=float)

        if best_x_solution is not None and best_u_solution is not None:
            self._previous_x_solution = np.asarray(x_solution, dtype=float)
            self._previous_u_solution = np.asarray(u_solution, dtype=float)

        self._last_cost_terms = self._evaluate_plan_cost_terms(
            x_traj=x_solution,
            u_traj=u_solution,
            x_ref_target=destination,
            object_snapshots=object_snapshots,
            current_acceleration_mps2=float(current_acceleration_mps2),
            current_steering_rad=float(current_steering_rad),
            lane_center_reference=lane_center_reference,
        )

        output: List[List[float]] = []
        for k in range(1, self.horizon_steps + 1):
            output.append(
                [
                    float(x_solution[k, 0]),
                    float(x_solution[k, 1]),
                    float(x_solution[k, 2]),
                    float(self._wrap_angle(float(x_solution[k, 3]))),
                ]
            )
        return output

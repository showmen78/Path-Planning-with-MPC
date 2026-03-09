"""
Trajectory PID controller for MPC_custom.

This module provides a lightweight trajectory-tracking controller used to
convert MPC-generated future states into executable vehicle controls:
    control = [acceleration, steering_angle]

Role in this project:
    - MPC generates the optimal future state sequence [x, y, v, psi].
    - The PID controller tracks that sequence in the simulation loop.
    - The ego vehicle is moved by applying the PID outputs through the
      vehicle's kinematic model (`Vehicle.step(...)`), instead of directly
      teleporting the state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple
import math


@dataclass
class PIDAxisConfig:
    """
    Intent:
        Store one PID loop gain set and numeric limits.

    Fields:
        k_p:
            float, proportional gain.
        k_i:
            float, integral gain.
        k_d:
            float, derivative gain.
        integral_limit:
            float >= 0, absolute clamp for the accumulated integral state.
            This prevents integral wind-up when large errors persist.
        output_limit:
            float >= 0, optional absolute clamp applied to the PID loop output
            before it is combined with other terms. Use 0 or negative to disable.
    """

    k_p: float
    k_i: float
    k_d: float
    integral_limit: float
    output_limit: float


class PIDAxis:
    """
    Intent:
        Implement one scalar PID control loop with anti-windup clamping.

    Logic / method:
        u = k_p * e + k_i * integral(e) + k_d * d(e)/dt

    Inputs / Outputs:
        - `step(error, dt_s)`:
            error:
                float scalar, current control error.
            dt_s:
                float [s], controller update period.
            Returns:
                float scalar PID output.
    """

    def __init__(self, cfg: PIDAxisConfig) -> None:
        self.cfg = cfg
        self._integral = 0.0
        self._prev_error = 0.0
        self._has_prev = False

    def reset(self) -> None:
        """Reset integral and derivative history."""

        self._integral = 0.0
        self._prev_error = 0.0
        self._has_prev = False

    def step(self, error: float, dt_s: float) -> float:
        """
        Intent:
            Compute one PID output sample.

        Inputs:
            error:
                float scalar control error.
            dt_s:
                float [s], update period (must be > 0).

        Output:
            float scalar PID output after optional output clipping.
        """

        dt_s = max(1e-6, float(dt_s))
        error = float(error)

        self._integral += error * dt_s
        i_lim = max(0.0, float(self.cfg.integral_limit))
        if i_lim > 0.0:
            self._integral = max(-i_lim, min(i_lim, self._integral))

        if self._has_prev:
            derivative = (error - self._prev_error) / dt_s
        else:
            derivative = 0.0
            self._has_prev = True

        output = (
            float(self.cfg.k_p) * error
            + float(self.cfg.k_i) * self._integral
            + float(self.cfg.k_d) * derivative
        )

        self._prev_error = error

        o_lim = float(self.cfg.output_limit)
        if o_lim > 0.0:
            output = max(-o_lim, min(o_lim, output))
        return float(output)


@dataclass
class TrajectoryPIDConfig:
    """
    Intent:
        Store trajectory-tracking controller configuration for the ego vehicle.

    Fields:
        dt_s:
            float [s], controller execution period. In this project it is set to
            the simulation step (`sim_dt_s`).
        lookahead_steps:
            int scalar, preview offset into the planned trajectory for smoother
            tracking. Example: 2 means track point `cursor + 2` while cursor
            progression still follows the current waypoint.
        max_index_advance_per_step:
            int scalar, maximum number of trajectory points the controller is
            allowed to skip in one simulation step when the ego is already close
            to / past earlier points.
        waypoint_reached_distance_m:
            float [m], distance threshold for advancing to the next trajectory
            waypoint.
        heading_reached_threshold_rad:
            float [rad], optional heading threshold used together with distance
            when deciding that the current waypoint is "reached enough".
        along_track_pass_threshold_m:
            float [m], if the ego has passed a waypoint by more than this amount
            along the target heading direction, the controller can skip it.
        heading_los_blend:
            float in [0, 1], blend between target-state heading and
            line-of-sight heading to the target point.
            0 -> use target psi only, 1 -> use LOS heading only.
        speed_preview_limit_mps:
            float [m/s], optional clamp on the internally adjusted target speed.
            If <= 0, no extra clamp beyond vehicle limits is applied.
        steering_rate_limit_rad_per_s:
            float [rad/s], limits steering command change rate for smoother
            actuation. If <= 0, disabled.
        lateral_cross_track_gain:
            float scalar, multiplies the lateral position PID contribution before
            summing with the heading PID contribution.
    """

    dt_s: float
    lookahead_steps: int
    max_index_advance_per_step: int
    waypoint_reached_distance_m: float
    heading_reached_threshold_rad: float
    along_track_pass_threshold_m: float
    heading_los_blend: float
    speed_preview_limit_mps: float
    steering_rate_limit_rad_per_s: float
    lateral_cross_track_gain: float


class TrajectoryPIDController:
    """
    Intent:
        Track an MPC-generated trajectory and generate [a, delta] commands for
        the ego vehicle.

    Logic / method used:
        1. Select a target state from the planned trajectory (with optional
           lookahead for smoother steering).
        2. Longitudinal loop:
             PID on speed error (optionally speed target is adjusted using
             along-track position error).
        3. Lateral loop:
             Steering = PID(heading_error) + gain * PID(cross_track_error)
           where cross-track error is measured in the ego body frame.
        4. The returned acceleration and steering are applied to the vehicle's
           kinematic bicycle model via `Vehicle.set_control` + `Vehicle.step`.

    Inputs / Outputs:
        - `advance_target_index(current_state, planned_states, plan_cursor)`:
            current_state: [x, y, v, psi]
            planned_states: sequence of [x, y, v, psi]
            plan_cursor: int
            Returns updated cursor index.
        - `compute_control(current_state, target_state, next_target_state, limits)`:
            current_state: [x, y, v, psi]
            target_state: [x, y, v, psi]
            next_target_state: optional preview state (unused for dynamics,
                used to improve heading selection if provided).
            limits: mapping with `max_acceleration_mps2`, `max_steer_rad`,
                optional min/max speed.
            Returns `(acceleration_mps2, steering_angle_rad, diagnostics_dict)`.
    """

    def __init__(self, pid_cfg: Mapping[str, object] | None = None, dt_s: float | None = None) -> None:
        cfg = dict(pid_cfg or {})

        lon_cfg_raw = dict(cfg.get("longitudinal", {}))
        lat_heading_cfg_raw = dict(cfg.get("lateral_heading", {}))
        lat_cte_cfg_raw = dict(cfg.get("lateral_cross_track", {}))
        tracking_cfg = dict(cfg.get("tracking", {}))

        self.config = TrajectoryPIDConfig(
            dt_s=max(1e-4, float(dt_s if dt_s is not None else cfg.get("dt_s", 0.05))),
            lookahead_steps=max(0, int(tracking_cfg.get("lookahead_steps", 1))),
            max_index_advance_per_step=max(1, int(tracking_cfg.get("max_index_advance_per_step", 4))),
            waypoint_reached_distance_m=max(0.05, float(tracking_cfg.get("waypoint_reached_distance_m", 0.6))),
            heading_reached_threshold_rad=max(0.05, float(tracking_cfg.get("heading_reached_threshold_rad", 0.7))),
            along_track_pass_threshold_m=max(0.0, float(tracking_cfg.get("along_track_pass_threshold_m", 0.25))),
            heading_los_blend=max(0.0, min(1.0, float(tracking_cfg.get("heading_los_blend", 0.35)))),
            speed_preview_limit_mps=float(tracking_cfg.get("speed_preview_limit_mps", 0.0)),
            steering_rate_limit_rad_per_s=max(0.0, float(tracking_cfg.get("steering_rate_limit_rad_per_s", 1.5))),
            lateral_cross_track_gain=float(tracking_cfg.get("lateral_cross_track_gain", 1.0)),
        )

        self._longitudinal_pid = PIDAxis(
            PIDAxisConfig(
                k_p=float(lon_cfg_raw.get("k_p", 2.0)),
                k_i=float(lon_cfg_raw.get("k_i", 0.05)),
                k_d=float(lon_cfg_raw.get("k_d", 0.15)),
                integral_limit=max(0.0, float(lon_cfg_raw.get("integral_limit", 10.0))),
                output_limit=max(0.0, float(lon_cfg_raw.get("output_limit", 0.0))),
            )
        )
        self._heading_pid = PIDAxis(
            PIDAxisConfig(
                k_p=float(lat_heading_cfg_raw.get("k_p", 2.2)),
                k_i=float(lat_heading_cfg_raw.get("k_i", 0.01)),
                k_d=float(lat_heading_cfg_raw.get("k_d", 0.20)),
                integral_limit=max(0.0, float(lat_heading_cfg_raw.get("integral_limit", 2.0))),
                output_limit=max(0.0, float(lat_heading_cfg_raw.get("output_limit", 0.0))),
            )
        )
        self._cross_track_pid = PIDAxis(
            PIDAxisConfig(
                k_p=float(lat_cte_cfg_raw.get("k_p", 0.65)),
                k_i=float(lat_cte_cfg_raw.get("k_i", 0.0)),
                k_d=float(lat_cte_cfg_raw.get("k_d", 0.08)),
                integral_limit=max(0.0, float(lat_cte_cfg_raw.get("integral_limit", 4.0))),
                output_limit=max(0.0, float(lat_cte_cfg_raw.get("output_limit", 0.0))),
            )
        )

        self._previous_steering_cmd_rad = 0.0

    @staticmethod
    def _wrap_angle(angle_rad: float) -> float:
        return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, float(value)))

    @staticmethod
    def _safe_state4(state: Sequence[float]) -> Tuple[float, float, float, float]:
        """
        Intent:
            Read a trajectory/state sample and normalize it to [x, y, v, psi].

        Input:
            state:
                sequence[float] of length >= 4.

        Output:
            tuple[float, float, float, float]
        """

        if len(state) < 4:
            raise ValueError("state must contain [x, y, v, psi].")
        return float(state[0]), float(state[1]), float(state[2]), float(state[3])

    def reset(self) -> None:
        """
        Intent:
            Reset all PID internal states (integral and derivative memories).

        Use:
            Call after a major replan or scenario reset to avoid carrying
            controller memory across unrelated trajectory segments.
        """

        self._longitudinal_pid.reset()
        self._heading_pid.reset()
        self._cross_track_pid.reset()
        self._previous_steering_cmd_rad = 0.0

    def update_dt(self, dt_s: float) -> None:
        """Update controller step time if the simulation step changes."""

        self.config.dt_s = max(1e-4, float(dt_s))

    def advance_target_index(
        self,
        current_state: Sequence[float],
        planned_states: Sequence[Sequence[float]],
        plan_cursor: int,
    ) -> int:
        """
        Intent:
            Advance the trajectory cursor when the ego has already reached or
            passed the current target state.

        Logic:
            - A point is considered reachable/consumed when:
              1) Euclidean distance <= waypoint_reached_distance_m and heading
                 error is small enough, OR
              2) The ego has passed the point along the target heading by more
                 than along_track_pass_threshold_m.
            - The number of skipped points per call is capped to maintain stable
              tracking behavior.

        Inputs:
            current_state:
                sequence[float], [x, y, v, psi] of ego.
            planned_states:
                sequence of future MPC states.
            plan_cursor:
                int current trajectory index.

        Output:
            int updated trajectory cursor index.
        """

        if len(planned_states) == 0:
            return int(plan_cursor)

        x_m, y_m, _, psi_rad = self._safe_state4(current_state)
        cursor = max(0, min(int(plan_cursor), len(planned_states) - 1))
        max_advances = int(self.config.max_index_advance_per_step)

        for _ in range(max_advances):
            if cursor >= len(planned_states) - 1:
                break

            tx_m, ty_m, _, tpsi_rad = self._safe_state4(planned_states[cursor])
            dx_m = tx_m - x_m
            dy_m = ty_m - y_m
            distance_m = math.hypot(dx_m, dy_m)
            heading_error_rad = abs(self._wrap_angle(tpsi_rad - psi_rad))

            passed_along_target_m = (
                math.cos(tpsi_rad) * (x_m - tx_m)
                + math.sin(tpsi_rad) * (y_m - ty_m)
            )

            reached = (
                distance_m <= float(self.config.waypoint_reached_distance_m)
                and heading_error_rad <= float(self.config.heading_reached_threshold_rad)
            )
            passed = passed_along_target_m > float(self.config.along_track_pass_threshold_m)

            if reached or passed:
                cursor += 1
            else:
                break

        return int(cursor)

    def compute_control(
        self,
        current_state: Sequence[float],
        target_state: Sequence[float],
        next_target_state: Sequence[float] | None,
        limits: Mapping[str, float],
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Intent:
            Convert a target trajectory state into executable acceleration and
            steering commands using PID tracking.

        Method:
            Longitudinal command:
                PID on speed error using target speed from the planned state,
                optionally boosted by along-track position error.

            Lateral command:
                steering = PID(heading_error) + gain * PID(cross_track_error)
                where cross-track error is computed in the ego body frame.

        Inputs:
            current_state:
                [x, y, v, psi] of ego.
            target_state:
                [x, y, v, psi] planned state currently being tracked.
            next_target_state:
                optional next planned state used to improve heading reference
                direction when the local path segment differs from target psi.
            limits:
                mapping with keys:
                    max_acceleration_mps2, max_steer_rad
                    optional min_velocity_mps, max_velocity_mps

        Outputs:
            tuple:
                acceleration_mps2:
                    float, command passed to `Vehicle.set_control`.
                steering_angle_rad:
                    float, command passed to `Vehicle.set_control`.
                diagnostics:
                    dict[str, float], useful for HUD/debugging.
        """

        x_m, y_m, v_mps, psi_rad = self._safe_state4(current_state)
        tx_m, ty_m, tv_mps, tpsi_rad = self._safe_state4(target_state)
        dt_s = float(self.config.dt_s)

        dx_m = tx_m - x_m
        dy_m = ty_m - y_m
        distance_m = math.hypot(dx_m, dy_m)

        # Line-of-sight heading to the target position. This helps convergence
        # when the ego is spatially off the planned point.
        if distance_m > 1e-9:
            los_heading_rad = math.atan2(dy_m, dx_m)
        else:
            los_heading_rad = tpsi_rad

        # If a next target exists, use the segment heading to reduce heading
        # noise when the target psi is slightly inconsistent with the path.
        if next_target_state is not None and len(next_target_state) >= 4:
            nx_m, ny_m, _, _ = self._safe_state4(next_target_state)
            seg_dx_m = nx_m - tx_m
            seg_dy_m = ny_m - ty_m
            if math.hypot(seg_dx_m, seg_dy_m) > 1e-9:
                tpsi_rad = math.atan2(seg_dy_m, seg_dx_m)

        blend = float(self.config.heading_los_blend)
        heading_ref_rad = self._wrap_angle(
            tpsi_rad + blend * self._wrap_angle(los_heading_rad - tpsi_rad)
        )

        heading_error_rad = self._wrap_angle(heading_ref_rad - psi_rad)
        cross_track_error_m = -math.sin(psi_rad) * dx_m + math.cos(psi_rad) * dy_m
        along_track_error_m = math.cos(psi_rad) * dx_m + math.sin(psi_rad) * dy_m

        # Track the planned target speed directly to avoid speed overshoot caused
        # by position-error speed boosting.
        target_speed_mps = float(tv_mps)
        speed_preview_cap = float(self.config.speed_preview_limit_mps)
        if speed_preview_cap > 0.0:
            target_speed_mps = self._clamp(target_speed_mps, 0.0, speed_preview_cap)

        if "max_velocity_mps" in limits:
            target_speed_mps = min(float(target_speed_mps), float(limits["max_velocity_mps"]))
        if "min_velocity_mps" in limits:
            target_speed_mps = max(float(target_speed_mps), float(limits["min_velocity_mps"]))

        speed_error_mps = float(target_speed_mps) - float(v_mps)

        accel_cmd_mps2 = self._longitudinal_pid.step(speed_error_mps, dt_s)
        steer_heading_rad = self._heading_pid.step(heading_error_rad, dt_s)
        steer_cte_rad = self._cross_track_pid.step(cross_track_error_m, dt_s)
        steer_cmd_rad = float(steer_heading_rad) + float(self.config.lateral_cross_track_gain) * float(steer_cte_rad)

        # Optional steering-rate limiting to reduce rapid oscillations.
        steer_rate_limit_rad_per_s = float(self.config.steering_rate_limit_rad_per_s)
        if steer_rate_limit_rad_per_s > 0.0:
            max_delta_change = steer_rate_limit_rad_per_s * dt_s
            lower = self._previous_steering_cmd_rad - max_delta_change
            upper = self._previous_steering_cmd_rad + max_delta_change
            steer_cmd_rad = self._clamp(steer_cmd_rad, lower, upper)

        max_accel = abs(float(limits.get("max_acceleration_mps2", 1e6)))
        max_steer = abs(float(limits.get("max_steer_rad", 1e6)))
        accel_cmd_mps2 = self._clamp(accel_cmd_mps2, -max_accel, max_accel)
        steer_cmd_rad = self._clamp(steer_cmd_rad, -max_steer, max_steer)

        self._previous_steering_cmd_rad = float(steer_cmd_rad)

        diagnostics = {
            "distance_to_target_m": float(distance_m),
            "heading_error_rad": float(heading_error_rad),
            "cross_track_error_m": float(cross_track_error_m),
            "along_track_error_m": float(along_track_error_m),
            "target_speed_mps": float(target_speed_mps),
            "speed_error_mps": float(speed_error_mps),
        }
        return float(accel_cmd_mps2), float(steer_cmd_rad), diagnostics

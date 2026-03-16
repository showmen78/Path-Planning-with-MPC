"""
Surrounding-vehicle summaries for the behavior-planner prompt.

This module converts tracker predictions into coarse intention labels and
provides the live super-ellipsoid safe-zone distance used by route lane-safety
checks.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Mapping, Sequence

from .global_planner import AStarGlobalPlanner
from utility import load_yaml_file


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MPC_YAML_PATH = os.path.join(PROJECT_ROOT, "MPC", "mpc.yaml")


@dataclass
class SurroundingVehicleSummary:
    """Prompt-facing summary for one surrounding vehicle."""

    vehicle_id: str
    x_m: float
    y_m: float
    v_mps: float
    psi_rad: float
    predicted_intention: str


def _wrap_angle(angle_rad: float) -> float:
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def _prediction_state_components(
    prediction_state: object,
    fallback_state: Mapping[str, object],
) -> tuple[float, float, float, float]:
    """
    Normalize one predicted state sample into (x, y, v, psi).

    Runtime tracker predictions in this project are stored as `[x, y, v, psi]`
    lists in `main.py`, but some call sites may also provide mapping-like
    samples. The behavior planner must accept both formats.
    """

    fallback_x_m = float(fallback_state.get("x", 0.0))
    fallback_y_m = float(fallback_state.get("y", 0.0))
    fallback_v_mps = float(fallback_state.get("v", 0.0))
    fallback_psi_rad = float(fallback_state.get("psi", 0.0))

    if isinstance(prediction_state, Mapping):
        return (
            float(prediction_state.get("x", fallback_x_m)),
            float(prediction_state.get("y", fallback_y_m)),
            float(prediction_state.get("v", fallback_v_mps)),
            float(prediction_state.get("psi", fallback_psi_rad)),
        )

    if isinstance(prediction_state, Sequence) and not isinstance(prediction_state, (str, bytes)):
        values = list(prediction_state)
        return (
            float(values[0]) if len(values) >= 1 else fallback_x_m,
            float(values[1]) if len(values) >= 2 else fallback_y_m,
            float(values[2]) if len(values) >= 3 else fallback_v_mps,
            float(values[3]) if len(values) >= 4 else fallback_psi_rad,
        )

    return fallback_x_m, fallback_y_m, fallback_v_mps, fallback_psi_rad


def _load_repulsive_cfg() -> Mapping[str, object]:
    cfg = load_yaml_file(MPC_YAML_PATH)
    return dict(cfg.get("mpc", {}).get("cost", {}).get("repulsive_potential", {}))


def compute_safe_zone_distance(
    ego_snapshot: Mapping[str, object],
    object_snapshot: Mapping[str, object],
    lane_width_m: float = 4.0,
 ) -> float:
    """
    Compute the live super-ellipsoid safe-zone normalized distance r_s.
    """

    repulsive_cfg = dict(_load_repulsive_cfg())

    ego_x_m = float(ego_snapshot.get("x", 0.0))
    ego_y_m = float(ego_snapshot.get("y", 0.0))
    ego_v_mps = max(0.0, float(ego_snapshot.get("v", 0.0)))
    ego_psi_rad = float(ego_snapshot.get("psi", 0.0))
    ego_length_m = max(1e-6, float(ego_snapshot.get("length_m", 4.5)))
    ego_width_m = max(1e-6, float(ego_snapshot.get("width_m", 2.0)))

    obs_x_m = float(object_snapshot.get("x", 0.0))
    obs_y_m = float(object_snapshot.get("y", 0.0))
    obs_v_mps = max(0.0, float(object_snapshot.get("v", 0.0)))
    obs_psi_rad = float(object_snapshot.get("psi", 0.0))
    obs_length_m = max(1e-6, float(object_snapshot.get("length_m", 4.5)))
    obs_width_m = max(1e-6, float(object_snapshot.get("width_m", 2.0)))

    heading_diff_rad = _wrap_angle(float(ego_psi_rad) - float(obs_psi_rad))
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
        float(repulsive_cfg.get("min_lateral_approach_speed_mps", 0.1)),
        abs(float(v_approach_lateral_mps)),
    )

    projected_ego_length_m = abs(float(ego_length_m) * math.cos(float(heading_diff_rad)))
    projected_ego_length_m += abs(float(ego_width_m) * math.sin(float(heading_diff_rad)))
    projected_ego_width_m = abs(float(ego_length_m) * math.sin(float(heading_diff_rad)))
    projected_ego_width_m += abs(float(ego_width_m) * math.cos(float(heading_diff_rad)))

    x0_m = 0.5 * (projected_ego_length_m + float(obs_length_m))
    x0_m += max(0.0, float(repulsive_cfg.get("static_longitudinal_buffer_m", 0.5)))
    y0_m = 0.5 * (projected_ego_width_m + float(obs_width_m))
    y0_m += max(0.0, float(repulsive_cfg.get("static_lateral_buffer_m", 1.0)))

    a_max_mps2 = max(1e-6, float(repulsive_cfg.get("max_braking_deceleration_mps2", 10.0)))
    a_comfort_mps2 = max(1e-6, float(repulsive_cfg.get("comfort_deceleration_mps2", 2.0)))
    reaction_time_s = max(0.0, float(repulsive_cfg.get("reaction_time_s", 1.0)))

    xc_m = x0_m + (delta_u_mps * delta_u_mps) / (2.0 * a_max_mps2)
    yc_m = y0_m + (delta_v_mps * delta_v_mps) / (2.0 * a_max_mps2)
    xs_m = x0_m + delta_u_mps * reaction_time_s + (delta_u_mps * delta_u_mps) / (2.0 * a_comfort_mps2)
    ys_m = y0_m + delta_v_mps * reaction_time_s + (delta_v_mps * delta_v_mps) / (2.0 * a_comfort_mps2)

    max_longitudinal_zone_length_m = max(
        0.0, float(repulsive_cfg.get("max_longitudinal_zone_length_m", 0.0))
    )
    if max_longitudinal_zone_length_m > 0.0:
        x_limit_m = 0.5 * float(max_longitudinal_zone_length_m)
        xc_m = min(float(xc_m), float(x_limit_m))
        xs_m = min(float(xs_m), float(x_limit_m))

    if bool(repulsive_cfg.get("limit_lateral_zone_to_lane_width", True)):
        lane_fraction = max(0.1, float(repulsive_cfg.get("max_lateral_zone_lane_fraction", 1.0)))
        y_limit_m = 0.5 * max(0.1, float(lane_width_m)) * lane_fraction
        yc_m = min(float(yc_m), float(y_limit_m))
        ys_m = min(float(ys_m), float(y_limit_m))

    exponent = max(1e-6, float(repulsive_cfg.get("shape_exponent", 4.0)))
    return (
        abs(float(x_local_m) / max(1e-6, float(xs_m))) ** exponent
        + abs(float(y_local_m) / max(1e-6, float(ys_m))) ** exponent
    ) ** (1.0 / exponent)


def infer_vehicle_intention(
    vehicle_snapshot: Mapping[str, object],
    lane_center_waypoints: Sequence[Mapping[str, object]],
    planner: AStarGlobalPlanner | None = None,
    speed_delta_threshold_mps: float = 0.5,
    stationary_speed_threshold_mps: float = 0.2,
    stationary_position_threshold_m: float = 0.5,
) -> str:
    """
    Infer a coarse intention from tracker predictions.

    Supported compact codes:
    - S  : stationary
    - KC : lane keeping + constant speed
    - KA : lane keeping + accelerating
    - KB : lane keeping + braking
    - LC / LA / LB
    - RC / RA / RB
    """

    prediction = vehicle_snapshot.get("predicted_trajectory", vehicle_snapshot.get("future_trajectory", []))
    predicted_states = list(prediction) if isinstance(prediction, Sequence) else []
    if planner is None:
        planner = AStarGlobalPlanner(lane_center_waypoints=lane_center_waypoints)

    current_x_m = float(vehicle_snapshot.get("x", 0.0))
    current_y_m = float(vehicle_snapshot.get("y", 0.0))
    current_v_mps = float(vehicle_snapshot.get("v", 0.0))
    current_psi_rad = float(vehicle_snapshot.get("psi", 0.0))

    current_context = planner.get_local_lane_context(
        x_m=float(current_x_m),
        y_m=float(current_y_m),
        heading_rad=float(current_psi_rad),
    )
    current_lane_id = int(current_context.get("lane_id", -1))

    if len(predicted_states) == 0:
        if abs(float(current_v_mps)) <= float(stationary_speed_threshold_mps):
            return "S"
        return "KC"

    final_pred = predicted_states[-1]
    future_x_m, future_y_m, future_v_mps, future_psi_rad = _prediction_state_components(
        prediction_state=final_pred,
        fallback_state=vehicle_snapshot,
    )

    future_context = planner.get_local_lane_context(
        x_m=float(future_x_m),
        y_m=float(future_y_m),
        heading_rad=float(future_psi_rad),
    )
    future_lane_id = int(future_context.get("lane_id", current_lane_id))

    displacement_m = math.hypot(
        float(future_x_m) - float(current_x_m),
        float(future_y_m) - float(current_y_m),
    )
    if (
        displacement_m <= float(stationary_position_threshold_m)
        and abs(float(current_v_mps)) <= float(stationary_speed_threshold_mps)
        and abs(float(future_v_mps)) <= float(stationary_speed_threshold_mps)
    ):
        return "S"

    speed_delta_mps = float(future_v_mps) - float(current_v_mps)
    if speed_delta_mps > float(speed_delta_threshold_mps):
        longitudinal_code = "A"
    elif speed_delta_mps < -float(speed_delta_threshold_mps):
        longitudinal_code = "B"
    else:
        longitudinal_code = "C"

    if future_lane_id != current_lane_id:
        lane_code = "L" if int(future_lane_id) > int(current_lane_id) else "R"
        return f"{lane_code}{longitudinal_code}"

    return f"K{longitudinal_code}"


def summarize_surrounding_vehicle(
    ego_snapshot: Mapping[str, object],
    vehicle_snapshot: Mapping[str, object],
    lane_center_waypoints: Sequence[Mapping[str, object]],
    lane_width_m: float = 4.0,
    planner: AStarGlobalPlanner | None = None,
) -> SurroundingVehicleSummary:
    """Build the exact surrounding-vehicle fields used in the LLM prompt."""

    if planner is None:
        planner = AStarGlobalPlanner(lane_center_waypoints=lane_center_waypoints)

    return SurroundingVehicleSummary(
        vehicle_id=str(vehicle_snapshot.get("vehicle_id", "unknown")),
        x_m=float(vehicle_snapshot.get("x", 0.0)),
        y_m=float(vehicle_snapshot.get("y", 0.0)),
        v_mps=float(vehicle_snapshot.get("v", 0.0)),
        psi_rad=float(vehicle_snapshot.get("psi", 0.0)),
        predicted_intention=str(
            infer_vehicle_intention(
                vehicle_snapshot=vehicle_snapshot,
                lane_center_waypoints=lane_center_waypoints,
                planner=planner,
            )
        ),
    )

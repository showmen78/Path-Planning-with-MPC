"""
Autonomous motion helpers for non-ego objects.

Supported modes:
- static: hold zero speed.
- lane_straight: keep current heading and speed (a=0, delta=0).
- lane_waypoint_follow: follow lane-center waypoints in the current lane.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple
import math

from .vehicle import Vehicle


def _wrap_angle(angle_rad: float) -> float:
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def _extract_lane_waypoint_target(
    vehicle_x_m: float,
    vehicle_y_m: float,
    lane_center_waypoints: Sequence[Mapping[str, object]],
) -> Tuple[float, float] | None:
    """
    Pick a forward target waypoint on the lane nearest to the vehicle lateral position.
    """

    valid_waypoints = []
    for wp in lane_center_waypoints:
        pos = wp.get("position")
        if not isinstance(pos, (list, tuple)) or len(pos) < 2:
            continue
        valid_waypoints.append(
            {
                "x": float(pos[0]),
                "y": float(pos[1]),
                "lane_id": int(wp.get("lane_id", -1)),
                "next": wp.get("next"),
            }
        )

    if len(valid_waypoints) == 0:
        return None

    # Determine lane from the nearest waypoint in full 2D space. Using only the
    # lateral y-distance works on straight roads but can pick the wrong lane on
    # curved roads when another lane elsewhere in x happens to share a similar y.
    nearest_waypoint = min(
        valid_waypoints,
        key=lambda wp: math.hypot(float(wp["x"]) - float(vehicle_x_m), float(wp["y"]) - float(vehicle_y_m)),
    )
    lane_id = int(nearest_waypoint["lane_id"])

    lane_wps = [wp for wp in valid_waypoints if int(wp["lane_id"]) == lane_id]
    if len(lane_wps) == 0:
        lane_wps = valid_waypoints

    # Prefer forward waypoint; fallback to nearest if none is forward.
    forward_wps = [wp for wp in lane_wps if float(wp["x"]) >= float(vehicle_x_m) - 1e-6]
    candidates = forward_wps if len(forward_wps) > 0 else lane_wps

    nearest_wp = min(candidates, key=lambda wp: math.hypot(float(wp["x"]) - float(vehicle_x_m), float(wp["y"]) - float(vehicle_y_m)))

    next_raw = nearest_wp.get("next")
    if isinstance(next_raw, (list, tuple)) and len(next_raw) >= 2:
        return float(next_raw[0]), float(next_raw[1])

    return float(nearest_wp["x"]), float(nearest_wp["y"])


def compute_non_ego_control(
    vehicle: Vehicle,
    vehicle_cfg: Mapping[str, Any],
    defaults_cfg: Mapping[str, Any] | None = None,
    lane_center_waypoints: Sequence[Mapping[str, object]] | None = None,
) -> Tuple[float, float]:
    """
    Compute acceleration and steering commands for non-ego objects.

    Output:
        (acceleration_mps2, steering_angle_rad)
    """

    defaults_cfg = dict(defaults_cfg or {})
    mode = str(vehicle_cfg.get("motion_mode", defaults_cfg.get("mode", "lane_straight"))).strip().lower()

    if mode == "static":
        vehicle.current_state[2] = 0.0
        return 0.0, 0.0

    if mode != "lane_waypoint_follow":
        # lane_straight (or unknown mode fallback)
        return 0.0, 0.0

    # Waypoint-follow controller.
    if lane_center_waypoints is None or len(lane_center_waypoints) == 0:
        return 0.0, 0.0

    x_m, y_m, v_mps, psi_rad = [float(value) for value in vehicle.current_state]
    target_xy = _extract_lane_waypoint_target(
        vehicle_x_m=x_m,
        vehicle_y_m=y_m,
        lane_center_waypoints=lane_center_waypoints,
    )
    if target_xy is None:
        return 0.0, 0.0

    target_x_m, target_y_m = float(target_xy[0]), float(target_xy[1])
    desired_heading_rad = math.atan2(target_y_m - y_m, target_x_m - x_m)
    heading_error_rad = _wrap_angle(desired_heading_rad - psi_rad)

    # Steering P-control.
    steer_kp = float(vehicle_cfg.get("steer_kp", 1.8))
    steer_cmd = max(-vehicle.max_steer_rad, min(vehicle.max_steer_rad, steer_kp * heading_error_rad))

    # Speed hold P-control around desired speed.
    desired_speed_mps = float(vehicle_cfg.get("desired_speed_mps", v_mps))
    accel_kp = float(vehicle_cfg.get("speed_kp", 1.5))
    accel_raw = accel_kp * (desired_speed_mps - v_mps)
    accel_cmd = max(-vehicle.max_acceleration_mps2, min(vehicle.max_acceleration_mps2, accel_raw))

    return float(accel_cmd), float(steer_cmd)

"""
Temporary destination ("blue dot") for MPC.

The global route only provides longitudinal progress and junction branch
selection. The behavior planner owns lane selection:

  **LANE_FOLLOW**         Hold the current selected lane
  **LANE_CHANGE_LEFT**    Shift the blue dot to the selected left lane
  **LANE_CHANGE_RIGHT**   Shift the blue dot to the selected right lane
  **REROUTE**             Let the new global route choose the lane

Mode is checked from the **previous blue-dot position**. When the next
macro maneuver is ``Left Turn`` or ``Right Turn`` and the blue dot is
within the intersection threshold, the mode is ``INTERSECTION``;
otherwise it is ``NORMAL``.

Key functions
-------------
* ``compute_temp_destination``  — single destination point for MPC
* ``build_reference_samples``   — horizon-length reference trajectory
* ``compute_ego_lane_offset``   — lateral offset from lane centre
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .planner import normalize_behavior_decision, normalize_macro_maneuver
from utility.carla_lane_graph import (
    canonical_lane_id_for_waypoint,
    canonical_lane_waypoint_for_lane_id,
)

# -------------------------------------------------------------------- #
# Constants                                                              #
# -------------------------------------------------------------------- #
DEFAULT_STEP_M: float = 2.0
INTERSECTION_THRESHOLD_M: float = 30.0


# -------------------------------------------------------------------- #
# Route geometry helpers                                                 #
# -------------------------------------------------------------------- #
_cached_route_id: int | None = None
_cached_cum_dists: List[float] = []


def _route_cum_dists(
    route_points: Sequence[Sequence[float]],
) -> List[float]:
    """Cumulative arc-length along the route polyline (cached by id)."""
    global _cached_route_id, _cached_cum_dists
    rid = id(route_points)
    if rid == _cached_route_id and len(_cached_cum_dists) == len(route_points):
        return _cached_cum_dists
    dists: List[float] = [0.0]
    for i in range(1, len(route_points)):
        d = math.hypot(
            float(route_points[i][0]) - float(route_points[i - 1][0]),
            float(route_points[i][1]) - float(route_points[i - 1][1]),
        )
        dists.append(dists[-1] + d)
    _cached_route_id = rid
    _cached_cum_dists = dists
    return dists


def project_ego_to_route(
    ego_x: float,
    ego_y: float,
    route_points: Sequence[Sequence[float]],
    cum_dists: Sequence[float],
) -> float:
    """
    Project ``(ego_x, ego_y)`` onto the route polyline.

    Returns the arc-length distance along the route at the projected
    point.
    """
    if len(route_points) < 2:
        return 0.0

    best_dist_sq = float("inf")
    best_arc = 0.0

    for i in range(len(route_points) - 1):
        ax = float(route_points[i][0])
        ay = float(route_points[i][1])
        bx = float(route_points[i + 1][0])
        by = float(route_points[i + 1][1])

        abx, aby = bx - ax, by - ay
        ab_sq = abx * abx + aby * aby

        if ab_sq < 1e-12:
            t = 0.0
        else:
            t = max(
                0.0,
                min(
                    1.0,
                    ((ego_x - ax) * abx + (ego_y - ay) * aby) / ab_sq,
                ),
            )

        px = ax + t * abx
        py = ay + t * aby
        d_sq = (ego_x - px) ** 2 + (ego_y - py) ** 2

        if d_sq < best_dist_sq:
            best_dist_sq = d_sq
            seg_len = math.sqrt(ab_sq)
            best_arc = float(cum_dists[i]) + t * seg_len

    return best_arc


def get_lookahead_route_point(
    route_points: Sequence[Sequence[float]],
    cum_dists: Sequence[float],
    ego_arc: float,
    lookahead_m: float,
) -> Tuple[float, float]:
    """
    Interpolate the route at ``ego_arc + lookahead_m``.

    Returns ``(x, y)``.
    """
    target = ego_arc + max(0.0, float(lookahead_m))
    total = float(cum_dists[-1]) if cum_dists else 0.0

    if target >= total:
        return float(route_points[-1][0]), float(route_points[-1][1])

    for i in range(len(cum_dists) - 1):
        if float(cum_dists[i + 1]) >= target:
            seg = max(1e-9, float(cum_dists[i + 1]) - float(cum_dists[i]))
            t = (target - float(cum_dists[i])) / seg
            x = float(route_points[i][0]) + t * (
                float(route_points[i + 1][0]) - float(route_points[i][0])
            )
            y = float(route_points[i][1]) + t * (
                float(route_points[i + 1][1]) - float(route_points[i][1])
            )
            return x, y

    return float(route_points[-1][0]), float(route_points[-1][1])


# -------------------------------------------------------------------- #
# CARLA helpers                                                          #
# -------------------------------------------------------------------- #
def _normalize_angle_deg(angle_deg: float) -> float:
    a = float(angle_deg) % 360.0
    if a > 180.0:
        a -= 360.0
    return a


def compute_distance_to_intersection_from_wp(
    wp: Any,
    step_m: float = DEFAULT_STEP_M,
    max_walk_m: float = 100.0,
) -> float:
    """
    Walk forward from *wp* via ``wp.next()`` until ``is_junction`` is True.

    Returns the distance in metres (0 if *wp* is already in a junction,
    ``inf`` if no junction is found within *max_walk_m*).
    """
    if wp.is_junction:
        return 0.0

    current = wp
    cumulative = 0.0
    while cumulative < max_walk_m:
        candidates = current.next(step_m)
        if not candidates:
            break
        current_yaw = current.transform.rotation.yaw
        current = min(
            candidates,
            key=lambda c: abs(
                _normalize_angle_deg(c.transform.rotation.yaw - current_yaw)
            ),
        )
        cumulative += step_m
        if current.is_junction:
            return cumulative

    return float("inf")


def _walk_forward(
    wp: Any,
    distance_m: float,
    step_m: float,
    route_points: Sequence[Sequence[float]] | None = None,
    cum_dists: Sequence[float] | None = None,
    maneuver: str | None = None,
) -> Any:
    """Walk *wp* forward by *distance_m* using ``wp.next()``.

    When *route_points* / *cum_dists* are provided, junction branches
    are resolved by picking the candidate closest to the route's
    lookahead point (route-guided).  Otherwise the straightest
    successor is picked (straight-ahead).
    """
    cumulative = 0.0
    while cumulative < distance_m:
        candidates = wp.next(step_m)
        if not candidates:
            break
        if len(candidates) == 1:
            wp = candidates[0]
        elif maneuver is not None:
            current_yaw = wp.transform.rotation.yaw
            maneuver_name = str(maneuver).strip().upper()
            if "LEFT" in maneuver_name:
                wp = max(
                    candidates,
                    key=lambda c: _normalize_angle_deg(
                        c.transform.rotation.yaw - current_yaw
                    ),
                )
            elif "RIGHT" in maneuver_name:
                wp = min(
                    candidates,
                    key=lambda c: _normalize_angle_deg(
                        c.transform.rotation.yaw - current_yaw
                    ),
                )
            else:
                wp = min(
                    candidates,
                    key=lambda c: abs(
                        _normalize_angle_deg(c.transform.rotation.yaw - current_yaw)
                    ),
                )
        elif route_points is not None and cum_dists is not None:
            # Route-guided branch selection
            wp_x = float(wp.transform.location.x)
            wp_y = float(wp.transform.location.y)
            arc = project_ego_to_route(wp_x, wp_y, route_points, cum_dists)
            look_x, look_y = get_lookahead_route_point(
                route_points, cum_dists, arc, step_m * 3,
            )
            wp = min(
                candidates,
                key=lambda c: (
                    (float(c.transform.location.x) - look_x) ** 2
                    + (float(c.transform.location.y) - look_y) ** 2
                ),
            )
        else:
            # Straightest successor
            current_yaw = wp.transform.rotation.yaw
            wp = min(
                candidates,
                key=lambda c: abs(
                    _normalize_angle_deg(c.transform.rotation.yaw - current_yaw)
                ),
            )
        cumulative += step_m
    return wp


def move_to_lane(carla: Any, wp: Any, target_lane_id: int) -> Any:
    """
    Move *wp* laterally to the lane whose project lane id is
    *target_lane_id*.

    Returns *wp* unchanged when inside a junction (lateral navigation is
    unreliable there) or when the target lane cannot be reached.
    """
    del carla
    if wp is None or bool(getattr(wp, "is_junction", False)):
        return wp
    return canonical_lane_waypoint_for_lane_id(wp, int(target_lane_id))


def _internal_lane_id(carla: Any, wp: Any) -> int:
    """Return the project lane id for *wp*."""
    del carla
    return int(canonical_lane_id_for_waypoint(wp))


# -------------------------------------------------------------------- #
# Mode constants (returned as 6th element of destination list)           #
# -------------------------------------------------------------------- #
MODE_NORMAL: float = 0.0
MODE_INTERSECTION: float = 1.0


def _apply_mode_override(
    is_intersection: bool,
    mode_override: str | float | None,
) -> bool:
    if mode_override is None:
        return bool(is_intersection)
    if isinstance(mode_override, (int, float)):
        return float(mode_override) > 0.5
    override_name = str(mode_override).strip().upper()
    if override_name == "INTERSECTION":
        return True
    if override_name == "NORMAL":
        return False
    return bool(is_intersection)


# -------------------------------------------------------------------- #
# Mode determination from blue-dot distance and next maneuver             #
# -------------------------------------------------------------------- #
def _requires_turn_intersection_mode(next_macro_maneuver: str | None) -> bool:
    maneuver_name = normalize_macro_maneuver(next_macro_maneuver)
    return maneuver_name in {"left", "right"}


def _determine_mode(
    ref_wp: Any,
    ego_wp: Any,
    step_m: float,
    intersection_threshold_m: float,
    prev_mode: float | None,
    prev_road_id: int | None,
    next_macro_maneuver: str | None = None,
    prev_entered_intersection: bool = False,
) -> Tuple[bool, int, bool]:
    """Determine NORMAL vs INTERSECTION from distance and next maneuver.

    Once INTERSECTION mode starts, it stays latched until the ego has
    entered and then exited the junction.

    Returns ``(is_intersection, road_id, entered_intersection)``.
    """
    del prev_road_id

    road_id = int(ref_wp.road_id)
    ego_in_junction = bool(getattr(ego_wp, "is_junction", False))
    was_intersection = prev_mode is not None and float(prev_mode) > 0.5
    entered_intersection = bool(prev_entered_intersection) or bool(ego_in_junction)

    if was_intersection:
        if entered_intersection:
            if ego_in_junction:
                return True, road_id, True
            return False, road_id, False
        return True, road_id, False

    if not _requires_turn_intersection_mode(next_macro_maneuver):
        return False, road_id, False
    dist = compute_distance_to_intersection_from_wp(
        ref_wp, step_m,
        max_walk_m=max(float(intersection_threshold_m) + 20.0, 100.0),
    )
    return (dist <= float(intersection_threshold_m)), road_id, entered_intersection


# -------------------------------------------------------------------- #
# Shared start-wp helper                                                 #
# -------------------------------------------------------------------- #
def _start_wp_for_decision(
    carla: Any, ego_wp: Any, decision: str, target_lane_id: int,
) -> Any:
    """Return the waypoint to walk forward from.

    The behavior planner owns lane choice. For every non-reroute motion
    decision, align the rolling target to the planner-selected lane.
    """
    normalized_decision = normalize_behavior_decision(decision)
    if normalized_decision != "reroute":
        return move_to_lane(carla, ego_wp, int(target_lane_id))
    return ego_wp


def _follow_route_lane_for_decision(
    decision: str,
    explicit_follow_global_route_lane: bool | None = None,
) -> bool:
    normalized_decision = str(normalize_behavior_decision(decision))
    if normalized_decision in {"lane_change_left", "lane_change_right"}:
        return False
    if explicit_follow_global_route_lane is not None:
        return bool(explicit_follow_global_route_lane)
    return normalized_decision == "reroute"


def _normalized_stop_target_state(
    stop_target_state: Sequence[float] | Mapping[str, object] | None,
) -> List[float] | None:
    if stop_target_state is None:
        return None
    if isinstance(stop_target_state, Mapping):
        try:
            return [
                float(stop_target_state.get("x_m", 0.0)),
                float(stop_target_state.get("y_m", 0.0)),
                0.0,
                float(stop_target_state.get("heading_rad", 0.0)),
                float(stop_target_state.get("lane_id", 0)),
                MODE_INTERSECTION,
                float(stop_target_state.get("road_id", -1)),
                0.0,
            ]
        except Exception:
            return None
    if not isinstance(stop_target_state, Sequence) or len(stop_target_state) < 5:
        return None
    normalized_state = [float(value) for value in list(stop_target_state[:8])]
    while len(normalized_state) < 8:
        if len(normalized_state) == 5:
            normalized_state.append(MODE_INTERSECTION)
        elif len(normalized_state) == 6:
            normalized_state.append(-1.0)
        else:
            normalized_state.append(0.0)
    normalized_state[2] = 0.0
    normalized_state[5] = MODE_INTERSECTION
    normalized_state[7] = 0.0
    return normalized_state


def _build_stop_reference_samples(
    world_map: Any,
    carla: Any,
    ego_transform: Any,
    stop_target_state: Sequence[float] | Mapping[str, object],
    global_route_points: Sequence[Sequence[float]],
    horizon_steps: int,
    step_distance_m: float,
) -> List[Dict[str, float]]:
    normalized_stop_target_state = _normalized_stop_target_state(stop_target_state)
    if normalized_stop_target_state is None:
        return []

    stop_x_m = float(normalized_stop_target_state[0])
    stop_y_m = float(normalized_stop_target_state[1])
    stop_heading_rad = float(normalized_stop_target_state[3])
    stop_lane_id = int(round(float(normalized_stop_target_state[4])))

    route_points_valid = (
        global_route_points
        if global_route_points is not None and len(global_route_points) >= 2
        else None
    )
    if route_points_valid is None:
        return [
            {
                "x_ref_m": float(stop_x_m),
                "y_ref_m": float(stop_y_m),
                "heading_rad": float(stop_heading_rad),
                "lane_id": int(stop_lane_id),
            }
            for _ in range(max(1, int(horizon_steps)))
        ]

    ego_x_m = float(ego_transform.location.x)
    ego_y_m = float(ego_transform.location.y)
    route_cum_dists = _route_cum_dists(route_points_valid)
    ego_arc_m = project_ego_to_route(
        ego_x=float(ego_x_m),
        ego_y=float(ego_y_m),
        route_points=route_points_valid,
        cum_dists=route_cum_dists,
    )
    stop_arc_m = project_ego_to_route(
        ego_x=float(stop_x_m),
        ego_y=float(stop_y_m),
        route_points=route_points_valid,
        cum_dists=route_cum_dists,
    )
    if float(stop_arc_m) <= float(ego_arc_m) + 1.0e-3:
        return [
            {
                "x_ref_m": float(stop_x_m),
                "y_ref_m": float(stop_y_m),
                "heading_rad": float(stop_heading_rad),
                "lane_id": int(stop_lane_id),
            }
            for _ in range(max(1, int(horizon_steps)))
        ]

    z_m = float(getattr(ego_transform.location, "z", 0.0))
    n = max(1, int(horizon_steps))
    sd = max(0.25, float(step_distance_m))
    samples: List[Dict[str, float]] = []

    for stage_idx in range(n):
        target_arc_m = min(
            float(stop_arc_m),
            float(ego_arc_m) + float(stage_idx) * float(sd),
        )
        if float(target_arc_m) >= float(stop_arc_m) - 1.0e-6:
            samples.append(
                {
                    "x_ref_m": float(stop_x_m),
                    "y_ref_m": float(stop_y_m),
                    "heading_rad": float(stop_heading_rad),
                    "lane_id": int(stop_lane_id),
                }
            )
            continue

        sample_x_m, sample_y_m = get_lookahead_route_point(
            route_points=route_points_valid,
            cum_dists=route_cum_dists,
            ego_arc=float(ego_arc_m),
            lookahead_m=float(target_arc_m) - float(ego_arc_m),
        )
        sample_wp = world_map.get_waypoint(
            carla.Location(
                x=float(sample_x_m),
                y=float(sample_y_m),
                z=float(z_m),
            ),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if sample_wp is None:
            next_x_m, next_y_m = get_lookahead_route_point(
                route_points=route_points_valid,
                cum_dists=route_cum_dists,
                ego_arc=float(ego_arc_m),
                lookahead_m=min(
                    float(stop_arc_m) - float(ego_arc_m),
                    float(target_arc_m) - float(ego_arc_m) + float(sd),
                ),
            )
            heading_rad = math.atan2(
                float(next_y_m) - float(sample_y_m),
                float(next_x_m) - float(sample_x_m),
            )
            samples.append(
                {
                    "x_ref_m": float(sample_x_m),
                    "y_ref_m": float(sample_y_m),
                    "heading_rad": float(heading_rad),
                    "lane_id": int(stop_lane_id),
                }
            )
            continue

        samples.append(
            {
                "x_ref_m": float(sample_wp.transform.location.x),
                "y_ref_m": float(sample_wp.transform.location.y),
                "heading_rad": float(math.radians(sample_wp.transform.rotation.yaw)),
                "lane_id": _internal_lane_id(carla, sample_wp),
            }
        )

    return samples


def _should_follow_turn_branch_from_route(
    is_intersection: bool,
    next_macro_maneuver: str | None,
    decision: str,
) -> bool:
    maneuver_name = normalize_macro_maneuver(next_macro_maneuver)
    return (
        bool(is_intersection)
        and maneuver_name in {"left", "right"}
        and str(normalize_behavior_decision(decision)) == "lane_follow"
    )


def _route_waypoint_from_anchor(
    world_map: Any,
    carla: Any,
    anchor_wp: Any,
    route_points: Sequence[Sequence[float]],
    lookahead_m: float,
    fallback_wp: Any,
    target_lane_id: int | None = None,
    follow_route_lane: bool = True,
) -> Any:
    if anchor_wp is None or route_points is None or len(route_points) < 2:
        return fallback_wp

    anchor_loc = getattr(getattr(anchor_wp, "transform", None), "location", None)
    fallback_loc = getattr(getattr(fallback_wp, "transform", None), "location", None)
    if anchor_loc is None:
        return fallback_wp

    cum_dists = _route_cum_dists(route_points)
    anchor_arc = project_ego_to_route(
        ego_x=float(anchor_loc.x),
        ego_y=float(anchor_loc.y),
        route_points=route_points,
        cum_dists=cum_dists,
    )
    target_x_m, target_y_m = get_lookahead_route_point(
        route_points=route_points,
        cum_dists=cum_dists,
        ego_arc=float(anchor_arc),
        lookahead_m=float(lookahead_m),
    )
    z_m = float(getattr(anchor_loc, "z", getattr(fallback_loc, "z", 0.0)))
    route_wp = world_map.get_waypoint(
        carla.Location(
            x=float(target_x_m),
            y=float(target_y_m),
            z=float(z_m),
        ),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    resolved_wp = route_wp if route_wp is not None else fallback_wp
    if (
        not bool(follow_route_lane)
        and route_wp is not None
        and bool(getattr(route_wp, "is_junction", False))
        and fallback_wp is not None
    ):
        resolved_wp = fallback_wp
    if bool(follow_route_lane) or resolved_wp is None or target_lane_id is None:
        return resolved_wp
    return move_to_lane(carla, resolved_wp, int(target_lane_id))


def _build_route_reference_samples_from_anchor(
    world_map: Any,
    carla: Any,
    anchor_wp: Any,
    route_points: Sequence[Sequence[float]],
    horizon_steps: int,
    step_distance_m: float,
    fallback_lane_id: int,
    target_lane_id: int | None = None,
    follow_route_lane: bool = True,
) -> List[Dict[str, float]]:
    if anchor_wp is None or route_points is None or len(route_points) < 2:
        return []

    anchor_loc = getattr(getattr(anchor_wp, "transform", None), "location", None)
    if anchor_loc is None:
        return []

    cum_dists = _route_cum_dists(route_points)
    anchor_arc = project_ego_to_route(
        ego_x=float(anchor_loc.x),
        ego_y=float(anchor_loc.y),
        route_points=route_points,
        cum_dists=cum_dists,
    )
    z_m = float(getattr(anchor_loc, "z", 0.0))
    samples: List[Dict[str, float]] = []
    n = max(1, int(horizon_steps))
    sd = max(0.25, float(step_distance_m))

    for k in range(n):
        x_ref_m, y_ref_m = get_lookahead_route_point(
            route_points=route_points,
            cum_dists=cum_dists,
            ego_arc=float(anchor_arc),
            lookahead_m=float(k) * float(sd),
        )
        route_wp = world_map.get_waypoint(
            carla.Location(
                x=float(x_ref_m),
                y=float(y_ref_m),
                z=float(z_m),
            ),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if route_wp is not None:
            sample_wp = route_wp
            if not bool(follow_route_lane) and target_lane_id is not None:
                sample_wp = move_to_lane(carla, route_wp, int(target_lane_id))
            samples.append({
                "x_ref_m": float(sample_wp.transform.location.x),
                "y_ref_m": float(sample_wp.transform.location.y),
                "heading_rad": float(math.radians(sample_wp.transform.rotation.yaw)),
                "lane_id": (
                    int(target_lane_id)
                    if not bool(follow_route_lane) and target_lane_id is not None
                    else _internal_lane_id(carla, sample_wp)
                ),
            })
            continue

        next_x_ref_m, next_y_ref_m = get_lookahead_route_point(
            route_points=route_points,
            cum_dists=cum_dists,
            ego_arc=float(anchor_arc),
            lookahead_m=float(k + 1) * float(sd),
        )
        heading_rad = math.atan2(
            float(next_y_ref_m) - float(y_ref_m),
            float(next_x_ref_m) - float(x_ref_m),
        )
        samples.append({
            "x_ref_m": float(x_ref_m),
            "y_ref_m": float(y_ref_m),
            "heading_rad": float(heading_rad),
            "lane_id": int(target_lane_id if target_lane_id is not None else fallback_lane_id),
        })
    return samples


def _build_forward_reference_samples(
    anchor_wp: Any,
    *,
    carla: Any,
    horizon_steps: int,
    step_distance_m: float,
    route_points: Sequence[Sequence[float]] | None = None,
    fallback_lane_id: int,
    maneuver: str | None = None,
) -> List[Dict[str, float]]:
    if anchor_wp is None:
        return []

    route_points_valid = route_points if route_points is not None and len(route_points) >= 2 else None
    cum_dists = _route_cum_dists(route_points_valid) if route_points_valid is not None else None
    samples: List[Dict[str, float]] = []
    wp = anchor_wp
    n = max(1, int(horizon_steps))
    sd = max(0.25, float(step_distance_m))

    for k in range(n):
        if k > 0:
            candidates = wp.next(sd)
            if candidates:
                if len(candidates) == 1:
                    wp = candidates[0]
                elif maneuver is not None:
                    current_yaw = wp.transform.rotation.yaw
                    maneuver_name = str(maneuver).strip().upper()
                    if "LEFT" in maneuver_name:
                        wp = max(
                            candidates,
                            key=lambda c: _normalize_angle_deg(
                                c.transform.rotation.yaw - current_yaw
                            ),
                        )
                    elif "RIGHT" in maneuver_name:
                        wp = min(
                            candidates,
                            key=lambda c: _normalize_angle_deg(
                                c.transform.rotation.yaw - current_yaw
                            ),
                        )
                    else:
                        wp = min(
                            candidates,
                            key=lambda c: abs(
                                _normalize_angle_deg(
                                    c.transform.rotation.yaw - current_yaw
                                )
                            ),
                        )
                elif route_points_valid is not None and cum_dists is not None:
                    wp_x = float(wp.transform.location.x)
                    wp_y = float(wp.transform.location.y)
                    arc = project_ego_to_route(wp_x, wp_y, route_points_valid, cum_dists)
                    lx, ly = get_lookahead_route_point(route_points_valid, cum_dists, arc, sd * 3)
                    wp = min(
                        candidates,
                        key=lambda c: (
                            (float(c.transform.location.x) - lx) ** 2
                            + (float(c.transform.location.y) - ly) ** 2
                        ),
                    )
                else:
                    current_yaw = wp.transform.rotation.yaw
                    wp = min(
                        candidates,
                        key=lambda c: abs(
                            _normalize_angle_deg(
                                c.transform.rotation.yaw - current_yaw
                            )
                        ),
                    )
        samples.append({
            "x_ref_m": float(wp.transform.location.x),
            "y_ref_m": float(wp.transform.location.y),
            "heading_rad": float(math.radians(wp.transform.rotation.yaw)),
            "lane_id": _internal_lane_id(carla, wp) if wp is not None else int(fallback_lane_id),
        })
    return samples


def _interpolate_heading_rad(start_heading_rad: float, end_heading_rad: float, alpha: float) -> float:
    alpha = min(1.0, max(0.0, float(alpha)))
    start_x = math.cos(float(start_heading_rad))
    start_y = math.sin(float(start_heading_rad))
    end_x = math.cos(float(end_heading_rad))
    end_y = math.sin(float(end_heading_rad))
    blend_x = (1.0 - alpha) * start_x + alpha * end_x
    blend_y = (1.0 - alpha) * start_y + alpha * end_y
    if math.hypot(blend_x, blend_y) <= 1e-9:
        return float(start_heading_rad)
    return float(math.atan2(blend_y, blend_x))


def _blend_reference_samples(
    source_samples: Sequence[Mapping[str, object]],
    target_samples: Sequence[Mapping[str, object]],
    *,
    blend_steps: int,
) -> List[Dict[str, float]]:
    if len(target_samples) == 0:
        return [dict(sample) for sample in source_samples if isinstance(sample, Mapping)]
    if len(source_samples) == 0:
        return [dict(sample) for sample in target_samples if isinstance(sample, Mapping)]

    normalized_source = [dict(sample) for sample in source_samples if isinstance(sample, Mapping)]
    normalized_target = [dict(sample) for sample in target_samples if isinstance(sample, Mapping)]
    if len(normalized_source) == 0:
        return normalized_target
    if len(normalized_target) == 0:
        return normalized_source

    while len(normalized_source) < len(normalized_target):
        normalized_source.append(dict(normalized_source[-1]))
    while len(normalized_target) < len(normalized_source):
        normalized_target.append(dict(normalized_target[-1]))

    transition_steps = max(1, min(int(blend_steps), len(normalized_target) - 1))
    blended_samples: List[Dict[str, float]] = []

    for stage_idx in range(len(normalized_target)):
        source_sample = normalized_source[stage_idx]
        target_sample = normalized_target[stage_idx]
        if stage_idx >= transition_steps:
            blended_samples.append(dict(target_sample))
            continue

        alpha = float(stage_idx) / float(transition_steps)
        source_heading_rad = float(source_sample.get("heading_rad", 0.0))
        target_heading_rad = float(target_sample.get("heading_rad", source_heading_rad))
        blended_samples.append(
            {
                "x_ref_m": (1.0 - alpha) * float(source_sample.get("x_ref_m", 0.0))
                + alpha * float(target_sample.get("x_ref_m", 0.0)),
                "y_ref_m": (1.0 - alpha) * float(source_sample.get("y_ref_m", 0.0))
                + alpha * float(target_sample.get("y_ref_m", 0.0)),
                "heading_rad": _interpolate_heading_rad(
                    source_heading_rad,
                    target_heading_rad,
                    alpha,
                ),
                "lane_id": int(
                    target_sample.get("lane_id", source_sample.get("lane_id", 0))
                    if alpha >= 0.5
                    else source_sample.get("lane_id", target_sample.get("lane_id", 0))
                ),
            }
        )
    return blended_samples


# -------------------------------------------------------------------- #
# Public API                                                             #
# -------------------------------------------------------------------- #
def compute_temp_destination_mode(
    world_map: Any,
    carla: Any,
    ego_transform: Any,
    mode_reference_xy: Tuple[float, float] | None = None,
    prev_mode: float | None = None,
    prev_road_id: int | None = None,
    prev_entered_intersection: bool = False,
    next_macro_maneuver: str | None = None,
    mode_override: str | float | None = None,
    intersection_threshold_m: float = INTERSECTION_THRESHOLD_M,
    step_m: float = DEFAULT_STEP_M,
) -> Tuple[float, int, bool]:
    """Return the current blue-dot mode using the same latch logic as the blue dot."""
    ego_z = float(ego_transform.location.z)

    ego_wp = world_map.get_waypoint(
        ego_transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if ego_wp is None:
        return MODE_NORMAL, -1, False

    if mode_reference_xy is not None:
        ref_wp = world_map.get_waypoint(
            carla.Location(
                x=float(mode_reference_xy[0]),
                y=float(mode_reference_xy[1]),
                z=ego_z,
            ),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if ref_wp is None:
            ref_wp = ego_wp
    else:
        ref_wp = ego_wp

    is_intersection, road_id, entered_intersection = _determine_mode(
        ref_wp,
        ego_wp,
        float(step_m),
        float(intersection_threshold_m),
        prev_mode,
        prev_road_id,
        next_macro_maneuver,
        prev_entered_intersection=bool(prev_entered_intersection),
    )
    is_intersection = _apply_mode_override(
        is_intersection=is_intersection,
        mode_override=mode_override,
    )
    return (
        MODE_INTERSECTION if bool(is_intersection) else MODE_NORMAL,
        int(road_id),
        bool(entered_intersection) and bool(is_intersection),
    )


def compute_temp_destination(
    world_map: Any,
    carla: Any,
    ego_transform: Any,
    target_lane_id: int,
    decision: str,
    lookahead_m: float,
    target_v_mps: float,
    global_route_points: Sequence[Sequence[float]],
    mode_reference_xy: Tuple[float, float] | None = None,
    prev_mode: float | None = None,
    prev_road_id: int | None = None,
    prev_entered_intersection: bool = False,
    next_macro_maneuver: str | None = None,
    mode_override: str | float | None = None,
    intersection_threshold_m: float = INTERSECTION_THRESHOLD_M,
    step_m: float = DEFAULT_STEP_M,
    stop_target_state: Sequence[float] | Mapping[str, object] | None = None,
    follow_global_route_lane: bool | None = None,
) -> List[float]:
    """
    Compute the temporary destination state for MPC.

    Parameters
    ----------
    decision : str
        ``"LANE_KEEP"``, ``"LANE_CHANGE_LEFT"``, or
        ``"LANE_CHANGE_RIGHT"``. The selected lane always comes from
        ``target_lane_id``; the global route only provides longitudinal
        progress except during ``reroute``.
    mode_reference_xy : tuple | None
        ``(x, y)`` of the **previous** blue-dot position.  Mode is
        determined from this point.  Falls back to ego on first frame.
    prev_mode : float | None
        Previous frame's mode (``MODE_NORMAL`` / ``MODE_INTERSECTION``).
    prev_road_id : int | None
        Previous frame's ``road_id`` (7th element of result).  Used for
        road-based mode latch.

    Returns
    -------
    [x, y, v_target, heading_rad, lane_id, mode, road_id, entered_intersection]

    *mode* is ``0.0`` for NORMAL and ``1.0`` for INTERSECTION.
    *road_id* is the CARLA ``road_id`` of the reference waypoint.
    *entered_intersection* is ``1.0`` while the ego has entered the
    current latched intersection and has not yet exited it.
    """
    ego_x = float(ego_transform.location.x)
    ego_y = float(ego_transform.location.y)
    ego_z = float(ego_transform.location.z)
    ego_psi = float(math.radians(ego_transform.rotation.yaw))

    ego_wp = world_map.get_waypoint(
        ego_transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if ego_wp is None:
        return [ego_x, ego_y, float(target_v_mps), ego_psi, 0,
                MODE_NORMAL, -1, 0.0]

    # ---- Reference wp for mode check (previous blue-dot pos) ---- #
    if mode_reference_xy is not None:
        ref_wp = world_map.get_waypoint(
            carla.Location(
                x=float(mode_reference_xy[0]),
                y=float(mode_reference_xy[1]),
                z=ego_z,
            ),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if ref_wp is None:
            ref_wp = ego_wp
    else:
        ref_wp = ego_wp

    # ---- Mode with road latch ---- #
    is_intersection, road_id, entered_intersection = _determine_mode(
        ref_wp, ego_wp, step_m, float(intersection_threshold_m),
        prev_mode, prev_road_id, next_macro_maneuver,
        prev_entered_intersection=bool(prev_entered_intersection),
    )
    is_intersection = _apply_mode_override(
        is_intersection=is_intersection,
        mode_override=mode_override,
    )
    if not bool(is_intersection):
        entered_intersection = False

    normalized_stop_target_state = _normalized_stop_target_state(stop_target_state)
    if (
        str(normalize_behavior_decision(decision)) == "stop"
        and normalized_stop_target_state is not None
    ):
        normalized_stop_target_state[6] = float(int(round(float(normalized_stop_target_state[6]))))
        return list(normalized_stop_target_state)

    normalized_decision = normalize_behavior_decision(decision)
    current_lane_id = _internal_lane_id(carla, ego_wp)
    route_alignment_lane_id = (
        int(target_lane_id)
        if int(target_lane_id) != 0
        else int(current_lane_id)
    )
    route_points_valid = (
        global_route_points
        if global_route_points is not None and len(global_route_points) >= 2
        else None
    )
    follow_route_lane = _follow_route_lane_for_decision(
        str(normalized_decision),
        explicit_follow_global_route_lane=follow_global_route_lane,
    )

    # ---- Start wp (lane shift only on lane-change decision) ---- #
    start_wp = _start_wp_for_decision(
        carla, ego_wp, str(decision), int(target_lane_id),
    )
    if route_points_valid is not None:
        fallback_wp = _walk_forward(
            start_wp,
            float(lookahead_m),
            step_m,
            maneuver=str(next_macro_maneuver or "") if bool(is_intersection) else None,
        )
        # Blue-dot longitudinal progress must always be measured from the
        # ego pose projected onto the global route, not from the previous
        # blue-dot position. ``mode_reference_xy`` is only for mode latching.
        anchor_wp = ego_wp
        route_wp = _route_waypoint_from_anchor(
            world_map=world_map,
            carla=carla,
            anchor_wp=anchor_wp,
            route_points=route_points_valid,
            lookahead_m=float(lookahead_m),
            fallback_wp=fallback_wp,
            target_lane_id=int(route_alignment_lane_id),
            follow_route_lane=bool(follow_route_lane),
        )
        mode = MODE_INTERSECTION if bool(is_intersection) else MODE_NORMAL
        return [
            float(route_wp.transform.location.x),
            float(route_wp.transform.location.y),
            float(target_v_mps),
            float(math.radians(route_wp.transform.rotation.yaw)),
            int(route_alignment_lane_id) if not bool(follow_route_lane) else _internal_lane_id(carla, route_wp),
            mode,
            int(getattr(route_wp, "road_id", road_id)),
            1.0 if bool(entered_intersection) and float(mode) > 0.5 else 0.0,
        ]

    if bool(is_intersection):
        wp = _walk_forward(
            start_wp,
            float(lookahead_m),
            step_m,
            maneuver=str(next_macro_maneuver or ""),
        )
        mode = MODE_INTERSECTION
    else:
        wp = _walk_forward(
            start_wp,
            float(lookahead_m),
            step_m,
        )
        mode = MODE_NORMAL

    return [
        float(wp.transform.location.x),
        float(wp.transform.location.y),
        float(target_v_mps),
        float(math.radians(wp.transform.rotation.yaw)),
        _internal_lane_id(carla, wp),
        mode,
        road_id,
        1.0 if bool(entered_intersection) and float(mode) > 0.5 else 0.0,
    ]


def build_reference_samples(
    world_map: Any,
    carla: Any,
    ego_transform: Any,
    target_lane_id: int,
    decision: str,
    horizon_steps: int,
    step_distance_m: float,
    global_route_points: Sequence[Sequence[float]],
    mode_reference_xy: Tuple[float, float] | None = None,
    prev_mode: float | None = None,
    prev_road_id: int | None = None,
    prev_entered_intersection: bool = False,
    next_macro_maneuver: str | None = None,
    mode_override: str | float | None = None,
    intersection_threshold_m: float = INTERSECTION_THRESHOLD_M,
    walk_step_m: float = DEFAULT_STEP_M,
    stop_target_state: Sequence[float] | Mapping[str, object] | None = None,
    follow_global_route_lane: bool | None = None,
) -> List[Dict[str, float]]:
    """
    Build a reference trajectory for MPC's lane-centre cost.

    Returns one sample per horizon step with
    ``{"x_ref_m", "y_ref_m", "heading_rad", "lane_id"}``.
    """
    ego_x = float(ego_transform.location.x)
    ego_y = float(ego_transform.location.y)
    ego_z = float(ego_transform.location.z)
    ego_psi = float(math.radians(ego_transform.rotation.yaw))
    n = max(1, int(horizon_steps))
    sd = max(0.25, float(step_distance_m))

    ego_wp = world_map.get_waypoint(
        ego_transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if ego_wp is None:
        return [
            {
                "x_ref_m": ego_x + float(k) * sd * math.cos(ego_psi),
                "y_ref_m": ego_y + float(k) * sd * math.sin(ego_psi),
                "heading_rad": ego_psi,
                "lane_id": int(target_lane_id),
            }
            for k in range(n)
        ]

    # ---- Reference wp for mode check ---- #
    if mode_reference_xy is not None:
        ref_wp = world_map.get_waypoint(
            carla.Location(
                x=float(mode_reference_xy[0]),
                y=float(mode_reference_xy[1]),
                z=ego_z,
            ),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if ref_wp is None:
            ref_wp = ego_wp
    else:
        ref_wp = ego_wp

    is_intersection, _, _ = _determine_mode(
        ref_wp, ego_wp, walk_step_m, float(intersection_threshold_m),
        prev_mode, prev_road_id, next_macro_maneuver,
        prev_entered_intersection=bool(prev_entered_intersection),
    )
    is_intersection = _apply_mode_override(
        is_intersection=is_intersection,
        mode_override=mode_override,
    )

    normalized_decision = normalize_behavior_decision(decision)
    if (
        str(normalized_decision) == "stop"
        and stop_target_state is not None
    ):
        stop_reference_samples = _build_stop_reference_samples(
            world_map=world_map,
            carla=carla,
            ego_transform=ego_transform,
            stop_target_state=stop_target_state,
            global_route_points=global_route_points,
            horizon_steps=n,
            step_distance_m=sd,
        )
        if len(stop_reference_samples) > 0:
            return stop_reference_samples
    current_lane_id = _internal_lane_id(carla, ego_wp)
    route_alignment_lane_id = (
        int(target_lane_id)
        if int(target_lane_id) != 0
        else int(current_lane_id)
    )
    follow_route_lane = _follow_route_lane_for_decision(
        str(normalized_decision),
        explicit_follow_global_route_lane=follow_global_route_lane,
    )

    route_points_valid = (
        global_route_points
        if global_route_points is not None and len(global_route_points) >= 2
        else None
    )
    if route_points_valid is not None:
        # Reference samples should originate from the ego's current route
        # progress; the previous blue-dot position is only used to decide mode.
        anchor_wp = ego_wp
        if (
            int(route_alignment_lane_id) != int(current_lane_id)
            and normalized_decision in ("lane_change_left", "lane_change_right", "reroute")
        ):
            source_route_samples = _build_route_reference_samples_from_anchor(
                world_map=world_map,
                carla=carla,
                anchor_wp=anchor_wp,
                route_points=route_points_valid,
                horizon_steps=n,
                step_distance_m=sd,
                fallback_lane_id=int(current_lane_id),
                target_lane_id=int(current_lane_id),
                follow_route_lane=False,
            )
            target_route_samples = _build_route_reference_samples_from_anchor(
                world_map=world_map,
                carla=carla,
                anchor_wp=anchor_wp,
                route_points=route_points_valid,
                horizon_steps=n,
                step_distance_m=sd,
                fallback_lane_id=int(route_alignment_lane_id),
                target_lane_id=int(route_alignment_lane_id),
                follow_route_lane=bool(follow_route_lane),
            )
            if len(source_route_samples) > 0 and len(target_route_samples) > 0:
                blend_steps = max(2, int(math.ceil(12.0 / max(0.25, sd))))
                return _blend_reference_samples(
                    source_samples=source_route_samples,
                    target_samples=target_route_samples,
                    blend_steps=blend_steps,
                )
        route_samples = _build_route_reference_samples_from_anchor(
            world_map=world_map,
            carla=carla,
            anchor_wp=anchor_wp,
            route_points=route_points_valid,
            horizon_steps=n,
            step_distance_m=sd,
            fallback_lane_id=int(route_alignment_lane_id),
            target_lane_id=int(route_alignment_lane_id),
            follow_route_lane=bool(follow_route_lane),
        )
        if len(route_samples) > 0:
            return route_samples

    start_wp = _start_wp_for_decision(
        carla, ego_wp, str(normalized_decision), int(target_lane_id),
    )
    walk_maneuver = str(next_macro_maneuver or "") if bool(is_intersection) else None
    source_samples = _build_forward_reference_samples(
        ego_wp,
        carla=carla,
        horizon_steps=n,
        step_distance_m=sd,
        route_points=None,
        fallback_lane_id=int(current_lane_id),
        maneuver=walk_maneuver,
    )
    target_samples = _build_forward_reference_samples(
        start_wp,
        carla=carla,
        horizon_steps=n,
        step_distance_m=sd,
        route_points=None,
        fallback_lane_id=int(target_lane_id),
        maneuver=walk_maneuver,
    )

    if (
        int(target_lane_id) != int(current_lane_id)
        and normalized_decision in ("lane_change_left", "lane_change_right", "reroute")
    ):
        blend_steps = max(2, int(math.ceil(12.0 / max(0.25, sd))))
        return _blend_reference_samples(
            source_samples=source_samples,
            target_samples=target_samples,
            blend_steps=blend_steps,
        )

    return target_samples if len(target_samples) > 0 else source_samples


def compute_ego_lane_offset(
    world_map: Any,
    carla: Any,
    ego_transform: Any,
) -> Dict[str, float]:
    """
    Lateral offset and heading error relative to lane centre.

    Returns ``{"lateral_offset_m", "heading_error_rad", "lane_id"}``.
    """
    ego_wp = world_map.get_waypoint(
        ego_transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if ego_wp is None:
        return {"lateral_offset_m": 0.0, "heading_error_rad": 0.0,
                "lane_id": 0}

    wp_loc = ego_wp.transform.location
    ego_loc = ego_transform.location
    dx = float(ego_loc.x) - float(wp_loc.x)
    dy = float(ego_loc.y) - float(wp_loc.y)
    lane_yaw = math.radians(float(ego_wp.transform.rotation.yaw))
    lateral = -math.sin(lane_yaw) * dx + math.cos(lane_yaw) * dy

    ego_yaw = math.radians(float(ego_transform.rotation.yaw))
    heading_error = math.atan2(
        math.sin(ego_yaw - lane_yaw),
        math.cos(ego_yaw - lane_yaw),
    )

    return {
        "lateral_offset_m": float(lateral),
        "heading_error_rad": float(heading_error),
        "lane_id": _internal_lane_id(carla, ego_wp),
    }

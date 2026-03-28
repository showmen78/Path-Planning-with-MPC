"""
Temporary destination ("blue dot") for MPC.

Two modes determined by the **blue dot's** distance to the next
intersection (not the ego's) plus the next global-route maneuver:

  **NORMAL**        (far from intersection)
      Walk forward from ego waypoint using ``wp.next()``, picking the
      straightest successor.  Lane shifts happen **only** when the
      behaviour-planner decision is ``LANE_CHANGE_LEFT`` or
      ``LANE_CHANGE_RIGHT``.  ``LANE_KEEP`` walks straight ahead in
      ego's current lane.  The global route is NOT used.

  **INTERSECTION**  (near / inside intersection)
      Walk forward from ego waypoint using ``wp.next()``, but at
      junction branches pick the successor closest to the global route
      direction.  Lane shifts still require a planner decision.

In **both** modes the blue dot never changes lane on its own — the
behaviour planner must issue ``LANE_CHANGE_LEFT`` or
``LANE_CHANGE_RIGHT``.

Mode is checked from the **previous blue-dot position**.  When the next
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
from typing import Any, Dict, List, Sequence, Tuple

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
    Move *wp* laterally to the lane whose **internal** id is
    *target_lane_id* (1 = rightmost driving lane, N = leftmost).

    Returns *wp* unchanged when inside a junction (lateral navigation is
    unreliable there) or when the target lane cannot be reached.
    """
    if wp.is_junction:
        return wp

    target = int(target_lane_id)

    # Find the rightmost driving lane in the same direction.
    rightmost = wp
    while True:
        right = rightmost.get_right_lane()
        if right is None or right.lane_type != carla.LaneType.Driving:
            break
        if right.lane_id * wp.lane_id < 0:
            break
        rightmost = right

    # Walk left, assigning internal IDs 1, 2, 3, …
    current = rightmost
    lid = 1
    if lid == target:
        return current

    while True:
        left = current.get_left_lane()
        if left is None or left.lane_type != carla.LaneType.Driving:
            break
        if left.lane_id * wp.lane_id < 0:
            break
        current = left
        lid += 1
        if lid == target:
            return current

    return wp  # target unreachable — stay on original lane


def _internal_lane_id(carla: Any, wp: Any) -> int:
    """Return the internal lane ID (1 = rightmost) for *wp*."""
    if wp.is_junction:
        return max(1, abs(int(wp.lane_id)))

    rightmost = wp
    while True:
        right = rightmost.get_right_lane()
        if right is None or right.lane_type != carla.LaneType.Driving:
            break
        if right.lane_id * wp.lane_id < 0:
            break
        rightmost = right

    current = rightmost
    lid = 1
    if current.road_id == wp.road_id and current.lane_id == wp.lane_id:
        return lid

    while True:
        left = current.get_left_lane()
        if left is None or left.lane_type != carla.LaneType.Driving:
            break
        if left.lane_id * wp.lane_id < 0:
            break
        current = left
        lid += 1
        if current.road_id == wp.road_id and current.lane_id == wp.lane_id:
            return lid

    return max(1, abs(int(wp.lane_id)))


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
    maneuver_name = str(next_macro_maneuver or "").strip().upper()
    return maneuver_name in {"LEFT TURN", "RIGHT TURN"}


def _determine_mode(
    ref_wp: Any,
    step_m: float,
    intersection_threshold_m: float,
    prev_mode: float | None,
    prev_road_id: int | None,
    next_macro_maneuver: str | None = None,
) -> Tuple[bool, int]:
    """Determine NORMAL vs INTERSECTION from distance and next maneuver.

    Returns ``(is_intersection, road_id)``.
    """
    del prev_mode
    del prev_road_id

    road_id = int(ref_wp.road_id)
    if not _requires_turn_intersection_mode(next_macro_maneuver):
        return False, road_id
    dist = compute_distance_to_intersection_from_wp(
        ref_wp, step_m,
        max_walk_m=max(float(intersection_threshold_m) + 20.0, 100.0),
    )
    return (dist <= float(intersection_threshold_m)), road_id


# -------------------------------------------------------------------- #
# Shared start-wp helper                                                 #
# -------------------------------------------------------------------- #
def _start_wp_for_decision(
    carla: Any, ego_wp: Any, decision: str, target_lane_id: int,
) -> Any:
    """Return the waypoint to walk forward from.

    LANE_CHANGE_LEFT / RIGHT -> shift to *target_lane_id* via
    ``move_to_lane``.  Everything else (LANE_KEEP) -> *ego_wp*
    unchanged.
    """
    if decision in ("LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT"):
        return move_to_lane(carla, ego_wp, int(target_lane_id))
    return ego_wp


def _should_follow_turn_branch_from_route(
    is_intersection: bool,
    next_macro_maneuver: str | None,
    decision: str,
) -> bool:
    maneuver_name = str(next_macro_maneuver or "").strip().upper()
    return (
        bool(is_intersection)
        and maneuver_name in {"LEFT TURN", "RIGHT TURN", "LANE FOLLOW"}
        and str(decision).strip().upper() == "LANE_KEEP"
    )


def _route_waypoint_from_anchor(
    world_map: Any,
    carla: Any,
    anchor_wp: Any,
    route_points: Sequence[Sequence[float]],
    lookahead_m: float,
    fallback_wp: Any,
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
    return route_wp if route_wp is not None else fallback_wp


def _build_route_reference_samples_from_anchor(
    world_map: Any,
    carla: Any,
    anchor_wp: Any,
    route_points: Sequence[Sequence[float]],
    horizon_steps: int,
    step_distance_m: float,
    fallback_lane_id: int,
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
        progress_m = float(anchor_arc) + float(k) * float(sd)
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
            samples.append({
                "x_ref_m": float(route_wp.transform.location.x),
                "y_ref_m": float(route_wp.transform.location.y),
                "heading_rad": float(math.radians(route_wp.transform.rotation.yaw)),
                "lane_id": _internal_lane_id(carla, route_wp),
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
            "lane_id": int(fallback_lane_id),
        })
    return samples


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
    next_macro_maneuver: str | None = None,
    mode_override: str | float | None = None,
    intersection_threshold_m: float = INTERSECTION_THRESHOLD_M,
    step_m: float = DEFAULT_STEP_M,
) -> Tuple[float, int]:
    """Return the current blue-dot mode using the same latch logic as the blue dot."""
    ego_z = float(ego_transform.location.z)

    ego_wp = world_map.get_waypoint(
        ego_transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if ego_wp is None:
        return MODE_NORMAL, -1

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

    is_intersection, road_id = _determine_mode(
        ref_wp,
        float(step_m),
        float(intersection_threshold_m),
        prev_mode,
        prev_road_id,
        next_macro_maneuver,
    )
    is_intersection = _apply_mode_override(
        is_intersection=is_intersection,
        mode_override=mode_override,
    )
    return (
        MODE_INTERSECTION if bool(is_intersection) else MODE_NORMAL,
        int(road_id),
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
    next_macro_maneuver: str | None = None,
    mode_override: str | float | None = None,
    intersection_threshold_m: float = INTERSECTION_THRESHOLD_M,
    step_m: float = DEFAULT_STEP_M,
) -> List[float]:
    """
    Compute the temporary destination state for MPC.

    Parameters
    ----------
    decision : str
        ``"LANE_KEEP"``, ``"LANE_CHANGE_LEFT"``, or
        ``"LANE_CHANGE_RIGHT"``.  In **both** modes, lane shifts only
        happen when the decision is a lane-change.
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
    [x, y, v_target, heading_rad, lane_id, mode, road_id]

    *mode* is ``0.0`` for NORMAL and ``1.0`` for INTERSECTION.
    *road_id* is the CARLA ``road_id`` of the reference waypoint (used
    for mode latch on the next call).
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
        return [ego_x, ego_y, float(target_v_mps), ego_psi, 1,
                MODE_NORMAL, -1]

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
    is_intersection, road_id = _determine_mode(
        ref_wp, step_m, float(intersection_threshold_m),
        prev_mode, prev_road_id, next_macro_maneuver,
    )
    is_intersection = _apply_mode_override(
        is_intersection=is_intersection,
        mode_override=mode_override,
    )

    # ---- Start wp (lane shift only on lane-change decision) ---- #
    start_wp = _start_wp_for_decision(
        carla, ego_wp, str(decision), int(target_lane_id),
    )
    follow_turn_branch_from_route = _should_follow_turn_branch_from_route(
        is_intersection=is_intersection,
        next_macro_maneuver=next_macro_maneuver,
        decision=str(decision),
    )

    if follow_turn_branch_from_route:
        wp = _route_waypoint_from_anchor(
            world_map=world_map,
            carla=carla,
            anchor_wp=start_wp,
            route_points=global_route_points,
            lookahead_m=float(lookahead_m),
            fallback_wp=start_wp,
        )
        return [
            float(wp.transform.location.x),
            float(wp.transform.location.y),
            float(target_v_mps),
            float(math.radians(wp.transform.rotation.yaw)),
            _internal_lane_id(carla, wp),
            MODE_INTERSECTION,
            road_id,
        ]

    if is_intersection:
        # Walk forward, route-guided at junction branches
        if global_route_points and len(global_route_points) >= 2:
            cd = _route_cum_dists(global_route_points)
            wp = _walk_forward(
                start_wp, float(lookahead_m), step_m,
                route_points=global_route_points, cum_dists=cd,
            )
        else:
            wp = _walk_forward(start_wp, float(lookahead_m), step_m)
        mode = MODE_INTERSECTION
    else:
        wp = _walk_forward(start_wp, float(lookahead_m), step_m)
        mode = MODE_NORMAL

    return [
        float(wp.transform.location.x),
        float(wp.transform.location.y),
        float(target_v_mps),
        float(math.radians(wp.transform.rotation.yaw)),
        _internal_lane_id(carla, wp),
        mode,
        road_id,
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
    next_macro_maneuver: str | None = None,
    mode_override: str | float | None = None,
    intersection_threshold_m: float = INTERSECTION_THRESHOLD_M,
    walk_step_m: float = DEFAULT_STEP_M,
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

    is_intersection, _ = _determine_mode(
        ref_wp, walk_step_m, float(intersection_threshold_m),
        prev_mode, prev_road_id, next_macro_maneuver,
    )
    is_intersection = _apply_mode_override(
        is_intersection=is_intersection,
        mode_override=mode_override,
    )

    # ---- Start wp (lane shift only on lane-change decision) ---- #
    start_wp = _start_wp_for_decision(
        carla, ego_wp, str(decision), int(target_lane_id),
    )
    follow_turn_branch_from_route = _should_follow_turn_branch_from_route(
        is_intersection=is_intersection,
        next_macro_maneuver=next_macro_maneuver,
        decision=str(decision),
    )

    if follow_turn_branch_from_route:
        route_samples = _build_route_reference_samples_from_anchor(
            world_map=world_map,
            carla=carla,
            anchor_wp=start_wp,
            route_points=global_route_points,
            horizon_steps=n,
            step_distance_m=sd,
            fallback_lane_id=int(target_lane_id),
        )
        if len(route_samples) > 0:
            return route_samples

    # Route data for guided branch selection in intersection mode
    use_route = (
        is_intersection
        and global_route_points is not None
        and len(global_route_points) >= 2
    )
    rp = global_route_points if use_route else None
    cd = _route_cum_dists(global_route_points) if use_route else None

    samples: List[Dict[str, float]] = []
    wp = start_wp
    for k in range(n):
        if k > 0:
            candidates = wp.next(sd)
            if candidates:
                if len(candidates) == 1:
                    wp = candidates[0]
                elif rp is not None and cd is not None:
                    wp_x = float(wp.transform.location.x)
                    wp_y = float(wp.transform.location.y)
                    arc = project_ego_to_route(wp_x, wp_y, rp, cd)
                    lx, ly = get_lookahead_route_point(rp, cd, arc, sd * 3)
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
            "lane_id": _internal_lane_id(carla, wp),
        })
    return samples


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
                "lane_id": 1}

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

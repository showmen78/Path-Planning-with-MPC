"""
Shared local-goal helpers for lookahead and temporary-destination placement.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence

from utility.carla_lane_graph import (
    _carla_waypoint_group_key,
    _internal_lane_id,
)


_LANE_TOPOLOGY_CACHE: dict[int, dict[str, object]] = {}


def _position_of_waypoint(waypoint: Mapping[str, object]) -> tuple[float, float] | None:
    position = waypoint.get("position", None)
    if not isinstance(position, (list, tuple)) or len(position) < 2:
        return None
    return float(position[0]), float(position[1])


def _waypoint_key(x_m: float, y_m: float) -> tuple[float, float]:
    return round(float(x_m), 3), round(float(y_m), 3)


def _wrap_angle_rad(angle_rad: float) -> float:
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def _lane_count_from_inputs(
    lane_center_waypoints: Sequence[Mapping[str, object]],
    road_cfg: Mapping[str, object] | None,
) -> int:
    road_cfg = dict(road_cfg or {})
    configured_lane_count = int(road_cfg.get("lane_count", 0))
    if configured_lane_count > 0:
        return configured_lane_count
    lane_ids = [int(item.get("lane_id", 0)) for item in lane_center_waypoints if item.get("lane_id", None) is not None]
    return max(1, max(lane_ids)) if len(lane_ids) > 0 else 1


def _distance_between_points(
    start_xy: Sequence[float],
    end_xy: Sequence[float],
) -> float:
    return math.hypot(
        float(end_xy[0]) - float(start_xy[0]),
        float(end_xy[1]) - float(start_xy[1]),
    )


def _build_lane_component_cache(
    lane_waypoints: Sequence[Mapping[str, object]],
    waypoint_by_xy: Mapping[tuple[float, float], Mapping[str, object]],
    successor_by_xy: Mapping[tuple[float, float], Mapping[str, object] | None],
) -> dict[str, object]:
    undirected_neighbors: dict[tuple[float, float], set[tuple[float, float]]] = {}
    for waypoint in lane_waypoints:
        position = _position_of_waypoint(waypoint)
        if position is None:
            continue
        waypoint_key = _waypoint_key(float(position[0]), float(position[1]))
        undirected_neighbors.setdefault(waypoint_key, set())

        successor_waypoint = successor_by_xy.get(waypoint_key, None)
        successor_position = _position_of_waypoint(successor_waypoint) if successor_waypoint is not None else None
        if successor_position is None:
            continue
        successor_key = _waypoint_key(float(successor_position[0]), float(successor_position[1]))
        if successor_key == waypoint_key:
            continue
        undirected_neighbors.setdefault(successor_key, set())
        undirected_neighbors[waypoint_key].add(successor_key)
        undirected_neighbors[successor_key].add(waypoint_key)

    component_id_by_key: dict[tuple[float, float], int] = {}
    component_waypoints_by_id: dict[int, list[Mapping[str, object]]] = {}
    component_waypoint_by_xy_by_id: dict[int, dict[tuple[float, float], Mapping[str, object]]] = {}
    component_successor_by_xy_by_id: dict[int, dict[tuple[float, float], Mapping[str, object] | None]] = {}

    visited_keys: set[tuple[float, float]] = set()
    for start_key in undirected_neighbors:
        if start_key in visited_keys:
            continue
        component_keys: list[tuple[float, float]] = []
        pending_keys = [start_key]
        while pending_keys:
            current_key = pending_keys.pop()
            if current_key in visited_keys:
                continue
            visited_keys.add(current_key)
            component_keys.append(current_key)
            for neighbor_key in undirected_neighbors.get(current_key, set()):
                if neighbor_key not in visited_keys:
                    pending_keys.append(neighbor_key)

        component_id = len(component_waypoints_by_id)
        component_waypoint_by_xy = {
            key: waypoint
            for key, waypoint in waypoint_by_xy.items()
            if key in component_keys
        }
        component_successor_by_xy: dict[tuple[float, float], Mapping[str, object] | None] = {}
        for key, successor in successor_by_xy.items():
            if key not in component_waypoint_by_xy:
                continue
            if successor is None:
                component_successor_by_xy[key] = None
                continue
            successor_position = _position_of_waypoint(successor)
            successor_key = (
                None
                if successor_position is None
                else _waypoint_key(float(successor_position[0]), float(successor_position[1]))
            )
            component_successor_by_xy[key] = (
                successor if successor_key in component_waypoint_by_xy else None
            )
        for component_key in component_keys:
            component_id_by_key[component_key] = int(component_id)
        component_waypoints_by_id[component_id] = list(component_waypoint_by_xy.values())
        component_waypoint_by_xy_by_id[component_id] = component_waypoint_by_xy
        component_successor_by_xy_by_id[component_id] = component_successor_by_xy

    return {
        "component_id_by_key": component_id_by_key,
        "component_waypoints_by_id": component_waypoints_by_id,
        "component_waypoint_by_xy_by_id": component_waypoint_by_xy_by_id,
        "component_successor_by_xy_by_id": component_successor_by_xy_by_id,
    }


def _ensure_lane_topology_cache(
    lane_center_waypoints: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    cache_key = id(lane_center_waypoints)
    cached = _LANE_TOPOLOGY_CACHE.get(cache_key, None)
    if (
        cached is None
        or cached.get("source") is not lane_center_waypoints
        or int(cached.get("size", -1)) != len(lane_center_waypoints)
    ):
        lane_data_by_id: dict[int, dict[str, object]] = {}
        internal_lane_id_by_raw_key: dict[tuple[int, int, int], int] = {}
        lane_ids_by_group: dict[tuple[int, int, str], set[int]] = {}

        for waypoint in lane_center_waypoints:
            if _position_of_waypoint(waypoint) is None:
                continue
            lane_id = int(waypoint.get("lane_id", -999999))
            lane_data_by_id.setdefault(lane_id, {"waypoints": []})
            lane_data_by_id[lane_id]["waypoints"].append(waypoint)

            carla_waypoint = waypoint.get("carla_waypoint", None)
            if carla_waypoint is None:
                continue
            raw_lane_id = int(getattr(carla_waypoint, "lane_id", 0))
            if raw_lane_id == 0:
                continue
            group_key = _carla_waypoint_group_key(carla_waypoint)
            lane_ids_by_group.setdefault(group_key, set()).add(raw_lane_id)
            internal_lane_id_by_raw_key[
                (
                    int(getattr(carla_waypoint, "road_id", 0)),
                    int(getattr(carla_waypoint, "section_id", 0)),
                    raw_lane_id,
                )
            ] = int(waypoint.get("lane_id", 0))

        for lane_data in lane_data_by_id.values():
            lane_waypoints = list(lane_data.get("waypoints", []))
            waypoint_by_xy = _waypoint_map(lane_waypoints)
            successor_by_xy = _successor_map_for_lane_waypoints(
                lane_waypoints=lane_waypoints,
                waypoint_by_xy=waypoint_by_xy,
            )
            lane_data["waypoint_by_xy"] = waypoint_by_xy
            lane_data["successor_by_xy"] = successor_by_xy
            lane_data.update(
                _build_lane_component_cache(
                    lane_waypoints=lane_waypoints,
                    waypoint_by_xy=waypoint_by_xy,
                    successor_by_xy=successor_by_xy,
                )
            )

        cached = {
            "source": lane_center_waypoints,
            "size": len(lane_center_waypoints),
            "lane_data_by_id": lane_data_by_id,
            "internal_lane_id_by_raw_key": internal_lane_id_by_raw_key,
            "lane_ids_by_group": lane_ids_by_group,
        }
        _LANE_TOPOLOGY_CACHE[cache_key] = cached
    return cached


def _carla_lane_lookup(
    lane_center_waypoints: Sequence[Mapping[str, object]],
) -> tuple[dict[tuple[int, int, int], int], dict[tuple[int, int, str], set[int]]]:
    cached = _ensure_lane_topology_cache(lane_center_waypoints=lane_center_waypoints)
    return (
        cached.get("internal_lane_id_by_raw_key", {}),
        cached.get("lane_ids_by_group", {}),
    )


def _internal_lane_id_for_carla_waypoint(
    carla_waypoint,
    internal_lane_id_by_raw_key: Mapping[tuple[int, int, int], int],
    lane_ids_by_group: Mapping[tuple[int, int, str], set[int]],
) -> int | None:
    raw_key = (
        int(getattr(carla_waypoint, "road_id", 0)),
        int(getattr(carla_waypoint, "section_id", 0)),
        int(getattr(carla_waypoint, "lane_id", 0)),
    )
    if raw_key in internal_lane_id_by_raw_key:
        return int(internal_lane_id_by_raw_key[raw_key])

    group_key = _carla_waypoint_group_key(carla_waypoint)
    if group_key not in lane_ids_by_group:
        return None

    try:
        return int(_internal_lane_id(carla_waypoint, lane_ids_by_group))
    except Exception:
        return None


def _normalized_next_maneuver_code(next_maneuver: str | None) -> str:
    lowered = str(next_maneuver or "").strip().lower()
    if lowered.startswith("l") or "left" in lowered:
        return "L"
    if lowered.startswith("r") or "right" in lowered:
        return "R"
    return "S"


def _signed_heading_delta_deg(
    from_heading_rad: float,
    to_heading_rad: float,
) -> float:
    delta_rad = math.atan2(
        math.sin(float(to_heading_rad) - float(from_heading_rad)),
        math.cos(float(to_heading_rad) - float(from_heading_rad)),
    )
    return float(math.degrees(delta_rad))


def _turn_code_from_heading_delta_deg(
    delta_deg: float,
    threshold_deg: float = 20.0,
) -> str:
    if float(delta_deg) > float(threshold_deg):
        return "L"
    if float(delta_deg) < -float(threshold_deg):
        return "R"
    return "S"


def _carla_waypoint_heading_rad(
    carla_waypoint,
    fallback_heading_rad: float,
) -> float:
    transform = getattr(carla_waypoint, "transform", None)
    rotation = getattr(transform, "rotation", None)
    if rotation is None:
        return float(fallback_heading_rad)
    return float(
        math.radians(
            float(getattr(rotation, "yaw", math.degrees(float(fallback_heading_rad))))
        )
    )


def _carla_waypoint_position(
    carla_waypoint,
) -> tuple[float, float] | None:
    transform = getattr(carla_waypoint, "transform", None)
    location = getattr(transform, "location", None)
    if location is None:
        return None
    return float(getattr(location, "x", 0.0)), float(getattr(location, "y", 0.0))


def _carla_branch_candidate_score(
    *,
    carla_waypoint,
    desired_maneuver_code: str,
    reference_heading_rad: float,
    target_lane_id: int,
    internal_lane_id_by_raw_key: Mapping[tuple[int, int, int], int],
    lane_ids_by_group: Mapping[tuple[int, int, str], set[int]],
) -> tuple[float, float, float, float]:
    candidate_heading_rad = _carla_waypoint_heading_rad(
        carla_waypoint=carla_waypoint,
        fallback_heading_rad=float(reference_heading_rad),
    )
    signed_heading_delta_deg = _signed_heading_delta_deg(
        from_heading_rad=float(reference_heading_rad),
        to_heading_rad=float(candidate_heading_rad),
    )
    candidate_internal_lane_id = _internal_lane_id_for_carla_waypoint(
        carla_waypoint=carla_waypoint,
        internal_lane_id_by_raw_key=internal_lane_id_by_raw_key,
        lane_ids_by_group=lane_ids_by_group,
    )
    lane_mismatch_penalty = (
        0.0
        if candidate_internal_lane_id is not None
        and int(candidate_internal_lane_id) == int(target_lane_id)
        else 1.0
    )

    if str(desired_maneuver_code) == "L":
        preferred_direction_penalty = 0.0 if float(signed_heading_delta_deg) > 1.0 else 1.0
        direction_alignment_score = -float(signed_heading_delta_deg)
        return (
            float(preferred_direction_penalty),
            float(direction_alignment_score),
            float(lane_mismatch_penalty),
            abs(float(signed_heading_delta_deg)),
        )

    if str(desired_maneuver_code) == "R":
        preferred_direction_penalty = 0.0 if float(signed_heading_delta_deg) < -1.0 else 1.0
        direction_alignment_score = float(signed_heading_delta_deg)
        return (
            float(preferred_direction_penalty),
            float(direction_alignment_score),
            float(lane_mismatch_penalty),
            abs(float(signed_heading_delta_deg)),
        )

    return (
        0.0 if abs(float(signed_heading_delta_deg)) <= 5.0 else 1.0,
        abs(float(signed_heading_delta_deg)),
        float(lane_mismatch_penalty),
        abs(float(signed_heading_delta_deg)),
    )


def _follow_carla_branch_for_distance(
    *,
    start_carla_waypoint,
    requested_distance_m: float,
    target_lane_id: int,
    desired_maneuver_code: str,
    internal_lane_id_by_raw_key: Mapping[tuple[int, int, int], int],
    lane_ids_by_group: Mapping[tuple[int, int, str], set[int]],
    fallback_heading_rad: float,
) -> tuple[object, int | None] | None:
    current_carla_waypoint = start_carla_waypoint
    current_heading_rad = _carla_waypoint_heading_rad(
        carla_waypoint=current_carla_waypoint,
        fallback_heading_rad=float(fallback_heading_rad),
    )
    last_internal_lane_id = _internal_lane_id_for_carla_waypoint(
        carla_waypoint=current_carla_waypoint,
        internal_lane_id_by_raw_key=internal_lane_id_by_raw_key,
        lane_ids_by_group=lane_ids_by_group,
    )
    remaining_distance_m = max(0.0, float(requested_distance_m))
    if remaining_distance_m <= 1e-6:
        return current_carla_waypoint, last_internal_lane_id

    step_distance_m = max(0.5, min(2.0, float(remaining_distance_m)))
    visited_positions = set()
    start_position = _carla_waypoint_position(current_carla_waypoint)
    if start_position is not None:
        visited_positions.add(_waypoint_key(float(start_position[0]), float(start_position[1])))

    while float(remaining_distance_m) > 1e-6:
        query_distance_m = min(float(step_distance_m), float(remaining_distance_m))
        try:
            next_candidates = [
                candidate
                for candidate in list(current_carla_waypoint.next(float(query_distance_m)))
                if int(getattr(candidate, "lane_id", 0)) != 0
            ]
        except Exception:
            next_candidates = []

        if len(next_candidates) == 0:
            break

        if len(next_candidates) == 1:
            selected_carla_waypoint = next_candidates[0]
        else:
            selected_carla_waypoint = min(
                next_candidates,
                key=lambda candidate: _carla_branch_candidate_score(
                    carla_waypoint=candidate,
                    desired_maneuver_code=str(desired_maneuver_code),
                    reference_heading_rad=float(current_heading_rad),
                    target_lane_id=int(target_lane_id),
                    internal_lane_id_by_raw_key=internal_lane_id_by_raw_key,
                    lane_ids_by_group=lane_ids_by_group,
                ),
            )

        current_position = _carla_waypoint_position(current_carla_waypoint)
        selected_position = _carla_waypoint_position(selected_carla_waypoint)
        if selected_position is None:
            break

        if current_position is None:
            traveled_distance_m = float(query_distance_m)
        else:
            traveled_distance_m = max(
                0.05,
                _distance_between_points(
                    current_position,
                    selected_position,
                ),
            )

        selected_key = _waypoint_key(float(selected_position[0]), float(selected_position[1]))
        if selected_key in visited_positions:
            current_carla_waypoint = selected_carla_waypoint
            break

        visited_positions.add(selected_key)
        current_carla_waypoint = selected_carla_waypoint
        current_heading_rad = _carla_waypoint_heading_rad(
            carla_waypoint=current_carla_waypoint,
            fallback_heading_rad=float(current_heading_rad),
        )
        selected_internal_lane_id = _internal_lane_id_for_carla_waypoint(
            carla_waypoint=current_carla_waypoint,
            internal_lane_id_by_raw_key=internal_lane_id_by_raw_key,
            lane_ids_by_group=lane_ids_by_group,
        )
        if selected_internal_lane_id is not None:
            last_internal_lane_id = int(selected_internal_lane_id)
        remaining_distance_m = max(0.0, float(remaining_distance_m) - float(traveled_distance_m))

    return current_carla_waypoint, last_internal_lane_id


def _snap_carla_target_to_internal_lane_state(
    carla_target_waypoint,
    lane_waypoints: Sequence[Mapping[str, object]],
    ego_state: Sequence[float],
) -> list[float] | None:
    carla_transform = getattr(carla_target_waypoint, "transform", None)
    if carla_transform is None:
        return None
    carla_location = getattr(carla_transform, "location", None)
    if carla_location is None:
        return None

    target_x_m = float(getattr(carla_location, "x", 0.0))
    target_y_m = float(getattr(carla_location, "y", 0.0))
    ego_x_m = float(ego_state[0])
    ego_y_m = float(ego_state[1])

    forward_candidates = []
    for waypoint in lane_waypoints:
        position = _position_of_waypoint(waypoint)
        if position is None:
            continue
        dx_m = float(position[0]) - ego_x_m
        dy_m = float(position[1]) - ego_y_m
        waypoint_heading_rad = float(waypoint.get("heading_rad", 0.0))
        forward_distance_m = (
            math.cos(float(waypoint_heading_rad)) * dx_m
            + math.sin(float(waypoint_heading_rad)) * dy_m
        )
        if forward_distance_m < -1e-6:
            continue
        forward_candidates.append(waypoint)

    snap_candidates = forward_candidates if len(forward_candidates) > 0 else list(lane_waypoints)
    if len(snap_candidates) == 0:
        return None

    snapped_waypoint = min(
        snap_candidates,
        key=lambda waypoint: math.hypot(
            float(_position_of_waypoint(waypoint)[0]) - target_x_m,
            float(_position_of_waypoint(waypoint)[1]) - target_y_m,
        ),
    )
    snapped_position = _position_of_waypoint(snapped_waypoint)
    if snapped_position is None:
        return None

    return [
        float(snapped_position[0]),
        float(snapped_position[1]),
        0.0,
        float(snapped_waypoint.get("heading_rad", float(ego_state[3]))),
    ]


def _lane_waypoints_by_lane_id(
    lane_center_waypoints: Sequence[Mapping[str, object]],
    target_lane_id: int,
) -> tuple[
    list[Mapping[str, object]],
    dict[tuple[float, float], Mapping[str, object]],
    dict[tuple[float, float], Mapping[str, object] | None],
]:
    cached = _ensure_lane_topology_cache(lane_center_waypoints=lane_center_waypoints)
    lane_data_by_id = cached.get("lane_data_by_id", {})
    lane_data = lane_data_by_id.get(int(target_lane_id), {})
    return (
        lane_data.get("waypoints", []),
        lane_data.get("waypoint_by_xy", {}),
        lane_data.get("successor_by_xy", {}),
    )


def _waypoint_map(
    waypoints: Sequence[Mapping[str, object]],
) -> Dict[tuple[float, float], Mapping[str, object]]:
    return {
        _waypoint_key(float(position[0]), float(position[1])): waypoint
        for waypoint in waypoints
        if (position := _position_of_waypoint(waypoint)) is not None
    }


def _successor_map_for_lane_waypoints(
    lane_waypoints: Sequence[Mapping[str, object]],
    waypoint_by_xy: Mapping[tuple[float, float], Mapping[str, object]],
) -> dict[tuple[float, float], Mapping[str, object] | None]:
    successor_by_xy: dict[tuple[float, float], Mapping[str, object] | None] = {}
    for waypoint in lane_waypoints:
        position = _position_of_waypoint(waypoint)
        if position is None:
            continue
        successor_by_xy[_waypoint_key(float(position[0]), float(position[1]))] = (
            _resolve_successor_waypoint_on_lane(
                current_waypoint=waypoint,
                lane_waypoints=lane_waypoints,
                waypoint_by_xy=waypoint_by_xy,
            )
        )
    return successor_by_xy


def _localized_lane_topology_near_pose(
    lane_waypoints: Sequence[Mapping[str, object]],
    waypoint_by_xy: Mapping[tuple[float, float], Mapping[str, object]],
    successor_by_xy: Mapping[tuple[float, float], Mapping[str, object] | None],
    reference_x_m: float,
    reference_y_m: float,
    reference_heading_rad: float | None = None,
    lane_component_cache: Mapping[str, object] | None = None,
) -> tuple[
    list[Mapping[str, object]],
    dict[tuple[float, float], Mapping[str, object]],
    dict[tuple[float, float], Mapping[str, object] | None],
]:
    if len(lane_waypoints) <= 1:
        return lane_waypoints, waypoint_by_xy, successor_by_xy

    def _seed_score(waypoint: Mapping[str, object]) -> tuple[float, float, float]:
        position = _position_of_waypoint(waypoint)
        if position is None:
            return (1.0e9, 1.0e9, 1.0e9)
        distance_m = math.hypot(
            float(position[0]) - float(reference_x_m),
            float(position[1]) - float(reference_y_m),
        )
        if reference_heading_rad is None:
            return (distance_m, 0.0, 0.0)
        heading_rad = float(waypoint.get("heading_rad", reference_heading_rad))
        heading_error_rad = abs(_wrap_angle_rad(float(heading_rad) - float(reference_heading_rad)))
        opposite_direction_penalty = 1.0 if math.cos(float(heading_error_rad)) < 0.0 else 0.0
        return (float(opposite_direction_penalty), float(distance_m), float(heading_error_rad))

    seed_waypoint = min(lane_waypoints, key=_seed_score)
    seed_position = _position_of_waypoint(seed_waypoint)
    if seed_position is None:
        return lane_waypoints, waypoint_by_xy, successor_by_xy

    seed_key = _waypoint_key(float(seed_position[0]), float(seed_position[1]))
    if lane_component_cache is not None:
        component_id_by_key = dict(lane_component_cache.get("component_id_by_key", {}))
        component_id = component_id_by_key.get(seed_key, None)
        if component_id is not None:
            component_waypoints_by_id = lane_component_cache.get("component_waypoints_by_id", {})
            component_waypoint_by_xy_by_id = lane_component_cache.get("component_waypoint_by_xy_by_id", {})
            component_successor_by_xy_by_id = lane_component_cache.get("component_successor_by_xy_by_id", {})
            return (
                component_waypoints_by_id.get(int(component_id), lane_waypoints),
                component_waypoint_by_xy_by_id.get(int(component_id), waypoint_by_xy),
                component_successor_by_xy_by_id.get(int(component_id), successor_by_xy),
            )

    undirected_neighbors: dict[tuple[float, float], set[tuple[float, float]]] = {}
    for waypoint in lane_waypoints:
        position = _position_of_waypoint(waypoint)
        if position is None:
            continue
        waypoint_key = _waypoint_key(float(position[0]), float(position[1]))
        undirected_neighbors.setdefault(waypoint_key, set())

        successor_waypoint = successor_by_xy.get(waypoint_key, None)
        if successor_waypoint is None:
            successor_waypoint = _resolve_successor_waypoint_on_lane(
                current_waypoint=waypoint,
                lane_waypoints=lane_waypoints,
                waypoint_by_xy=waypoint_by_xy,
            )
        successor_position = _position_of_waypoint(successor_waypoint) if successor_waypoint is not None else None
        if successor_position is None:
            continue
        successor_key = _waypoint_key(float(successor_position[0]), float(successor_position[1]))
        if successor_key == waypoint_key:
            continue
        undirected_neighbors.setdefault(successor_key, set())
        undirected_neighbors[waypoint_key].add(successor_key)
        undirected_neighbors[successor_key].add(waypoint_key)

    if seed_key not in undirected_neighbors:
        return lane_waypoints, waypoint_by_xy, successor_by_xy

    component_keys: set[tuple[float, float]] = set()
    pending_keys = [seed_key]
    while pending_keys:
        current_key = pending_keys.pop()
        if current_key in component_keys:
            continue
        component_keys.add(current_key)
        for neighbor_key in undirected_neighbors.get(current_key, set()):
            if neighbor_key not in component_keys:
                pending_keys.append(neighbor_key)

    if len(component_keys) == 0:
        return lane_waypoints, waypoint_by_xy, successor_by_xy

    component_waypoint_by_xy = {
        key: waypoint
        for key, waypoint in waypoint_by_xy.items()
        if key in component_keys
    }
    component_waypoints = list(component_waypoint_by_xy.values())
    component_successor_by_xy: dict[tuple[float, float], Mapping[str, object] | None] = {}
    for key, waypoint in component_waypoint_by_xy.items():
        successor_waypoint = successor_by_xy.get(key, None)
        successor_position = _position_of_waypoint(successor_waypoint) if successor_waypoint is not None else None
        successor_key = (
            None
            if successor_position is None
            else _waypoint_key(float(successor_position[0]), float(successor_position[1]))
        )
        component_successor_by_xy[key] = (
            successor_waypoint if successor_key in component_keys else None
        )

    return component_waypoints, component_waypoint_by_xy, component_successor_by_xy


def _resolve_successor_waypoint_on_lane(
    current_waypoint: Mapping[str, object],
    lane_waypoints: Sequence[Mapping[str, object]],
    waypoint_by_xy: Mapping[tuple[float, float], Mapping[str, object]],
    visited_keys: set[tuple[float, float]] | None = None,
    successor_by_xy: Mapping[tuple[float, float], Mapping[str, object] | None] | None = None,
) -> Mapping[str, object] | None:
    visited_keys = set(visited_keys or set())
    current_position = _position_of_waypoint(current_waypoint)
    if current_position is None:
        return None

    current_key = _waypoint_key(float(current_position[0]), float(current_position[1]))
    if successor_by_xy is not None:
        preferred_successor = successor_by_xy.get(current_key, None)
        if preferred_successor is not None:
            preferred_position = _position_of_waypoint(preferred_successor)
            if preferred_position is not None:
                preferred_key = _waypoint_key(float(preferred_position[0]), float(preferred_position[1]))
                if preferred_key not in visited_keys:
                    return preferred_successor

    next_position_raw = current_waypoint.get("next", None)
    if isinstance(next_position_raw, (list, tuple)) and len(next_position_raw) >= 2:
        next_key = _waypoint_key(float(next_position_raw[0]), float(next_position_raw[1]))
        if next_key not in visited_keys:
            next_waypoint = waypoint_by_xy.get(next_key)
            if next_waypoint is not None:
                return next_waypoint

    current_heading_rad = float(current_waypoint.get("heading_rad", 0.0))
    current_direction = str(current_waypoint.get("direction", ""))
    fallback_candidates: list[tuple[tuple[float, float, float], Mapping[str, object]]] = []

    for waypoint in lane_waypoints:
        waypoint_position = _position_of_waypoint(waypoint)
        if waypoint_position is None:
            continue
        waypoint_key = _waypoint_key(float(waypoint_position[0]), float(waypoint_position[1]))
        if waypoint_key in visited_keys:
            continue
        if waypoint_key == _waypoint_key(float(current_position[0]), float(current_position[1])):
            continue
        if current_direction and str(waypoint.get("direction", "")) not in {"", current_direction}:
            continue

        dx_m = float(waypoint_position[0]) - float(current_position[0])
        dy_m = float(waypoint_position[1]) - float(current_position[1])
        forward_distance_m = (
            math.cos(float(current_heading_rad)) * dx_m
            + math.sin(float(current_heading_rad)) * dy_m
        )
        if forward_distance_m <= 1e-6:
            continue

        lateral_offset_m = abs(
            -math.sin(float(current_heading_rad)) * dx_m
            + math.cos(float(current_heading_rad)) * dy_m
        )
        distance_from_current_m = math.hypot(dx_m, dy_m)
        score = (
            float(lateral_offset_m),
            float(abs(distance_from_current_m - max(1e-6, float(forward_distance_m)))),
            float(distance_from_current_m),
        )

        if isinstance(next_position_raw, (list, tuple)) and len(next_position_raw) >= 2:
            score = (
                math.hypot(
                    float(waypoint_position[0]) - float(next_position_raw[0]),
                    float(waypoint_position[1]) - float(next_position_raw[1]),
                ),
                float(lateral_offset_m),
                float(distance_from_current_m),
            )

        fallback_candidates.append((score, waypoint))

    if len(fallback_candidates) == 0:
        return None
    fallback_candidates.sort(key=lambda item: item[0])
    return fallback_candidates[0][1]


def _forward_seed_waypoint_on_lane(
    lane_waypoints: Sequence[Mapping[str, object]],
    ego_state: Sequence[float],
) -> Mapping[str, object] | None:
    if len(lane_waypoints) == 0:
        return None

    ego_x_m = float(ego_state[0])
    ego_y_m = float(ego_state[1])
    forward_waypoints = [
        waypoint
        for waypoint in lane_waypoints
        if (
            math.cos(float(waypoint.get("heading_rad", 0.0)))
            * (float(_position_of_waypoint(waypoint)[0]) - ego_x_m)
            + math.sin(float(waypoint.get("heading_rad", 0.0)))
            * (float(_position_of_waypoint(waypoint)[1]) - ego_y_m)
        ) >= -1e-6
    ]
    seed_candidates = forward_waypoints if len(forward_waypoints) > 0 else list(lane_waypoints)
    return min(
        seed_candidates,
        key=lambda waypoint: math.hypot(
            float(_position_of_waypoint(waypoint)[0]) - ego_x_m,
            float(_position_of_waypoint(waypoint)[1]) - ego_y_m,
        ),
    )


def _collect_forward_sample_positions(
    start_waypoint: Mapping[str, object],
    lane_waypoints: Sequence[Mapping[str, object]],
    waypoint_by_xy: Mapping[tuple[float, float], Mapping[str, object]],
    successor_by_xy: Mapping[tuple[float, float], Mapping[str, object] | None],
    origin_xy: Sequence[float],
    target_distances_m: Sequence[float],
) -> list[tuple[float, float]]:
    if len(target_distances_m) == 0:
        return []

    start_position = _position_of_waypoint(start_waypoint)
    if start_position is None:
        return []

    sample_positions: list[tuple[float, float]] = []
    cumulative_distance_m = _distance_between_points(origin_xy, start_position)
    current_waypoint = start_waypoint
    current_position = start_position
    visited_keys = {_waypoint_key(float(current_position[0]), float(current_position[1]))}

    while (
        len(sample_positions) < len(target_distances_m)
        and cumulative_distance_m >= float(target_distances_m[len(sample_positions)])
    ):
        sample_positions.append((float(current_position[0]), float(current_position[1])))

    while len(sample_positions) < len(target_distances_m):
        next_waypoint = _resolve_successor_waypoint_on_lane(
            current_waypoint=current_waypoint,
            lane_waypoints=lane_waypoints,
            waypoint_by_xy=waypoint_by_xy,
            visited_keys=visited_keys,
            successor_by_xy=successor_by_xy,
        )
        if next_waypoint is None:
            break
        next_position = _position_of_waypoint(next_waypoint)
        if next_position is None:
            break

        cumulative_distance_m += _distance_between_points(current_position, next_position)
        current_waypoint = next_waypoint
        current_position = next_position
        visited_keys.add(_waypoint_key(float(current_position[0]), float(current_position[1])))

        while (
            len(sample_positions) < len(target_distances_m)
            and cumulative_distance_m >= float(target_distances_m[len(sample_positions)])
        ):
            sample_positions.append((float(current_position[0]), float(current_position[1])))

    return sample_positions


def _collect_forward_reference_samples(
    start_waypoint: Mapping[str, object],
    lane_waypoints: Sequence[Mapping[str, object]],
    waypoint_by_xy: Mapping[tuple[float, float], Mapping[str, object]],
    successor_by_xy: Mapping[tuple[float, float], Mapping[str, object] | None],
    origin_xy: Sequence[float],
    target_distances_m: Sequence[float],
    lane_id: int,
    fallback_heading_rad: float = 0.0,
) -> List[Dict[str, float]]:
    """Single forward walk collecting reference samples (position + heading) at target distances."""
    if len(target_distances_m) == 0:
        return []

    start_position = _position_of_waypoint(start_waypoint)
    if start_position is None:
        return []

    samples: List[Dict[str, float]] = []
    cumulative_distance_m = _distance_between_points(origin_xy, start_position)
    current_waypoint = start_waypoint
    current_position = start_position
    visited_keys = {_waypoint_key(float(current_position[0]), float(current_position[1]))}

    def _append_current() -> None:
        samples.append({
            "x_ref_m": float(current_position[0]),
            "y_ref_m": float(current_position[1]),
            "heading_rad": float(current_waypoint.get("heading_rad", fallback_heading_rad)),
            "lane_id": int(lane_id),
        })

    while (
        len(samples) < len(target_distances_m)
        and cumulative_distance_m >= float(target_distances_m[len(samples)])
    ):
        _append_current()

    while len(samples) < len(target_distances_m):
        next_waypoint = _resolve_successor_waypoint_on_lane(
            current_waypoint=current_waypoint,
            lane_waypoints=lane_waypoints,
            waypoint_by_xy=waypoint_by_xy,
            visited_keys=visited_keys,
            successor_by_xy=successor_by_xy,
        )
        if next_waypoint is None:
            break
        next_position = _position_of_waypoint(next_waypoint)
        if next_position is None:
            break

        cumulative_distance_m += _distance_between_points(current_position, next_position)
        current_waypoint = next_waypoint
        current_position = next_position
        visited_keys.add(_waypoint_key(float(current_position[0]), float(current_position[1])))

        while (
            len(samples) < len(target_distances_m)
            and cumulative_distance_m >= float(target_distances_m[len(samples)])
        ):
            _append_current()

    return samples


def _estimate_path_curvature_on_lane(
    start_waypoint: Mapping[str, object],
    lane_waypoints: Sequence[Mapping[str, object]],
    waypoint_by_xy: Mapping[tuple[float, float], Mapping[str, object]],
    successor_by_xy: Mapping[tuple[float, float], Mapping[str, object] | None],
    origin_xy: Sequence[float],
    sample_spacing_m: float,
) -> float:
    if float(sample_spacing_m) <= 0.0:
        return 0.0

    sample_positions = _collect_forward_sample_positions(
        start_waypoint=start_waypoint,
        lane_waypoints=lane_waypoints,
        waypoint_by_xy=waypoint_by_xy,
        successor_by_xy=successor_by_xy,
        origin_xy=origin_xy,
        target_distances_m=[
            float(sample_spacing_m),
            2.0 * float(sample_spacing_m),
            3.0 * float(sample_spacing_m),
        ],
    )
    if len(sample_positions) < 3:
        return 0.0

    p1, p2, p3 = sample_positions[0], sample_positions[1], sample_positions[2]
    side_a = _distance_between_points(p1, p2)
    side_b = _distance_between_points(p2, p3)
    side_c = _distance_between_points(p1, p3)
    if min(side_a, side_b, side_c) <= 1e-9:
        return 0.0

    triangle_area = 0.5 * abs(
        (float(p2[0]) - float(p1[0])) * (float(p3[1]) - float(p1[1]))
        - (float(p3[0]) - float(p1[0])) * (float(p2[1]) - float(p1[1]))
    )
    if triangle_area <= 1e-12:
        return 0.0

    return float((4.0 * triangle_area) / (side_a * side_b * side_c))


def compute_lane_lookahead_distance(
    ego_state: Sequence[float],
    lane_center_waypoints: Sequence[Mapping[str, object]],
    target_lane_id: int,
    local_goal_cfg: Mapping[str, object],
) -> float | None:
    if not bool(local_goal_cfg.get("dynamic_lookahead_enabled", True)):
        return None

    min_distance_m = max(0.0, float(local_goal_cfg.get("dynamic_lookahead_min_distance_m", 20.0)))
    max_distance_m = max(
        float(min_distance_m),
        float(local_goal_cfg.get("dynamic_lookahead_max_distance_m", min_distance_m)),
    )
    speed_gain = float(local_goal_cfg.get("dynamic_lookahead_speed_gain", 3.0))
    curvature_gain = float(local_goal_cfg.get("dynamic_lookahead_curvature_gain", 20.0))
    sample_spacing_m = max(
        0.1,
        float(local_goal_cfg.get("dynamic_lookahead_curvature_sample_spacing_m", 5.0)),
    )

    lane_waypoints, waypoint_by_xy, successor_by_xy = _lane_waypoints_by_lane_id(
        lane_center_waypoints=lane_center_waypoints,
        target_lane_id=int(target_lane_id),
    )
    if len(lane_waypoints) == 0:
        return float(min_distance_m)
    lane_topology_cache = _ensure_lane_topology_cache(lane_center_waypoints=lane_center_waypoints)
    lane_data = lane_topology_cache.get("lane_data_by_id", {}).get(int(target_lane_id), {})
    lane_waypoints, waypoint_by_xy, successor_by_xy = _localized_lane_topology_near_pose(
        lane_waypoints=lane_waypoints,
        waypoint_by_xy=waypoint_by_xy,
        successor_by_xy=successor_by_xy,
        reference_x_m=float(ego_state[0]),
        reference_y_m=float(ego_state[1]),
        reference_heading_rad=float(ego_state[3]),
        lane_component_cache=lane_data,
    )

    current_waypoint = _forward_seed_waypoint_on_lane(
        lane_waypoints=lane_waypoints,
        ego_state=ego_state,
    )
    if current_waypoint is None:
        return float(min_distance_m)

    local_curvature = _estimate_path_curvature_on_lane(
        start_waypoint=current_waypoint,
        lane_waypoints=lane_waypoints,
        waypoint_by_xy=waypoint_by_xy,
        successor_by_xy=successor_by_xy,
        origin_xy=(float(ego_state[0]), float(ego_state[1])),
        sample_spacing_m=float(sample_spacing_m),
    )
    raw_lookahead_distance_m = (
        float(min_distance_m)
        + float(speed_gain) * max(0.0, float(ego_state[2]))
        - float(curvature_gain) * abs(float(local_curvature))
    )
    return float(
        min(
            float(max_distance_m),
            max(float(min_distance_m), float(raw_lookahead_distance_m)),
        )
    )


def _nearest_progress_along_route(
    route_points: Sequence[Sequence[float]],
    xy: Sequence[float],
) -> tuple[float, float]:
    if len(route_points) <= 1:
        return 0.0, 0.0

    total_progress_m = 0.0
    best_progress_m = 0.0
    best_distance_m = float("inf")
    px_m = float(xy[0])
    py_m = float(xy[1])

    for idx in range(len(route_points) - 1):
        x0_m, y0_m = float(route_points[idx][0]), float(route_points[idx][1])
        x1_m, y1_m = float(route_points[idx + 1][0]), float(route_points[idx + 1][1])
        dx_m = x1_m - x0_m
        dy_m = y1_m - y0_m
        seg_len_sq = dx_m * dx_m + dy_m * dy_m
        if seg_len_sq <= 1e-9:
            continue
        proj = ((px_m - x0_m) * dx_m + (py_m - y0_m) * dy_m) / seg_len_sq
        proj = min(1.0, max(0.0, proj))
        cx_m = x0_m + proj * dx_m
        cy_m = y0_m + proj * dy_m
        distance_m = math.hypot(px_m - cx_m, py_m - cy_m)
        if distance_m < best_distance_m:
            best_distance_m = distance_m
            best_progress_m = total_progress_m + proj * math.sqrt(seg_len_sq)
        total_progress_m += math.sqrt(seg_len_sq)
    return best_progress_m, total_progress_m


def _sample_route_at_progress(
    route_points: Sequence[Sequence[float]],
    progress_m: float,
) -> List[float]:
    if len(route_points) == 0:
        return [0.0, 0.0, 0.0]
    if len(route_points) == 1:
        return [float(route_points[0][0]), float(route_points[0][1]), 0.0]

    remaining_m = max(0.0, float(progress_m))
    for idx in range(len(route_points) - 1):
        x0_m, y0_m = float(route_points[idx][0]), float(route_points[idx][1])
        x1_m, y1_m = float(route_points[idx + 1][0]), float(route_points[idx + 1][1])
        segment_length_m = math.hypot(x1_m - x0_m, y1_m - y0_m)
        if segment_length_m <= 1e-9:
            continue
        if remaining_m <= segment_length_m:
            alpha = remaining_m / segment_length_m
            heading_rad = math.atan2(y1_m - y0_m, x1_m - x0_m)
            return [
                float(x0_m + alpha * (x1_m - x0_m)),
                float(y0_m + alpha * (y1_m - y0_m)),
                float(heading_rad),
            ]
        remaining_m -= segment_length_m

    last_idx = len(route_points) - 1
    prev_idx = max(0, last_idx - 1)
    heading_rad = math.atan2(
        float(route_points[last_idx][1]) - float(route_points[prev_idx][1]),
        float(route_points[last_idx][0]) - float(route_points[prev_idx][0]),
    )
    return [float(route_points[last_idx][0]), float(route_points[last_idx][1]), float(heading_rad)]


def build_route_reference_samples(
    ego_snapshot: Mapping[str, object],
    route_points: Sequence[Sequence[float]],
    horizon_steps: int,
    step_distance_m: float,
    target_lane_id: int = 1,
) -> List[Dict[str, float]]:
    """Build MPC reference samples by sampling along stored route waypoints.

    Unlike ``build_lane_center_reference_to_destination`` which walks lane-
    center waypoints (and can pick the wrong branch at intersections), this
    function samples directly from the pre-computed route polyline that
    already encodes the correct path through every junction.
    """
    if len(route_points) < 2 or int(horizon_steps) < 0:
        return []

    ego_x_m = float(ego_snapshot.get("x", 0.0))
    ego_y_m = float(ego_snapshot.get("y", 0.0))
    ego_psi_rad = float(ego_snapshot.get("psi", 0.0))

    progress_m, route_length_m = _nearest_progress_along_route(
        route_points=route_points,
        xy=[ego_x_m, ego_y_m],
    )

    reference_samples: List[Dict[str, float]] = []
    for step_idx in range(int(horizon_steps) + 1):
        target_progress_m = min(
            float(route_length_m),
            float(progress_m) + float(step_idx) * float(step_distance_m),
        )
        sample = _sample_route_at_progress(
            route_points=route_points,
            progress_m=float(target_progress_m),
        )
        heading_rad = float(sample[2]) if len(sample) >= 3 and not math.isnan(float(sample[2])) else ego_psi_rad
        reference_samples.append({
            "x_ref_m": float(sample[0]),
            "y_ref_m": float(sample[1]),
            "heading_rad": float(heading_rad),
            "lane_id": int(max(1, int(target_lane_id))),
        })

    return reference_samples


def _triangle_area(
    p1: Sequence[float],
    p2: Sequence[float],
    p3: Sequence[float],
) -> float:
    return abs(
        0.5
        * (
            float(p1[0]) * (float(p2[1]) - float(p3[1]))
            + float(p2[0]) * (float(p3[1]) - float(p1[1]))
            + float(p3[0]) * (float(p1[1]) - float(p2[1]))
        )
    )


def _route_curvature(
    route_points: Sequence[Sequence[float]],
    progress_m: float,
    spacing_m: float,
) -> float:
    p1 = _sample_route_at_progress(route_points, progress_m + float(spacing_m))
    p2 = _sample_route_at_progress(route_points, progress_m + 2.0 * float(spacing_m))
    p3 = _sample_route_at_progress(route_points, progress_m + 3.0 * float(spacing_m))
    a = math.hypot(float(p2[0]) - float(p1[0]), float(p2[1]) - float(p1[1]))
    b = math.hypot(float(p3[0]) - float(p2[0]), float(p3[1]) - float(p2[1]))
    c = math.hypot(float(p3[0]) - float(p1[0]), float(p3[1]) - float(p1[1]))
    if min(a, b, c) <= 1e-6:
        return 0.0
    area = _triangle_area(p1, p2, p3)
    return float((4.0 * area) / (a * b * c))


def compute_route_lookahead_distance(
    ego_state: Sequence[float],
    route_points: Sequence[Sequence[float]],
    local_goal_cfg: Mapping[str, object] | None = None,
) -> float | None:
    if len(route_points) < 2:
        return None

    local_goal_cfg = dict(local_goal_cfg or {})
    progress_m, _route_length_m = _nearest_progress_along_route(route_points, ego_state[:2])
    d_min = float(local_goal_cfg.get("dynamic_lookahead_min_distance_m", 20.0))
    if not bool(local_goal_cfg.get("dynamic_lookahead_enabled", True)):
        return float(max(d_min, d_min))

    d_max = float(local_goal_cfg.get("dynamic_lookahead_max_distance_m", 100.0))
    k_v = float(local_goal_cfg.get("dynamic_lookahead_speed_gain", 3.0))
    k_c = float(local_goal_cfg.get("dynamic_lookahead_curvature_gain", 20.0))
    spacing_m = float(local_goal_cfg.get("dynamic_lookahead_curvature_sample_spacing_m", 10.0))
    curvature = _route_curvature(
        route_points=route_points,
        progress_m=float(progress_m),
        spacing_m=max(1.0, float(spacing_m)),
    )
    raw_distance_m = float(d_min + k_v * float(ego_state[2]) - k_c * abs(float(curvature)))
    return float(min(max(float(raw_distance_m), float(d_min)), float(d_max)))


def _nearest_waypoint_any_lane(
    lane_center_waypoints: Sequence[Mapping[str, object]],
    reference_x_m: float,
    reference_y_m: float,
) -> Mapping[str, object] | None:
    valid_waypoints = [
        waypoint
        for waypoint in lane_center_waypoints
        if isinstance(waypoint.get("position", None), (list, tuple))
        and len(waypoint.get("position", [])) >= 2
    ]
    if len(valid_waypoints) == 0:
        return None
    return min(
        valid_waypoints,
        key=lambda waypoint: math.hypot(
            float(waypoint["position"][0]) - float(reference_x_m),
            float(waypoint["position"][1]) - float(reference_y_m),
        ),
    )


def _nearest_waypoint_on_lane(
    lane_center_waypoints: Sequence[Mapping[str, object]],
    target_lane_id: int,
    reference_x_m: float,
    reference_y_m: float,
) -> Mapping[str, object] | None:
    target_lane_id = int(target_lane_id)
    valid_waypoints = [
        waypoint
        for waypoint in lane_center_waypoints
        if int(waypoint.get("lane_id", -1)) == int(target_lane_id)
        and isinstance(waypoint.get("position", None), (list, tuple))
        and len(waypoint.get("position", [])) >= 2
    ]
    if len(valid_waypoints) == 0:
        return _nearest_waypoint_any_lane(
            lane_center_waypoints=lane_center_waypoints,
            reference_x_m=float(reference_x_m),
            reference_y_m=float(reference_y_m),
        )
    return min(
        valid_waypoints,
        key=lambda waypoint: math.hypot(
            float(waypoint["position"][0]) - float(reference_x_m),
            float(waypoint["position"][1]) - float(reference_y_m),
        ),
    )


def _select_waypoint_ahead_on_lane(
    lane_center_waypoints: Sequence[Mapping[str, object]],
    ego_snapshot: Mapping[str, object],
    target_lane_id: int,
    target_distance_m: float,
    next_maneuver: str | None = None,
) -> List[float] | None:
    lane_waypoints, waypoint_by_key, successor_by_xy = _lane_waypoints_by_lane_id(
        lane_center_waypoints=lane_center_waypoints,
        target_lane_id=int(target_lane_id),
    )
    if len(lane_waypoints) == 0:
        return None
    lane_topology_cache = _ensure_lane_topology_cache(lane_center_waypoints=lane_center_waypoints)
    lane_data = lane_topology_cache.get("lane_data_by_id", {}).get(int(target_lane_id), {})
    lane_waypoints, waypoint_by_key, successor_by_xy = _localized_lane_topology_near_pose(
        lane_waypoints=lane_waypoints,
        waypoint_by_xy=waypoint_by_key,
        successor_by_xy=successor_by_xy,
        reference_x_m=float(ego_snapshot.get("x", 0.0)),
        reference_y_m=float(ego_snapshot.get("y", 0.0)),
        reference_heading_rad=float(ego_snapshot.get("psi", 0.0)),
        lane_component_cache=lane_data,
    )

    ego_x_m = float(ego_snapshot.get("x", 0.0))
    ego_y_m = float(ego_snapshot.get("y", 0.0))
    ego_psi_rad = float(ego_snapshot.get("psi", 0.0))

    def _best_segment_for_waypoints(
        waypoints: Sequence[Mapping[str, object]],
        waypoint_by_key_local: Mapping[tuple[float, float], Mapping[str, object]],
        successor_by_xy_local: Mapping[tuple[float, float], Mapping[str, object] | None],
    ) -> tuple[Mapping[str, object], Mapping[str, object], float, float] | None:
        best_segment_local: tuple[Mapping[str, object], Mapping[str, object], float, float] | None = None
        best_projection_distance_m_local = float("inf")

        for waypoint in waypoints:
            start_position = _position_of_waypoint(waypoint)
            if start_position is None:
                continue
            next_waypoint = _resolve_successor_waypoint_on_lane(
                current_waypoint=waypoint,
                lane_waypoints=waypoints,
                waypoint_by_xy=waypoint_by_key_local,
                successor_by_xy=successor_by_xy_local,
            )
            if next_waypoint is None:
                continue
            end_position = _position_of_waypoint(next_waypoint)
            if end_position is None:
                continue

            dx_m = float(end_position[0]) - float(start_position[0])
            dy_m = float(end_position[1]) - float(start_position[1])
            segment_length_sq = dx_m * dx_m + dy_m * dy_m
            if segment_length_sq <= 1e-9:
                continue

            alpha = (
                ((float(ego_x_m) - float(start_position[0])) * dx_m)
                + ((float(ego_y_m) - float(start_position[1])) * dy_m)
            ) / segment_length_sq
            alpha = min(1.0, max(0.0, float(alpha)))
            proj_x_m = float(start_position[0]) + float(alpha) * dx_m
            proj_y_m = float(start_position[1]) + float(alpha) * dy_m
            projection_distance_m = math.hypot(float(ego_x_m) - proj_x_m, float(ego_y_m) - proj_y_m)
            if projection_distance_m < best_projection_distance_m_local:
                best_projection_distance_m_local = float(projection_distance_m)
                best_segment_local = (
                    waypoint,
                    next_waypoint,
                    float(alpha),
                    math.sqrt(segment_length_sq),
                )

        return best_segment_local

    seed_waypoint = min(
        lane_waypoints,
        key=lambda waypoint: math.hypot(
            float(_position_of_waypoint(waypoint)[0]) - ego_x_m,
            float(_position_of_waypoint(waypoint)[1]) - ego_y_m,
        ),
    )
    seed_position = _position_of_waypoint(seed_waypoint)
    if seed_position is None:
        return None

    best_segment = _best_segment_for_waypoints(
        lane_waypoints,
        waypoint_by_key_local=waypoint_by_key,
        successor_by_xy_local=successor_by_xy,
    )
    if best_segment is not None:
        current_waypoint = best_segment[0]
        current_direction = str(current_waypoint.get("direction", ""))
        directional_lane_waypoints = [
            waypoint
            for waypoint in lane_waypoints
            if str(waypoint.get("direction", "")) == current_direction
        ]
        if len(directional_lane_waypoints) > 0:
            lane_waypoints = directional_lane_waypoints
            waypoint_by_key = _waypoint_map(lane_waypoints)
            successor_by_xy = _successor_map_for_lane_waypoints(
                lane_waypoints=lane_waypoints,
                waypoint_by_xy=waypoint_by_key,
            )
            localized_best_segment = _best_segment_for_waypoints(
                lane_waypoints,
                waypoint_by_key_local=waypoint_by_key,
                successor_by_xy_local=successor_by_xy,
            )
            if localized_best_segment is not None:
                best_segment = localized_best_segment

    if best_segment is None:
        current_waypoint = seed_waypoint
        remaining_distance_m = max(0.0, float(target_distance_m))
        visited_keys = {_waypoint_key(float(seed_position[0]), float(seed_position[1]))}

        while float(remaining_distance_m) > 1e-6:
            current_position = _position_of_waypoint(current_waypoint)
            if current_position is None:
                return None

            next_waypoint = _resolve_successor_waypoint_on_lane(
                current_waypoint=current_waypoint,
                lane_waypoints=lane_waypoints,
                waypoint_by_xy=waypoint_by_key,
                visited_keys=visited_keys,
                successor_by_xy=successor_by_xy,
            )
            if next_waypoint is None:
                break
            next_position = _position_of_waypoint(next_waypoint)
            if next_position is None:
                break

            dx_m = float(next_position[0]) - float(current_position[0])
            dy_m = float(next_position[1]) - float(current_position[1])
            segment_length_m = math.hypot(dx_m, dy_m)
            if segment_length_m <= 1e-9:
                current_waypoint = next_waypoint
                continue

            if float(remaining_distance_m) <= float(segment_length_m) + 1e-6:
                return [
                    float(next_position[0]),
                    float(next_position[1]),
                    0.0,
                    float(next_waypoint.get("heading_rad", math.atan2(dy_m, dx_m))),
                    float(max(1, int(target_lane_id))),
                ]

            remaining_distance_m -= float(segment_length_m)
            current_waypoint = next_waypoint
            visited_keys.add(_waypoint_key(float(next_position[0]), float(next_position[1])))

        final_position = _position_of_waypoint(current_waypoint)
        if final_position is None:
            return None
        return [
            float(final_position[0]),
            float(final_position[1]),
            0.0,
            float(current_waypoint.get("heading_rad", ego_psi_rad)),
            float(max(1, int(target_lane_id))),
        ]

    current_waypoint, next_waypoint, current_alpha, current_segment_length_m = best_segment
    current_position = _position_of_waypoint(current_waypoint)
    next_position = _position_of_waypoint(next_waypoint)
    if current_position is None or next_position is None:
        return None

    current_carla_waypoint = current_waypoint.get("carla_waypoint", None)
    internal_lane_id_by_raw_key, lane_ids_by_group = _carla_lane_lookup(
        lane_center_waypoints=lane_center_waypoints,
    )
    if current_carla_waypoint is not None and hasattr(current_carla_waypoint, "next"):
        local_segment_offset_m = max(0.0, float(current_alpha) * float(current_segment_length_m))
        requested_distance_m = max(0.0, float(target_distance_m) + float(local_segment_offset_m))
        desired_maneuver_code = _normalized_next_maneuver_code(next_maneuver)
        carla_follow_result = _follow_carla_branch_for_distance(
            start_carla_waypoint=current_carla_waypoint,
            requested_distance_m=float(requested_distance_m),
            target_lane_id=int(target_lane_id),
            desired_maneuver_code=str(desired_maneuver_code),
            internal_lane_id_by_raw_key=internal_lane_id_by_raw_key,
            lane_ids_by_group=lane_ids_by_group,
            fallback_heading_rad=float(current_waypoint.get("heading_rad", ego_psi_rad)),
        )

        if carla_follow_result is not None:
            selected_carla_waypoint, selected_internal_lane_id = carla_follow_result
            snap_lane_id = int(selected_internal_lane_id) if selected_internal_lane_id is not None else int(target_lane_id)
            selected_transform = getattr(selected_carla_waypoint, "transform", None)
            selected_location = getattr(selected_transform, "location", None)
            selected_rotation = getattr(selected_transform, "rotation", None)

            snap_lane_waypoints, _, _ = _lane_waypoints_by_lane_id(
                lane_center_waypoints=lane_center_waypoints,
                target_lane_id=int(snap_lane_id),
            )
            snapped_state = _snap_carla_target_to_internal_lane_state(
                carla_target_waypoint=selected_carla_waypoint,
                lane_waypoints=snap_lane_waypoints if len(snap_lane_waypoints) > 0 else lane_waypoints,
                ego_state=(ego_x_m, ego_y_m, 0.0, ego_psi_rad),
            )
            if snapped_state is not None:
                snapped_state.append(float(max(1, int(snap_lane_id))))
                return snapped_state

            if selected_location is not None:
                return [
                    float(getattr(selected_location, "x", float(current_position[0]))),
                    float(getattr(selected_location, "y", float(current_position[1]))),
                    0.0,
                    math.radians(float(getattr(selected_rotation, "yaw", math.degrees(ego_psi_rad)))),
                    float(max(1, int(snap_lane_id))),
                ]

    remaining_distance_m = max(0.0, float(target_distance_m))
    distance_to_segment_end_m = max(0.0, (1.0 - float(current_alpha)) * float(current_segment_length_m))
    if remaining_distance_m <= distance_to_segment_end_m + 1e-6:
        return [
            float(next_position[0]),
            float(next_position[1]),
            0.0,
            float(next_waypoint.get("heading_rad", math.atan2(
                float(next_position[1]) - float(current_position[1]),
                float(next_position[0]) - float(current_position[0]),
            ))),
            float(max(1, int(target_lane_id))),
        ]

    remaining_distance_m -= float(distance_to_segment_end_m)
    current_waypoint = next_waypoint
    visited_keys = {
        _waypoint_key(float(current_position[0]), float(current_position[1])),
        _waypoint_key(float(next_position[0]), float(next_position[1])),
    }

    while True:
        current_position = _position_of_waypoint(current_waypoint)
        if current_position is None:
            return None
        current_key = _waypoint_key(float(current_position[0]), float(current_position[1]))
        if current_key in visited_keys:
            visited_keys.discard(current_key)
        visited_keys.add(current_key)

        next_waypoint = _resolve_successor_waypoint_on_lane(
            current_waypoint=current_waypoint,
            lane_waypoints=lane_waypoints,
            waypoint_by_xy=waypoint_by_key,
            visited_keys=visited_keys,
            successor_by_xy=successor_by_xy,
        )
        if next_waypoint is None:
            break
        next_position = _position_of_waypoint(next_waypoint)
        if next_position is None:
            break

        dx_m = float(next_position[0]) - float(current_position[0])
        dy_m = float(next_position[1]) - float(current_position[1])
        segment_length_m = math.hypot(dx_m, dy_m)
        if segment_length_m <= 1e-9:
            current_waypoint = next_waypoint
            continue

        if remaining_distance_m <= segment_length_m + 1e-6:
            return [
                float(next_position[0]),
                float(next_position[1]),
                0.0,
                float(next_waypoint.get("heading_rad", math.atan2(dy_m, dx_m))),
                float(max(1, int(target_lane_id))),
            ]

        remaining_distance_m -= float(segment_length_m)
        current_waypoint = next_waypoint

    final_position = _position_of_waypoint(current_waypoint)
    if final_position is None:
        return None
    return [
        float(final_position[0]),
        float(final_position[1]),
        0.0,
        float(current_waypoint.get("heading_rad", ego_psi_rad)),
        float(max(1, int(target_lane_id))),
    ]


def build_destination_on_lane(
    ego_snapshot: Mapping[str, object],
    lane_center_waypoints: Sequence[Mapping[str, object]],
    target_lane_id: int,
    target_distance_m: float,
    road_cfg: Mapping[str, object] | None = None,
    next_maneuver: str | None = None,
) -> List[float] | None:
    lane_count = _lane_count_from_inputs(
        lane_center_waypoints=lane_center_waypoints,
        road_cfg=road_cfg,
    )
    clamped_lane_id = min(
        max(1, int(target_lane_id)),
        max(1, int(lane_count)),
    )
    return _select_waypoint_ahead_on_lane(
        lane_center_waypoints=lane_center_waypoints,
        ego_snapshot=ego_snapshot,
        target_lane_id=clamped_lane_id,
        target_distance_m=float(target_distance_m),
        next_maneuver=next_maneuver,
    )


def build_lane_center_reference_to_destination(
    ego_snapshot: Mapping[str, object],
    lane_center_waypoints: Sequence[Mapping[str, object]],
    destination_state: Sequence[float],
    horizon_steps: int,
    step_distance_m: float,
    target_distance_m: float | None = None,
    road_cfg: Mapping[str, object] | None = None,
    next_maneuver: str | None = None,
) -> List[Dict[str, float]]:
    if int(horizon_steps) < 0 or len(lane_center_waypoints) == 0:
        return []
    if len(destination_state) < 2:
        return []

    ego_x_m = float(ego_snapshot.get("x", 0.0))
    ego_y_m = float(ego_snapshot.get("y", 0.0))
    ego_psi_rad = float(ego_snapshot.get("psi", 0.0))

    lane_count = _lane_count_from_inputs(
        lane_center_waypoints=lane_center_waypoints,
        road_cfg=road_cfg,
    )
    destination_lane_id = (
        int(destination_state[4])
        if len(destination_state) >= 5
        else -1
    )
    if destination_lane_id <= 0:
        snapped_destination_waypoint = _nearest_waypoint_any_lane(
            lane_center_waypoints=lane_center_waypoints,
            reference_x_m=float(destination_state[0]),
            reference_y_m=float(destination_state[1]),
        )
        if snapped_destination_waypoint is not None:
            destination_lane_id = int(snapped_destination_waypoint.get("lane_id", destination_lane_id))
    destination_lane_id = min(
        max(1, int(destination_lane_id)),
        max(1, int(lane_count)),
    )

    if target_distance_m is None:
        target_distance_m = _distance_between_points(
            start_xy=[float(ego_x_m), float(ego_y_m)],
            end_xy=[float(destination_state[0]), float(destination_state[1])],
        )
    clamped_target_distance_m = max(0.0, float(target_distance_m))
    sample_step_distance_m = max(0.25, float(step_distance_m))

    # --- Optimized: single forward walk instead of N separate walks ---
    lane_waypoints, waypoint_by_xy, successor_by_xy = _lane_waypoints_by_lane_id(
        lane_center_waypoints=lane_center_waypoints,
        target_lane_id=int(destination_lane_id),
    )
    if len(lane_waypoints) == 0:
        return []

    lane_topology_cache = _ensure_lane_topology_cache(
        lane_center_waypoints=lane_center_waypoints,
    )
    lane_data = lane_topology_cache.get("lane_data_by_id", {}).get(
        int(destination_lane_id), {},
    )
    lane_waypoints, waypoint_by_xy, successor_by_xy = _localized_lane_topology_near_pose(
        lane_waypoints=lane_waypoints,
        waypoint_by_xy=waypoint_by_xy,
        successor_by_xy=successor_by_xy,
        reference_x_m=float(ego_x_m),
        reference_y_m=float(ego_y_m),
        reference_heading_rad=float(ego_psi_rad),
        lane_component_cache=lane_data,
    )

    ego_state_tuple = (ego_x_m, ego_y_m, 0.0, ego_psi_rad)
    seed_waypoint = _forward_seed_waypoint_on_lane(
        lane_waypoints=lane_waypoints,
        ego_state=ego_state_tuple,
    )
    if seed_waypoint is None:
        return []

    reference_samples: List[Dict[str, float]] = []
    seed_position = _position_of_waypoint(seed_waypoint)
    if seed_position is not None:
        reference_samples.append({
            "x_ref_m": float(seed_position[0]),
            "y_ref_m": float(seed_position[1]),
            "heading_rad": float(seed_waypoint.get("heading_rad", ego_psi_rad)),
            "lane_id": int(destination_lane_id),
        })

    target_distances = [
        min(float(clamped_target_distance_m), float(stage_idx) * float(sample_step_distance_m))
        for stage_idx in range(1, int(horizon_steps) + 1)
    ]

    forward_samples = _collect_forward_reference_samples(
        start_waypoint=seed_waypoint,
        lane_waypoints=lane_waypoints,
        waypoint_by_xy=waypoint_by_xy,
        successor_by_xy=successor_by_xy,
        origin_xy=(float(ego_x_m), float(ego_y_m)),
        target_distances_m=target_distances,
        lane_id=int(destination_lane_id),
        fallback_heading_rad=float(ego_psi_rad),
    )
    reference_samples.extend(forward_samples)

    # Pad with the last sample if the walk ended early
    if len(reference_samples) > 0:
        while len(reference_samples) < int(horizon_steps) + 1:
            reference_samples.append(dict(reference_samples[-1]))

    return reference_samples


def compute_temporary_destination_state_from_route(
    ego_snapshot: Mapping[str, object],
    route_points: Sequence[Sequence[float]],
    target_distance_m: float,
    target_v_mps: float,
    final_destination_state: Sequence[float] | None = None,
    lane_center_waypoints: Sequence[Mapping[str, object]] | None = None,
    target_lane_id: int | None = None,
    local_goal_cfg: Mapping[str, object] | None = None,
    snap_to_lane: bool = False,
) -> List[float] | None:
    if len(route_points) == 0:
        return None

    local_goal_cfg = dict(local_goal_cfg or {})
    ego_x_m = float(ego_snapshot.get("x", 0.0))
    ego_y_m = float(ego_snapshot.get("y", 0.0))
    ego_psi_rad = float(ego_snapshot.get("psi", 0.0))
    progress_m, route_length_m = _nearest_progress_along_route(
        route_points=route_points,
        xy=[float(ego_x_m), float(ego_y_m)],
    )
    target_progress_m = min(float(route_length_m), float(progress_m) + max(0.0, float(target_distance_m)))
    target_x_m, target_y_m, target_heading_rad = _sample_route_at_progress(
        route_points=route_points,
        progress_m=float(target_progress_m),
    )

    if final_destination_state is not None and len(final_destination_state) >= 4:
        final_distance_m = math.hypot(
            float(final_destination_state[0]) - float(ego_x_m),
            float(final_destination_state[1]) - float(ego_y_m),
        )
        if final_distance_m <= float(local_goal_cfg.get("lock_to_final_distance_m", 30.0)):
            final_state = list(final_destination_state[:5])
            final_state[2] = float(target_v_mps)
            if len(final_state) < 5:
                final_state.append(float(max(1, int(target_lane_id or 1))))
            return final_state

    inferred_lane_id = int(max(1, int(target_lane_id or 1)))
    if lane_center_waypoints is not None and len(lane_center_waypoints) > 0:
        if bool(snap_to_lane) and target_lane_id is not None:
            snapped_waypoint = _nearest_waypoint_on_lane(
                lane_center_waypoints=lane_center_waypoints,
                target_lane_id=int(target_lane_id),
                reference_x_m=float(target_x_m),
                reference_y_m=float(target_y_m),
            )
        else:
            snapped_waypoint = _nearest_waypoint_any_lane(
                lane_center_waypoints=lane_center_waypoints,
                reference_x_m=float(target_x_m),
                reference_y_m=float(target_y_m),
            )

        if snapped_waypoint is not None:
            snapped_position = _position_of_waypoint(snapped_waypoint)
            if snapped_position is not None:
                target_x_m = float(snapped_position[0])
                target_y_m = float(snapped_position[1])
            target_heading_rad = float(snapped_waypoint.get("heading_rad", target_heading_rad))
            inferred_lane_id = int(max(1, int(snapped_waypoint.get("lane_id", inferred_lane_id))))

    return [
        float(target_x_m),
        float(target_y_m),
        float(target_v_mps),
        float(target_heading_rad if not math.isnan(float(target_heading_rad)) else ego_psi_rad),
        float(inferred_lane_id),
    ]


def compute_temporary_destination_state(
    ego_snapshot: Mapping[str, object],
    lane_center_waypoints: Sequence[Mapping[str, object]],
    target_lane_id: int,
    target_distance_m: float,
    target_v_mps: float,
    road_cfg: Mapping[str, object] | None = None,
    next_maneuver: str | None = None,
) -> List[float] | None:
    lane_count = _lane_count_from_inputs(
        lane_center_waypoints=lane_center_waypoints,
        road_cfg=road_cfg,
    )
    clamped_lane_id = min(
        max(1, int(target_lane_id)),
        max(1, int(lane_count)),
    )
    destination_state = build_destination_on_lane(
        ego_snapshot=ego_snapshot,
        lane_center_waypoints=lane_center_waypoints,
        target_lane_id=int(clamped_lane_id),
        target_distance_m=float(target_distance_m),
        road_cfg=road_cfg,
        next_maneuver=next_maneuver,
    )
    if destination_state is None:
        return None
    destination_state = list(destination_state)
    destination_state[2] = float(target_v_mps)
    if len(destination_state) >= 5:
        destination_state[4] = float(max(1, int(destination_state[4])))
    else:
        destination_state.append(float(clamped_lane_id))
    return destination_state

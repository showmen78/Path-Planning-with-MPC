"""
Shared local-goal helpers for lookahead and temporary-destination placement.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence

from utility.carla_lane_graph import (
    _carla_waypoint_group_key,
    _internal_lane_id,
    direction_key,
)


_LANE_TOPOLOGY_CACHE: dict[int, dict[str, object]] = {}


def _position_of_waypoint(waypoint: Mapping[str, object]) -> tuple[float, float] | None:
    position = waypoint.get("position", None)
    if not isinstance(position, (list, tuple)) or len(position) < 2:
        return None
    return float(position[0]), float(position[1])


def _waypoint_key(x_m: float, y_m: float) -> tuple[float, float]:
    return round(float(x_m), 3), round(float(y_m), 3)


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


def _carla_lane_lookup(
    lane_center_waypoints: Sequence[Mapping[str, object]],
) -> tuple[dict[tuple[int, int, int], int], dict[tuple[int, int, str], set[int]]]:
    internal_lane_id_by_raw_key: dict[tuple[int, int, int], int] = {}
    lane_ids_by_group: dict[tuple[int, int, str], set[int]] = {}

    for waypoint in lane_center_waypoints:
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

    return internal_lane_id_by_raw_key, lane_ids_by_group


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
    cache_key = id(lane_center_waypoints)
    cached = _LANE_TOPOLOGY_CACHE.get(cache_key, None)
    if (
        cached is None
        or cached.get("source") is not lane_center_waypoints
        or int(cached.get("size", -1)) != len(lane_center_waypoints)
    ):
        lane_data_by_id: dict[int, dict[str, object]] = {}
        for waypoint in lane_center_waypoints:
            if _position_of_waypoint(waypoint) is None:
                continue
            lane_id = int(waypoint.get("lane_id", -999999))
            lane_data_by_id.setdefault(lane_id, {"waypoints": []})
            lane_data_by_id[lane_id]["waypoints"].append(waypoint)

        for lane_data in lane_data_by_id.values():
            lane_waypoints = list(lane_data.get("waypoints", []))
            waypoint_by_xy = _waypoint_map(lane_waypoints)
            successor_by_xy = _successor_map_for_lane_waypoints(
                lane_waypoints=lane_waypoints,
                waypoint_by_xy=waypoint_by_xy,
            )
            lane_data["waypoint_by_xy"] = waypoint_by_xy
            lane_data["successor_by_xy"] = successor_by_xy

        cached = {
            "source": lane_center_waypoints,
            "size": len(lane_center_waypoints),
            "lane_data_by_id": lane_data_by_id,
        }
        _LANE_TOPOLOGY_CACHE[cache_key] = cached

    lane_data_by_id = dict(cached.get("lane_data_by_id", {}))
    lane_data = dict(lane_data_by_id.get(int(target_lane_id), {}))
    return (
        list(lane_data.get("waypoints", [])),
        dict(lane_data.get("waypoint_by_xy", {})),
        dict(lane_data.get("successor_by_xy", {})),
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


def _select_waypoint_ahead_on_lane(
    lane_center_waypoints: Sequence[Mapping[str, object]],
    ego_snapshot: Mapping[str, object],
    target_lane_id: int,
    target_distance_m: float,
) -> List[float] | None:
    lane_waypoints, waypoint_by_key, successor_by_xy = _lane_waypoints_by_lane_id(
        lane_center_waypoints=lane_center_waypoints,
        target_lane_id=int(target_lane_id),
    )
    if len(lane_waypoints) == 0:
        return None

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
        return [
            float(seed_position[0]),
            float(seed_position[1]),
            0.0,
            float(seed_waypoint.get("heading_rad", ego_psi_rad)),
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
        try:
            carla_candidates = list(current_carla_waypoint.next(float(requested_distance_m)))
        except Exception:
            carla_candidates = []

        if len(carla_candidates) > 0:
            current_direction = str(current_waypoint.get("direction", ""))

            def _candidate_score(carla_waypoint) -> tuple[float, float, float, float]:
                candidate_internal_lane_id = _internal_lane_id_for_carla_waypoint(
                    carla_waypoint=carla_waypoint,
                    internal_lane_id_by_raw_key=internal_lane_id_by_raw_key,
                    lane_ids_by_group=lane_ids_by_group,
                )
                same_internal_lane = (
                    candidate_internal_lane_id is not None
                    and int(candidate_internal_lane_id) == int(target_lane_id)
                )
                same_direction = (
                    str(current_direction) == ""
                    or direction_key(int(getattr(carla_waypoint, "lane_id", 0))) == str(current_direction)
                )
                candidate_transform = getattr(carla_waypoint, "transform", None)
                candidate_location = getattr(candidate_transform, "location", None)
                candidate_rotation = getattr(candidate_transform, "rotation", None)
                candidate_x_m = float(getattr(candidate_location, "x", float(current_position[0])))
                candidate_y_m = float(getattr(candidate_location, "y", float(current_position[1])))
                candidate_heading_rad = math.radians(float(getattr(candidate_rotation, "yaw", math.degrees(ego_psi_rad))))
                heading_delta_rad = abs(
                    math.atan2(
                        math.sin(candidate_heading_rad - float(current_waypoint.get("heading_rad", ego_psi_rad))),
                        math.cos(candidate_heading_rad - float(current_waypoint.get("heading_rad", ego_psi_rad))),
                    )
                )
                return (
                    0.0 if same_internal_lane else 1.0,
                    0.0 if same_direction else 1.0,
                    float(heading_delta_rad),
                    math.hypot(candidate_x_m - ego_x_m, candidate_y_m - ego_y_m),
                )

            selected_carla_waypoint = min(carla_candidates, key=_candidate_score)
            snapped_state = _snap_carla_target_to_internal_lane_state(
                carla_target_waypoint=selected_carla_waypoint,
                lane_waypoints=lane_waypoints,
                ego_state=(ego_x_m, ego_y_m, 0.0, ego_psi_rad),
            )
            if snapped_state is not None:
                return snapped_state

            selected_transform = getattr(selected_carla_waypoint, "transform", None)
            selected_location = getattr(selected_transform, "location", None)
            selected_rotation = getattr(selected_transform, "rotation", None)
            if selected_location is not None:
                return [
                    float(getattr(selected_location, "x", float(current_position[0]))),
                    float(getattr(selected_location, "y", float(current_position[1]))),
                    0.0,
                    math.radians(float(getattr(selected_rotation, "yaw", math.degrees(ego_psi_rad)))),
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
    ]


def build_destination_on_lane(
    ego_snapshot: Mapping[str, object],
    lane_center_waypoints: Sequence[Mapping[str, object]],
    target_lane_id: int,
    target_distance_m: float,
    road_cfg: Mapping[str, object] | None = None,
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
    )


def compute_temporary_destination_state(
    ego_snapshot: Mapping[str, object],
    lane_center_waypoints: Sequence[Mapping[str, object]],
    target_lane_id: int,
    target_distance_m: float,
    target_v_mps: float,
    road_cfg: Mapping[str, object] | None = None,
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
    )
    if destination_state is None:
        return None
    destination_state = list(destination_state)
    destination_state[2] = float(target_v_mps)
    destination_state.append(float(clamped_lane_id))
    return destination_state

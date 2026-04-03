"""
Behavior-planner reroute helpers.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from utility.carla_lane_graph import (
    canonical_lane_id_for_waypoint,
    canonical_lane_waypoints,
    direction_key,
)


CP_MESSAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cp_message.json")
DEFAULT_REROUTE_PENALTY = 1.0e9


def ensure_cp_message_file_exists(message_path: str = CP_MESSAGE_PATH) -> None:
    if os.path.exists(message_path):
        return
    with open(message_path, "w", encoding="utf-8") as message_file:
        json.dump([], message_file, indent=2)


def load_cp_messages(message_path: str = CP_MESSAGE_PATH) -> List[dict]:
    ensure_cp_message_file_exists(message_path=message_path)
    try:
        with open(message_path, "r", encoding="utf-8") as message_file:
            payload = json.load(message_file)
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [dict(item) for item in payload if isinstance(item, Mapping)]


def write_cp_messages(messages: Sequence[Mapping[str, object]], message_path: str = CP_MESSAGE_PATH) -> None:
    ensure_cp_message_file_exists(message_path=message_path)
    normalized_messages = [dict(message) for message in list(messages or []) if isinstance(message, Mapping)]
    with open(message_path, "w", encoding="utf-8") as message_file:
        json.dump(normalized_messages, message_file, indent=2)


def remove_cp_messages_by_id(
    message_ids: Sequence[object],
    message_path: str = CP_MESSAGE_PATH,
) -> List[dict]:
    retained_messages: List[dict] = []
    remove_ids = {
        str(message_id).strip()
        for message_id in list(message_ids or [])
        if str(message_id).strip()
    }
    for message in load_cp_messages(message_path=message_path):
        message_id = str(message.get("id", "")).strip()
        if message_id and message_id in remove_ids:
            continue
        retained_messages.append(dict(message))
    write_cp_messages(retained_messages, message_path=message_path)
    return retained_messages


def lane_closure_messages(
    messages: Sequence[Mapping[str, object]],
) -> List[dict]:
    valid_messages: List[dict] = []
    for message in list(messages or []):
        if not isinstance(message, Mapping):
            continue
        if str(message.get("type", "")).strip().lower() != "lane_closure":
            continue
        if not str(message.get("id", "")).strip():
            continue
        valid_messages.append(dict(message))
    return valid_messages


def load_lane_closure_messages(message_path: str = CP_MESSAGE_PATH) -> List[dict]:
    return lane_closure_messages(load_cp_messages(message_path=message_path))


def pop_lane_closure_messages(message_path: str = CP_MESSAGE_PATH) -> List[dict]:
    current_messages = load_cp_messages(message_path=message_path)
    closure_messages = lane_closure_messages(current_messages)
    if len(closure_messages) == 0:
        return []
    closure_ids = {
        str(message.get("id", "")).strip()
        for message in closure_messages
        if str(message.get("id", "")).strip()
    }
    retained_messages: List[dict] = []
    for message in current_messages:
        message_id = str(message.get("id", "")).strip()
        message_type = str(message.get("type", "")).strip().lower()
        if message_type == "lane_closure" and message_id in closure_ids:
            continue
        retained_messages.append(dict(message))
    write_cp_messages(retained_messages, message_path=message_path)
    return [dict(message) for message in closure_messages]


def _coerce_position_xy(raw_position: object) -> List[float] | None:
    if not isinstance(raw_position, (list, tuple)) or len(raw_position) < 2:
        return None
    return [float(raw_position[0]), float(raw_position[1])]


def _road_numeric_id(raw_road_id: object) -> int | None:
    if raw_road_id is None:
        return None
    raw_text = str(raw_road_id).strip()
    if not raw_text:
        return None
    if ":" in raw_text:
        raw_text = raw_text.split(":", 1)[0]
    try:
        return int(raw_text)
    except Exception:
        return None


def _section_numeric_id(raw_section_id: object) -> int | None:
    if raw_section_id is None:
        return None
    raw_text = str(raw_section_id).strip()
    if not raw_text:
        return None
    try:
        return int(raw_text)
    except Exception:
        return None


def _road_segment_key_from_waypoint(waypoint) -> str:
    return f"{int(getattr(waypoint, 'road_id', 0))}:{int(getattr(waypoint, 'section_id', 0))}"


def _route_points_from_summary(route_summary: object) -> List[List[float]]:
    return [
        [float(item[0]), float(item[1])]
        for item in list(getattr(route_summary, "route_waypoints", []) or [])
        if isinstance(item, Sequence) and len(item) >= 2
    ]


def _distance_point_to_segment_m(
    point_xy: Sequence[float],
    segment_start_xy: Sequence[float],
    segment_end_xy: Sequence[float],
) -> float:
    px_m = float(point_xy[0])
    py_m = float(point_xy[1])
    ax_m = float(segment_start_xy[0])
    ay_m = float(segment_start_xy[1])
    bx_m = float(segment_end_xy[0])
    by_m = float(segment_end_xy[1])
    dx_m = float(bx_m) - float(ax_m)
    dy_m = float(by_m) - float(ay_m)
    segment_len_sq = dx_m * dx_m + dy_m * dy_m
    if float(segment_len_sq) <= 1.0e-9:
        return float(math.hypot(float(px_m) - float(ax_m), float(py_m) - float(ay_m)))
    projection = (
        (float(px_m) - float(ax_m)) * float(dx_m)
        + (float(py_m) - float(ay_m)) * float(dy_m)
    ) / float(segment_len_sq)
    projection = max(0.0, min(1.0, float(projection)))
    closest_x_m = float(ax_m) + float(projection) * float(dx_m)
    closest_y_m = float(ay_m) + float(projection) * float(dy_m)
    return float(math.hypot(float(px_m) - float(closest_x_m), float(py_m) - float(closest_y_m)))


def _route_min_distance_to_point_m(
    route_points: Sequence[Sequence[float]],
    point_xy: Sequence[float],
) -> float:
    normalized_route_points = [
        [float(route_point[0]), float(route_point[1])]
        for route_point in list(route_points or [])
        if isinstance(route_point, Sequence) and len(route_point) >= 2
    ]
    if len(normalized_route_points) == 0:
        return float("inf")
    if len(normalized_route_points) == 1:
        return float(
            math.hypot(
                float(point_xy[0]) - float(normalized_route_points[0][0]),
                float(point_xy[1]) - float(normalized_route_points[0][1]),
            )
        )
    return min(
        _distance_point_to_segment_m(
            point_xy=point_xy,
            segment_start_xy=segment_start_xy,
            segment_end_xy=segment_end_xy,
        )
        for segment_start_xy, segment_end_xy in zip(
            normalized_route_points[:-1],
            normalized_route_points[1:],
        )
    )


def _route_overlaps_blocked_positions(
    route_points: Sequence[Sequence[float]],
    blocked_positions_xy: Sequence[Sequence[float]],
    *,
    clearance_m: float,
) -> bool:
    for blocked_position_xy in list(blocked_positions_xy or []):
        if not isinstance(blocked_position_xy, Sequence) or len(blocked_position_xy) < 2:
            continue
        if _route_min_distance_to_point_m(route_points, blocked_position_xy) <= float(clearance_m):
            return True
    return False


def _expand_specific_lane_segments(
    *,
    global_planner,
    road_id: object,
    lane_id: int,
) -> List[Tuple[str, int]]:
    if hasattr(global_planner, "segment_keys_for_road_and_lane"):
        return list(
            global_planner.segment_keys_for_road_and_lane(
                road_id=road_id,
                lane_id=int(lane_id),
            )
        )
    return []


def _expand_same_direction_road_segments(
    *,
    global_planner,
    road_id: object,
    ego_direction: str,
) -> List[Tuple[str, int]]:
    if hasattr(global_planner, "segment_keys_for_road"):
        return list(
            global_planner.segment_keys_for_road(
                road_id=road_id,
                direction=str(ego_direction),
            )
        )
    return []


def _select_reroute_start_waypoint(
    *,
    start_waypoint,
    blocked_segments: Sequence[Tuple[object, int]] | None,
):
    if start_waypoint is None:
        return None
    blocked_segment_keys = {
        (str(segment_key[0]), int(segment_key[1]))
        for segment_key in list(blocked_segments or [])
        if isinstance(segment_key, (list, tuple)) and len(segment_key) >= 2
    }
    if len(blocked_segment_keys) == 0:
        return start_waypoint

    candidate_waypoints = canonical_lane_waypoints(start_waypoint)
    if len(candidate_waypoints) == 0:
        candidate_waypoints = [start_waypoint]
    current_lane_id = int(canonical_lane_id_for_waypoint(start_waypoint))

    def _candidate_sort_key(candidate_waypoint) -> Tuple[int, int]:
        candidate_lane_id = int(canonical_lane_id_for_waypoint(candidate_waypoint))
        is_blocked = (
            str(_road_segment_key_from_waypoint(candidate_waypoint)),
            int(candidate_lane_id),
        ) in blocked_segment_keys
        return (
            1 if bool(is_blocked) else 0,
            abs(int(candidate_lane_id) - int(current_lane_id)),
        )

    for candidate_waypoint in sorted(candidate_waypoints, key=_candidate_sort_key):
        candidate_segment_key = (
            str(_road_segment_key_from_waypoint(candidate_waypoint)),
            int(canonical_lane_id_for_waypoint(candidate_waypoint)),
        )
        if candidate_segment_key not in blocked_segment_keys:
            return candidate_waypoint
    return start_waypoint


def _route_cumulative_distances(route_points: Sequence[Sequence[float]]) -> List[float]:
    if len(route_points) == 0:
        return []
    cumulative = [0.0]
    for idx in range(1, len(route_points)):
        prev_point = route_points[idx - 1]
        current_point = route_points[idx]
        cumulative.append(
            float(cumulative[-1])
            + float(
                math.hypot(
                    float(current_point[0]) - float(prev_point[0]),
                    float(current_point[1]) - float(prev_point[1]),
                )
            )
        )
    return cumulative


def _sample_route_points(route_points: Sequence[Sequence[float]], *, sample_count: int = 40) -> List[List[float]]:
    valid_route_points = [
        [float(point[0]), float(point[1])]
        for point in list(route_points or [])
        if isinstance(point, Sequence) and len(point) >= 2
    ]
    if len(valid_route_points) <= 1:
        return list(valid_route_points)
    cumulative = _route_cumulative_distances(valid_route_points)
    total_length_m = float(cumulative[-1])
    if total_length_m <= 1.0e-6:
        return [list(valid_route_points[0])]

    sample_points: List[List[float]] = []
    requested_samples = max(2, int(sample_count))
    for sample_idx in range(requested_samples):
        target_arc_m = float(sample_idx) * float(total_length_m) / float(requested_samples - 1)
        upper_index = 1
        while upper_index < len(cumulative) and float(cumulative[upper_index]) < float(target_arc_m):
            upper_index += 1
        if upper_index >= len(cumulative):
            sample_points.append(list(valid_route_points[-1]))
            continue
        lower_index = max(0, int(upper_index) - 1)
        lower_arc_m = float(cumulative[lower_index])
        upper_arc_m = float(cumulative[upper_index])
        if upper_arc_m <= lower_arc_m + 1.0e-9:
            sample_points.append(list(valid_route_points[upper_index]))
            continue
        alpha = float(target_arc_m - lower_arc_m) / float(upper_arc_m - lower_arc_m)
        lower_point = valid_route_points[lower_index]
        upper_point = valid_route_points[upper_index]
        sample_points.append(
            [
                float(lower_point[0]) + float(alpha) * (float(upper_point[0]) - float(lower_point[0])),
                float(lower_point[1]) + float(alpha) * (float(upper_point[1]) - float(lower_point[1])),
            ]
        )
    return sample_points


def _routes_effectively_same(
    reference_route_points: Sequence[Sequence[float]] | None,
    candidate_route_points: Sequence[Sequence[float]] | None,
    *,
    mean_distance_threshold_m: float = 0.35,
    max_distance_threshold_m: float = 1.0,
) -> bool:
    reference_samples = _sample_route_points(reference_route_points or [])
    candidate_samples = _sample_route_points(candidate_route_points or [])
    if len(reference_samples) < 2 or len(candidate_samples) < 2:
        return False
    sample_count = min(len(reference_samples), len(candidate_samples))
    if sample_count < 2:
        return False

    sample_distances_m: List[float] = []
    for idx in range(sample_count):
        ref_point = reference_samples[idx]
        cand_point = candidate_samples[idx]
        sample_distances_m.append(
            float(
                math.hypot(
                    float(cand_point[0]) - float(ref_point[0]),
                    float(cand_point[1]) - float(ref_point[1]),
                )
            )
        )
    if len(sample_distances_m) == 0:
        return False
    return (
        float(sum(sample_distances_m) / len(sample_distances_m)) <= float(mean_distance_threshold_m)
        and float(max(sample_distances_m)) <= float(max_distance_threshold_m)
    )


def _advance_waypoint_forward(waypoint, *, distance_m: float, step_m: float = 2.0):
    current_waypoint = waypoint
    remaining_distance_m = max(0.0, float(distance_m))
    while current_waypoint is not None and remaining_distance_m > 1.0e-3:
        next_fn = getattr(current_waypoint, "next", None)
        if not callable(next_fn):
            break
        try:
            next_candidates = list(next_fn(min(float(step_m), float(remaining_distance_m))) or [])
        except Exception:
            break
        if len(next_candidates) == 0:
            break
        current_lane_id = int(getattr(current_waypoint, "lane_id", 0))
        preferred_candidates = [
            candidate
            for candidate in next_candidates
            if int(getattr(candidate, "lane_id", 0)) == int(current_lane_id)
        ]
        selection_pool = preferred_candidates if len(preferred_candidates) > 0 else next_candidates
        current_waypoint = selection_pool[0]
        remaining_distance_m -= min(float(step_m), float(remaining_distance_m))
    return current_waypoint if current_waypoint is not None else waypoint


def _select_bypass_waypoint(
    *,
    resolved_messages: Sequence[Mapping[str, object]],
    blocked_segment_keys: Sequence[Tuple[object, int]],
    world_map,
    carla,
):
    blocked_segment_key_set = {
        (str(segment_key[0]), int(segment_key[1]))
        for segment_key in list(blocked_segment_keys or [])
        if isinstance(segment_key, (list, tuple)) and len(segment_key) >= 2
    }
    for message in list(resolved_messages or []):
        position_xy = _coerce_position_xy(message.get("position", None))
        if position_xy is None:
            continue
        blocked_waypoint = world_map.get_waypoint(
            carla.Location(x=float(position_xy[0]), y=float(position_xy[1]), z=0.0),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if blocked_waypoint is None:
            continue
        blocked_lane_id = int(canonical_lane_id_for_waypoint(blocked_waypoint))
        lane_candidates = list(canonical_lane_waypoints(blocked_waypoint))
        if len(lane_candidates) == 0:
            lane_candidates = [blocked_waypoint]
        unblocked_candidates = [
            candidate_waypoint
            for candidate_waypoint in lane_candidates
            if (
                str(_road_segment_key_from_waypoint(candidate_waypoint)),
                int(canonical_lane_id_for_waypoint(candidate_waypoint)),
            ) not in blocked_segment_key_set
            and int(canonical_lane_id_for_waypoint(candidate_waypoint)) != int(blocked_lane_id)
        ]
        if len(unblocked_candidates) == 0:
            continue
        unblocked_candidates.sort(
            key=lambda candidate_waypoint: abs(
                int(canonical_lane_id_for_waypoint(candidate_waypoint)) - int(blocked_lane_id)
            )
        )
        return _advance_waypoint_forward(
            unblocked_candidates[0],
            distance_m=15.0,
            step_m=2.0,
        )
    return None


def resolve_lane_closure_segments(
    *,
    messages: Sequence[Mapping[str, object]],
    world_map,
    carla,
    global_planner,
    ego_transform,
) -> Dict[str, object]:
    ego_waypoint = world_map.get_waypoint(
        ego_transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    ego_direction = direction_key(int(getattr(ego_waypoint, "lane_id", 1))) if ego_waypoint is not None else "positive"

    blocked_segments: Dict[Tuple[str, int], float] = {}
    resolved_messages: List[dict] = []
    blocked_positions_xy: List[List[float]] = []

    for message in lane_closure_messages(messages):
        raw_position_xy = _coerce_position_xy(message.get("position", None))
        inferred_waypoint = None
        inferred_segment_key = None
        inferred_lane_id = None
        snapped_graph_query = None
        snapped_carla_key = None
        if raw_position_xy is not None:
            inferred_waypoint = world_map.get_waypoint(
                carla.Location(x=float(raw_position_xy[0]), y=float(raw_position_xy[1]), z=0.0),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if inferred_waypoint is not None:
                inferred_segment_key = _road_segment_key_from_waypoint(inferred_waypoint)
                inferred_lane_id = int(canonical_lane_id_for_waypoint(inferred_waypoint))
            if hasattr(global_planner, "nearest_waypoint_query"):
                snapped_graph_query = global_planner.nearest_waypoint_query(
                    x_m=float(raw_position_xy[0]),
                    y_m=float(raw_position_xy[1]),
                )
            if hasattr(global_planner, "nearest_waypoint_carla_key"):
                try:
                    snapped_carla_key = global_planner.nearest_waypoint_carla_key(
                        x_m=float(raw_position_xy[0]),
                        y_m=float(raw_position_xy[1]),
                    )
                except TypeError:
                    snapped_carla_key = global_planner.nearest_waypoint_carla_key(
                        float(raw_position_xy[0]),
                        float(raw_position_xy[1]),
                    )

        raw_road_id = message.get("road_id", None)
        raw_section_id = message.get("section_id", None)
        raw_lane_id = message.get("lane_id", None)
        expanded_segments: List[Tuple[str, int]] = []
        resolved_road_id = raw_road_id
        resolved_section_id = raw_section_id
        resolved_lane_id = raw_lane_id
        resolved_carla_lane_id = raw_lane_id

        if raw_road_id is not None and raw_lane_id is None:
            expanded_segments = _expand_same_direction_road_segments(
                global_planner=global_planner,
                road_id=raw_road_id,
                ego_direction=str(ego_direction),
            )
        elif snapped_graph_query is not None and snapped_carla_key is not None:
            expanded_segments = [
                (str(getattr(snapped_graph_query, "road_id", "")), int(getattr(snapped_graph_query, "lane_id", 0)))
            ]
            resolved_road_id = int(snapped_carla_key[0])
            resolved_section_id = int(snapped_carla_key[1])
            resolved_lane_id = int(getattr(snapped_graph_query, "lane_id", 0))
            resolved_carla_lane_id = int(snapped_carla_key[2])
        elif raw_road_id is not None and raw_lane_id is not None:
            if hasattr(global_planner, "segment_keys_for_raw_carla_lane"):
                expanded_segments = list(
                    global_planner.segment_keys_for_raw_carla_lane(
                        road_id=raw_road_id,
                        section_id=_section_numeric_id(raw_section_id),
                        lane_id=int(raw_lane_id),
                    )
                )
            if len(expanded_segments) == 0:
                expanded_segments = _expand_specific_lane_segments(
                    global_planner=global_planner,
                    road_id=raw_road_id,
                    lane_id=int(raw_lane_id),
                )
            if len(expanded_segments) > 0:
                resolved_lane_id = int(expanded_segments[0][1])
        elif inferred_segment_key is not None and inferred_lane_id is not None:
            expanded_segments = [(str(inferred_segment_key), int(inferred_lane_id))]
            resolved_road_id = int(getattr(inferred_waypoint, "road_id", 0))
            resolved_section_id = int(getattr(inferred_waypoint, "section_id", 0))
            resolved_lane_id = int(inferred_lane_id)
            resolved_carla_lane_id = int(getattr(inferred_waypoint, "lane_id", 0))

        if len(expanded_segments) == 0 and inferred_segment_key is not None and inferred_lane_id is not None:
            expanded_segments = [(str(inferred_segment_key), int(inferred_lane_id))]
            resolved_road_id = int(getattr(inferred_waypoint, "road_id", 0))
            resolved_section_id = int(getattr(inferred_waypoint, "section_id", 0))
            resolved_lane_id = int(inferred_lane_id)
            resolved_carla_lane_id = int(getattr(inferred_waypoint, "lane_id", 0))

        if len(expanded_segments) == 0:
            continue

        for segment_key in expanded_segments:
            blocked_segments[(str(segment_key[0]), int(segment_key[1]))] = float(DEFAULT_REROUTE_PENALTY)
        if raw_position_xy is not None:
            blocked_positions_xy.append([float(raw_position_xy[0]), float(raw_position_xy[1])])
        resolved_messages.append(
            {
                "id": str(message.get("id", "")).strip(),
                "type": "lane_closure",
                "position": None if raw_position_xy is None else [float(raw_position_xy[0]), float(raw_position_xy[1])],
                "road_id": resolved_road_id,
                "section_id": resolved_section_id,
                "lane_id": resolved_lane_id,
                "carla_lane_id": resolved_carla_lane_id,
                "expanded_segments": [
                    [str(segment[0]), int(segment[1])]
                    for segment in expanded_segments
                ],
            }
        )

    return {
        "ego_road_id": None if ego_waypoint is None else _road_segment_key_from_waypoint(ego_waypoint),
        "ego_lane_id": None if ego_waypoint is None else int(canonical_lane_id_for_waypoint(ego_waypoint)),
        "blocked_segments": dict(blocked_segments),
        "blocked_positions_xy": list(blocked_positions_xy),
        "resolved_messages": list(resolved_messages),
    }


def reroute_from_lane_closure_messages(
    *,
    messages: Sequence[Mapping[str, object]],
    world_map,
    carla,
    global_planner,
    ego_transform,
    goal_location,
    penalty_value: float = DEFAULT_REROUTE_PENALTY,
    current_route_points: Sequence[Sequence[float]] | None = None,
) -> Dict[str, object]:
    resolution = resolve_lane_closure_segments(
        messages=messages,
        world_map=world_map,
        carla=carla,
        global_planner=global_planner,
        ego_transform=ego_transform,
    )
    blocked_segments = dict(resolution.get("blocked_segments", {}))
    blocked_positions_xy = [
        [float(point_xy[0]), float(point_xy[1])]
        for point_xy in list(resolution.get("blocked_positions_xy", []))
        if isinstance(point_xy, Sequence) and len(point_xy) >= 2
    ]
    if len(blocked_segments) == 0:
        return {
            "route_summary": None,
            "route_points": [],
            "handled_message_ids": [],
            "resolved_messages": [],
            "ego_road_id": resolution.get("ego_road_id", None),
            "ego_lane_id": resolution.get("ego_lane_id", None),
        }

    blocked_segment_keys = [
        (str(segment_key[0]), int(segment_key[1]))
        for segment_key in blocked_segments.keys()
    ]
    segment_penalties = {
        (str(segment_key[0]), int(segment_key[1])): float(penalty_value)
        for segment_key in blocked_segment_keys
    }
    ego_start_waypoint = world_map.get_waypoint(
        ego_transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    start_waypoint = _select_reroute_start_waypoint(
        start_waypoint=ego_start_waypoint,
        blocked_segments=blocked_segment_keys,
    )
    start_location = getattr(getattr(start_waypoint, "transform", None), "location", None)
    if start_location is None:
        start_location = ego_transform.location
    goal_waypoint = world_map.get_waypoint(
        goal_location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    blocked_lane_ids = sorted({int(segment_key[1]) for segment_key in blocked_segment_keys})
    blocked_raw_lanes: List[Tuple[int, int | None, int]] = []
    seen_blocked_raw_lanes: set[Tuple[int, int | None, int]] = set()
    for resolved_message in list(resolution.get("resolved_messages", []) or []):
        raw_road_id = _road_numeric_id(resolved_message.get("road_id", None))
        raw_section_id = _section_numeric_id(resolved_message.get("section_id", None))
        raw_lane_id = resolved_message.get("carla_lane_id", resolved_message.get("lane_id", None))
        try:
            normalized_raw_lane_id = int(raw_lane_id)
        except Exception:
            normalized_raw_lane_id = None
        if raw_road_id is None or normalized_raw_lane_id is None:
            continue
        blocked_raw_lane = (
            int(raw_road_id),
            None if raw_section_id is None else int(raw_section_id),
            int(normalized_raw_lane_id),
        )
        if blocked_raw_lane in seen_blocked_raw_lanes:
            continue
        seen_blocked_raw_lanes.add(blocked_raw_lane)
        blocked_raw_lanes.append(blocked_raw_lane)

    def _route_requires_retry(route_summary: object | None, *, phase_name: str) -> bool:
        if not bool(getattr(route_summary, "route_found", False)):
            return True
        route_points = _route_points_from_summary(route_summary)
        if _routes_effectively_same(current_route_points, route_points):
            print(
                "[BEHAVIOR] reroute warning: "
                f"{phase_name} returned a route that is effectively unchanged; retrying with a stronger avoidance mode."
            )
            return True
        if hasattr(global_planner, "route_penalty_from_waypoints"):
            route_penalty = float(
                global_planner.route_penalty_from_waypoints(
                    route_points,
                    segment_penalties=segment_penalties,
                )
            )
            if route_penalty > 0.0:
                print(
                    "[BEHAVIOR] reroute warning: "
                    f"{phase_name} still uses the blocked lane segments; retrying with a stronger avoidance mode."
                )
                return True
        if len(blocked_positions_xy) == 0:
            return False
        if _route_overlaps_blocked_positions(
            route_points,
            blocked_positions_xy,
            clearance_m=2.25,
        ):
            print(
                "[BEHAVIOR] reroute warning: "
                f"{phase_name} still overlaps the blocked workzone position; retrying with a stronger avoidance mode."
            )
            return True
        return False

    route_summary = None
    if len(blocked_raw_lanes) > 0 and hasattr(global_planner, "plan_route_carla_with_blocked_lanes"):
        route_summary = global_planner.plan_route_carla_with_blocked_lanes(
            start_location=start_location,
            goal_location=goal_location,
            blocked_raw_lanes=blocked_raw_lanes,
            replace_stored_route=True,
            fallback_start_xy=[
                float(start_location.x),
                float(start_location.y),
            ],
            fallback_goal_xy=[
                float(goal_location.x),
                float(goal_location.y),
            ],
        )
    if _route_requires_retry(route_summary, phase_name="blocked-CARLA-lane reroute"):
        route_summary = global_planner.plan_route_astar_blocking_segments(
            start_xy=[
                float(start_location.x),
                float(start_location.y),
            ],
            goal_xy=[
                float(goal_location.x),
                float(goal_location.y),
            ],
            blocked_segments=blocked_segment_keys,
            replace_stored_route=True,
            start_waypoint=start_waypoint,
            goal_waypoint=goal_waypoint,
        )
    if _route_requires_retry(route_summary, phase_name="blocked-segment reroute"):
        route_summary = global_planner.plan_route_astar_avoiding_points(
            start_xy=[
                float(start_location.x),
                float(start_location.y),
            ],
            goal_xy=[
                float(goal_location.x),
                float(goal_location.y),
            ],
            blocked_points_xy=list(blocked_positions_xy),
            blocked_lane_ids=blocked_lane_ids if len(blocked_lane_ids) > 0 else None,
            block_radius_m=8.0,
            replace_stored_route=True,
            start_waypoint=start_waypoint,
            goal_waypoint=goal_waypoint,
        )
    if _route_requires_retry(route_summary, phase_name="point-avoidance reroute"):
        route_summary = global_planner.plan_route_from_locations_with_segment_penalties(
            start_location=start_location,
            goal_location=goal_location,
            segment_penalties=segment_penalties,
            replace_stored_route=True,
            start_waypoint=start_waypoint,
        )
    if _route_requires_retry(route_summary, phase_name="segment-penalty reroute"):
        bypass_waypoint = _select_bypass_waypoint(
            resolved_messages=resolution.get("resolved_messages", []),
            blocked_segment_keys=blocked_segment_keys,
            world_map=world_map,
            carla=carla,
        )
        bypass_location = getattr(getattr(bypass_waypoint, "transform", None), "location", None)
        if bypass_location is not None and hasattr(global_planner, "plan_route_from_locations_via_locations"):
            route_summary = global_planner.plan_route_from_locations_via_locations(
                start_location=start_location,
                goal_location=goal_location,
                intermediate_locations=[bypass_location],
                replace_stored_route=True,
            )
    route_points = _route_points_from_summary(route_summary)
    if (
        bool(getattr(route_summary, "route_found", False))
        and len(route_points) > 0
        and float(
            math.hypot(
                float(route_points[0][0]) - float(start_location.x),
                float(route_points[0][1]) - float(start_location.y),
            )
        ) > 1.0e-3
    ):
        route_points = [[float(start_location.x), float(start_location.y)]] + list(route_points)
        if route_summary is not None:
            setattr(route_summary, "route_waypoints", list(route_points))
    if (
        bool(getattr(route_summary, "route_found", False))
        and hasattr(global_planner, "route_penalty_from_waypoints")
        and float(
            global_planner.route_penalty_from_waypoints(
                route_points,
                segment_penalties=segment_penalties,
            )
        ) > 0.0
    ):
        print(
            "[BEHAVIOR] reroute failed: generated route still uses the blocked lane segments."
        )
        route_summary = None
        route_points = []
    if (
        bool(getattr(route_summary, "route_found", False))
        and _route_overlaps_blocked_positions(
            route_points,
            blocked_positions_xy,
            clearance_m=2.25,
        )
    ):
        print(
            "[BEHAVIOR] reroute failed: generated route still overlaps the blocked workzone position."
        )
        route_summary = None
        route_points = []
    if not bool(getattr(route_summary, "route_found", False)):
        print(
            "[BEHAVIOR] reroute failed: "
            f"{str(getattr(route_summary, 'debug_reason', '')).strip() or 'no route found'}"
        )
    return {
        "route_summary": route_summary if bool(getattr(route_summary, "route_found", False)) else None,
        "route_points": route_points,
        "handled_message_ids": [
            str(message.get("id", "")).strip()
            for message in list(resolution.get("resolved_messages", []))
            if str(message.get("id", "")).strip()
        ],
        "resolved_messages": list(resolution.get("resolved_messages", [])),
        "ego_road_id": resolution.get("ego_road_id", None),
        "ego_lane_id": resolution.get("ego_lane_id", None),
    }

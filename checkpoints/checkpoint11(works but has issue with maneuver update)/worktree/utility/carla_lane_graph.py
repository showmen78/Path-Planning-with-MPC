"""
Shared CARLA lane-center waypoint graph helpers.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping

import numpy as np


def direction_key(carla_lane_id: int) -> str:
    return "positive" if int(carla_lane_id) > 0 else "negative"


def round_xy(x_m: float, y_m: float) -> tuple[float, float]:
    return round(float(x_m), 3), round(float(y_m), 3)


def carla_waypoint_graph_key(waypoint) -> tuple[int, int, int, float]:
    return (
        int(waypoint.road_id),
        int(waypoint.section_id),
        int(waypoint.lane_id),
        round(float(getattr(waypoint, "s", 0.0)), 3),
    )


def _carla_waypoint_group_key(waypoint) -> tuple[int, int, str]:
    return int(waypoint.road_id), int(waypoint.section_id), direction_key(int(waypoint.lane_id))


def is_driving_waypoint(waypoint) -> bool:
    if waypoint is None:
        return False
    lane_type = getattr(waypoint, "lane_type", None)
    if lane_type is None:
        return True
    lane_type_name = getattr(lane_type, "name", lane_type)
    normalized_name = str(lane_type_name).strip().upper()
    return normalized_name.endswith("DRIVING")


def _same_lane_group(base_waypoint, candidate_waypoint) -> bool:
    if base_waypoint is None or candidate_waypoint is None:
        return False
    if not is_driving_waypoint(candidate_waypoint):
        return False

    base_lane_id = int(getattr(base_waypoint, "lane_id", 0))
    candidate_lane_id = int(getattr(candidate_waypoint, "lane_id", 0))
    if base_lane_id == 0 or candidate_lane_id == 0:
        return False
    if base_lane_id * candidate_lane_id < 0:
        return False

    return (
        int(getattr(base_waypoint, "road_id", 0)) == int(getattr(candidate_waypoint, "road_id", 0))
        and int(getattr(base_waypoint, "section_id", 0)) == int(getattr(candidate_waypoint, "section_id", 0))
    )


def canonical_lane_waypoints(waypoint) -> List[object]:
    if waypoint is None:
        return []

    rightmost_waypoint = waypoint
    while True:
        right_waypoint = rightmost_waypoint.get_right_lane()
        if not _same_lane_group(waypoint, right_waypoint):
            break
        rightmost_waypoint = right_waypoint

    lanes: List[object] = [rightmost_waypoint]
    current_waypoint = rightmost_waypoint
    while True:
        left_waypoint = current_waypoint.get_left_lane()
        if not _same_lane_group(waypoint, left_waypoint):
            break
        lanes.append(left_waypoint)
        current_waypoint = left_waypoint
    return lanes


def canonical_lane_ids_for_waypoint(waypoint) -> List[int]:
    lane_waypoints = canonical_lane_waypoints(waypoint)
    return list(range(1, len(lane_waypoints) + 1))


def canonical_lane_id_for_waypoint(waypoint) -> int:
    lane_waypoints = canonical_lane_waypoints(waypoint)
    if len(lane_waypoints) == 0:
        raw_lane_id = int(getattr(waypoint, "lane_id", 0)) if waypoint is not None else 0
        return max(1, abs(raw_lane_id))

    for lane_index, lane_waypoint in enumerate(lane_waypoints, start=1):
        if (
            int(getattr(lane_waypoint, "road_id", 0)) == int(getattr(waypoint, "road_id", 0))
            and int(getattr(lane_waypoint, "section_id", 0)) == int(getattr(waypoint, "section_id", 0))
            and int(getattr(lane_waypoint, "lane_id", 0)) == int(getattr(waypoint, "lane_id", 0))
        ):
            return int(lane_index)

    raw_lane_id = int(getattr(waypoint, "lane_id", 0)) if waypoint is not None else 0
    return max(1, abs(raw_lane_id))


def canonical_lane_waypoint_for_lane_id(waypoint, target_lane_id: int):
    lane_waypoints = canonical_lane_waypoints(waypoint)
    if len(lane_waypoints) == 0:
        return waypoint
    lane_index = int(target_lane_id) - 1
    if 0 <= lane_index < len(lane_waypoints):
        return lane_waypoints[lane_index]
    return waypoint


def _internal_lane_id(waypoint, lane_ids_by_group: Mapping[tuple[int, int, str], set[int]]) -> int:
    del lane_ids_by_group
    return int(canonical_lane_id_for_waypoint(waypoint))


def build_lane_center_waypoints(map_obj, carla, sample_distance_m: float) -> tuple[list[dict], dict]:
    sampled_waypoints = [
        waypoint
        for waypoint in map_obj.generate_waypoints(float(sample_distance_m))
        if waypoint.lane_type == carla.LaneType.Driving and int(waypoint.lane_id) != 0
    ]
    lane_ids_by_group: Dict[tuple[int, int, str], set[int]] = {}
    for waypoint in sampled_waypoints:
        lane_ids_by_group.setdefault(_carla_waypoint_group_key(waypoint), set()).add(int(waypoint.lane_id))

    valid_waypoints = []
    for waypoint in sampled_waypoints:
        internal_lane_id = _internal_lane_id(waypoint, lane_ids_by_group)
        valid_waypoints.append(
            {
                "carla_waypoint": waypoint,
                "carla_waypoint_key": carla_waypoint_graph_key(waypoint),
                "position": [float(waypoint.transform.location.x), float(waypoint.transform.location.y)],
                "heading_rad": math.radians(float(waypoint.transform.rotation.yaw)),
                "lane_id": int(internal_lane_id),
                "road_id": f"{int(waypoint.road_id)}:{int(waypoint.section_id)}",
                "direction": direction_key(int(waypoint.lane_id)),
                "lane_width_m": float(waypoint.lane_width),
                "is_intersection": bool(waypoint.is_junction),
                "maneuver": "straight",
            }
        )

    for item in valid_waypoints:
        carla_waypoint = item["carla_waypoint"]
        next_candidates = [
            waypoint
            for waypoint in carla_waypoint.next(float(sample_distance_m))
            if waypoint.lane_type == carla.LaneType.Driving
        ]
        if next_candidates:
            successor_positions_by_key: Dict[tuple[float, float], list[float]] = {}
            successor_keys = []
            for next_waypoint in next_candidates:
                successor_xy = [
                    float(next_waypoint.transform.location.x),
                    float(next_waypoint.transform.location.y),
                ]
                successor_positions_by_key[round_xy(successor_xy[0], successor_xy[1])] = successor_xy
                successor_key = carla_waypoint_graph_key(next_waypoint)
                if successor_key not in successor_keys:
                    successor_keys.append(successor_key)
            item["successors"] = list(successor_positions_by_key.values())
            item["successor_keys"] = list(successor_keys)

            preferred_next_candidates = [
                waypoint
                for waypoint in next_candidates
                if direction_key(int(waypoint.lane_id)) == direction_key(int(carla_waypoint.lane_id))
            ]
            selection_pool = preferred_next_candidates if len(preferred_next_candidates) > 0 else next_candidates
            next_waypoint = min(
                selection_pool,
                key=lambda waypoint: abs(
                    math.atan2(
                        math.sin(
                            math.radians(float(waypoint.transform.rotation.yaw))
                            - math.radians(float(carla_waypoint.transform.rotation.yaw))
                        ),
                        math.cos(
                            math.radians(float(waypoint.transform.rotation.yaw))
                            - math.radians(float(carla_waypoint.transform.rotation.yaw))
                        ),
                    )
                ),
            )
            item["next"] = [
                float(next_waypoint.transform.location.x),
                float(next_waypoint.transform.location.y),
            ]
            item["next_key"] = carla_waypoint_graph_key(next_waypoint)

    lane_width_m = (
        float(np.median([float(item["lane_width_m"]) for item in valid_waypoints]))
        if valid_waypoints
        else 4.0
    )
    lane_count = max(int(item["lane_id"]) for item in valid_waypoints) if valid_waypoints else 1
    road_cfg = {
        "lane_width_m": float(lane_width_m),
        "lane_count": int(lane_count),
    }
    return valid_waypoints, road_cfg

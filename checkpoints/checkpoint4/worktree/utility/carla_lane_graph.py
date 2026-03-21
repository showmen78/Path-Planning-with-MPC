"""
Shared CARLA lane-center waypoint graph helpers.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping

import numpy as np


def direction_key(carla_lane_id: int) -> str:
    return "positive" if int(carla_lane_id) > 0 else "negative"


def round_xy(x_m: float, y_m: float) -> tuple[float, float]:
    return round(float(x_m), 3), round(float(y_m), 3)


def _carla_waypoint_group_key(waypoint) -> tuple[int, int, str]:
    return int(waypoint.road_id), int(waypoint.section_id), direction_key(int(waypoint.lane_id))


def _internal_lane_id(waypoint, lane_ids_by_group: Mapping[tuple[int, int, str], set[int]]) -> int:
    group_key = _carla_waypoint_group_key(waypoint)
    lane_id = int(waypoint.lane_id)
    if lane_id > 0:
        return int(lane_id)
    max_abs_lane_id = max(abs(int(value)) for value in lane_ids_by_group[group_key])
    return int(max_abs_lane_id - abs(lane_id) + 1)


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
            and direction_key(int(waypoint.lane_id)) == direction_key(int(carla_waypoint.lane_id))
        ]
        if next_candidates:
            next_waypoint = min(
                next_candidates,
                key=lambda waypoint: abs(
                    math.radians(float(waypoint.transform.rotation.yaw))
                    - math.radians(float(carla_waypoint.transform.rotation.yaw))
                ),
            )
            item["next"] = [
                float(next_waypoint.transform.location.x),
                float(next_waypoint.transform.location.y),
            ]

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

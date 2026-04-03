"""
Traffic-light stop helpers for the rule-based behavior planner.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Sequence

from utility.carla_lane_graph import canonical_lane_id_for_waypoint


def normalize_signal_state(signal_state: object) -> str:
    raw_name = str(signal_state or "").strip().upper()
    if "." in raw_name:
        raw_name = raw_name.rsplit(".", 1)[-1]
    if raw_name in {"GREEN"}:
        return "green"
    if raw_name in {"YELLOW", "AMBER"}:
        return "yellow"
    if raw_name in {"RED"}:
        return "red"
    return "unknown"


def _actor_transform(actor: Any):
    direct_transform = getattr(actor, "transform", None)
    if direct_transform is not None:
        return direct_transform
    get_transform_fn = getattr(actor, "get_transform", None)
    if callable(get_transform_fn):
        try:
            return get_transform_fn()
        except Exception:
            return None
    return None


def _traffic_light_actor_name(actor: Any) -> str:
    if actor is None:
        return ""
    raw_attributes = getattr(actor, "attributes", None)
    if isinstance(raw_attributes, Mapping):
        for key in ("name", "role_name", "object_name"):
            normalized_value = str(raw_attributes.get(key, "")).strip()
            if normalized_value:
                return normalized_value
    return str(getattr(actor, "type_id", "")).strip()


def _iter_world_traffic_lights(world) -> Iterable[Any]:
    for actor in list(world.get_actors() if hasattr(world, "get_actors") else []):
        if callable(getattr(actor, "get_state", None)):
            yield actor


def _traffic_light_stop_waypoints(actor: Any) -> list[Any]:
    stop_waypoints: list[Any] = []
    for method_name in ("get_stop_waypoints", "get_affected_lane_waypoints"):
        method = getattr(actor, method_name, None)
        if not callable(method):
            continue
        try:
            returned_waypoints = list(method() or [])
        except Exception:
            continue
        for waypoint in returned_waypoints:
            if waypoint is not None:
                stop_waypoints.append(waypoint)
        if len(stop_waypoints) > 0:
            break
    return stop_waypoints


def _signal_distance_from_actor(
    *,
    ego_transform,
    actor: Any,
) -> float | None:
    ego_location = getattr(ego_transform, "location", None)
    actor_transform = _actor_transform(actor)
    actor_location = getattr(actor_transform, "location", None)
    if ego_location is None or actor_location is None:
        return None
    return float(
        math.hypot(
            float(ego_location.x) - float(actor_location.x),
            float(ego_location.y) - float(actor_location.y),
        )
    )


def find_relevant_signal_context(
    *,
    world,
    ego_vehicle,
    ego_transform,
    stop_target: Mapping[str, object] | None,
    max_stop_waypoint_match_distance_m: float = 12.0,
    max_actor_position_match_distance_m: float = 40.0,
) -> Dict[str, object]:
    default_context: Dict[str, object] = {
        "signal_found": False,
        "signal_state": "unknown",
        "signal_distance_m": None,
        "signal_actor_id": None,
        "signal_actor_name": "",
        "signal_source": "none",
        "signal_match_distance_m": None,
    }

    ego_associated_actor = None
    get_traffic_light_fn = getattr(ego_vehicle, "get_traffic_light", None)
    if callable(get_traffic_light_fn):
        try:
            ego_associated_actor = get_traffic_light_fn()
        except Exception:
            ego_associated_actor = None

    stop_target_xy = None
    stop_target_distance_m = None
    if isinstance(stop_target, Mapping):
        try:
            stop_target_xy = (
                float(stop_target.get("x_m", 0.0)),
                float(stop_target.get("y_m", 0.0)),
            )
            stop_target_distance_m = max(
                0.0,
                float(stop_target.get("distance_m", 0.0)),
            )
        except Exception:
            stop_target_xy = None
            stop_target_distance_m = None

    candidate_actors: list[Any] = []
    seen_candidate_ids: set[int] = set()
    for candidate_actor in [ego_associated_actor, *_iter_world_traffic_lights(world)]:
        if candidate_actor is None:
            continue
        candidate_id = id(candidate_actor)
        if candidate_id in seen_candidate_ids:
            continue
        seen_candidate_ids.add(candidate_id)
        candidate_actors.append(candidate_actor)

    best_candidate: Dict[str, object] | None = None
    for candidate_actor in candidate_actors:
        candidate_state = "unknown"
        get_state_fn = getattr(candidate_actor, "get_state", None)
        if callable(get_state_fn):
            try:
                candidate_state = normalize_signal_state(get_state_fn())
            except Exception:
                candidate_state = "unknown"

        actor_name = _traffic_light_actor_name(candidate_actor)
        signal_distance_m = _signal_distance_from_actor(
            ego_transform=ego_transform,
            actor=candidate_actor,
        )
        match_distance_m = float("inf")
        signal_source = "actor_position_match"

        if stop_target_xy is not None:
            stop_waypoints = _traffic_light_stop_waypoints(candidate_actor)
            if len(stop_waypoints) > 0:
                waypoint_distances = []
                for stop_waypoint in stop_waypoints:
                    stop_location = getattr(
                        getattr(stop_waypoint, "transform", None),
                        "location",
                        None,
                    )
                    if stop_location is None:
                        continue
                    waypoint_distances.append(
                        math.hypot(
                            float(stop_location.x) - float(stop_target_xy[0]),
                            float(stop_location.y) - float(stop_target_xy[1]),
                        )
                    )
                if len(waypoint_distances) > 0:
                    match_distance_m = min(float(distance) for distance in waypoint_distances)
                    signal_source = "stop_waypoint_match"
            if not math.isfinite(match_distance_m):
                actor_transform = _actor_transform(candidate_actor)
                actor_location = getattr(actor_transform, "location", None)
                if actor_location is not None:
                    match_distance_m = float(
                        math.hypot(
                            float(actor_location.x) - float(stop_target_xy[0]),
                            float(actor_location.y) - float(stop_target_xy[1]),
                        )
                    )
        elif signal_distance_m is not None:
            match_distance_m = float(signal_distance_m)

        if candidate_actor is ego_associated_actor:
            signal_source = "ego_vehicle_association"
            if stop_target_distance_m is not None and (
                not math.isfinite(match_distance_m) or float(match_distance_m) > float(stop_target_distance_m) + 20.0
            ):
                match_distance_m = float(stop_target_distance_m)

        if signal_source == "stop_waypoint_match":
            if float(match_distance_m) > float(max_stop_waypoint_match_distance_m):
                continue
        elif signal_source == "actor_position_match":
            if float(match_distance_m) > float(max_actor_position_match_distance_m):
                continue

        candidate_context = {
            "signal_found": True,
            "signal_state": str(candidate_state),
            "signal_distance_m": (
                None
                if stop_target_distance_m is None
                else float(stop_target_distance_m)
            ) if stop_target_distance_m is not None else signal_distance_m,
            "signal_actor_id": getattr(candidate_actor, "id", None),
            "signal_actor_name": str(actor_name),
            "signal_source": str(signal_source),
            "signal_match_distance_m": (
                None if not math.isfinite(match_distance_m) else float(match_distance_m)
            ),
            "signal_actor": candidate_actor,
        }
        if best_candidate is None:
            best_candidate = candidate_context
            continue

        source_priority = {
            "stop_waypoint_match": 0,
            "ego_vehicle_association": 1,
            "actor_position_match": 2,
        }
        current_priority = (
            int(source_priority.get(str(candidate_context["signal_source"]), 9)),
            float(candidate_context.get("signal_match_distance_m") or float("inf")),
            float(candidate_context.get("signal_distance_m") or float("inf")),
        )
        best_priority = (
            int(source_priority.get(str(best_candidate["signal_source"]), 9)),
            float(best_candidate.get("signal_match_distance_m") or float("inf")),
            float(best_candidate.get("signal_distance_m") or float("inf")),
        )
        if current_priority < best_priority:
            best_candidate = candidate_context

    if best_candidate is None:
        return dict(default_context)
    best_candidate.pop("signal_actor", None)
    return best_candidate


def _route_cum_dists(route_points: Sequence[Sequence[float]]) -> list[float]:
    if len(route_points) == 0:
        return []
    dists = [0.0]
    for idx in range(1, len(route_points)):
        dists.append(
            float(dists[-1])
            + float(
                math.hypot(
                    float(route_points[idx][0]) - float(route_points[idx - 1][0]),
                    float(route_points[idx][1]) - float(route_points[idx - 1][1]),
                )
            )
        )
    return dists


def _project_to_route_arc(
    x_m: float,
    y_m: float,
    route_points: Sequence[Sequence[float]],
    cum_dists: Sequence[float],
) -> float:
    if len(route_points) < 2:
        return 0.0

    best_dist_sq = float("inf")
    best_arc_m = 0.0
    for idx in range(len(route_points) - 1):
        ax_m = float(route_points[idx][0])
        ay_m = float(route_points[idx][1])
        bx_m = float(route_points[idx + 1][0])
        by_m = float(route_points[idx + 1][1])
        dx_m = float(bx_m) - float(ax_m)
        dy_m = float(by_m) - float(ay_m)
        segment_len_sq = dx_m * dx_m + dy_m * dy_m
        if segment_len_sq <= 1.0e-12:
            alpha = 0.0
        else:
            alpha = (
                (float(x_m) - float(ax_m)) * float(dx_m)
                + (float(y_m) - float(ay_m)) * float(dy_m)
            ) / float(segment_len_sq)
            alpha = max(0.0, min(1.0, float(alpha)))
        proj_x_m = float(ax_m) + float(alpha) * float(dx_m)
        proj_y_m = float(ay_m) + float(alpha) * float(dy_m)
        dist_sq = (
            (float(x_m) - float(proj_x_m)) ** 2
            + (float(y_m) - float(proj_y_m)) ** 2
        )
        if dist_sq < best_dist_sq:
            best_dist_sq = float(dist_sq)
            best_arc_m = float(cum_dists[idx]) + float(alpha) * float(math.sqrt(max(segment_len_sq, 0.0)))
    return float(best_arc_m)


def find_stop_target_from_ego(
    *,
    world_map,
    carla,
    ego_transform,
    global_route_points: Sequence[Sequence[float]],
    search_distance_m: float = 100.0,
) -> Dict[str, float] | None:
    route_points = [
        [float(point[0]), float(point[1])]
        for point in list(global_route_points or [])
        if isinstance(point, Sequence) and len(point) >= 2
    ]
    if len(route_points) < 2:
        return None

    ego_location = getattr(ego_transform, "location", None)
    if ego_location is None:
        return None
    ego_x_m = float(getattr(ego_location, "x", 0.0))
    ego_y_m = float(getattr(ego_location, "y", 0.0))
    ego_z_m = float(getattr(ego_location, "z", 0.0))

    route_cum_dists = _route_cum_dists(route_points)
    if len(route_cum_dists) != len(route_points):
        return None
    ego_arc_m = _project_to_route_arc(
        x_m=float(ego_x_m),
        y_m=float(ego_y_m),
        route_points=route_points,
        cum_dists=route_cum_dists,
    )
    max_search_distance_m = max(0.0, float(search_distance_m))

    previous_waypoint = None
    previous_arc_m: float | None = None
    for idx, route_point in enumerate(route_points):
        point_arc_m = float(route_cum_dists[idx])
        if float(point_arc_m) < float(ego_arc_m) - 1.0e-3:
            continue
        forward_distance_m = float(point_arc_m) - float(ego_arc_m)
        if float(forward_distance_m) > float(max_search_distance_m):
            break

        route_waypoint = world_map.get_waypoint(
            carla.Location(
                x=float(route_point[0]),
                y=float(route_point[1]),
                z=float(ego_z_m),
            ),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if route_waypoint is None:
            continue

        if bool(getattr(route_waypoint, "is_junction", False)):
            if previous_waypoint is None or previous_arc_m is None:
                return None
            stop_distance_m = float(previous_arc_m) - float(ego_arc_m)
            if stop_distance_m <= 1.0e-3:
                return None
            stop_location = getattr(getattr(previous_waypoint, "transform", None), "location", None)
            stop_rotation = getattr(getattr(previous_waypoint, "transform", None), "rotation", None)
            if stop_location is None:
                return None
            return {
                "x_m": float(getattr(stop_location, "x", 0.0)),
                "y_m": float(getattr(stop_location, "y", 0.0)),
                "heading_rad": float(
                    math.radians(float(getattr(stop_rotation, "yaw", 0.0)))
                ) if stop_rotation is not None else 0.0,
                "lane_id": int(canonical_lane_id_for_waypoint(previous_waypoint)),
                "road_id": int(getattr(previous_waypoint, "road_id", 0)),
                "section_id": int(getattr(previous_waypoint, "section_id", 0)),
                "distance_m": float(stop_distance_m),
            }

        previous_waypoint = route_waypoint
        previous_arc_m = float(point_arc_m)

    return None


def should_stop_for_signal(
    *,
    signal_state: object,
    stop_target: Mapping[str, object] | None,
    ego_velocity_mps: float,
    ego_max_deceleration_mps2: float,
    ego_in_junction: bool,
    stop_buffer_m: float = 2.0,
) -> bool:
    if bool(ego_in_junction):
        return False

    normalized_signal_state = normalize_signal_state(signal_state)
    if normalized_signal_state == "green":
        return False
    if normalized_signal_state not in {"yellow", "red"}:
        return False
    if not isinstance(stop_target, Mapping):
        return False

    try:
        stop_distance_m = max(0.0, float(stop_target.get("distance_m", 0.0)))
    except Exception:
        return False
    if stop_distance_m <= 1.0e-3:
        return False

    ego_speed_mps = max(0.0, float(ego_velocity_mps))
    max_deceleration_mps2 = max(1.0e-6, float(ego_max_deceleration_mps2))
    required_stop_distance_m = (
        float(ego_speed_mps) * float(ego_speed_mps)
    ) / (2.0 * float(max_deceleration_mps2))
    required_stop_distance_m += max(0.0, float(stop_buffer_m))

    return float(stop_distance_m) >= float(required_stop_distance_m)

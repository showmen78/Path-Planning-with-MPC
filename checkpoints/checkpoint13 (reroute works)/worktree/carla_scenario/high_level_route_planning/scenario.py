"""
Scenario-local logic for the high_level_route_planning experiment.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from behavior_planner.reroute import (
    CP_MESSAGE_PATH,
    ensure_cp_message_file_exists,
    load_cp_messages,
    remove_cp_messages_by_id,
    write_cp_messages,
)
def _best_partial_match(candidates: List[Tuple[int, Any]]) -> Any | None:
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _find_environment_marker_by_name(world, carla, marker_name: str):
    marker_name_lower = str(marker_name).strip().lower()
    partial_candidates: List[Tuple[int, Any]] = []
    for env_obj in world.get_environment_objects(carla.CityObjectLabel.Any):
        env_name = str(getattr(env_obj, "name", "")).strip().lower()
        if env_name == marker_name_lower:
            return env_obj
        if marker_name_lower and marker_name_lower in env_name:
            partial_candidates.append((len(env_name), env_obj))
    return _best_partial_match(partial_candidates)


def _find_actor_by_name(world, object_name: str):
    object_name_lower = str(object_name).strip().lower()
    if not object_name_lower:
        return None

    partial_candidates: List[Tuple[int, Any]] = []
    for actor in list(world.get_actors() if hasattr(world, "get_actors") else []):
        for candidate_name in _object_name_candidates(actor):
            normalized_candidate = str(candidate_name).strip().lower()
            if normalized_candidate == object_name_lower:
                return actor
            if object_name_lower in normalized_candidate:
                partial_candidates.append((len(normalized_candidate), actor))
                break
    return _best_partial_match(partial_candidates)


def _object_name_candidates(obj: Any) -> List[str]:
    candidate_values: List[str] = []
    for raw_value in (
        getattr(obj, "name", ""),
        getattr(obj, "type_id", ""),
        getattr(obj, "id", ""),
    ):
        normalized_value = str(raw_value).strip()
        if normalized_value:
            candidate_values.append(normalized_value)
    raw_attributes = getattr(obj, "attributes", None)
    if isinstance(raw_attributes, Mapping):
        for key in ("role_name", "object_name", "name"):
            normalized_value = str(raw_attributes.get(key, "")).strip()
            if normalized_value:
                candidate_values.append(normalized_value)
    return candidate_values


def _find_world_object_by_name(world, carla, object_name: str):
    object_name_lower = str(object_name).strip().lower()
    if not object_name_lower:
        return None

    actor_match = _find_actor_by_name(world, object_name_lower)
    if actor_match is not None:
        return actor_match

    environment_match = _find_environment_marker_by_name(world, carla, object_name_lower)
    if environment_match is not None:
        return environment_match

    return None


def _normalized_name_candidates(raw_names: Sequence[object]) -> List[str]:
    normalized_names: List[str] = []
    for raw_name in list(raw_names or []):
        normalized_name = str(raw_name).strip()
        if not normalized_name:
            continue
        if normalized_name.lower() in {name.lower() for name in normalized_names}:
            continue
        normalized_names.append(normalized_name)
    return normalized_names


def _workzone_name_candidates(
    *,
    scenario_cfg: Mapping[str, object] | None = None,
    runtime_state: Mapping[str, object] | None = None,
) -> List[str]:
    if isinstance(runtime_state, Mapping):
        stored_name = str(runtime_state.get("workzone_name", "")).strip()
        stored_names = _normalized_name_candidates([stored_name])
        if len(stored_names) > 0:
            return stored_names

    scenario_cfg = dict(scenario_cfg or {})
    runtime_cfg = dict(scenario_cfg.get("runtime", {}))
    return _normalized_name_candidates([runtime_cfg.get("workzone_name", "workzone")])


def _resolve_workzone_object(
    *,
    world,
    world_map,
    carla,
    candidate_names: Sequence[object],
) -> Tuple[Any | None, str | None, Dict[str, object] | None]:
    for candidate_name in _normalized_name_candidates(candidate_names):
        world_object = _find_environment_marker_by_name(world, carla, candidate_name)
        if world_object is None:
            continue
        return (
            world_object,
            str(candidate_name),
            _nearest_driving_waypoint_info(
                world_map=world_map,
                carla=carla,
                world_object=world_object,
            ),
        )
    return None, None, None


def _object_location_xy(world_object: Any) -> List[float] | None:
    transform = getattr(world_object, "transform", None)
    if transform is None and hasattr(world_object, "get_transform"):
        try:
            transform = world_object.get_transform()
        except RuntimeError:
            transform = None
    if transform is None:
        return None
    location = getattr(transform, "location", None)
    if location is None:
        return None
    return [float(location.x), float(location.y)]


def _object_transform(world_object: Any):
    transform = getattr(world_object, "transform", None)
    if transform is None and hasattr(world_object, "get_transform"):
        try:
            transform = world_object.get_transform()
        except RuntimeError:
            transform = None
    return transform


def _nearest_driving_waypoint_info(
    *,
    world_map,
    carla,
    world_object: Any,
) -> Dict[str, object] | None:
    object_transform = getattr(world_object, "transform", None)
    if object_transform is None:
        object_transform = _object_transform(world_object)
    object_location = getattr(object_transform, "location", None)
    if object_location is None or world_map is None or not hasattr(world_map, "get_waypoint"):
        return None
    try:
        waypoint = world_map.get_waypoint(
            object_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
    except Exception:
        waypoint = None
    waypoint_location = getattr(getattr(waypoint, "transform", None), "location", None)
    if waypoint_location is None:
        return None
    return {
        "position_xy": [float(waypoint_location.x), float(waypoint_location.y)],
        "road_id": int(getattr(waypoint, "road_id", 0)),
        "section_id": int(getattr(waypoint, "section_id", 0)),
        "lane_id": int(getattr(waypoint, "lane_id", 0)),
    }

def _append_lane_closure_message(
    *,
    message_path: str,
    message_id: str,
    position_xy: Sequence[float],
    road_id: object = None,
    section_id: object = None,
    lane_id: object = None,
) -> None:
    ensure_cp_message_file_exists(message_path=message_path)
    current_messages = load_cp_messages(message_path=message_path)
    normalized_message_id = str(message_id).strip()
    retained_messages = [
        dict(message)
        for message in current_messages
        if str(message.get("id", "")).strip() != normalized_message_id
    ]
    retained_messages.append(
        {
            "id": normalized_message_id,
            "type": "lane_closure",
            "position": [float(position_xy[0]), float(position_xy[1])],
            **({"road_id": road_id} if road_id is not None else {}),
            **({"section_id": int(section_id)} if section_id is not None else {}),
            **({"lane_id": int(lane_id)} if lane_id is not None else {}),
        }
    )
    write_cp_messages(retained_messages, message_path=message_path)


def _resolved_wall_time_s(wall_time_s: float | None = None) -> float:
    if wall_time_s is not None:
        return float(wall_time_s)
    return float(time.perf_counter())


def _clone_transform_with_pose(location_transform, rotation_transform, carla, z_m: float):
    return carla.Transform(
        carla.Location(
            x=float(location_transform.location.x),
            y=float(location_transform.location.y),
            z=float(z_m),
        ),
        carla.Rotation(
            pitch=float(rotation_transform.rotation.pitch),
            yaw=float(rotation_transform.rotation.yaw),
            roll=float(rotation_transform.rotation.roll),
        ),
    )


def _spawn_attempt_transforms(base_transform, waypoint_transform, carla, base_z_offset_m: float) -> List[Any]:
    attempts: List[Any] = []
    ground_z_m = float(base_transform.location.z)
    rotation_source_transform = base_transform
    if waypoint_transform is not None:
        ground_z_m = float(waypoint_transform.location.z)
        rotation_source_transform = waypoint_transform

    if waypoint_transform is not None:
        for extra_z_m in (0.0, 0.10, 0.20):
            attempts.append(
                _clone_transform_with_pose(
                    location_transform=waypoint_transform,
                    rotation_transform=rotation_source_transform,
                    carla=carla,
                    z_m=float(waypoint_transform.location.z) + float(base_z_offset_m) + float(extra_z_m),
                )
            )

    for extra_z_m in (0.0, 0.10, 0.20):
        attempts.append(
            _clone_transform_with_pose(
                location_transform=base_transform,
                rotation_transform=rotation_source_transform,
                carla=carla,
                z_m=float(ground_z_m) + float(base_z_offset_m) + float(extra_z_m),
            )
        )
    return attempts


def _configure_static_vehicle(vehicle, carla) -> None:
    try:
        vehicle.set_simulate_physics(True)
    except RuntimeError:
        pass
    try:
        vehicle.set_autopilot(False)
    except RuntimeError:
        pass
    try:
        vehicle.set_target_velocity(carla.Vector3D(x=0.0, y=0.0, z=0.0))
    except RuntimeError:
        pass
    try:
        vehicle.set_target_angular_velocity(carla.Vector3D(x=0.0, y=0.0, z=0.0))
    except RuntimeError:
        pass
    try:
        vehicle.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                brake=1.0,
                steer=0.0,
                hand_brake=True,
                reverse=False,
                manual_gear_shift=False,
            )
        )
    except RuntimeError:
        pass


def spawn_obstacles(
    *,
    world,
    world_map,
    carla,
    blueprint_library,
    scenario_cfg: Mapping[str, object],
    route_summary=None,
    route_points: Sequence[Sequence[float]] | None = None,
    **_,
) -> List[Any]:
    del route_summary, route_points

    obstacle_cfg = dict(scenario_cfg.get("obstacles", {}))
    marker_names = [
        str(marker_name).strip()
        for marker_name in list(obstacle_cfg.get("marker_names", ["obstacle1"]))
        if str(marker_name).strip()
    ]
    vehicle_blueprint_id = str(obstacle_cfg.get("vehicle_blueprint", "vehicle.tesla.model3")).strip()
    color_rgb = str(obstacle_cfg.get("color_rgb", "90,90,90")).strip()
    spawn_z_offset_m = float(obstacle_cfg.get("spawn_z_offset_m", 0.05))

    spawned_vehicles: List[Any] = []
    for marker_name in marker_names:
        marker = _find_environment_marker_by_name(world, carla, marker_name)
        if marker is None:
            print(f"[HIGH LEVEL ROUTE] Marker '{marker_name}' was not found.")
            continue

        try:
            blueprint = blueprint_library.find(vehicle_blueprint_id)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Obstacle vehicle blueprint '{vehicle_blueprint_id}' was not found."
            ) from exc
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", str(marker_name))
        if color_rgb and blueprint.has_attribute("color"):
            blueprint.set_attribute("color", color_rgb)

        nearest_waypoint = world_map.get_waypoint(
            marker.transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        waypoint_transform = None if nearest_waypoint is None else nearest_waypoint.transform
        spawn_vehicle = None
        for attempt_transform in _spawn_attempt_transforms(
            base_transform=marker.transform,
            waypoint_transform=waypoint_transform,
            carla=carla,
            base_z_offset_m=spawn_z_offset_m,
        ):
            spawn_vehicle = world.try_spawn_actor(blueprint, attempt_transform)
            if spawn_vehicle is not None:
                break

        if spawn_vehicle is None:
            print(f"[HIGH LEVEL ROUTE] Failed to spawn obstacle '{marker_name}'.")
            continue

        _configure_static_vehicle(spawn_vehicle, carla)
        spawned_vehicles.append(spawn_vehicle)
        transform = spawn_vehicle.get_transform()
        print(
            f"[HIGH LEVEL ROUTE] Spawned obstacle '{marker_name}' at "
            f"({float(transform.location.x):.3f}, {float(transform.location.y):.3f}, {float(transform.location.z):.3f})."
        )
    return spawned_vehicles


def initialize_runtime(
    *,
    scenario_cfg: Mapping[str, object],
    world,
    world_map=None,
    carla,
    wall_time_s: float | None = None,
    **_,
) -> Dict[str, object]:
    runtime_cfg = dict(scenario_cfg.get("runtime", {}))
    workzone_name_candidates = _workzone_name_candidates(scenario_cfg=scenario_cfg)
    preferred_workzone_name = (
        workzone_name_candidates[0] if len(workzone_name_candidates) > 0 else "workzone"
    )
    cp_message_path = str(runtime_cfg.get("cp_message_path", CP_MESSAGE_PATH))
    cp_message_id = str(
        runtime_cfg.get(
            "cp_message_id",
            f"high_level_route_planning_{str(preferred_workzone_name).strip().lower()}",
        )
    )
    if cp_message_id.strip():
        remove_cp_messages_by_id(
            [cp_message_id],
            message_path=cp_message_path,
        )
    workzone_object, resolved_workzone_name, workzone_waypoint_info = _resolve_workzone_object(
        world=world,
        world_map=world_map,
        carla=carla,
        candidate_names=workzone_name_candidates,
    )
    workzone_position_xy = None if workzone_waypoint_info is None else list(workzone_waypoint_info.get("position_xy", []))
    if workzone_position_xy is None:
        print(
            "[HIGH LEVEL ROUTE] Warning: workzone object was not found. "
            f"candidates={workzone_name_candidates}"
        )
    else:
        print(
            "[HIGH LEVEL ROUTE] Found workzone object "
            f"'{resolved_workzone_name}' at ({float(workzone_position_xy[0]):.3f}, {float(workzone_position_xy[1]):.3f}) "
            f"road_id={workzone_waypoint_info.get('road_id')} "
            f"section_id={workzone_waypoint_info.get('section_id')} "
            f"lane_id={workzone_waypoint_info.get('lane_id')}."
        )
    runtime_start_wall_time_s = _resolved_wall_time_s(wall_time_s)
    cooperative_message_delay_s = float(
        runtime_cfg.get("cooperative_message_delay_s", 5.0)
    )
    return {
        "workzone_name": str(preferred_workzone_name),
        "workzone_name_candidates": list(workzone_name_candidates),
        "workzone_object_name": None if resolved_workzone_name is None else str(resolved_workzone_name),
        "workzone_position_xy": workzone_position_xy,
        "workzone_road_id": None if workzone_waypoint_info is None else workzone_waypoint_info.get("road_id", None),
        "workzone_section_id": None if workzone_waypoint_info is None else workzone_waypoint_info.get("section_id", None),
        "workzone_lane_id": None if workzone_waypoint_info is None else workzone_waypoint_info.get("lane_id", None),
        "cooperative_message_delay_s": float(cooperative_message_delay_s),
        "runtime_start_wall_time_s": float(runtime_start_wall_time_s),
        "cooperative_message_publish_wall_time_s": float(runtime_start_wall_time_s + cooperative_message_delay_s),
        "cp_message_path": str(cp_message_path),
        "cp_message_id": str(cp_message_id),
        "cp_message_inserted": False,
        "workzone_missing_warning_emitted": workzone_position_xy is None,
    }


def maybe_replan_global_route(
    *,
    runtime_state,
    world,
    world_map=None,
    carla,
    sim_time_s: float,
    wall_time_s: float | None = None,
    **_,
) -> Tuple[object | None, List[List[float]] | None, Dict[str, object]]:
    next_runtime_state = dict(runtime_state or {})
    if bool(next_runtime_state.get("cp_message_inserted", False)):
        return None, None, next_runtime_state

    publish_wall_time_s = float(
        next_runtime_state.get(
            "cooperative_message_publish_wall_time_s",
            _resolved_wall_time_s(wall_time_s)
            + float(next_runtime_state.get("cooperative_message_delay_s", 5.0)),
        )
    )
    if float(_resolved_wall_time_s(wall_time_s)) < float(publish_wall_time_s):
        return None, None, next_runtime_state

    workzone_position_xy = next_runtime_state.get("workzone_position_xy", None)
    if not isinstance(workzone_position_xy, Sequence) or len(workzone_position_xy) < 2:
        _workzone_object, resolved_workzone_name, workzone_waypoint_info = _resolve_workzone_object(
            world=world,
            world_map=world_map,
            carla=carla,
            candidate_names=_workzone_name_candidates(runtime_state=next_runtime_state),
        )
        workzone_position_xy = None if workzone_waypoint_info is None else list(workzone_waypoint_info.get("position_xy", []))
        if isinstance(workzone_position_xy, Sequence) and len(workzone_position_xy) >= 2:
            next_runtime_state["workzone_position_xy"] = [
                float(workzone_position_xy[0]),
                float(workzone_position_xy[1]),
            ]
            next_runtime_state["workzone_road_id"] = (
                None if workzone_waypoint_info is None else workzone_waypoint_info.get("road_id", None)
            )
            next_runtime_state["workzone_section_id"] = (
                None if workzone_waypoint_info is None else workzone_waypoint_info.get("section_id", None)
            )
            next_runtime_state["workzone_lane_id"] = (
                None if workzone_waypoint_info is None else workzone_waypoint_info.get("lane_id", None)
            )
            if str(resolved_workzone_name or "").strip():
                next_runtime_state["workzone_object_name"] = str(resolved_workzone_name)
            print(
                "[HIGH LEVEL ROUTE] Detected workzone object during runtime at "
                f"({float(workzone_position_xy[0]):.3f}, {float(workzone_position_xy[1]):.3f}) "
                f"road_id={next_runtime_state.get('workzone_road_id')} "
                f"section_id={next_runtime_state.get('workzone_section_id')} "
                f"lane_id={next_runtime_state.get('workzone_lane_id')}."
            )

    if not isinstance(workzone_position_xy, Sequence) or len(workzone_position_xy) < 2:
        if not bool(next_runtime_state.get("workzone_missing_warning_emitted", False)):
            print(
                "[HIGH LEVEL ROUTE] Warning: workzone object could not be located before cooperative message insertion. "
                f"candidates={_workzone_name_candidates(runtime_state=next_runtime_state)}"
            )
            next_runtime_state["workzone_missing_warning_emitted"] = True
        return None, None, next_runtime_state

    _append_lane_closure_message(
        message_path=str(next_runtime_state.get("cp_message_path", CP_MESSAGE_PATH)),
        message_id=str(next_runtime_state.get("cp_message_id", "high_level_route_planning_workzone")),
        position_xy=workzone_position_xy,
        road_id=next_runtime_state.get("workzone_road_id", None),
        section_id=next_runtime_state.get("workzone_section_id", None),
        lane_id=next_runtime_state.get("workzone_lane_id", None),
    )
    next_runtime_state["cp_message_inserted"] = True
    print(
        "[HIGH LEVEL ROUTE] Inserted workzone cooperative message into cp_message.json "
        f"at ({float(workzone_position_xy[0]):.3f}, {float(workzone_position_xy[1]):.3f})."
    )
    return None, None, next_runtime_state


def filter_dynamic_obstacle_snapshots(
    *,
    runtime_state,
    world,
    world_map,
    carla,
    object_snapshots: Sequence[Mapping[str, object]],
    sim_time_s: float,
    wall_time_s: float | None = None,
    **_,
) -> Tuple[List[dict], Dict[str, object]]:
    next_runtime_state = dict(runtime_state or {})
    filtered_snapshots = [dict(snapshot) for snapshot in list(object_snapshots or [])]
    return filtered_snapshots, next_runtime_state

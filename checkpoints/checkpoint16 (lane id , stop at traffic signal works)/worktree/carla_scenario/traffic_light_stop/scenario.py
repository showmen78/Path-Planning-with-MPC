"""
Scenario-local traffic-light control for the stop experiment.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def _resolved_wall_time_s(wall_time_s: float | None = None) -> float:
    if wall_time_s is not None:
        return float(wall_time_s)
    return float(time.perf_counter())


def _object_name_candidates(obj: Any) -> List[str]:
    candidate_values: List[str] = []
    for raw_value in (
        getattr(obj, "name", ""),
        getattr(obj, "type_id", ""),
    ):
        normalized_value = str(raw_value).strip()
        if normalized_value:
            candidate_values.append(normalized_value)
    raw_attributes = getattr(obj, "attributes", None)
    if isinstance(raw_attributes, Mapping):
        for key in ("name", "role_name", "object_name"):
            normalized_value = str(raw_attributes.get(key, "")).strip()
            if normalized_value:
                candidate_values.append(normalized_value)
    return candidate_values


def _object_transform(obj: Any):
    direct_transform = getattr(obj, "transform", None)
    if direct_transform is not None:
        return direct_transform
    get_transform_fn = getattr(obj, "get_transform", None)
    if callable(get_transform_fn):
        try:
            return get_transform_fn()
        except Exception:
            return None
    return None


def _best_partial_match(candidates: List[Tuple[int, Any]]) -> Any | None:
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _find_environment_object_by_name(world, carla, object_name: str):
    object_name_lower = str(object_name).strip().lower()
    if not object_name_lower:
        return None
    partial_candidates: List[Tuple[int, Any]] = []
    for env_obj in list(world.get_environment_objects(carla.CityObjectLabel.Any)):
        env_name = str(getattr(env_obj, "name", "")).strip().lower()
        if env_name == object_name_lower:
            return env_obj
        if object_name_lower in env_name:
            partial_candidates.append((len(env_name), env_obj))
    return _best_partial_match(partial_candidates)


def _find_actor_by_name(world, object_name: str):
    object_name_lower = str(object_name).strip().lower()
    if not object_name_lower:
        return None
    partial_candidates: List[Tuple[int, Any]] = []
    for actor in list(world.get_actors() if hasattr(world, "get_actors") else []):
        attr_name = str(getattr(actor, "attributes", {}).get("name", "")).strip().lower()
        role_name = str(getattr(actor, "attributes", {}).get("role_name", "")).strip().lower()
        type_id = str(getattr(actor, "type_id", "")).strip().lower()
        if (
            attr_name == object_name_lower
            or role_name == object_name_lower
            or type_id.endswith(object_name_lower)
        ):
            return actor
        if object_name_lower in attr_name:
            partial_candidates.append((len(attr_name), actor))
        if object_name_lower in role_name:
            partial_candidates.append((len(role_name), actor))
        if object_name_lower in type_id:
            partial_candidates.append((len(type_id), actor))
    return _best_partial_match(partial_candidates)


def _is_traffic_light_actor(actor: Any) -> bool:
    if actor is None:
        return False
    if callable(getattr(actor, "set_state", None)):
        return True
    return "traffic_light" in str(getattr(actor, "type_id", "")).strip().lower()


def _iter_traffic_light_actors(world) -> Iterable[Any]:
    for actor in list(world.get_actors() if hasattr(world, "get_actors") else []):
        if _is_traffic_light_actor(actor):
            yield actor


def _distance_between_transforms_m(transform_a: Any, transform_b: Any) -> float:
    location_a = getattr(transform_a, "location", None)
    location_b = getattr(transform_b, "location", None)
    if location_a is None or location_b is None:
        return float("inf")
    return float(
        math.hypot(
            float(location_a.x) - float(location_b.x),
            float(location_a.y) - float(location_b.y),
        )
    )


def _resolve_exact_trigger_transform(world, carla, marker_name: str):
    marker = _find_environment_object_by_name(world, carla, marker_name)
    if marker is not None:
        return _object_transform(marker)
    marker_actor = _find_actor_by_name(world, marker_name)
    if marker_actor is not None:
        return _object_transform(marker_actor)
    return None


def _resolve_named_traffic_light_actor(
    *,
    world,
    carla,
    traffic_light_name: str,
    max_actor_distance_m: float = 12.0,
) -> Tuple[Any | None, str]:
    exact_actor = _find_actor_by_name(world, traffic_light_name)
    if _is_traffic_light_actor(exact_actor):
        return exact_actor, "actor"

    exact_env_object = _find_environment_object_by_name(
        world,
        carla,
        traffic_light_name,
    )
    if exact_env_object is None:
        return None, "missing"

    env_transform = _object_transform(exact_env_object)
    if env_transform is None:
        return None, "missing"

    best_actor = None
    best_distance_m = float("inf")
    for candidate_actor in _iter_traffic_light_actors(world):
        candidate_transform = _object_transform(candidate_actor)
        if candidate_transform is None:
            continue
        distance_m = _distance_between_transforms_m(env_transform, candidate_transform)
        if distance_m < best_distance_m:
            best_distance_m = float(distance_m)
            best_actor = candidate_actor
    if best_actor is None or float(best_distance_m) > float(max_actor_distance_m):
        return None, "unresolved"
    return best_actor, "environment_object"


def _traffic_light_state_enum(carla, state_name: str):
    traffic_light_state_cls = getattr(carla, "TrafficLightState", None)
    if traffic_light_state_cls is None:
        return None
    return getattr(traffic_light_state_cls, str(state_name).capitalize(), None)


def _set_frozen_traffic_light_state(traffic_light_actor: Any, carla, state_name: str) -> bool:
    if traffic_light_actor is None:
        return False
    state_enum = _traffic_light_state_enum(carla, state_name)
    if state_enum is None:
        return False
    freeze_fn = getattr(traffic_light_actor, "freeze", None)
    if callable(freeze_fn):
        try:
            freeze_fn(False)
        except Exception:
            pass
    set_state_fn = getattr(traffic_light_actor, "set_state", None)
    if not callable(set_state_fn):
        return False
    try:
        set_state_fn(state_enum)
    except Exception:
        return False
    if callable(freeze_fn):
        try:
            freeze_fn(True)
        except Exception:
            pass
    return True


def _warning(message: str) -> None:
    print(f"[TRAFFIC LIGHT STOP] Warning: {message}")


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


def _spawn_attempt_transforms(
    base_transform,
    rotation_transform,
    carla,
    base_z_offset_m: float,
) -> List[Any]:
    attempts: List[Any] = []
    for extra_z_m in (0.0, 0.10, 0.20):
        attempts.append(
            _clone_transform_with_pose(
                location_transform=base_transform,
                rotation_transform=rotation_transform,
                carla=carla,
                z_m=float(base_transform.location.z) + float(base_z_offset_m) + float(extra_z_m),
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
    world_map=None,
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
        for marker_name in list(obstacle_cfg.get("marker_names", ["traffic_obstacle1"]))
        if str(marker_name).strip()
    ]
    vehicle_blueprint_id = str(obstacle_cfg.get("vehicle_blueprint", "vehicle.tesla.model3")).strip()
    color_rgb = str(obstacle_cfg.get("color_rgb", "90,90,90")).strip()
    spawn_z_offset_m = float(obstacle_cfg.get("spawn_z_offset_m", 0.05))

    spawned_vehicles: List[Any] = []
    for marker_name in marker_names:
        marker_transform = _resolve_exact_trigger_transform(
            world=world,
            carla=carla,
            marker_name=str(marker_name),
        )
        if marker_transform is None:
            _warning(f"obstacle marker '{marker_name}' was not found.")
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

        rotation_transform = marker_transform
        if world_map is not None and hasattr(world_map, "get_waypoint"):
            try:
                spawn_waypoint = world_map.get_waypoint(
                    marker_transform.location,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
            except Exception:
                spawn_waypoint = None
            waypoint_transform = getattr(spawn_waypoint, "transform", None)
            if waypoint_transform is not None:
                rotation_transform = waypoint_transform

        spawn_vehicle = None
        for attempt_transform in _spawn_attempt_transforms(
            base_transform=marker_transform,
            rotation_transform=rotation_transform,
            carla=carla,
            base_z_offset_m=spawn_z_offset_m,
        ):
            spawn_vehicle = world.try_spawn_actor(blueprint, attempt_transform)
            if spawn_vehicle is not None:
                break

        if spawn_vehicle is None:
            print(f"[TRAFFIC LIGHT STOP] Failed to spawn obstacle '{marker_name}'.")
            continue

        _configure_static_vehicle(spawn_vehicle, carla)
        spawned_vehicles.append(spawn_vehicle)
        transform = spawn_vehicle.get_transform()
        print(
            f"[TRAFFIC LIGHT STOP] Spawned obstacle '{marker_name}' at "
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
    del world_map
    runtime_cfg = dict(scenario_cfg.get("runtime", {}))
    traffic_light_name = str(
        runtime_cfg.get(
            "traffic_light_name",
            "BP_TrafficLightNew_T10_master_largeBIG_rsc11",
        )
    ).strip()
    trigger_yellow_name = str(runtime_cfg.get("trigger_yellow_marker", "trigger_yellow")).strip()
    trigger_red_name = str(runtime_cfg.get("trigger_red_marker", "trigger_red")).strip()
    trigger_distance_m = max(0.1, float(runtime_cfg.get("trigger_distance_m", 5.0)))
    red_hold_duration_s = max(0.0, float(runtime_cfg.get("red_hold_duration_s", 5.0)))
    traffic_light_actor, traffic_light_resolution = _resolve_named_traffic_light_actor(
        world=world,
        carla=carla,
        traffic_light_name=str(traffic_light_name),
    )
    trigger_yellow_transform = _resolve_exact_trigger_transform(
        world=world,
        carla=carla,
        marker_name=str(trigger_yellow_name),
    )
    trigger_red_transform = _resolve_exact_trigger_transform(
        world=world,
        carla=carla,
        marker_name=str(trigger_red_name),
    )

    if traffic_light_actor is None:
        _warning(
            f"traffic light '{traffic_light_name}' was not found for scenario control."
        )
    else:
        _set_frozen_traffic_light_state(traffic_light_actor, carla, "green")
        print(
            "[TRAFFIC LIGHT STOP] Controlled signal "
            f"'{traffic_light_name}' initialized to GREEN (freeze)."
        )

    if trigger_yellow_transform is None:
        _warning(f"trigger marker '{trigger_yellow_name}' was not found.")
    if trigger_red_transform is None:
        _warning(f"trigger marker '{trigger_red_name}' was not found.")

    return {
        "traffic_light_name": str(traffic_light_name),
        "traffic_light_actor": traffic_light_actor,
        "traffic_light_resolution": str(traffic_light_resolution),
        "trigger_yellow_name": str(trigger_yellow_name),
        "trigger_red_name": str(trigger_red_name),
        "trigger_yellow_transform": trigger_yellow_transform,
        "trigger_red_transform": trigger_red_transform,
        "trigger_distance_m": float(trigger_distance_m),
        "red_hold_duration_s": float(red_hold_duration_s),
        "phase": "green_initial",
        "red_release_wall_time_s": None,
        "runtime_start_wall_time_s": float(_resolved_wall_time_s(wall_time_s)),
        "missing_ego_warning_emitted": False,
        "ego_actor": None,
        "lookup_retry_period_s": 1.0,
        "next_lookup_retry_wall_time_s": float(_resolved_wall_time_s(wall_time_s)),
    }


def filter_dynamic_obstacle_snapshots(
    *,
    runtime_state,
    world,
    world_map=None,
    carla=None,
    object_snapshots: Sequence[Mapping[str, object]],
    sim_time_s: float | None = None,
    wall_time_s: float | None = None,
    **_,
) -> Tuple[List[dict], Dict[str, object]]:
    del world_map
    del sim_time_s
    filtered_snapshots = [dict(snapshot) for snapshot in list(object_snapshots or [])]
    next_runtime_state = dict(runtime_state or {})

    current_wall_time_s = float(_resolved_wall_time_s(wall_time_s))
    traffic_light_actor = next_runtime_state.get("traffic_light_actor", None)
    retry_deadline_s = float(
        next_runtime_state.get(
            "next_lookup_retry_wall_time_s",
            float(current_wall_time_s),
        )
    )
    should_retry_lookup = (
        (
            traffic_light_actor is None
            or next_runtime_state.get("trigger_yellow_transform", None) is None
            or next_runtime_state.get("trigger_red_transform", None) is None
            or next_runtime_state.get("ego_actor", None) is None
        )
        and float(current_wall_time_s) >= float(retry_deadline_s)
    )
    if should_retry_lookup:
        if traffic_light_actor is None:
            traffic_light_actor, traffic_light_resolution = _resolve_named_traffic_light_actor(
                world=world,
                carla=carla,
                traffic_light_name=str(next_runtime_state.get("traffic_light_name", "")),
            )
            if traffic_light_actor is not None:
                next_runtime_state["traffic_light_actor"] = traffic_light_actor
                next_runtime_state["traffic_light_resolution"] = str(traffic_light_resolution)
                _set_frozen_traffic_light_state(traffic_light_actor, carla, "green")
                print(
                    "[TRAFFIC LIGHT STOP] Controlled signal "
                    f"'{next_runtime_state.get('traffic_light_name', '')}' initialized to GREEN (freeze)."
                )

        if next_runtime_state.get("trigger_yellow_transform", None) is None:
            next_runtime_state["trigger_yellow_transform"] = _resolve_exact_trigger_transform(
                world=world,
                carla=carla,
                marker_name=str(next_runtime_state.get("trigger_yellow_name", "trigger_yellow")),
            )
        if next_runtime_state.get("trigger_red_transform", None) is None:
            next_runtime_state["trigger_red_transform"] = _resolve_exact_trigger_transform(
                world=world,
                carla=carla,
                marker_name=str(next_runtime_state.get("trigger_red_name", "trigger_red")),
            )
        if next_runtime_state.get("ego_actor", None) is None:
            next_runtime_state["ego_actor"] = _find_actor_by_name(world, "ego")
        next_runtime_state["next_lookup_retry_wall_time_s"] = (
            float(current_wall_time_s)
            + max(0.1, float(next_runtime_state.get("lookup_retry_period_s", 1.0)))
        )

    traffic_light_actor = next_runtime_state.get("traffic_light_actor", None)
    ego_actor = next_runtime_state.get("ego_actor", None)
    if ego_actor is None:
        if not bool(next_runtime_state.get("missing_ego_warning_emitted", False)):
            _warning("ego actor 'ego' was not found during traffic-light scenario runtime.")
            next_runtime_state["missing_ego_warning_emitted"] = True
        return filtered_snapshots, next_runtime_state

    ego_transform = _object_transform(ego_actor)
    if ego_transform is None:
        return filtered_snapshots, next_runtime_state

    current_phase = str(next_runtime_state.get("phase", "green_initial"))
    trigger_distance_m = float(next_runtime_state.get("trigger_distance_m", 5.0))
    yellow_trigger_transform = next_runtime_state.get("trigger_yellow_transform", None)
    red_trigger_transform = next_runtime_state.get("trigger_red_transform", None)

    if (
        current_phase == "green_initial"
        and yellow_trigger_transform is not None
        and _distance_between_transforms_m(ego_transform, yellow_trigger_transform) <= float(trigger_distance_m)
    ):
        if _set_frozen_traffic_light_state(traffic_light_actor, carla, "yellow"):
            next_runtime_state["phase"] = "yellow"
            print("[TRAFFIC LIGHT STOP] Signal set to YELLOW (freeze).")
        return filtered_snapshots, next_runtime_state

    if (
        current_phase == "yellow"
        and red_trigger_transform is not None
        and _distance_between_transforms_m(ego_transform, red_trigger_transform) <= float(trigger_distance_m)
    ):
        if _set_frozen_traffic_light_state(traffic_light_actor, carla, "red"):
            next_runtime_state["phase"] = "red"
            next_runtime_state["red_release_wall_time_s"] = (
                float(current_wall_time_s)
                + float(next_runtime_state.get("red_hold_duration_s", 5.0))
            )
            print("[TRAFFIC LIGHT STOP] Signal set to RED (freeze).")
        return filtered_snapshots, next_runtime_state

    if current_phase == "red":
        release_wall_time_s = next_runtime_state.get("red_release_wall_time_s", None)
        if (
            release_wall_time_s is not None
            and float(current_wall_time_s) >= float(release_wall_time_s)
            and _set_frozen_traffic_light_state(traffic_light_actor, carla, "green")
        ):
            next_runtime_state["phase"] = "green_released"
            print("[TRAFFIC LIGHT STOP] Signal set to GREEN (freeze) after red hold.")

    return filtered_snapshots, next_runtime_state

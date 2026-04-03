"""
Scenario-local logic for the roadway_hazard use case.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple


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


def _configure_autopilot_vehicle(vehicle, traffic_manager, traffic_manager_port: int) -> None:
    try:
        vehicle.set_simulate_physics(True)
    except RuntimeError:
        pass

    try:
        vehicle.set_autopilot(True, int(traffic_manager_port))
    except TypeError:
        try:
            vehicle.set_autopilot(True)
        except RuntimeError:
            pass
    except RuntimeError:
        pass

    if traffic_manager is None:
        return

    try:
        traffic_manager.auto_lane_change(vehicle, False)
    except Exception:
        pass
    try:
        traffic_manager.distance_to_leading_vehicle(vehicle, 1.0)
    except Exception:
        pass


def _lowercase_name_set(values: Sequence[object] | None, default_values: Sequence[str]) -> List[str]:
    raw_values = list(values) if values is not None else list(default_values)
    normalized: List[str] = []
    for value in raw_values:
        name = str(value).strip()
        if name:
            normalized.append(name)
    return normalized


def spawn_obstacles(
    *,
    world,
    world_map,
    carla,
    blueprint_library,
    traffic_manager=None,
    traffic_manager_port: int = 8000,
    scenario_cfg: Mapping[str, object],
    route_summary=None,
    route_points: Sequence[Sequence[float]] | None = None,
) -> List[Any]:
    del route_summary, route_points

    obstacle_cfg = dict(scenario_cfg.get("obstacles", {}))
    marker_names = _lowercase_name_set(
        obstacle_cfg.get("marker_names", None),
        default_values=[f"obstacle{idx}" for idx in range(1, 7)],
    )
    autopilot_marker_names = {
        str(name).strip().lower()
        for name in _lowercase_name_set(
            obstacle_cfg.get("autopilot_marker_names", None),
            default_values=["obstacle6"],
        )
    }
    vehicle_blueprint_id = str(obstacle_cfg.get("vehicle_blueprint", "vehicle.tesla.model3")).strip()
    color_rgb = str(obstacle_cfg.get("color_rgb", "90,90,90")).strip()
    spawn_z_offset_m = float(obstacle_cfg.get("spawn_z_offset_m", 0.05))

    spawned_vehicles: List[Any] = []
    for marker_name in marker_names:
        marker = _find_environment_marker_by_name(world, carla, marker_name)
        if marker is None:
            print(f"[ROADWAY HAZARD] Marker '{marker_name}' was not found.")
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
            print(f"[ROADWAY HAZARD] Failed to spawn vehicle at marker '{marker_name}'.")
            continue

        marker_name_lower = str(marker_name).strip().lower()
        if marker_name_lower in autopilot_marker_names:
            _configure_autopilot_vehicle(
                spawn_vehicle,
                traffic_manager=traffic_manager,
                traffic_manager_port=int(traffic_manager_port),
            )
        else:
            _configure_static_vehicle(spawn_vehicle, carla)

        spawned_vehicles.append(spawn_vehicle)
        transform = spawn_vehicle.get_transform()
        print(
            f"[ROADWAY HAZARD] Spawned '{marker_name}' at "
            f"({float(transform.location.x):.3f}, {float(transform.location.y):.3f}, {float(transform.location.z):.3f}) "
            f"autopilot={'on' if marker_name_lower in autopilot_marker_names else 'off'}"
        )

    return spawned_vehicles


def initialize_runtime(
    *,
    scenario_cfg: Mapping[str, object],
    **_,
) -> Dict[str, object]:
    runtime_cfg = dict(scenario_cfg.get("runtime", {}))
    hidden_obstacle_id = str(runtime_cfg.get("hidden_obstacle_id", "obstacle4")).strip().lower()
    relay_obstacle_id = str(runtime_cfg.get("relay_obstacle_id", "obstacle6")).strip().lower()
    reveal_distance_m = float(runtime_cfg.get("reveal_distance_m", 20.0))
    return {
        "hidden_obstacle_id": hidden_obstacle_id,
        "relay_obstacle_id": relay_obstacle_id,
        "reveal_distance_m": reveal_distance_m,
        "hidden_obstacle_revealed": False,
    }


def _snapshot_by_id(
    object_snapshots: Sequence[Mapping[str, object]],
) -> Dict[str, Mapping[str, object]]:
    snapshots_by_id: Dict[str, Mapping[str, object]] = {}
    for snapshot in object_snapshots:
        obstacle_id = str(snapshot.get("vehicle_id", "")).strip().lower()
        if obstacle_id:
            snapshots_by_id[obstacle_id] = snapshot
    return snapshots_by_id


def _distance_between_snapshots(
    first_snapshot: Mapping[str, object] | None,
    second_snapshot: Mapping[str, object] | None,
) -> float | None:
    if first_snapshot is None or second_snapshot is None:
        return None
    return float(
        math.hypot(
            float(first_snapshot.get("x", 0.0)) - float(second_snapshot.get("x", 0.0)),
            float(first_snapshot.get("y", 0.0)) - float(second_snapshot.get("y", 0.0)),
        )
    )


def filter_dynamic_obstacle_snapshots(
    *,
    runtime_state,
    object_snapshots: Sequence[Mapping[str, object]],
    **_,
) -> Tuple[List[dict], Dict[str, object]]:
    next_runtime_state = dict(runtime_state or {})
    hidden_obstacle_id = str(next_runtime_state.get("hidden_obstacle_id", "obstacle4")).strip().lower()
    relay_obstacle_id = str(next_runtime_state.get("relay_obstacle_id", "obstacle6")).strip().lower()
    reveal_distance_m = float(next_runtime_state.get("reveal_distance_m", 20.0))
    hidden_obstacle_revealed = bool(next_runtime_state.get("hidden_obstacle_revealed", False))

    snapshots_by_id = _snapshot_by_id(object_snapshots)
    hidden_snapshot = snapshots_by_id.get(hidden_obstacle_id, None)
    relay_snapshot = snapshots_by_id.get(relay_obstacle_id, None)
    obstacle_distance_m = _distance_between_snapshots(hidden_snapshot, relay_snapshot)

    if (
        not hidden_obstacle_revealed
        and obstacle_distance_m is not None
        and float(obstacle_distance_m) < float(reveal_distance_m)
    ):
        hidden_obstacle_revealed = True
        print("[ROADWAY HAZARD] Cooperative message received for obstacle4; enabling obstacle field.")

    next_runtime_state["hidden_obstacle_revealed"] = bool(hidden_obstacle_revealed)

    filtered_snapshots: List[dict] = []
    for snapshot in object_snapshots:
        snapshot_id = str(snapshot.get("vehicle_id", "")).strip().lower()
        if snapshot_id == hidden_obstacle_id and not hidden_obstacle_revealed:
            continue
        filtered_snapshots.append(dict(snapshot))

    return filtered_snapshots, next_runtime_state

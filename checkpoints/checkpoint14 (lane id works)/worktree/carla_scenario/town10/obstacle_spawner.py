"""
Scenario-local obstacle spawner for Town10.
"""

from __future__ import annotations

import re
from typing import Any, List, Mapping, Sequence


def _obstacle_sort_key(env_obj, prefix: str) -> tuple[int, int, str]:
    name = str(getattr(env_obj, "name", ""))
    match = re.fullmatch(rf"{re.escape(prefix)}(\d+)", name.lower())
    if match is not None:
        return (0, int(match.group(1)), name.lower())
    return (1, 0, name.lower())


def _find_obstacle_markers(world, carla, prefix: str) -> List[Any]:
    prefix_lower = str(prefix).lower()
    markers = [
        env_obj
        for env_obj in world.get_environment_objects(carla.CityObjectLabel.Any)
        if str(getattr(env_obj, "name", "")).lower().startswith(prefix_lower)
    ]
    markers.sort(key=lambda env_obj: _obstacle_sort_key(env_obj, prefix_lower))
    return markers


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
) -> List[Any]:
    del route_summary, route_points

    obstacle_cfg = dict(scenario_cfg.get("obstacles", {}))
    cube_name_prefix = str(obstacle_cfg.get("cube_name_prefix", "obstacle")).strip() or "obstacle"
    vehicle_blueprint_id = str(obstacle_cfg.get("vehicle_blueprint", "vehicle.tesla.model3")).strip()
    role_name_prefix = str(obstacle_cfg.get("role_name_prefix", "static_obstacle")).strip() or "static_obstacle"
    color_rgb = str(obstacle_cfg.get("color_rgb", "90,90,90")).strip()
    spawn_z_offset_m = float(obstacle_cfg.get("spawn_z_offset_m", 0.05))

    markers = _find_obstacle_markers(world, carla, cube_name_prefix)
    if len(markers) == 0:
        print(
            f"[TOWN10 OBSTACLES] No EnvironmentObject markers with prefix '{cube_name_prefix}' were found."
        )
        return []

    print(
        f"[TOWN10 OBSTACLES] Found {len(markers)} obstacle marker(s) "
        f"with prefix '{cube_name_prefix}'."
    )
    spawned_vehicles: List[Any] = []
    for index, marker in enumerate(markers, start=1):
        marker_name = str(getattr(marker, "name", f"{cube_name_prefix}{index}"))
        marker_transform = marker.transform
        marker_location = marker_transform.location
        print(
            f"[TOWN10 OBSTACLES] Marker '{marker_name}' at "
            f"({float(marker_location.x):.3f}, {float(marker_location.y):.3f}, {float(marker_location.z):.3f})"
        )

        try:
            blueprint = blueprint_library.find(vehicle_blueprint_id)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Obstacle vehicle blueprint '{vehicle_blueprint_id}' was not found."
            ) from exc
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", f"{role_name_prefix}_{marker_name}")
        if color_rgb and blueprint.has_attribute("color"):
            blueprint.set_attribute("color", color_rgb)

        nearest_waypoint = world_map.get_waypoint(
            marker_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        waypoint_transform = None if nearest_waypoint is None else nearest_waypoint.transform
        spawn_vehicle = None
        for attempt_transform in _spawn_attempt_transforms(
            base_transform=marker_transform,
            waypoint_transform=waypoint_transform,
            carla=carla,
            base_z_offset_m=spawn_z_offset_m,
        ):
            spawn_vehicle = world.try_spawn_actor(blueprint, attempt_transform)
            if spawn_vehicle is not None:
                break

        if spawn_vehicle is None:
            print(
                f"[TOWN10 OBSTACLES] Failed to spawn vehicle at marker '{marker_name}'."
            )
            continue

        _configure_static_vehicle(spawn_vehicle, carla)
        spawned_vehicles.append(spawn_vehicle)
        transform = spawn_vehicle.get_transform()
        print(
            f"[TOWN10 OBSTACLES] Spawned static vehicle '{marker_name}' at "
            f"({float(transform.location.x):.3f}, {float(transform.location.y):.3f}, {float(transform.location.z):.3f}) "
            f"yaw={float(transform.rotation.yaw):.3f}"
        )

    return spawned_vehicles

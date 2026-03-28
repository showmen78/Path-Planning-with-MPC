"""
Scenario-local logic for the high_level_route_planning experiment.
"""

from __future__ import annotations

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
    **_,
) -> Dict[str, object]:
    runtime_cfg = dict(scenario_cfg.get("runtime", {}))
    blocked_obstacle_ids = [
        str(obstacle_id).strip().lower()
        for obstacle_id in list(runtime_cfg.get("blocked_obstacle_ids", ["obstacle1"]))
        if str(obstacle_id).strip()
    ]
    return {
        "obstacle_aware_replan_delay_s": float(
            runtime_cfg.get("obstacle_aware_replan_delay_s", 5.0)
        ),
        "blocked_obstacle_ids": blocked_obstacle_ids,
        "block_radius_m": float(runtime_cfg.get("block_radius_m", 8.0)),
        "route_replan_applied": False,
    }


def maybe_replan_global_route(
    *,
    runtime_state,
    global_planner,
    ego_transform,
    goal_location,
    object_snapshots: Sequence[Mapping[str, object]],
    sim_time_s: float,
    **_,
) -> Tuple[object | None, List[List[float]] | None, Dict[str, object]]:
    next_runtime_state = dict(runtime_state or {})
    if bool(next_runtime_state.get("route_replan_applied", False)):
        return None, None, next_runtime_state

    replan_delay_s = float(next_runtime_state.get("obstacle_aware_replan_delay_s", 5.0))
    if float(sim_time_s) < float(replan_delay_s):
        return None, None, next_runtime_state

    blocked_obstacle_ids = {
        str(obstacle_id).strip().lower()
        for obstacle_id in list(next_runtime_state.get("blocked_obstacle_ids", []))
        if str(obstacle_id).strip()
    }
    blocked_snapshots = [
        dict(snapshot)
        for snapshot in list(object_snapshots or [])
        if (
            len(blocked_obstacle_ids) == 0
            or str(snapshot.get("vehicle_id", "")).strip().lower() in blocked_obstacle_ids
        )
    ]
    next_runtime_state["route_replan_applied"] = True
    if len(blocked_snapshots) == 0:
        return None, None, next_runtime_state

    start_xy = [
        float(ego_transform.location.x),
        float(ego_transform.location.y),
    ]
    goal_xy = [
        float(goal_location.x),
        float(goal_location.y),
    ]
    blocked_points_xy = [
        [float(snapshot.get("x", 0.0)), float(snapshot.get("y", 0.0))]
        for snapshot in blocked_snapshots
    ]
    route_summary = global_planner.plan_route_astar_avoiding_points(
        start_xy=start_xy,
        goal_xy=goal_xy,
        blocked_points_xy=blocked_points_xy,
        block_radius_m=float(next_runtime_state.get("block_radius_m", 8.0)),
        replace_stored_route=True,
    )
    if not bool(getattr(route_summary, "route_found", False)):
        print("[HIGH LEVEL ROUTE] Obstacle-aware global reroute failed.")
        return None, None, next_runtime_state

    route_points = [
        [float(item[0]), float(item[1])]
        for item in list(getattr(route_summary, "route_waypoints", []) or [])
        if isinstance(item, Sequence) and len(item) >= 2
    ]
    print("[HIGH LEVEL ROUTE] Regenerated global route considering obstacles.")
    return route_summary, route_points, next_runtime_state

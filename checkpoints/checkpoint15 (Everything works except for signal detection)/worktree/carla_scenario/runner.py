"""
Shared CARLA scenario runner.
"""

from __future__ import annotations

import importlib
import inspect
import math
import os
import queue
import threading
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from MPC import (
    MPC,
    compute_lane_lookahead_distance,
)
from behavior_planner import (
    LaneSafetyScorer,
    RuleBasedBehaviorPlanner,
    build_reference_samples,
    find_relevant_signal_context,
    find_stop_target_from_ego,
    evaluate_intersection_obstacle_response,
    intersection_route_follow_maneuver,
    normalize_behavior_decision,
    normalize_macro_maneuver,
    compute_temp_destination_mode,
    compute_temp_destination,
    compute_ego_lane_offset,
)
from behavior_planner.reroute import (
    CP_MESSAGE_PATH,
    reroute_from_lane_closure_messages,
)
from utility import AStarGlobalPlanner, Tracker, build_lane_center_waypoints, load_yaml_file

try:
    import pygame
except ImportError:  # pragma: no cover
    pygame = None  # type: ignore[assignment]


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MPC_CONFIG_PATH = os.path.join(PROJECT_ROOT, "MPC", "mpc.yaml")
TRACKER_CONFIG_PATH = os.path.join(PROJECT_ROOT, "utility", "tracker.yaml")

# CARLA simulation tick [s]. Fixed at 20 Hz — do NOT change at runtime.
# MPC prediction step (plan_dt_s) is independent and configured in mpc.yaml.
CARLA_FIXED_DELTA_SECONDS: float = 0.05
WORLD_DEBUG_ROUTE_REFRESH_PERIOD_S: float = 0.10
WORLD_DEBUG_ROUTE_LIFE_TIME_S: float = 0.15
REQUIRED_CONSTRAINT_KEYS = (
    "min_velocity_mps",
    "max_velocity_mps",
    "min_acceleration_mps2",
    "max_acceleration_mps2",
    "max_jerk_mps3",
    "min_steer_rad",
    "max_steer_rad",
    "min_steer_rate_rps",
    "max_steer_rate_rps",
    "enforce_terminal_velocity_constraint",
    "terminal_velocity_mps",
)


def _best_partial_match(candidates: List[tuple[int, Any]]) -> Any | None:
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _find_environment_object_by_name(world, carla, name: str):
    name_lower = str(name).lower()
    partial_candidates = []
    for env_obj in world.get_environment_objects(carla.CityObjectLabel.Any):
        env_name = str(env_obj.name).lower()
        if env_name == name_lower:
            return env_obj
        if name_lower in env_name:
            partial_candidates.append((len(env_name), env_obj))
    return _best_partial_match(partial_candidates)


def _find_environment_objects_by_prefix(world, carla, prefix: str) -> List[Any]:
    prefix_lower = str(prefix).lower()
    matches: List[Any] = []
    for env_obj in world.get_environment_objects(carla.CityObjectLabel.Any):
        env_name = str(env_obj.name).lower()
        if env_name.startswith(prefix_lower):
            matches.append(env_obj)
    matches.sort(key=lambda item: str(getattr(item, "name", "")).lower())
    return matches


def _find_actor_by_name(world, name: str):
    name_lower = str(name).lower()
    partial_candidates = []
    for actor in world.get_actors():
        attr_name = str(actor.attributes.get("name", "")).lower()
        role_name = str(actor.attributes.get("role_name", "")).lower()
        type_id = str(actor.type_id).lower()
        if attr_name == name_lower or role_name == name_lower or type_id.endswith(name_lower):
            return actor
        if name_lower in attr_name:
            partial_candidates.append((len(attr_name), actor))
        if name_lower in role_name:
            partial_candidates.append((len(role_name), actor))
        if name_lower in type_id:
            partial_candidates.append((len(type_id), actor))
    return _best_partial_match(partial_candidates)


def _resolve_anchor_transform(world, carla, name: str):
    env_obj = _find_environment_object_by_name(world, carla, name)
    if env_obj is not None:
        return env_obj.transform

    actor = _find_actor_by_name(world, name)
    if actor is not None:
        return actor.get_transform()

    raise RuntimeError(
        f"Could not find an EnvironmentObject or Actor named '{name}'. "
        "Make sure the cube exists in the Unreal level and its name contains the keyword."
    )


def _get_location_from_anchor(world, carla, name: str):
    """Check EnvironmentObjects first, then Actors, and return a CARLA location."""
    env_obj = _find_environment_object_by_name(world, carla, name)
    if env_obj is not None:
        return env_obj.transform.location

    actor = _find_actor_by_name(world, name)
    if actor is not None:
        return actor.get_location()

    print(f"[CARLA GLOBAL ROUTE OUTPUT] Anchor '{name}' was not found in the loaded world.")
    return None


def _print_anchor_lookup(world, carla, name: str) -> None:
    env_obj = _find_environment_object_by_name(world, carla, name)
    if env_obj is not None:
        transform = env_obj.transform
        print(
            f"[CARLA SCENARIO] Found anchor '{name}' as EnvironmentObject "
            f"'{env_obj.name}' at "
            f"({transform.location.x:.3f}, {transform.location.y:.3f}, {transform.location.z:.3f}) "
            f"yaw={transform.rotation.yaw:.3f}"
        )
        return

    actor = _find_actor_by_name(world, name)
    if actor is not None:
        transform = actor.get_transform()
        print(
            f"[CARLA SCENARIO] Found anchor '{name}' as Actor "
            f"'{actor.type_id}' at "
            f"({transform.location.x:.3f}, {transform.location.y:.3f}, {transform.location.z:.3f}) "
            f"yaw={transform.rotation.yaw:.3f}"
        )
        return

    print(f"[CARLA SCENARIO] Anchor '{name}' was not found in the loaded world.")


def _align_transform_to_lane(map_obj, carla, transform):
    waypoint = map_obj.get_waypoint(
        transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if waypoint is None:
        return transform, None
    return waypoint.transform, waypoint


def _vehicle_speed_mps(vehicle) -> float:
    velocity = vehicle.get_velocity()
    return float(math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z))


def _allowed_lane_ids_from_context(
    local_context: Mapping[str, object] | None,
    fallback_lane_count: int,
    fallback_lane_id: int | None = None,
) -> list[int]:
    local_context = dict(local_context or {})
    lane_ids = [
        int(lane_id)
        for lane_id in list(local_context.get("lane_ids", []))
        if int(lane_id) != 0
    ]
    if len(lane_ids) > 0:
        return list(dict.fromkeys(lane_ids))

    context_lane_id = int(local_context.get("lane_id", 0))
    if int(context_lane_id) != 0:
        return [int(context_lane_id)]
    if fallback_lane_id is not None and int(fallback_lane_id) != 0:
        return [int(fallback_lane_id)]
    del fallback_lane_count
    return []


def _clamp_optional_lane_id_to_allowed(
    raw_lane_id: int | None,
    allowed_lane_ids: Sequence[int],
) -> int:
    if raw_lane_id is None or int(raw_lane_id) == 0:
        return 0
    return _clamp_lane_id_to_allowed(int(raw_lane_id), allowed_lane_ids)


def _clamp_lane_id_to_allowed(raw_lane_id: int | None, allowed_lane_ids: Sequence[int]) -> int:
    if len(allowed_lane_ids) == 0:
        return int(raw_lane_id or 0)
    lane_id = int(raw_lane_id if raw_lane_id is not None else allowed_lane_ids[0])
    if lane_id in allowed_lane_ids:
        return int(lane_id)
    return int(
        min(
            allowed_lane_ids,
            key=lambda candidate_lane_id: abs(int(candidate_lane_id) - int(lane_id)),
        )
    )


def _behavior_runtime_value(
    behavior_runtime_cfg: Mapping[str, object] | None,
    legacy_rule_based_cfg: Mapping[str, object] | None,
    *,
    key: str,
    legacy_key: str | None = None,
    default: object,
) -> object:
    runtime_cfg = dict(behavior_runtime_cfg or {})
    legacy_cfg = dict(legacy_rule_based_cfg or {})

    if key in runtime_cfg:
        return runtime_cfg[key]
    if legacy_key is not None and legacy_key in legacy_cfg:
        return legacy_cfg[legacy_key]
    if key in legacy_cfg:
        return legacy_cfg[key]
    return default


def _world_state_from_vehicle(vehicle) -> List[float]:
    transform = vehicle.get_transform()
    location = transform.location
    speed_mps = _vehicle_speed_mps(vehicle)
    yaw_rad = math.radians(float(transform.rotation.yaw))
    return [float(location.x), float(location.y), float(speed_mps), float(yaw_rad)]


def _spawn_vehicle(world, blueprint_library, scenario_cfg: Mapping[str, object], spawn_transform):
    ego_cfg = dict(scenario_cfg.get("ego", {}))
    blueprint_filter = str(ego_cfg.get("blueprint", "vehicle.tesla.model3"))
    role_name = str(ego_cfg.get("role_name", "ego"))
    z_offset_m = float(ego_cfg.get("spawn_z_offset_m", 1.0))

    blueprints = blueprint_library.filter(blueprint_filter)
    if not blueprints:
        raise RuntimeError(f"No CARLA vehicle blueprint matched '{blueprint_filter}'.")
    blueprint = blueprints[0]
    blueprint.set_attribute("role_name", role_name)
    color_rgb = ego_cfg.get("color_rgb", None)
    if color_rgb and blueprint.has_attribute("color"):
        blueprint.set_attribute("color", str(color_rgb))

    spawn_candidates = [0.0, z_offset_m, z_offset_m + 0.5, z_offset_m + 1.0]
    for extra_z_m in spawn_candidates:
        attempt_transform = spawn_transform.__class__(
            spawn_transform.location.__class__(
                x=float(spawn_transform.location.x),
                y=float(spawn_transform.location.y),
                z=float(spawn_transform.location.z),
            ),
            spawn_transform.rotation.__class__(
                pitch=float(spawn_transform.rotation.pitch),
                yaw=float(spawn_transform.rotation.yaw),
                roll=float(spawn_transform.rotation.roll),
            ),
        )
        attempt_transform.location.z += float(extra_z_m)
        vehicle = world.try_spawn_actor(blueprint, attempt_transform)
        if vehicle is not None:
            return vehicle
    raise RuntimeError("Failed to spawn the ego vehicle at the requested anchor.")


def _camera_blueprint(world, width_px: int, height_px: int, fov_deg: float):
    blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
    blueprint.set_attribute("image_size_x", str(int(width_px)))
    blueprint.set_attribute("image_size_y", str(int(height_px)))
    blueprint.set_attribute("fov", str(float(fov_deg)))
    return blueprint


def _spawn_camera(world, carla, blueprint, transform, parent=None):
    if parent is None:
        sensor = world.spawn_actor(
            blueprint,
            transform,
        )
    else:
        sensor = world.spawn_actor(
            blueprint,
            transform,
            attach_to=parent,
            attachment_type=carla.AttachmentType.Rigid,
        )
    image_queue: "queue.Queue[Any]" = queue.Queue(maxsize=1)

    def _on_image(image) -> None:
        if image_queue.full():
            try:
                image_queue.get_nowait()
            except queue.Empty:
                pass
        image_queue.put(image)

    sensor.listen(_on_image)
    return sensor, image_queue


def _camera_calibration_matrix(width_px: int, height_px: int, fov_deg: float) -> np.ndarray:
    focal_length_px = float(width_px) / (2.0 * math.tan(math.radians(float(fov_deg)) / 2.0))
    calibration_matrix = np.identity(3)
    calibration_matrix[0, 0] = focal_length_px
    calibration_matrix[1, 1] = focal_length_px
    calibration_matrix[0, 2] = float(width_px) / 2.0
    calibration_matrix[1, 2] = float(height_px) / 2.0
    return calibration_matrix


def _world_fixed_topdown_transform(
    carla,
    focus_points_xy: Sequence[Sequence[float]],
    image_width_px: int,
    image_height_px: int,
    fov_deg: float,
    min_height_m: float,
    padding_m: float,
):
    valid_points = [
        (float(point[0]), float(point[1]))
        for point in focus_points_xy
        if isinstance(point, Sequence) and len(point) >= 2
    ]
    if len(valid_points) == 0:
        return carla.Transform(
            carla.Location(x=0.0, y=0.0, z=float(min_height_m)),
            carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
        )

    x_values_m = [float(point[0]) for point in valid_points]
    y_values_m = [float(point[1]) for point in valid_points]
    center_x_m = 0.5 * (min(x_values_m) + max(x_values_m))
    center_y_m = 0.5 * (min(y_values_m) + max(y_values_m))
    half_span_x_m = 0.5 * max(1.0, max(x_values_m) - min(x_values_m)) + max(0.0, float(padding_m))
    half_span_y_m = 0.5 * max(1.0, max(y_values_m) - min(y_values_m)) + max(0.0, float(padding_m))

    horizontal_fov_rad = math.radians(float(fov_deg))
    vertical_fov_rad = 2.0 * math.atan(
        math.tan(0.5 * horizontal_fov_rad) * float(image_height_px) / max(1.0, float(image_width_px))
    )
    required_height_x_m = half_span_x_m / max(1e-6, math.tan(0.5 * horizontal_fov_rad))
    required_height_y_m = half_span_y_m / max(1e-6, math.tan(0.5 * vertical_fov_rad))
    camera_height_m = max(float(min_height_m), float(required_height_x_m), float(required_height_y_m))

    return carla.Transform(
        carla.Location(x=float(center_x_m), y=float(center_y_m), z=float(camera_height_m)),
        carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
    )


def _project_world_to_image(
    camera_transform,
    calibration_matrix: np.ndarray,
    world_xyz: Sequence[float],
    image_width_px: int,
    image_height_px: int,
) -> tuple[int, int] | None:
    world_point = np.array(
        [float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2]), 1.0],
        dtype=np.float64,
    )
    world_to_camera = np.array(camera_transform.get_inverse_matrix(), dtype=np.float64)
    point_camera = np.dot(world_to_camera, world_point)
    point_camera = np.array(
        [float(point_camera[1]), -float(point_camera[2]), float(point_camera[0])],
        dtype=np.float64,
    )
    if float(point_camera[2]) <= 1e-6:
        return None

    image_point = np.dot(calibration_matrix, point_camera)
    pixel_x = int(round(float(image_point[0] / image_point[2])))
    pixel_y = int(round(float(image_point[1] / image_point[2])))
    if pixel_x < 0 or pixel_x >= int(image_width_px) or pixel_y < 0 or pixel_y >= int(image_height_px):
        return None
    return pixel_x, pixel_y


def _draw_dotted_polyline(surface, points_px: Sequence[tuple[int, int]], color_rgb=(35, 210, 70), dot_spacing_px: int = 12, radius_px: int = 3) -> None:
    if pygame is None or len(points_px) < 2:
        return

    spacing_px = max(2, int(dot_spacing_px))
    radius_px = max(1, int(radius_px))
    for idx in range(len(points_px) - 1):
        x0_px, y0_px = points_px[idx]
        x1_px, y1_px = points_px[idx + 1]
        dx_px = float(x1_px - x0_px)
        dy_px = float(y1_px - y0_px)
        segment_length_px = math.hypot(dx_px, dy_px)
        if segment_length_px <= 1e-6:
            pygame.draw.circle(surface, color_rgb, (int(x0_px), int(y0_px)), radius_px)
            continue
        steps = max(1, int(segment_length_px / float(spacing_px)))
        for step_idx in range(steps + 1):
            alpha = float(step_idx) / float(steps)
            dot_x_px = int(round(float(x0_px) + alpha * dx_px))
            dot_y_px = int(round(float(y0_px) + alpha * dy_px))
            pygame.draw.circle(surface, color_rgb, (dot_x_px, dot_y_px), radius_px)


def _split_projected_polyline_segments(
    projected_points: Sequence[tuple[int, int] | None],
) -> List[List[tuple[int, int]]]:
    segments: List[List[tuple[int, int]]] = []
    current_segment: List[tuple[int, int]] = []
    for point_px in projected_points:
        if point_px is None:
            if len(current_segment) >= 2:
                segments.append(list(current_segment))
            current_segment = []
            continue
        current_segment.append((int(point_px[0]), int(point_px[1])))
    if len(current_segment) >= 2:
        segments.append(list(current_segment))
    return segments


def _split_route_world_segments(
    route_points: Sequence[Sequence[float]],
    *,
    max_gap_m: float = 12.0,
) -> List[List[tuple[float, float]]]:
    segments: List[List[tuple[float, float]]] = []
    current_segment: List[tuple[float, float]] = []
    previous_point_xy: tuple[float, float] | None = None
    gap_threshold_m = max(0.5, float(max_gap_m))

    for point_xy in route_points:
        if len(point_xy) < 2:
            continue
        current_point_xy = (float(point_xy[0]), float(point_xy[1]))
        if (
            previous_point_xy is not None
            and math.hypot(
                current_point_xy[0] - previous_point_xy[0],
                current_point_xy[1] - previous_point_xy[1],
            ) > gap_threshold_m
        ):
            if len(current_segment) >= 2:
                segments.append(list(current_segment))
            current_segment = []
        current_segment.append(current_point_xy)
        previous_point_xy = current_point_xy

    if len(current_segment) >= 2:
        segments.append(list(current_segment))
    return segments


def _draw_world_debug_route(world, carla, route_points: Sequence[Sequence[float]], life_time_s: float = 60.0) -> None:
    if len(route_points) == 0:
        return
    debug = getattr(world, "debug", None)
    if debug is None:
        return
    yellow = carla.Color(255, 255, 0)
    for segment_world_points in _split_route_world_segments(route_points):
        elevated_points = [
            carla.Location(x=float(point_xy[0]), y=float(point_xy[1]), z=0.5)
            for point_xy in segment_world_points
        ]
        for idx, point in enumerate(elevated_points):
            debug.draw_point(
                point,
                size=0.15,
                color=yellow,
                life_time=float(life_time_s),
            )
            if idx == 0:
                continue
            debug.draw_line(
                elevated_points[idx - 1],
                point,
                thickness=0.1,
                color=yellow,
                life_time=float(life_time_s),
            )


def _route_points_for_visualization(
    route_points: Sequence[Sequence[float]] | None,
    *,
    enabled: bool,
) -> List[List[float]] | None:
    if not bool(enabled) or route_points is None:
        return None
    return [
        [float(point[0]), float(point[1])]
        for point in route_points
        if len(point) >= 2
    ]


def _draw_planning_overlay(
    *,
    surface,
    camera_transform,
    calibration_matrix: np.ndarray,
    image_width_px: int,
    image_height_px: int,
    overlay_z_m: float,
    global_route_points: Sequence[Sequence[float]] | None,
    temporary_destination_state: Sequence[float] | None,
    planned_trajectory_states: Sequence[Sequence[float]] | None,
    obstacle_field_contours: Sequence[Mapping[str, object]] | None,
) -> None:
    if pygame is None:
        return

    if global_route_points is not None and len(global_route_points) >= 2:
        for route_segment_world in _split_route_world_segments(global_route_points):
            projected_points_px = [
                _project_world_to_image(
                    camera_transform=camera_transform,
                    calibration_matrix=calibration_matrix,
                    world_xyz=[float(point_xy[0]), float(point_xy[1]), float(overlay_z_m)],
                    image_width_px=image_width_px,
                    image_height_px=image_height_px,
                )
                for point_xy in route_segment_world
            ]
            for route_points_px in _split_projected_polyline_segments(projected_points_px):
                if len(route_points_px) < 2:
                    continue
                _draw_dotted_polyline(
                    surface,
                    route_points_px,
                    color_rgb=(255, 220, 60),
                    dot_spacing_px=12,
                    radius_px=4,
                )

    if temporary_destination_state is not None and len(temporary_destination_state) >= 2:
        temp_destination_pixel = _project_world_to_image(
            camera_transform=camera_transform,
            calibration_matrix=calibration_matrix,
            world_xyz=[
                float(temporary_destination_state[0]),
                float(temporary_destination_state[1]),
                float(overlay_z_m),
            ],
            image_width_px=image_width_px,
            image_height_px=image_height_px,
        )
        if temp_destination_pixel is not None:
            pygame.draw.circle(surface, (40, 170, 255), temp_destination_pixel, 7)
            pygame.draw.circle(surface, (20, 20, 20), temp_destination_pixel, 7, width=1)

    if planned_trajectory_states is not None and len(planned_trajectory_states) >= 2:
        trajectory_points_px: List[tuple[int, int]] = []
        for state in planned_trajectory_states:
            if len(state) < 2:
                continue
            pixel = _project_world_to_image(
                camera_transform=camera_transform,
                calibration_matrix=calibration_matrix,
                world_xyz=[float(state[0]), float(state[1]), float(overlay_z_m)],
                image_width_px=image_width_px,
                image_height_px=image_height_px,
            )
            if pixel is not None:
                trajectory_points_px.append(pixel)
        _draw_dotted_polyline(surface, trajectory_points_px)

    if obstacle_field_contours is not None:
        for contour in obstacle_field_contours:
            collision_points_px: List[tuple[int, int]] = []
            safe_points_px: List[tuple[int, int]] = []
            for world_xyz in contour.get("collision_points_world", []) or []:
                pixel = _project_world_to_image(
                    camera_transform=camera_transform,
                    calibration_matrix=calibration_matrix,
                    world_xyz=world_xyz,
                    image_width_px=image_width_px,
                    image_height_px=image_height_px,
                )
                if pixel is not None:
                    collision_points_px.append(pixel)
            for world_xyz in contour.get("safe_points_world", []) or []:
                pixel = _project_world_to_image(
                    camera_transform=camera_transform,
                    calibration_matrix=calibration_matrix,
                    world_xyz=world_xyz,
                    image_width_px=image_width_px,
                    image_height_px=image_height_px,
                )
                if pixel is not None:
                    safe_points_px.append(pixel)
            if len(safe_points_px) >= 3:
                pygame.draw.lines(surface, (255, 210, 60), True, safe_points_px, width=2)
            if len(collision_points_px) >= 3:
                pygame.draw.lines(surface, (255, 80, 80), True, collision_points_px, width=2)


def _draw_hud_lines(surface, font, lines: Sequence[str], top_left_px: tuple[int, int]) -> None:
    if pygame is None or font is None or len(lines) == 0:
        return

    x0_px, y0_px = int(top_left_px[0]), int(top_left_px[1])
    line_height_px = int(font.get_linesize())
    padding_px = 6
    text_surfaces = [font.render(str(line), True, (255, 255, 255)) for line in lines]
    box_width_px = max(text_surface.get_width() for text_surface in text_surfaces) + 2 * padding_px
    box_height_px = len(text_surfaces) * line_height_px + 2 * padding_px
    box_surface = pygame.Surface((box_width_px, box_height_px), pygame.SRCALPHA)
    box_surface.fill((0, 0, 0, 140))
    surface.blit(box_surface, (x0_px, y0_px))

    for idx, text_surface in enumerate(text_surfaces):
        surface.blit(
            text_surface,
            (
                x0_px + padding_px,
                y0_px + padding_px + idx * line_height_px,
            ),
        )


def _render_camera_pair(
    display,
    left_image,
    right_image,
    topdown_overlay: Mapping[str, object] | None = None,
    hud_lines: Sequence[str] | None = None,
    hud_font=None,
):
    if pygame is None:
        return
    if left_image is not None:
        left_array = np.frombuffer(left_image.raw_data, dtype=np.uint8)
        left_array = left_array.reshape((left_image.height, left_image.width, 4))
        left_rgb = left_array[:, :, :3][:, :, ::-1]
        left_surface = pygame.surfarray.make_surface(left_rgb.swapaxes(0, 1))
        if isinstance(topdown_overlay, Mapping):
            _draw_planning_overlay(
                surface=left_surface,
                camera_transform=topdown_overlay["camera_transform"],
                calibration_matrix=topdown_overlay["calibration_matrix"],
                image_width_px=int(topdown_overlay["image_width_px"]),
                image_height_px=int(topdown_overlay["image_height_px"]),
                overlay_z_m=float(topdown_overlay["overlay_z_m"]),
                global_route_points=topdown_overlay.get("global_route_points", None),
                temporary_destination_state=topdown_overlay.get("temporary_destination_state", None),
                planned_trajectory_states=topdown_overlay.get("planned_trajectory_states", None),
                obstacle_field_contours=topdown_overlay.get("obstacle_field_contours", None),
            )
        display.blit(left_surface, (0, 0))

    if right_image is not None:
        right_array = np.frombuffer(right_image.raw_data, dtype=np.uint8)
        right_array = right_array.reshape((right_image.height, right_image.width, 4))
        right_rgb = right_array[:, :, :3][:, :, ::-1]
        right_surface = pygame.surfarray.make_surface(right_rgb.swapaxes(0, 1))
        display.blit(right_surface, (right_image.width, 0))

    if hud_lines:
        _draw_hud_lines(display, hud_font, hud_lines, (16, 16))

    pygame.display.flip()

def _merge_tracker_predictions(
    object_snapshots: Sequence[Mapping[str, object]],
    predictions: Mapping[str, Sequence[Mapping[str, float]]],
) -> List[dict]:
    merged: List[dict] = []
    for snapshot in object_snapshots:
        snapshot_copy = dict(snapshot)
        prediction = predictions.get(str(snapshot_copy.get("vehicle_id", "")), [])
        snapshot_copy["predicted_trajectory"] = [
            [
                float(item.get("x", 0.0)),
                float(item.get("y", 0.0)),
                float(item.get("v", 0.0)),
                float(item.get("psi", 0.0)),
            ]
            for item in prediction
        ]
        merged.append(snapshot_copy)
    return merged


def _call_with_supported_kwargs(func, **kwargs):
    signature = inspect.signature(func)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return func(**kwargs)
    supported_kwargs = {
        name: value
        for name, value in kwargs.items()
        if name in signature.parameters
    }
    return func(**supported_kwargs)


def _load_optional_module(*, module_name: str, purpose: str):
    normalized_module_name = str(module_name).strip()
    if not normalized_module_name:
        return None
    try:
        return importlib.import_module(normalized_module_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import {purpose} module '{normalized_module_name}': {exc}"
        ) from exc


def _initialize_scenario_runtime_state(
    *,
    module,
    world,
    world_map,
    carla,
    scenario_cfg: Mapping[str, object],
):
    if module is None:
        return None
    initialize_fn = getattr(module, "initialize_runtime", None)
    if not callable(initialize_fn):
        return None
    return _call_with_supported_kwargs(
        initialize_fn,
        world=world,
        world_map=world_map,
        carla=carla,
        scenario_cfg=scenario_cfg,
    )


def _apply_scenario_dynamic_obstacle_filter(
    *,
    module,
    runtime_state,
    world,
    world_map,
    carla,
    scenario_cfg: Mapping[str, object],
    object_snapshots: Sequence[Mapping[str, object]],
    sim_time_s: float,
    wall_time_s: float | None = None,
) -> Tuple[List[dict], Any]:
    snapshots_copy = [dict(snapshot) for snapshot in list(object_snapshots)]
    if module is None:
        return snapshots_copy, runtime_state

    filter_fn = getattr(module, "filter_dynamic_obstacle_snapshots", None)
    if not callable(filter_fn):
        return snapshots_copy, runtime_state

    filter_result = _call_with_supported_kwargs(
        filter_fn,
        world=world,
        world_map=world_map,
        carla=carla,
        scenario_cfg=scenario_cfg,
        runtime_state=runtime_state,
        object_snapshots=snapshots_copy,
        sim_time_s=float(sim_time_s),
        wall_time_s=None if wall_time_s is None else float(wall_time_s),
    )
    if isinstance(filter_result, tuple) and len(filter_result) == 2:
        filtered_snapshots, updated_runtime_state = filter_result
    else:
        filtered_snapshots = filter_result
        updated_runtime_state = runtime_state
    filtered_list = [dict(snapshot) for snapshot in list(filtered_snapshots or [])]
    return filtered_list, updated_runtime_state


def _maybe_apply_scenario_global_route_update(
    *,
    module,
    runtime_state,
    world,
    world_map,
    carla,
    scenario_cfg: Mapping[str, object],
    global_planner: AStarGlobalPlanner,
    ego_transform,
    goal_location,
    object_snapshots: Sequence[Mapping[str, object]],
    current_route_summary,
    active_global_route_points: Sequence[Sequence[float]],
    sim_time_s: float,
    wall_time_s: float | None = None,
) -> Tuple[object | None, List[List[float]] | None, Any]:
    if module is None:
        return None, None, runtime_state

    update_fn = getattr(module, "maybe_replan_global_route", None)
    if not callable(update_fn):
        return None, None, runtime_state

    update_result = _call_with_supported_kwargs(
        update_fn,
        world=world,
        world_map=world_map,
        carla=carla,
        scenario_cfg=scenario_cfg,
        runtime_state=runtime_state,
        global_planner=global_planner,
        ego_transform=ego_transform,
        goal_location=goal_location,
        object_snapshots=[dict(snapshot) for snapshot in list(object_snapshots or [])],
        current_route_summary=current_route_summary,
        active_global_route_points=[list(point) for point in list(active_global_route_points or [])],
        sim_time_s=float(sim_time_s),
        wall_time_s=None if wall_time_s is None else float(wall_time_s),
    )
    if isinstance(update_result, tuple):
        if len(update_result) == 3:
            route_summary, route_points, updated_runtime_state = update_result
            return route_summary, route_points, updated_runtime_state
        if len(update_result) == 2:
            route_summary, route_points = update_result
            return route_summary, route_points, runtime_state
    if isinstance(update_result, Mapping):
        return (
            update_result.get("route_summary", None),
            update_result.get("route_points", None),
            update_result.get("runtime_state", runtime_state),
        )
    return None, None, runtime_state


def _filter_obstacle_snapshots_by_vertical_overlap(
    *,
    ego_z_m: float,
    ego_height_m: float,
    object_snapshots: Sequence[Mapping[str, object]],
    vertical_clearance_margin_m: float,
    default_obstacle_height_m: float,
) -> List[dict]:
    filtered_snapshots: List[dict] = []
    ego_height_value_m = max(0.5, float(ego_height_m))
    margin_m = max(0.0, float(vertical_clearance_margin_m))
    fallback_height_m = max(0.5, float(default_obstacle_height_m))

    for snapshot in object_snapshots:
        snapshot_copy = dict(snapshot)
        obstacle_z_raw = snapshot_copy.get("z", None)
        if obstacle_z_raw is None:
            filtered_snapshots.append(snapshot_copy)
            continue

        obstacle_height_m = max(
            0.5,
            float(snapshot_copy.get("height_m", fallback_height_m)),
        )
        vertical_gap_m = abs(float(obstacle_z_raw) - float(ego_z_m))
        overlap_threshold_m = 0.5 * (ego_height_value_m + obstacle_height_m) + margin_m
        if vertical_gap_m <= overlap_threshold_m:
            filtered_snapshots.append(snapshot_copy)

    return filtered_snapshots


def _road_numeric_id_from_context(local_context: Mapping[str, object] | None) -> int:
    if not isinstance(local_context, Mapping):
        return -1
    raw_numeric_id = local_context.get("road_numeric_id", None)
    if raw_numeric_id is not None:
        try:
            return int(raw_numeric_id)
        except Exception:
            pass
    raw_road_id = str(local_context.get("road_id", "") or "").strip()
    if ":" in raw_road_id:
        raw_road_id = raw_road_id.split(":", 1)[0]
    try:
        return int(raw_road_id)
    except Exception:
        return -1


def _same_lane_safety_corridor(
    ego_lane_context: Mapping[str, object] | None,
    obstacle_lane_context: Mapping[str, object] | None,
) -> bool:
    if not isinstance(ego_lane_context, Mapping) or not isinstance(obstacle_lane_context, Mapping):
        return False
    ego_direction = str(ego_lane_context.get("direction", "") or "").strip().lower()
    obstacle_direction = str(obstacle_lane_context.get("direction", "") or "").strip().lower()
    if ego_direction and obstacle_direction and ego_direction != obstacle_direction:
        return False

    ego_road_numeric_id = _road_numeric_id_from_context(ego_lane_context)
    obstacle_road_numeric_id = _road_numeric_id_from_context(obstacle_lane_context)
    if ego_road_numeric_id > 0 and obstacle_road_numeric_id > 0:
        return int(ego_road_numeric_id) == int(obstacle_road_numeric_id)

    ego_road_id = str(ego_lane_context.get("road_id", "") or "").strip()
    obstacle_road_id = str(obstacle_lane_context.get("road_id", "") or "").strip()
    if ego_road_id and obstacle_road_id:
        return str(ego_road_id) == str(obstacle_road_id)
    return False


def _nearest_front_obstacle_by_lane(
    *,
    ego_snapshot: Mapping[str, float],
    obstacle_snapshots: Sequence[Mapping[str, object]],
    lane_assignments: Mapping[str, int],
    available_lane_ids: Sequence[int],
) -> Dict[int, dict]:
    ego_x_m = float(ego_snapshot.get("x", 0.0))
    ego_y_m = float(ego_snapshot.get("y", 0.0))
    ego_psi_rad = float(ego_snapshot.get("psi", 0.0))
    cos_heading = math.cos(float(ego_psi_rad))
    sin_heading = math.sin(float(ego_psi_rad))

    nearest_obstacles: Dict[int, dict] = {}
    valid_lane_ids = {int(lane_id) for lane_id in list(available_lane_ids or [])}

    for obstacle_snapshot in obstacle_snapshots:
        obstacle_id = str(obstacle_snapshot.get("vehicle_id", ""))
        lane_id = int(lane_assignments.get(obstacle_id, -1))
        if lane_id not in valid_lane_ids:
            continue

        dx_m = float(obstacle_snapshot.get("x", 0.0)) - float(ego_x_m)
        dy_m = float(obstacle_snapshot.get("y", 0.0)) - float(ego_y_m)
        longitudinal_m = float(cos_heading) * float(dx_m) + float(sin_heading) * float(dy_m)
        if float(longitudinal_m) < 0.0:
            continue

        previous_distance_m = float(
            nearest_obstacles.get(int(lane_id), {}).get("front_distance_m", float("inf"))
        )
        if float(longitudinal_m) < float(previous_distance_m):
            obstacle_copy = dict(obstacle_snapshot)
            obstacle_copy["front_distance_m"] = float(longitudinal_m)
            nearest_obstacles[int(lane_id)] = obstacle_copy

    return nearest_obstacles


def _nearest_front_obstacle_distance_by_lane(
    *,
    ego_snapshot: Mapping[str, float],
    obstacle_snapshots: Sequence[Mapping[str, object]],
    lane_assignments: Mapping[str, int],
    available_lane_ids: Sequence[int],
) -> Dict[int, float]:
    nearest_obstacles = _nearest_front_obstacle_by_lane(
        ego_snapshot=ego_snapshot,
        obstacle_snapshots=obstacle_snapshots,
        lane_assignments=lane_assignments,
        available_lane_ids=available_lane_ids,
    )
    return {
        int(lane_id): float(obstacle.get("front_distance_m", float("inf")))
        for lane_id, obstacle in nearest_obstacles.items()
    }


def _should_force_intersection_reroute(
    *,
    mode: str,
    ego_lane_id: int,
    lane_safety_scores: Mapping[int, float],
    nearest_front_obstacles_by_lane: Mapping[int, Mapping[str, object]],
    safety_threshold: float,
) -> bool:
    if str(mode or "NORMAL").strip().upper() != "INTERSECTION":
        return False
    if int(ego_lane_id) not in nearest_front_obstacles_by_lane:
        return False
    return float(lane_safety_scores.get(int(ego_lane_id), 1.0)) < float(safety_threshold)


def _static_obstacle_replan_candidate_lane_ids(
    *,
    current_lane_id: int,
    available_lane_ids: Sequence[int],
    lane_safety_scores: Mapping[int, float],
) -> List[int]:
    lane_ids = [int(lane_id) for lane_id in list(available_lane_ids or [])]
    if len(lane_ids) == 0:
        return []

    normalized_current_lane_id = int(current_lane_id)
    if normalized_current_lane_id not in lane_ids:
        normalized_current_lane_id = min(
            lane_ids,
            key=lambda lane_id: abs(int(lane_id) - int(current_lane_id)),
        )

    alternative_lane_ids = [
        lane_id
        for lane_id in lane_ids
        if int(lane_id) != int(normalized_current_lane_id)
    ]
    alternative_lane_ids.sort(
        key=lambda lane_id: (
            -float(lane_safety_scores.get(int(lane_id), 0.0)),
            abs(int(lane_id) - int(normalized_current_lane_id)),
            int(lane_id),
        )
    )
    return list(alternative_lane_ids) + [int(normalized_current_lane_id)]


def _replan_route_around_static_intersection_obstacle(
    *,
    global_planner: AStarGlobalPlanner,
    ego_transform,
    goal_location,
    blocked_obstacle_snapshot: Mapping[str, object] | None,
    blocked_lane_id: int,
) -> tuple[object | None, List[List[float]]]:
    if blocked_obstacle_snapshot is None:
        return None, []

    start_xy = [
        float(ego_transform.location.x),
        float(ego_transform.location.y),
    ]
    goal_xy = [
        float(goal_location.x),
        float(goal_location.y),
    ]
    blocked_point_xy = [
        float(blocked_obstacle_snapshot.get("x", 0.0)),
        float(blocked_obstacle_snapshot.get("y", 0.0)),
    ]
    obstacle_length_m = max(0.0, float(blocked_obstacle_snapshot.get("length_m", 4.5)))
    obstacle_width_m = max(0.0, float(blocked_obstacle_snapshot.get("width_m", 2.0)))
    block_radius_m = max(
        6.0,
        0.5 * max(float(obstacle_length_m), float(obstacle_width_m)) + 4.0,
    )
    route_summary = global_planner.plan_route_astar_avoiding_points(
        start_xy=start_xy,
        goal_xy=goal_xy,
        blocked_points_xy=[blocked_point_xy],
        blocked_lane_ids=[int(blocked_lane_id)],
        block_radius_m=float(block_radius_m),
        replace_stored_route=True,
    )
    if not bool(getattr(route_summary, "route_found", False)):
        return None, []

    route_points = [
        [float(item[0]), float(item[1])]
        for item in list(getattr(route_summary, "route_waypoints", []) or [])
        if isinstance(item, Sequence) and len(item) >= 2
    ]
    if len(route_points) == 0:
        return None, []
    return route_summary, route_points


def _apply_final_destination_snap(
    *,
    temporary_destination_state: Sequence[float] | None,
    final_destination_state: Sequence[float],
    ego_state: Sequence[float],
    lock_to_final_distance_m: float,
    original_max_velocity_mps: float,
) -> tuple[List[float] | None, float]:
    if temporary_destination_state is None:
        return None, float(original_max_velocity_mps)
    if len(temporary_destination_state) < 4 or len(final_destination_state) < 4:
        return list(temporary_destination_state), float(original_max_velocity_mps)

    snap_distance_threshold_m = max(0.0, float(lock_to_final_distance_m))
    distance_temp_to_final_m = math.hypot(
        float(temporary_destination_state[0]) - float(final_destination_state[0]),
        float(temporary_destination_state[1]) - float(final_destination_state[1]),
    )
    if distance_temp_to_final_m > float(snap_distance_threshold_m):
        return list(temporary_destination_state), float(original_max_velocity_mps)

    snapped_destination_state = list(temporary_destination_state)
    snapped_destination_state[0] = float(final_destination_state[0])
    snapped_destination_state[1] = float(final_destination_state[1])
    snapped_destination_state[3] = float(final_destination_state[3])

    if len(ego_state) < 2 or float(snap_distance_threshold_m) <= 1e-6:
        active_max_velocity_mps = 0.0
    else:
        distance_ego_to_final_m = math.hypot(
            float(ego_state[0]) - float(final_destination_state[0]),
            float(ego_state[1]) - float(final_destination_state[1]),
        )
        stop_speed_scale = max(
            0.0,
            min(1.0, float(distance_ego_to_final_m) / float(snap_distance_threshold_m)),
        )
        active_max_velocity_mps = float(original_max_velocity_mps) * float(stop_speed_scale)

    if len(snapped_destination_state) >= 3:
        snapped_destination_state[2] = float(active_max_velocity_mps)

    return snapped_destination_state, float(active_max_velocity_mps)


def _apply_stop_target_speed_cap(
    *,
    temporary_destination_state: Sequence[float] | None,
    ego_state: Sequence[float],
    stop_target_distance_m: float | None,
    original_max_velocity_mps: float,
    braking_deceleration_mps2: float,
    stop_buffer_m: float,
) -> tuple[List[float] | None, float]:
    if temporary_destination_state is None:
        return None, float(original_max_velocity_mps)

    shaped_destination_state = list(temporary_destination_state)
    if len(shaped_destination_state) < 3:
        return shaped_destination_state, float(original_max_velocity_mps)

    if stop_target_distance_m is None:
        if len(ego_state) >= 2 and len(shaped_destination_state) >= 2:
            remaining_distance_m = math.hypot(
                float(shaped_destination_state[0]) - float(ego_state[0]),
                float(shaped_destination_state[1]) - float(ego_state[1]),
            )
        else:
            remaining_distance_m = 0.0
    else:
        remaining_distance_m = max(0.0, float(stop_target_distance_m))

    effective_braking_deceleration_mps2 = max(1.0e-6, float(braking_deceleration_mps2))
    remaining_brake_distance_m = max(
        0.0,
        float(remaining_distance_m) - max(0.0, float(stop_buffer_m)),
    )
    speed_cap_mps = min(
        float(original_max_velocity_mps),
        math.sqrt(2.0 * float(effective_braking_deceleration_mps2) * float(remaining_brake_distance_m)),
    )
    shaped_destination_state[2] = float(speed_cap_mps)
    return shaped_destination_state, float(speed_cap_mps)


def _stop_target_state_from_behavior_output(
    stop_target: Mapping[str, object] | None,
) -> List[float] | None:
    if not isinstance(stop_target, Mapping):
        return None
    try:
        return [
            float(stop_target.get("x_m", 0.0)),
            float(stop_target.get("y_m", 0.0)),
            0.0,
            float(stop_target.get("heading_rad", 0.0)),
            float(stop_target.get("lane_id", 0)),
            0.0,
            float(stop_target.get("road_id", -1)),
            0.0,
        ]
    except Exception:
        return None


def _sample_superellipse_contour_world(
    *,
    center_x_m: float,
    center_y_m: float,
    center_z_m: float,
    heading_rad: float,
    half_length_m: float,
    half_width_m: float,
    shape_exponent: float,
    num_points: int = 72,
) -> List[List[float]]:
    half_length_m = max(1e-6, float(half_length_m))
    half_width_m = max(1e-6, float(half_width_m))
    shape_exponent = max(2.0, float(shape_exponent))
    cos_heading = math.cos(float(heading_rad))
    sin_heading = math.sin(float(heading_rad))
    contour_world: List[List[float]] = []
    exponent = 2.0 / float(shape_exponent)
    for idx in range(max(8, int(num_points))):
        theta_rad = 2.0 * math.pi * float(idx) / float(max(8, int(num_points)))
        cos_theta = math.cos(theta_rad)
        sin_theta = math.sin(theta_rad)
        local_x_m = float(half_length_m) * math.copysign(abs(cos_theta) ** exponent, cos_theta)
        local_y_m = float(half_width_m) * math.copysign(abs(sin_theta) ** exponent, sin_theta)
        world_x_m = float(center_x_m) + local_x_m * cos_heading - local_y_m * sin_heading
        world_y_m = float(center_y_m) + local_x_m * sin_heading + local_y_m * cos_heading
        contour_world.append([float(world_x_m), float(world_y_m), float(center_z_m)])
    return contour_world


def _static_obstacle_prediction(snapshot: Mapping[str, object], horizon_steps: int) -> List[List[float]]:
    state = [
        float(snapshot.get("x", 0.0)),
        float(snapshot.get("y", 0.0)),
        float(snapshot.get("v", 0.0)),
        float(snapshot.get("psi", 0.0)),
    ]
    return [list(state) for _ in range(max(1, int(horizon_steps)))]


def _collect_vehicle_snapshots(world, ego_vehicle) -> List[dict]:
    snapshots: List[dict] = []
    for actor in world.get_actors().filter("vehicle.*"):
        if int(actor.id) == int(ego_vehicle.id):
            continue
        transform = actor.get_transform()
        velocity = actor.get_velocity()
        speed_mps = math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z)
        snapshots.append(
            {
                "vehicle_id": str(actor.attributes.get("role_name", actor.id)),
                "x": float(transform.location.x),
                "y": float(transform.location.y),
                "z": float(transform.location.z),
                "v": float(speed_mps),
                "psi": float(math.radians(transform.rotation.yaw)),
                "length_m": float(actor.bounding_box.extent.x * 2.0),
                "width_m": float(actor.bounding_box.extent.y * 2.0),
                "height_m": float(actor.bounding_box.extent.z * 2.0),
            }
        )
    return snapshots


def _collect_environment_obstacle_snapshots(world, map_obj, carla, obstacle_prefix: str) -> List[dict]:
    snapshots: List[dict] = []
    if not str(obstacle_prefix).strip():
        return snapshots

    for env_obj in _find_environment_objects_by_prefix(world, carla, obstacle_prefix):
        transform = env_obj.transform
        location = transform.location
        nearest_waypoint = map_obj.get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        heading_rad = float(math.radians(transform.rotation.yaw))
        if nearest_waypoint is not None:
            heading_rad = float(math.radians(nearest_waypoint.transform.rotation.yaw))

        half_length_m = 2.25
        half_width_m = 1.0
        half_height_m = 1.0
        bbox = getattr(env_obj, "bounding_box", None)
        if bbox is not None:
            half_length_m = max(0.05, float(getattr(bbox.extent, "x", half_length_m)))
            half_width_m = max(0.05, float(getattr(bbox.extent, "y", half_width_m)))
            half_height_m = max(0.05, float(getattr(bbox.extent, "z", half_height_m)))

        snapshots.append(
            {
                "vehicle_id": str(getattr(env_obj, "name", f"{obstacle_prefix}_{len(snapshots)+1}")),
                "x": float(location.x),
                "y": float(location.y),
                "z": float(location.z),
                "v": 0.0,
                "psi": float(heading_rad),
                "length_m": 2.0 * float(half_length_m),
                "width_m": 2.0 * float(half_width_m),
                "height_m": 2.0 * float(half_height_m),
                "type": "static_obstacle",
                "predicted_trajectory": _static_obstacle_prediction(
                    {
                        "x": float(location.x),
                        "y": float(location.y),
                        "v": 0.0,
                        "psi": float(heading_rad),
                    },
                    horizon_steps=1,
                ),
            }
        )
    return snapshots


def _build_obstacle_field_contours(
    *,
    mpc: MPC,
    ego_state: Sequence[float],
    object_snapshots: Sequence[Mapping[str, object]],
) -> List[dict]:
    contours: List[dict] = []
    for snapshot in object_snapshots:
        obstacle_length_m = float(snapshot.get("length_m", 4.5))
        obstacle_width_m = float(snapshot.get("width_m", 2.0))
        obstacle_state = [
            float(snapshot.get("x", 0.0)),
            float(snapshot.get("y", 0.0)),
            float(snapshot.get("v", 0.0)),
            float(snapshot.get("psi", 0.0)),
        ]
        geometry = mpc._superellipsoid_zone_geometry(
            ego_state=ego_state,
            obstacle_state=obstacle_state,
            obstacle_length_m=obstacle_length_m,
            obstacle_width_m=obstacle_width_m,
        )
        obstacle_z_m = float(snapshot.get("z", 0.0))
        contours.append(
            {
                "label": str(snapshot.get("vehicle_id", "")),
                "safe_points_world": _sample_superellipse_contour_world(
                    center_x_m=float(geometry["obstacle_x_m"]),
                    center_y_m=float(geometry["obstacle_y_m"]),
                    center_z_m=obstacle_z_m,
                    heading_rad=float(geometry["obstacle_psi_rad"]),
                    half_length_m=float(geometry["xs_m"]),
                    half_width_m=float(geometry["ys_m"]),
                    shape_exponent=float(geometry["shape_exponent"]),
                ),
                "collision_points_world": _sample_superellipse_contour_world(
                    center_x_m=float(geometry["obstacle_x_m"]),
                    center_y_m=float(geometry["obstacle_y_m"]),
                    center_z_m=obstacle_z_m,
                    heading_rad=float(geometry["obstacle_psi_rad"]),
                    half_length_m=float(geometry["xc_m"]),
                    half_width_m=float(geometry["yc_m"]),
                    shape_exponent=float(geometry["shape_exponent"]),
                ),
            }
        )
    return contours


def _control_from_mpc(mpc: MPC, carla, acceleration_mps2: float, steering_angle_rad: float):
    max_accel_mps2 = max(1e-6, float(mpc.constraints.max_acceleration_mps2))
    max_brake_mps2 = max(1e-6, abs(float(mpc.constraints.min_acceleration_mps2)))
    max_steer_rad = max(1e-6, float(mpc.constraints.max_steer_rad))

    throttle = 0.0
    brake = 0.0
    if float(acceleration_mps2) >= 0.0:
        throttle = min(1.0, max(0.0, float(acceleration_mps2) / max_accel_mps2))
    else:
        brake = min(1.0, max(0.0, abs(float(acceleration_mps2)) / max_brake_mps2))

    steer = min(1.0, max(-1.0, float(steering_angle_rad) / max_steer_rad))
    return carla.VehicleControl(
        throttle=float(throttle),
        brake=float(brake),
        steer=float(steer),
        hand_brake=False,
        reverse=False,
        manual_gear_shift=False,
    )


def _destroy_actors(actors: Iterable[Any]) -> None:
    for actor in actors:
        if actor is None:
            continue
        try:
            if hasattr(actor, "stop"):
                actor.stop()
        except RuntimeError:
            pass
        try:
            actor.destroy()
        except RuntimeError:
            pass


def _spawn_scenario_obstacles_from_module(
    *,
    client,
    world,
    map_obj,
    carla,
    blueprint_library,
    traffic_manager,
    traffic_manager_port: int,
    scenario_cfg: Mapping[str, object],
    route_summary,
    route_points: Sequence[Sequence[float]],
) -> List[Any]:
    obstacle_cfg = dict(scenario_cfg.get("obstacles", {}))
    module_name = str(obstacle_cfg.get("spawner_module", "")).strip()
    if not module_name:
        return []

    print(
        "[CARLA SCENARIO] Spawning scenario obstacles after startup global route generation "
        f"using module '{module_name}'."
    )
    module = _load_optional_module(
        module_name=str(module_name),
        purpose="scenario obstacle spawner",
    )

    spawn_fn = getattr(module, "spawn_obstacles", None)
    if not callable(spawn_fn):
        raise RuntimeError(
            f"Scenario obstacle spawner module '{module_name}' does not expose "
            "spawn_obstacles(...)."
        )

    spawned_actors = _call_with_supported_kwargs(
        spawn_fn,
        client=client,
        world=world,
        world_map=map_obj,
        carla=carla,
        blueprint_library=blueprint_library,
        traffic_manager=traffic_manager,
        traffic_manager_port=int(traffic_manager_port),
        scenario_cfg=scenario_cfg,
        route_summary=route_summary,
        route_points=route_points,
    )
    if spawned_actors is None:
        return []
    actors = [actor for actor in list(spawned_actors) if actor is not None]
    print(
        f"[CARLA SCENARIO] Scenario obstacle spawner '{module_name}' "
        f"spawned {len(actors)} actor(s)."
    )
    return actors


def _scenario_constraints_cfg(scenario_cfg: Mapping[str, object]) -> Dict[str, object]:
    constraints_cfg = dict(scenario_cfg.get("constraints", {}))
    missing_keys = [key for key in REQUIRED_CONSTRAINT_KEYS if key not in constraints_cfg]
    if len(missing_keys) > 0:
        raise RuntimeError(
            "Scenario constraints are incomplete. "
            f"Missing keys in scenario '{scenario_cfg.get('name', '<unknown>')}': {missing_keys}"
        )
    return constraints_cfg


def run_loaded_world(client, world, scenario_cfg: Mapping[str, object], carla) -> int:
    scenario_cfg = dict(scenario_cfg or {})
    carla_cfg = dict(scenario_cfg.get("carla", {}))
    anchors_cfg = dict(scenario_cfg.get("anchors", {}))
    planning_cfg = dict(scenario_cfg.get("planning", {}))
    camera_cfg = dict(scenario_cfg.get("camera", {}))
    obstacle_cfg = dict(scenario_cfg.get("obstacles", {}))
    runtime_cfg = dict(scenario_cfg.get("runtime", {}))
    traffic_manager_cfg = dict(scenario_cfg.get("traffic_manager", {}))

    if pygame is None and bool(camera_cfg.get("enabled", True)):
        raise RuntimeError("pygame is required for the two-camera CARLA scenario window.")

    world_map = world.get_map()
    blueprint_library = world.get_blueprint_library()
    mpc_payload = load_yaml_file(MPC_CONFIG_PATH)
    tracker_payload = load_yaml_file(TRACKER_CONFIG_PATH)
    mpc_cfg = dict(mpc_payload.get("mpc", mpc_payload))
    scenario_constraints_cfg = _scenario_constraints_cfg(scenario_cfg)
    mpc_cfg["constraints"] = dict(scenario_constraints_cfg)
    tracker_cfg = dict(tracker_payload.get("tracker", tracker_payload))
    behavior_runtime_cfg = dict(mpc_cfg.get("behavior_planner_runtime", {}))
    local_goal_cfg = dict(mpc_cfg.get("local_goal", {}))
    mpc_constraints_cfg = dict(mpc_cfg.get("constraints", {}))
    obstacle_filter_cfg = dict(mpc_cfg.get("obstacle_filter", {}))
    visualization_cfg = dict(mpc_cfg.get("visualization", {}))
    global_route_visualization_enabled = bool(
        visualization_cfg.get("show_global_route", True)
    )

    sample_distance_m = float(planning_cfg.get("waypoint_sample_distance_m", 2.0))
    lane_center_waypoints, road_cfg = build_lane_center_waypoints(
        map_obj=world_map,
        carla=carla,
        sample_distance_m=float(sample_distance_m),
    )
    global_planner = AStarGlobalPlanner(
        lane_center_waypoints=lane_center_waypoints,
        world_map=world_map,
        route_sample_distance_m=float(sample_distance_m),
    )

    ego_anchor_name = str(anchors_cfg.get("ego_spawn", "cav_spawn"))
    destination_anchor_name = str(anchors_cfg.get("final_destination", "final_destination"))
    global_route_start_location = _get_location_from_anchor(world, carla, ego_anchor_name)
    global_route_goal_location = _get_location_from_anchor(world, carla, destination_anchor_name)
    if global_route_start_location is None or global_route_goal_location is None:
        raise RuntimeError(
            "Could not find the required global-route anchors. "
            f"ego_spawn='{ego_anchor_name}' final_destination='{destination_anchor_name}'."
        )
    spawn_anchor = _resolve_anchor_transform(world, carla, ego_anchor_name)
    destination_anchor = _resolve_anchor_transform(world, carla, destination_anchor_name)
    aligned_spawn_transform, spawn_waypoint = _align_transform_to_lane(world_map, carla, spawn_anchor)
    aligned_destination_transform, destination_waypoint = _align_transform_to_lane(world_map, carla, destination_anchor)
    if spawn_waypoint is None or destination_waypoint is None:
        raise RuntimeError("Could not align the spawn or destination anchors to a driving lane.")

    initial_global_route_summary = global_planner.plan_route_from_locations(
        start_location=global_route_start_location,
        goal_location=global_route_goal_location,
        fallback_start_xy=[
            float(global_route_start_location.x),
            float(global_route_start_location.y),
        ],
        fallback_goal_xy=[
            float(global_route_goal_location.x),
            float(global_route_goal_location.y),
        ],
    )
    initial_route_points: List[List[float]] = []
    if bool(initial_global_route_summary.route_found):
        initial_route_points = [
            [float(item[0]), float(item[1])]
            for item in initial_global_route_summary.route_waypoints
        ]
    topdown_focus_points: List[List[float]] = list(initial_route_points)
    if len(topdown_focus_points) == 0:
        topdown_focus_points = [
            [float(global_route_start_location.x), float(global_route_start_location.y)],
            [float(global_route_goal_location.x), float(global_route_goal_location.y)],
        ]

    ego_vehicle = _spawn_vehicle(
        world=world,
        blueprint_library=blueprint_library,
        scenario_cfg=scenario_cfg,
        spawn_transform=aligned_spawn_transform,
    )

    traffic_manager = None
    traffic_manager_port = int(traffic_manager_cfg.get("port", 8000))
    if bool(traffic_manager_cfg.get("enabled", False)):
        traffic_manager = client.get_trafficmanager(int(traffic_manager_port))

    actors_to_destroy = [ego_vehicle]
    actors_to_destroy.extend(
        _spawn_scenario_obstacles_from_module(
            client=client,
            world=world,
            map_obj=world_map,
            carla=carla,
            blueprint_library=blueprint_library,
            traffic_manager=traffic_manager,
            traffic_manager_port=int(traffic_manager_port),
            scenario_cfg=scenario_cfg,
            route_summary=initial_global_route_summary,
            route_points=initial_route_points,
        )
    )
    previous_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = bool(carla_cfg.get("synchronous_mode", True))
    settings.fixed_delta_seconds = CARLA_FIXED_DELTA_SECONDS
    world.apply_settings(settings)
    if traffic_manager is not None:
        try:
            traffic_manager.set_synchronous_mode(
                bool(traffic_manager_cfg.get("synchronous_mode", settings.synchronous_mode))
            )
        except Exception:
            pass
    print(
        f"[CARLA SCENARIO] Simulation tick = {CARLA_FIXED_DELTA_SECONDS:.3f}s "
        f"({1.0 / CARLA_FIXED_DELTA_SECONDS:.0f} Hz), "
        f"MPC prediction dt = {float(mpc_cfg.get('plan_dt_s', 0.2)):.3f}s"
    )
    realtime_pacing_enabled = bool(carla_cfg.get("realtime_pacing_enabled", False))
    realtime_pacing_factor = max(1e-3, float(carla_cfg.get("realtime_pacing_factor", 1.0)))
    realtime_loop_period_s = 0.0
    if realtime_pacing_enabled:
        realtime_loop_period_s = CARLA_FIXED_DELTA_SECONDS / float(realtime_pacing_factor)
        print(
            "[CARLA SCENARIO] Real-time pacing enabled "
            f"(wall_period={float(realtime_loop_period_s):.3f}s sim_dt={CARLA_FIXED_DELTA_SECONDS:.3f}s factor={float(realtime_pacing_factor):.3f})"
        )

    image_width_px = int(camera_cfg.get("image_size_x", 960))
    image_height_px = int(camera_cfg.get("image_size_y", 540))
    camera_fov_deg = float(camera_cfg.get("fov", 90.0))
    camera_blueprint = _camera_blueprint(
        world=world,
        width_px=image_width_px,
        height_px=image_height_px,
        fov_deg=float(camera_fov_deg),
    )
    topdown_calibration_matrix = _camera_calibration_matrix(
        width_px=image_width_px,
        height_px=image_height_px,
        fov_deg=float(camera_fov_deg),
    )

    topdown_camera = None
    topdown_queue = None
    chase_camera = None
    chase_queue = None
    display = None

    if bool(camera_cfg.get("enabled", True)):
        pygame.init()
        pygame.font.init()
        display = pygame.display.set_mode((int(image_width_px * 2), int(image_height_px)))
        pygame.display.set_caption(f"CARLA {scenario_cfg.get('name', 'scenario')} - Topdown | Chase")

        topdown_cfg = dict(camera_cfg.get("topdown", {}))
        chase_cfg = dict(camera_cfg.get("chase", {}))
        topdown_world_fixed = bool(topdown_cfg.get("world_fixed", False))
        if topdown_world_fixed:
            topdown_transform = _world_fixed_topdown_transform(
                carla=carla,
                focus_points_xy=topdown_focus_points,
                image_width_px=int(image_width_px),
                image_height_px=int(image_height_px),
                fov_deg=float(camera_fov_deg),
                min_height_m=float(topdown_cfg.get("height", 65.0)),
                padding_m=float(topdown_cfg.get("padding_m", 20.0)),
            )
        else:
            topdown_transform = carla.Transform(
                carla.Location(x=0.0, y=0.0, z=float(topdown_cfg.get("height", 65.0))),
                carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
            )
        chase_transform = carla.Transform(
            carla.Location(
                x=-float(chase_cfg.get("back", 8.0)),
                y=0.0,
                z=float(chase_cfg.get("height", 2.8)),
            ),
            carla.Rotation(
                pitch=float(chase_cfg.get("pitch", -10.0)),
                yaw=0.0,
                roll=0.0,
            ),
        )
        topdown_camera, topdown_queue = _spawn_camera(
            world,
            carla,
            camera_blueprint,
            topdown_transform,
            parent=None if topdown_world_fixed else ego_vehicle,
        )
        chase_camera, chase_queue = _spawn_camera(
            world,
            carla,
            camera_blueprint,
            chase_transform,
            parent=ego_vehicle,
        )
        actors_to_destroy.extend([topdown_camera, chase_camera])
    hud_font = pygame.font.SysFont("monospace", 18) if display is not None else None

    tracker = Tracker(tracker_cfg=tracker_cfg)
    mpc = MPC(mpc_cfg=mpc_cfg, road_cfg=road_cfg)
    obstacle_height_filter_enabled = bool(obstacle_filter_cfg.get("enable_height_filter", True))
    obstacle_vertical_clearance_margin_m = float(
        obstacle_filter_cfg.get("vertical_clearance_margin_m", 1.0)
    )
    default_obstacle_height_m = float(
        obstacle_filter_cfg.get("default_obstacle_height_m", 2.0)
    )
    ego_bbox = getattr(ego_vehicle, "bounding_box", None)
    ego_bbox_extent = getattr(ego_bbox, "extent", None)
    ego_height_m = max(
        0.5,
        float(getattr(ego_bbox_extent, "z", 0.9)) * 2.0,
    )

    final_destination_state = [
        float(aligned_destination_transform.location.x),
        float(aligned_destination_transform.location.y),
        0.0,
        float(math.radians(aligned_destination_transform.rotation.yaw)),
    ]
    initial_lane_context = global_planner.get_local_lane_context(
        x_m=float(aligned_spawn_transform.location.x),
        y_m=float(aligned_spawn_transform.location.y),
        heading_rad=float(math.radians(aligned_spawn_transform.rotation.yaw)),
        z_m=float(aligned_spawn_transform.location.z),
    )
    initial_lane_count = int(initial_lane_context.get("lane_count", 0))
    if initial_lane_count > 0:
        road_cfg["lane_count"] = int(initial_lane_count)
    lane_count = max(1, int(road_cfg.get("lane_count", 1)))
    initial_allowed_lane_ids = _allowed_lane_ids_from_context(
        local_context=initial_lane_context,
        fallback_lane_count=int(lane_count),
        fallback_lane_id=int(initial_lane_context.get("lane_id", 0)),
    )
    selected_lane_id = _clamp_lane_id_to_allowed(
        int(initial_lane_context.get("lane_id", 0)),
        initial_allowed_lane_ids,
    )
    current_applied_behavior = "lane_follow"
    original_max_velocity_mps = float(mpc.constraints.max_velocity_mps)
    current_target_v_mps = float(original_max_velocity_mps)
    lane_scores: Dict[int, float] = {
        int(lane_id): 1.0 for lane_id in initial_allowed_lane_ids
    }
    current_route_optimal_lane_id = _clamp_optional_lane_id_to_allowed(
        getattr(initial_global_route_summary, "optimal_lane_id", 0),
        initial_allowed_lane_ids,
    )
    display_reference_maneuver = normalize_macro_maneuver(
        getattr(initial_global_route_summary, "next_macro_maneuver", "straight")
    )
    traffic_light_debug: Dict[str, object] = {
        "signal_state": "unknown",
        "signal_distance_m": None,
        "signal_found": False,
        "should_stop_now": False,
        "stop_latched": False,
        "stop_decision_active": False,
        "signal_actor_name": "",
    }

    current_acceleration_mps2 = 0.0
    current_steering_rad = 0.0
    active_global_route_points: List[List[float]] = list(initial_route_points)
    current_route_summary = initial_global_route_summary
    temporary_route_summary = initial_global_route_summary
    active_reference_maneuver = normalize_macro_maneuver(
        getattr(initial_global_route_summary, "next_macro_maneuver", "straight")
    )
    temporary_destination_state: List[float] | None = None
    planned_trajectory: List[List[float]] = []
    cached_control_sequence: np.ndarray | None = None
    cached_control_step_idx = 0
    last_world_debug_route_draw_time_s = -float("inf")

    # ---- Rule-based behavior planner -------------------------------- #
    rule_planner_cfg = dict(behavior_runtime_cfg.get("rule_based", {}))
    lane_safety_scorer = LaneSafetyScorer(
        d_safe_m=float(
            _behavior_runtime_value(
                behavior_runtime_cfg,
                rule_planner_cfg,
                key="lane_safety_distance_safe_m",
                legacy_key="d_safe_m",
                default=12.0,
            )
        ),
        ttc_safe_s=float(
            _behavior_runtime_value(
                behavior_runtime_cfg,
                rule_planner_cfg,
                key="lane_safety_ttc_safe_s",
                legacy_key="ttc_safe_s",
                default=3.0,
            )
        ),
        sigmoid_k=float(
            _behavior_runtime_value(
                behavior_runtime_cfg,
                rule_planner_cfg,
                key="lane_safety_ttc_sigmoid_gain",
                legacy_key="sigmoid_k",
                default=2.0,
            )
        ),
        ttc_history_size=int(
            _behavior_runtime_value(
                behavior_runtime_cfg,
                rule_planner_cfg,
                key="lane_safety_ttc_history_length",
                legacy_key="ttc_history_size",
                default=8,
            )
        ),
    )
    rule_planner = RuleBasedBehaviorPlanner(
        hysteresis_delta=float(
            _behavior_runtime_value(
                behavior_runtime_cfg,
                rule_planner_cfg,
                key="decision_hysteresis_delta",
                legacy_key="hysteresis_delta",
                default=0.10,
            )
        ),
        lateral_complete_m=float(
            _behavior_runtime_value(
                behavior_runtime_cfg,
                rule_planner_cfg,
                key="lane_change_completion_lateral_threshold_m",
                legacy_key="lateral_complete_m",
                default=0.75,
            )
        ),
        heading_complete_rad=float(
            _behavior_runtime_value(
                behavior_runtime_cfg,
                rule_planner_cfg,
                key="lane_change_completion_heading_threshold_rad",
                legacy_key="heading_complete_rad",
                default=0.20,
            )
        ),
        lane_change_target_safety_threshold=float(
            _behavior_runtime_value(
                behavior_runtime_cfg,
                rule_planner_cfg,
                key="lane_change_target_safety_threshold",
                legacy_key="intersection_lane_change_safety_threshold",
                default=0.10,
            )
        ),
        optimal_lane_unsafe_threshold=float(
            _behavior_runtime_value(
                behavior_runtime_cfg,
                rule_planner_cfg,
                key="optimal_lane_unsafe_threshold",
                default=0.50,
            )
        ),
        cooperative_message_check_frequency_hz=float(
            _behavior_runtime_value(
                behavior_runtime_cfg,
                rule_planner_cfg,
                key="cooperative_message_check_frequency_hz",
                legacy_key="cooperative_message_check_frequency_hz",
                default=1.0,
            )
        ),
        cp_message_path=str(
            rule_planner_cfg.get(
                "cooperative_message_path",
                behavior_runtime_cfg.get("cooperative_message_path", CP_MESSAGE_PATH),
            )
        ).strip()
        or CP_MESSAGE_PATH,
    )
    moving_obstacle_speed_threshold_mps = float(
        _behavior_runtime_value(
            behavior_runtime_cfg,
            rule_planner_cfg,
            key="intersection_obstacle_moving_speed_threshold_mps",
            legacy_key="intersection_obstacle_moving_speed_threshold_mps",
            default=0.5,
        )
    )
    static_obstacle_replan_lane_safety_threshold = float(
        _behavior_runtime_value(
            behavior_runtime_cfg,
            rule_planner_cfg,
            key="intersection_static_obstacle_replan_lane_safety_threshold",
            legacy_key="intersection_static_obstacle_replan_lane_safety_threshold",
            default=0.5,
        )
    )
    static_obstacle_replan_cooldown_s = float(
        _behavior_runtime_value(
            behavior_runtime_cfg,
            rule_planner_cfg,
            key="intersection_static_obstacle_replan_cooldown_s",
            legacy_key="intersection_static_obstacle_replan_cooldown_s",
            default=1.0,
        )
    )
    traffic_light_stop_cfg = dict(
        rule_planner_cfg.get(
            "traffic_light_stop",
            behavior_runtime_cfg.get("traffic_light_stop", {}),
        )
    )
    traffic_light_stop_enabled = bool(traffic_light_stop_cfg.get("enabled", True))
    traffic_light_stop_search_distance_m = max(
        1.0,
        float(traffic_light_stop_cfg.get("search_distance_m", 100.0)),
    )
    traffic_light_stop_buffer_m = max(
        0.0,
        float(traffic_light_stop_cfg.get("stop_buffer_m", 2.0)),
    )
    last_static_intersection_replan_time_s = -float("inf")
    print("[CARLA SCENARIO] Rule-based behavior planner initialized.")
    scenario_runtime_module = _load_optional_module(
        module_name=str(runtime_cfg.get("module", "")).strip(),
        purpose="scenario runtime",
    )
    scenario_runtime_state = _initialize_scenario_runtime_state(
        module=scenario_runtime_module,
        world=world,
        world_map=world_map,
        carla=carla,
        scenario_cfg=scenario_cfg,
    )

    try:
        obstacle_prefix = str(obstacle_cfg.get("environment_name_prefix", "")).strip()
        cached_static_environment_obstacles: List[dict] = []
        if obstacle_prefix:
            cached_static_environment_obstacles = _collect_environment_obstacle_snapshots(
                world=world,
                map_obj=world_map,
                carla=carla,
                obstacle_prefix=obstacle_prefix,
            )
            print(
                f"[CARLA SCENARIO] Found {len(cached_static_environment_obstacles)} static environment obstacles "
                f"with prefix '{obstacle_prefix}'."
            )
            for snapshot in cached_static_environment_obstacles:
                print(
                    "[CARLA SCENARIO] Obstacle "
                    f"{snapshot['vehicle_id']} at ({snapshot['x']:.3f}, {snapshot['y']:.3f}) "
                    f"size=({snapshot['length_m']:.2f}m, {snapshot['width_m']:.2f}m)"
                )
        next_tick_wall_time_s = time.monotonic()
        # Cached planning state (updated at MPC replan rate, not every tick)
        rolling_target_distance_m = float(local_goal_cfg.get("dynamic_lookahead_min_distance_m", 20.0))
        ego_snapshot: Dict[str, float] = {
            "x": float(aligned_spawn_transform.location.x),
            "y": float(aligned_spawn_transform.location.y),
            "v": 0.0,
            "psi": float(math.radians(aligned_spawn_transform.rotation.yaw)),
        }
        base_destination_state: List[float] = list(final_destination_state) + [0.0]
        target_distance_for_destination_m = float(rolling_target_distance_m)
        local_allowed_lane_ids = list(initial_allowed_lane_ids)
        # Last known good lane values from a non-junction road segment.
        # Inside a junction, canonical_lane_id_for_waypoint() always returns 1
        # for every connector (unique road_ids, no lane siblings).  We freeze
        # these values at junction entry so ego_lane_id, selected_lane_id, and
        # optimal_lane_id stay coherent throughout the intersection.
        _pre_junction_lane_id: int = 0
        _pre_junction_allowed_lane_ids: List[int] = list(initial_allowed_lane_ids)
        _pre_junction_optimal_lane_id: int = 0
        cached_obstacle_contours: List[dict] = []
        cached_hud_temp_lane_prompt = 0
        render_tick_counter = 0
        while True:
            if realtime_loop_period_s > 0.0:
                now_wall_time_s = time.monotonic()
                sleep_duration_s = float(next_tick_wall_time_s) - float(now_wall_time_s)
                if sleep_duration_s > 0.0:
                    time.sleep(float(sleep_duration_s))
                else:
                    next_tick_wall_time_s = float(now_wall_time_s)
            world.tick()
            if realtime_loop_period_s > 0.0:
                next_tick_wall_time_s += float(realtime_loop_period_s)
            for event in pygame.event.get() if display is not None else []:
                if event.type == pygame.QUIT:
                    return 0

            ego_state = _world_state_from_vehicle(ego_vehicle)
            if math.hypot(
                float(ego_state[0]) - float(final_destination_state[0]),
                float(ego_state[1]) - float(final_destination_state[1]),
            ) <= float(mpc.destination_reached_threshold_m):
                ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
                print("Reached final destination.")
                return 0

            sim_time_s = float(world.get_snapshot().timestamp.elapsed_seconds)
            wall_time_s = float(time.perf_counter())

            # Tracker update every tick for accuracy; predictions only at replan
            dynamic_object_snapshots, scenario_runtime_state = _apply_scenario_dynamic_obstacle_filter(
                module=scenario_runtime_module,
                runtime_state=scenario_runtime_state,
                world=world,
                world_map=world_map,
                carla=carla,
                scenario_cfg=scenario_cfg,
                object_snapshots=_collect_vehicle_snapshots(world, ego_vehicle),
                sim_time_s=float(sim_time_s),
                wall_time_s=float(wall_time_s),
            )
            scenario_route_summary, scenario_route_points, scenario_runtime_state = _maybe_apply_scenario_global_route_update(
                module=scenario_runtime_module,
                runtime_state=scenario_runtime_state,
                world=world,
                world_map=world_map,
                carla=carla,
                scenario_cfg=scenario_cfg,
                global_planner=global_planner,
                ego_transform=ego_vehicle.get_transform(),
                goal_location=aligned_destination_transform.location,
                object_snapshots=dynamic_object_snapshots,
                current_route_summary=current_route_summary,
                active_global_route_points=active_global_route_points,
                sim_time_s=float(sim_time_s),
                wall_time_s=float(wall_time_s),
            )
            if scenario_route_summary is not None and scenario_route_points is not None:
                if len(scenario_route_points) > 0:
                    active_global_route_points = [
                        [float(point[0]), float(point[1])]
                        for point in list(scenario_route_points)
                        if isinstance(point, Sequence) and len(point) >= 2
                    ]
                current_route_summary = scenario_route_summary
            tick_traffic_stop_target = None
            tick_traffic_signal_context = None
            if bool(traffic_light_stop_enabled):
                tick_ego_transform = ego_vehicle.get_transform()
                tick_traffic_stop_target = find_stop_target_from_ego(
                    world_map=world_map,
                    carla=carla,
                    ego_transform=tick_ego_transform,
                    global_route_points=active_global_route_points,
                    search_distance_m=float(traffic_light_stop_search_distance_m),
                )
                tick_traffic_signal_context = find_relevant_signal_context(
                    world=world,
                    ego_vehicle=ego_vehicle,
                    ego_transform=tick_ego_transform,
                    stop_target=tick_traffic_stop_target,
                )
            tracker.update(
                obstacle_snapshots=dynamic_object_snapshots,
                timestamp_s=sim_time_s,
                next_signal_context=tick_traffic_signal_context,
                next_stop_target=tick_traffic_stop_target,
            )

            # --- Heavy planning: only at MPC replan rate ---
            if mpc.should_replan(sim_time_s):
                static_object_snapshots = list(cached_static_environment_obstacles)
                predicted_snapshots = _merge_tracker_predictions(
                    object_snapshots=dynamic_object_snapshots,
                    predictions=tracker.predict(step_dt_s=float(mpc.dt_s), horizon_s=float(mpc.horizon_s)),
                )
                predicted_snapshots.extend(list(static_object_snapshots))
                ego_transform = ego_vehicle.get_transform()
                ego_z_m = float(ego_transform.location.z)
                # Determine junction status early — before the lane-context
                # call — so we can freeze lane IDs while traversing an
                # intersection (junction connectors have unique road_ids with
                # no lane siblings, so canonical IDs would collapse to 1).
                ego_waypoint = world_map.get_waypoint(
                    ego_transform.location,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
                ego_in_junction = bool(getattr(ego_waypoint, "is_junction", False))
                if obstacle_height_filter_enabled:
                    predicted_snapshots = _filter_obstacle_snapshots_by_vertical_overlap(
                        ego_z_m=float(ego_z_m),
                        ego_height_m=float(ego_height_m),
                        object_snapshots=predicted_snapshots,
                        vertical_clearance_margin_m=float(obstacle_vertical_clearance_margin_m),
                        default_obstacle_height_m=float(default_obstacle_height_m),
                    )

                # Cheap O(N) lookup on stored initial route (no A* replan)
                current_route_summary = global_planner.get_current_route_info(
                    x_m=float(ego_state[0]),
                    y_m=float(ego_state[1]),
                    query_key="ego",
                )

                current_lane_context = global_planner.get_local_lane_context(
                    x_m=float(ego_state[0]),
                    y_m=float(ego_state[1]),
                    heading_rad=float(ego_state[3]),
                    z_m=float(ego_z_m),
                )
                if ego_in_junction and int(_pre_junction_lane_id) != 0:
                    # Junction connectors have unique road_ids with no lane
                    # siblings → canonical_lane_id_for_waypoint() returns 1
                    # for every connector regardless of the approach lane.
                    # Use the last known pre-junction values so ego_lane_id,
                    # selected_lane_id, and optimal_lane_id stay coherent
                    # throughout the intersection and lane-change completion
                    # can be checked correctly on exit.
                    current_lane_id = int(_pre_junction_lane_id)
                    local_allowed_lane_ids = list(_pre_junction_allowed_lane_ids)
                else:
                    current_lane_id = int(current_lane_context.get("lane_id", selected_lane_id))
                    local_allowed_lane_ids = _allowed_lane_ids_from_context(
                        local_context=current_lane_context,
                        fallback_lane_count=int(lane_count),
                        fallback_lane_id=int(selected_lane_id),
                    )
                    if int(current_lane_id) == 0:
                        current_lane_id = _clamp_lane_id_to_allowed(selected_lane_id, local_allowed_lane_ids)
                    # Persist pre-junction values while on a normal road segment.
                    if int(current_lane_id) != 0:
                        _pre_junction_lane_id = int(current_lane_id)
                    if len(local_allowed_lane_ids) > 0:
                        _pre_junction_allowed_lane_ids = list(local_allowed_lane_ids)
                local_lane_count = len(local_allowed_lane_ids)
                selected_lane_id = _clamp_lane_id_to_allowed(selected_lane_id, local_allowed_lane_ids)

                # Use selected_lane_id (target) for lookahead distance so that
                # during a lane change the lookahead is computed for the
                # destination lane, not the lane the ego is currently straddling.
                base_target_lane_id = (
                    int(selected_lane_id)
                    if int(selected_lane_id) != 0
                    else int(current_lane_id)
                )
                base_lookahead_distance_raw_m = compute_lane_lookahead_distance(
                    ego_state=ego_state,
                    lane_center_waypoints=lane_center_waypoints,
                    target_lane_id=int(base_target_lane_id),
                    local_goal_cfg=local_goal_cfg,
                )
                rolling_target_distance_m = float(
                    max(
                        1.0,
                        round(
                            float(
                                base_lookahead_distance_raw_m
                                if base_lookahead_distance_raw_m is not None
                                else local_goal_cfg.get("dynamic_lookahead_min_distance_m", 20.0)
                            )
                        ),
                    )
                )
                ego_snapshot = {
                    "x": float(ego_state[0]),
                    "y": float(ego_state[1]),
                    "v": float(ego_state[2]),
                    "psi": float(ego_state[3]),
                }
                # ---- Assign obstacles to lanes (for safety thread) ---- #
                lane_assignments: Dict[str, int] = {}
                lane_safety_obstacle_snapshots: List[dict] = []
                for obs in predicted_snapshots:
                    obs_id = str(obs.get("vehicle_id", ""))
                    obs_ctx = global_planner.get_local_lane_context(
                        x_m=float(obs.get("x", 0.0)),
                        y_m=float(obs.get("y", 0.0)),
                        heading_rad=float(obs.get("psi", 0.0)),
                        z_m=float(obs.get("z", 0.0)) if obs.get("z", None) is not None else None,
                    )
                    if _same_lane_safety_corridor(current_lane_context, obs_ctx):
                        lane_assignments[obs_id] = int(obs_ctx.get("lane_id", 0))
                        lane_safety_obstacle_snapshots.append(dict(obs))
                    else:
                        lane_assignments[obs_id] = 0

                raw_lane_scores = lane_safety_scorer.compute_lane_scores(
                    ego_snapshot=ego_snapshot,
                    obstacle_snapshots=lane_safety_obstacle_snapshots,
                    lane_assignments=lane_assignments,
                    ego_lane_id=int(current_lane_id),
                    available_lane_ids=local_allowed_lane_ids,
                    timestamp_s=sim_time_s,
                )
                lane_safety_scorer.cleanup_stale_obstacles(
                    {
                        str(snapshot.get("vehicle_id", ""))
                        for snapshot in lane_safety_obstacle_snapshots
                    }
                )
                lane_scores = {
                    int(lane_id): float(raw_lane_scores.get(lane_id, 1.0))
                    for lane_id in local_allowed_lane_ids
                }
                nearest_front_obstacles_by_lane = _nearest_front_obstacle_by_lane(
                    ego_snapshot=ego_snapshot,
                    obstacle_snapshots=lane_safety_obstacle_snapshots,
                    lane_assignments=lane_assignments,
                    available_lane_ids=local_allowed_lane_ids,
                )
                front_obstacle_distance_by_lane = _nearest_front_obstacle_distance_by_lane(
                    ego_snapshot=ego_snapshot,
                    obstacle_snapshots=lane_safety_obstacle_snapshots,
                    lane_assignments=lane_assignments,
                    available_lane_ids=local_allowed_lane_ids,
                )

                # ---- Ego lane offset for lane-change completion ------ #
                # ego_waypoint and ego_in_junction were already computed above
                # (before the lane-context call) so they are not repeated here.
                ego_offset_info = compute_ego_lane_offset(world_map, carla, ego_transform)
                tracked_signal_context = tracker.get_next_signal_context()
                traffic_signal_context = dict(tracked_signal_context or {})
                tracked_stop_target = traffic_signal_context.get("stop_target", None)
                traffic_stop_target = (
                    dict(tracked_stop_target)
                    if isinstance(tracked_stop_target, Mapping)
                    else None
                )
                traffic_signal_state = str(
                    traffic_signal_context.get("signal_state", "unknown")
                )

                # ---- Blue-dot route context and mode ----------------- #
                _prev_dest_xy = (
                    (float(temporary_destination_state[0]), float(temporary_destination_state[1]))
                    if temporary_destination_state is not None
                    and len(temporary_destination_state) >= 2
                    else None
                )
                _prev_mode = (
                    float(temporary_destination_state[5])
                    if temporary_destination_state is not None
                    and len(temporary_destination_state) >= 6
                    else None
                )
                _prev_road_id = (
                    int(temporary_destination_state[6])
                    if temporary_destination_state is not None
                    and len(temporary_destination_state) >= 7
                    else None
                )
                _prev_entered_intersection = (
                    float(temporary_destination_state[7]) > 0.5
                    if temporary_destination_state is not None
                    and len(temporary_destination_state) >= 8
                    else False
                )
                planning_temporary_route_summary = current_route_summary
                if _prev_dest_xy is not None:
                    planning_temporary_route_summary = global_planner.get_current_route_info(
                        x_m=float(_prev_dest_xy[0]),
                        y_m=float(_prev_dest_xy[1]),
                        query_key="blue_dot",
                    )
                planning_next_maneuver = normalize_macro_maneuver(
                    getattr(planning_temporary_route_summary, "next_macro_maneuver", "straight")
                )
                _raw_planning_optimal = _clamp_optional_lane_id_to_allowed(
                    getattr(planning_temporary_route_summary, "optimal_lane_id", 0),
                    local_allowed_lane_ids,
                )
                if ego_in_junction and int(_pre_junction_optimal_lane_id) != 0:
                    # Freeze the route-optimal lane at the pre-junction value.
                    # Inside a junction, route waypoints are connector segments
                    # with unreliable lane IDs; using the preparatory approach
                    # lane keeps the behavior planner stable through the turn.
                    planning_optimal_lane_id = _clamp_optional_lane_id_to_allowed(
                        int(_pre_junction_optimal_lane_id), local_allowed_lane_ids
                    )
                else:
                    planning_optimal_lane_id = int(_raw_planning_optimal)
                    if int(planning_optimal_lane_id) != 0:
                        _pre_junction_optimal_lane_id = int(planning_optimal_lane_id)
                raw_temp_mode_value, _temp_mode_road_id, _temp_mode_entered_intersection = compute_temp_destination_mode(
                    world_map=world_map,
                    carla=carla,
                    ego_transform=ego_transform,
                    mode_reference_xy=_prev_dest_xy,
                    prev_mode=_prev_mode,
                    prev_road_id=_prev_road_id,
                    prev_entered_intersection=bool(_prev_entered_intersection),
                    next_macro_maneuver=str(planning_next_maneuver),
                    intersection_threshold_m=float(
                        _behavior_runtime_value(
                            behavior_runtime_cfg,
                            rule_planner_cfg,
                            key="intersection_distance_threshold_m",
                            legacy_key="intersection_threshold_m",
                            default=30.0,
                        )
                    ),
                )
                temp_mode_str = (
                    "INTERSECTION" if float(raw_temp_mode_value) > 0.5 else "NORMAL"
                )
                intersection_front_obstacle = nearest_front_obstacles_by_lane.get(
                    int(current_lane_id),
                    None,
                )
                current_lane_safety = float(
                    lane_scores.get(int(current_lane_id), 1.0)
                )
                intersection_obstacle_response = evaluate_intersection_obstacle_response(
                    mode=str(temp_mode_str),
                    front_obstacle_speed_mps=(
                        None
                        if intersection_front_obstacle is None
                        else float(intersection_front_obstacle.get("v", 0.0))
                    ),
                    original_max_velocity_mps=float(original_max_velocity_mps),
                    moving_obstacle_speed_threshold_mps=float(
                        moving_obstacle_speed_threshold_mps
                    ),
                    route_lane_safety_score=float(current_lane_safety),
                    static_obstacle_replan_lane_safety_threshold=float(
                        static_obstacle_replan_lane_safety_threshold
                    ),
                )
                behavior_target_max_velocity_mps = float(
                    intersection_obstacle_response.get(
                        "speed_cap_mps",
                        float(original_max_velocity_mps),
                    )
                )
                current_target_v_mps = float(behavior_target_max_velocity_mps)

                # ---- Rule-based behavior planner (synchronous) ------- #
                planner_output = rule_planner.update(
                    lane_safety_scores=lane_scores,
                    ego_lane_id=int(current_lane_id),
                    selected_lane_id=int(selected_lane_id),
                    ego_lateral_offset_m=float(ego_offset_info.get("lateral_offset_m", 0.0)),
                    ego_heading_error_rad=float(ego_offset_info.get("heading_error_rad", 0.0)),
                    mode=str(temp_mode_str),
                    route_optimal_lane_id=int(planning_optimal_lane_id),
                    next_macro_maneuver=str(planning_next_maneuver),
                    front_obstacle_distance_by_lane=front_obstacle_distance_by_lane,
                    current_time_s=float(sim_time_s),
                    traffic_signal_state=str(traffic_signal_state),
                    traffic_stop_target=traffic_stop_target,
                    traffic_signal_context=traffic_signal_context,
                    ego_speed_mps=float(ego_state[2]),
                    ego_max_deceleration_mps2=abs(float(mpc.constraints.min_acceleration_mps2)),
                    ego_in_junction=bool(ego_in_junction),
                )
                current_applied_behavior = normalize_behavior_decision(
                    planner_output.get("decision", "lane_follow")
                )
                planner_mode_override = str(
                    planner_output.get("mode_override", temp_mode_str) or temp_mode_str
                ).strip().upper()
                if str(planner_mode_override) not in {"NORMAL", "INTERSECTION"}:
                    planner_mode_override = str(temp_mode_str)
                blue_dot_rolling = bool(
                    planner_output.get(
                        "blue_dot_rolling",
                        str(current_applied_behavior) != "stop",
                    )
                )
                traffic_light_debug = dict(
                    planner_output.get("traffic_light_debug", {}) or {}
                )
                signal_distance_value = traffic_light_debug.get("signal_distance_m", None)
                signal_distance_str = (
                    "n/a"
                    if signal_distance_value is None
                    else f"{float(signal_distance_value):.1f}m"
                )
                stop_decision_active = bool(
                    traffic_light_debug.get(
                        "stop_decision_active",
                        traffic_light_debug.get("stop_latched", False),
                    )
                )
                print(
                    "[BEHAVIOR][SIGNAL] "
                    f"found={bool(traffic_light_debug.get('signal_found', False))} "
                    f"name={str(traffic_light_debug.get('signal_actor_name', '')) or '<none>'} "
                    f"state={str(traffic_light_debug.get('signal_state', 'unknown'))} "
                    f"signal_dist={signal_distance_str} "
                    f"should_stop={stop_decision_active} "
                    f"feasible_stop={bool(traffic_light_debug.get('should_stop_now', False))} "
                    f"latched={bool(traffic_light_debug.get('stop_latched', False))}"
                )
                stop_target_state = _stop_target_state_from_behavior_output(
                    planner_output.get("stop_target", None)
                )
                stop_target_distance_m = None
                if stop_target_state is not None:
                    try:
                        stop_target_distance_m = float(
                            planner_output.get("stop_target", {}).get("distance_m", 0.0)
                        )
                    except Exception:
                        stop_target_distance_m = None
                if str(current_applied_behavior) == "reroute":
                    print("[BEHAVIOR] executing reroute request via global planner.")
                    reroute_result = reroute_from_lane_closure_messages(
                        messages=planner_output.get("reroute_messages", []),
                        world_map=world_map,
                        carla=carla,
                        global_planner=global_planner,
                        ego_transform=ego_transform,
                        goal_location=aligned_destination_transform.location,
                        current_route_points=active_global_route_points,
                    )
                    rerouted_route_summary = reroute_result.get("route_summary", None)
                    rerouted_route_points = list(reroute_result.get("route_points", []))
                    if rerouted_route_summary is not None and len(rerouted_route_points) > 0:
                        print(
                            "[BEHAVIOR] reroute completed with a new global route "
                            f"({len(rerouted_route_points)} route points)."
                        )
                        active_global_route_points = [
                            [float(point[0]), float(point[1])]
                            for point in rerouted_route_points
                            if isinstance(point, Sequence) and len(point) >= 2
                        ]
                        last_world_debug_route_draw_time_s = -float("inf")
                        current_route_summary = global_planner.get_current_route_info(
                            x_m=float(ego_state[0]),
                            y_m=float(ego_state[1]),
                            query_key="ego",
                        )
                        planning_temporary_route_summary = current_route_summary
                        _prev_dest_xy = None
                        _prev_mode = None
                        _prev_road_id = None
                        _prev_entered_intersection = False
                        planning_next_maneuver = normalize_macro_maneuver(
                            getattr(planning_temporary_route_summary, "next_macro_maneuver", "straight")
                        )
                        planning_optimal_lane_id = _clamp_optional_lane_id_to_allowed(
                            getattr(planning_temporary_route_summary, "optimal_lane_id", 0),
                            local_allowed_lane_ids,
                        )
                    else:
                        print("[BEHAVIOR] reroute requested, but global planner did not return a new route.")
                motion_behavior_decision = (
                    "lane_follow" if str(current_applied_behavior) == "reroute" else str(current_applied_behavior)
                )
                selected_lane_id = _clamp_lane_id_to_allowed(
                    int(
                        planner_output.get(
                            "selected_lane_id",
                            planner_output.get("target_lane_id", current_lane_id),
                        )
                    ),
                    local_allowed_lane_ids,
                )
                active_planning_maneuver = intersection_route_follow_maneuver(
                    mode=str(planner_mode_override),
                    next_macro_maneuver=str(planning_next_maneuver),
                    decision=str(motion_behavior_decision),
                    target_lane_id=int(selected_lane_id),
                    available_lane_ids=local_allowed_lane_ids,
                    current_road_option=str(
                        getattr(planning_temporary_route_summary, "current_road_option", "")
                    ),
                )

                # ---- Temporary destination via CARLA waypoints ------- #
                temp_destination_decision = (
                    str(current_applied_behavior)
                    if str(current_applied_behavior) == "reroute"
                    else str(motion_behavior_decision)
                )
                should_follow_global_route_lane = bool(
                    str(temp_destination_decision) == "reroute"
                    or (
                        str(temp_destination_decision) == "lane_follow"
                        and int(selected_lane_id) != 0
                        and int(planning_optimal_lane_id) != 0
                        and int(selected_lane_id) == int(planning_optimal_lane_id)
                    )
                )
                temporary_destination_state = compute_temp_destination(
                    world_map=world_map,
                    carla=carla,
                    ego_transform=ego_transform,
                    target_lane_id=int(selected_lane_id),
                    decision=str(temp_destination_decision),
                    lookahead_m=float(rolling_target_distance_m),
                    target_v_mps=float(current_target_v_mps),
                    global_route_points=active_global_route_points,
                    mode_reference_xy=_prev_dest_xy,
                    prev_mode=_prev_mode,
                    prev_road_id=_prev_road_id,
                    prev_entered_intersection=bool(_prev_entered_intersection),
                    next_macro_maneuver=str(active_planning_maneuver),
                    mode_override=str(planner_mode_override),
                    stop_target_state=stop_target_state,
                    follow_global_route_lane=bool(should_follow_global_route_lane),
                )
                if str(current_applied_behavior) == "stop" and stop_target_state is not None:
                    temporary_destination_state, active_plan_max_velocity_mps = _apply_stop_target_speed_cap(
                        temporary_destination_state=temporary_destination_state,
                        ego_state=ego_state,
                        stop_target_distance_m=stop_target_distance_m,
                        original_max_velocity_mps=float(behavior_target_max_velocity_mps),
                        braking_deceleration_mps2=abs(float(mpc.constraints.min_acceleration_mps2)),
                        stop_buffer_m=float(traffic_light_stop_buffer_m),
                    )
                else:
                    active_plan_max_velocity_mps = float(behavior_target_max_velocity_mps)
                temporary_destination_state, active_plan_max_velocity_mps = _apply_final_destination_snap(
                    temporary_destination_state=temporary_destination_state,
                    final_destination_state=final_destination_state,
                    ego_state=ego_state,
                    lock_to_final_distance_m=float(
                        local_goal_cfg.get("lock_to_final_distance_m", 5.0)
                    ),
                    original_max_velocity_mps=float(active_plan_max_velocity_mps),
                )

                temporary_route_summary = planning_temporary_route_summary
                if temporary_destination_state is not None and len(temporary_destination_state) >= 2:
                    temporary_route_summary = global_planner.get_current_route_info(
                        x_m=float(temporary_destination_state[0]),
                        y_m=float(temporary_destination_state[1]),
                        query_key="blue_dot",
                    )
                current_temp_mode_str = (
                    "INTERSECTION"
                    if (
                        temporary_destination_state is not None
                        and len(temporary_destination_state) >= 6
                        and float(temporary_destination_state[5]) > 0.5
                    )
                    else "NORMAL"
                )
                current_temp_reference_xy = (
                    (float(temporary_destination_state[0]), float(temporary_destination_state[1]))
                    if temporary_destination_state is not None
                    and len(temporary_destination_state) >= 2
                    else _prev_dest_xy
                )
                current_temp_mode_value = (
                    float(temporary_destination_state[5])
                    if temporary_destination_state is not None
                    and len(temporary_destination_state) >= 6
                    else raw_temp_mode_value
                )
                current_temp_road_id = (
                    int(temporary_destination_state[6])
                    if temporary_destination_state is not None
                    and len(temporary_destination_state) >= 7
                    else _temp_mode_road_id
                )
                current_temp_entered_intersection = (
                    float(temporary_destination_state[7]) > 0.5
                    if temporary_destination_state is not None
                    and len(temporary_destination_state) >= 8
                    else bool(_temp_mode_entered_intersection)
                )
                reference_next_maneuver = normalize_macro_maneuver(
                    getattr(temporary_route_summary, "next_macro_maneuver", planning_next_maneuver)
                )
                if ego_in_junction and int(_pre_junction_optimal_lane_id) != 0:
                    current_route_optimal_lane_id = _clamp_optional_lane_id_to_allowed(
                        int(_pre_junction_optimal_lane_id), local_allowed_lane_ids
                    )
                else:
                    current_route_optimal_lane_id = _clamp_optional_lane_id_to_allowed(
                        getattr(temporary_route_summary, "optimal_lane_id", 0),
                        local_allowed_lane_ids,
                    )
                display_reference_maneuver = str(reference_next_maneuver)
                active_reference_maneuver = intersection_route_follow_maneuver(
                    mode=str(current_temp_mode_str),
                    next_macro_maneuver=str(reference_next_maneuver),
                    decision=str(current_applied_behavior),
                    target_lane_id=int(selected_lane_id),
                    available_lane_ids=local_allowed_lane_ids,
                    current_road_option=str(
                        getattr(temporary_route_summary, "current_road_option", "")
                    ),
                )
                should_follow_global_route_lane_for_reference = bool(
                    str(current_applied_behavior) == "reroute"
                    or (
                        str(current_applied_behavior) == "lane_follow"
                        and int(selected_lane_id) != 0
                        and int(current_route_optimal_lane_id) != 0
                        and int(selected_lane_id) == int(current_route_optimal_lane_id)
                    )
                )

                # ---- MPC solve to the blue dot with path shaping ----- #
                mpc.constraints.max_velocity_mps = float(active_plan_max_velocity_mps)
                lane_reference_speed_mps = max(
                    1.0,
                    float(ego_state[2]),
                    abs(float(temporary_destination_state[2])) if len(temporary_destination_state) >= 3 else 0.0,
                )
                local_lane_center_reference = build_reference_samples(
                    world_map=world_map,
                    carla=carla,
                    ego_transform=ego_transform,
                    target_lane_id=int(selected_lane_id),
                    decision=str(current_applied_behavior),
                    horizon_steps=int(mpc.horizon_steps),
                    step_distance_m=max(0.5, float(lane_reference_speed_mps) * float(mpc.dt_s)),
                    global_route_points=active_global_route_points,
                    mode_reference_xy=current_temp_reference_xy,
                    prev_mode=current_temp_mode_value,
                    prev_road_id=current_temp_road_id,
                    prev_entered_intersection=bool(current_temp_entered_intersection),
                    next_macro_maneuver=str(active_reference_maneuver),
                    mode_override=str(current_temp_mode_str),
                    stop_target_state=stop_target_state,
                    follow_global_route_lane=bool(should_follow_global_route_lane_for_reference),
                )
                new_planned_trajectory = mpc.plan_trajectory(
                    current_state=ego_state,
                    destination_state=temporary_destination_state,
                    object_snapshots=predicted_snapshots,
                    current_acceleration_mps2=float(current_acceleration_mps2),
                    current_steering_rad=float(current_steering_rad),
                    lane_center_waypoints=[],
                    lane_center_reference_samples=local_lane_center_reference,
                    stop_goal_active=bool(str(current_applied_behavior) == "stop"),
                )
                new_control_sequence = getattr(mpc, "_last_u_solution", None)
                mpc.mark_replanned(sim_time_s)

                if new_planned_trajectory:
                    planned_trajectory = list(new_planned_trajectory)
                if new_control_sequence is not None and len(new_control_sequence) > 0:
                    cached_control_sequence = np.asarray(new_control_sequence, dtype=float)
                    cached_control_step_idx = 0

                # Cache obstacle contours and HUD lane for rendering
                cached_obstacle_contours = _build_obstacle_field_contours(
                    mpc=mpc,
                    ego_state=ego_state,
                    object_snapshots=predicted_snapshots,
                )
                if len(temporary_destination_state) >= 5:
                    cached_hud_temp_lane_prompt = int(temporary_destination_state[4])
            # --- End heavy planning block ---

            if not planned_trajectory:
                ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
            elif cached_control_sequence is None or len(cached_control_sequence) == 0:
                ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.3, steer=0.0))
            else:
                control_step_idx = min(int(cached_control_step_idx), int(len(cached_control_sequence) - 1))
                current_acceleration_mps2 = float(cached_control_sequence[control_step_idx][0])
                current_steering_rad = float(cached_control_sequence[control_step_idx][1])
                ego_vehicle.apply_control(
                    _control_from_mpc(
                        mpc=mpc,
                        carla=carla,
                        acceleration_mps2=float(current_acceleration_mps2),
                        steering_angle_rad=float(current_steering_rad),
                    )
                )
                if cached_control_step_idx < int(len(cached_control_sequence) - 1):
                    cached_control_step_idx += 1

            if display is not None and topdown_queue is not None and chase_queue is not None:
                render_tick_counter += 1
                topdown_image = None
                chase_image = None
                try:
                    topdown_image = topdown_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    chase_image = chase_queue.get_nowait()
                except queue.Empty:
                    pass
                if topdown_image is None and chase_image is None:
                    continue

                # ---- HUD text (rule-based planner info) -------------- #
                # Safety scores as compact string
                safety_str = " ".join(
                    f"L{int(lid)}={float(lane_scores.get(int(lid), 1.0)):.2f}"
                    for lid in list(local_allowed_lane_ids)
                )
                allowed_lanes_str = ",".join(str(int(lane_id)) for lane_id in list(local_allowed_lane_ids))
                # Mode from temp destination (6th element)
                td_mode_str = "INTERSECTION" if (len(temporary_destination_state) >= 6 and float(temporary_destination_state[5]) > 0.5) else "NORMAL"

                terminal_planned_velocity_mps = float("nan")
                if planned_trajectory and len(planned_trajectory[-1]) >= 3:
                    terminal_planned_velocity_mps = float(planned_trajectory[-1][2])
                hud_signal_distance_value = traffic_light_debug.get("signal_distance_m", None)
                hud_signal_distance_str = (
                    "n/a"
                    if hud_signal_distance_value is None
                    else f"{float(hud_signal_distance_value):.1f}m"
                )
                hud_stop_decision = bool(
                    traffic_light_debug.get(
                        "stop_decision_active",
                        traffic_light_debug.get("stop_latched", False),
                    )
                )
                hud_lines = [
                    f"v={float(ego_state[2]):.2f}  v_ref={float(temporary_destination_state[2]):.2f}  v_max={float(mpc.constraints.max_velocity_mps):.2f} m/s",
                    f"ego_lane={int(current_lane_id)}  sel_lane={int(selected_lane_id)}  blue_lane={int(cached_hud_temp_lane_prompt)}",
                    f"lanes=[{allowed_lanes_str}]",
                    f"safety: {safety_str}",
                    f"mode={td_mode_str}  maneuver={str(display_reference_maneuver)}",
                    f"route_opt_lane={int(current_route_optimal_lane_id)}",
                    f"decision={str(current_applied_behavior)}  lc_state={rule_planner.lc_state}",
                    f"signal={str(traffic_light_debug.get('signal_state', 'unknown'))}  signal_dist={hud_signal_distance_str}  stop={hud_stop_decision}",
                    f"traj_v_end={float(terminal_planned_velocity_mps):.2f}  lookahead={int(round(float(rolling_target_distance_m)))}m",
                ]
                topdown_overlay = None
                if topdown_camera is not None:
                    topdown_camera_transform = (
                        getattr(topdown_image, "transform", None)
                        if topdown_image is not None
                        else None
                    )
                    if topdown_camera_transform is None:
                        topdown_camera_transform = topdown_camera.get_transform()
                    topdown_overlay = {
                        "camera_transform": topdown_camera_transform,
                        "calibration_matrix": topdown_calibration_matrix,
                        "image_width_px": int(image_width_px),
                        "image_height_px": int(image_height_px),
                        "overlay_z_m": float(ego_vehicle.get_location().z),
                        "global_route_points": _route_points_for_visualization(
                            active_global_route_points,
                            enabled=bool(global_route_visualization_enabled),
                        ),
                        "temporary_destination_state": list(temporary_destination_state),
                        "planned_trajectory_states": list(planned_trajectory or []),
                        "obstacle_field_contours": cached_obstacle_contours,
                    }
                _render_camera_pair(
                    display,
                    topdown_image,
                    chase_image,
                    topdown_overlay=topdown_overlay,
                    hud_lines=hud_lines,
                    hud_font=hud_font,
                )
    finally:
        if traffic_manager is not None:
            try:
                traffic_manager.set_synchronous_mode(False)
            except Exception:
                pass
        world.apply_settings(previous_settings)
        _destroy_actors(actors_to_destroy)
        if pygame is not None:
            pygame.quit()

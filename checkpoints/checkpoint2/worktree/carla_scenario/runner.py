"""
Shared CARLA scenario runner.
"""

from __future__ import annotations

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
    compute_temporary_destination_state,
)
from behavior_planner import (
    BehaviorPlannerAPIClient,
    BehaviorPlannerDecision,
    BehaviorPlannerPromptBuilder,
    apply_behavior_planner_decision,
    parse_behavior_planner_response,
)
from utility import AStarGlobalPlanner, Tracker, build_lane_center_waypoints, load_yaml_file

try:
    import pygame
except ImportError:  # pragma: no cover
    pygame = None  # type: ignore[assignment]


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MPC_CONFIG_PATH = os.path.join(PROJECT_ROOT, "MPC", "mpc.yaml")
TRACKER_CONFIG_PATH = os.path.join(PROJECT_ROOT, "utility", "tracker.yaml")
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
) -> list[int]:
    if int(fallback_lane_count) > 0:
        return list(range(1, max(1, int(fallback_lane_count)) + 1))

    local_context = dict(local_context or {})
    lane_ids = sorted(
        {
            int(lane_id)
            for lane_id in list(local_context.get("lane_ids", []))
            if int(lane_id) > 0
        }
    )
    if len(lane_ids) > 0:
        return lane_ids
    return list(range(1, max(1, int(fallback_lane_count)) + 1))


def _clamp_lane_id_to_allowed(raw_lane_id: int | None, allowed_lane_ids: Sequence[int]) -> int:
    if len(allowed_lane_ids) == 0:
        return max(1, int(raw_lane_id or 1))
    lane_id = int(raw_lane_id if raw_lane_id is not None else allowed_lane_ids[0])
    if lane_id in allowed_lane_ids:
        return int(lane_id)
    return int(
        min(
            allowed_lane_ids,
            key=lambda candidate_lane_id: abs(int(candidate_lane_id) - int(lane_id)),
        )
    )


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


def _draw_planning_overlay(
    *,
    surface,
    camera_transform,
    calibration_matrix: np.ndarray,
    image_width_px: int,
    image_height_px: int,
    overlay_z_m: float,
    temporary_destination_state: Sequence[float] | None,
    planned_trajectory_states: Sequence[Sequence[float]] | None,
    obstacle_field_contours: Sequence[Mapping[str, object]] | None,
) -> None:
    if pygame is None:
        return

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

def _nearest_progress_along_route(route_points: Sequence[Sequence[float]], xy: Sequence[float]) -> tuple[float, float]:
    if len(route_points) <= 1:
        return 0.0, 0.0

    total_progress_m = 0.0
    best_progress_m = 0.0
    best_distance_m = float("inf")
    px_m = float(xy[0])
    py_m = float(xy[1])

    for idx in range(len(route_points) - 1):
        x0_m, y0_m = float(route_points[idx][0]), float(route_points[idx][1])
        x1_m, y1_m = float(route_points[idx + 1][0]), float(route_points[idx + 1][1])
        dx_m = x1_m - x0_m
        dy_m = y1_m - y0_m
        seg_len_sq = dx_m * dx_m + dy_m * dy_m
        if seg_len_sq <= 1e-9:
            continue
        proj = ((px_m - x0_m) * dx_m + (py_m - y0_m) * dy_m) / seg_len_sq
        proj = min(1.0, max(0.0, proj))
        cx_m = x0_m + proj * dx_m
        cy_m = y0_m + proj * dy_m
        distance_m = math.hypot(px_m - cx_m, py_m - cy_m)
        if distance_m < best_distance_m:
            best_distance_m = distance_m
            best_progress_m = total_progress_m + proj * math.sqrt(seg_len_sq)
        total_progress_m += math.sqrt(seg_len_sq)
    return best_progress_m, total_progress_m


def _sample_route_at_progress(route_points: Sequence[Sequence[float]], progress_m: float) -> List[float]:
    if len(route_points) == 0:
        return [0.0, 0.0, 0.0]
    if len(route_points) == 1:
        return [float(route_points[0][0]), float(route_points[0][1]), 0.0]

    remaining_m = max(0.0, float(progress_m))
    for idx in range(len(route_points) - 1):
        x0_m, y0_m = float(route_points[idx][0]), float(route_points[idx][1])
        x1_m, y1_m = float(route_points[idx + 1][0]), float(route_points[idx + 1][1])
        segment_length_m = math.hypot(x1_m - x0_m, y1_m - y0_m)
        if segment_length_m <= 1e-9:
            continue
        if remaining_m <= segment_length_m:
            alpha = remaining_m / segment_length_m
            heading_rad = math.atan2(y1_m - y0_m, x1_m - x0_m)
            return [
                float(x0_m + alpha * (x1_m - x0_m)),
                float(y0_m + alpha * (y1_m - y0_m)),
                float(heading_rad),
            ]
        remaining_m -= segment_length_m

    last_idx = len(route_points) - 1
    prev_idx = max(0, last_idx - 1)
    heading_rad = math.atan2(
        float(route_points[last_idx][1]) - float(route_points[prev_idx][1]),
        float(route_points[last_idx][0]) - float(route_points[prev_idx][0]),
    )
    return [float(route_points[last_idx][0]), float(route_points[last_idx][1]), float(heading_rad)]


def _triangle_area(p1: Sequence[float], p2: Sequence[float], p3: Sequence[float]) -> float:
    return abs(
        0.5
        * (
            float(p1[0]) * (float(p2[1]) - float(p3[1]))
            + float(p2[0]) * (float(p3[1]) - float(p1[1]))
            + float(p3[0]) * (float(p1[1]) - float(p2[1]))
        )
    )


def _route_curvature(route_points: Sequence[Sequence[float]], progress_m: float, spacing_m: float) -> float:
    p1 = _sample_route_at_progress(route_points, progress_m + float(spacing_m))
    p2 = _sample_route_at_progress(route_points, progress_m + 2.0 * float(spacing_m))
    p3 = _sample_route_at_progress(route_points, progress_m + 3.0 * float(spacing_m))
    a = math.hypot(float(p2[0]) - float(p1[0]), float(p2[1]) - float(p1[1]))
    b = math.hypot(float(p3[0]) - float(p2[0]), float(p3[1]) - float(p2[1]))
    c = math.hypot(float(p3[0]) - float(p1[0]), float(p3[1]) - float(p1[1]))
    if min(a, b, c) <= 1e-6:
        return 0.0
    area = _triangle_area(p1, p2, p3)
    return float((4.0 * area) / (a * b * c))


def _lookahead_distance_m(speed_mps: float, route_points: Sequence[Sequence[float]], progress_m: float, local_goal_cfg: Mapping[str, object]) -> float:
    d_min = float(local_goal_cfg.get("dynamic_lookahead_min_distance_m", 20.0))
    if not bool(local_goal_cfg.get("dynamic_lookahead_enabled", True)):
        return float(max(d_min, d_min))

    d_max = float(local_goal_cfg.get("dynamic_lookahead_max_distance_m", 100.0))
    k_v = float(local_goal_cfg.get("dynamic_lookahead_speed_gain", 3.0))
    k_c = float(local_goal_cfg.get("dynamic_lookahead_curvature_gain", 20.0))
    spacing_m = float(local_goal_cfg.get("dynamic_lookahead_curvature_sample_spacing_m", 10.0))
    curvature = _route_curvature(
        route_points=route_points,
        progress_m=float(progress_m),
        spacing_m=max(1.0, float(spacing_m)),
    )
    raw_distance_m = float(d_min + k_v * float(speed_mps) - k_c * abs(float(curvature)))
    clamped_distance_m = min(max(float(raw_distance_m), float(d_min)), float(d_max))
    return float(max(float(d_min), float(clamped_distance_m)))


def _temporary_destination_state(
    ego_state: Sequence[float],
    route_points: Sequence[Sequence[float]],
    final_destination_state: Sequence[float],
    mpc_cfg: Mapping[str, object],
    lookahead_distance_m: float | None = None,
) -> List[float]:
    local_goal_cfg = dict(mpc_cfg.get("local_goal", {}))
    progress_m, route_length_m = _nearest_progress_along_route(route_points, ego_state[:2])
    if lookahead_distance_m is None:
        lookahead_m = _lookahead_distance_m(
            speed_mps=float(ego_state[2]),
            route_points=route_points,
            progress_m=float(progress_m),
            local_goal_cfg=local_goal_cfg,
        )
    else:
        lookahead_m = max(1.0, float(lookahead_distance_m))
    target_progress_m = min(float(route_length_m), float(progress_m) + float(lookahead_m))
    target_x_m, target_y_m, target_heading_rad = _sample_route_at_progress(route_points, target_progress_m)

    final_distance_m = math.hypot(
        float(final_destination_state[0]) - float(ego_state[0]),
        float(final_destination_state[1]) - float(ego_state[1]),
    )
    if final_distance_m <= float(local_goal_cfg.get("lock_to_final_distance_m", 30.0)):
        return list(final_destination_state[:4])

    return [
        float(target_x_m),
        float(target_y_m),
        float(local_goal_cfg.get("v_ref_mps", 10.0)),
        float(target_heading_rad),
    ]


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
                "v": float(speed_mps),
                "psi": float(math.radians(transform.rotation.yaw)),
                "length_m": float(actor.bounding_box.extent.x * 2.0),
                "width_m": float(actor.bounding_box.extent.y * 2.0),
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
        bbox = getattr(env_obj, "bounding_box", None)
        if bbox is not None:
            half_length_m = max(0.05, float(getattr(bbox.extent, "x", half_length_m)))
            half_width_m = max(0.05, float(getattr(bbox.extent, "y", half_width_m)))

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


def _latest_route_line(prompt_text: str) -> str:
    for raw_line in str(prompt_text or "").splitlines():
        line = str(raw_line).strip()
        if line.startswith("ROUTE:["):
            return line
    return "ROUTE:[pending]"


def _default_lane_keep_decision(request_id: str = "0") -> BehaviorPlannerDecision:
    return BehaviorPlannerDecision(
        request_id=str(request_id),
        behavior="LANE_KEEP",
        target_v_mps=None,
        reasoning="",
        cav_broadcast={},
        raw_response_text='{"id":"%s","behavior":"LANE_KEEP"}' % str(request_id),
    )


def _scenario_constraints_cfg(scenario_cfg: Mapping[str, object]) -> Dict[str, object]:
    constraints_cfg = dict(scenario_cfg.get("constraints", {}))
    missing_keys = [key for key in REQUIRED_CONSTRAINT_KEYS if key not in constraints_cfg]
    if len(missing_keys) > 0:
        raise RuntimeError(
            "Scenario constraints are incomplete. "
            f"Missing keys in scenario '{scenario_cfg.get('name', '<unknown>')}': {missing_keys}"
        )
    return constraints_cfg


def _behavior_planner_worker(
    *,
    stop_event: threading.Event,
    shared_state: Dict[str, Any],
    shared_lock: threading.Lock,
    prompt_builder: BehaviorPlannerPromptBuilder,
    api_client: BehaviorPlannerAPIClient | None,
    system_instruction: str,
    behavior_runtime_cfg: Mapping[str, object],
    lane_center_waypoints: Sequence[Mapping[str, object]],
    road_cfg: Mapping[str, object],
    mpc_constraints: Mapping[str, object],
    ego_vehicle_id: str = "Ego01",
) -> None:
    frequency_hz = float(max(0.1, behavior_runtime_cfg.get("frequency_hz", 1.0)))
    period_s = 1.0 / frequency_hz
    api_enabled = bool(behavior_runtime_cfg.get("api_enabled", False))
    print_prompt = bool(behavior_runtime_cfg.get("print_prompt", True))
    print_response = bool(behavior_runtime_cfg.get("print_response", True))
    print_system_instruction = bool(behavior_runtime_cfg.get("print_system_instruction", False))

    if print_system_instruction and len(system_instruction.strip()) > 0:
        print("\n[CARLA BEHAVIOR PLANNER SYSTEM]")
        print(system_instruction.strip())

    if api_enabled and api_client is not None:
        try:
            api_client.prime_system_instruction(system_instruction)
        except Exception as exc:  # pragma: no cover - runtime integration path
            print(f"[CARLA BEHAVIOR PLANNER] Failed to prime system instruction: {exc}")

    next_tick_wall_time_s = time.monotonic()
    while not stop_event.is_set():
        with shared_lock:
            latest_inputs = shared_state.get("latest_inputs", None)
            previous_behavior = str(
                shared_state.get(
                    "previous_behavior_for_prompt",
                    shared_state.get("current_applied_behavior", "LANE_KEEP"),
                )
            )
            prompt_seq = int(shared_state.get("prompt_seq", 0)) + 1
            shared_state["prompt_seq"] = prompt_seq

        if latest_inputs is not None:
            prompt_text = prompt_builder.build_prompt(
                ego_snapshot=dict(latest_inputs["ego_snapshot"]),
                destination_state=list(latest_inputs["destination_state"]),
                temporary_destination_state=list(latest_inputs["temporary_destination_state"]),
                lane_center_waypoints=lane_center_waypoints,
                object_snapshots=list(latest_inputs["object_snapshots"]),
                road_cfg=road_cfg,
                behavior_planner_runtime_cfg=dict(behavior_runtime_cfg),
                ego_vehicle_id=ego_vehicle_id,
                mpc_constraints=mpc_constraints,
                prompt_id=prompt_seq,
                previous_behavior=previous_behavior,
            )

            with shared_lock:
                shared_state["latest_prompt"] = prompt_text
                shared_state["latest_prompt_id"] = str(prompt_seq)

            if print_prompt:
                print("\n[CARLA BEHAVIOR PLANNER INPUT]")
                print(prompt_text)

            if api_enabled and api_client is not None:
                start_wall_time_s = time.monotonic()
                try:
                    raw_response_text, _response_id = api_client.request_decision(
                        system_instruction=system_instruction,
                        prompt=prompt_text,
                    )
                    latency_s = time.monotonic() - start_wall_time_s
                    decision = parse_behavior_planner_response(raw_response_text)
                    if str(decision.request_id) != str(prompt_seq):
                        print(
                            "[CARLA BEHAVIOR PLANNER] Ignoring mismatched response id "
                            f"{decision.request_id!r} for prompt {prompt_seq}."
                        )
                    else:
                        with shared_lock:
                            shared_state["latest_decision"] = decision
                            shared_state["previous_behavior_for_prompt"] = str(decision.behavior)
                        if print_response:
                            print("\n[CARLA BEHAVIOR PLANNER RESPONSE]")
                            print(
                                f"[CARLA BEHAVIOR PLANNER RESPONSE] seq={prompt_seq} "
                                f"latency={latency_s:.3f}s"
                            )
                            print(raw_response_text)
                except Exception as exc:  # pragma: no cover - runtime integration path
                    print(f"[CARLA BEHAVIOR PLANNER] API request failed: {exc}")

        next_tick_wall_time_s += period_s
        wait_s = max(0.0, next_tick_wall_time_s - time.monotonic())
        stop_event.wait(wait_s)


def run_loaded_world(client, world, scenario_cfg: Mapping[str, object], carla) -> int:
    if pygame is None:
        raise RuntimeError("pygame is required for the two-camera CARLA scenario window.")

    scenario_cfg = dict(scenario_cfg or {})
    carla_cfg = dict(scenario_cfg.get("carla", {}))
    anchors_cfg = dict(scenario_cfg.get("anchors", {}))
    planning_cfg = dict(scenario_cfg.get("planning", {}))
    camera_cfg = dict(scenario_cfg.get("camera", {}))
    obstacle_cfg = dict(scenario_cfg.get("obstacles", {}))

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

    sample_distance_m = float(planning_cfg.get("waypoint_sample_distance_m", 2.0))
    lane_center_waypoints, road_cfg = build_lane_center_waypoints(
        map_obj=world_map,
        carla=carla,
        sample_distance_m=float(sample_distance_m),
    )
    global_planner = AStarGlobalPlanner(lane_center_waypoints=lane_center_waypoints)

    ego_anchor_name = str(anchors_cfg.get("ego_spawn", "cav_u"))
    destination_anchor_name = str(anchors_cfg.get("final_destination", "cav_w"))
    _print_anchor_lookup(world, carla, ego_anchor_name)
    _print_anchor_lookup(world, carla, destination_anchor_name)
    spawn_anchor = _resolve_anchor_transform(world, carla, ego_anchor_name)
    destination_anchor = _resolve_anchor_transform(world, carla, destination_anchor_name)
    aligned_spawn_transform, spawn_waypoint = _align_transform_to_lane(world_map, carla, spawn_anchor)
    aligned_destination_transform, destination_waypoint = _align_transform_to_lane(world_map, carla, destination_anchor)
    if spawn_waypoint is None or destination_waypoint is None:
        raise RuntimeError("Could not align the spawn or destination anchors to a driving lane.")

    initial_route_summary = global_planner.plan_route(
        start_xy=[
            float(aligned_spawn_transform.location.x),
            float(aligned_spawn_transform.location.y),
        ],
        goal_xy=[
            float(aligned_destination_transform.location.x),
            float(aligned_destination_transform.location.y),
        ],
    )
    initial_route_points: List[List[float]] = [
        [float(item[0]), float(item[1])]
        for item in initial_route_summary.route_waypoints
    ]
    if len(initial_route_points) == 0:
        initial_route_points = [
            [float(spawn_waypoint.transform.location.x), float(spawn_waypoint.transform.location.y)],
            [float(destination_waypoint.transform.location.x), float(destination_waypoint.transform.location.y)],
        ]

    ego_vehicle = _spawn_vehicle(
        world=world,
        blueprint_library=blueprint_library,
        scenario_cfg=scenario_cfg,
        spawn_transform=aligned_spawn_transform,
    )

    actors_to_destroy = [ego_vehicle]
    previous_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = bool(carla_cfg.get("synchronous_mode", True))
    settings.fixed_delta_seconds = float(carla_cfg.get("fixed_delta_seconds", 0.05))
    world.apply_settings(settings)
    carla_plan_dt_s = world.get_settings().fixed_delta_seconds
    if carla_plan_dt_s is not None and float(carla_plan_dt_s) > 0.0:
        configured_plan_dt_s = float(mpc_cfg.get("plan_dt_s", carla_plan_dt_s))
        if abs(float(configured_plan_dt_s) - float(carla_plan_dt_s)) > 1e-9:
            print(
                "[CARLA SCENARIO] Overriding mpc.plan_dt_s "
                f"from {configured_plan_dt_s:.6f}s to CARLA fixed_delta_seconds={float(carla_plan_dt_s):.6f}s"
            )
        mpc_cfg["plan_dt_s"] = float(carla_plan_dt_s)

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
                focus_points_xy=initial_route_points,
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

    prompt_builder = BehaviorPlannerPromptBuilder()
    tracker = Tracker(tracker_cfg=tracker_cfg)
    mpc = MPC(mpc_cfg=mpc_cfg, road_cfg=road_cfg)
    behavior_api_client = None
    if bool(behavior_runtime_cfg.get("enabled", False)) and bool(behavior_runtime_cfg.get("api_enabled", False)):
        behavior_api_client = BehaviorPlannerAPIClient(
            api_key_env_var=str(behavior_runtime_cfg.get("api_key_env_var", "OPENAI_API_KEY")),
            model=str(behavior_runtime_cfg.get("model", "gpt-4o")),
            temperature=float(behavior_runtime_cfg.get("temperature", 0.0)),
            request_timeout_s=float(behavior_runtime_cfg.get("request_timeout_s", 30.0)),
            max_output_tokens=int(behavior_runtime_cfg.get("max_output_tokens", 80)),
            enabled=bool(behavior_runtime_cfg.get("api_enabled", True)),
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
    )
    initial_lane_count = int(initial_lane_context.get("lane_count", 0))
    if initial_lane_count > 0:
        road_cfg["lane_count"] = int(initial_lane_count)
    selected_lane_id = int(initial_lane_context.get("lane_id", 1))
    lane_count = max(1, int(road_cfg.get("lane_count", 1)))
    current_applied_behavior = "LANE_KEEP"
    latest_decision = _default_lane_keep_decision(request_id="0")
    last_applied_decision_id = str(latest_decision.request_id)
    current_target_v_mps = float(local_goal_cfg.get("v_ref_mps", 10.0))
    current_max_velocity_override_mps: float | None = None
    original_max_velocity_mps = float(mpc.constraints.max_velocity_mps)
    saved_max_velocity_before_emergency_mps: float | None = None

    current_acceleration_mps2 = 0.0
    current_steering_rad = 0.0
    route_rebuild_period_s = float(planning_cfg.get("route_rebuild_period_s", 1.0))
    last_route_rebuild_wall_time_s = -1.0
    route_points: List[List[float]] = list(initial_route_points)
    behavior_stop_event = threading.Event()
    behavior_shared_lock = threading.Lock()
    behavior_shared_state: Dict[str, Any] = {
        "latest_inputs": None,
        "latest_prompt": "",
        "latest_prompt_id": "0",
        "latest_decision": latest_decision,
        "prompt_seq": 0,
        "current_applied_behavior": current_applied_behavior,
        "previous_behavior_for_prompt": current_applied_behavior,
    }
    behavior_thread = None
    if bool(behavior_runtime_cfg.get("enabled", False)):
        system_instruction = prompt_builder.load_system_instruction()
        behavior_thread = threading.Thread(
            target=_behavior_planner_worker,
            kwargs={
                "stop_event": behavior_stop_event,
                "shared_state": behavior_shared_state,
                "shared_lock": behavior_shared_lock,
                "prompt_builder": prompt_builder,
                "api_client": behavior_api_client,
                "system_instruction": system_instruction,
                "behavior_runtime_cfg": behavior_runtime_cfg,
                "lane_center_waypoints": lane_center_waypoints,
                "road_cfg": road_cfg,
                "mpc_constraints": mpc_constraints_cfg,
                "ego_vehicle_id": "Ego01",
            },
            daemon=True,
        )
        behavior_thread.start()

    try:
        obstacle_prefix = str(obstacle_cfg.get("environment_name_prefix", "")).strip()
        if obstacle_prefix:
            static_obstacles = _collect_environment_obstacle_snapshots(
                world=world,
                map_obj=world_map,
                carla=carla,
                obstacle_prefix=obstacle_prefix,
            )
            print(
                f"[CARLA SCENARIO] Found {len(static_obstacles)} static environment obstacles "
                f"with prefix '{obstacle_prefix}'."
            )
            for snapshot in static_obstacles:
                print(
                    "[CARLA SCENARIO] Obstacle "
                    f"{snapshot['vehicle_id']} at ({snapshot['x']:.3f}, {snapshot['y']:.3f}) "
                    f"size=({snapshot['length_m']:.2f}m, {snapshot['width_m']:.2f}m)"
                )
        while True:
            world.tick()
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

            now_wall_time_s = time.monotonic()
            if last_route_rebuild_wall_time_s < 0.0 or (now_wall_time_s - last_route_rebuild_wall_time_s) >= route_rebuild_period_s:
                route_summary = global_planner.plan_route(
                    start_xy=ego_state[:2],
                    goal_xy=final_destination_state[:2],
                )
                if route_summary.route_waypoints:
                    route_points = [[float(item[0]), float(item[1])] for item in route_summary.route_waypoints]
                last_route_rebuild_wall_time_s = now_wall_time_s

            current_lane_context = global_planner.get_local_lane_context(
                x_m=float(ego_state[0]),
                y_m=float(ego_state[1]),
                heading_rad=float(ego_state[3]),
            )
            current_lane_id = int(current_lane_context.get("lane_id", selected_lane_id))
            local_allowed_lane_ids = _allowed_lane_ids_from_context(
                local_context=current_lane_context,
                fallback_lane_count=int(lane_count),
            )
            local_lane_count = len(local_allowed_lane_ids)
            if int(current_lane_id) <= 0:
                current_lane_id = _clamp_lane_id_to_allowed(selected_lane_id, local_allowed_lane_ids)
            selected_lane_id = _clamp_lane_id_to_allowed(selected_lane_id, local_allowed_lane_ids)

            dynamic_object_snapshots = _collect_vehicle_snapshots(world, ego_vehicle)
            static_object_snapshots = []
            if obstacle_prefix:
                static_object_snapshots = _collect_environment_obstacle_snapshots(
                    world=world,
                    map_obj=world_map,
                    carla=carla,
                    obstacle_prefix=obstacle_prefix,
                )
            sim_time_s = float(world.get_snapshot().timestamp.elapsed_seconds)
            tracker.update(obstacle_snapshots=dynamic_object_snapshots, timestamp_s=sim_time_s)
            predicted_snapshots = _merge_tracker_predictions(
                object_snapshots=dynamic_object_snapshots,
                predictions=tracker.predict(step_dt_s=float(mpc.dt_s), horizon_s=float(mpc.horizon_s)),
            )
            predicted_snapshots.extend(list(static_object_snapshots))

            base_target_lane_id = (
                int(current_lane_id)
                if int(current_lane_id) > 0
                else min(max(1, int(selected_lane_id)), int(lane_count))
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
            base_destination_state = compute_temporary_destination_state(
                ego_snapshot={
                    "x": float(ego_state[0]),
                    "y": float(ego_state[1]),
                    "v": float(ego_state[2]),
                    "psi": float(ego_state[3]),
                },
                lane_center_waypoints=lane_center_waypoints,
                target_lane_id=int(base_target_lane_id),
                target_distance_m=float(rolling_target_distance_m),
                target_v_mps=float(current_target_v_mps),
                road_cfg=road_cfg,
            )
            if base_destination_state is None:
                base_destination_state = list(final_destination_state)
                base_destination_state[2] = float(current_target_v_mps)

            if bool(behavior_runtime_cfg.get("enabled", False)):
                with behavior_shared_lock:
                    latest_decision_from_worker = behavior_shared_state.get("latest_decision", None)
                if (
                    isinstance(latest_decision_from_worker, BehaviorPlannerDecision)
                    and str(latest_decision_from_worker.request_id) != str(last_applied_decision_id)
                ):
                    execution = apply_behavior_planner_decision(
                        decision=latest_decision_from_worker,
                        ego_snapshot={
                            "x": float(ego_state[0]),
                            "y": float(ego_state[1]),
                            "v": float(ego_state[2]),
                            "psi": float(ego_state[3]),
                        },
                        base_destination_state=base_destination_state,
                        final_destination_state=final_destination_state,
                        lane_center_waypoints=lane_center_waypoints,
                        selected_lane_id=int(selected_lane_id),
                        previous_applied_behavior=str(current_applied_behavior),
                        road_cfg=road_cfg,
                        local_goal_cfg=local_goal_cfg,
                        mpc_constraints=mpc_constraints_cfg,
                        target_distance_m=float(rolling_target_distance_m),
                    )
                    latest_decision = latest_decision_from_worker
                    last_applied_decision_id = str(latest_decision_from_worker.request_id)
                    current_applied_behavior = str(execution.applied_behavior or current_applied_behavior)
                    current_target_v_mps = float(execution.destination_state[2])
                    current_max_velocity_override_mps = execution.max_velocity_override_mps
                    selected_lane_id = _clamp_lane_id_to_allowed(
                        int(execution.selected_lane_id),
                        local_allowed_lane_ids,
                    )

            behavior_upper = str(current_applied_behavior).upper()
            active_target_lane_id = _clamp_lane_id_to_allowed(
                int(selected_lane_id),
                local_allowed_lane_ids,
            )

            target_distance_for_destination_m = float(rolling_target_distance_m)
            if behavior_upper == "EMERGENCY_BRAKE":
                braking_mps2 = max(1e-6, abs(float(mpc_constraints_cfg.get("min_acceleration_mps2", -3.0))))
                target_distance_for_destination_m = max(
                    1.0,
                    float(max(0.0, ego_state[2])) ** 2 / (2.0 * braking_mps2),
                )
            elif behavior_upper == "STOP_AT_LINE":
                target_distance_for_destination_m = max(
                    1.0,
                    min(
                        float(road_cfg.get("distance_to_signal_m", 10000.0)),
                        float(road_cfg.get("distance_to_intersection_m", 10000.0)),
                    ),
                )

            if (
                int(active_target_lane_id) == int(base_target_lane_id)
                and abs(float(target_distance_for_destination_m) - float(rolling_target_distance_m)) <= 1e-6
                and len(base_destination_state) >= 4
            ):
                temporary_destination_state = list(base_destination_state)
            else:
                temporary_destination_state = compute_temporary_destination_state(
                    ego_snapshot={
                        "x": float(ego_state[0]),
                        "y": float(ego_state[1]),
                        "v": float(ego_state[2]),
                        "psi": float(ego_state[3]),
                    },
                    lane_center_waypoints=lane_center_waypoints,
                    target_lane_id=int(active_target_lane_id),
                    target_distance_m=float(target_distance_for_destination_m),
                    target_v_mps=float(current_target_v_mps),
                    road_cfg=road_cfg,
                )
            if temporary_destination_state is None:
                temporary_destination_state = list(base_destination_state)

            if behavior_upper == "EMERGENCY_BRAKE":
                temporary_destination_state[2] = 0.0
            elif behavior_upper == "STOP_AT_LINE":
                temporary_destination_state[2] = 0.0

            if bool(behavior_runtime_cfg.get("enabled", False)):
                with behavior_shared_lock:
                    behavior_shared_state["latest_inputs"] = {
                        "ego_snapshot": {
                            "x": float(ego_state[0]),
                            "y": float(ego_state[1]),
                            "v": float(ego_state[2]),
                            "psi": float(ego_state[3]),
                        },
                        "destination_state": list(final_destination_state),
                        "temporary_destination_state": list(temporary_destination_state),
                        "object_snapshots": list(predicted_snapshots),
                    }
                    behavior_shared_state["current_applied_behavior"] = str(current_applied_behavior)

            if current_applied_behavior == "EMERGENCY_BRAKE":
                if saved_max_velocity_before_emergency_mps is None:
                    saved_max_velocity_before_emergency_mps = float(mpc.constraints.max_velocity_mps)
                mpc.constraints.max_velocity_mps = 0.0
            else:
                if saved_max_velocity_before_emergency_mps is not None:
                    mpc.constraints.max_velocity_mps = float(saved_max_velocity_before_emergency_mps)
                    saved_max_velocity_before_emergency_mps = None
                else:
                    mpc.constraints.max_velocity_mps = float(original_max_velocity_mps)

            if current_max_velocity_override_mps is not None:
                mpc.constraints.max_velocity_mps = float(current_max_velocity_override_mps)

            planned_trajectory = mpc.plan_trajectory(
                current_state=ego_state,
                destination_state=temporary_destination_state,
                object_snapshots=predicted_snapshots,
                current_acceleration_mps2=float(current_acceleration_mps2),
                current_steering_rad=float(current_steering_rad),
                lane_center_waypoints=lane_center_waypoints,
            )
            if not planned_trajectory:
                ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
            else:
                control_solution = getattr(mpc, "_last_u_solution", None)
                if control_solution is None or len(control_solution) == 0:
                    ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.3, steer=0.0))
                else:
                    current_acceleration_mps2 = float(control_solution[0][0])
                    current_steering_rad = float(control_solution[0][1])
                    ego_vehicle.apply_control(
                        _control_from_mpc(
                            mpc=mpc,
                            carla=carla,
                            acceleration_mps2=float(current_acceleration_mps2),
                            steering_angle_rad=float(current_steering_rad),
                        )
                    )

            if display is not None and topdown_queue is not None and chase_queue is not None:
                topdown_image = None
                chase_image = None
                try:
                    topdown_image = topdown_queue.get(timeout=0.25)
                except queue.Empty:
                    pass
                try:
                    chase_image = chase_queue.get(timeout=0.25)
                except queue.Empty:
                    pass
                route_line = "ROUTE:[pending]"
                with behavior_shared_lock:
                    route_line = _latest_route_line(str(behavior_shared_state.get("latest_prompt", "")))

                temp_lane_context = global_planner.get_local_lane_context(
                    x_m=float(temporary_destination_state[0]),
                    y_m=float(temporary_destination_state[1]),
                    heading_rad=float(temporary_destination_state[3]) if len(temporary_destination_state) >= 4 else 0.0,
                )
                temp_lane_prompt = prompt_builder._prompt_lane_id(
                    raw_lane_id=(
                        int(temporary_destination_state[4])
                        if len(temporary_destination_state) >= 5
                        else int(temp_lane_context.get("lane_id", 1))
                    ),
                    lane_count=int(lane_count),
                )
                terminal_planned_velocity_mps = float("nan")
                if planned_trajectory and len(planned_trajectory[-1]) >= 3:
                    terminal_planned_velocity_mps = float(planned_trajectory[-1][2])
                lookahead_cmd_text = f"{int(round(float(rolling_target_distance_m)))} m"
                hud_lines = [
                    route_line,
                    f"v={float(ego_state[2]):.2f} m/s",
                    f"v_ref={float(temporary_destination_state[2]):.2f} m/s",
                    f"v_max={float(mpc.constraints.max_velocity_mps):.2f} m/s",
                    f"traj_v_end={float(terminal_planned_velocity_mps):.2f} m/s",
                    f"temp_lane={int(temp_lane_prompt)}",
                    f"lookahead={lookahead_cmd_text}",
                    f"decision={str(current_applied_behavior)}",
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
                        "temporary_destination_state": list(temporary_destination_state),
                        "planned_trajectory_states": list(planned_trajectory or []),
                        "obstacle_field_contours": _build_obstacle_field_contours(
                            mpc=mpc,
                            ego_state=ego_state,
                            object_snapshots=predicted_snapshots,
                        ),
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
        behavior_stop_event.set()
        if behavior_thread is not None:
            behavior_thread.join(timeout=1.0)
        world.apply_settings(previous_settings)
        _destroy_actors(actors_to_destroy)
        if pygame is not None:
            pygame.quit()

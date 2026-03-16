"""
Rendering helper functions for MPC_custom.

These helpers keep pygame drawing logic out of `main.py` so the simulation loop
remains focused on state updates and planning calls.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence, Tuple
import math

import numpy as np
import pygame


ColorRGB = Tuple[int, int, int]


def world_to_screen(
    x_m: float,
    y_m: float,
    camera_center_world: Tuple[float, float],
    pixels_per_meter: float,
    screen_center_px: Tuple[float, float],
) -> Tuple[int, int]:
    """
    Intent:
        Convert world coordinates [m] into pygame pixel coordinates.

    Coordinate convention:
        - +x is right in world and on screen.
        - +y is up in world but pygame y increases downward, so y is inverted.
    """

    cam_x_m, cam_y_m = float(camera_center_world[0]), float(camera_center_world[1])
    cx_px, cy_px = float(screen_center_px[0]), float(screen_center_px[1])
    px = int(round(cx_px + (float(x_m) - cam_x_m) * float(pixels_per_meter)))
    py = int(round(cy_px - (float(y_m) - cam_y_m) * float(pixels_per_meter)))
    return px, py


def screen_to_world(
    x_px: float,
    y_px: float,
    camera_center_world: Tuple[float, float],
    pixels_per_meter: float,
    screen_center_px: Tuple[float, float],
) -> Tuple[float, float]:
    """
    Intent:
        Convert a mouse click or pixel coordinate into world coordinates [m].
    """

    cam_x_m, cam_y_m = float(camera_center_world[0]), float(camera_center_world[1])
    cx_px, cy_px = float(screen_center_px[0]), float(screen_center_px[1])
    x_m = cam_x_m + (float(x_px) - cx_px) / max(1e-9, float(pixels_per_meter))
    y_m = cam_y_m - (float(y_px) - cy_px) / max(1e-9, float(pixels_per_meter))
    return float(x_m), float(y_m)


def draw_destination(
    surface: pygame.Surface,
    destination_state: Sequence[float],
    camera_center_world: Tuple[float, float],
    pixels_per_meter: float,
    fill_color_rgb: ColorRGB = (220, 35, 35),
    outline_color_rgb: ColorRGB = (245, 245, 245),
    radius_px: int = 7,
) -> None:
    """
    Intent:
        Draw a destination target marker as a filled circle with an outline.

    Inputs:
        destination_state:
            sequence[float], expects at least [x, y, ...].
    """

    width_px, height_px = surface.get_size()
    screen_center = (0.5 * width_px, 0.5 * height_px)
    px, py = world_to_screen(
        float(destination_state[0]),
        float(destination_state[1]),
        camera_center_world,
        pixels_per_meter,
        screen_center,
    )
    pygame.draw.circle(surface, outline_color_rgb, (px, py), int(radius_px) + 2)
    pygame.draw.circle(surface, fill_color_rgb, (px, py), int(radius_px))


def _draw_dotted_polyline(
    surface: pygame.Surface,
    points_px: Sequence[Tuple[int, int]],
    color_rgb: ColorRGB,
    segment_dot_spacing_px: int,
    dot_radius_px: int,
) -> None:
    """
    Intent:
        Draw a dotted line along a polyline by placing dots at fixed pixel
        spacing along each segment.
    """

    if len(points_px) < 2:
        return

    spacing_px = max(2, int(segment_dot_spacing_px))
    radius_px = max(1, int(dot_radius_px))

    for idx in range(len(points_px) - 1):
        x0, y0 = points_px[idx]
        x1, y1 = points_px[idx + 1]
        dx = float(x1 - x0)
        dy = float(y1 - y0)
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-6:
            pygame.draw.circle(surface, color_rgb, (x0, y0), radius_px)
            continue
        dot_count = max(1, int(seg_len // spacing_px))
        for dot_idx in range(dot_count + 1):
            t = min(1.0, (dot_idx * spacing_px) / seg_len)
            px = int(round(x0 + t * dx))
            py = int(round(y0 + t * dy))
            pygame.draw.circle(surface, color_rgb, (px, py), radius_px)


def draw_dotted_trajectory(
    surface: pygame.Surface,
    trajectory_states: Sequence[Sequence[float]],
    camera_center_world: Tuple[float, float],
    pixels_per_meter: float,
    color_rgb: ColorRGB = (25, 25, 25),
    dot_spacing_px: int = 14,
    dot_radius_px: int = 2,
) -> None:
    """
    Intent:
        Draw an MPC trajectory (future ego states) as a dotted polyline.

    Inputs:
        trajectory_states:
            sequence of [x, y, v, psi] states. Only x and y are rendered.
    """

    if len(trajectory_states) == 0:
        return

    width_px, height_px = surface.get_size()
    screen_center = (0.5 * width_px, 0.5 * height_px)
    points_px = [
        world_to_screen(float(state[0]), float(state[1]), camera_center_world, pixels_per_meter, screen_center)
        for state in trajectory_states
        if len(state) >= 2
    ]
    _draw_dotted_polyline(surface, points_px, color_rgb, dot_spacing_px, dot_radius_px)


def draw_predicted_object_trajectories(
    surface: pygame.Surface,
    prediction_by_object_id: Mapping[str, Sequence[Mapping[str, float]]],
    camera_center_world: Tuple[float, float],
    pixels_per_meter: float,
    color_rgb: ColorRGB = (220, 45, 45),
) -> None:
    """
    Intent:
        Draw predicted future trajectories for surrounding objects as red dotted
        lines so the user can visually compare tracker output and MPC behavior.
    """

    width_px, height_px = surface.get_size()
    screen_center = (0.5 * width_px, 0.5 * height_px)
    for _, samples in prediction_by_object_id.items():
        points_px = []
        for sample in samples:
            if "x" not in sample or "y" not in sample:
                continue
            points_px.append(
                world_to_screen(
                    float(sample["x"]),
                    float(sample["y"]),
                    camera_center_world,
                    pixels_per_meter,
                    screen_center,
                )
            )
        _draw_dotted_polyline(surface, points_px, color_rgb, 10, 2)


def _compute_superellipsoid_zone_geometry(
    ego_snapshot: Mapping[str, object],
    obstacle_snapshot: Mapping[str, object],
    repulsive_cfg: Mapping[str, object],
    lane_width_m: float,
) -> Mapping[str, float] | None:
    """
    Compute the live MPC super-ellipsoid geometry and the tighter visual
    drawing sizes used for the overlay.
    """

    ego_state = list(ego_snapshot.get("current_state", []))
    obstacle_state = list(obstacle_snapshot.get("current_state", []))
    if len(ego_state) < 4 or len(obstacle_state) < 4:
        return None

    ego_v_mps = max(0.0, float(ego_state[2]))
    ego_psi_rad = float(ego_state[3])
    obs_v_mps = max(0.0, float(obstacle_state[2]))
    obs_psi_rad = float(obstacle_state[3])

    ego_length_m = max(1e-6, float(ego_snapshot.get("length_m", 4.5)))
    ego_width_m = max(1e-6, float(ego_snapshot.get("width_m", 2.0)))
    obstacle_length_m = max(1e-6, float(obstacle_snapshot.get("length_m", 4.5)))
    obstacle_width_m = max(1e-6, float(obstacle_snapshot.get("width_m", 2.0)))

    heading_diff_rad = float(ego_psi_rad) - float(obs_psi_rad)
    v_approach_longitudinal_mps = float(ego_v_mps) * math.cos(heading_diff_rad) - float(obs_v_mps)
    v_approach_lateral_mps = float(ego_v_mps) * math.sin(heading_diff_rad)
    delta_u_mps = max(0.0, float(v_approach_longitudinal_mps))
    delta_v_mps = max(
        max(0.0, float(repulsive_cfg.get("min_lateral_approach_speed_mps", 0.1))),
        abs(float(v_approach_lateral_mps)),
    )

    projected_ego_length_m = abs(float(ego_length_m) * math.cos(heading_diff_rad))
    projected_ego_length_m += abs(float(ego_width_m) * math.sin(heading_diff_rad))
    projected_ego_width_m = abs(float(ego_length_m) * math.sin(heading_diff_rad))
    projected_ego_width_m += abs(float(ego_width_m) * math.cos(heading_diff_rad))

    x0_m = 0.5 * (projected_ego_length_m + float(obstacle_length_m))
    x0_m += max(0.0, float(repulsive_cfg.get("static_longitudinal_buffer_m", 0.5)))
    y0_m = 0.5 * (projected_ego_width_m + float(obstacle_width_m))
    y0_m += max(0.0, float(repulsive_cfg.get("static_lateral_buffer_m", 1.0)))

    max_braking_mps2 = max(1e-6, float(repulsive_cfg.get("max_braking_deceleration_mps2", 5.0)))
    comfort_mps2 = max(1e-6, float(repulsive_cfg.get("comfort_deceleration_mps2", 2.0)))
    reaction_time_s = max(0.0, float(repulsive_cfg.get("reaction_time_s", 1.0)))

    xc_raw_m = x0_m + (delta_u_mps * delta_u_mps) / (2.0 * max_braking_mps2)
    yc_raw_m = y0_m + (delta_v_mps * delta_v_mps) / (2.0 * max_braking_mps2)
    xs_raw_m = x0_m + delta_u_mps * reaction_time_s + (delta_u_mps * delta_u_mps) / (2.0 * comfort_mps2)
    ys_raw_m = y0_m + delta_v_mps * reaction_time_s + (delta_v_mps * delta_v_mps) / (2.0 * comfort_mps2)

    xc_m = float(xc_raw_m)
    yc_m = float(yc_raw_m)
    xs_m = float(xs_raw_m)
    ys_m = float(ys_raw_m)

    longitudinal_half_limit_m = 0.5 * max(1e-6, float(repulsive_cfg.get("max_longitudinal_zone_length_m", 10.0)))
    xc_m = min(float(xc_m), longitudinal_half_limit_m)
    xs_m = min(float(xs_m), longitudinal_half_limit_m)

    if bool(repulsive_cfg.get("limit_lateral_zone_to_lane_width", True)):
        lateral_half_limit_m = 0.5 * max(1e-6, float(lane_width_m))
        lateral_half_limit_m *= max(1e-3, float(repulsive_cfg.get("max_lateral_zone_lane_fraction", 1.0)))
        yc_m = min(float(yc_m), lateral_half_limit_m)
        ys_m = min(float(ys_m), lateral_half_limit_m)

    # Match the visual convention used in super_ellipsoid.py: draw the zone
    # around the obstacle body itself instead of the full ego-center Minkowski
    # sum used internally by the MPC cost.
    visual_longitudinal_inflation_m = 0.5 * projected_ego_length_m
    visual_longitudinal_inflation_m += max(0.0, float(repulsive_cfg.get("static_longitudinal_buffer_m", 0.5)))
    visual_lateral_inflation_m = 0.5 * projected_ego_width_m
    visual_lateral_inflation_m += max(0.0, float(repulsive_cfg.get("static_lateral_buffer_m", 1.0)))

    draw_xc_m = max(0.5 * float(obstacle_length_m), float(xc_raw_m) - visual_longitudinal_inflation_m)
    draw_xs_m = max(0.5 * float(obstacle_length_m), float(xs_raw_m) - visual_longitudinal_inflation_m)
    draw_yc_m = max(0.5 * float(obstacle_width_m), float(yc_raw_m) - visual_lateral_inflation_m)
    draw_ys_m = max(0.5 * float(obstacle_width_m), float(ys_raw_m) - visual_lateral_inflation_m)

    draw_xc_m = min(float(draw_xc_m), longitudinal_half_limit_m)
    draw_xs_m = min(float(draw_xs_m), longitudinal_half_limit_m)

    if bool(repulsive_cfg.get("limit_lateral_zone_to_lane_width", True)):
        draw_lateral_half_limit_m = 0.5 * max(1e-6, float(lane_width_m))
        draw_lateral_half_limit_m *= max(1e-3, float(repulsive_cfg.get("max_lateral_zone_lane_fraction", 1.0)))
        draw_yc_m = min(float(draw_yc_m), draw_lateral_half_limit_m)
        draw_ys_m = min(float(draw_ys_m), draw_lateral_half_limit_m)

    return {
        "xc_m": float(max(1e-3, xc_m)),
        "yc_m": float(max(1e-3, yc_m)),
        "xs_m": float(max(1e-3, xs_m)),
        "ys_m": float(max(1e-3, ys_m)),
        "draw_xc_m": float(max(0.1, draw_xc_m)),
        "draw_yc_m": float(max(0.1, draw_yc_m)),
        "draw_xs_m": float(max(0.1, draw_xs_m)),
        "draw_ys_m": float(max(0.1, draw_ys_m)),
    }


def _superellipse_polygon_points(
    center_x_m: float,
    center_y_m: float,
    heading_rad: float,
    half_length_m: float,
    half_width_m: float,
    exponent: float,
    sample_count: int,
    camera_center_world: Tuple[float, float],
    pixels_per_meter: float,
    screen_center_px: Tuple[float, float],
) -> Sequence[Tuple[int, int]]:
    """
    Sample a super-ellipse in the obstacle local frame and transform it into
    screen points using the same body-frame rotation convention as vehicle.draw.
    """

    half_length_m = max(1e-6, float(half_length_m))
    half_width_m = max(1e-6, float(half_width_m))
    exponent = max(2.0, float(exponent))
    sample_count = max(32, int(sample_count))

    cos_psi = math.cos(float(heading_rad))
    sin_psi = math.sin(float(heading_rad))
    points_px = []
    for theta_rad in np.linspace(0.0, 2.0 * math.pi, sample_count, endpoint=False):
        cos_theta = math.cos(float(theta_rad))
        sin_theta = math.sin(float(theta_rad))
        local_x_m = math.copysign(
            half_length_m * (abs(cos_theta) ** (2.0 / exponent)),
            cos_theta,
        )
        local_y_m = math.copysign(
            half_width_m * (abs(sin_theta) ** (2.0 / exponent)),
            sin_theta,
        )
        world_x_m = float(center_x_m) + local_x_m * cos_psi - local_y_m * sin_psi
        world_y_m = float(center_y_m) + local_x_m * sin_psi + local_y_m * cos_psi
        points_px.append(
            world_to_screen(
                world_x_m,
                world_y_m,
                camera_center_world,
                pixels_per_meter,
                screen_center_px,
            )
        )
    return points_px


def draw_obstacle_potential_fields(
    surface: pygame.Surface,
    ego_snapshot: Mapping[str, object],
    object_snapshots: Sequence[Mapping[str, object]],
    camera_center_world: Tuple[float, float],
    pixels_per_meter: float,
    repulsive_cfg: Mapping[str, object],
    lane_width_m: float,
) -> None:
    """
    Visualize the live super-ellipsoid safe/collision zones around each
    non-ego object using the same geometry model as the MPC obstacle cost.
    """

    visualization_cfg = dict(repulsive_cfg.get("visualization", {}))
    if not bool(visualization_cfg.get("enabled", False)):
        return

    width_px, height_px = surface.get_size()
    screen_center = (0.5 * width_px, 0.5 * height_px)
    ppm = max(1e-6, float(pixels_per_meter))
    contour_resolution = max(48, int(visualization_cfg.get("grid_resolution", 120)))
    draw_margin_m = max(0.0, float(visualization_cfg.get("draw_margin_m", 0.5)))
    safe_color_rgba = tuple(int(v) for v in visualization_cfg.get("safe_zone_color_rgba", [0, 200, 0, 70]))
    collision_color_rgba = tuple(int(v) for v in visualization_cfg.get("collision_zone_color_rgba", [220, 40, 40, 110]))
    shape_exponent = max(2.0, float(repulsive_cfg.get("shape_exponent", 4.0)))
    overlay_surface = pygame.Surface((width_px, height_px), pygame.SRCALPHA)

    for obstacle_snapshot in object_snapshots:
        if str(obstacle_snapshot.get("type", "")).strip().lower() == "ego":
            continue

        zone_geometry = _compute_superellipsoid_zone_geometry(
            ego_snapshot=ego_snapshot,
            obstacle_snapshot=obstacle_snapshot,
            repulsive_cfg=repulsive_cfg,
            lane_width_m=lane_width_m,
        )
        if zone_geometry is None:
            continue

        draw_xc_m = float(zone_geometry["draw_xc_m"])
        draw_yc_m = float(zone_geometry["draw_yc_m"])
        draw_xs_m = float(zone_geometry["draw_xs_m"])
        draw_ys_m = float(zone_geometry["draw_ys_m"])
        half_draw_x_m = max(draw_xc_m, draw_xs_m) + draw_margin_m
        half_draw_y_m = max(draw_yc_m, draw_ys_m) + draw_margin_m
        draw_width_px = max(2, int(round(2.0 * half_draw_x_m * ppm)))
        draw_height_px = max(2, int(round(2.0 * half_draw_y_m * ppm)))

        obs_x_m = float(obstacle_snapshot.get("x", 0.0))
        obs_y_m = float(obstacle_snapshot.get("y", 0.0))
        center_px = world_to_screen(obs_x_m, obs_y_m, camera_center_world, pixels_per_meter, screen_center)
        if (
            center_px[0] + draw_width_px < 0
            or center_px[0] - draw_width_px > width_px
            or center_px[1] + draw_height_px < 0
            or center_px[1] - draw_height_px > height_px
        ):
            continue

        obstacle_heading_rad = float(obstacle_snapshot.get("psi", 0.0))
        safe_polygon_px = _superellipse_polygon_points(
            center_x_m=obs_x_m,
            center_y_m=obs_y_m,
            heading_rad=obstacle_heading_rad,
            half_length_m=draw_xs_m,
            half_width_m=draw_ys_m,
            exponent=shape_exponent,
            sample_count=contour_resolution,
            camera_center_world=camera_center_world,
            pixels_per_meter=pixels_per_meter,
            screen_center_px=screen_center,
        )
        collision_polygon_px = _superellipse_polygon_points(
            center_x_m=obs_x_m,
            center_y_m=obs_y_m,
            heading_rad=obstacle_heading_rad,
            half_length_m=draw_xc_m,
            half_width_m=draw_yc_m,
            exponent=shape_exponent,
            sample_count=contour_resolution,
            camera_center_world=camera_center_world,
            pixels_per_meter=pixels_per_meter,
            screen_center_px=screen_center,
        )
        pygame.draw.polygon(overlay_surface, safe_color_rgba, safe_polygon_px)
        pygame.draw.polygon(overlay_surface, collision_color_rgba, collision_polygon_px)

    surface.blit(overlay_surface, (0, 0))


def _nice_scale_step_m(min_step_m: float) -> float:
    """
    Compute a readable tick step in meters using 1-2-5 progression.
    """

    value = max(1e-6, float(min_step_m))
    exponent = math.floor(math.log10(value))
    base = value / (10.0 ** exponent)
    if base <= 1.0:
        nice_base = 1.0
    elif base <= 2.0:
        nice_base = 2.0
    elif base <= 5.0:
        nice_base = 5.0
    else:
        nice_base = 10.0
    return float(nice_base * (10.0 ** exponent))


def draw_world_scale(
    surface: pygame.Surface,
    font: pygame.font.Font,
    camera_center_world: Tuple[float, float],
    pixels_per_meter: float,
    step_m: float | None = None,
    min_tick_spacing_px: int = 80,
    margin_px: int = 14,
    axis_color_rgb: ColorRGB = (240, 240, 240),
) -> None:
    """
    Draw X/Y meter scales on the screen border.

    - X scale: bottom ruler with world-X labels [m].
    - Y scale: left ruler with world-Y labels [m].
    """

    width_px, height_px = surface.get_size()
    ppm = max(1e-6, float(pixels_per_meter))
    cam_x_m, cam_y_m = float(camera_center_world[0]), float(camera_center_world[1])

    world_x_min = cam_x_m - 0.5 * float(width_px) / ppm
    world_x_max = cam_x_m + 0.5 * float(width_px) / ppm
    world_y_min = cam_y_m - 0.5 * float(height_px) / ppm
    world_y_max = cam_y_m + 0.5 * float(height_px) / ppm

    if step_m is None or float(step_m) <= 0.0:
        tick_step_m = _nice_scale_step_m(float(min_tick_spacing_px) / ppm)
    else:
        tick_step_m = max(1e-6, float(step_m))

    tick_label_decimals = 0 if tick_step_m >= 1.0 else 1

    x_axis_y_px = int(height_px - margin_px)
    pygame.draw.line(surface, axis_color_rgb, (margin_px, x_axis_y_px), (width_px - margin_px, x_axis_y_px), 1)

    x_tick_m = math.ceil(world_x_min / tick_step_m) * tick_step_m
    while x_tick_m <= world_x_max + 1e-9:
        tick_x_px = int(round(0.5 * float(width_px) + (x_tick_m - cam_x_m) * ppm))
        if margin_px <= tick_x_px <= width_px - margin_px:
            pygame.draw.line(surface, axis_color_rgb, (tick_x_px, x_axis_y_px - 5), (tick_x_px, x_axis_y_px + 5), 1)
            tick_text = f"{x_tick_m:.{tick_label_decimals}f}"
            tick_surface = font.render(tick_text, True, axis_color_rgb)
            surface.blit(tick_surface, (tick_x_px - 0.5 * tick_surface.get_width(), x_axis_y_px - tick_surface.get_height() - 8))
        x_tick_m += tick_step_m

    x_label = font.render("X [m]", True, axis_color_rgb)
    surface.blit(x_label, (width_px - margin_px - x_label.get_width(), x_axis_y_px - x_label.get_height() - 6))

    y_axis_x_px = int(margin_px)
    pygame.draw.line(surface, axis_color_rgb, (y_axis_x_px, margin_px), (y_axis_x_px, height_px - margin_px), 1)

    y_tick_m = math.ceil(world_y_min / tick_step_m) * tick_step_m
    while y_tick_m <= world_y_max + 1e-9:
        tick_y_px = int(round(0.5 * float(height_px) - (y_tick_m - cam_y_m) * ppm))
        if margin_px <= tick_y_px <= height_px - margin_px:
            pygame.draw.line(surface, axis_color_rgb, (y_axis_x_px - 5, tick_y_px), (y_axis_x_px + 5, tick_y_px), 1)
            tick_text = f"{y_tick_m:.{tick_label_decimals}f}"
            tick_surface = font.render(tick_text, True, axis_color_rgb)
            surface.blit(tick_surface, (y_axis_x_px + 8, tick_y_px - 0.5 * tick_surface.get_height()))
        y_tick_m += tick_step_m

    y_label = font.render("Y [m]", True, axis_color_rgb)
    surface.blit(y_label, (y_axis_x_px + 8, margin_px))


def draw_hud_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    lines: Iterable[str],
    top_left_px: Tuple[int, int],
    anchor: str = "left",
) -> None:
    """
    Intent:
        Render a simple text HUD with one line per row.
    """

    x0, y0 = int(top_left_px[0]), int(top_left_px[1])
    line_height = int(font.get_linesize()) + 2
    for idx, text in enumerate(lines):
        text_surface = font.render(str(text), True, (245, 245, 245))
        shadow_surface = font.render(str(text), True, (20, 20, 20))
        y = y0 + idx * line_height
        if str(anchor).strip().lower() == "right":
            x = x0 - int(text_surface.get_width())
        else:
            x = x0
        surface.blit(shadow_surface, (x + 1, y + 1))
        surface.blit(text_surface, (x, y))

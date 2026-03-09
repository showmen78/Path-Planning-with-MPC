"""
Rendering helper functions for MPC_custom.

These helpers keep pygame drawing logic out of `main.py` so the simulation loop
remains focused on state updates and planning calls.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence, Tuple
import math

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


def draw_hud_text(surface: pygame.Surface, font: pygame.font.Font, lines: Iterable[str], top_left_px: Tuple[int, int]) -> None:
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
        surface.blit(shadow_surface, (x0 + 1, y + 1))
        surface.blit(text_surface, (x0, y))

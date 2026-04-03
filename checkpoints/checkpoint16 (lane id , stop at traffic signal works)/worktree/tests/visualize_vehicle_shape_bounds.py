#!/usr/bin/env python3
"""
Pygame demo: ellipse vs superellipse (n=4) as vehicle-shape bounds.

Run:
    python tests/visualize_vehicle_shape_bounds.py
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import pygame


WINDOW_WIDTH_PX = 1320
WINDOW_HEIGHT_PX = 760
BACKGROUND_RGB = (255, 255, 255)
RECTANGLE_FILL_RGB = (160, 160, 160)
RECTANGLE_BORDER_RGB = (110, 110, 110)
SHAPE_RGB = (220, 30, 30)
TEXT_RGB = (0, 0, 0)

VEHICLE_HALF_LENGTH_PX = 140.0
VEHICLE_HALF_WIDTH_PX = 70.0
SUPERELLIPSE_EXPONENT = 4.0


def _corner_touch_scale(shape_exponent: float) -> float:
    """
    Scale the semi-axes so the shape passes through the rectangle corners.

    For |x/a|^n + |y/b|^n = 1 and a rectangle corner at (L, W), choosing
    a = sL and b = sW with s = 2^(1/n) makes the corner lie on the boundary.
    """

    return 2.0 ** (1.0 / max(1e-6, float(shape_exponent)))


def _sample_superellipse_points(
    *,
    center_xy: Sequence[float],
    semi_axis_x_px: float,
    semi_axis_y_px: float,
    shape_exponent: float,
    num_points: int = 360,
) -> List[Tuple[int, int]]:
    points: List[Tuple[int, int]] = []
    exponent = max(2.0, float(shape_exponent))
    center_x_px = float(center_xy[0])
    center_y_px = float(center_xy[1])

    for idx in range(max(32, int(num_points))):
        theta_rad = 2.0 * math.pi * float(idx) / float(max(32, int(num_points)))
        cos_theta = math.cos(theta_rad)
        sin_theta = math.sin(theta_rad)
        local_x_px = float(semi_axis_x_px) * math.copysign(abs(cos_theta) ** (2.0 / exponent), cos_theta)
        local_y_px = float(semi_axis_y_px) * math.copysign(abs(sin_theta) ** (2.0 / exponent), sin_theta)
        points.append(
            (
                int(round(center_x_px + local_x_px)),
                int(round(center_y_px + local_y_px)),
            )
        )
    return points


def _draw_vehicle(surface, center_xy: Sequence[float], half_length_px: float, half_width_px: float) -> None:
    rect = pygame.Rect(
        int(round(float(center_xy[0]) - float(half_length_px))),
        int(round(float(center_xy[1]) - float(half_width_px))),
        int(round(2.0 * float(half_length_px))),
        int(round(2.0 * float(half_width_px))),
    )
    pygame.draw.rect(surface, RECTANGLE_FILL_RGB, rect, border_radius=6)
    pygame.draw.rect(surface, RECTANGLE_BORDER_RGB, rect, width=2, border_radius=6)


def _draw_demo_panel(
    *,
    surface,
    center_xy: Sequence[float],
    title: str,
    equation: str,
    caption: str,
    shape_exponent: float,
    font,
) -> None:
    _draw_vehicle(
        surface,
        center_xy=center_xy,
        half_length_px=VEHICLE_HALF_LENGTH_PX,
        half_width_px=VEHICLE_HALF_WIDTH_PX,
    )

    scale = _corner_touch_scale(float(shape_exponent))
    semi_axis_x_px = float(scale * VEHICLE_HALF_LENGTH_PX)
    semi_axis_y_px = float(scale * VEHICLE_HALF_WIDTH_PX)
    shape_points = _sample_superellipse_points(
        center_xy=center_xy,
        semi_axis_x_px=semi_axis_x_px,
        semi_axis_y_px=semi_axis_y_px,
        shape_exponent=float(shape_exponent),
    )
    pygame.draw.lines(surface, SHAPE_RGB, True, shape_points, width=4)

    title_surface = font.render(title, True, TEXT_RGB)
    equation_surface = font.render(equation, True, TEXT_RGB)
    caption_surface = font.render(caption, True, TEXT_RGB)
    title_x_px = int(round(float(center_xy[0]) - 0.5 * float(title_surface.get_width())))
    equation_x_px = int(round(float(center_xy[0]) - 0.5 * float(equation_surface.get_width())))
    caption_x_px = int(round(float(center_xy[0]) - 0.5 * float(caption_surface.get_width())))
    surface.blit(title_surface, (title_x_px, int(round(center_xy[1] - 145.0))))
    surface.blit(equation_surface, (equation_x_px, int(round(center_xy[1] - 118.0))))
    surface.blit(caption_surface, (caption_x_px, int(round(center_xy[1] + 138.0))))


def main() -> int:
    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode((WINDOW_WIDTH_PX, WINDOW_HEIGHT_PX))
    pygame.display.set_caption("Ellipse vs Superellipse Vehicle Bound")

    title_font = pygame.font.SysFont("Times New Roman", 14, bold=True)
    body_font = pygame.font.SysFont("Times New Roman", 14)

    left_center_xy = (0.28 * WINDOW_WIDTH_PX, 0.42 * WINDOW_HEIGHT_PX)
    right_center_xy = (0.72 * WINDOW_WIDTH_PX, 0.42 * WINDOW_HEIGHT_PX)

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False

        screen.fill(BACKGROUND_RGB)

        heading_surface = title_font.render(
            "Ellipse cuts corners more than a superellipse (n = 4)",
            True,
            TEXT_RGB,
        )
        screen.blit(
            heading_surface,
            (
                int(round(0.5 * (WINDOW_WIDTH_PX - heading_surface.get_width()))),
                28,
            ),
        )

        _draw_demo_panel(
            surface=screen,
            center_xy=left_center_xy,
            title="Ellipse Bound",
            equation="(x/a)^2 + (y/b)^2 = 1",
            caption="Figure 1: Rectangle bounded by an ellipse.",
            shape_exponent=2.0,
            font=body_font,
        )
        _draw_demo_panel(
            surface=screen,
            center_xy=right_center_xy,
            title="Superellipse Bound",
            equation="(|x|/a)^4 + (|y|/b)^4 = 1",
            caption="Figure 2: Rectangle bounded by a superellipse (n = 4).",
            shape_exponent=SUPERELLIPSE_EXPONENT,
            font=body_font,
        )

        note_surface = body_font.render(
            "Gray = vehicle rectangle, Red = enclosing shape. Press Esc or Q to quit.",
            True,
            TEXT_RGB,
        )
        screen.blit(
            note_surface,
            (
                int(round(0.5 * (WINDOW_WIDTH_PX - note_surface.get_width()))),
                WINDOW_HEIGHT_PX - 40,
            ),
        )

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

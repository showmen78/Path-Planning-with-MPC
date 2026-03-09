"""
Road geometry, rendering, and lane-center waypoint generation.

This module owns road-related logic so scenarios can switch between a straight
road and a 4-way intersection road without duplicating drawing logic.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pygame

from utility.rendering import world_to_screen


class RoadModel:
    """
    Intent:
        Encapsulate road geometry used by both rendering and planning support
        data (lane-center waypoints).

    Supported road types:
        - `straight`: one corridor with lane center waypoints on parallel lines.
        - `intersection_4way`: two perpendicular 4-lane corridors crossing at
          the center, with straight + left-turn + right-turn waypoint branches.

    Waypoint fields (all road types):
        - `position`: [x, y]
        - `lane_id`: 1-based lane index within the corridor cross section
        - `lane_width_m`: lane width
        - `direction`: lane travel direction label
    """

    def __init__(self) -> None:
        self._latest_lane_waypoints: List[Dict[str, object]] = []

    def get_latest_lane_waypoints(self) -> List[Dict[str, object]]:
        return deepcopy(self._latest_lane_waypoints)

    def _compute_visible_x_range_m(
        self,
        surface_width_px: int,
        camera_center_world: Tuple[float, float],
        pixels_per_meter: float,
        road_cfg: Mapping[str, Any],
    ) -> Tuple[float, float]:
        visible_half_width_m = 0.5 * float(surface_width_px) / max(1e-6, float(pixels_per_meter))
        x_margin_m = float(road_cfg.get("visible_x_margin_m", 20.0))
        x_start_m = float(camera_center_world[0]) - visible_half_width_m - x_margin_m
        x_end_m = float(camera_center_world[0]) + visible_half_width_m + x_margin_m
        return float(x_start_m), float(x_end_m)

    def _compute_visible_y_range_m(
        self,
        surface_height_px: int,
        camera_center_world: Tuple[float, float],
        pixels_per_meter: float,
        road_cfg: Mapping[str, Any],
    ) -> Tuple[float, float]:
        visible_half_height_m = 0.5 * float(surface_height_px) / max(1e-6, float(pixels_per_meter))
        y_margin_m = float(road_cfg.get("visible_y_margin_m", road_cfg.get("visible_x_margin_m", 20.0)))
        y_start_m = float(camera_center_world[1]) - visible_half_height_m - y_margin_m
        y_end_m = float(camera_center_world[1]) + visible_half_height_m + y_margin_m
        return float(y_start_m), float(y_end_m)

    @staticmethod
    def _sample_axis_points(start_m: float, end_m: float, spacing_m: float) -> List[float]:
        points: List[float] = []
        spacing_m = max(0.2, float(spacing_m))
        if end_m < start_m:
            start_m, end_m = end_m, start_m
        cursor_m = float(start_m)
        while cursor_m <= float(end_m) + 1e-9:
            points.append(float(cursor_m))
            cursor_m += spacing_m
        return points

    @staticmethod
    def _lane_direction_from_map(
        lane_id: int,
        default_direction: str,
        lane_direction_map: Mapping[str, Any] | None,
    ) -> str:
        if lane_direction_map is None:
            return str(default_direction)
        return str(
            lane_direction_map.get(str(int(lane_id)), lane_direction_map.get(int(lane_id), default_direction))
        )

    @staticmethod
    def _lane_progress_coordinate(waypoint: Mapping[str, object]) -> float:
        """
        Compute a scalar progress coordinate along the lane direction for sorting.

        Uses waypoint heading so the same logic works for straight/curved lanes.
        """

        position = waypoint.get("position", [0.0, 0.0])
        if not isinstance(position, Sequence) or len(position) < 2:
            return 0.0
        x_m = float(position[0])
        y_m = float(position[1])
        heading_rad = float(waypoint.get("heading_rad", 0.0))
        return float(x_m * math.cos(heading_rad) + y_m * math.sin(heading_rad))

    def _attach_next_waypoint_positions(self, waypoints: List[Dict[str, object]]) -> List[Dict[str, object]]:
        """
        Attach a `next` position field for each waypoint inside the same lane path.

        Grouping key keeps independent lane paths separated (lane, road, direction,
        maneuver/axis where available). The next pointer is stored as:
            waypoint["next"] = [x_next, y_next] or None
        """

        for waypoint in waypoints:
            waypoint["next"] = None
            waypoint["next_heading_rad"] = None
            # Ensure every lane-center waypoint explicitly carries heading.
            # If a specific builder omitted it, default to 0.0 so consumers can
            # always read `heading_rad` safely.
            if "heading_rad" not in waypoint:
                waypoint["heading_rad"] = 0.0

        groups: Dict[Tuple[object, object, object, object, object], List[int]] = {}
        for index, waypoint in enumerate(waypoints):
            key = (
                waypoint.get("lane_id"),
                waypoint.get("road_id"),
                waypoint.get("direction"),
                waypoint.get("maneuver", "straight"),
                waypoint.get("road_axis", "main"),
            )
            groups.setdefault(key, []).append(index)

        for indices in groups.values():
            if len(indices) <= 1:
                continue
            indices.sort(key=lambda idx: self._lane_progress_coordinate(waypoints[idx]))
            for cursor in range(len(indices) - 1):
                current_idx = indices[cursor]
                next_idx = indices[cursor + 1]
                next_position = waypoints[next_idx].get("position", [0.0, 0.0])
                if isinstance(next_position, Sequence) and len(next_position) >= 2:
                    waypoints[current_idx]["next"] = [float(next_position[0]), float(next_position[1])]
                    waypoints[current_idx]["next_heading_rad"] = float(waypoints[next_idx].get("heading_rad", 0.0))

        return waypoints

    def build_lane_center_waypoints(
        self,
        road_cfg: Mapping[str, Any],
        x_start_m: float,
        x_end_m: float,
    ) -> List[Dict[str, object]]:
        lane_count = max(1, int(road_cfg.get("lane_count", 3)))
        lane_width_m = float(road_cfg.get("lane_width_m", 4.0))
        road_width_m = float(lane_count * lane_width_m)
        center_y_m = float(road_cfg.get("center_y_m", 0.0))
        spacing_m = max(0.5, float(road_cfg.get("waypoint_spacing_m", 3.0)))
        road_direction = str(road_cfg.get("direction", "positive_x"))
        lane_direction_map = road_cfg.get("lane_directions", None)
        road_id = str(road_cfg.get("road_id", "road_main"))

        road_min_y_m = center_y_m - 0.5 * lane_count * lane_width_m
        x_points = self._sample_axis_points(float(x_start_m), float(x_end_m), spacing_m)

        waypoints: List[Dict[str, object]] = []
        for lane_index in range(lane_count):
            lane_id = lane_index + 1
            lane_center_y_m = road_min_y_m + (lane_index + 0.5) * lane_width_m
            lane_direction = self._lane_direction_from_map(
                lane_id=lane_id,
                default_direction=road_direction,
                lane_direction_map=lane_direction_map if isinstance(lane_direction_map, Mapping) else None,
            )
            heading_rad = 0.0 if "positive" in lane_direction or "east" in lane_direction else math.pi
            for x_m in x_points:
                waypoints.append(
                    {
                        "position": [float(x_m), float(lane_center_y_m)],
                        "lane_id": int(lane_id),
                        "lane_width_m": float(lane_width_m),
                        "road_id": str(road_id),
                        "road_width_m": float(road_width_m),
                        "direction": str(lane_direction),
                        "heading_rad": float(heading_rad),
                        "is_intersection": False,
                    }
                )
        return self._attach_next_waypoint_positions(waypoints)

    def build_curved_lane_center_waypoints(
        self,
        road_cfg: Mapping[str, Any],
        x_start_m: float,
        x_end_m: float,
    ) -> List[Dict[str, object]]:
        """
        Build lane-center waypoints for a sinusoidal curved road.

        The road centerline is:
            y_c(x) = center_y + A * sin(k * (x - x_ref) + phase)
        with k = 2*pi / wavelength.

        Lane center offsets are applied along the local normal direction so lane
        spacing remains approximately constant on the curve.
        """

        lane_count = max(1, int(road_cfg.get("lane_count", 4)))
        lane_width_m = float(road_cfg.get("lane_width_m", 3.66))
        road_width_m = float(lane_count * lane_width_m)
        spacing_m = max(0.5, float(road_cfg.get("waypoint_spacing_m", 1.0)))
        road_direction = str(road_cfg.get("direction", "positive_x"))
        lane_direction_map = road_cfg.get("lane_directions", None)
        road_id = str(road_cfg.get("road_id", "road_curved"))

        curve_cfg = dict(road_cfg.get("curve", {}))
        center_y_m = float(road_cfg.get("center_y_m", 0.0))
        amplitude_m = float(curve_cfg.get("amplitude_m", 1.5))
        wavelength_m = max(1e-6, float(curve_cfg.get("wavelength_m", 160.0)))
        x_reference_m = float(curve_cfg.get("x_reference_m", 0.0))
        phase_rad = float(curve_cfg.get("phase_rad", 0.0))
        wave_number = (2.0 * math.pi) / wavelength_m

        x_points = self._sample_axis_points(float(x_start_m), float(x_end_m), spacing_m)
        half_lane_count = 0.5 * float(lane_count)

        waypoints: List[Dict[str, object]] = []
        for x_m in x_points:
            curve_theta = wave_number * (float(x_m) - x_reference_m) + phase_rad
            centerline_y_m = center_y_m + amplitude_m * math.sin(curve_theta)
            dy_dx = amplitude_m * wave_number * math.cos(curve_theta)

            tangent_norm = math.hypot(1.0, dy_dx)
            tangent_x = 1.0 / max(1e-9, tangent_norm)
            tangent_y = dy_dx / max(1e-9, tangent_norm)
            normal_x = -tangent_y
            normal_y = tangent_x

            for lane_index in range(lane_count):
                lane_id = lane_index + 1
                lane_direction = self._lane_direction_from_map(
                    lane_id=lane_id,
                    default_direction=road_direction,
                    lane_direction_map=lane_direction_map if isinstance(lane_direction_map, Mapping) else None,
                )

                offset_m = (float(lane_index) + 0.5 - half_lane_count) * lane_width_m
                lane_x_m = float(x_m + offset_m * normal_x)
                lane_y_m = float(centerline_y_m + offset_m * normal_y)

                heading_rad = math.atan2(tangent_y, tangent_x)
                if "negative" in lane_direction or "west" in lane_direction:
                    heading_rad = (heading_rad + math.pi + math.pi) % (2.0 * math.pi) - math.pi

                waypoints.append(
                    {
                        "position": [lane_x_m, lane_y_m],
                        "lane_id": int(lane_id),
                        "lane_width_m": float(lane_width_m),
                        "road_id": str(road_id),
                        "road_width_m": float(road_width_m),
                        "direction": str(lane_direction),
                        "heading_rad": float(heading_rad),
                        "is_intersection": False,
                    }
                )
        return self._attach_next_waypoint_positions(waypoints)

    @staticmethod
    def _sample_quadratic_bezier(
        p0: Tuple[float, float],
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        sample_count: int,
    ) -> List[Tuple[float, float, float]]:
        points: List[Tuple[float, float, float]] = []
        n = max(2, int(sample_count))
        for i in range(n):
            t = float(i) / float(n - 1)
            one_t = 1.0 - t
            x_m = one_t * one_t * p0[0] + 2.0 * one_t * t * p1[0] + t * t * p2[0]
            y_m = one_t * one_t * p0[1] + 2.0 * one_t * t * p1[1] + t * t * p2[1]
            dx_dt = 2.0 * one_t * (p1[0] - p0[0]) + 2.0 * t * (p2[0] - p1[0])
            dy_dt = 2.0 * one_t * (p1[1] - p0[1]) + 2.0 * t * (p2[1] - p1[1])
            heading_rad = math.atan2(dy_dt, dx_dt) if (abs(dx_dt) + abs(dy_dt)) > 1e-9 else 0.0
            points.append((float(x_m), float(y_m), float(heading_rad)))
        return points

    def build_intersection_waypoints(
        self,
        road_cfg: Mapping[str, Any],
        x_start_m: float,
        x_end_m: float,
        y_start_m: float,
        y_end_m: float,
    ) -> List[Dict[str, object]]:
        lane_count = max(2, int(road_cfg.get("lane_count", 4)))
        lane_width_m = float(road_cfg.get("lane_width_m", 3.66))
        lanes_per_direction = max(1, int(road_cfg.get("lanes_per_direction", lane_count // 2)))
        lanes_per_direction = min(lanes_per_direction, lane_count)
        road_width_m = float(lane_count * lane_width_m)
        spacing_m = max(0.4, float(road_cfg.get("waypoint_spacing_m", 2.0)))
        intersection_half_m = float(road_cfg.get("intersection_half_size_m", 12.0))
        road_half_m = lanes_per_direction * lane_width_m

        # 2 lanes for each direction when lane_count=4.
        eastbound_y = [(-road_half_m + (idx + 0.5) * lane_width_m) for idx in range(lanes_per_direction)]
        westbound_y = [(road_half_m - (idx + 0.5) * lane_width_m) for idx in range(lanes_per_direction)]
        southbound_x = [(-road_half_m + (idx + 0.5) * lane_width_m) for idx in range(lanes_per_direction)]
        northbound_x = [(road_half_m - (idx + 0.5) * lane_width_m) for idx in range(lanes_per_direction)]

        x_points = self._sample_axis_points(float(x_start_m), float(x_end_m), spacing_m)
        y_points = self._sample_axis_points(float(y_start_m), float(y_end_m), spacing_m)

        waypoints: List[Dict[str, object]] = []

        def add_waypoint(
            x_m: float,
            y_m: float,
            lane_id: int,
            direction: str,
            heading_rad: float,
            axis: str,
            maneuver: str,
        ) -> None:
            waypoints.append(
                {
                    "position": [float(x_m), float(y_m)],
                    "lane_id": int(lane_id),
                    "lane_width_m": float(lane_width_m),
                    "road_id": (
                        "horizontal_road"
                        if axis == "horizontal"
                        else ("vertical_road" if axis == "vertical" else "intersection_connector")
                    ),
                    "road_width_m": float(road_width_m),
                    "direction": str(direction),
                    "heading_rad": float(heading_rad),
                    "road_axis": str(axis),
                    "maneuver": str(maneuver),
                    "is_intersection": bool(
                        abs(float(x_m)) <= float(intersection_half_m) + 1e-9
                        and abs(float(y_m)) <= float(intersection_half_m) + 1e-9
                    ),
                }
            )

        # Straight lane-center waypoints (similar to CARLA lane center traces).
        for idx, lane_y_m in enumerate(eastbound_y):
            lane_id = idx + 1
            for x_m in x_points:
                add_waypoint(x_m, lane_y_m, lane_id, "eastbound", 0.0, "horizontal", "straight")
        for idx, lane_y_m in enumerate(westbound_y):
            lane_id = lanes_per_direction + idx + 1
            for x_m in x_points:
                add_waypoint(x_m, lane_y_m, lane_id, "westbound", math.pi, "horizontal", "straight")
        for idx, lane_x_m in enumerate(northbound_x):
            lane_id = idx + 1
            for y_m in y_points:
                add_waypoint(lane_x_m, y_m, lane_id, "northbound", 0.5 * math.pi, "vertical", "straight")
        for idx, lane_x_m in enumerate(southbound_x):
            lane_id = lanes_per_direction + idx + 1
            for y_m in y_points:
                add_waypoint(lane_x_m, y_m, lane_id, "southbound", -0.5 * math.pi, "vertical", "straight")

        # Intersection connector waypoints for left/right turns (CARLA-like branch options).
        # Inbound from west (eastbound lanes).
        for idx in range(lanes_per_direction):
            start = (-intersection_half_m, eastbound_y[idx])
            right_end = (southbound_x[idx], -intersection_half_m)
            left_end = (northbound_x[idx], intersection_half_m)
            right_curve = self._sample_quadratic_bezier(start, (-intersection_half_m, -intersection_half_m), right_end, 12)
            left_curve = self._sample_quadratic_bezier(start, (intersection_half_m, intersection_half_m), left_end, 18)
            for x_m, y_m, heading in right_curve:
                add_waypoint(x_m, y_m, idx + 1, "eastbound", heading, "intersection", "right")
            for x_m, y_m, heading in left_curve:
                add_waypoint(x_m, y_m, idx + 1, "eastbound", heading, "intersection", "left")

        # Inbound from east (westbound lanes).
        for idx in range(lanes_per_direction):
            start = (intersection_half_m, westbound_y[idx])
            right_end = (northbound_x[idx], intersection_half_m)
            left_end = (southbound_x[idx], -intersection_half_m)
            right_curve = self._sample_quadratic_bezier(start, (intersection_half_m, intersection_half_m), right_end, 12)
            left_curve = self._sample_quadratic_bezier(start, (-intersection_half_m, -intersection_half_m), left_end, 18)
            lane_id = lanes_per_direction + idx + 1
            for x_m, y_m, heading in right_curve:
                add_waypoint(x_m, y_m, lane_id, "westbound", heading, "intersection", "right")
            for x_m, y_m, heading in left_curve:
                add_waypoint(x_m, y_m, lane_id, "westbound", heading, "intersection", "left")

        # Inbound from south (northbound lanes).
        for idx in range(lanes_per_direction):
            start = (northbound_x[idx], -intersection_half_m)
            right_end = (intersection_half_m, eastbound_y[idx])
            left_end = (-intersection_half_m, westbound_y[idx])
            right_curve = self._sample_quadratic_bezier(start, (intersection_half_m, -intersection_half_m), right_end, 12)
            left_curve = self._sample_quadratic_bezier(start, (-intersection_half_m, intersection_half_m), left_end, 18)
            for x_m, y_m, heading in right_curve:
                add_waypoint(x_m, y_m, idx + 1, "northbound", heading, "intersection", "right")
            for x_m, y_m, heading in left_curve:
                add_waypoint(x_m, y_m, idx + 1, "northbound", heading, "intersection", "left")

        # Inbound from north (southbound lanes).
        for idx in range(lanes_per_direction):
            start = (southbound_x[idx], intersection_half_m)
            right_end = (-intersection_half_m, westbound_y[idx])
            left_end = (intersection_half_m, eastbound_y[idx])
            right_curve = self._sample_quadratic_bezier(start, (-intersection_half_m, intersection_half_m), right_end, 12)
            left_curve = self._sample_quadratic_bezier(start, (intersection_half_m, -intersection_half_m), left_end, 18)
            lane_id = lanes_per_direction + idx + 1
            for x_m, y_m, heading in right_curve:
                add_waypoint(x_m, y_m, lane_id, "southbound", heading, "intersection", "right")
            for x_m, y_m, heading in left_curve:
                add_waypoint(x_m, y_m, lane_id, "southbound", heading, "intersection", "left")

        return self._attach_next_waypoint_positions(waypoints)

    def _draw_straight_road(
        self,
        surface: pygame.Surface,
        road_cfg: Mapping[str, Any],
        camera_center_world: Tuple[float, float],
        pixels_per_meter: float,
        screen_center: Tuple[float, float],
    ) -> None:
        width_px, _ = surface.get_size()
        lane_count = max(1, int(road_cfg.get("lane_count", 3)))
        lane_width_m = float(road_cfg.get("lane_width_m", 4.0))
        center_y_m = float(road_cfg.get("center_y_m", 0.0))
        road_min_y_m = center_y_m - 0.5 * lane_count * lane_width_m
        road_max_y_m = center_y_m + 0.5 * lane_count * lane_width_m

        road_color = tuple(road_cfg.get("road_color_rgb", [56, 56, 56]))
        boundary_color = tuple(road_cfg.get("boundary_color_rgb", [245, 245, 245]))
        lane_line_color = tuple(road_cfg.get("lane_line_color_rgb", [230, 230, 230]))

        x_start_m, x_end_m = self._compute_visible_x_range_m(
            surface_width_px=width_px,
            camera_center_world=camera_center_world,
            pixels_per_meter=pixels_per_meter,
            road_cfg=road_cfg,
        )

        p1 = world_to_screen(x_start_m, road_max_y_m, camera_center_world, pixels_per_meter, screen_center)
        p2 = world_to_screen(x_end_m, road_max_y_m, camera_center_world, pixels_per_meter, screen_center)
        p3 = world_to_screen(x_end_m, road_min_y_m, camera_center_world, pixels_per_meter, screen_center)
        p4 = world_to_screen(x_start_m, road_min_y_m, camera_center_world, pixels_per_meter, screen_center)
        pygame.draw.polygon(surface, road_color, [p1, p2, p3, p4])
        pygame.draw.line(surface, boundary_color, p1, p2, 3)
        pygame.draw.line(surface, boundary_color, p4, p3, 3)

        dash_length_m = 3.0
        dash_gap_m = 2.5
        for divider_idx in range(1, lane_count):
            lane_y = road_min_y_m + divider_idx * lane_width_m
            x_cursor_m = x_start_m
            while x_cursor_m < x_end_m:
                dash_end_m = min(x_end_m, x_cursor_m + dash_length_m)
                s = world_to_screen(x_cursor_m, lane_y, camera_center_world, pixels_per_meter, screen_center)
                e = world_to_screen(dash_end_m, lane_y, camera_center_world, pixels_per_meter, screen_center)
                pygame.draw.line(surface, lane_line_color, s, e, 2)
                x_cursor_m += dash_length_m + dash_gap_m

        self._latest_lane_waypoints = self.build_lane_center_waypoints(
            road_cfg=road_cfg,
            x_start_m=x_start_m,
            x_end_m=x_end_m,
        )

    def _draw_curved_road(
        self,
        surface: pygame.Surface,
        road_cfg: Mapping[str, Any],
        camera_center_world: Tuple[float, float],
        pixels_per_meter: float,
        screen_center: Tuple[float, float],
    ) -> None:
        """
        Draw a one-way curved road and generate curved lane-center waypoints.
        """

        width_px, _ = surface.get_size()
        lane_count = max(1, int(road_cfg.get("lane_count", 4)))
        lane_width_m = float(road_cfg.get("lane_width_m", 3.66))
        half_road_width_m = 0.5 * float(lane_count) * lane_width_m

        road_color = tuple(road_cfg.get("road_color_rgb", [56, 56, 56]))
        boundary_color = tuple(road_cfg.get("boundary_color_rgb", [245, 245, 245]))
        lane_line_color = tuple(road_cfg.get("lane_line_color_rgb", [230, 230, 230]))

        x_start_m, x_end_m = self._compute_visible_x_range_m(
            surface_width_px=width_px,
            camera_center_world=camera_center_world,
            pixels_per_meter=pixels_per_meter,
            road_cfg=road_cfg,
        )

        curve_cfg = dict(road_cfg.get("curve", {}))
        center_y_m = float(road_cfg.get("center_y_m", 0.0))
        amplitude_m = float(curve_cfg.get("amplitude_m", 1.5))
        wavelength_m = max(1e-6, float(curve_cfg.get("wavelength_m", 160.0)))
        x_reference_m = float(curve_cfg.get("x_reference_m", 0.0))
        phase_rad = float(curve_cfg.get("phase_rad", 0.0))
        wave_number = (2.0 * math.pi) / wavelength_m

        sample_step_m = max(0.25, float(curve_cfg.get("draw_step_m", 0.5)))
        x_points = self._sample_axis_points(float(x_start_m), float(x_end_m), sample_step_m)
        if len(x_points) < 2:
            self._latest_lane_waypoints = []
            return

        def centerline_with_frame(x_m: float) -> Tuple[float, float, float, float]:
            curve_theta = wave_number * (float(x_m) - x_reference_m) + phase_rad
            yc_m = center_y_m + amplitude_m * math.sin(curve_theta)
            dy_dx = amplitude_m * wave_number * math.cos(curve_theta)
            tangent_norm = math.hypot(1.0, dy_dx)
            tangent_x = 1.0 / max(1e-9, tangent_norm)
            tangent_y = dy_dx / max(1e-9, tangent_norm)
            normal_x = -tangent_y
            normal_y = tangent_x
            return float(yc_m), float(normal_x), float(normal_y), float(tangent_norm)

        left_boundary_world: List[Tuple[float, float]] = []
        right_boundary_world: List[Tuple[float, float]] = []
        for x_m in x_points:
            yc_m, nx_m, ny_m, _ = centerline_with_frame(x_m)
            left_boundary_world.append((float(x_m + half_road_width_m * nx_m), float(yc_m + half_road_width_m * ny_m)))
            right_boundary_world.append((float(x_m - half_road_width_m * nx_m), float(yc_m - half_road_width_m * ny_m)))

        left_boundary_px = [
            world_to_screen(px, py, camera_center_world, pixels_per_meter, screen_center)
            for (px, py) in left_boundary_world
        ]
        right_boundary_px = [
            world_to_screen(px, py, camera_center_world, pixels_per_meter, screen_center)
            for (px, py) in right_boundary_world
        ]

        road_polygon = left_boundary_px + list(reversed(right_boundary_px))
        if len(road_polygon) >= 3:
            pygame.draw.polygon(surface, road_color, road_polygon)
        if len(left_boundary_px) >= 2:
            pygame.draw.lines(surface, boundary_color, False, left_boundary_px, 3)
        if len(right_boundary_px) >= 2:
            pygame.draw.lines(surface, boundary_color, False, right_boundary_px, 3)

        dash_length_m = float(road_cfg.get("dash_length_m", 3.0))
        dash_gap_m = float(road_cfg.get("dash_gap_m", 2.5))
        dash_draw_count = max(1, int(round(dash_length_m / sample_step_m)))
        dash_skip_count = max(1, int(round(dash_gap_m / sample_step_m)))
        dash_cycle = dash_draw_count + dash_skip_count

        for divider_idx in range(1, lane_count):
            offset_m = -half_road_width_m + float(divider_idx) * lane_width_m
            divider_world: List[Tuple[float, float]] = []
            for x_m in x_points:
                yc_m, nx_m, ny_m, _ = centerline_with_frame(x_m)
                divider_world.append((float(x_m + offset_m * nx_m), float(yc_m + offset_m * ny_m)))
            divider_px = [
                world_to_screen(px, py, camera_center_world, pixels_per_meter, screen_center)
                for (px, py) in divider_world
            ]
            for seg_idx in range(max(0, len(divider_px) - 1)):
                if (seg_idx % dash_cycle) < dash_draw_count:
                    pygame.draw.line(surface, lane_line_color, divider_px[seg_idx], divider_px[seg_idx + 1], 2)

        self._latest_lane_waypoints = self.build_curved_lane_center_waypoints(
            road_cfg=road_cfg,
            x_start_m=x_start_m,
            x_end_m=x_end_m,
        )

    def _draw_intersection_road(
        self,
        surface: pygame.Surface,
        road_cfg: Mapping[str, Any],
        camera_center_world: Tuple[float, float],
        pixels_per_meter: float,
        screen_center: Tuple[float, float],
    ) -> None:
        width_px, height_px = surface.get_size()
        lane_count = max(2, int(road_cfg.get("lane_count", 4)))
        lanes_per_direction = max(1, int(road_cfg.get("lanes_per_direction", lane_count // 2)))
        lane_width_m = float(road_cfg.get("lane_width_m", 3.66))
        road_half_m = float(lanes_per_direction * lane_width_m)

        road_color = tuple(road_cfg.get("road_color_rgb", [56, 56, 56]))
        boundary_color = tuple(road_cfg.get("boundary_color_rgb", [245, 245, 245]))
        lane_line_color = tuple(road_cfg.get("lane_line_color_rgb", [230, 230, 230]))
        center_line_color = tuple(road_cfg.get("center_line_color_rgb", [230, 210, 70]))

        x_start_m, x_end_m = self._compute_visible_x_range_m(
            surface_width_px=width_px,
            camera_center_world=camera_center_world,
            pixels_per_meter=pixels_per_meter,
            road_cfg=road_cfg,
        )
        y_start_m, y_end_m = self._compute_visible_y_range_m(
            surface_height_px=height_px,
            camera_center_world=camera_center_world,
            pixels_per_meter=pixels_per_meter,
            road_cfg=road_cfg,
        )

        # Horizontal corridor polygon.
        hp1 = world_to_screen(x_start_m, road_half_m, camera_center_world, pixels_per_meter, screen_center)
        hp2 = world_to_screen(x_end_m, road_half_m, camera_center_world, pixels_per_meter, screen_center)
        hp3 = world_to_screen(x_end_m, -road_half_m, camera_center_world, pixels_per_meter, screen_center)
        hp4 = world_to_screen(x_start_m, -road_half_m, camera_center_world, pixels_per_meter, screen_center)
        pygame.draw.polygon(surface, road_color, [hp1, hp2, hp3, hp4])

        # Vertical corridor polygon.
        vp1 = world_to_screen(road_half_m, y_end_m, camera_center_world, pixels_per_meter, screen_center)
        vp2 = world_to_screen(road_half_m, y_start_m, camera_center_world, pixels_per_meter, screen_center)
        vp3 = world_to_screen(-road_half_m, y_start_m, camera_center_world, pixels_per_meter, screen_center)
        vp4 = world_to_screen(-road_half_m, y_end_m, camera_center_world, pixels_per_meter, screen_center)
        pygame.draw.polygon(surface, road_color, [vp1, vp2, vp3, vp4])

        # Outer boundaries.
        pygame.draw.line(surface, boundary_color, hp1, hp2, 3)
        pygame.draw.line(surface, boundary_color, hp4, hp3, 3)
        pygame.draw.line(surface, boundary_color, vp1, vp2, 3)
        pygame.draw.line(surface, boundary_color, vp4, vp3, 3)

        dash_length_m = 3.0
        dash_gap_m = 2.5
        for divider_idx in range(1, lane_count):
            y_line_m = -road_half_m + divider_idx * lane_width_m
            x_line_m = -road_half_m + divider_idx * lane_width_m
            is_center_divider = divider_idx == lanes_per_direction
            line_color = center_line_color if is_center_divider else lane_line_color
            line_width = 2 if not is_center_divider else 3

            # Horizontal divider at y = const.
            x_cursor_m = x_start_m
            while x_cursor_m < x_end_m:
                dash_end_m = min(x_end_m, x_cursor_m + dash_length_m)
                s = world_to_screen(x_cursor_m, y_line_m, camera_center_world, pixels_per_meter, screen_center)
                e = world_to_screen(dash_end_m, y_line_m, camera_center_world, pixels_per_meter, screen_center)
                pygame.draw.line(surface, line_color, s, e, line_width)
                x_cursor_m += dash_length_m + dash_gap_m

            # Vertical divider at x = const.
            y_cursor_m = y_start_m
            while y_cursor_m < y_end_m:
                dash_end_m = min(y_end_m, y_cursor_m + dash_length_m)
                s = world_to_screen(x_line_m, y_cursor_m, camera_center_world, pixels_per_meter, screen_center)
                e = world_to_screen(x_line_m, dash_end_m, camera_center_world, pixels_per_meter, screen_center)
                pygame.draw.line(surface, line_color, s, e, line_width)
                y_cursor_m += dash_length_m + dash_gap_m

        self._latest_lane_waypoints = self.build_intersection_waypoints(
            road_cfg=road_cfg,
            x_start_m=x_start_m,
            x_end_m=x_end_m,
            y_start_m=y_start_m,
            y_end_m=y_end_m,
        )

    def draw(
        self,
        surface: pygame.Surface,
        road_cfg: Mapping[str, Any],
        camera_center_world: Tuple[float, float],
        pixels_per_meter: float,
    ) -> None:
        width_px, height_px = surface.get_size()
        screen_center = (0.5 * width_px, 0.5 * height_px)
        road_type = str(road_cfg.get("type", "straight")).strip().lower()

        if road_type == "intersection_4way":
            self._draw_intersection_road(
                surface=surface,
                road_cfg=road_cfg,
                camera_center_world=camera_center_world,
                pixels_per_meter=pixels_per_meter,
                screen_center=screen_center,
            )
        elif road_type == "curved":
            self._draw_curved_road(
                surface=surface,
                road_cfg=road_cfg,
                camera_center_world=camera_center_world,
                pixels_per_meter=pixels_per_meter,
                screen_center=screen_center,
            )
        else:
            self._draw_straight_road(
                surface=surface,
                road_cfg=road_cfg,
                camera_center_world=camera_center_world,
                pixels_per_meter=pixels_per_meter,
                screen_center=screen_center,
            )

        if not bool(road_cfg.get("show_lane_waypoints", True)):
            return

        waypoint_color = tuple(road_cfg.get("waypoint_color_rgb", [70, 190, 90]))
        waypoint_radius_px = max(1, int(road_cfg.get("waypoint_radius_px", 2)))
        waypoint_stride = max(1, int(road_cfg.get("waypoint_draw_stride", 1)))
        for idx, waypoint in enumerate(self._latest_lane_waypoints):
            if idx % waypoint_stride != 0:
                continue
            position = waypoint.get("position", [0.0, 0.0])
            if not isinstance(position, Sequence) or len(position) < 2:
                continue
            px, py = world_to_screen(
                float(position[0]),
                float(position[1]),
                camera_center_world,
                pixels_per_meter,
                screen_center,
            )
            if -4 <= px <= width_px + 4 and -4 <= py <= height_px + 4:
                pygame.draw.circle(surface, waypoint_color, (px, py), waypoint_radius_px)

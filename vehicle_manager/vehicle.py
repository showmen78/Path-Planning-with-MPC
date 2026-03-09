"""
Vehicle manager core module.

This module defines the generic object model used by the simulation. The same
kinematic state definition is used for ego, surrounding vehicles, emergency
vehicles, and VRUs so the MPC interface remains consistent.

State vector (all movable objects):
    [x, y, v, psi]
Control vector (for kinematic bicycle objects):
    [a, delta]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import math

import pygame


@dataclass
class VehicleRenderSpec:
    """
    Intent:
        Store rendering geometry and color for a rectangular object.

    Fields:
        length_m:
            float [m], rectangle length in the local body x-direction.
        width_m:
            float [m], rectangle width in the local body y-direction.
        color_rgb:
            tuple[int, int, int], display color in pygame.
    """

    length_m: float
    width_m: float
    color_rgb: Tuple[int, int, int]


class Vehicle:
    """
    Intent:
        Represent one dynamic object (ego/surrounding/emergency/VRU) and expose
        a common state interface to the state manager, tracker, collision
        checker, and MPC.

    Model / method used:
        Non-linear kinematic bicycle model (CG-reference) with forward Euler
        integration. The position state [x, y] is treated as the vehicle center
        / CG point used by rendering and collision geometry.

        beta    = atan((l_r / L) * tan(delta))
        x_dot   = v cos(psi + beta)
        y_dot   = v sin(psi + beta)
        v_dot   = a
        psi_dot = (v / l_r) sin(beta)

    State and control definitions:
        current_state:
            list[float], shape (4), [x_m, y_m, v_mps, psi_rad]
        controls:
            acceleration_mps2 (a), steering_angle_rad (delta)

    Notes:
        - The simulation keeps the same state layout for all object classes.
        - For VRUs the same kinematic update is used as a simplification so the
          code path remains consistent; future scenarios can specialize this.
    """

    def __init__(
        self,
        vehicle_id: str,
        vehicle_type: str,
        object_class: str,
        initial_state: Sequence[float],
        wheelbase_m: float,
        min_velocity_mps: float,
        max_velocity_mps: float,
        max_acceleration_mps2: float,
        max_steer_rad: float,
        render_spec: VehicleRenderSpec,
    ) -> None:
        if len(initial_state) != 4:
            raise ValueError("initial_state must be [x, y, v, psi].")
        if wheelbase_m <= 0.0:
            raise ValueError("wheelbase_m must be > 0.")
        if max_velocity_mps <= min_velocity_mps:
            raise ValueError("max_velocity_mps must be > min_velocity_mps.")
        if max_acceleration_mps2 <= 0.0:
            raise ValueError("max_acceleration_mps2 must be > 0.")
        if max_steer_rad <= 0.0:
            raise ValueError("max_steer_rad must be > 0.")

        self.vehicle_id = str(vehicle_id)
        self.vehicle_type = str(vehicle_type)  # e.g. ego, obstacle, vru
        self.object_class = str(object_class)  # surrounding_vehicle, emergency_vehicle, vru

        self.current_state: List[float] = [float(value) for value in initial_state]
        self.wheelbase_m = float(wheelbase_m)
        self.min_velocity_mps = float(min_velocity_mps)
        self.max_velocity_mps = float(max_velocity_mps)
        self.max_acceleration_mps2 = float(max_acceleration_mps2)
        self.max_steer_rad = float(max_steer_rad)
        self.render_spec = render_spec

        self.acceleration_mps2 = 0.0
        self.steering_angle_rad = 0.0
        self.future_trajectory: List[List[float]] = []

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    @staticmethod
    def _wrap_angle(angle_rad: float) -> float:
        return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi

    def set_control(self, acceleration_mps2: float, steering_angle_rad: float) -> None:
        """
        Intent:
            Clamp and store the current commanded acceleration and steering.
        """

        self.acceleration_mps2 = self._clamp(
            float(acceleration_mps2), -self.max_acceleration_mps2, self.max_acceleration_mps2
        )
        self.steering_angle_rad = self._clamp(float(steering_angle_rad), -self.max_steer_rad, self.max_steer_rad)

    def step(self, dt_s: float) -> List[float]:
        """
        Intent:
            Propagate one time step using the kinematic bicycle model.

        Inputs:
            dt_s:
                float [s], integration time step.

        Output:
            list[float], updated state [x, y, v, psi].
        """

        if dt_s <= 0.0:
            raise ValueError("dt_s must be > 0.")

        x_m, y_m, v_mps, psi_rad = [float(value) for value in self.current_state]
        a_mps2 = float(self.acceleration_mps2)
        delta_rad = float(self.steering_angle_rad)

        # CG-reference kinematic bicycle model. We assume the CG is midway
        # between axles unless a more detailed axle split is introduced later.
        l_r_m = 0.5 * float(self.wheelbase_m)
        l_r_m = max(1e-9, l_r_m)
        beta_rad = math.atan((l_r_m / float(self.wheelbase_m)) * math.tan(delta_rad))

        x_dot = v_mps * math.cos(psi_rad + beta_rad)
        y_dot = v_mps * math.sin(psi_rad + beta_rad)
        v_dot = a_mps2
        psi_dot = (v_mps / l_r_m) * math.sin(beta_rad)

        x_next = x_m + dt_s * x_dot
        y_next = y_m + dt_s * y_dot
        v_next = self._clamp(v_mps + dt_s * v_dot, self.min_velocity_mps, self.max_velocity_mps)
        psi_next = self._wrap_angle(psi_rad + dt_s * psi_dot)

        self.current_state = [float(x_next), float(y_next), float(v_next), float(psi_next)]
        return [float(value) for value in self.current_state]

    def set_future_trajectory(self, future_trajectory: Sequence[Sequence[float]]) -> None:
        """
        Intent:
            Store future states for visualization/debugging.

        Meaning:
            - Ego: MPC-generated future states.
            - Other objects: tracker-predicted future states.
        """

        stored: List[List[float]] = []
        for state in future_trajectory:
            if len(state) < 4:
                continue
            stored.append([float(state[0]), float(state[1]), float(state[2]), float(state[3])])
        self.future_trajectory = stored

    def to_snapshot(self) -> Dict[str, object]:
        """
        Intent:
            Export a serializable snapshot consumed by the state manager,
            tracker, collision checker, and MPC.
        """

        x_m, y_m, v_mps, psi_rad = [float(value) for value in self.current_state]
        return {
            "vehicle_id": self.vehicle_id,
            "type": self.vehicle_type,
            "object_class": self.object_class,
            "current_state": [x_m, y_m, v_mps, psi_rad],
            "x": x_m,
            "y": y_m,
            "v": v_mps,
            "psi": psi_rad,
            "acceleration_mps2": float(self.acceleration_mps2),
            "steering_angle_rad": float(self.steering_angle_rad),
            "wheelbase_m": float(self.wheelbase_m),
            "min_velocity_mps": float(self.min_velocity_mps),
            "max_velocity_mps": float(self.max_velocity_mps),
            "max_acceleration_mps2": float(self.max_acceleration_mps2),
            "max_steer_rad": float(self.max_steer_rad),
            "length_m": float(self.render_spec.length_m),
            "width_m": float(self.render_spec.width_m),
            "color_rgb": tuple(self.render_spec.color_rgb),
            "future_trajectory": [list(state) for state in self.future_trajectory],
        }

    def draw(
        self,
        surface: pygame.Surface,
        pixels_per_meter: float,
        camera_center_world: Tuple[float, float],
        screen_center_px: Tuple[float, float],
    ) -> None:
        """
        Intent:
            Draw the object as a rotated rectangle using its current state.

        Inputs:
            pixels_per_meter:
                float [px/m], world-to-screen scale.
            camera_center_world:
                tuple[float, float], world point shown at screen center.
            screen_center_px:
                tuple[float, float], screen center in pixels.
        """

        x_m, y_m, _, psi_rad = self.current_state
        half_l = 0.5 * float(self.render_spec.length_m)
        half_w = 0.5 * float(self.render_spec.width_m)

        # Body-frame rectangle corners (rear/front not distinguished visually;
        # they are only geometric corners for rendering).
        body_corners = [
            (half_l, half_w),
            (half_l, -half_w),
            (-half_l, -half_w),
            (-half_l, half_w),
        ]

        cos_psi = math.cos(float(psi_rad))
        sin_psi = math.sin(float(psi_rad))
        cam_x, cam_y = float(camera_center_world[0]), float(camera_center_world[1])
        cx_px, cy_px = float(screen_center_px[0]), float(screen_center_px[1])

        polygon_points_px = []
        for local_x, local_y in body_corners:
            world_x = float(x_m) + local_x * cos_psi - local_y * sin_psi
            world_y = float(y_m) + local_x * sin_psi + local_y * cos_psi
            px = int(round(cx_px + (world_x - cam_x) * float(pixels_per_meter)))
            py = int(round(cy_px - (world_y - cam_y) * float(pixels_per_meter)))
            polygon_points_px.append((px, py))

        pygame.draw.polygon(surface, self.render_spec.color_rgb, polygon_points_px)
        pygame.draw.polygon(surface, (20, 20, 20), polygon_points_px, 2)

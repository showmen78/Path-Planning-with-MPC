"""
Vehicle factory utilities.

This module converts scenario YAML vehicle entries into `Vehicle` objects while
applying folder-level rendering/geometry defaults from `vehicle_manager/vehicle_manager.yaml`.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence

from .vehicle import Vehicle, VehicleRenderSpec


NON_EGO_DEFAULT_CLASS = "surrounding_vehicle"


def infer_object_class_from_entry(vehicle_cfg: Mapping[str, Any]) -> str:
    """
    Intent:
        Infer the semantic safety class for an object used in the PDF safety
        cost (surrounding/emergency/VRU).

    Logic:
        - If `object_class` is explicitly set in YAML, use it.
        - Else classify ego as `ego`.
        - Else classify all non-ego objects as `surrounding_vehicle` so current
          scenarios remain backward compatible without YAML edits.
    """

    if "object_class" in vehicle_cfg:
        return str(vehicle_cfg.get("object_class", NON_EGO_DEFAULT_CLASS)).strip().lower()
    vehicle_type = str(vehicle_cfg.get("type", "obstacle")).strip().lower()
    if vehicle_type == "ego":
        return "ego"
    return NON_EGO_DEFAULT_CLASS


def build_vehicles_from_config(
    config: Mapping[str, Any],
    vehicle_manager_cfg: Mapping[str, Any] | None = None,
) -> List[Vehicle]:
    """
    Intent:
        Build simulation objects from scenario configuration.

    Inputs:
        config:
            mapping containing `vehicles` list in scenario yaml.
        vehicle_manager_cfg:
            optional mapping of defaults from vehicle_manager folder yaml.

    Output:
        list[Vehicle], initialized dynamic objects.
    """

    manager_cfg = dict(vehicle_manager_cfg or {})
    defaults_cfg = dict(manager_cfg.get("defaults", {}))
    type_defaults_cfg = dict(manager_cfg.get("type_defaults", {}))
    scenario_constraints_cfg = dict(dict(config.get("mpc", {})).get("constraints", {}))
    scenario_min_velocity_mps = float(scenario_constraints_cfg.get("min_velocity_mps", 0.0))
    scenario_max_velocity_mps = float(scenario_constraints_cfg.get("max_velocity_mps", 15.0))
    scenario_max_acceleration_mps2 = float(scenario_constraints_cfg.get("max_acceleration_mps2", 3.0))
    scenario_min_steer_rad = float(scenario_constraints_cfg.get("min_steer_rad", -0.3))
    scenario_max_steer_rad = float(scenario_constraints_cfg.get("max_steer_rad", 0.3))
    scenario_vehicle_max_steer_rad = max(abs(scenario_min_steer_rad), abs(scenario_max_steer_rad))

    vehicle_cfg_list = list(config.get("vehicles", []))
    if len(vehicle_cfg_list) == 0:
        raise ValueError("Scenario config must contain a non-empty 'vehicles' list.")

    vehicles: List[Vehicle] = []
    for vehicle_cfg in vehicle_cfg_list:
        if not isinstance(vehicle_cfg, Mapping):
            raise TypeError("Each vehicles[] entry must be a mapping.")

        vehicle_type = str(vehicle_cfg.get("type", "obstacle"))
        object_class = infer_object_class_from_entry(vehicle_cfg)

        # Merge defaults in this order: global defaults -> per-type defaults -> entry overrides.
        per_type_defaults = dict(type_defaults_cfg.get(vehicle_type.lower(), {}))

        def _get(key: str, fallback: Any) -> Any:
            if key in vehicle_cfg:
                return vehicle_cfg[key]
            if key in per_type_defaults:
                return per_type_defaults[key]
            return defaults_cfg.get(key, fallback)

        render_spec = VehicleRenderSpec(
            length_m=float(_get("length_m", 4.5)),
            width_m=float(_get("width_m", 2.0)),
            color_rgb=tuple(_get("color_rgb", [120, 120, 120])),
        )

        # Constraint values are sourced from scenario.mpc.constraints to avoid
        # redundant limit definitions across multiple YAML files.
        min_velocity_mps = float(vehicle_cfg.get("min_velocity_mps", scenario_min_velocity_mps))
        max_velocity_mps = float(vehicle_cfg.get("max_velocity_mps", scenario_max_velocity_mps))
        max_acceleration_mps2 = float(vehicle_cfg.get("max_acceleration_mps2", scenario_max_acceleration_mps2))
        max_steer_rad = float(vehicle_cfg.get("max_steer_rad", scenario_vehicle_max_steer_rad))

        vehicle = Vehicle(
            vehicle_id=str(vehicle_cfg["vehicle_id"]),
            vehicle_type=str(vehicle_type),
            object_class=str(object_class),
            initial_state=[float(value) for value in vehicle_cfg.get("initial_state", [0.0, 0.0, 0.0, 0.0])],
            wheelbase_m=float(_get("wheelbase_m", 2.7)),
            min_velocity_mps=min_velocity_mps,
            max_velocity_mps=max_velocity_mps,
            max_acceleration_mps2=max_acceleration_mps2,
            max_steer_rad=max_steer_rad,
            render_spec=render_spec,
        )
        vehicle.set_control(
            acceleration_mps2=float(vehicle_cfg.get("initial_acceleration_mps2", 0.0)),
            steering_angle_rad=float(vehicle_cfg.get("initial_steering_angle_rad", 0.0)),
        )
        vehicles.append(vehicle)

    return vehicles


def find_ego_vehicle(vehicles: Sequence[Vehicle]) -> Vehicle:
    """
    Intent:
        Return the first vehicle whose `type` is `ego`.
    """

    for vehicle in vehicles:
        if str(vehicle.vehicle_type).lower() == "ego":
            return vehicle
    raise ValueError("No ego vehicle found in scenario vehicles list.")

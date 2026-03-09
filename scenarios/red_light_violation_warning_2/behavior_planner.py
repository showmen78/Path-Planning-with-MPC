"""
Behavior planner mock for red-light-violation-warning scenario.

Rule implemented:
- If ego is within trigger distance of the traffic-light proxy, set final
  destination to a stop point placed `stop_offset_before_light_m` before the
  light along the x-axis.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence
import math


class RedLightBehaviorPlanner:
    """Simple stateful behavior planner used by scenario-level hook."""

    def __init__(self, cfg: Mapping[str, object] | None = None) -> None:
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.traffic_light_vehicle_id = str(cfg.get("traffic_light_vehicle_id", "traffic_light_proxy"))
        self.trigger_distance_m = max(0.0, float(cfg.get("trigger_distance_m", 10.0)))
        self.stop_offset_before_light_m = max(0.0, float(cfg.get("stop_offset_before_light_m", 2.0)))

        # Traffic-signal render colors: default yellow before trigger, red after trigger.
        yellow_default = [235, 190, 40]
        red_default = [230, 40, 40]
        self.yellow_color_rgb = tuple(int(v) for v in cfg.get("yellow_color_rgb", yellow_default))
        self.red_color_rgb = tuple(int(v) for v in cfg.get("red_color_rgb", red_default))

        self._locked_stop_destination: List[float] | None = None

    @staticmethod
    def _find_object_by_id(
        object_snapshots: Sequence[Mapping[str, object]],
        vehicle_id: str,
    ) -> Mapping[str, object] | None:
        for snapshot in object_snapshots:
            if str(snapshot.get("vehicle_id", "")) == str(vehicle_id):
                return snapshot
        return None

    @property
    def is_triggered(self) -> bool:
        """True once the red-light stop target has been locked."""

        return self._locked_stop_destination is not None

    def get_traffic_light_color_rgb(self) -> tuple[int, int, int]:
        """Return traffic-light color based on trigger state."""

        return tuple(self.red_color_rgb if self.is_triggered else self.yellow_color_rgb)

    def update_final_destination_state(
        self,
        ego_snapshot: Mapping[str, object],
        object_snapshots: Sequence[Mapping[str, object]],
        current_final_destination_state: Sequence[float],
        simulation_time_s: float,
    ) -> List[float]:
        """
        Return the final destination for the current step.

        Once the stop destination is triggered, it stays locked.
        """

        _ = float(simulation_time_s)
        current = list(current_final_destination_state) if len(current_final_destination_state) >= 2 else [0.0, 0.0, 0.0, 0.0]
        while len(current) < 4:
            current.append(0.0)

        if not self.enabled:
            return current

        if self._locked_stop_destination is not None:
            return list(self._locked_stop_destination)

        traffic_light_snapshot = self._find_object_by_id(
            object_snapshots=object_snapshots,
            vehicle_id=self.traffic_light_vehicle_id,
        )
        if traffic_light_snapshot is None:
            return current

        ego_x_m = float(ego_snapshot.get("x", 0.0))
        light_x_m = float(traffic_light_snapshot.get("x", 0.0))

        # Trigger when ego is within configured x-distance of the light.
        x_distance_to_light_m = abs(light_x_m - ego_x_m)
        if x_distance_to_light_m > self.trigger_distance_m:
            return current

        # Place stop point before the light along approach direction.
        # If ego is left of light (moving +x), stop at x = light_x - offset.
        # If ego is right of light (moving -x), stop at x = light_x + offset.
        approach_sign = 1.0 if ego_x_m <= light_x_m else -1.0
        stop_x_m = float(light_x_m - approach_sign * self.stop_offset_before_light_m)

        stop_destination = [
            float(stop_x_m),
            float(current[1]),
            0.0,
            float(current[3]),
        ]

        self._locked_stop_destination = list(stop_destination)
        return stop_destination

"""
Behavior planner mock for VRU scenario.

Rule implemented:
- When ego crosses trigger x-position (default: x >= -40 m), set final
  destination to a fixed stop point.
"""

from __future__ import annotations

from typing import List, Mapping, Sequence


class VRUBehaviorPlanner:
    """Simple stateful planner for scenario-level destination override."""

    def __init__(self, cfg: Mapping[str, object] | None = None) -> None:
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.trigger_x_m = float(cfg.get("trigger_x_m", -40.0))
        self.trigger_when_ego_x_ge = bool(cfg.get("trigger_when_ego_x_ge", True))

        dest = list(cfg.get("trigger_destination_state", [-15.0, -1.0, 0.0, 0.0]))
        while len(dest) < 4:
            dest.append(0.0)
        self.trigger_destination_state = [
            float(dest[0]),
            float(dest[1]),
            float(dest[2]),
            float(dest[3]),
        ]

        self._locked_destination: List[float] | None = None

    @property
    def is_triggered(self) -> bool:
        return self._locked_destination is not None

    def update_final_destination_state(
        self,
        ego_snapshot: Mapping[str, object],
        object_snapshots: Sequence[Mapping[str, object]],
        current_final_destination_state: Sequence[float],
        simulation_time_s: float,
    ) -> List[float]:
        """
        Return final destination for this step.

        Once triggered, destination remains locked.
        """

        _ = object_snapshots
        _ = simulation_time_s

        current = list(current_final_destination_state) if len(current_final_destination_state) >= 2 else [0.0, 0.0, 0.0, 0.0]
        while len(current) < 4:
            current.append(0.0)

        if not self.enabled:
            return current

        if self._locked_destination is not None:
            return list(self._locked_destination)

        ego_x_m = float(ego_snapshot.get("x", 0.0))
        triggered_now = (ego_x_m >= self.trigger_x_m) if self.trigger_when_ego_x_ge else (ego_x_m <= self.trigger_x_m)
        if not triggered_now:
            return current

        self._locked_destination = list(self.trigger_destination_state)
        return list(self._locked_destination)

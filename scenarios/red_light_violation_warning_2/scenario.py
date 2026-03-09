"""
Red-light-violation-warning scenario.

This scenario reuses Scenario4 behavior and adds a simple behavior-planner hook
that can modify the final destination near the traffic light.
"""

from __future__ import annotations

import os
from typing import List, Mapping, Sequence

from scenarios.scenario4.scenario import Scenario4
from .behavior_planner import RedLightBehaviorPlanner


class RedLightViolationWarningScenario(Scenario4):
    """Scenario variant with custom map geometry and behavior-planner hook."""

    def __init__(self, scenario_yaml_path: str | None = None) -> None:
        default_yaml_path = os.path.join(os.path.dirname(__file__), "scenario.yaml")
        super().__init__(scenario_yaml_path=scenario_yaml_path or default_yaml_path)

        cfg = self.get_config()
        behavior_cfg = dict(cfg.get("behavior_planner", {}))
        self._behavior_planner = RedLightBehaviorPlanner(cfg=behavior_cfg)

    def get_vehicle_color_overrides(
        self,
        simulation_time_s: float,
    ) -> Mapping[str, Sequence[int]]:
        """
        Optional main-loop hook:
        return per-vehicle render color overrides for this frame.
        """

        _ = float(simulation_time_s)
        return {
            str(self._behavior_planner.traffic_light_vehicle_id): tuple(self._behavior_planner.get_traffic_light_color_rgb()),
        }

    def get_final_destination_state(
        self,
        ego_snapshot: Mapping[str, object],
        object_snapshots: Sequence[Mapping[str, object]],
        current_final_destination_state: Sequence[float],
        simulation_time_s: float,
    ) -> List[float]:
        """
        Optional main-loop hook:
        return potentially-updated final destination state.
        """

        return self._behavior_planner.update_final_destination_state(
            ego_snapshot=ego_snapshot,
            object_snapshots=object_snapshots,
            current_final_destination_state=current_final_destination_state,
            simulation_time_s=simulation_time_s,
        )


def create_scenario() -> RedLightViolationWarningScenario:
    return RedLightViolationWarningScenario()

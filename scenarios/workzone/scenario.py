"""
Workzone scenario on a curved 4-lane road.

This scenario reuses the curved-road rolling-goal behavior from scenario4 and
only swaps in a dedicated YAML setup for the workzone layout.
"""

from __future__ import annotations

import os

from scenarios.scenario4.scenario import Scenario4 as BaseScenario4
from .behavior_planner import WorkzoneBehaviorPlanner


class WorkzoneScenario(BaseScenario4):
    """Curved-road workzone scenario with local configuration."""

    def __init__(self, scenario_yaml_path: str | None = None) -> None:
        default_yaml_path = os.path.join(os.path.dirname(__file__), "scenario.yaml")
        super().__init__(scenario_yaml_path=scenario_yaml_path or default_yaml_path)
        cfg = self.get_config()
        behavior_cfg = dict(cfg.get("behavior_planner", {}))
        self._behavior_planner = WorkzoneBehaviorPlanner(cfg=behavior_cfg)

    def get_final_destination_state(
        self,
        ego_snapshot: dict[str, object],
        object_snapshots: list[dict[str, object]],
        current_final_destination_state: list[float] | tuple[float, ...],
        simulation_time_s: float,
    ) -> list[float]:
        return self._behavior_planner.update_final_destination_state(
            ego_snapshot=ego_snapshot,
            object_snapshots=object_snapshots,
            current_final_destination_state=current_final_destination_state,
            simulation_time_s=simulation_time_s,
            lane_center_waypoints=self.get_latest_lane_waypoints(),
        )


def create_scenario() -> WorkzoneScenario:
    return WorkzoneScenario()

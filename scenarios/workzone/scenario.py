"""
Workzone scenario on a curved 4-lane road.

This scenario now uses the shared rolling-goal logic without any scenario-local
behavior planner overriding destination updates.
"""

from __future__ import annotations

import os

from scenarios.scenario4.scenario import Scenario4 as BaseScenario4


class WorkzoneScenario(BaseScenario4):
    """Curved-road workzone scenario with local configuration only."""

    def __init__(self, scenario_yaml_path: str | None = None) -> None:
        default_yaml_path = os.path.join(os.path.dirname(__file__), "scenario.yaml")
        super().__init__(scenario_yaml_path=scenario_yaml_path or default_yaml_path)


def create_scenario() -> WorkzoneScenario:
    return WorkzoneScenario()

"""
Red-light-violation-warning scenario.

This scenario keeps its custom YAML geometry but no longer uses a scenario-local
behavior planner to rewrite destination updates.
"""

from __future__ import annotations

import os

from scenarios.scenario4.scenario import Scenario4


class RedLightViolationWarningScenario(Scenario4):
    """Scenario variant with custom map geometry only."""

    def __init__(self, scenario_yaml_path: str | None = None) -> None:
        default_yaml_path = os.path.join(os.path.dirname(__file__), "scenario.yaml")
        super().__init__(scenario_yaml_path=scenario_yaml_path or default_yaml_path)


def create_scenario() -> RedLightViolationWarningScenario:
    return RedLightViolationWarningScenario()

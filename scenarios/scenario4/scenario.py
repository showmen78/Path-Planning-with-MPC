"""
Scenario 4: curved 4-lane road scenario.

This scenario now relies on the shared rolling-goal base for temporary-
destination updates. Scenario-local code no longer modifies the rolling
temporary destination.
"""

from __future__ import annotations

import os

from scenarios.rolling_goal_base import RollingGoalScenario


class Scenario4(RollingGoalScenario):
    """Curved-road scenario using the shared rolling-goal destination logic."""

    def __init__(self, scenario_yaml_path: str | None = None) -> None:
        default_yaml_path = os.path.join(os.path.dirname(__file__), "scenario.yaml")
        super().__init__(scenario_yaml_path=scenario_yaml_path or default_yaml_path)


def create_scenario() -> Scenario4:
    return Scenario4()

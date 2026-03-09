"""
Scenario package loader for MPC_custom.

`main.py` uses this helper so scenarios can be switched by name:
    python main.py scenario1
    python main.py scenario2
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple


def load_scenario_by_name(scenario_name: str) -> Tuple[object, Dict[str, Any]]:
    """
    Intent:
        Dynamically import a scenario module and return the instantiated scenario
        object plus its configuration dictionary.
    """

    sanitized = str(scenario_name).strip()
    if len(sanitized) == 0:
        raise ValueError("scenario_name cannot be empty.")

    module_name = f"scenarios.{sanitized}.scenario"
    scenario_module = import_module(module_name)
    if not hasattr(scenario_module, "create_scenario"):
        raise AttributeError(f"{module_name} must define create_scenario().")

    scenario_object = scenario_module.create_scenario()
    if not hasattr(scenario_object, "get_config"):
        raise AttributeError(f"{module_name} scenario object must define get_config().")
    scenario_cfg = scenario_object.get_config()
    if not isinstance(scenario_cfg, dict):
        raise TypeError(f"{module_name}.get_config() must return dict.")
    return scenario_object, scenario_cfg

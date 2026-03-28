"""
Helpers for loading CARLA scenario definitions from YAML files.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from utility import load_yaml_file


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARLA_SCENARIO_DIR = os.path.join(PROJECT_ROOT, "carla_scenario")


def _candidate_file_names(name: str) -> List[str]:
    candidate = name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"
    return [candidate, os.path.splitext(candidate)[0] + ".yml"]


def get_scenario_path(name: str) -> str:
    """
    Resolve a scenario name to a YAML file anywhere under `carla_scenario/`.
    """

    for root, _, filenames in os.walk(CARLA_SCENARIO_DIR):
        for candidate in _candidate_file_names(name):
            if candidate in filenames:
                return os.path.join(root, candidate)
    return os.path.join(CARLA_SCENARIO_DIR, f"{name}.yaml")


def list_available_scenarios() -> List[str]:
    """
    Return all available CARLA scenario names without file extensions.
    """

    if not os.path.isdir(CARLA_SCENARIO_DIR):
        return []

    names: List[str] = []
    for root, _, filenames in os.walk(CARLA_SCENARIO_DIR):
        for entry in sorted(filenames):
            if entry.endswith((".yaml", ".yml")):
                names.append(os.path.splitext(entry)[0])
    return sorted(set(names))


def load_carla_scenario(name: str) -> Dict[str, Any]:
    """
    Load a CARLA scenario YAML file by scenario name.
    """

    path = get_scenario_path(name)
    if not os.path.exists(path):
        available = ", ".join(list_available_scenarios()) or "<none>"
        raise FileNotFoundError(
            f"CARLA scenario '{name}' was not found at {path}. "
            f"Available scenarios: {available}"
        )

    payload = load_yaml_file(path)
    payload.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    payload.setdefault("carla", {})
    return payload

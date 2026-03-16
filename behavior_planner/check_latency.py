"""
One-off latency probe for the LLM behavior planner.

This utility builds a real prompt from a scenario's initial state, sends a
single request to the configured OpenAI model, and prints the round-trip
latency plus the response text.

Run:
    conda run -n mpc_custom python behavior_planner/check_latency.py --scenario scenario4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, Mapping

import pygame

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from behavior_planner import BehaviorPlannerAPIClient, BehaviorPlannerPromptBuilder
from main import _compose_runtime_config
from scenarios import load_scenario_by_name
from vehicle_manager import build_vehicles_from_config, find_ego_vehicle


def _build_demo_prompt(scenario_name: str) -> tuple[str, str, Dict[str, Any]]:
    scenario_handler, scenario_config = load_scenario_by_name(scenario_name)
    config = _compose_runtime_config(scenario_config)

    road_cfg = dict(config.get("road", {}))
    wnd_cfg = dict(config.get("window", {}))
    vehicle_manager_cfg = dict(config.get("vehicle_manager", {}))
    mpc_cfg = dict(config.get("mpc", {}))

    pygame.init()
    surface = pygame.Surface(
        (
            int(wnd_cfg.get("width_px", 1500)),
            int(wnd_cfg.get("height_px", 900)),
        )
    )
    camera_center_world = (
        float(road_cfg.get("camera_center_x_m", 0.0)),
        float(road_cfg.get("center_y_m", 0.0)),
    )
    pixels_per_meter = float(wnd_cfg.get("pixels_per_meter", 14.0))
    scenario_handler.draw_road(
        surface=surface,
        road_cfg=road_cfg,
        camera_center_world=camera_center_world,
        pixels_per_meter=pixels_per_meter,
        world_to_screen_fn=None,
    )
    lane_center_waypoints = list(scenario_handler.get_latest_lane_waypoints())

    vehicles = build_vehicles_from_config(config=config, vehicle_manager_cfg=vehicle_manager_cfg)
    ego_vehicle = find_ego_vehicle(vehicles)
    ego_snapshot = ego_vehicle.to_snapshot()
    object_snapshots = [
        vehicle.to_snapshot()
        for vehicle in vehicles
        if str(vehicle.vehicle_type).lower() != "ego"
    ]

    prompt_builder = BehaviorPlannerPromptBuilder()
    system_instruction = prompt_builder.load_system_instruction()
    prompt = prompt_builder.build_prompt(
        ego_snapshot=ego_snapshot,
        destination_state=list(config.get("destination", [50.0, 0.0, 0.0, 0.0])),
        temporary_destination_state=list(config.get("destination", [50.0, 0.0, 0.0, 0.0])),
        lane_center_waypoints=lane_center_waypoints,
        object_snapshots=object_snapshots,
        road_cfg=road_cfg,
        v2x_broadcasts=[],
        ego_vehicle_id="Ego01",
        mpc_constraints=dict(mpc_cfg.get("constraints", {})),
    )
    runtime_cfg = dict(mpc_cfg.get("behavior_planner_runtime", {}))
    return system_instruction, prompt, runtime_cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure one behavior-planner API call latency.")
    parser.add_argument("--scenario", default="scenario4", help="Scenario name used to build the demo prompt.")
    args = parser.parse_args()

    system_instruction, prompt, runtime_cfg = _build_demo_prompt(str(args.scenario))
    client = BehaviorPlannerAPIClient(
        api_key_env_var=str(runtime_cfg.get("api_key_env_var", "OPENAI_API_KEY")),
        model=str(runtime_cfg.get("model", "gpt-4o")),
        temperature=float(runtime_cfg.get("temperature", 0.0)),
        request_timeout_s=float(runtime_cfg.get("request_timeout_s", 30.0)),
        max_output_tokens=int(runtime_cfg.get("max_output_tokens", 300)),
        enabled=bool(runtime_cfg.get("api_enabled", True)),
    )

    if not client.is_ready():
        print("Behavior-planner API client is not ready. Check OPENAI_API_KEY in .env.")
        return 1

    print(f"[LATENCY CHECK] scenario={args.scenario}")
    print("[LATENCY CHECK] prompt:")
    print(prompt)

    start_s = time.perf_counter()
    response_text, _response_id = client.request_decision(
        system_instruction=system_instruction,
        prompt=prompt,
    )
    latency_s = max(0.0, time.perf_counter() - start_s)

    print(f"[LATENCY CHECK] latency={latency_s:.3f}s")
    print("[LATENCY CHECK] response:")
    print(response_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

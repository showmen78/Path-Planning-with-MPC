"""
One-off latency probe for the LLM behavior planner.

This planning-only utility sends one compact sample prompt, prints the
round-trip latency, and shows the raw JSON response.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from behavior_planner import BehaviorPlannerAPIClient, BehaviorPlannerPromptBuilder
from main import load_mpc_config
from test import build_sample_prompt


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure one behavior-planner API call latency.")
    parser.add_argument(
        "--request-id",
        type=int,
        default=1,
        help="Request index used to build the compact sample prompt.",
    )
    args = parser.parse_args()

    runtime_cfg = dict(load_mpc_config().get("behavior_planner_runtime", {}))
    prompt_builder = BehaviorPlannerPromptBuilder()
    system_instruction = prompt_builder.load_system_instruction()
    prompt = build_sample_prompt(max(0, int(args.request_id) - 1))
    client = BehaviorPlannerAPIClient(
        api_key_env_var=str(runtime_cfg.get("api_key_env_var", "OPENAI_API_KEY")),
        model=str(runtime_cfg.get("model", "gpt-4o")),
        temperature=float(runtime_cfg.get("temperature", 0.0)),
        request_timeout_s=float(runtime_cfg.get("request_timeout_s", 30.0)),
        max_output_tokens=int(runtime_cfg.get("max_output_tokens", 80)),
        enabled=bool(runtime_cfg.get("api_enabled", True)),
    )

    if not client.is_ready():
        print("Behavior-planner API client is not ready. Set OPENAI_API_KEY in .env.")
        return 1

    started_s = time.perf_counter()
    response_text, response_id = client.request_decision(
        system_instruction=system_instruction,
        prompt=prompt,
    )
    latency_s = max(0.0, time.perf_counter() - started_s)

    print(f"[LATENCY CHECK] latency={latency_s:.3f}s response_id={response_id}")
    print("[LATENCY CHECK] prompt:")
    print(prompt)
    print("[LATENCY CHECK] response:")
    print(response_text)
    print("[LATENCY CHECK] summary:")
    print(f"mean_latency_s={statistics.mean([latency_s]):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

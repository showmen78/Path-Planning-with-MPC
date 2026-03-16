"""
Standalone behavior-planner API probe.

This script sends compact sample prompts in the same format used by the
project's behavior planner. The system instruction is loaded once, then the
script sends prompt-only requests sequentially. Each loop waits for the
response before sending the next request and prints the measured API latency.

Run:
    python test.py
    python test.py --duration-s 20
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from typing import List

import yaml

from behavior_planner import BehaviorPlannerAPIClient, BehaviorPlannerPromptBuilder


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MPC_CONFIG_PATH = os.path.join(PROJECT_ROOT, "MPC", "mpc.yaml")


def load_behavior_runtime_cfg() -> dict:
    with open(MPC_CONFIG_PATH, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    mpc_cfg = dict(payload.get("mpc", {}))
    return dict(mpc_cfg.get("behavior_planner_runtime", {}))


def build_sample_prompt(request_index: int) -> str:
    """
    Build a compact sample prompt in the same format used by the live project.

    The values change slightly per request so repeated calls are not identical.
    """

    ego_x_m = -80.0 + 1.5 * float(request_index)
    obstacle_x_m = -42.0 + 0.6 * float(request_index)
    adjacent_x_m = -18.0 + 0.3 * float(request_index)

    return (
        f"ID:[{request_index + 1}]\n\n"
        f"Ego01:[{ego_x_m:.2f},-1.83,6.0,0.0,-3]\n\n"
        "ROUTE:[1,0,1]\n\n"
        "PREV:[LANE_KEEP]\n\n"
        f"V101:[{obstacle_x_m:.2f},-1.83,2.0,0.0,-3,KC]\n\n"
        f"V205:[{adjacent_x_m:.2f},1.83,7.0,0.0,-2,KC]"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Send sequential behavior-planner test prompts.")
    parser.add_argument(
        "--duration-s",
        type=float,
        default=15.0,
        help="Total wall-clock duration for the request loop.",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print each sample prompt before sending it.",
    )
    args = parser.parse_args()

    runtime_cfg = load_behavior_runtime_cfg()
    prompt_builder = BehaviorPlannerPromptBuilder()
    system_instruction = prompt_builder.load_system_instruction()
    client = BehaviorPlannerAPIClient(
        api_key_env_var=str(runtime_cfg.get("api_key_env_var", "OPENAI_API_KEY")),
        model=str(runtime_cfg.get("model", "gpt-4o")),
        temperature=float(runtime_cfg.get("temperature", 0.0)),
        request_timeout_s=float(runtime_cfg.get("request_timeout_s", 30.0)),
        max_output_tokens=int(runtime_cfg.get("max_output_tokens", 300)),
        enabled=bool(runtime_cfg.get("api_enabled", True)),
    )

    if not client.is_ready():
        print("Behavior-planner API client is not ready. Set OPENAI_API_KEY in .env.")
        return 1

    duration_s = max(0.1, float(args.duration_s))
    end_time_s = time.perf_counter() + duration_s
    latencies_s: List[float] = []
    request_index = 0

    print(f"[TEST] duration={duration_s:.2f}s model={client.model}")
    print("[TEST] system instruction is sent only on the first request.")

    while time.perf_counter() < end_time_s:
        prompt = build_sample_prompt(request_index=request_index)
        if bool(args.print_prompt):
            print("\n[TEST PROMPT]")
            print(prompt)

        started_s = time.perf_counter()
        response_text, response_id = client.request_decision(
            system_instruction=system_instruction,
            prompt=prompt,
        )
        latency_s = max(0.0, time.perf_counter() - started_s)
        latencies_s.append(latency_s)

        print(
            f"\n[TEST RESPONSE] request={request_index + 1} "
            f"latency={latency_s:.3f}s response_id={response_id}"
        )
        print(response_text)

        expected_id = str(request_index + 1)
        if f'"id": "{expected_id}"' not in response_text and f'"id":"{expected_id}"' not in response_text:
            print(f"[TEST WARN] Response did not echo the expected id={expected_id}.")

        request_index += 1

    if len(latencies_s) > 0:
        print("\n[TEST SUMMARY]")
        print(f"requests_sent={len(latencies_s)}")
        print(f"min_latency_s={min(latencies_s):.3f}")
        print(f"max_latency_s={max(latencies_s):.3f}")
        print(f"mean_latency_s={statistics.mean(latencies_s):.3f}")
        print(f"median_latency_s={statistics.median(latencies_s):.3f}")
        if len(latencies_s) > 1:
            print(f"mean_latency_requests_2plus_s={statistics.mean(latencies_s[1:]):.3f}")
    else:
        print("\n[TEST SUMMARY]")
        print("requests_sent=0")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

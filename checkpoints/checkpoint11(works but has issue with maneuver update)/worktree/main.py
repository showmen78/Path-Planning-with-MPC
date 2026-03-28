"""
Planning-only entrypoint for CARLA integration work.

This branch no longer contains the old pygame/scenario simulation runtime.
Instead, it keeps the MPC planner, behavior-planner modules, and lightweight
helpers for loading configuration and constructing planning objects.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Mapping
import glob
import os
import platform
import socket
import subprocess
import sys
import time

from MPC import MPC
from behavior_planner import BehaviorPlannerAPIClient, BehaviorPlannerPromptBuilder
from carla_scenario import list_available_scenarios, load_carla_scenario
from utility import Tracker, load_yaml_file


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MPC_CONFIG_PATH = os.path.join(PROJECT_ROOT, "MPC", "mpc.yaml")
TRACKER_CONFIG_PATH = os.path.join(PROJECT_ROOT, "utility", "tracker.yaml")
DEFAULT_CARLA_ROOT = os.environ.get("CARLA_ROOT", "/home/umd-user/carla_source/carla")


def load_mpc_config(config_path: str = MPC_CONFIG_PATH) -> Dict[str, Any]:
    """Load the MPC configuration tree from YAML."""

    payload = load_yaml_file(config_path)
    return dict(payload.get("mpc", payload))


def load_tracker_config(config_path: str = TRACKER_CONFIG_PATH) -> Dict[str, Any]:
    """Load the tracker configuration tree from YAML."""

    payload = load_yaml_file(config_path)
    return dict(payload.get("tracker", payload))


def build_mpc_planner(
    mpc_cfg: Mapping[str, Any] | None = None,
    road_cfg: Mapping[str, Any] | None = None,
) -> MPC:
    """
    Build the live MPC planner.

    The caller is expected to inject CARLA-specific ego geometry and any
    runtime constraint overrides into `mpc_cfg` before calling this helper.
    """

    if mpc_cfg is None:
        mpc_cfg = load_mpc_config()
    return MPC(mpc_cfg=dict(mpc_cfg), road_cfg=dict(road_cfg or {}))


def build_tracker(tracker_cfg: Mapping[str, Any] | None = None) -> Tracker:
    """Build the polynomial motion tracker used for surrounding vehicles."""

    if tracker_cfg is None:
        tracker_cfg = load_tracker_config()
    return Tracker(tracker_cfg=dict(tracker_cfg))


def build_behavior_planner_prompt_builder() -> BehaviorPlannerPromptBuilder:
    """Create the compact prompt builder used by the behavior planner."""

    return BehaviorPlannerPromptBuilder()


def build_behavior_planner_api_client(
    behavior_runtime_cfg: Mapping[str, Any] | None = None,
) -> BehaviorPlannerAPIClient:
    """Create the OpenAI API client for the behavior planner."""

    runtime_cfg = dict(behavior_runtime_cfg or load_mpc_config().get("behavior_planner_runtime", {}))
    return BehaviorPlannerAPIClient(
        api_key_env_var=str(runtime_cfg.get("api_key_env_var", "OPENAI_API_KEY")),
        model=str(runtime_cfg.get("model", "gpt-4o")),
        temperature=float(runtime_cfg.get("temperature", 0.0)),
        request_timeout_s=float(runtime_cfg.get("request_timeout_s", 30.0)),
        max_output_tokens=int(runtime_cfg.get("max_output_tokens", 80)),
        enabled=bool(runtime_cfg.get("api_enabled", True)),
    )


def _get_carla_egg_glob(carla_root: str) -> str:
    machine = platform.machine().lower()
    if sys.platform.startswith("linux"):
        platform_tag = "linux-x86_64" if machine in {"x86_64", "amd64"} else f"linux-{machine}"
    elif sys.platform == "win32":
        platform_tag = "win-amd64"
    else:
        platform_tag = "*"
    return os.path.join(
        carla_root,
        "PythonAPI",
        "carla",
        "dist",
        f"carla-*{sys.version_info.major}.{sys.version_info.minor}-{platform_tag}.egg",
    )


def import_carla(carla_root: str = DEFAULT_CARLA_ROOT):
    """
    Import the CARLA Python API, using the local source build as a fallback.
    """

    try:
        import carla  # type: ignore

        return carla
    except ImportError:
        egg_matches = glob.glob(_get_carla_egg_glob(carla_root))
        if egg_matches:
            sys.path.append(egg_matches[0])
            import carla  # type: ignore

            return carla

    raise ImportError(
        "Unable to import the CARLA Python API. "
        f"Set CARLA_ROOT to your source build or install the matching carla package. "
        f"Tried source root: {carla_root}"
    )


def _is_tcp_port_open(host: str, port: int, timeout_s: float = 1.0) -> bool:
    """
    Return True if a TCP connection to the target host/port succeeds.
    """

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_s)
        try:
            sock.connect((host, port))
            return True
        except OSError:
            return False


def launch_carla_server(carla_cfg: Mapping[str, Any]) -> subprocess.Popen[bytes]:
    """
    Launch the CARLA server process for a source-build installation.
    """

    carla_root = str(carla_cfg.get("carla_root", DEFAULT_CARLA_ROOT))
    launch_mode = str(carla_cfg.get("launch_mode", "make_launch_only"))
    map_name = str(carla_cfg.get("map", ""))
    if launch_mode == "ue4editor_map":
        ue4_root = os.environ.get("UE4_ROOT", "")
        if not ue4_root:
            raise RuntimeError(
                "UE4_ROOT is not set. It is required to launch the CARLA source build directly."
            )
        editor_binary = os.path.join(ue4_root, "Engine", "Binaries", "Linux", "UE4Editor")
        uproject_path = os.path.join(carla_root, "Unreal", "CarlaUE4", "CarlaUE4.uproject")
        rhi = str(carla_cfg.get("rhi", "-vulkan"))
        extra_args = list(carla_cfg.get("launch_extra_args", []))
        command = [editor_binary, uproject_path, map_name, "-game", rhi]
        command.extend(extra_args)
    else:
        command = list(carla_cfg.get("launch_command", ["make", "launch-only"]))
    log_path = os.path.join(PROJECT_ROOT, "carla_launch.log")
    log_file = open(log_path, "ab")
    process = subprocess.Popen(
        command,
        cwd=carla_root,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
        start_new_session=True,
    )
    print(f"Launching CARLA: {' '.join(command)}")
    print(f"CARLA root: {carla_root}")
    print(f"Launch log: {log_path}")
    return process


def wait_for_carla_server(carla, host: str, port: int, timeout_s: float) -> Any:
    """
    Wait until the CARLA server accepts API requests.
    """

    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if not _is_tcp_port_open(host, port, timeout_s=1.0):
            time.sleep(1.0)
            continue
        try:
            client = carla.Client(host, port)
            client.set_timeout(2.0)
            client.get_server_version()
            client.get_world()
            return client
        except RuntimeError as exc:
            last_error = exc
            time.sleep(1.0)
    raise RuntimeError(
        f"Timed out while waiting for CARLA at {host}:{port} to become ready. "
        f"Last error: {last_error}"
    )


def _is_retriable_world_ready_error(exc: Exception) -> bool:
    if isinstance(exc, RuntimeError):
        return True
    if isinstance(exc, ValueError):
        error_text = str(exc).strip().lower()
        if "integer overflow in color channel" in error_text:
            return True
        if "color channel" in error_text and "overflow" in error_text:
            return True
    return False


def wait_for_carla_world_ready(client, world, timeout_s: float) -> Any:
    """
    Wait until the currently loaded world responds to the core API calls used
    by the scenario runner.
    """

    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            current_world = client.get_world()
            current_world.get_map()
            current_world.get_settings()
            current_world.get_blueprint_library()
            return current_world
        except Exception as exc:
            if not _is_retriable_world_ready_error(exc):
                raise
            last_error = exc
            time.sleep(1.0)
    raise RuntimeError(
        "Timed out while waiting for the loaded CARLA world to become ready. "
        f"Last error: {last_error}"
    )


def _build_map_load_candidates(map_name: str) -> list[str]:
    """
    Build a small set of plausible CARLA map identifiers for imported maps.
    """

    candidates: list[str] = []

    def add(candidate: str) -> None:
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    normalized = map_name.strip()
    add(normalized)

    last_token = normalized.rstrip("/").split("/")[-1]
    if not normalized.startswith("/Game/"):
        add(last_token)

    if normalized.startswith("/Game/"):
        folder_token = normalized.rstrip("/").split("/")[-1]
        parent_folder = normalized.rstrip("/").split("/")[-2] if "/" in normalized.rstrip("/") else folder_token
        if folder_token == parent_folder:
            add(f"{normalized.rstrip('/')}/{folder_token}")
        else:
            add(parent_folder)
            add(f"{normalized.rstrip('/')}/{last_token}")

    return candidates


def _map_name_matches_requested(current_map_name: str, candidates: list[str]) -> bool:
    normalized_current = str(current_map_name or "").strip()
    if normalized_current in candidates:
        return True

    current_leaf = normalized_current.rstrip("/").split("/")[-1]
    for candidate in candidates:
        candidate_leaf = str(candidate).rstrip("/").split("/")[-1]
        if current_leaf == candidate_leaf:
            return True
    return False


def run_carla_scenario(name: str) -> int:
    """
    Connect to a running CARLA server and load the requested map.
    """

    scenario_cfg = load_carla_scenario(name)
    carla_cfg = dict(scenario_cfg.get("carla", {}))
    carla = import_carla()
    host = str(carla_cfg.get("host", "127.0.0.1"))
    port = int(carla_cfg.get("port", 2000))
    request_timeout_s = float(carla_cfg.get("timeout_s", 10.0))
    launch_timeout_s = float(carla_cfg.get("launch_timeout_s", 120.0))

    client = None
    if _is_tcp_port_open(host, port, timeout_s=1.0):
        client = carla.Client(host, port)
        client.set_timeout(request_timeout_s)
    elif bool(carla_cfg.get("launch_if_needed", False)):
        launch_carla_server(carla_cfg)
        client = wait_for_carla_server(carla, host, port, launch_timeout_s)
        client.set_timeout(request_timeout_s)
    else:
        raise RuntimeError(
            f"CARLA is not reachable at {host}:{port}. "
            "Start the server first or set launch_if_needed: true in the scenario."
        )

    requested_map = str(carla_cfg["map"])
    load_candidates = _build_map_load_candidates(requested_map)
    world = None
    last_error: Exception | None = None
    current_world = client.get_world()
    current_map_name = str(current_world.get_map().name)
    if _map_name_matches_requested(current_map_name, load_candidates):
        world = current_world
        requested_map = current_map_name
    else:
        for candidate in load_candidates:
            try:
                world = client.load_world(candidate)
                requested_map = candidate
                break
            except RuntimeError as exc:
                last_error = exc

    if world is None:
        available_maps = list(client.get_available_maps())
        raise RuntimeError(
            "Unable to load the requested CARLA map. "
            f"Tried: {load_candidates}. "
            f"Available maps reported by server: {available_maps}. "
            f"Last error: {last_error}"
        )

    world = wait_for_carla_world_ready(
        client=client,
        world=world,
        timeout_s=max(10.0, float(request_timeout_s)),
    )

    from carla_scenario.runner import CARLA_FIXED_DELTA_SECONDS
    settings = world.get_settings()
    settings.synchronous_mode = bool(carla_cfg.get("synchronous_mode", False))
    settings.fixed_delta_seconds = CARLA_FIXED_DELTA_SECONDS
    world.apply_settings(settings)

    print(f"Loaded CARLA scenario: {scenario_cfg['name']}")
    print(f"Map asset: {requested_map}")
    print(f"Current map: {world.get_map().name}")
    print(f"Host/port: {carla_cfg.get('host', '127.0.0.1')}:{carla_cfg.get('port', 2000)}")

    runner_module_name = str(scenario_cfg.get("runner_module", "")).strip()
    if runner_module_name:
        runner_module = importlib.import_module(runner_module_name)
        if not hasattr(runner_module, "run_loaded_world"):
            raise AttributeError(
                f"Scenario runner module '{runner_module_name}' does not expose run_loaded_world(...)."
            )
        return int(
            runner_module.run_loaded_world(
                client=client,
                world=world,
                scenario_cfg=scenario_cfg,
                carla=carla,
            )
        )

    return 0


def main() -> int:
    if len(sys.argv) > 1:
        return run_carla_scenario(sys.argv[1])

    print("This branch is planning-only.")
    print("Available CARLA scenarios:", ", ".join(list_available_scenarios()) or "<none>")
    print("Usage: python main.py <scenario_name>")
    print(f"MPC config: {MPC_CONFIG_PATH}")
    print(f"Tracker config: {TRACKER_CONFIG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Find the custom_map2 anchor cubes in a running CARLA editor world."""

from __future__ import annotations

import glob
import os
import platform
import sys
from typing import Any, List


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from carla_scenario.loader import load_carla_scenario


DEFAULT_CARLA_ROOT = os.environ.get("CARLA_ROOT", "/home/umd-user/carla_source/carla")


def _best_partial_match(candidates: List[tuple[int, Any]]) -> Any | None:
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


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


def _find_environment_object_by_name(world, carla, name: str):
    name_lower = str(name).lower()
    partial_candidates = []
    for env_obj in world.get_environment_objects(carla.CityObjectLabel.Any):
        env_name = str(env_obj.name).lower()
        if env_name == name_lower:
            return env_obj
        if name_lower in env_name:
            partial_candidates.append((len(env_name), env_obj))
    return _best_partial_match(partial_candidates)


def _find_actor_by_name(world, name: str):
    name_lower = str(name).lower()
    partial_candidates = []
    for actor in world.get_actors():
        attr_name = str(actor.attributes.get("name", "")).lower()
        role_name = str(actor.attributes.get("role_name", "")).lower()
        type_id = str(actor.type_id).lower()
        if attr_name == name_lower or role_name == name_lower or type_id.endswith(name_lower):
            return actor
        if name_lower in attr_name:
            partial_candidates.append((len(attr_name), actor))
        if name_lower in role_name:
            partial_candidates.append((len(role_name), actor))
        if name_lower in type_id:
            partial_candidates.append((len(type_id), actor))
    return _best_partial_match(partial_candidates)


def _print_found_anchor(world, carla, anchor_name: str) -> bool:
    env_obj = _find_environment_object_by_name(world, carla, anchor_name)
    if env_obj is not None:
        transform = env_obj.transform
        print(
            f"FOUND EnvironmentObject anchor='{anchor_name}' actual_name='{env_obj.name}' "
            f"location=({transform.location.x:.3f}, {transform.location.y:.3f}, {transform.location.z:.3f}) "
            f"yaw={transform.rotation.yaw:.3f}"
        )
        return True

    actor = _find_actor_by_name(world, anchor_name)
    if actor is not None:
        transform = actor.get_transform()
        print(
            f"FOUND Actor anchor='{anchor_name}' type_id='{actor.type_id}' "
            f"location=({transform.location.x:.3f}, {transform.location.y:.3f}, {transform.location.z:.3f}) "
            f"yaw={transform.rotation.yaw:.3f}"
        )
        return True

    print(f"NOT FOUND anchor='{anchor_name}'")
    return False


def main() -> int:
    scenario_cfg = load_carla_scenario("custom_map2")
    carla_cfg = dict(scenario_cfg.get("carla", {}))
    anchors_cfg = dict(scenario_cfg.get("anchors", {}))

    carla_root = str(carla_cfg.get("carla_root", DEFAULT_CARLA_ROOT))
    carla = import_carla(carla_root=carla_root)

    client = carla.Client(
        str(carla_cfg.get("host", "127.0.0.1")),
        int(carla_cfg.get("port", 2000)),
    )
    client.set_timeout(float(carla_cfg.get("timeout_s", 10.0)))

    world = client.get_world()
    print(f"Connected world map: {world.get_map().name}")

    ego_anchor_name = str(anchors_cfg.get("ego_spawn", "cav_spawn"))
    destination_anchor_name = str(anchors_cfg.get("final_destination", "final_destination"))

    found_ego = _print_found_anchor(world, carla, ego_anchor_name)
    found_goal = _print_found_anchor(world, carla, destination_anchor_name)
    return 0 if found_ego and found_goal else 1


if __name__ == "__main__":
    raise SystemExit(main())

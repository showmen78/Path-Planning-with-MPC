"""
Global planner helpers for CARLA lane context and route summaries.

Local lane context still uses the project's sampled lane-center waypoint graph.
When a CARLA map is available, route planning uses CARLA's built-in
GlobalRoutePlanner so intersection maneuvers come from the HD map graph.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
import glob
import heapq
import math
import os
import platform
import sys
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .carla_lane_graph import carla_waypoint_graph_key, direction_key


_CARLA_ROUTE_PLANNER_LOAD_ERROR = ""


def _set_carla_route_planner_load_error(message: str) -> None:
    global _CARLA_ROUTE_PLANNER_LOAD_ERROR
    _CARLA_ROUTE_PLANNER_LOAD_ERROR = str(message or "").strip()


def _get_carla_route_planner_load_error() -> str:
    return str(_CARLA_ROUTE_PLANNER_LOAD_ERROR or "").strip()


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


def _ensure_carla_api_loaded(carla_root: str):
    try:
        import carla  # type: ignore

        return carla
    except Exception as exc:
        first_exc = exc

    egg_matches = glob.glob(_get_carla_egg_glob(carla_root))
    if egg_matches:
        egg_path = egg_matches[0]
        if egg_path not in sys.path:
            sys.path.append(egg_path)
        try:
            import carla  # type: ignore

            return carla
        except Exception as exc:
            _set_carla_route_planner_load_error(
                f"CARLA import from egg failed: {exc}"
            )
            return None

    _set_carla_route_planner_load_error(
        f"CARLA import failed: {first_exc}. No matching CARLA egg was found under {carla_root}."
    )
    return None


def _candidate_site_packages_paths() -> List[str]:
    candidates: List[str] = []
    conda_prefix = str(os.environ.get("CONDA_PREFIX", "")).strip()
    if conda_prefix:
        candidates.extend(glob.glob(os.path.join(conda_prefix, "lib", "python*", "site-packages")))

    for envs_root in (
        os.path.expanduser("~/miniconda3/envs"),
        os.path.expanduser("~/anaconda3/envs"),
    ):
        if os.path.isdir(envs_root):
            candidates.extend(glob.glob(os.path.join(envs_root, "*", "lib", "python*", "site-packages")))

    for base_root in (
        os.path.expanduser("~/miniconda3"),
        os.path.expanduser("~/anaconda3"),
    ):
        if os.path.isdir(base_root):
            candidates.extend(glob.glob(os.path.join(base_root, "lib", "python*", "site-packages")))

    unique_candidates: List[str] = []
    for candidate in candidates:
        normalized_candidate = os.path.abspath(str(candidate))
        if normalized_candidate not in unique_candidates:
            unique_candidates.append(normalized_candidate)
    return unique_candidates


def _ensure_python_module_available(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        pass

    for site_packages_path in _candidate_site_packages_paths():
        module_init_path = os.path.join(site_packages_path, module_name, "__init__.py")
        if not os.path.exists(module_init_path):
            continue
        if site_packages_path not in sys.path:
            sys.path.append(site_packages_path)
        try:
            __import__(module_name)
            return True
        except Exception:
            continue
    return False


def _load_carla_global_route_planner():
    _set_carla_route_planner_load_error("")
    carla_root = os.environ.get("CARLA_ROOT", "/home/umd-user/carla_source/carla")
    carla_api = _ensure_carla_api_loaded(carla_root=carla_root)
    if carla_api is None:
        return None

    if not _ensure_python_module_available("networkx"):
        _set_carla_route_planner_load_error(
            "Required dependency 'networkx' is missing for CARLA GlobalRoutePlanner."
        )
        return None

    try:
        from agents.navigation.global_route_planner import GlobalRoutePlanner as planner_cls

        return planner_cls
    except Exception as exc:
        first_exc = exc

    agents_root = os.path.join(carla_root, "PythonAPI", "carla")
    if os.path.isdir(agents_root) and agents_root not in sys.path:
        sys.path.append(agents_root)

    try:
        from agents.navigation.global_route_planner import GlobalRoutePlanner as planner_cls

        return planner_cls
    except Exception as exc:
        _set_carla_route_planner_load_error(
            f"GlobalRoutePlanner import failed: {exc} (initial import error: {first_exc})"
        )
        return None


@dataclass(frozen=True)
class WaypointNode:
    """Normalized waypoint representation used by the A* graph."""

    index: int
    x_m: float
    y_m: float
    lane_id: int
    lane_width_m: float
    road_id: str
    direction: str
    heading_rad: float
    progress_m: float
    maneuver: str
    is_intersection: bool
    next_index: int | None
    successor_indices: Tuple[int, ...]
    carla_waypoint: object | None


@dataclass(frozen=True)
class WaypointQueryResult:
    """Nearest sampled-graph waypoint for a query point."""

    index: int
    distance_m: float
    x_m: float
    y_m: float
    road_id: str
    lane_id: int
    direction: str


@dataclass
class RoutePlanSummary:
    """Compact route description exposed to the prompt builder."""

    route_found: bool
    start_road_id: str
    start_lane_id: int
    goal_road_id: str
    goal_lane_id: int
    optimal_lane_id: int
    distance_to_destination_m: float
    next_macro_maneuver: str
    route_waypoints: List[List[float]]
    road_options: List[str] = field(default_factory=list)
    debug_reason: str = ""
    start_graph_index: int = -1
    goal_graph_index: int = -1
    start_graph_xy: Tuple[float, float] | None = None
    goal_graph_xy: Tuple[float, float] | None = None
    start_query_distance_m: float = float("inf")
    goal_query_distance_m: float = float("inf")


class AStarGlobalPlanner:
    """Build a lightweight waypoint graph and plan A* routes on it."""

    _INTERSECTION_TURN_THRESHOLD_RAD = math.radians(20.0)

    def __init__(
        self,
        lane_center_waypoints: Sequence[Mapping[str, object]],
        world_map: object | None = None,
        route_sample_distance_m: float = 2.0,
        lane_change_penalty_m: float = 5.0,
        lane_change_progress_tolerance_m: float = 5.0,
        lane_change_distance_factor: float = 1.8,
        max_heading_diff_rad: float = 0.8,
    ) -> None:
        self._lane_change_penalty_m = max(0.0, float(lane_change_penalty_m))
        self._lane_change_progress_tolerance_m = max(0.5, float(lane_change_progress_tolerance_m))
        self._lane_change_distance_factor = max(1.0, float(lane_change_distance_factor))
        self._max_heading_diff_rad = max(0.1, float(max_heading_diff_rad))
        self._world_map = world_map
        self._route_sample_distance_m = max(0.5, float(route_sample_distance_m))

        self._nodes = self._normalize_waypoints(lane_center_waypoints=lane_center_waypoints)
        self._node_x_m = np.asarray([float(node.x_m) for node in self._nodes], dtype=float)
        self._node_y_m = np.asarray([float(node.y_m) for node in self._nodes], dtype=float)
        self._adjacency = self._build_adjacency()
        self._internal_lane_id_by_raw_key = self._build_internal_lane_lookup()
        self._carla_route_planner = self._build_carla_route_planner()

    @staticmethod
    def _position_of_waypoint(waypoint: Mapping[str, object]) -> Tuple[float, float] | None:
        position = waypoint.get("position", None)
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            return None
        return float(position[0]), float(position[1])

    @staticmethod
    def _waypoint_key(x_m: float, y_m: float) -> Tuple[float, float]:
        return round(float(x_m), 3), round(float(y_m), 3)

    @staticmethod
    def _coerce_carla_waypoint_key(raw_key: object) -> Tuple[int, int, int, float] | None:
        if not isinstance(raw_key, (list, tuple)) or len(raw_key) < 4:
            return None
        return (
            int(raw_key[0]),
            int(raw_key[1]),
            int(raw_key[2]),
            round(float(raw_key[3]), 3),
        )

    @staticmethod
    def _wrap_angle(angle_rad: float) -> float:
        return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _progress_of_waypoint(x_m: float, y_m: float, heading_rad: float) -> float:
        return float(x_m * math.cos(float(heading_rad)) + y_m * math.sin(float(heading_rad)))

    def _normalize_waypoints(
        self,
        lane_center_waypoints: Sequence[Mapping[str, object]],
    ) -> List[WaypointNode]:
        raw_items: List[Tuple[Mapping[str, object], Tuple[float, float]]] = []
        for waypoint in lane_center_waypoints:
            position = self._position_of_waypoint(waypoint)
            if position is None:
                continue
            raw_items.append((waypoint, position))

        index_by_key: Dict[Tuple[float, float], int] = {}
        index_by_carla_key: Dict[Tuple[int, int, int, float], int] = {}
        for idx, (waypoint, position) in enumerate(raw_items):
            index_by_key[self._waypoint_key(*position)] = idx
            carla_key = self._coerce_carla_waypoint_key(waypoint.get("carla_waypoint_key", None))
            if carla_key is None:
                carla_waypoint = waypoint.get("carla_waypoint", None)
                if carla_waypoint is not None:
                    try:
                        carla_key = carla_waypoint_graph_key(carla_waypoint)
                    except Exception:
                        carla_key = None
            if carla_key is not None:
                index_by_carla_key[carla_key] = idx

        nodes: List[WaypointNode] = []
        for idx, (waypoint, position) in enumerate(raw_items):
            next_index = None
            next_key = self._coerce_carla_waypoint_key(waypoint.get("next_key", None))
            if next_key is not None:
                next_index = index_by_carla_key.get(next_key)
            next_position = waypoint.get("next", None)
            if isinstance(next_position, (list, tuple)) and len(next_position) >= 2:
                if next_index is None:
                    next_index = index_by_key.get(
                        self._waypoint_key(float(next_position[0]), float(next_position[1]))
                    )

            successor_indices: List[int] = []
            successor_keys_raw = waypoint.get("successor_keys", [])
            if isinstance(successor_keys_raw, Sequence) and not isinstance(successor_keys_raw, (str, bytes)):
                for successor_key_raw in successor_keys_raw:
                    successor_key = self._coerce_carla_waypoint_key(successor_key_raw)
                    if successor_key is None:
                        continue
                    successor_index = index_by_carla_key.get(successor_key)
                    if successor_index is None:
                        continue
                    if int(successor_index) not in successor_indices:
                        successor_indices.append(int(successor_index))
            successors_raw = waypoint.get("successors", [])
            if isinstance(successors_raw, Sequence) and not isinstance(successors_raw, (str, bytes)):
                for successor_raw in successors_raw:
                    if not isinstance(successor_raw, (list, tuple)) or len(successor_raw) < 2:
                        continue
                    successor_index = index_by_key.get(
                        self._waypoint_key(float(successor_raw[0]), float(successor_raw[1]))
                    )
                    if successor_index is None:
                        continue
                    if int(successor_index) not in successor_indices:
                        successor_indices.append(int(successor_index))
            if next_index is not None and int(next_index) not in successor_indices:
                successor_indices.insert(0, int(next_index))

            heading_rad = float(waypoint.get("heading_rad", 0.0))
            nodes.append(
                WaypointNode(
                    index=int(idx),
                    x_m=float(position[0]),
                    y_m=float(position[1]),
                    lane_id=int(waypoint.get("lane_id", -1)),
                    lane_width_m=max(0.1, float(waypoint.get("lane_width_m", 4.0))),
                    road_id=str(waypoint.get("road_id", "unknown_road")),
                    direction=str(waypoint.get("direction", "unknown")),
                    heading_rad=heading_rad,
                    progress_m=self._progress_of_waypoint(
                        x_m=float(position[0]),
                        y_m=float(position[1]),
                        heading_rad=heading_rad,
                    ),
                    maneuver=str(waypoint.get("maneuver", "straight")),
                    is_intersection=bool(waypoint.get("is_intersection", False)),
                    next_index=next_index,
                    successor_indices=tuple(successor_indices),
                    carla_waypoint=waypoint.get("carla_waypoint", None),
                )
            )
        return nodes

    def _build_internal_lane_lookup(self) -> Dict[Tuple[int, int, int], int]:
        lookup: Dict[Tuple[int, int, int], int] = {}
        for node in self._nodes:
            carla_waypoint = node.carla_waypoint
            if carla_waypoint is None:
                continue
            lookup[
                (
                    int(getattr(carla_waypoint, "road_id", 0)),
                    int(getattr(carla_waypoint, "section_id", 0)),
                    int(getattr(carla_waypoint, "lane_id", 0)),
                )
            ] = int(node.lane_id)
        return lookup

    def _build_carla_route_planner(self):
        if self._world_map is None:
            _set_carla_route_planner_load_error("CARLA world map was not provided to the global planner.")
            return None
        planner_cls = _load_carla_global_route_planner()
        if planner_cls is None:
            return None
        try:
            return planner_cls(self._world_map, self._route_sample_distance_m)
        except Exception as exc:
            _set_carla_route_planner_load_error(
                f"CARLA GlobalRoutePlanner construction failed: {exc}"
            )
            return None

    def _build_adjacency(self) -> Dict[int, List[Tuple[int, float]]]:
        adjacency: Dict[int, List[Tuple[int, float]]] = {node.index: [] for node in self._nodes}
        if len(self._nodes) == 0:
            return adjacency

        for node in self._nodes:
            successor_indices = list(node.successor_indices)
            if len(successor_indices) == 0 and node.next_index is not None:
                successor_indices = [int(node.next_index)]
            seen_successor_indices: set[int] = set()
            for successor_index in successor_indices:
                if int(successor_index) in seen_successor_indices:
                    continue
                seen_successor_indices.add(int(successor_index))
                if not (0 <= int(successor_index) < len(self._nodes)):
                    continue
                next_node = self._nodes[int(successor_index)]
                forward_cost = math.hypot(next_node.x_m - node.x_m, next_node.y_m - node.y_m)
                adjacency[node.index].append((next_node.index, max(1e-6, float(forward_cost))))

        grouped: Dict[Tuple[str, str], Dict[int, List[WaypointNode]]] = {}
        for node in self._nodes:
            grouped.setdefault((node.road_id, node.direction), {}).setdefault(node.lane_id, []).append(node)

        for lane_groups in grouped.values():
            for lane_nodes in lane_groups.values():
                lane_nodes.sort(key=lambda node: float(node.progress_m))

            lane_ids = sorted(lane_groups.keys())
            for lane_id in lane_ids:
                base_lane_nodes = lane_groups.get(lane_id, [])
                for neighbor_lane_id in (lane_id - 1, lane_id + 1):
                    neighbor_lane_nodes = lane_groups.get(neighbor_lane_id, [])
                    if len(base_lane_nodes) == 0 or len(neighbor_lane_nodes) == 0:
                        continue
                    neighbor_progress = [float(node.progress_m) for node in neighbor_lane_nodes]
                    for node in base_lane_nodes:
                        insert_at = bisect_left(neighbor_progress, float(node.progress_m))
                        candidate_indices = [insert_at - 1, insert_at, insert_at + 1]
                        best_candidate = None
                        best_score = float("inf")
                        for candidate_idx in candidate_indices:
                            if not (0 <= candidate_idx < len(neighbor_lane_nodes)):
                                continue
                            candidate = neighbor_lane_nodes[candidate_idx]
                            progress_gap_m = abs(float(candidate.progress_m) - float(node.progress_m))
                            if progress_gap_m > float(self._lane_change_progress_tolerance_m):
                                continue
                            heading_gap_rad = abs(
                                self._wrap_angle(float(candidate.heading_rad) - float(node.heading_rad))
                            )
                            if heading_gap_rad > float(self._max_heading_diff_rad):
                                continue
                            distance_m = math.hypot(candidate.x_m - node.x_m, candidate.y_m - node.y_m)
                            max_lane_change_distance_m = (
                                float(node.lane_width_m) * float(self._lane_change_distance_factor)
                            )
                            if distance_m > max_lane_change_distance_m:
                                continue
                            score = distance_m + progress_gap_m
                            if score < best_score:
                                best_score = score
                                best_candidate = candidate
                        if best_candidate is None:
                            continue
                        lateral_cost = math.hypot(
                            best_candidate.x_m - node.x_m,
                            best_candidate.y_m - node.y_m,
                        ) + float(self._lane_change_penalty_m)
                        adjacency[node.index].append((best_candidate.index, float(max(1e-6, lateral_cost))))

        return adjacency

    def nearest_waypoint_index(self, x_m: float, y_m: float) -> int | None:
        nearest_query = self.nearest_waypoint_query(x_m=x_m, y_m=y_m)
        if nearest_query is None:
            return None
        return int(nearest_query.index)

    def nearest_waypoint_query(self, x_m: float, y_m: float) -> WaypointQueryResult | None:
        if len(self._nodes) == 0:
            return None
        delta_x = self._node_x_m - float(x_m)
        delta_y = self._node_y_m - float(y_m)
        distance_sq = delta_x * delta_x + delta_y * delta_y
        nearest_index = int(np.argmin(distance_sq))
        nearest_node = self._nodes[nearest_index]
        distance_m = math.sqrt(float(distance_sq[nearest_index]))
        return WaypointQueryResult(
            index=int(nearest_node.index),
            distance_m=float(distance_m),
            x_m=float(nearest_node.x_m),
            y_m=float(nearest_node.y_m),
            road_id=str(nearest_node.road_id),
            lane_id=int(nearest_node.lane_id),
            direction=str(nearest_node.direction),
        )

    def _internal_lane_id_for_carla_waypoint(self, carla_waypoint) -> int | None:
        if carla_waypoint is None:
            return None
        raw_key = (
            int(getattr(carla_waypoint, "road_id", 0)),
            int(getattr(carla_waypoint, "section_id", 0)),
            int(getattr(carla_waypoint, "lane_id", 0)),
        )
        if raw_key in self._internal_lane_id_by_raw_key:
            return int(self._internal_lane_id_by_raw_key[raw_key])

        location = getattr(getattr(carla_waypoint, "transform", None), "location", None)
        if location is None:
            return None
        nearest_index = self.nearest_waypoint_index(
            x_m=float(getattr(location, "x", 0.0)),
            y_m=float(getattr(location, "y", 0.0)),
        )
        if nearest_index is None:
            return None
        return int(self._nodes[int(nearest_index)].lane_id)

    def _route_waypoints_from_carla_trace(
        self,
        route_trace: Sequence[tuple[object, object]],
    ) -> tuple[List[List[float]], List[int], List[str], str, int | None]:
        route_waypoints: List[List[float]] = []
        route_lane_ids: List[int] = []
        road_options: List[str] = []
        preferred_lane_id: int | None = None
        maneuver_lane_id: int | None = None

        for carla_waypoint, road_option in route_trace:
            transform = getattr(carla_waypoint, "transform", None)
            location = getattr(transform, "location", None)
            if location is None:
                continue
            waypoint_xy = [float(getattr(location, "x", 0.0)), float(getattr(location, "y", 0.0))]
            if len(route_waypoints) == 0 or math.hypot(
                float(route_waypoints[-1][0]) - float(waypoint_xy[0]),
                float(route_waypoints[-1][1]) - float(waypoint_xy[1]),
            ) > 1e-3:
                route_waypoints.append(waypoint_xy)

            lane_id = self._internal_lane_id_for_carla_waypoint(carla_waypoint)
            if lane_id is not None:
                route_lane_ids.append(int(lane_id))
                if preferred_lane_id is None:
                    preferred_lane_id = int(lane_id)
                elif int(lane_id) != int(preferred_lane_id):
                    preferred_lane_id = int(lane_id)

            option_name = str(getattr(road_option, "name", road_option)).upper()
            if option_name and (len(road_options) == 0 or str(road_options[-1]) != str(option_name)):
                road_options.append(str(option_name))

            if option_name in {"CHANGELANELEFT", "CHANGELANERIGHT"} and lane_id is not None:
                preferred_lane_id = int(lane_id)
            elif option_name in {"LEFT", "RIGHT", "STRAIGHT"} and maneuver_lane_id is None and lane_id is not None:
                maneuver_lane_id = int(lane_id)

        if maneuver_lane_id is not None:
            preferred_lane_id = int(maneuver_lane_id)
        elif preferred_lane_id is None and len(route_lane_ids) > 0:
            preferred_lane_id = int(route_lane_ids[0])

        return (
            route_waypoints,
            route_lane_ids,
            road_options,
            self._next_macro_maneuver_from_road_options(road_options),
            preferred_lane_id,
        )

    @staticmethod
    def _next_macro_maneuver_from_road_options(road_options: Sequence[str]) -> str:
        normalized_options = [str(option).strip().upper() for option in list(road_options or []) if str(option).strip()]
        for option_name in normalized_options:
            if option_name == "LEFT":
                return "Left Turn"
            if option_name == "RIGHT":
                return "Right Turn"
        for option_name in normalized_options:
            if option_name == "STRAIGHT":
                return "Continue Straight"
        return "Continue Straight"

    @staticmethod
    def _location_xy(location) -> Tuple[float, float] | None:
        if location is None:
            return None
        if not hasattr(location, "x") or not hasattr(location, "y"):
            return None
        return float(getattr(location, "x", 0.0)), float(getattr(location, "y", 0.0))

    @staticmethod
    def _route_length_from_waypoints(route_waypoints: Sequence[Sequence[float]]) -> float:
        if len(route_waypoints) <= 1:
            return 0.0
        total_length_m = 0.0
        for start_xy, end_xy in zip(route_waypoints[:-1], route_waypoints[1:]):
            total_length_m += math.hypot(
                float(end_xy[0]) - float(start_xy[0]),
                float(end_xy[1]) - float(start_xy[1]),
            )
        return float(total_length_m)

    @staticmethod
    def _optimal_lane_id_from_sequence(route_lane_ids: Sequence[int], fallback_lane_id: int) -> int:
        if len(route_lane_ids) == 0:
            return int(fallback_lane_id)
        first_lane_id = int(route_lane_ids[0])
        for lane_id in route_lane_ids[1:]:
            if int(lane_id) != int(first_lane_id):
                return int(lane_id)
        return int(route_lane_ids[-1])

    def _build_route_summary_from_carla_trace(
        self,
        route_trace: Sequence[tuple[object, object]],
        start_xy: Sequence[float],
        goal_xy: Sequence[float],
    ) -> RoutePlanSummary | None:
        route_waypoints, route_lane_ids, road_options, next_macro_maneuver, preferred_lane_id = self._route_waypoints_from_carla_trace(route_trace)
        if len(route_waypoints) == 0:
            return None

        start_x_m = float(start_xy[0]) if len(start_xy) >= 1 else 0.0
        start_y_m = float(start_xy[1]) if len(start_xy) >= 2 else 0.0
        goal_x_m = float(goal_xy[0]) if len(goal_xy) >= 1 else 0.0
        goal_y_m = float(goal_xy[1]) if len(goal_xy) >= 2 else 0.0

        start_query = self.nearest_waypoint_query(x_m=float(start_x_m), y_m=float(start_y_m))
        goal_query = self.nearest_waypoint_query(x_m=float(goal_x_m), y_m=float(goal_y_m))
        start_lane_id = int(getattr(start_query, "lane_id", -1))
        goal_lane_id = int(getattr(goal_query, "lane_id", -1))
        start_road_id = str(getattr(start_query, "road_id", "unknown_road"))
        goal_road_id = str(getattr(goal_query, "road_id", "unknown_road"))

        route_length_m = self._route_length_from_waypoints(route_waypoints)
        route_length_m += math.hypot(route_waypoints[0][0] - float(start_x_m), route_waypoints[0][1] - float(start_y_m))
        route_length_m += math.hypot(float(goal_x_m) - route_waypoints[-1][0], float(goal_y_m) - route_waypoints[-1][1])

        if preferred_lane_id is None:
            optimal_lane_id = self._optimal_lane_id_from_sequence(
                route_lane_ids=route_lane_ids,
                fallback_lane_id=int(goal_lane_id),
            )
        else:
            optimal_lane_id = int(preferred_lane_id)

        return RoutePlanSummary(
            route_found=True,
            start_road_id=str(start_road_id),
            start_lane_id=int(start_lane_id),
            goal_road_id=str(goal_road_id),
            goal_lane_id=int(goal_lane_id),
            optimal_lane_id=int(optimal_lane_id),
            distance_to_destination_m=float(route_length_m),
            next_macro_maneuver=str(next_macro_maneuver),
            route_waypoints=route_waypoints,
            road_options=list(road_options),
            start_graph_index=int(getattr(start_query, "index", -1)),
            goal_graph_index=int(getattr(goal_query, "index", -1)),
            start_graph_xy=(
                float(getattr(start_query, "x_m", 0.0)),
                float(getattr(start_query, "y_m", 0.0)),
            ) if start_query is not None else None,
            goal_graph_xy=(
                float(getattr(goal_query, "x_m", 0.0)),
                float(getattr(goal_query, "y_m", 0.0)),
            ) if goal_query is not None else None,
            start_query_distance_m=float(getattr(start_query, "distance_m", float("inf"))),
            goal_query_distance_m=float(getattr(goal_query, "distance_m", float("inf"))),
        )

    def _plan_route_with_carla(
        self,
        start_x_m: float,
        start_y_m: float,
        goal_x_m: float,
        goal_y_m: float,
        start_node: WaypointNode,
        goal_node: WaypointNode,
    ) -> RoutePlanSummary | None:
        if self._carla_route_planner is None:
            return None
        start_carla_waypoint = start_node.carla_waypoint
        goal_carla_waypoint = goal_node.carla_waypoint
        if start_carla_waypoint is None or goal_carla_waypoint is None:
            return None

        start_location = getattr(getattr(start_carla_waypoint, "transform", None), "location", None)
        goal_location = getattr(getattr(goal_carla_waypoint, "transform", None), "location", None)
        if start_location is None or goal_location is None:
            return None

        return self.plan_route_from_locations(
            start_location=start_location,
            goal_location=goal_location,
            fallback_start_xy=[float(start_x_m), float(start_y_m)],
            fallback_goal_xy=[float(goal_x_m), float(goal_y_m)],
        )

    def plan_route_from_locations(
        self,
        start_location,
        goal_location,
        fallback_start_xy: Sequence[float] | None = None,
        fallback_goal_xy: Sequence[float] | None = None,
    ) -> RoutePlanSummary:
        if self._carla_route_planner is None:
            start_xy = self._location_xy(start_location)
            goal_xy = self._location_xy(goal_location)
            start_x_m = float(start_xy[0]) if start_xy is not None else 0.0
            start_y_m = float(start_xy[1]) if start_xy is not None else 0.0
            goal_x_m = float(goal_xy[0]) if goal_xy is not None else 0.0
            goal_y_m = float(goal_xy[1]) if goal_xy is not None else 0.0
            load_error = _get_carla_route_planner_load_error()
            return RoutePlanSummary(
                route_found=False,
                start_road_id="unknown_road",
                start_lane_id=-1,
                goal_road_id="unknown_road",
                goal_lane_id=-1,
                optimal_lane_id=-1,
                distance_to_destination_m=float(math.hypot(goal_x_m - start_x_m, goal_y_m - start_y_m)),
                next_macro_maneuver="Continue Straight",
                route_waypoints=[],
                debug_reason=(
                    "CARLA GlobalRoutePlanner is not available for the loaded map."
                    if not load_error
                    else f"CARLA GlobalRoutePlanner is not available: {load_error}"
                ),
            )

        start_xy = self._location_xy(start_location)
        goal_xy = self._location_xy(goal_location)
        if start_xy is None or goal_xy is None:
            return RoutePlanSummary(
                route_found=False,
                start_road_id="unknown_road",
                start_lane_id=-1,
                goal_road_id="unknown_road",
                goal_lane_id=-1,
                optimal_lane_id=-1,
                distance_to_destination_m=0.0,
                next_macro_maneuver="Continue Straight",
                route_waypoints=[],
                debug_reason="Start or goal location for CARLA GlobalRoutePlanner was invalid.",
            )

        try:
            route_trace = list(self._carla_route_planner.trace_route(start_location, goal_location))
        except Exception as exc:
            return RoutePlanSummary(
                route_found=False,
                start_road_id="unknown_road",
                start_lane_id=-1,
                goal_road_id="unknown_road",
                goal_lane_id=-1,
                optimal_lane_id=-1,
                distance_to_destination_m=float(math.hypot(float(goal_xy[0]) - float(start_xy[0]), float(goal_xy[1]) - float(start_xy[1]))),
                next_macro_maneuver="Continue Straight",
                route_waypoints=[],
                debug_reason=f"CARLA GlobalRoutePlanner.trace_route failed: {exc}",
            )
        if len(route_trace) == 0:
            return RoutePlanSummary(
                route_found=False,
                start_road_id="unknown_road",
                start_lane_id=-1,
                goal_road_id="unknown_road",
                goal_lane_id=-1,
                optimal_lane_id=-1,
                distance_to_destination_m=float(math.hypot(float(goal_xy[0]) - float(start_xy[0]), float(goal_xy[1]) - float(start_xy[1]))),
                next_macro_maneuver="Continue Straight",
                route_waypoints=[],
                debug_reason="CARLA GlobalRoutePlanner returned an empty route trace.",
            )

        route_summary = self._build_route_summary_from_carla_trace(
            route_trace=route_trace,
            start_xy=list(fallback_start_xy or start_xy),
            goal_xy=list(fallback_goal_xy or goal_xy),
        )
        if route_summary is not None:
            return route_summary
        return RoutePlanSummary(
            route_found=False,
            start_road_id="unknown_road",
            start_lane_id=-1,
            goal_road_id="unknown_road",
            goal_lane_id=-1,
            optimal_lane_id=-1,
            distance_to_destination_m=float(math.hypot(float(goal_xy[0]) - float(start_xy[0]), float(goal_xy[1]) - float(start_xy[1]))),
            next_macro_maneuver="Continue Straight",
            route_waypoints=[],
            debug_reason="CARLA GlobalRoutePlanner returned a route trace but no usable waypoints were extracted.",
        )

    def get_local_lane_context(
        self,
        x_m: float,
        y_m: float,
        heading_rad: float | None = None,
    ) -> Dict[str, object]:
        """
        Return road/lane identifiers plus left/right lane-change availability.
        """

        nearest_index = self.nearest_waypoint_index(x_m=x_m, y_m=y_m)
        if nearest_index is None:
            return {
                "road_id": "unknown_road",
                "lane_id": -1,
                "lane_ids": [],
                "lane_count": 0,
                "min_lane_id": -1,
                "max_lane_id": -1,
                "can_change_left": False,
                "can_change_right": False,
            }

        nearest_node = self._nodes[int(nearest_index)]
        reference_heading_rad = float(nearest_node.heading_rad if heading_rad is None else heading_rad)
        corridor_nodes = [
            node
            for node in self._nodes
            if str(node.road_id) == str(nearest_node.road_id)
            and str(node.direction) == str(nearest_node.direction)
        ]
        local_lane_ids = sorted(
            {
                int(node.lane_id)
                for node in corridor_nodes
                if int(node.lane_id) > 0
            }
        )

        nearby_same_corridor = [
            node
            for node in corridor_nodes
            if abs(float(node.progress_m) - float(nearest_node.progress_m))
            <= float(self._lane_change_progress_tolerance_m)
        ]

        lane_offsets: Dict[int, float] = {}
        for node in nearby_same_corridor:
            dx_m = float(node.x_m) - float(nearest_node.x_m)
            dy_m = float(node.y_m) - float(nearest_node.y_m)
            lateral_offset_m = (
                -math.sin(reference_heading_rad) * dx_m + math.cos(reference_heading_rad) * dy_m
            )
            existing = lane_offsets.get(int(node.lane_id), None)
            if existing is None or abs(float(lateral_offset_m)) < abs(float(existing)):
                lane_offsets[int(node.lane_id)] = float(lateral_offset_m)

        current_lane_id = int(nearest_node.lane_id)
        can_change_left = int(current_lane_id + 1) in lane_offsets
        can_change_right = int(current_lane_id - 1) in lane_offsets

        return {
            "road_id": str(nearest_node.road_id),
            "lane_id": int(current_lane_id),
            "lane_ids": list(local_lane_ids),
            "lane_count": int(len(local_lane_ids)),
            "min_lane_id": int(local_lane_ids[0]) if len(local_lane_ids) > 0 else -1,
            "max_lane_id": int(local_lane_ids[-1]) if len(local_lane_ids) > 0 else -1,
            "can_change_left": bool(can_change_left),
            "can_change_right": bool(can_change_right),
        }

    @staticmethod
    def _append_unique_route_point(
        route_waypoints: List[List[float]],
        point_xy: Sequence[float],
        min_distance_m: float = 1e-3,
    ) -> None:
        if len(point_xy) < 2:
            return
        point = [float(point_xy[0]), float(point_xy[1])]
        if len(route_waypoints) == 0:
            route_waypoints.append(point)
            return
        if math.hypot(
            float(route_waypoints[-1][0]) - float(point[0]),
            float(route_waypoints[-1][1]) - float(point[1]),
        ) <= float(min_distance_m):
            return
        route_waypoints.append(point)

    def _unreachable_route_reason(
        self,
        start_index: int,
        goal_index: int,
        start_query: WaypointQueryResult,
        goal_query: WaypointQueryResult,
        reachable_node_count: int,
    ) -> str:
        start_node = self._nodes[int(start_index)]
        goal_node = self._nodes[int(goal_index)]
        details = [
            "No directed path exists between the closest sampled waypoints.",
            (
                f"start_index={int(start_index)} road={str(start_node.road_id)} lane={int(start_node.lane_id)} "
                f"dir={str(start_node.direction)} query_distance_m={float(start_query.distance_m):.3f}"
            ),
            (
                f"goal_index={int(goal_index)} road={str(goal_node.road_id)} lane={int(goal_node.lane_id)} "
                f"dir={str(goal_node.direction)} query_distance_m={float(goal_query.distance_m):.3f}"
            ),
            f"reachable_nodes_from_start={int(reachable_node_count)}",
        ]
        if len(self._adjacency.get(int(start_index), [])) == 0:
            details.append("the chosen start waypoint has no outgoing graph edges")
        if str(start_node.direction) != str(goal_node.direction):
            details.append("the chosen start and goal waypoints point in different travel directions")
        return "; ".join(details) + "."

    def _build_route_waypoints_from_indices(
        self,
        route_indices: Sequence[int],
        start_xy: Sequence[float],
        goal_xy: Sequence[float],
    ) -> List[List[float]]:
        route_waypoints: List[List[float]] = []
        self._append_unique_route_point(route_waypoints, start_xy)
        for index in route_indices:
            node = self._nodes[int(index)]
            self._append_unique_route_point(route_waypoints, [float(node.x_m), float(node.y_m)])
        self._append_unique_route_point(route_waypoints, goal_xy)
        return route_waypoints

    def _plan_route_with_internal_astar(
        self,
        start_x_m: float,
        start_y_m: float,
        goal_x_m: float,
        goal_y_m: float,
        start_index: int,
        goal_index: int,
        start_node: WaypointNode,
        goal_node: WaypointNode,
        start_query: WaypointQueryResult,
        goal_query: WaypointQueryResult,
    ) -> RoutePlanSummary:
        frontier: List[Tuple[float, int]] = []
        heapq.heappush(frontier, (0.0, int(start_index)))
        came_from: Dict[int, int | None] = {int(start_index): None}
        g_cost: Dict[int, float] = {int(start_index): 0.0}

        while frontier:
            _, current_index = heapq.heappop(frontier)
            if int(current_index) == int(goal_index):
                break
            for neighbor_index, edge_cost_m in self._adjacency.get(int(current_index), []):
                new_cost = float(g_cost[int(current_index)]) + float(edge_cost_m)
                if int(neighbor_index) not in g_cost or new_cost < float(g_cost[int(neighbor_index)]):
                    g_cost[int(neighbor_index)] = float(new_cost)
                    heuristic_m = math.hypot(
                        float(self._nodes[int(neighbor_index)].x_m) - float(goal_x_m),
                        float(self._nodes[int(neighbor_index)].y_m) - float(goal_y_m),
                    )
                    priority = float(new_cost + heuristic_m)
                    heapq.heappush(frontier, (priority, int(neighbor_index)))
                    came_from[int(neighbor_index)] = int(current_index)

        if int(goal_index) not in came_from:
            direct_distance_m = math.hypot(goal_x_m - start_x_m, goal_y_m - start_y_m)
            return RoutePlanSummary(
                route_found=False,
                start_road_id=str(start_node.road_id),
                start_lane_id=int(start_node.lane_id),
                goal_road_id=str(goal_node.road_id),
                goal_lane_id=int(goal_node.lane_id),
                optimal_lane_id=int(goal_node.lane_id),
                distance_to_destination_m=float(direct_distance_m),
                next_macro_maneuver="Continue Straight",
                route_waypoints=[],
                debug_reason=self._unreachable_route_reason(
                    start_index=int(start_index),
                    goal_index=int(goal_index),
                    start_query=start_query,
                    goal_query=goal_query,
                    reachable_node_count=len(came_from),
                ),
                start_graph_index=int(start_index),
                goal_graph_index=int(goal_index),
                start_graph_xy=(float(start_node.x_m), float(start_node.y_m)),
                goal_graph_xy=(float(goal_node.x_m), float(goal_node.y_m)),
                start_query_distance_m=float(start_query.distance_m),
                goal_query_distance_m=float(goal_query.distance_m),
            )

        route_indices: List[int] = []
        cursor = int(goal_index)
        while cursor is not None:
            route_indices.append(int(cursor))
            previous_cursor = came_from.get(int(cursor), None)
            cursor = int(previous_cursor) if previous_cursor is not None else None
        route_indices.reverse()

        route_waypoints = self._build_route_waypoints_from_indices(
            route_indices=route_indices,
            start_xy=[float(start_x_m), float(start_y_m)],
            goal_xy=[float(goal_x_m), float(goal_y_m)],
        )
        route_length_m = math.hypot(start_node.x_m - start_x_m, start_node.y_m - start_y_m)
        route_length_m += float(g_cost.get(int(goal_index), 0.0))
        route_length_m += math.hypot(goal_x_m - goal_node.x_m, goal_y_m - goal_node.y_m)

        optimal_lane_id = self._infer_optimal_lane_id(route_indices=route_indices)
        next_macro_maneuver = self._infer_macro_maneuver(route_indices=route_indices)

        return RoutePlanSummary(
            route_found=True,
            start_road_id=str(start_node.road_id),
            start_lane_id=int(start_node.lane_id),
            goal_road_id=str(goal_node.road_id),
            goal_lane_id=int(goal_node.lane_id),
            optimal_lane_id=int(optimal_lane_id),
            distance_to_destination_m=float(route_length_m),
            next_macro_maneuver=str(next_macro_maneuver),
            route_waypoints=route_waypoints,
            start_graph_index=int(start_index),
            goal_graph_index=int(goal_index),
            start_graph_xy=(float(start_node.x_m), float(start_node.y_m)),
            goal_graph_xy=(float(goal_node.x_m), float(goal_node.y_m)),
            start_query_distance_m=float(start_query.distance_m),
            goal_query_distance_m=float(goal_query.distance_m),
        )

    def plan_route(
        self,
        start_xy: Sequence[float],
        goal_xy: Sequence[float],
    ) -> RoutePlanSummary:
        """
        Plan an A* path from the ego position to the destination position.
        """

        start_x_m = float(start_xy[0]) if len(start_xy) >= 1 else 0.0
        start_y_m = float(start_xy[1]) if len(start_xy) >= 2 else 0.0
        goal_x_m = float(goal_xy[0]) if len(goal_xy) >= 1 else 0.0
        goal_y_m = float(goal_xy[1]) if len(goal_xy) >= 2 else 0.0

        start_query = self.nearest_waypoint_query(x_m=start_x_m, y_m=start_y_m)
        goal_query = self.nearest_waypoint_query(x_m=goal_x_m, y_m=goal_y_m)

        if start_query is None or goal_query is None or len(self._nodes) == 0:
            direct_distance_m = math.hypot(goal_x_m - start_x_m, goal_y_m - start_y_m)
            return RoutePlanSummary(
                route_found=False,
                start_road_id="unknown_road",
                start_lane_id=-1,
                goal_road_id="unknown_road",
                goal_lane_id=-1,
                optimal_lane_id=-1,
                distance_to_destination_m=float(direct_distance_m),
                next_macro_maneuver="Continue Straight",
                route_waypoints=[],
                debug_reason="No sampled lane-center waypoint was available near the start or goal query point.",
            )

        start_index = int(start_query.index)
        goal_index = int(goal_query.index)
        start_node = self._nodes[int(start_index)]
        goal_node = self._nodes[int(goal_index)]

        carla_route_summary = self._plan_route_with_carla(
            start_x_m=float(start_x_m),
            start_y_m=float(start_y_m),
            goal_x_m=float(goal_x_m),
            goal_y_m=float(goal_y_m),
            start_node=start_node,
            goal_node=goal_node,
        )
        if carla_route_summary is not None:
            return carla_route_summary
        return self._plan_route_with_internal_astar(
            start_x_m=float(start_x_m),
            start_y_m=float(start_y_m),
            goal_x_m=float(goal_x_m),
            goal_y_m=float(goal_y_m),
            start_index=int(start_index),
            goal_index=int(goal_index),
            start_node=start_node,
            goal_node=goal_node,
            start_query=start_query,
            goal_query=goal_query,
        )

    def plan_route_astar(
        self,
        start_xy: Sequence[float],
        goal_xy: Sequence[float],
    ) -> RoutePlanSummary:
        """Plan the shortest path on the internal lane-center graph only."""

        start_x_m = float(start_xy[0]) if len(start_xy) >= 1 else 0.0
        start_y_m = float(start_xy[1]) if len(start_xy) >= 2 else 0.0
        goal_x_m = float(goal_xy[0]) if len(goal_xy) >= 1 else 0.0
        goal_y_m = float(goal_xy[1]) if len(goal_xy) >= 2 else 0.0

        start_query = self.nearest_waypoint_query(x_m=start_x_m, y_m=start_y_m)
        goal_query = self.nearest_waypoint_query(x_m=goal_x_m, y_m=goal_y_m)

        if start_query is None or goal_query is None or len(self._nodes) == 0:
            direct_distance_m = math.hypot(goal_x_m - start_x_m, goal_y_m - start_y_m)
            return RoutePlanSummary(
                route_found=False,
                start_road_id="unknown_road",
                start_lane_id=-1,
                goal_road_id="unknown_road",
                goal_lane_id=-1,
                optimal_lane_id=-1,
                distance_to_destination_m=float(direct_distance_m),
                next_macro_maneuver="Continue Straight",
                route_waypoints=[],
                debug_reason="No sampled lane-center waypoint was available near the start or goal query point.",
            )

        start_index = int(start_query.index)
        goal_index = int(goal_query.index)
        start_node = self._nodes[int(start_index)]
        goal_node = self._nodes[int(goal_index)]

        return self._plan_route_with_internal_astar(
            start_x_m=float(start_x_m),
            start_y_m=float(start_y_m),
            goal_x_m=float(goal_x_m),
            goal_y_m=float(goal_y_m),
            start_index=int(start_index),
            goal_index=int(goal_index),
            start_node=start_node,
            goal_node=goal_node,
            start_query=start_query,
            goal_query=goal_query,
        )

    def _infer_optimal_lane_id(self, route_indices: Sequence[int]) -> int:
        if len(route_indices) == 0:
            return -1
        first_lane_id = int(self._nodes[int(route_indices[0])].lane_id)
        for route_index in route_indices[1:]:
            node_lane_id = int(self._nodes[int(route_index)].lane_id)
            if node_lane_id != first_lane_id:
                return int(node_lane_id)
        return int(self._nodes[int(route_indices[-1])].lane_id)

    def _infer_macro_maneuver(self, route_indices: Sequence[int]) -> str:
        if len(route_indices) <= 1:
            return "Continue Straight"

        start_node = self._nodes[int(route_indices[0])]
        start_heading_rad = float(start_node.heading_rad)
        encountered_intersection = False

        for route_index in route_indices[1:]:
            node = self._nodes[int(route_index)]
            encountered_intersection = encountered_intersection or bool(node.is_intersection)
            if not encountered_intersection:
                continue

            heading_delta_rad = self._wrap_angle(float(node.heading_rad) - float(start_heading_rad))
            if heading_delta_rad >= float(self._INTERSECTION_TURN_THRESHOLD_RAD):
                return "Left Turn"
            if heading_delta_rad <= -float(self._INTERSECTION_TURN_THRESHOLD_RAD):
                return "Right Turn"

        return "Continue Straight"

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
import threading
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .carla_lane_graph import (
    INVALID_LANE_ID,
    canonical_lane_id_for_waypoint,
    canonical_lane_ids_for_waypoint,
    canonical_lane_waypoints,
    carla_waypoint_graph_key,
    direction_key,
)


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
    current_road_option: str = "LANEFOLLOW"
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

        # --- Position-based cache for get_local_lane_context ---
        self._lane_context_lock = threading.Lock()
        self._lane_context_cache_state: Tuple[float, float, float | None, float | None] | None = None
        self._lane_context_cache_result: Dict[str, object] | None = None
        self._lane_context_cache_threshold_m: float = 1.0
        self._lane_context_cache_heading_threshold_rad: float = math.radians(20.0)
        self._lane_context_cache_z_threshold_m: float = 2.5

        # --- Stored initial route (compute once, look up cheaply) ---
        self._stored_route_xy: np.ndarray | None = None          # (N, 2)
        self._stored_route_cum_dists: np.ndarray | None = None   # (N,)
        self._stored_route_options: List[str] | None = None       # per-waypoint road option
        self._stored_route_lane_ids: List[int] | None = None      # per-waypoint lane id
        self._stored_route_summary: RoutePlanSummary | None = None
        self._route_info_query_state: Dict[str, Tuple[float, float, float]] = {}
        self._last_trace_per_waypoint_options: List[str] = []
        self._last_trace_per_waypoint_lane_ids: List[int] = []

    def store_initial_route(self, summary: RoutePlanSummary, per_waypoint_options: List[str], per_waypoint_lane_ids: List[int]) -> None:
        """Store the initial route so get_current_route_info() can do cheap lookups."""
        self._stored_route_summary = summary
        if summary.route_waypoints and len(summary.route_waypoints) > 0:
            self._stored_route_xy = np.asarray(summary.route_waypoints, dtype=float)
            self._stored_route_cum_dists = self._route_cumulative_distances(
                route_xy=self._stored_route_xy,
            )
            self._stored_route_options = list(per_waypoint_options)
            self._stored_route_lane_ids = list(per_waypoint_lane_ids)
        else:
            self._stored_route_xy = None
            self._stored_route_cum_dists = None
            self._stored_route_options = None
            self._stored_route_lane_ids = None
        self._route_info_query_state = {}

    def replace_stored_route(
        self,
        summary: RoutePlanSummary,
        per_waypoint_options: Sequence[str],
        per_waypoint_lane_ids: Sequence[int],
    ) -> None:
        """Replace the active stored route used by get_current_route_info()."""

        self.store_initial_route(
            summary=summary,
            per_waypoint_options=list(per_waypoint_options),
            per_waypoint_lane_ids=[int(lane_id) for lane_id in list(per_waypoint_lane_ids)],
        )

    @staticmethod
    def _route_points_for_console(route_summary: RoutePlanSummary) -> List[List[float]]:
        return [
            [float(item[0]), float(item[1])]
            for item in list(getattr(route_summary, "route_waypoints", []) or [])
            if isinstance(item, Sequence) and len(item) >= 2
        ]

    def _finalize_route_plan(
        self,
        route_summary: RoutePlanSummary,
        *,
        replace_stored_route: bool = False,
    ) -> RoutePlanSummary:
        if bool(replace_stored_route) and bool(getattr(route_summary, "route_found", False)):
            self.replace_stored_route(
                summary=route_summary,
                per_waypoint_options=list(self._last_trace_per_waypoint_options),
                per_waypoint_lane_ids=list(self._last_trace_per_waypoint_lane_ids),
            )
        if bool(getattr(route_summary, "route_found", False)):
            print("new path generated by global planner")
        return route_summary

    @staticmethod
    def _default_road_option_for_node(node: WaypointNode | None) -> str:
        if node is None:
            return "LANEFOLLOW"
        maneuver_name = str(getattr(node, "maneuver", "straight") or "straight").strip().upper()
        if bool(getattr(node, "is_intersection", False)) and maneuver_name in {"LEFT", "RIGHT", "STRAIGHT"}:
            return maneuver_name
        return "LANEFOLLOW"

    def _internal_metadata_from_route_waypoints(
        self,
        route_waypoints: Sequence[Sequence[float]],
    ) -> Tuple[List[str], List[int]]:
        per_waypoint_options: List[str] = []
        per_waypoint_lane_ids: List[int] = []
        for waypoint_xy in list(route_waypoints or []):
            if not isinstance(waypoint_xy, Sequence) or len(waypoint_xy) < 2:
                per_waypoint_options.append("LANEFOLLOW")
                per_waypoint_lane_ids.append(int(INVALID_LANE_ID))
                continue
            waypoint_query = self.nearest_waypoint_query(
                x_m=float(waypoint_xy[0]),
                y_m=float(waypoint_xy[1]),
            )
            if waypoint_query is None or not (0 <= int(waypoint_query.index) < len(self._nodes)):
                per_waypoint_options.append("LANEFOLLOW")
                per_waypoint_lane_ids.append(int(INVALID_LANE_ID))
                continue
            node = self._nodes[int(waypoint_query.index)]
            per_waypoint_options.append(self._default_road_option_for_node(node))
            per_waypoint_lane_ids.append(int(node.lane_id))
        return per_waypoint_options, per_waypoint_lane_ids

    def _blocked_node_indices_for_points(
        self,
        blocked_points_xy: Sequence[Sequence[float]],
        *,
        block_radius_m: float,
        blocked_lane_ids: Sequence[int] | None = None,
    ) -> set[int]:
        normalized_points = [
            (float(point_xy[0]), float(point_xy[1]))
            for point_xy in list(blocked_points_xy or [])
            if isinstance(point_xy, Sequence) and len(point_xy) >= 2
        ]
        if len(normalized_points) == 0 or len(self._nodes) == 0:
            return set()

        lane_id_filter = None
        if blocked_lane_ids is not None:
            lane_id_filter = {int(lane_id) for lane_id in list(blocked_lane_ids or [])}

        radius_sq_m = max(0.25, float(block_radius_m)) ** 2
        blocked_indices: set[int] = set()
        for node in self._nodes:
            if lane_id_filter is not None and int(node.lane_id) not in lane_id_filter:
                continue
            for point_x_m, point_y_m in normalized_points:
                dx_m = float(node.x_m) - float(point_x_m)
                dy_m = float(node.y_m) - float(point_y_m)
                if dx_m * dx_m + dy_m * dy_m <= float(radius_sq_m):
                    blocked_indices.add(int(node.index))
                    break
        return blocked_indices

    @staticmethod
    def _road_numeric_component(road_id: object) -> int | None:
        raw_text = str(road_id).strip()
        if not raw_text:
            return None
        if ":" in raw_text:
            raw_text = raw_text.split(":", 1)[0]
        try:
            return int(raw_text)
        except Exception:
            return None

    def segment_keys_for_road(
        self,
        road_id: object,
        *,
        direction: str | None = None,
    ) -> List[Tuple[str, int]]:
        requested_road_text = str(road_id).strip()
        requested_road_numeric = self._road_numeric_component(road_id)
        normalized_direction = str(direction).strip().lower() if direction is not None else ""
        segment_keys: List[Tuple[str, int]] = []
        for node in self._nodes:
            node_road_text = str(node.road_id)
            node_road_numeric = self._road_numeric_component(node_road_text)
            matches_road = False
            if requested_road_text and ":" in requested_road_text:
                matches_road = node_road_text == requested_road_text
            elif requested_road_numeric is not None and node_road_numeric is not None:
                matches_road = int(node_road_numeric) == int(requested_road_numeric)
            if not matches_road:
                continue
            if normalized_direction and str(node.direction).strip().lower() != normalized_direction:
                continue
            segment_key = (str(node.road_id), int(node.lane_id))
            if segment_key not in segment_keys:
                segment_keys.append(segment_key)
        return segment_keys

    def segment_keys_for_road_and_lane(
        self,
        *,
        road_id: object,
        lane_id: int,
        direction: str | None = None,
    ) -> List[Tuple[str, int]]:
        return [
            segment_key
            for segment_key in self.segment_keys_for_road(
                road_id=road_id,
                direction=direction,
            )
            if int(segment_key[1]) == int(lane_id)
        ]

    def segment_keys_for_raw_carla_lane(
        self,
        *,
        road_id: object,
        lane_id: int,
        section_id: int | None = None,
        direction: str | None = None,
    ) -> List[Tuple[str, int]]:
        requested_road_numeric = self._road_numeric_component(road_id)
        requested_lane_id = int(lane_id)
        requested_section_id = None if section_id is None else int(section_id)
        normalized_direction = str(direction).strip().lower() if direction is not None else ""
        segment_keys: List[Tuple[str, int]] = []
        for node in self._nodes:
            carla_waypoint = getattr(node, "carla_waypoint", None)
            if carla_waypoint is None:
                continue
            if requested_road_numeric is not None and int(getattr(carla_waypoint, "road_id", 0)) != int(requested_road_numeric):
                continue
            if requested_section_id is not None and int(getattr(carla_waypoint, "section_id", 0)) != int(requested_section_id):
                continue
            if int(getattr(carla_waypoint, "lane_id", 0)) != int(requested_lane_id):
                continue
            if normalized_direction and str(node.direction).strip().lower() != normalized_direction:
                continue
            segment_key = (str(node.road_id), int(node.lane_id))
            if segment_key not in segment_keys:
                segment_keys.append(segment_key)
        return segment_keys

    def nearest_waypoint_carla_key(
        self,
        *,
        x_m: float,
        y_m: float,
    ) -> Tuple[int, int, int] | None:
        nearest_query = self.nearest_waypoint_query(
            x_m=float(x_m),
            y_m=float(y_m),
        )
        if nearest_query is None or not (0 <= int(nearest_query.index) < len(self._nodes)):
            return None
        carla_waypoint = getattr(self._nodes[int(nearest_query.index)], "carla_waypoint", None)
        if carla_waypoint is None:
            return None
        return (
            int(getattr(carla_waypoint, "road_id", 0)),
            int(getattr(carla_waypoint, "section_id", 0)),
            int(getattr(carla_waypoint, "lane_id", 0)),
        )

    def blocked_node_indices_for_segments(
        self,
        blocked_segments: Sequence[Tuple[object, int]] | Mapping[Tuple[object, int], object] | None,
    ) -> set[int]:
        normalized_segments = set()
        if isinstance(blocked_segments, Mapping):
            blocked_segment_items = list(blocked_segments.keys())
        else:
            blocked_segment_items = list(blocked_segments or [])
        for raw_segment in blocked_segment_items:
            if not isinstance(raw_segment, (list, tuple)) or len(raw_segment) < 2:
                continue
            normalized_segments.add((str(raw_segment[0]), int(raw_segment[1])))
        if len(normalized_segments) == 0:
            return set()
        return {
            int(node.index)
            for node in self._nodes
            if (str(node.road_id), int(node.lane_id)) in normalized_segments
        }

    @staticmethod
    def _normalized_segment_penalties(
        segment_penalties: Mapping[Tuple[str, int], float] | None,
    ) -> Dict[Tuple[str, int], float]:
        normalized_penalties: Dict[Tuple[str, int], float] = {}
        for raw_key, raw_penalty in dict(segment_penalties or {}).items():
            if not isinstance(raw_key, (list, tuple)) or len(raw_key) < 2:
                continue
            normalized_penalties[(str(raw_key[0]), int(raw_key[1]))] = float(raw_penalty)
        return normalized_penalties

    def route_penalty_from_waypoints(
        self,
        route_waypoints: Sequence[Sequence[float]] | None,
        segment_penalties: Mapping[Tuple[str, int], float] | None,
    ) -> float:
        normalized_penalties = self._normalized_segment_penalties(segment_penalties)
        if len(normalized_penalties) == 0:
            return 0.0

        visited_segments: set[Tuple[str, int]] = set()
        total_penalty = 0.0
        for waypoint_xy in list(route_waypoints or []):
            if not isinstance(waypoint_xy, Sequence) or len(waypoint_xy) < 2:
                continue
            waypoint_query = self.nearest_waypoint_query(
                x_m=float(waypoint_xy[0]),
                y_m=float(waypoint_xy[1]),
            )
            if waypoint_query is None:
                continue
            segment_key = (
                str(getattr(waypoint_query, "road_id", "")),
                int(getattr(waypoint_query, "lane_id", INVALID_LANE_ID)),
            )
            if segment_key in visited_segments:
                continue
            visited_segments.add(segment_key)
            total_penalty += float(normalized_penalties.get(segment_key, 0.0))
        return float(total_penalty)

    def get_current_route_info(
        self,
        x_m: float,
        y_m: float,
        query_key: str | None = None,
    ) -> RoutePlanSummary:
        """Return route info by looking up ego position on the stored initial route.

        This replaces repeated A* plan_route() calls with a single O(N) nearest-
        waypoint lookup on the pre-computed route, then derives next_macro_maneuver
        and optimal_lane_id from the remaining portion of the route.
        """
        if self._stored_route_xy is None or self._stored_route_summary is None:
            return self._stored_route_summary or RoutePlanSummary(
                route_found=False,
                start_road_id="unknown_road", start_lane_id=int(INVALID_LANE_ID),
                goal_road_id="unknown_road", goal_lane_id=int(INVALID_LANE_ID),
                optimal_lane_id=int(INVALID_LANE_ID), distance_to_destination_m=0.0,
                next_macro_maneuver="Continue Straight", route_waypoints=[],
            )

        route_index = self._stored_route_index_for_position(
            x_m=float(x_m),
            y_m=float(y_m),
            query_key=query_key,
        )

        # Remaining route from ego's position onward
        remaining_options = self._stored_route_options[route_index:]
        remaining_lane_ids = self._stored_route_lane_ids[route_index:]
        remaining_xy = self._stored_route_xy[route_index:]
        current_option = "LANEFOLLOW"
        if self._stored_route_options is not None and len(self._stored_route_options) > int(route_index):
            current_option = str(self._stored_route_options[int(route_index)] or "LANEFOLLOW")

        # Next macro maneuver from the remaining route. If the blue dot is
        # still inside the previous junction's turn block, skip that active
        # block so the returned maneuver describes the *upcoming* decision.
        next_maneuver = self._next_macro_maneuver_from_road_options(
            remaining_options,
            skip_active_turn_block=bool(str(current_option).strip().upper() in {"LEFT", "RIGHT", "STRAIGHT"}),
        )

        # Optimal lane = the lane the global route is travelling through at the
        # blue dot's closest position on the stored route. The behavior planner
        # then decides whether to stay on that lane or move away from it.
        optimal_lane_id = self._active_lane_id_from_remaining_sequence(
            remaining_lane_ids=list(remaining_lane_ids),
            fallback_lane_id=int(self._stored_route_summary.optimal_lane_id),
        )

        # Remaining distance along route
        if len(remaining_xy) >= 2:
            seg_diffs = np.diff(remaining_xy, axis=0)
            remaining_distance_m = float(np.sum(np.sqrt(np.sum(seg_diffs * seg_diffs, axis=1))))
        else:
            remaining_distance_m = 0.0

        # Deduplicated road_options for the remaining portion
        dedup_options: List[str] = []
        for opt in remaining_options:
            if not dedup_options or str(dedup_options[-1]) != str(opt):
                dedup_options.append(str(opt))

        base = self._stored_route_summary
        return RoutePlanSummary(
            route_found=True,
            start_road_id=str(base.start_road_id),
            start_lane_id=int(base.start_lane_id),
            goal_road_id=str(base.goal_road_id),
            goal_lane_id=int(base.goal_lane_id),
            optimal_lane_id=int(optimal_lane_id),
            distance_to_destination_m=float(remaining_distance_m),
            next_macro_maneuver=str(next_maneuver),
            route_waypoints=remaining_xy.tolist(),
            road_options=dedup_options,
            current_road_option=str(current_option),
        )

    @staticmethod
    def _route_cumulative_distances(route_xy: np.ndarray) -> np.ndarray:
        if route_xy is None or len(route_xy) == 0:
            return np.zeros((0,), dtype=float)
        if len(route_xy) == 1:
            return np.zeros((1,), dtype=float)
        diffs = np.diff(route_xy, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs * diffs, axis=1))
        return np.concatenate((np.zeros((1,), dtype=float), np.cumsum(segment_lengths)))

    def _stored_route_index_for_position(
        self,
        x_m: float,
        y_m: float,
        query_key: str | None = None,
    ) -> int:
        if self._stored_route_xy is None or len(self._stored_route_xy) == 0:
            return 0
        if len(self._stored_route_xy) == 1:
            return 0

        route_cum_dists = self._stored_route_cum_dists
        if route_cum_dists is None or len(route_cum_dists) != len(self._stored_route_xy):
            route_cum_dists = self._route_cumulative_distances(self._stored_route_xy)
            self._stored_route_cum_dists = route_cum_dists

        candidates: List[Tuple[float, float, int]] = []
        best_distance_m = float("inf")
        query_x_m = float(x_m)
        query_y_m = float(y_m)
        route_xy = self._stored_route_xy
        ambiguity_margin_m = max(0.75, 0.75 * float(self._route_sample_distance_m))

        for segment_index in range(len(route_xy) - 1):
            ax_m = float(route_xy[segment_index][0])
            ay_m = float(route_xy[segment_index][1])
            bx_m = float(route_xy[segment_index + 1][0])
            by_m = float(route_xy[segment_index + 1][1])
            dx_m = float(bx_m) - float(ax_m)
            dy_m = float(by_m) - float(ay_m)
            segment_len_sq = dx_m * dx_m + dy_m * dy_m
            if float(segment_len_sq) <= 1.0e-9:
                projection = 0.0
                projected_x_m = float(ax_m)
                projected_y_m = float(ay_m)
                progress_m = float(route_cum_dists[segment_index])
            else:
                projection = (
                    (float(query_x_m) - float(ax_m)) * float(dx_m)
                    + (float(query_y_m) - float(ay_m)) * float(dy_m)
                ) / float(segment_len_sq)
                projection = min(1.0, max(0.0, float(projection)))
                projected_x_m = float(ax_m) + float(projection) * float(dx_m)
                projected_y_m = float(ay_m) + float(projection) * float(dy_m)
                progress_m = (
                    float(route_cum_dists[segment_index])
                    + float(projection) * math.sqrt(float(segment_len_sq))
                )

            distance_m = math.hypot(
                float(query_x_m) - float(projected_x_m),
                float(query_y_m) - float(projected_y_m),
            )
            best_distance_m = min(float(best_distance_m), float(distance_m))
            candidates.append((float(distance_m), float(progress_m), int(segment_index)))

        plausible_candidates = [
            candidate
            for candidate in candidates
            if float(candidate[0]) <= float(best_distance_m) + float(ambiguity_margin_m)
        ]
        selected_candidates = plausible_candidates if len(plausible_candidates) > 0 else candidates
        if len(selected_candidates) == 0:
            return 0

        selected_candidate = min(
            selected_candidates,
            key=lambda candidate: (float(candidate[1]), float(candidate[0])),
        )

        if query_key is not None:
            cache_key = str(query_key).strip()
            if cache_key:
                previous_state = self._route_info_query_state.get(cache_key)
                if previous_state is not None:
                    previous_progress_m, previous_x_m, previous_y_m = previous_state
                    movement_m = math.hypot(
                        float(query_x_m) - float(previous_x_m),
                        float(query_y_m) - float(previous_y_m),
                    )
                    progress_backtrack_m = max(
                        1.0,
                        1.5 * float(self._route_sample_distance_m),
                    )
                    progress_advance_m = max(
                        4.0,
                        float(movement_m) + 3.0 * float(self._route_sample_distance_m),
                    )
                    preferred_progress_m = float(previous_progress_m) + min(
                        float(movement_m),
                        max(1.0, 2.0 * float(self._route_sample_distance_m)),
                    )
                    continuity_candidates = [
                        candidate
                        for candidate in selected_candidates
                        if (
                            float(previous_progress_m) - float(progress_backtrack_m)
                            <= float(candidate[1])
                            <= float(previous_progress_m) + float(progress_advance_m)
                        )
                    ]
                    if len(continuity_candidates) > 0:
                        selected_candidate = min(
                            continuity_candidates,
                            key=lambda candidate: (
                                abs(float(candidate[1]) - float(preferred_progress_m)),
                                float(candidate[0]),
                                float(candidate[1]),
                            ),
                        )

                self._route_info_query_state[cache_key] = (
                    float(selected_candidate[1]),
                    float(query_x_m),
                    float(query_y_m),
                )

        _distance_m, _progress_m, segment_index = selected_candidate
        return int(segment_index)

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
                    lane_id=int(waypoint.get("lane_id", INVALID_LANE_ID)),
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

    def _candidate_waypoint_queries(
        self,
        *,
        x_m: float,
        y_m: float,
        max_count: int = 24,
        direction: str | None = None,
        road_id: str | None = None,
    ) -> List[WaypointQueryResult]:
        if len(self._nodes) == 0:
            return []
        candidate_count = min(len(self._nodes), max(1, int(max_count)))
        delta_x = self._node_x_m - float(x_m)
        delta_y = self._node_y_m - float(y_m)
        distance_sq = delta_x * delta_x + delta_y * delta_y
        candidate_indices = np.argpartition(distance_sq, candidate_count - 1)[:candidate_count]
        normalized_direction = str(direction).strip().lower() if direction is not None else ""
        normalized_road_id = str(road_id).strip() if road_id is not None else ""

        queries: List[WaypointQueryResult] = []
        for candidate_index in candidate_indices:
            node = self._nodes[int(candidate_index)]
            if normalized_direction and str(node.direction).strip().lower() != normalized_direction:
                continue
            if normalized_road_id and str(node.road_id).strip() != normalized_road_id:
                continue
            queries.append(
                WaypointQueryResult(
                    index=int(node.index),
                    distance_m=float(math.sqrt(float(distance_sq[int(candidate_index)]))),
                    x_m=float(node.x_m),
                    y_m=float(node.y_m),
                    road_id=str(node.road_id),
                    lane_id=int(node.lane_id),
                    direction=str(node.direction),
                )
            )
        queries.sort(key=lambda query: float(query.distance_m))
        return queries

    @staticmethod
    def _append_unique_query(
        candidates: List[WaypointQueryResult],
        candidate: WaypointQueryResult | None,
    ) -> None:
        if candidate is None:
            return
        if any(int(existing.index) == int(candidate.index) for existing in candidates):
            return
        candidates.append(candidate)

    def _waypoint_query_for_carla_waypoint(self, carla_waypoint) -> WaypointQueryResult | None:
        if carla_waypoint is None or len(self._nodes) == 0:
            return None

        internal_lane_id = self._internal_lane_id_for_carla_waypoint(carla_waypoint)
        road_id = int(getattr(carla_waypoint, "road_id", 0))
        section_id = int(getattr(carla_waypoint, "section_id", 0))
        road_key = f"{int(road_id)}:{int(section_id)}"
        waypoint_location = getattr(getattr(carla_waypoint, "transform", None), "location", None)
        if waypoint_location is None:
            return None
        waypoint_x_m = float(getattr(waypoint_location, "x", 0.0))
        waypoint_y_m = float(getattr(waypoint_location, "y", 0.0))

        best_node = None
        best_distance_sq = float("inf")
        for node in self._nodes:
            if str(node.road_id) != str(road_key):
                continue
            if internal_lane_id is not None and int(node.lane_id) != int(internal_lane_id):
                continue
            dx_m = float(node.x_m) - float(waypoint_x_m)
            dy_m = float(node.y_m) - float(waypoint_y_m)
            distance_sq = float(dx_m * dx_m + dy_m * dy_m)
            if distance_sq < float(best_distance_sq):
                best_distance_sq = float(distance_sq)
                best_node = node

        if best_node is not None:
            return WaypointQueryResult(
                index=int(best_node.index),
                distance_m=float(math.sqrt(best_distance_sq)),
                x_m=float(best_node.x_m),
                y_m=float(best_node.y_m),
                road_id=str(best_node.road_id),
                lane_id=int(best_node.lane_id),
                direction=str(best_node.direction),
            )

        waypoint_heading_rad = math.radians(
            float(getattr(getattr(carla_waypoint, "transform", None), "rotation", None).yaw)
        ) if getattr(getattr(carla_waypoint, "transform", None), "rotation", None) is not None else None
        fallback_index = self._nearest_waypoint_index_for_heading(
            x_m=float(waypoint_x_m),
            y_m=float(waypoint_y_m),
            heading_rad=waypoint_heading_rad,
        )
        if fallback_index is None:
            return None
        fallback_node = self._nodes[int(fallback_index)]
        return WaypointQueryResult(
            index=int(fallback_node.index),
            distance_m=float(
                math.hypot(
                    float(fallback_node.x_m) - float(waypoint_x_m),
                    float(fallback_node.y_m) - float(waypoint_y_m),
                )
            ),
            x_m=float(fallback_node.x_m),
            y_m=float(fallback_node.y_m),
            road_id=str(fallback_node.road_id),
            lane_id=int(fallback_node.lane_id),
            direction=str(fallback_node.direction),
        )

    def _route_with_endpoint_candidates(
        self,
        *,
        start_x_m: float,
        start_y_m: float,
        goal_x_m: float,
        goal_y_m: float,
        start_query: WaypointQueryResult,
        goal_query: WaypointQueryResult,
        blocked_node_indices: set[int] | None = None,
        segment_penalties: Mapping[Tuple[str, int], float] | None = None,
        start_candidate_limit: int = 12,
        goal_candidate_limit: int = 32,
    ) -> RoutePlanSummary:
        def _plan_for_queries(
            candidate_start_query: WaypointQueryResult,
            candidate_goal_query: WaypointQueryResult,
        ) -> RoutePlanSummary:
            return self._plan_route_with_internal_astar(
                start_x_m=float(start_x_m),
                start_y_m=float(start_y_m),
                goal_x_m=float(goal_x_m),
                goal_y_m=float(goal_y_m),
                start_index=int(candidate_start_query.index),
                goal_index=int(candidate_goal_query.index),
                start_node=self._nodes[int(candidate_start_query.index)],
                goal_node=self._nodes[int(candidate_goal_query.index)],
                start_query=candidate_start_query,
                goal_query=candidate_goal_query,
                blocked_node_indices=blocked_node_indices,
                segment_penalties=segment_penalties,
            )

        route_summary = _plan_for_queries(start_query, goal_query)
        if bool(getattr(route_summary, "route_found", False)):
            return route_summary

        additional_start_candidates: List[WaypointQueryResult] = []
        for candidate_start_query in self._candidate_waypoint_queries(
            x_m=float(start_x_m),
            y_m=float(start_y_m),
            max_count=int(start_candidate_limit),
            direction=str(start_query.direction),
            road_id=str(start_query.road_id),
        ):
            self._append_unique_query(additional_start_candidates, candidate_start_query)
        for candidate_start_query in self._candidate_waypoint_queries(
            x_m=float(start_x_m),
            y_m=float(start_y_m),
            max_count=int(start_candidate_limit),
            direction=str(start_query.direction),
        ):
            self._append_unique_query(additional_start_candidates, candidate_start_query)

        additional_goal_candidates: List[WaypointQueryResult] = []
        for candidate_goal_query in self._candidate_waypoint_queries(
            x_m=float(goal_x_m),
            y_m=float(goal_y_m),
            max_count=int(goal_candidate_limit),
            direction=str(start_query.direction),
        ):
            self._append_unique_query(additional_goal_candidates, candidate_goal_query)

        additional_start_candidates = [
            candidate
            for candidate in additional_start_candidates
            if int(candidate.index) != int(start_query.index)
        ]
        additional_goal_candidates = [
            candidate
            for candidate in additional_goal_candidates
            if int(candidate.index) != int(goal_query.index)
        ]

        for candidate_start_query in additional_start_candidates:
            candidate_summary = _plan_for_queries(candidate_start_query, goal_query)
            if bool(getattr(candidate_summary, "route_found", False)):
                return candidate_summary

        for candidate_goal_query in additional_goal_candidates:
            candidate_summary = _plan_for_queries(start_query, candidate_goal_query)
            if bool(getattr(candidate_summary, "route_found", False)):
                return candidate_summary

        for candidate_start_query in additional_start_candidates:
            for candidate_goal_query in additional_goal_candidates:
                candidate_summary = _plan_for_queries(
                    candidate_start_query,
                    candidate_goal_query,
                )
                if bool(getattr(candidate_summary, "route_found", False)):
                    return candidate_summary

        return route_summary

    def _internal_lane_id_for_carla_waypoint(self, carla_waypoint) -> int | None:
        if carla_waypoint is None:
            return None
        canonical_lane_id = canonical_lane_id_for_waypoint(carla_waypoint)
        if int(canonical_lane_id) != int(INVALID_LANE_ID):
            return int(canonical_lane_id)
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

    def _project_runtime_waypoint(self, x_m: float, y_m: float, z_m: float | None = None):
        world_map = getattr(self, "_world_map", None)
        if world_map is None or not hasattr(world_map, "get_waypoint"):
            return None

        location_template = None
        for node in getattr(self, "_nodes", []):
            candidate_location = getattr(
                getattr(getattr(getattr(node, "carla_waypoint", None), "transform", None), "location", None),
                "__class__",
                None,
            )
            if candidate_location is not None:
                location_template = getattr(getattr(node.carla_waypoint, "transform", None), "location", None)
                break
        if location_template is None:
            return None

        location_cls = location_template.__class__
        try:
            query_location = location_cls(
                x=float(x_m),
                y=float(y_m),
                z=float(getattr(location_template, "z", 0.0) if z_m is None else z_m),
            )
            # Specify LaneType.Driving so we never project onto a shoulder,
            # parking lane, or other non-driving surface.  carla is already
            # imported by this point (we hold a live world_map reference), so
            # we retrieve the module from sys.modules without a hard import.
            _carla_mod = sys.modules.get("carla")
            if _carla_mod is not None:
                return world_map.get_waypoint(
                    query_location,
                    project_to_road=True,
                    lane_type=_carla_mod.LaneType.Driving,
                )
            return world_map.get_waypoint(query_location, project_to_road=True)
        except Exception:
            return None

    def _route_waypoints_from_carla_trace(
        self,
        route_trace: Sequence[tuple[object, object]],
    ) -> tuple[List[List[float]], List[int], List[str], str, int | None, List[str], List[int]]:
        route_waypoints: List[List[float]] = []
        route_lane_ids: List[int] = []
        road_options: List[str] = []
        per_waypoint_options: List[str] = []
        per_waypoint_lane_ids: List[int] = []
        preferred_lane_id: int | None = None

        for carla_waypoint, road_option in route_trace:
            transform = getattr(carla_waypoint, "transform", None)
            location = getattr(transform, "location", None)
            if location is None:
                continue
            waypoint_xy = [float(getattr(location, "x", 0.0)), float(getattr(location, "y", 0.0))]
            option_name = str(getattr(road_option, "name", road_option)).upper()
            lane_id = self._internal_lane_id_for_carla_waypoint(carla_waypoint)

            is_new_wp = len(route_waypoints) == 0 or math.hypot(
                float(route_waypoints[-1][0]) - float(waypoint_xy[0]),
                float(route_waypoints[-1][1]) - float(waypoint_xy[1]),
            ) > 1e-3
            if is_new_wp:
                route_waypoints.append(waypoint_xy)
                per_waypoint_options.append(str(option_name))
                per_waypoint_lane_ids.append(
                    int(lane_id) if lane_id is not None else int(INVALID_LANE_ID)
                )

            if lane_id is not None:
                route_lane_ids.append(int(lane_id))
                if preferred_lane_id is None:
                    preferred_lane_id = int(lane_id)
                elif int(lane_id) != int(preferred_lane_id):
                    preferred_lane_id = int(lane_id)

            if option_name and (len(road_options) == 0 or str(road_options[-1]) != str(option_name)):
                road_options.append(str(option_name))

            if option_name in {"CHANGELANELEFT", "CHANGELANERIGHT"} and lane_id is not None:
                preferred_lane_id = int(lane_id)

        if preferred_lane_id is None and len(route_lane_ids) > 0:
            preferred_lane_id = int(route_lane_ids[0])
        preferred_lane_id = self._active_lane_id_from_remaining_sequence(
            remaining_lane_ids=per_waypoint_lane_ids,
            fallback_lane_id=int(
                preferred_lane_id
                if preferred_lane_id is not None
                else int(INVALID_LANE_ID)
            ),
        )

        return (
            route_waypoints,
            route_lane_ids,
            road_options,
            self._next_macro_maneuver_from_road_options(road_options),
            preferred_lane_id,
            per_waypoint_options,
            per_waypoint_lane_ids,
        )

    def _candidate_start_carla_waypoints(
        self,
        *,
        start_waypoint,
        start_x_m: float,
        start_y_m: float,
        max_count: int = 12,
    ) -> List[object]:
        candidates: List[object] = []
        seen_keys: set[tuple[int, int, int, float]] = set()

        def _append_candidate(waypoint) -> None:
            if waypoint is None:
                return
            try:
                waypoint_key = carla_waypoint_graph_key(waypoint)
            except Exception:
                return
            if waypoint_key in seen_keys:
                return
            seen_keys.add(waypoint_key)
            candidates.append(waypoint)

        _append_candidate(start_waypoint)
        for candidate_waypoint in canonical_lane_waypoints(start_waypoint):
            _append_candidate(candidate_waypoint)
        for candidate_query in self._candidate_waypoint_queries(
            x_m=float(start_x_m),
            y_m=float(start_y_m),
            max_count=int(max_count),
            direction=(
                str(direction_key(int(getattr(start_waypoint, "lane_id", 0))))
                if start_waypoint is not None
                else None
            ),
            road_id=(
                f"{int(getattr(start_waypoint, 'road_id', 0))}:{int(getattr(start_waypoint, 'section_id', 0))}"
                if start_waypoint is not None
                else None
            ),
        ):
            _append_candidate(getattr(self._nodes[int(candidate_query.index)], "carla_waypoint", None))
        return candidates

    def _carla_trace_segment_penalty(
        self,
        route_trace: Sequence[tuple[object, object]],
        *,
        segment_penalties: Mapping[Tuple[str, int], float] | None,
    ) -> float:
        normalized_penalties = self._normalized_segment_penalties(segment_penalties)
        if len(normalized_penalties) == 0:
            return 0.0
        traversed_segments: set[Tuple[str, int]] = set()
        penalty_m = 0.0
        for route_item in list(route_trace or []):
            if not isinstance(route_item, Sequence) or len(route_item) < 1:
                continue
            carla_waypoint = route_item[0]
            if carla_waypoint is None:
                continue
            internal_lane_id = self._internal_lane_id_for_carla_waypoint(carla_waypoint)
            if internal_lane_id is None:
                continue
            segment_key = (
                f"{int(getattr(carla_waypoint, 'road_id', 0))}:{int(getattr(carla_waypoint, 'section_id', 0))}",
                int(internal_lane_id),
            )
            if segment_key in traversed_segments:
                continue
            traversed_segments.add(segment_key)
            penalty_m += float(normalized_penalties.get(segment_key, 0.0))
        return float(penalty_m)

    def _set_road_edges_inf(
        self,
        *,
        raw_road_id: int,
        raw_lane_id: int,
        raw_section_id: int | None = None,
    ) -> int:
        graph = getattr(self._carla_route_planner, "_graph", None)
        road_id_to_edge = getattr(self._carla_route_planner, "_road_id_to_edge", None)
        if graph is None or road_id_to_edge is None:
            return 0

        blocked_edge_count = 0
        sections = road_id_to_edge.get(int(raw_road_id), {})
        for section_id, section_edges in dict(sections or {}).items():
            if raw_section_id is not None and int(section_id) != int(raw_section_id):
                continue
            edge_info = dict(section_edges or {}).get(int(raw_lane_id))
            if edge_info is None or len(edge_info) < 2:
                continue
            edge_start, edge_end = edge_info[0], edge_info[1]
            if not graph.has_edge(edge_start, edge_end):
                continue
            graph[edge_start][edge_end]["length"] = float("inf")
            if "weight" in graph[edge_start][edge_end]:
                graph[edge_start][edge_end]["weight"] = float("inf")
            blocked_edge_count += 1
        return int(blocked_edge_count)

    @staticmethod
    def _next_macro_maneuver_from_road_options(
        road_options: Sequence[str],
        *,
        skip_active_turn_block: bool = False,
    ) -> str:
        normalized_options = [
            str(option).strip().upper()
            for option in list(road_options or [])
            if str(option).strip()
        ]
        if skip_active_turn_block and len(normalized_options) > 0:
            while len(normalized_options) > 0 and normalized_options[0] in {"LEFT", "RIGHT", "STRAIGHT"}:
                normalized_options = normalized_options[1:]
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

    @staticmethod
    def _active_lane_id_from_remaining_sequence(
        remaining_lane_ids: Sequence[int],
        fallback_lane_id: int,
    ) -> int:
        """Return the lane id of the route at the current remaining position.

        The route-optimal lane should update exactly when the blue dot reaches
        a point on the global route where the route itself changes lane.
        """
        for lane_id in list(remaining_lane_ids or []):
            if int(lane_id) != int(INVALID_LANE_ID):
                return int(lane_id)
        return int(fallback_lane_id)

    @staticmethod
    def _last_valid_lane_id_before_index(
        lane_ids: Sequence[int],
        end_index: int,
    ) -> int:
        stop_index = min(max(0, int(end_index)), len(list(lane_ids or [])))
        for idx in range(int(stop_index) - 1, -1, -1):
            lane_id = int(lane_ids[idx])
            if lane_id != int(INVALID_LANE_ID):
                return int(lane_id)
        return int(INVALID_LANE_ID)

    def _route_optimal_lane_id_for_stored_route_index(
        self,
        route_index: int,
        fallback_lane_id: int,
    ) -> int:
        """Return the route-optimal lane for the blue-dot position.

        For intersection maneuvers, the optimal lane is the preparatory lane on
        the approach road, not the lane id of the junction connector. That
        preparatory lane stays active until the turn block is fully passed.
        Outside turn blocks, the route-optimal lane follows the active route
        lane at the current remaining position.
        """
        stored_options = list(self._stored_route_options or [])
        stored_lane_ids = list(self._stored_route_lane_ids or [])
        if len(stored_options) == 0 or len(stored_lane_ids) == 0:
            return int(fallback_lane_id)

        index = max(0, min(int(route_index), len(stored_options) - 1))
        normalized_options = [str(option).strip().upper() for option in stored_options]
        turn_options = {"LEFT", "RIGHT", "STRAIGHT"}

        current_lane_id = self._active_lane_id_from_remaining_sequence(
            remaining_lane_ids=stored_lane_ids[index:],
            fallback_lane_id=int(fallback_lane_id),
        )

        current_option = normalized_options[index]
        if current_option in turn_options:
            block_start = int(index)
            while block_start > 0 and normalized_options[block_start - 1] in turn_options:
                block_start -= 1
            preparatory_lane_id = self._last_valid_lane_id_before_index(
                lane_ids=stored_lane_ids,
                end_index=int(block_start),
            )
            if int(preparatory_lane_id) != int(INVALID_LANE_ID):
                return int(preparatory_lane_id)
            return int(current_lane_id)

        for future_index in range(int(index), len(normalized_options)):
            if normalized_options[future_index] in turn_options:
                preparatory_lane_id = self._last_valid_lane_id_before_index(
                    lane_ids=stored_lane_ids,
                    end_index=int(future_index),
                )
                if int(preparatory_lane_id) != int(INVALID_LANE_ID):
                    return int(preparatory_lane_id)
                break

        return int(current_lane_id)

    def _build_route_summary_from_carla_trace(
        self,
        route_trace: Sequence[tuple[object, object]],
        start_xy: Sequence[float],
        goal_xy: Sequence[float],
    ) -> RoutePlanSummary | None:
        route_waypoints, route_lane_ids, road_options, next_macro_maneuver, preferred_lane_id, per_wp_options, per_wp_lane_ids = self._route_waypoints_from_carla_trace(route_trace)
        if len(route_waypoints) == 0:
            return None

        start_x_m = float(start_xy[0]) if len(start_xy) >= 1 else 0.0
        start_y_m = float(start_xy[1]) if len(start_xy) >= 2 else 0.0
        goal_x_m = float(goal_xy[0]) if len(goal_xy) >= 1 else 0.0
        goal_y_m = float(goal_xy[1]) if len(goal_xy) >= 2 else 0.0

        start_query = self.nearest_waypoint_query(x_m=float(start_x_m), y_m=float(start_y_m))
        goal_query = self.nearest_waypoint_query(x_m=float(goal_x_m), y_m=float(goal_y_m))
        start_lane_id = int(getattr(start_query, "lane_id", INVALID_LANE_ID))
        goal_lane_id = int(getattr(goal_query, "lane_id", INVALID_LANE_ID))
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

        summary = RoutePlanSummary(
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
        self._last_trace_per_waypoint_options = list(per_wp_options)
        self._last_trace_per_waypoint_lane_ids = [int(lane_id) for lane_id in list(per_wp_lane_ids)]
        # Only store on the very first successful call (startup).  Subsequent
        # calls (e.g. from the behavior-planner prompt builder) must NOT
        # overwrite the stored route, because get_current_route_info() relies
        # on the full original route for its O(N) nearest-waypoint lookup.
        if self._stored_route_xy is None:
            self.store_initial_route(summary, per_wp_options, per_wp_lane_ids)
        return summary

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
        replace_stored_route: bool = False,
    ) -> RoutePlanSummary:
        if self._carla_route_planner is None:
            start_xy = self._location_xy(start_location)
            goal_xy = self._location_xy(goal_location)
            start_x_m = float(start_xy[0]) if start_xy is not None else 0.0
            start_y_m = float(start_xy[1]) if start_xy is not None else 0.0
            goal_x_m = float(goal_xy[0]) if goal_xy is not None else 0.0
            goal_y_m = float(goal_xy[1]) if goal_xy is not None else 0.0
            load_error = _get_carla_route_planner_load_error()
            return self._finalize_route_plan(RoutePlanSummary(
                route_found=False,
                start_road_id="unknown_road",
                start_lane_id=int(INVALID_LANE_ID),
                goal_road_id="unknown_road",
                goal_lane_id=int(INVALID_LANE_ID),
                optimal_lane_id=int(INVALID_LANE_ID),
                distance_to_destination_m=float(math.hypot(goal_x_m - start_x_m, goal_y_m - start_y_m)),
                next_macro_maneuver="Continue Straight",
                route_waypoints=[],
                debug_reason=(
                    "CARLA GlobalRoutePlanner is not available for the loaded map."
                    if not load_error
                    else f"CARLA GlobalRoutePlanner is not available: {load_error}"
                ),
            ))

        start_xy = self._location_xy(start_location)
        goal_xy = self._location_xy(goal_location)
        if start_xy is None or goal_xy is None:
            return self._finalize_route_plan(RoutePlanSummary(
                route_found=False,
                start_road_id="unknown_road",
                start_lane_id=int(INVALID_LANE_ID),
                goal_road_id="unknown_road",
                goal_lane_id=int(INVALID_LANE_ID),
                optimal_lane_id=int(INVALID_LANE_ID),
                distance_to_destination_m=0.0,
                next_macro_maneuver="Continue Straight",
                route_waypoints=[],
                debug_reason="Start or goal location for CARLA GlobalRoutePlanner was invalid.",
            ))

        try:
            route_trace = list(self._carla_route_planner.trace_route(start_location, goal_location))
        except Exception as exc:
            return self._finalize_route_plan(RoutePlanSummary(
                route_found=False,
                start_road_id="unknown_road",
                start_lane_id=int(INVALID_LANE_ID),
                goal_road_id="unknown_road",
                goal_lane_id=int(INVALID_LANE_ID),
                optimal_lane_id=int(INVALID_LANE_ID),
                distance_to_destination_m=float(math.hypot(float(goal_xy[0]) - float(start_xy[0]), float(goal_xy[1]) - float(start_xy[1]))),
                next_macro_maneuver="Continue Straight",
                route_waypoints=[],
                debug_reason=f"CARLA GlobalRoutePlanner.trace_route failed: {exc}",
            ))
        if len(route_trace) == 0:
            return self._finalize_route_plan(RoutePlanSummary(
                route_found=False,
                start_road_id="unknown_road",
                start_lane_id=int(INVALID_LANE_ID),
                goal_road_id="unknown_road",
                goal_lane_id=int(INVALID_LANE_ID),
                optimal_lane_id=int(INVALID_LANE_ID),
                distance_to_destination_m=float(math.hypot(float(goal_xy[0]) - float(start_xy[0]), float(goal_xy[1]) - float(start_xy[1]))),
                next_macro_maneuver="Continue Straight",
                route_waypoints=[],
                debug_reason="CARLA GlobalRoutePlanner returned an empty route trace.",
            ))

        route_summary = self._build_route_summary_from_carla_trace(
            route_trace=route_trace,
            start_xy=list(fallback_start_xy or start_xy),
            goal_xy=list(fallback_goal_xy or goal_xy),
        )
        if route_summary is not None:
            return self._finalize_route_plan(
                route_summary,
                replace_stored_route=bool(replace_stored_route),
            )
        return self._finalize_route_plan(RoutePlanSummary(
            route_found=False,
            start_road_id="unknown_road",
            start_lane_id=int(INVALID_LANE_ID),
            goal_road_id="unknown_road",
            goal_lane_id=int(INVALID_LANE_ID),
            optimal_lane_id=int(INVALID_LANE_ID),
            distance_to_destination_m=float(math.hypot(float(goal_xy[0]) - float(start_xy[0]), float(goal_xy[1]) - float(start_xy[1]))),
            next_macro_maneuver="Continue Straight",
            route_waypoints=[],
            debug_reason="CARLA GlobalRoutePlanner returned a route trace but no usable waypoints were extracted.",
        ))

    def plan_route_from_locations_with_segment_penalties(
        self,
        *,
        start_location,
        goal_location,
        segment_penalties: Mapping[Tuple[str, int], float],
        replace_stored_route: bool = False,
        start_waypoint=None,
        start_candidate_limit: int = 12,
    ) -> RoutePlanSummary:
        if self._carla_route_planner is None:
            return self.plan_route_from_locations(
                start_location=start_location,
                goal_location=goal_location,
                replace_stored_route=replace_stored_route,
            )

        start_xy = self._location_xy(start_location)
        goal_xy = self._location_xy(goal_location)
        if start_xy is None or goal_xy is None:
            return RoutePlanSummary(
                route_found=False,
                start_road_id="unknown_road",
                start_lane_id=int(INVALID_LANE_ID),
                goal_road_id="unknown_road",
                goal_lane_id=int(INVALID_LANE_ID),
                optimal_lane_id=int(INVALID_LANE_ID),
                distance_to_destination_m=0.0,
                next_macro_maneuver="Continue Straight",
                route_waypoints=[],
                debug_reason="Start or goal location for CARLA penalized reroute was invalid.",
            )

        candidate_start_waypoints = self._candidate_start_carla_waypoints(
            start_waypoint=start_waypoint,
            start_x_m=float(start_xy[0]),
            start_y_m=float(start_xy[1]),
            max_count=int(start_candidate_limit),
        )
        if len(candidate_start_waypoints) == 0:
            candidate_start_waypoints = [start_waypoint] if start_waypoint is not None else []

        best_summary: RoutePlanSummary | None = None
        best_penalty_m = float("inf")
        best_distance_m = float("inf")
        failure_reason = "CARLA GlobalRoutePlanner penalized reroute did not find any valid start-lane candidate."

        for candidate_start_waypoint in list(candidate_start_waypoints or []):
            candidate_location = getattr(getattr(candidate_start_waypoint, "transform", None), "location", None)
            if candidate_location is None:
                continue
            try:
                route_trace = list(self._carla_route_planner.trace_route(candidate_location, goal_location))
            except Exception as exc:
                failure_reason = f"CARLA GlobalRoutePlanner.trace_route failed during penalized reroute: {exc}"
                continue
            if len(route_trace) == 0:
                failure_reason = "CARLA GlobalRoutePlanner returned an empty route trace during penalized reroute."
                continue

            route_summary = self._build_route_summary_from_carla_trace(
                route_trace=route_trace,
                start_xy=[float(start_xy[0]), float(start_xy[1])],
                goal_xy=[float(goal_xy[0]), float(goal_xy[1])],
            )
            if route_summary is None or not bool(getattr(route_summary, "route_found", False)):
                failure_reason = "CARLA penalized reroute trace did not produce a usable route summary."
                continue

            penalty_m = self._carla_trace_segment_penalty(
                route_trace=route_trace,
                segment_penalties=segment_penalties,
            )
            route_distance_m = float(getattr(route_summary, "distance_to_destination_m", float("inf")))
            if (
                best_summary is None
                or float(penalty_m) < float(best_penalty_m)
                or (
                    math.isclose(float(penalty_m), float(best_penalty_m))
                    and float(route_distance_m) < float(best_distance_m)
                )
            ):
                best_summary = route_summary
                best_penalty_m = float(penalty_m)
                best_distance_m = float(route_distance_m)
                if float(best_penalty_m) <= 0.0:
                    break

        if best_summary is not None:
            if self.route_penalty_from_waypoints(
                getattr(best_summary, "route_waypoints", []),
                segment_penalties=segment_penalties,
            ) > 0.0:
                return RoutePlanSummary(
                    route_found=False,
                    start_road_id="unknown_road",
                    start_lane_id=int(INVALID_LANE_ID),
                    goal_road_id="unknown_road",
                    goal_lane_id=int(INVALID_LANE_ID),
                    optimal_lane_id=int(INVALID_LANE_ID),
                    distance_to_destination_m=float(
                        math.hypot(
                            float(goal_xy[0]) - float(start_xy[0]),
                            float(goal_xy[1]) - float(start_xy[1]),
                        )
                    ),
                    next_macro_maneuver="Continue Straight",
                    route_waypoints=[],
                    debug_reason=(
                        "CARLA penalized reroute did not find any route that avoids the blocked lane segments."
                    ),
                )
            return self._finalize_route_plan(
                best_summary,
                replace_stored_route=bool(replace_stored_route),
            )

        return RoutePlanSummary(
            route_found=False,
            start_road_id="unknown_road",
            start_lane_id=int(INVALID_LANE_ID),
            goal_road_id="unknown_road",
            goal_lane_id=int(INVALID_LANE_ID),
            optimal_lane_id=int(INVALID_LANE_ID),
            distance_to_destination_m=float(math.hypot(float(goal_xy[0]) - float(start_xy[0]), float(goal_xy[1]) - float(start_xy[1]))),
            next_macro_maneuver="Continue Straight",
            route_waypoints=[],
            debug_reason=str(failure_reason),
        )

    def plan_route_carla_with_blocked_lanes(
        self,
        *,
        start_location,
        goal_location,
        blocked_raw_lanes: Sequence[Tuple[int, int | None, int]] | None,
        replace_stored_route: bool = False,
        fallback_start_xy: Sequence[float] | None = None,
        fallback_goal_xy: Sequence[float] | None = None,
    ) -> RoutePlanSummary:
        if self._carla_route_planner is None:
            return self.plan_route_from_locations(
                start_location=start_location,
                goal_location=goal_location,
                fallback_start_xy=fallback_start_xy,
                fallback_goal_xy=fallback_goal_xy,
                replace_stored_route=replace_stored_route,
            )

        normalized_blocked_lanes: List[Tuple[int, int | None, int]] = []
        for blocked_raw_lane in list(blocked_raw_lanes or []):
            if not isinstance(blocked_raw_lane, (list, tuple)) or len(blocked_raw_lane) < 3:
                continue
            normalized_blocked_lanes.append(
                (
                    int(blocked_raw_lane[0]),
                    None if blocked_raw_lane[1] is None else int(blocked_raw_lane[1]),
                    int(blocked_raw_lane[2]),
                )
            )

        blocked_edge_count = 0
        for raw_road_id, raw_section_id, raw_lane_id in normalized_blocked_lanes:
            blocked_edge_count += self._set_road_edges_inf(
                raw_road_id=int(raw_road_id),
                raw_section_id=None if raw_section_id is None else int(raw_section_id),
                raw_lane_id=int(raw_lane_id),
            )

        route_summary = self.plan_route_from_locations(
            start_location=start_location,
            goal_location=goal_location,
            fallback_start_xy=fallback_start_xy,
            fallback_goal_xy=fallback_goal_xy,
            replace_stored_route=replace_stored_route,
        )
        if (
            blocked_edge_count <= 0
            and not bool(getattr(route_summary, "route_found", False))
            and not str(getattr(route_summary, "debug_reason", "")).strip()
        ):
            setattr(
                route_summary,
                "debug_reason",
                "No CARLA route-planner edge matched the blocked raw road/lane identifiers.",
            )
        return route_summary

    def plan_route_from_locations_via_locations(
        self,
        *,
        start_location,
        goal_location,
        intermediate_locations: Sequence[object] | None = None,
        replace_stored_route: bool = False,
    ) -> RoutePlanSummary:
        if self._carla_route_planner is None:
            return self.plan_route_from_locations(
                start_location=start_location,
                goal_location=goal_location,
                replace_stored_route=replace_stored_route,
            )

        route_locations: List[object] = [start_location]
        route_locations.extend([location for location in list(intermediate_locations or []) if location is not None])
        route_locations.append(goal_location)
        if len(route_locations) < 2:
            return self.plan_route_from_locations(
                start_location=start_location,
                goal_location=goal_location,
                replace_stored_route=replace_stored_route,
            )

        start_xy = self._location_xy(start_location)
        goal_xy = self._location_xy(goal_location)
        if start_xy is None or goal_xy is None:
            return RoutePlanSummary(
                route_found=False,
                start_road_id="unknown_road",
                start_lane_id=int(INVALID_LANE_ID),
                goal_road_id="unknown_road",
                goal_lane_id=int(INVALID_LANE_ID),
                optimal_lane_id=int(INVALID_LANE_ID),
                distance_to_destination_m=0.0,
                next_macro_maneuver="Continue Straight",
                route_waypoints=[],
                debug_reason="Start or goal location for CARLA via-location reroute was invalid.",
            )

        combined_trace: List[tuple[object, object]] = []
        for segment_index in range(len(route_locations) - 1):
            segment_start = route_locations[segment_index]
            segment_goal = route_locations[segment_index + 1]
            try:
                segment_trace = list(self._carla_route_planner.trace_route(segment_start, segment_goal))
            except Exception as exc:
                return RoutePlanSummary(
                    route_found=False,
                    start_road_id="unknown_road",
                    start_lane_id=int(INVALID_LANE_ID),
                    goal_road_id="unknown_road",
                    goal_lane_id=int(INVALID_LANE_ID),
                    optimal_lane_id=int(INVALID_LANE_ID),
                    distance_to_destination_m=float(
                        math.hypot(float(goal_xy[0]) - float(start_xy[0]), float(goal_xy[1]) - float(start_xy[1]))
                    ),
                    next_macro_maneuver="Continue Straight",
                    route_waypoints=[],
                    debug_reason=f"CARLA GlobalRoutePlanner.trace_route failed during via-location reroute: {exc}",
                )
            if len(segment_trace) == 0:
                return RoutePlanSummary(
                    route_found=False,
                    start_road_id="unknown_road",
                    start_lane_id=int(INVALID_LANE_ID),
                    goal_road_id="unknown_road",
                    goal_lane_id=int(INVALID_LANE_ID),
                    optimal_lane_id=int(INVALID_LANE_ID),
                    distance_to_destination_m=float(
                        math.hypot(float(goal_xy[0]) - float(start_xy[0]), float(goal_xy[1]) - float(start_xy[1]))
                    ),
                    next_macro_maneuver="Continue Straight",
                    route_waypoints=[],
                    debug_reason="CARLA GlobalRoutePlanner returned an empty via-location route trace.",
                )
            if len(combined_trace) > 0:
                segment_trace = list(segment_trace[1:])
            combined_trace.extend(list(segment_trace))

        route_summary = self._build_route_summary_from_carla_trace(
            route_trace=combined_trace,
            start_xy=list(start_xy),
            goal_xy=list(goal_xy),
        )
        if route_summary is not None:
            finalized_summary = self._finalize_route_plan(
                route_summary,
                replace_stored_route=bool(replace_stored_route),
            )
            if bool(getattr(finalized_summary, "route_found", False)):
                print("new path generated by global planner")
            return finalized_summary

        return RoutePlanSummary(
            route_found=False,
            start_road_id="unknown_road",
            start_lane_id=int(INVALID_LANE_ID),
            goal_road_id="unknown_road",
            goal_lane_id=int(INVALID_LANE_ID),
            optimal_lane_id=int(INVALID_LANE_ID),
            distance_to_destination_m=float(
                math.hypot(float(goal_xy[0]) - float(start_xy[0]), float(goal_xy[1]) - float(start_xy[1]))
            ),
            next_macro_maneuver="Continue Straight",
            route_waypoints=[],
            debug_reason="CARLA via-location reroute returned a route trace but no usable waypoints were extracted.",
        )

    def get_local_lane_context(
        self,
        x_m: float,
        y_m: float,
        heading_rad: float | None = None,
        z_m: float | None = None,
    ) -> Dict[str, object]:
        """
        Return road/lane identifiers plus left/right lane-change availability.

        Results are cached and reused when the ego position has moved less than
        ``_lane_context_cache_threshold_m`` since the last computation, avoiding
        redundant O(n) scans on every tick.  Thread-safe: the cache is guarded
        by ``_lane_context_lock`` because the main loop and the behavior-planner
        worker thread both call this method concurrently.
        """
        with self._lane_context_lock:
            if self._lane_context_cache_state is not None and self._lane_context_cache_result is not None:
                dx = float(x_m) - self._lane_context_cache_state[0]
                dy = float(y_m) - self._lane_context_cache_state[1]
                heading_matches = True
                cached_heading_rad = self._lane_context_cache_state[2]
                cached_z_m = self._lane_context_cache_state[3]
                if heading_rad is not None and cached_heading_rad is not None:
                    heading_matches = abs(
                        self._wrap_angle(float(heading_rad) - float(cached_heading_rad))
                    ) <= float(self._lane_context_cache_heading_threshold_rad)
                if heading_rad is not None and cached_heading_rad is None:
                    heading_matches = False
                z_matches = True
                if z_m is not None and cached_z_m is not None:
                    z_matches = abs(float(z_m) - float(cached_z_m)) <= float(self._lane_context_cache_z_threshold_m)
                if z_m is not None and cached_z_m is None:
                    z_matches = False
                if (dx * dx + dy * dy) < self._lane_context_cache_threshold_m ** 2 and heading_matches and z_matches:
                    return dict(self._lane_context_cache_result)

        result = self._compute_local_lane_context(x_m=x_m, y_m=y_m, heading_rad=heading_rad, z_m=z_m)
        with self._lane_context_lock:
            self._lane_context_cache_state = (
                float(x_m),
                float(y_m),
                None if heading_rad is None else float(heading_rad),
                None if z_m is None else float(z_m),
            )
            self._lane_context_cache_result = dict(result)
        return result

    def _nearest_waypoint_index_for_heading(
        self,
        x_m: float,
        y_m: float,
        heading_rad: float | None = None,
    ) -> int | None:
        if len(self._nodes) == 0:
            return None
        if heading_rad is None:
            return self.nearest_waypoint_index(x_m=x_m, y_m=y_m)

        delta_x = self._node_x_m - float(x_m)
        delta_y = self._node_y_m - float(y_m)
        distance_sq = delta_x * delta_x + delta_y * delta_y
        candidate_count = min(len(self._nodes), 24)
        if candidate_count <= 0:
            return None
        candidate_indices = np.argpartition(distance_sq, candidate_count - 1)[:candidate_count]

        best_index: int | None = None
        best_score = float("inf")
        max_heading_gap_rad = math.pi / 2.0
        for candidate_index in candidate_indices:
            node = self._nodes[int(candidate_index)]
            heading_gap_rad = abs(
                self._wrap_angle(float(node.heading_rad) - float(heading_rad))
            )
            distance_m = math.sqrt(float(distance_sq[int(candidate_index)]))
            if heading_gap_rad > max_heading_gap_rad and distance_m > max(2.0, float(node.lane_width_m)):
                continue
            score = float(distance_m) + float(node.lane_width_m) * (
                float(heading_gap_rad) / max(1e-6, float(max_heading_gap_rad))
            )
            if score < best_score:
                best_score = float(score)
                best_index = int(candidate_index)

        if best_index is not None:
            return int(best_index)
        return self.nearest_waypoint_index(x_m=x_m, y_m=y_m)

    def _compute_local_lane_context(
        self,
        x_m: float,
        y_m: float,
        heading_rad: float | None = None,
        z_m: float | None = None,
    ) -> Dict[str, object]:
        """Uncached implementation of lane context lookup."""

        nearest_index = self._nearest_waypoint_index_for_heading(
            x_m=x_m,
            y_m=y_m,
            heading_rad=heading_rad,
        )
        if nearest_index is None:
            return {
                "road_id": "unknown_road",
                "road_numeric_id": -1,
                "section_id": -1,
                "direction": "unknown",
                "lane_id": int(INVALID_LANE_ID),
                "lane_ids": [],
                "lane_count": 0,
                "min_lane_id": int(INVALID_LANE_ID),
                "max_lane_id": int(INVALID_LANE_ID),
                "can_change_left": False,
                "can_change_right": False,
            }

        nearest_node = self._nodes[int(nearest_index)]
        runtime_waypoint = self._project_runtime_waypoint(x_m=x_m, y_m=y_m, z_m=z_m)
        if runtime_waypoint is not None:
            current_lane_id = int(canonical_lane_id_for_waypoint(runtime_waypoint))
            local_lane_ids = list(canonical_lane_ids_for_waypoint(runtime_waypoint))
            current_lane_index = (
                local_lane_ids.index(int(current_lane_id))
                if int(current_lane_id) in local_lane_ids
                else -1
            )
            return {
                "road_id": f"{int(getattr(runtime_waypoint, 'road_id', 0))}:{int(getattr(runtime_waypoint, 'section_id', 0))}",
                "road_numeric_id": int(getattr(runtime_waypoint, "road_id", 0)),
                "section_id": int(getattr(runtime_waypoint, "section_id", 0)),
                "direction": str(direction_key(int(getattr(runtime_waypoint, "lane_id", 0)))),
                "lane_id": int(current_lane_id),
                "lane_ids": list(local_lane_ids),
                "lane_count": int(len(local_lane_ids)),
                "min_lane_id": min(local_lane_ids) if len(local_lane_ids) > 0 else int(INVALID_LANE_ID),
                "max_lane_id": max(local_lane_ids) if len(local_lane_ids) > 0 else int(INVALID_LANE_ID),
                "can_change_left": bool(
                    0 <= int(current_lane_index) < len(local_lane_ids) - 1
                ),
                "can_change_right": bool(int(current_lane_index) > 0),
            }

        reference_heading_rad = float(nearest_node.heading_rad if heading_rad is None else heading_rad)
        corridor_nodes = [
            node
            for node in self._nodes
            if str(node.road_id) == str(nearest_node.road_id)
            and str(node.direction) == str(nearest_node.direction)
        ]
        nearby_same_corridor = [
            node
            for node in corridor_nodes
            if abs(float(node.progress_m) - float(nearest_node.progress_m))
            <= float(self._lane_change_progress_tolerance_m)
        ]

        candidate_lane_ids = sorted(
            {
                int(node.lane_id)
                for node in nearby_same_corridor
                if int(node.lane_id) != int(INVALID_LANE_ID)
            }
        )
        if len(candidate_lane_ids) == 0:
            candidate_lane_ids = sorted(
                {
                    int(node.lane_id)
                    for node in corridor_nodes
                    if int(node.lane_id) != int(INVALID_LANE_ID)
                }
            )

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

        if len(lane_offsets) > 0:
            local_lane_ids = [
                int(lane_id)
                for lane_id, _offset in sorted(
                    lane_offsets.items(),
                    key=lambda item: float(item[1]),
                )
            ]
        else:
            local_lane_ids = list(candidate_lane_ids)

        current_lane_id = int(nearest_node.lane_id)
        current_lane_index = (
            local_lane_ids.index(int(current_lane_id))
            if int(current_lane_id) in local_lane_ids
            else -1
        )
        can_change_left = bool(0 <= int(current_lane_index) < len(local_lane_ids) - 1)
        can_change_right = bool(int(current_lane_index) > 0)

        return {
            "road_id": str(nearest_node.road_id),
            "road_numeric_id": int(str(nearest_node.road_id).split(":", 1)[0]) if ":" in str(nearest_node.road_id) else -1,
            "section_id": int(str(nearest_node.road_id).split(":", 1)[1]) if ":" in str(nearest_node.road_id) else -1,
            "direction": str(nearest_node.direction),
            "lane_id": int(current_lane_id),
            "lane_ids": list(local_lane_ids),
            "lane_count": int(len(local_lane_ids)),
            "min_lane_id": min(local_lane_ids) if len(local_lane_ids) > 0 else int(INVALID_LANE_ID),
            "max_lane_id": max(local_lane_ids) if len(local_lane_ids) > 0 else int(INVALID_LANE_ID),
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
        blocked_node_indices: set[int] | None = None,
        segment_penalties: Mapping[Tuple[str, int], float] | None = None,
    ) -> RoutePlanSummary:
        blocked_indices = {int(node_index) for node_index in list(blocked_node_indices or set())}
        blocked_indices.discard(int(start_index))
        blocked_indices.discard(int(goal_index))
        normalized_segment_penalties = self._normalized_segment_penalties(segment_penalties)
        frontier: List[Tuple[float, int]] = []
        heapq.heappush(frontier, (0.0, int(start_index)))
        came_from: Dict[int, int | None] = {int(start_index): None}
        g_cost: Dict[int, float] = {int(start_index): 0.0}

        while frontier:
            _, current_index = heapq.heappop(frontier)
            if int(current_index) == int(goal_index):
                break
            for neighbor_index, edge_cost_m in self._adjacency.get(int(current_index), []):
                if int(neighbor_index) in blocked_indices:
                    continue
                neighbor_node = self._nodes[int(neighbor_index)]
                segment_penalty_m = float(
                    normalized_segment_penalties.get(
                        (str(neighbor_node.road_id), int(neighbor_node.lane_id)),
                        0.0,
                    )
                )
                new_cost = (
                    float(g_cost[int(current_index)])
                    + float(edge_cost_m)
                    + float(segment_penalty_m)
                )
                if int(neighbor_index) not in g_cost or new_cost < float(g_cost[int(neighbor_index)]):
                    g_cost[int(neighbor_index)] = float(new_cost)
                    heuristic_m = math.hypot(
                        float(neighbor_node.x_m) - float(goal_x_m),
                        float(neighbor_node.y_m) - float(goal_y_m),
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
                start_lane_id=int(INVALID_LANE_ID),
                goal_road_id="unknown_road",
                goal_lane_id=int(INVALID_LANE_ID),
                optimal_lane_id=int(INVALID_LANE_ID),
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
                start_lane_id=int(INVALID_LANE_ID),
                goal_road_id="unknown_road",
                goal_lane_id=int(INVALID_LANE_ID),
                optimal_lane_id=int(INVALID_LANE_ID),
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

    def plan_route_astar_with_segment_penalties(
        self,
        start_xy: Sequence[float],
        goal_xy: Sequence[float],
        *,
        segment_penalties: Mapping[Tuple[str, int], float],
        replace_stored_route: bool = False,
        start_waypoint=None,
        goal_waypoint=None,
    ) -> RoutePlanSummary:
        start_x_m = float(start_xy[0]) if len(start_xy) >= 1 else 0.0
        start_y_m = float(start_xy[1]) if len(start_xy) >= 2 else 0.0
        goal_x_m = float(goal_xy[0]) if len(goal_xy) >= 1 else 0.0
        goal_y_m = float(goal_xy[1]) if len(goal_xy) >= 2 else 0.0

        start_query = (
            self._waypoint_query_for_carla_waypoint(start_waypoint)
            if start_waypoint is not None
            else self.nearest_waypoint_query(x_m=start_x_m, y_m=start_y_m)
        )
        goal_query = (
            self._waypoint_query_for_carla_waypoint(goal_waypoint)
            if goal_waypoint is not None
            else self.nearest_waypoint_query(x_m=goal_x_m, y_m=goal_y_m)
        )

        if start_query is None or goal_query is None or len(self._nodes) == 0:
            direct_distance_m = math.hypot(goal_x_m - start_x_m, goal_y_m - start_y_m)
            return RoutePlanSummary(
                route_found=False,
                start_road_id="unknown_road",
                start_lane_id=int(INVALID_LANE_ID),
                goal_road_id="unknown_road",
                goal_lane_id=int(INVALID_LANE_ID),
                optimal_lane_id=int(INVALID_LANE_ID),
                distance_to_destination_m=float(direct_distance_m),
                next_macro_maneuver="Continue Straight",
                route_waypoints=[],
                debug_reason="No sampled lane-center waypoint was available near the start or goal query point.",
            )

        start_index = int(start_query.index)
        goal_index = int(goal_query.index)
        route_summary = self._route_with_endpoint_candidates(
            start_x_m=float(start_x_m),
            start_y_m=float(start_y_m),
            goal_x_m=float(goal_x_m),
            goal_y_m=float(goal_y_m),
            start_query=start_query,
            goal_query=goal_query,
            segment_penalties=segment_penalties,
        )
        if bool(getattr(route_summary, "route_found", False)):
            if bool(replace_stored_route):
                per_waypoint_options, per_waypoint_lane_ids = self._internal_metadata_from_route_waypoints(
                    route_waypoints=list(getattr(route_summary, "route_waypoints", []) or []),
                )
                self._last_trace_per_waypoint_options = list(per_waypoint_options)
                self._last_trace_per_waypoint_lane_ids = [int(lane_id) for lane_id in list(per_waypoint_lane_ids)]
                self.replace_stored_route(
                    summary=route_summary,
                    per_waypoint_options=per_waypoint_options,
                    per_waypoint_lane_ids=per_waypoint_lane_ids,
                )
            print("new path generated by global planner")
        return route_summary

    def plan_route_astar_blocking_segments(
        self,
        start_xy: Sequence[float],
        goal_xy: Sequence[float],
        *,
        blocked_segments: Sequence[Tuple[object, int]] | Mapping[Tuple[object, int], object],
        replace_stored_route: bool = False,
        start_waypoint=None,
        goal_waypoint=None,
    ) -> RoutePlanSummary:
        start_x_m = float(start_xy[0]) if len(start_xy) >= 1 else 0.0
        start_y_m = float(start_xy[1]) if len(start_xy) >= 2 else 0.0
        goal_x_m = float(goal_xy[0]) if len(goal_xy) >= 1 else 0.0
        goal_y_m = float(goal_xy[1]) if len(goal_xy) >= 2 else 0.0

        start_query = (
            self._waypoint_query_for_carla_waypoint(start_waypoint)
            if start_waypoint is not None
            else self.nearest_waypoint_query(x_m=start_x_m, y_m=start_y_m)
        )
        goal_query = (
            self._waypoint_query_for_carla_waypoint(goal_waypoint)
            if goal_waypoint is not None
            else self.nearest_waypoint_query(x_m=goal_x_m, y_m=goal_y_m)
        )

        if start_query is None or goal_query is None or len(self._nodes) == 0:
            direct_distance_m = math.hypot(goal_x_m - start_x_m, goal_y_m - start_y_m)
            return RoutePlanSummary(
                route_found=False,
                start_road_id="unknown_road",
                start_lane_id=int(INVALID_LANE_ID),
                goal_road_id="unknown_road",
                goal_lane_id=int(INVALID_LANE_ID),
                optimal_lane_id=int(INVALID_LANE_ID),
                distance_to_destination_m=float(direct_distance_m),
                next_macro_maneuver="Continue Straight",
                route_waypoints=[],
                debug_reason="No sampled lane-center waypoint was available near the start or goal query point.",
            )

        blocked_node_indices = self.blocked_node_indices_for_segments(blocked_segments)
        blocked_node_indices.discard(int(start_query.index))
        blocked_node_indices.discard(int(goal_query.index))

        route_summary = self._route_with_endpoint_candidates(
            start_x_m=float(start_x_m),
            start_y_m=float(start_y_m),
            goal_x_m=float(goal_x_m),
            goal_y_m=float(goal_y_m),
            start_query=start_query,
            goal_query=goal_query,
            blocked_node_indices=blocked_node_indices,
            segment_penalties=None,
        )
        if bool(getattr(route_summary, "route_found", False)):
            if bool(replace_stored_route):
                per_waypoint_options, per_waypoint_lane_ids = self._internal_metadata_from_route_waypoints(
                    route_waypoints=list(getattr(route_summary, "route_waypoints", []) or []),
                )
                self.replace_stored_route(
                    summary=route_summary,
                    per_waypoint_options=per_waypoint_options,
                    per_waypoint_lane_ids=per_waypoint_lane_ids,
                )
            print("new path generated by global planner")
        return route_summary

    def plan_route_astar_avoiding_points(
        self,
        start_xy: Sequence[float],
        goal_xy: Sequence[float],
        *,
        blocked_points_xy: Sequence[Sequence[float]],
        blocked_lane_ids: Sequence[int] | None = None,
        block_radius_m: float = 8.0,
        replace_stored_route: bool = False,
        start_waypoint=None,
        goal_waypoint=None,
    ) -> RoutePlanSummary:
        start_x_m = float(start_xy[0]) if len(start_xy) >= 1 else 0.0
        start_y_m = float(start_xy[1]) if len(start_xy) >= 2 else 0.0
        goal_x_m = float(goal_xy[0]) if len(goal_xy) >= 1 else 0.0
        goal_y_m = float(goal_xy[1]) if len(goal_xy) >= 2 else 0.0

        start_query = (
            self._waypoint_query_for_carla_waypoint(start_waypoint)
            if start_waypoint is not None
            else self.nearest_waypoint_query(x_m=start_x_m, y_m=start_y_m)
        )
        goal_query = (
            self._waypoint_query_for_carla_waypoint(goal_waypoint)
            if goal_waypoint is not None
            else self.nearest_waypoint_query(x_m=goal_x_m, y_m=goal_y_m)
        )

        if start_query is None or goal_query is None or len(self._nodes) == 0:
            direct_distance_m = math.hypot(goal_x_m - start_x_m, goal_y_m - start_y_m)
            return RoutePlanSummary(
                route_found=False,
                start_road_id="unknown_road",
                start_lane_id=int(INVALID_LANE_ID),
                goal_road_id="unknown_road",
                goal_lane_id=int(INVALID_LANE_ID),
                optimal_lane_id=int(INVALID_LANE_ID),
                distance_to_destination_m=float(direct_distance_m),
                next_macro_maneuver="Continue Straight",
                route_waypoints=[],
                debug_reason="No sampled lane-center waypoint was available near the start or goal query point.",
            )

        start_index = int(start_query.index)
        goal_index = int(goal_query.index)
        blocked_node_indices = self._blocked_node_indices_for_points(
            blocked_points_xy=blocked_points_xy,
            block_radius_m=float(block_radius_m),
            blocked_lane_ids=blocked_lane_ids,
        )
        blocked_node_indices.discard(int(start_index))
        blocked_node_indices.discard(int(goal_index))

        route_summary = self._route_with_endpoint_candidates(
            start_x_m=float(start_x_m),
            start_y_m=float(start_y_m),
            goal_x_m=float(goal_x_m),
            goal_y_m=float(goal_y_m),
            start_query=start_query,
            goal_query=goal_query,
            blocked_node_indices=blocked_node_indices,
        )
        if bool(getattr(route_summary, "route_found", False)):
            if bool(replace_stored_route):
                per_waypoint_options, per_waypoint_lane_ids = self._internal_metadata_from_route_waypoints(
                    route_waypoints=list(getattr(route_summary, "route_waypoints", []) or []),
                )
                self._last_trace_per_waypoint_options = list(per_waypoint_options)
                self._last_trace_per_waypoint_lane_ids = [int(lane_id) for lane_id in list(per_waypoint_lane_ids)]
                self.replace_stored_route(
                    summary=route_summary,
                    per_waypoint_options=per_waypoint_options,
                    per_waypoint_lane_ids=per_waypoint_lane_ids,
                )
            print("new path generated by global planner")
        return route_summary

    def _infer_optimal_lane_id(self, route_indices: Sequence[int]) -> int:
        if len(route_indices) == 0:
            return int(INVALID_LANE_ID)
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

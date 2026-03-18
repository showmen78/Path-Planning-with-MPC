"""
A* global planner over road-model lane-center waypoints.

This planner operates only on the project's existing lane-center waypoint graph.
It does not depend on CARLA HD-map APIs. The route summary is tailored to the
fields required by the LLM behavior-planner prompt.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import heapq
import math
from typing import Dict, List, Mapping, Sequence, Tuple


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


class AStarGlobalPlanner:
    """Build a lightweight waypoint graph and plan A* routes on it."""

    def __init__(
        self,
        lane_center_waypoints: Sequence[Mapping[str, object]],
        lane_change_penalty_m: float = 5.0,
        lane_change_progress_tolerance_m: float = 5.0,
        lane_change_distance_factor: float = 1.8,
        max_heading_diff_rad: float = 0.8,
    ) -> None:
        self._lane_change_penalty_m = max(0.0, float(lane_change_penalty_m))
        self._lane_change_progress_tolerance_m = max(0.5, float(lane_change_progress_tolerance_m))
        self._lane_change_distance_factor = max(1.0, float(lane_change_distance_factor))
        self._max_heading_diff_rad = max(0.1, float(max_heading_diff_rad))

        self._nodes = self._normalize_waypoints(lane_center_waypoints=lane_center_waypoints)
        self._adjacency = self._build_adjacency()

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
        for idx, (_, position) in enumerate(raw_items):
            index_by_key[self._waypoint_key(*position)] = idx

        nodes: List[WaypointNode] = []
        for idx, (waypoint, position) in enumerate(raw_items):
            next_position = waypoint.get("next", None)
            next_index = None
            if isinstance(next_position, (list, tuple)) and len(next_position) >= 2:
                next_index = index_by_key.get(
                    self._waypoint_key(float(next_position[0]), float(next_position[1]))
                )

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
                )
            )
        return nodes

    def _build_adjacency(self) -> Dict[int, List[Tuple[int, float]]]:
        adjacency: Dict[int, List[Tuple[int, float]]] = {node.index: [] for node in self._nodes}
        if len(self._nodes) == 0:
            return adjacency

        for node in self._nodes:
            if node.next_index is not None and 0 <= int(node.next_index) < len(self._nodes):
                next_node = self._nodes[int(node.next_index)]
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
        if len(self._nodes) == 0:
            return None
        nearest_node = min(
            self._nodes,
            key=lambda node: math.hypot(float(node.x_m) - float(x_m), float(node.y_m) - float(y_m)),
        )
        return int(nearest_node.index)

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

        start_index = self.nearest_waypoint_index(x_m=start_x_m, y_m=start_y_m)
        goal_index = self.nearest_waypoint_index(x_m=goal_x_m, y_m=goal_y_m)

        if start_index is None or goal_index is None or len(self._nodes) == 0:
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
            )

        start_node = self._nodes[int(start_index)]
        goal_node = self._nodes[int(goal_index)]

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
                route_waypoints=[[float(start_node.x_m), float(start_node.y_m)], [float(goal_node.x_m), float(goal_node.y_m)]],
            )

        route_indices: List[int] = []
        cursor = int(goal_index)
        while cursor is not None:
            route_indices.append(int(cursor))
            previous_cursor = came_from.get(int(cursor), None)
            cursor = int(previous_cursor) if previous_cursor is not None else None
        route_indices.reverse()

        route_waypoints = [
            [float(self._nodes[index].x_m), float(self._nodes[index].y_m)]
            for index in route_indices
        ]
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

        for route_index in route_indices[1:]:
            node = self._nodes[int(route_index)]
            maneuver = str(node.maneuver).strip().lower()
            if maneuver == "left":
                return "Left Turn"
            if maneuver == "right":
                return "Right Turn"

        start_lane_id = int(start_node.lane_id)
        for route_index in route_indices[1:]:
            node = self._nodes[int(route_index)]
            if int(node.lane_id) == start_lane_id:
                continue
            dx_m = float(node.x_m) - float(start_node.x_m)
            dy_m = float(node.y_m) - float(start_node.y_m)
            lateral_offset_m = (
                -math.sin(float(start_node.heading_rad)) * dx_m
                + math.cos(float(start_node.heading_rad)) * dy_m
            )
            return "Lane Change Left" if lateral_offset_m >= 0.0 else "Lane Change Right"

        return "Continue Straight"

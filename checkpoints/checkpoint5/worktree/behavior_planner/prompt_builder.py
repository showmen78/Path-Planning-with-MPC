"""Behavior-planner prompt builder."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import List, Mapping, Sequence

from .intention import (
    SurroundingVehicleSummary,
    summarize_surrounding_vehicle,
)
from utility import AStarGlobalPlanner, RoutePlanSummary


ALLOWED_BEHAVIORS = [
    "LANE_KEEP",
    "LANE_CHANGE_LEFT",
    "LANE_CHANGE_RIGHT",
    "STOP",
    "EMERGENCY_BRAKE",
]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYSTEM_INSTRUCTION_PATH = os.path.join(PROJECT_ROOT, "behavior_planner", "system_instruction.txt")
DEFAULT_LANE_SAFE_FORWARD_BUFFER_M = 10.0
DEFAULT_LANE_SAFE_REAR_LONGITUDINAL_DISTANCE_M = 20.0


@dataclass
class BehaviorPlannerPromptContext:
    """Structured runtime data used to build the prompt lines."""

    ego_lane_id: int
    ego_lane_count: int
    maneuver_possible: int
    lane_safe_current: int
    lane_safe_left: int
    lane_safe_right: int
    route_next_maneuver: str
    distance_to_signal_m: float
    signal_phase_code: str
    route_summary: RoutePlanSummary
    surrounding_vehicles: List[SurroundingVehicleSummary]


class BehaviorPlannerPromptBuilder:
    """Create system instruction text and compact runtime prompts."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def _normalize_broadcasts(v2x_broadcasts: Sequence[Mapping[str, object]] | None) -> List[dict]:
        normalized: List[dict] = []
        for message in list(v2x_broadcasts or []):
            normalized.append(
                {
                    "id": message.get("id", "unknown"),
                    "type": str(message.get("type", "n")),
                    "message": str(message.get("message", "")),
                }
            )
        return normalized

    @staticmethod
    def _lane_count_from_inputs(
        lane_center_waypoints: Sequence[Mapping[str, object]],
        road_cfg: Mapping[str, object] | None,
    ) -> int:
        road_cfg = dict(road_cfg or {})
        configured_lane_count = int(road_cfg.get("lane_count", 0))
        if configured_lane_count > 0:
            return configured_lane_count

        lane_ids = []
        for waypoint in lane_center_waypoints:
            lane_id = waypoint.get("lane_id", None)
            if lane_id is None:
                continue
            lane_ids.append(int(lane_id))
        if len(lane_ids) == 0:
            return 1
        return max(1, max(lane_ids))

    @staticmethod
    def _prompt_lane_id(raw_lane_id: int, lane_count: int) -> int:
        """
        Map project lane ids 1..N into the compact prompt convention:
        0  -> reference line at the left edge of the left-most lane
        -1 -> left-most lane
        -2 -> next lane to the right
        ...
        """

        lane_count = max(1, int(lane_count))
        return int(raw_lane_id) - (lane_count + 1)

    @staticmethod
    def _macro_maneuver_code(maneuver_text: str) -> str:
        lowered = str(maneuver_text).strip().lower()
        if "left turn" in lowered or lowered == "left":
            return "L"
        if "right turn" in lowered or lowered == "right":
            return "R"
        return "S"

    @staticmethod
    def _signal_phase_code(road_cfg: Mapping[str, object]) -> str:
        _ = road_cfg
        return "G"

    @staticmethod
    def _simplify_intention_code(predicted_intention: str) -> str:
        normalized = str(predicted_intention or "").strip().upper()
        if normalized == "S":
            return "S"
        if normalized.startswith("L"):
            return "L"
        if normalized.startswith("R"):
            return "R"
        if normalized.endswith("B"):
            return "B"
        if normalized.endswith("A"):
            return "A"
        return "K"

    @staticmethod
    def _participant_type_code(vehicle_snapshot: Mapping[str, object]) -> int:
        raw_type = str(vehicle_snapshot.get("type", "")).strip().lower()
        raw_vehicle_id = str(vehicle_snapshot.get("vehicle_id", "")).strip().lower()
        combined = f"{raw_type} {raw_vehicle_id}"
        if any(keyword in combined for keyword in ("ambulance", "police", "fire", "rescue", "emergency")):
            return 1
        if any(keyword in combined for keyword in ("pedestrian", "walker", "cyclist", "bicycle", "bike", "vru")):
            return 2
        return 0

    @staticmethod
    def _stopping_distance_m(speed_mps: float, min_acceleration_mps2: float) -> float:
        braking_mps2 = max(1e-6, abs(float(min_acceleration_mps2)))
        return float(speed_mps * speed_mps / (2.0 * braking_mps2))

    @staticmethod
    def _format_numeric(value: float, decimals: int = 3) -> str:
        formatted = f"{float(value):.{int(decimals)}f}"
        if "." in formatted:
            formatted = formatted.rstrip("0").rstrip(".")
        return formatted if len(formatted) > 0 else "0"

    @staticmethod
    def _nearest_waypoint_on_lane(
        lane_center_waypoints: Sequence[Mapping[str, object]],
        target_lane_id: int,
        reference_x_m: float,
        reference_y_m: float,
    ) -> Mapping[str, object] | None:
        lane_waypoints = [
            waypoint
            for waypoint in lane_center_waypoints
            if int(waypoint.get("lane_id", -1)) == int(target_lane_id)
            and isinstance(waypoint.get("position", None), (list, tuple))
            and len(waypoint.get("position", [])) >= 2
        ]
        if len(lane_waypoints) == 0:
            return None
        return min(
            lane_waypoints,
            key=lambda waypoint: math.hypot(
                float(waypoint["position"][0]) - float(reference_x_m),
                float(waypoint["position"][1]) - float(reference_y_m),
            ),
        )

    def _assumed_ego_snapshot_for_lane(
        self,
        ego_snapshot: Mapping[str, object],
        lane_center_waypoints: Sequence[Mapping[str, object]],
        target_lane_id: int,
    ) -> dict:
        assumed_snapshot = dict(ego_snapshot)
        waypoint = self._nearest_waypoint_on_lane(
            lane_center_waypoints=lane_center_waypoints,
            target_lane_id=int(target_lane_id),
            reference_x_m=float(ego_snapshot.get("x", 0.0)),
            reference_y_m=float(ego_snapshot.get("y", 0.0)),
        )
        if waypoint is None:
            return assumed_snapshot

        position = list(waypoint.get("position", []))
        if len(position) >= 2:
            assumed_snapshot["x"] = float(position[0])
            assumed_snapshot["y"] = float(position[1])
        if "heading_rad" in waypoint:
            assumed_snapshot["psi"] = float(waypoint.get("heading_rad", ego_snapshot.get("psi", 0.0)))
        return assumed_snapshot

    def _lane_safe_flag(
        self,
        ego_snapshot: Mapping[str, object],
        object_snapshots: Sequence[Mapping[str, object]],
        lane_center_waypoints: Sequence[Mapping[str, object]],
        planner: AStarGlobalPlanner,
        target_lane_id: int,
        forward_longitudinal_safe_distance_m: float,
        rear_longitudinal_safe_distance_m: float,
    ) -> int:
        lane_exists = any(
            int(waypoint.get("lane_id", -1)) == int(target_lane_id)
            for waypoint in lane_center_waypoints
        )
        if not lane_exists:
            return 0

        assumed_ego_snapshot = self._assumed_ego_snapshot_for_lane(
            ego_snapshot=ego_snapshot,
            lane_center_waypoints=lane_center_waypoints,
            target_lane_id=int(target_lane_id),
        )
        lane_obstacles = []
        for vehicle_snapshot in object_snapshots:
            vehicle_lane_context = planner.get_local_lane_context(
                x_m=float(vehicle_snapshot.get("x", 0.0)),
                y_m=float(vehicle_snapshot.get("y", 0.0)),
                heading_rad=float(vehicle_snapshot.get("psi", 0.0)),
            )
            if int(vehicle_lane_context.get("lane_id", -1)) != int(target_lane_id):
                continue
            lane_obstacles.append(vehicle_snapshot)

        if len(lane_obstacles) == 0:
            return 1

        ego_x_m = float(assumed_ego_snapshot.get("x", 0.0))
        ego_y_m = float(assumed_ego_snapshot.get("y", 0.0))
        ego_heading_rad = float(assumed_ego_snapshot.get("psi", 0.0))
        forward_longitudinal_safe_distance_m = max(0.0, float(forward_longitudinal_safe_distance_m))
        rear_longitudinal_safe_distance_m = max(0.0, float(rear_longitudinal_safe_distance_m))

        for vehicle_snapshot in lane_obstacles:
            signed_longitudinal_gap_m = (
                math.cos(float(ego_heading_rad))
                * (float(vehicle_snapshot.get("x", 0.0)) - float(ego_x_m))
                + math.sin(float(ego_heading_rad))
                * (float(vehicle_snapshot.get("y", 0.0)) - float(ego_y_m))
            )
            if float(signed_longitudinal_gap_m) >= 0.0:
                active_threshold_m = float(forward_longitudinal_safe_distance_m)
                longitudinal_gap_m = float(signed_longitudinal_gap_m)
            else:
                active_threshold_m = float(rear_longitudinal_safe_distance_m)
                longitudinal_gap_m = abs(float(signed_longitudinal_gap_m))

            if float(longitudinal_gap_m) <= float(active_threshold_m):
                return 0
        return 1

    def load_system_instruction(self) -> str:
        """Return the system instruction text as authored."""

        with open(SYSTEM_INSTRUCTION_PATH, "r", encoding="utf-8") as handle:
            full_text = handle.read()

        split_marker = "# Example INPUT (Prompt)"
        if split_marker in full_text:
            full_text = full_text.split(split_marker, 1)[0]
        return full_text.strip()

    def build_context(
        self,
        ego_snapshot: Mapping[str, object],
        destination_state: Sequence[float],
        temporary_destination_state: Sequence[float] | None,
        lane_center_waypoints: Sequence[Mapping[str, object]],
        object_snapshots: Sequence[Mapping[str, object]],
        road_cfg: Mapping[str, object] | None = None,
        behavior_planner_runtime_cfg: Mapping[str, object] | None = None,
        planner: AStarGlobalPlanner | None = None,
    ) -> BehaviorPlannerPromptContext:
        """Build the structured runtime data needed for the prompt."""

        road_cfg = dict(road_cfg or {})
        behavior_planner_runtime_cfg = dict(behavior_planner_runtime_cfg or {})
        planner = planner or AStarGlobalPlanner(lane_center_waypoints=lane_center_waypoints)

        ego_x_m = float(ego_snapshot.get("x", 0.0))
        ego_y_m = float(ego_snapshot.get("y", 0.0))
        ego_psi_rad = float(ego_snapshot.get("psi", 0.0))

        local_context = planner.get_local_lane_context(
            x_m=float(ego_x_m),
            y_m=float(ego_y_m),
            heading_rad=float(ego_psi_rad),
        )
        temporary_destination_state = (
            list(temporary_destination_state)
            if temporary_destination_state is not None
            else None
        )
        route_start_xy = (
            list(temporary_destination_state[:2])
            if temporary_destination_state is not None and len(temporary_destination_state) >= 2
            else [float(ego_x_m), float(ego_y_m)]
        )
        route_summary = planner.plan_route(
            start_xy=route_start_xy,
            goal_xy=list(destination_state[:2]) if len(destination_state) >= 2 else [ego_x_m, ego_y_m],
        )
        lane_width_m = float(road_cfg.get("lane_width_m", 4.0))
        forward_longitudinal_safe_buffer_m = max(
            0.0,
            float(
                behavior_planner_runtime_cfg.get(
                    "lane_safe_forward_buffer_m",
                    DEFAULT_LANE_SAFE_FORWARD_BUFFER_M,
                )
            ),
        )
        rear_longitudinal_safe_distance_m = max(
            0.0,
            float(
                behavior_planner_runtime_cfg.get(
                    "lane_safe_rear_longitudinal_distance_m",
                    behavior_planner_runtime_cfg.get(
                        "lane_safe_longitudinal_distance_m",
                        DEFAULT_LANE_SAFE_REAR_LONGITUDINAL_DISTANCE_M,
                    ),
                )
            ),
        )
        forward_longitudinal_safe_distance_m = 0.0
        if temporary_destination_state is not None and len(temporary_destination_state) >= 2:
            forward_longitudinal_safe_distance_m = math.hypot(
                float(temporary_destination_state[0]) - float(ego_x_m),
                float(temporary_destination_state[1]) - float(ego_y_m),
            )
        forward_longitudinal_safe_distance_m += float(forward_longitudinal_safe_buffer_m)
        ego_lane_id = int(local_context.get("lane_id", -1))
        lane_safe_current = self._lane_safe_flag(
            ego_snapshot=ego_snapshot,
            object_snapshots=object_snapshots,
            lane_center_waypoints=lane_center_waypoints,
            planner=planner,
            target_lane_id=ego_lane_id,
            forward_longitudinal_safe_distance_m=forward_longitudinal_safe_distance_m,
            rear_longitudinal_safe_distance_m=rear_longitudinal_safe_distance_m,
        )
        lane_safe_left = 0
        if bool(local_context.get("can_change_left", False)):
            lane_safe_left = self._lane_safe_flag(
                ego_snapshot=ego_snapshot,
                object_snapshots=object_snapshots,
                lane_center_waypoints=lane_center_waypoints,
                planner=planner,
                target_lane_id=int(ego_lane_id + 1),
                forward_longitudinal_safe_distance_m=forward_longitudinal_safe_distance_m,
                rear_longitudinal_safe_distance_m=rear_longitudinal_safe_distance_m,
            )
        lane_safe_right = 0
        if bool(local_context.get("can_change_right", False)):
            lane_safe_right = self._lane_safe_flag(
                ego_snapshot=ego_snapshot,
                object_snapshots=object_snapshots,
                lane_center_waypoints=lane_center_waypoints,
                planner=planner,
                target_lane_id=int(ego_lane_id - 1),
                forward_longitudinal_safe_distance_m=forward_longitudinal_safe_distance_m,
                rear_longitudinal_safe_distance_m=rear_longitudinal_safe_distance_m,
            )

        surrounding_summaries = [
            summarize_surrounding_vehicle(
                ego_snapshot=ego_snapshot,
                vehicle_snapshot=vehicle_snapshot,
                lane_center_waypoints=lane_center_waypoints,
                lane_width_m=lane_width_m,
                planner=planner,
            )
            for vehicle_snapshot in object_snapshots
        ]

        route_next_maneuver = self._macro_maneuver_code(str(route_summary.next_macro_maneuver))
        maneuver_possible = 1
        if (
            int(route_summary.optimal_lane_id) > 0
            and int(ego_lane_id) > 0
            and str(route_next_maneuver) in {"L", "R"}
        ):
            maneuver_possible = int(int(route_summary.optimal_lane_id) == int(ego_lane_id))

        return BehaviorPlannerPromptContext(
            ego_lane_id=ego_lane_id,
            ego_lane_count=max(
                1,
                int(
                    road_cfg.get(
                        "lane_count",
                        self._lane_count_from_inputs(
                            lane_center_waypoints=lane_center_waypoints,
                            road_cfg=road_cfg,
                        ),
                    )
                ),
            ),
            maneuver_possible=int(maneuver_possible),
            lane_safe_current=int(lane_safe_current),
            lane_safe_left=int(lane_safe_left),
            lane_safe_right=int(lane_safe_right),
            route_next_maneuver=str(route_next_maneuver),
            distance_to_signal_m=1000.0,
            signal_phase_code=str(self._signal_phase_code(road_cfg)),
            route_summary=route_summary,
            surrounding_vehicles=surrounding_summaries,
        )

    def build_prompt(
        self,
        ego_snapshot: Mapping[str, object],
        destination_state: Sequence[float],
        temporary_destination_state: Sequence[float] | None,
        lane_center_waypoints: Sequence[Mapping[str, object]],
        object_snapshots: Sequence[Mapping[str, object]],
        road_cfg: Mapping[str, object] | None = None,
        behavior_planner_runtime_cfg: Mapping[str, object] | None = None,
        v2x_broadcasts: Sequence[Mapping[str, object]] | None = None,
        ego_vehicle_id: str = "Ego01",
        mpc_constraints: Mapping[str, object] | None = None,
        prompt_id: str | int | None = None,
        previous_behavior: str = "LANE_KEEP",
        planner: AStarGlobalPlanner | None = None,
    ) -> str:
        """Build the environment-input prompt expected by the current system instruction."""

        road_cfg = dict(road_cfg or {})
        mpc_constraints = dict(mpc_constraints or {})

        context = self.build_context(
            ego_snapshot=ego_snapshot,
            destination_state=destination_state,
            temporary_destination_state=temporary_destination_state,
            lane_center_waypoints=lane_center_waypoints,
            object_snapshots=object_snapshots,
            road_cfg=road_cfg,
            behavior_planner_runtime_cfg=behavior_planner_runtime_cfg,
            planner=planner,
        )
        lane_count = max(1, int(context.ego_lane_count))

        ego_x_m = float(ego_snapshot.get("x", 0.0))
        ego_y_m = float(ego_snapshot.get("y", 0.0))
        ego_v_mps = float(ego_snapshot.get("v", 0.0))
        ego_psi_rad = float(ego_snapshot.get("psi", 0.0))
        ego_lane_prompt = self._prompt_lane_id(
            raw_lane_id=int(context.ego_lane_id),
            lane_count=lane_count,
        )

        lines = []
        if prompt_id is not None:
            lines.append(f"ID: {int(prompt_id)}")
        lines.extend([
            (
                "ego = ["
                f"{self._format_numeric(ego_x_m)},"
                f"{self._format_numeric(ego_y_m)},"
                f"{self._format_numeric(ego_v_mps)},"
                f"{self._format_numeric(ego_psi_rad, decimals=6)},"
                f"{ego_lane_prompt},"
                f"{int(context.maneuver_possible)}]"
            ),
            (
                "lane = ["
                f"{int(context.lane_safe_left)},"
                f"{int(context.lane_safe_current)},"
                f"{int(context.lane_safe_right)}]"
            ),
            (
                "route = ["
                f"{str(context.route_next_maneuver)},"
                f"{self._format_numeric(float(context.distance_to_signal_m))},"
                f"{str(context.signal_phase_code)}]"
            ),
        ])

        for vehicle_snapshot, vehicle_summary in zip(object_snapshots, context.surrounding_vehicles):
            vehicle_id = str(vehicle_summary.vehicle_id)
            intention_code = self._simplify_intention_code(vehicle_summary.predicted_intention)
            participant_type_code = self._participant_type_code(vehicle_snapshot)

            lines.extend(
                [
                    (
                        f"v[{vehicle_id}] = ["
                        f"{self._format_numeric(vehicle_summary.x_m)},"
                        f"{self._format_numeric(vehicle_summary.y_m)},"
                        f"{self._format_numeric(vehicle_summary.v_mps)},"
                        f"{self._format_numeric(vehicle_summary.psi_rad, decimals=6)},"
                        f"{intention_code},"
                        f"{int(participant_type_code)}]"
                    ),
                ]
            )

        return "\n".join(lines)


def build_behavior_planner_prompt(
    ego_snapshot: Mapping[str, object],
    destination_state: Sequence[float],
    temporary_destination_state: Sequence[float] | None,
    lane_center_waypoints: Sequence[Mapping[str, object]],
    object_snapshots: Sequence[Mapping[str, object]],
    road_cfg: Mapping[str, object] | None = None,
    behavior_planner_runtime_cfg: Mapping[str, object] | None = None,
    v2x_broadcasts: Sequence[Mapping[str, object]] | None = None,
    ego_vehicle_id: str = "Ego01",
    mpc_constraints: Mapping[str, object] | None = None,
    prompt_id: str | int | None = None,
    previous_behavior: str = "LANE_KEEP",
) -> str:
    """Functional wrapper for the prompt builder."""

    builder = BehaviorPlannerPromptBuilder()
    return builder.build_prompt(
        ego_snapshot=ego_snapshot,
        destination_state=destination_state,
        temporary_destination_state=temporary_destination_state,
        lane_center_waypoints=lane_center_waypoints,
        object_snapshots=object_snapshots,
        road_cfg=road_cfg,
        behavior_planner_runtime_cfg=behavior_planner_runtime_cfg,
        v2x_broadcasts=v2x_broadcasts,
        ego_vehicle_id=ego_vehicle_id,
        mpc_constraints=mpc_constraints,
        prompt_id=prompt_id,
        previous_behavior=previous_behavior,
    )

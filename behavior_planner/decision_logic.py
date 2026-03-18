"""
Behavior-planner response parsing and rule execution.

The runtime logic in this file intentionally does not read
`behavior_planner_rules.txt`. The rules are implemented directly here so the
simulation behavior stays stable even if that text file changes later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Dict, List, Mapping, Sequence

from .global_planner import AStarGlobalPlanner


ALLOWED_BEHAVIOR_DECISIONS = {
    "LANE_KEEP",
    "LANE_CHANGE_LEFT",
    "LANE_CHANGE_RIGHT",
    "STOP_AT_LINE",
    "EMERGENCY_BRAKE",
    "GO_STRAIGHT",
    "TURN_LEFT",
    "TURN_RIGHT",
}


@dataclass
class BehaviorPlannerDecision:
    """Normalized LLM decision."""

    request_id: str
    behavior: str
    target_v_mps: float | None
    reasoning: str
    cav_broadcast: Dict[str, str]
    raw_response_text: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class BehaviorExecutionResult:
    """Planner overrides derived from one behavior decision."""

    destination_state: List[float]
    selected_lane_id: int
    max_velocity_override_mps: float | None
    applied_behavior: str
    cav_broadcast: Dict[str, str]


def _clean_json_text(raw_text: str) -> str:
    text = str(raw_text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            if lines[0].startswith("```"):
                lines = lines[1:]
            if len(lines) > 0 and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
    return text


def parse_behavior_planner_response(raw_text: str) -> BehaviorPlannerDecision:
    """Parse and validate the JSON response returned by the LLM."""

    cleaned_text = _clean_json_text(raw_text)
    payload = json.loads(cleaned_text)
    if not isinstance(payload, Mapping):
        raise ValueError("Behavior-planner response must be a JSON object.")

    request_id = str(payload.get("id", payload.get("request_id", ""))).strip()
    if len(request_id) == 0:
        raise ValueError("Behavior-planner response must include a non-empty id.")

    behavior = str(payload.get("behavior", "")).strip().upper()
    if behavior not in ALLOWED_BEHAVIOR_DECISIONS:
        raise ValueError(f"Unsupported behavior decision: {behavior!r}")

    target_v_value = payload.get(
        "Target_v (m/s)",
        payload.get("target_v", payload.get("target_v_mps", None)),
    )
    target_v_mps = None if target_v_value is None else float(target_v_value)
    reasoning = str(payload.get("reasoning", "")).strip()

    broadcast_raw = payload.get("cav_broadcast", payload.get("Cav_broadcast", {}))
    if not isinstance(broadcast_raw, Mapping):
        broadcast_raw = {}
    cav_broadcast = {
        "id": str(broadcast_raw.get("id", "Ego01")),
        "type": str(broadcast_raw.get("type", "n")),
        "message": str(broadcast_raw.get("message", "")),
    }

    return BehaviorPlannerDecision(
        request_id=request_id,
        behavior=behavior,
        target_v_mps=None if target_v_mps is None else float(target_v_mps),
        reasoning=reasoning,
        cav_broadcast=cav_broadcast,
        raw_response_text=str(raw_text),
    )


def decision_from_mapping(data: Mapping[str, object] | None) -> BehaviorPlannerDecision | None:
    if not isinstance(data, Mapping) or len(data) == 0:
        return None
    return BehaviorPlannerDecision(
        request_id=str(data.get("request_id", data.get("id", ""))),
        behavior=str(data.get("behavior", "")),
        target_v_mps=(
            None
            if data.get("target_v_mps", None) is None
            else float(data.get("target_v_mps", 0.0))
        ),
        reasoning=str(data.get("reasoning", "")),
        cav_broadcast=dict(data.get("cav_broadcast", {})) if isinstance(data.get("cav_broadcast", {}), Mapping) else {},
        raw_response_text=str(data.get("raw_response_text", "")),
    )


def _position_of_waypoint(waypoint: Mapping[str, object]) -> tuple[float, float] | None:
    position = waypoint.get("position", None)
    if not isinstance(position, (list, tuple)) or len(position) < 2:
        return None
    return float(position[0]), float(position[1])


def _waypoint_key(x_m: float, y_m: float) -> tuple[float, float]:
    return round(float(x_m), 3), round(float(y_m), 3)


def _lane_count_from_inputs(
    lane_center_waypoints: Sequence[Mapping[str, object]],
    road_cfg: Mapping[str, object] | None,
) -> int:
    road_cfg = dict(road_cfg or {})
    configured_lane_count = int(road_cfg.get("lane_count", 0))
    if configured_lane_count > 0:
        return configured_lane_count
    lane_ids = [int(item.get("lane_id", 0)) for item in lane_center_waypoints if item.get("lane_id", None) is not None]
    return max(1, max(lane_ids)) if len(lane_ids) > 0 else 1


def _normalize_destination_state(destination_state: Sequence[float], fallback_heading_rad: float = 0.0) -> List[float]:
    dest = [float(value) for value in list(destination_state[:4])]
    while len(dest) < 4:
        dest.append(0.0)
    if abs(float(dest[3])) <= 1e-9:
        dest[3] = float(fallback_heading_rad)
    return dest


def _clip_target_speed(target_v_mps: float, speed_limit_mps: float) -> float:
    return float(min(max(0.0, float(target_v_mps)), max(0.0, float(speed_limit_mps))))


def _default_target_speed_for_behavior(
    behavior: str,
    speed_limit_mps: float,
    llm_target_v_mps: float | None,
) -> float:
    """
    Resolve the effective V_ref from the hardcoded runtime rules.

    The updated system instruction may return only {"behavior": ...}, so the
    runtime must not depend on Target_v being present.
    """

    if llm_target_v_mps is not None:
        return _clip_target_speed(float(llm_target_v_mps), float(speed_limit_mps))

    normalized_behavior = str(behavior or "").strip().upper()
    if normalized_behavior in {"STOP_AT_LINE", "EMERGENCY_BRAKE"}:
        return 0.0
    return float(max(0.0, speed_limit_mps))


def _same_destination_xy(
    lhs_state: Sequence[float],
    rhs_state: Sequence[float],
    tolerance_m: float = 1e-3,
) -> bool:
    """Return True when two destination states represent the same XY point."""

    if len(lhs_state) < 2 or len(rhs_state) < 2:
        return False
    return math.hypot(
        float(lhs_state[0]) - float(rhs_state[0]),
        float(lhs_state[1]) - float(rhs_state[1]),
    ) <= float(max(0.0, tolerance_m))


def _lookahead_distance_from_destination(
    ego_snapshot: Mapping[str, object],
    base_destination_state: Sequence[float],
    local_goal_cfg: Mapping[str, object] | None = None,
) -> float:
    local_goal_cfg = dict(local_goal_cfg or {})
    ego_x_m = float(ego_snapshot.get("x", 0.0))
    ego_y_m = float(ego_snapshot.get("y", 0.0))
    minimum_lookahead_distance_m = float(
        max(
            1.0,
            local_goal_cfg.get(
                "dynamic_lookahead_min_distance_m",
                local_goal_cfg.get("lock_to_final_distance_m", 10.0),
            ),
        )
    )
    if len(base_destination_state) >= 2:
        lookahead_distance_m = math.hypot(
            float(base_destination_state[0]) - ego_x_m,
            float(base_destination_state[1]) - ego_y_m,
        )
        if lookahead_distance_m > 1e-3:
            return float(max(minimum_lookahead_distance_m, lookahead_distance_m))
    return float(minimum_lookahead_distance_m)


def _behavior_destination_distance_m(
    ego_snapshot: Mapping[str, object],
    base_destination_state: Sequence[float],
    local_goal_cfg: Mapping[str, object] | None = None,
) -> float:
    local_goal_cfg = dict(local_goal_cfg or {})
    lookahead_distance_m = _lookahead_distance_from_destination(
        ego_snapshot=ego_snapshot,
        base_destination_state=base_destination_state,
        local_goal_cfg=local_goal_cfg,
    )
    extra_buffer_m = max(
        0.0,
        float(local_goal_cfg.get("behavior_destination_distance_buffer_m", 0.0)),
    )
    return float(max(1.0, float(lookahead_distance_m) + float(extra_buffer_m)))


def _select_waypoint_ahead_on_lane(
    lane_center_waypoints: Sequence[Mapping[str, object]],
    ego_snapshot: Mapping[str, object],
    target_lane_id: int,
    target_distance_m: float,
) -> List[float] | None:
    lane_waypoints = [
        waypoint
        for waypoint in lane_center_waypoints
        if int(waypoint.get("lane_id", -999999)) == int(target_lane_id)
        and _position_of_waypoint(waypoint) is not None
    ]
    if len(lane_waypoints) == 0:
        return None

    ego_x_m = float(ego_snapshot.get("x", 0.0))
    ego_y_m = float(ego_snapshot.get("y", 0.0))
    ego_psi_rad = float(ego_snapshot.get("psi", 0.0))

    def _waypoint_map(
        waypoints: Sequence[Mapping[str, object]],
    ) -> Dict[tuple[float, float], Mapping[str, object]]:
        return {
            _waypoint_key(float(position[0]), float(position[1])): waypoint
            for waypoint in waypoints
            if (position := _position_of_waypoint(waypoint)) is not None
        }

    def _best_segment_for_waypoints(
        waypoints: Sequence[Mapping[str, object]],
    ) -> tuple[Mapping[str, object], Mapping[str, object], float, float] | None:
        waypoint_by_key_local = _waypoint_map(waypoints)
        best_segment_local: tuple[Mapping[str, object], Mapping[str, object], float, float] | None = None
        best_projection_distance_m_local = float("inf")

        for waypoint in waypoints:
            start_position = _position_of_waypoint(waypoint)
            next_position_raw = waypoint.get("next", None)
            if start_position is None or not isinstance(next_position_raw, (list, tuple)) or len(next_position_raw) < 2:
                continue
            next_waypoint = waypoint_by_key_local.get(
                _waypoint_key(float(next_position_raw[0]), float(next_position_raw[1]))
            )
            if next_waypoint is None:
                continue
            end_position = _position_of_waypoint(next_waypoint)
            if end_position is None:
                continue

            dx_m = float(end_position[0]) - float(start_position[0])
            dy_m = float(end_position[1]) - float(start_position[1])
            segment_length_sq = dx_m * dx_m + dy_m * dy_m
            if segment_length_sq <= 1e-9:
                continue

            alpha = (
                ((float(ego_x_m) - float(start_position[0])) * dx_m)
                + ((float(ego_y_m) - float(start_position[1])) * dy_m)
            ) / segment_length_sq
            alpha = min(1.0, max(0.0, float(alpha)))
            proj_x_m = float(start_position[0]) + float(alpha) * dx_m
            proj_y_m = float(start_position[1]) + float(alpha) * dy_m
            projection_distance_m = math.hypot(float(ego_x_m) - proj_x_m, float(ego_y_m) - proj_y_m)
            if projection_distance_m < best_projection_distance_m_local:
                best_projection_distance_m_local = float(projection_distance_m)
                best_segment_local = (
                    waypoint,
                    next_waypoint,
                    float(alpha),
                    math.sqrt(segment_length_sq),
                )

        return best_segment_local

    seed_waypoint = min(
        lane_waypoints,
        key=lambda waypoint: math.hypot(
            float(_position_of_waypoint(waypoint)[0]) - ego_x_m,
            float(_position_of_waypoint(waypoint)[1]) - ego_y_m,
        ),
    )
    seed_position = _position_of_waypoint(seed_waypoint)
    if seed_position is None:
        return None

    best_segment = _best_segment_for_waypoints(lane_waypoints)
    if best_segment is not None:
        current_waypoint = best_segment[0]
        current_road_id = str(current_waypoint.get("road_id", ""))
        current_direction = str(current_waypoint.get("direction", ""))
        local_lane_waypoints = [
            waypoint
            for waypoint in lane_waypoints
            if str(waypoint.get("road_id", "")) == current_road_id
            and str(waypoint.get("direction", "")) == current_direction
        ]
        if len(local_lane_waypoints) > 0:
            lane_waypoints = local_lane_waypoints
            localized_best_segment = _best_segment_for_waypoints(lane_waypoints)
            if localized_best_segment is not None:
                best_segment = localized_best_segment

    waypoint_by_key = _waypoint_map(lane_waypoints)

    if best_segment is None:
        return [
            float(seed_position[0]),
            float(seed_position[1]),
            0.0,
            float(seed_waypoint.get("heading_rad", ego_psi_rad)),
        ]

    current_waypoint, next_waypoint, current_alpha, current_segment_length_m = best_segment
    current_position = _position_of_waypoint(current_waypoint)
    next_position = _position_of_waypoint(next_waypoint)
    if current_position is None or next_position is None:
        return None

    remaining_distance_m = max(0.0, float(target_distance_m))
    distance_to_segment_end_m = max(0.0, (1.0 - float(current_alpha)) * float(current_segment_length_m))
    if remaining_distance_m <= distance_to_segment_end_m and current_segment_length_m > 1e-9:
        target_alpha = (
            (float(current_alpha) * float(current_segment_length_m)) + float(remaining_distance_m)
        ) / float(current_segment_length_m)
        target_alpha = min(1.0, max(0.0, float(target_alpha)))
        interp_x_m = float(current_position[0]) + target_alpha * (float(next_position[0]) - float(current_position[0]))
        interp_y_m = float(current_position[1]) + target_alpha * (float(next_position[1]) - float(current_position[1]))
        interp_heading_rad = math.atan2(
            float(next_position[1]) - float(current_position[1]),
            float(next_position[0]) - float(current_position[0]),
        )
        return [float(interp_x_m), float(interp_y_m), 0.0, float(interp_heading_rad)]

    remaining_distance_m -= float(distance_to_segment_end_m)
    current_waypoint = next_waypoint
    visited_keys = {
        _waypoint_key(float(current_position[0]), float(current_position[1])),
    }

    while True:
        current_position = _position_of_waypoint(current_waypoint)
        if current_position is None:
            return None
        current_key = _waypoint_key(float(current_position[0]), float(current_position[1]))
        if current_key in visited_keys:
            break
        visited_keys.add(current_key)

        next_position_raw = current_waypoint.get("next", None)
        if not isinstance(next_position_raw, (list, tuple)) or len(next_position_raw) < 2:
            break
        next_waypoint = waypoint_by_key.get(
            _waypoint_key(float(next_position_raw[0]), float(next_position_raw[1]))
        )
        if next_waypoint is None:
            break
        next_position = _position_of_waypoint(next_waypoint)
        if next_position is None:
            break

        dx_m = float(next_position[0]) - float(current_position[0])
        dy_m = float(next_position[1]) - float(current_position[1])
        segment_length_m = math.hypot(dx_m, dy_m)
        if segment_length_m <= 1e-9:
            current_waypoint = next_waypoint
            continue

        if remaining_distance_m <= segment_length_m:
            alpha = min(1.0, max(0.0, float(remaining_distance_m) / float(segment_length_m)))
            interp_x_m = float(current_position[0]) + alpha * dx_m
            interp_y_m = float(current_position[1]) + alpha * dy_m
            interp_heading_rad = math.atan2(dy_m, dx_m)
            return [float(interp_x_m), float(interp_y_m), 0.0, float(interp_heading_rad)]

        remaining_distance_m -= float(segment_length_m)
        current_waypoint = next_waypoint

    final_position = _position_of_waypoint(current_waypoint)
    if final_position is None:
        return None
    return [
        float(final_position[0]),
        float(final_position[1]),
        0.0,
        float(current_waypoint.get("heading_rad", ego_psi_rad)),
    ]


def build_destination_on_lane(
    ego_snapshot: Mapping[str, object],
    lane_center_waypoints: Sequence[Mapping[str, object]],
    target_lane_id: int,
    target_distance_m: float,
    road_cfg: Mapping[str, object] | None = None,
) -> List[float] | None:
    """
    Build a rolling temporary destination at a specified distance ahead of the
    ego vehicle on the requested lane.
    """

    lane_count = _lane_count_from_inputs(
        lane_center_waypoints=lane_center_waypoints,
        road_cfg=road_cfg,
    )
    clamped_lane_id = min(
        max(1, int(target_lane_id)),
        max(1, int(lane_count)),
    )
    return _select_waypoint_ahead_on_lane(
        lane_center_waypoints=lane_center_waypoints,
        ego_snapshot=ego_snapshot,
        target_lane_id=clamped_lane_id,
        target_distance_m=float(target_distance_m),
    )


def apply_behavior_planner_decision(
    decision: BehaviorPlannerDecision | None,
    ego_snapshot: Mapping[str, object],
    base_destination_state: Sequence[float],
    final_destination_state: Sequence[float] | None,
    lane_center_waypoints: Sequence[Mapping[str, object]],
    selected_lane_id: int | None = None,
    previous_applied_behavior: str | None = None,
    road_cfg: Mapping[str, object] | None = None,
    local_goal_cfg: Mapping[str, object] | None = None,
    mpc_constraints: Mapping[str, object] | None = None,
    target_distance_m: float | None = None,
) -> BehaviorExecutionResult:
    """
    Convert one LLM decision into MPC-friendly destination and speed overrides.

    This hardcodes the current behavior rules instead of reading the text file.
    """

    road_cfg = dict(road_cfg or {})
    local_goal_cfg = dict(local_goal_cfg or {})
    mpc_constraints = dict(mpc_constraints or {})
    previous_applied_behavior = str(previous_applied_behavior or "").strip().upper()

    ego_psi_rad = float(ego_snapshot.get("psi", 0.0))
    destination_state = _normalize_destination_state(
        destination_state=base_destination_state,
        fallback_heading_rad=ego_psi_rad,
    )
    final_destination_state = _normalize_destination_state(
        destination_state=(final_destination_state or base_destination_state),
        fallback_heading_rad=ego_psi_rad,
    )
    speed_limit_mps = float(max(0.0, mpc_constraints.get("max_velocity_mps", destination_state[2])))
    if decision is None:
        return BehaviorExecutionResult(
            destination_state=destination_state,
            selected_lane_id=int(selected_lane_id or 1),
            max_velocity_override_mps=None,
            applied_behavior="NONE",
            cav_broadcast={},
        )

    target_v_mps = _default_target_speed_for_behavior(
        behavior=decision.behavior,
        speed_limit_mps=speed_limit_mps,
        llm_target_v_mps=decision.target_v_mps,
    )
    active_target_distance_m = (
        float(target_distance_m)
        if target_distance_m is not None
        else _behavior_destination_distance_m(
            ego_snapshot=ego_snapshot,
            base_destination_state=destination_state,
            local_goal_cfg=local_goal_cfg,
        )
    )

    planner = AStarGlobalPlanner(lane_center_waypoints=lane_center_waypoints)
    local_context = planner.get_local_lane_context(
        x_m=float(ego_snapshot.get("x", 0.0)),
        y_m=float(ego_snapshot.get("y", 0.0)),
        heading_rad=float(ego_snapshot.get("psi", 0.0)),
    )
    current_lane_id = int(local_context.get("lane_id", -1))
    lane_count = _lane_count_from_inputs(
        lane_center_waypoints=lane_center_waypoints,
        road_cfg=road_cfg,
    )
    active_selected_lane_id = int(selected_lane_id) if selected_lane_id is not None else int(current_lane_id)
    active_selected_lane_id = min(
        max(1, int(active_selected_lane_id)),
        max(1, int(lane_count)),
    )

    def _destination_on_lane(target_lane_id: int, target_distance_m: float) -> List[float] | None:
        return build_destination_on_lane(
            ego_snapshot=ego_snapshot,
            lane_center_waypoints=lane_center_waypoints,
            target_lane_id=int(target_lane_id),
            target_distance_m=float(target_distance_m),
            road_cfg=road_cfg,
        )

    def _candidate_lane_after_change(raw_lane_delta: int) -> int:
        return min(
            max(1, int(active_selected_lane_id) + int(raw_lane_delta)),
            max(1, int(lane_count)),
        )

    def _try_lane_change(raw_lane_delta: int) -> tuple[int, List[float] | None]:
        candidate_lane_id = _candidate_lane_after_change(raw_lane_delta=raw_lane_delta)
        if abs(int(candidate_lane_id) - int(current_lane_id)) > 1:
            return int(active_selected_lane_id), _destination_on_lane(
                target_lane_id=int(active_selected_lane_id),
                target_distance_m=active_target_distance_m,
            )
        candidate_destination = _destination_on_lane(
            target_lane_id=int(candidate_lane_id),
            target_distance_m=active_target_distance_m,
        )
        return int(candidate_lane_id), candidate_destination

    behavior = str(decision.behavior)
    max_velocity_override_mps = None

    # Once the rolling/temporary destination reaches the final destination,
    # hold the stop target and force V_ref to zero regardless of later
    # high-level behavior outputs.
    if _same_destination_xy(destination_state, final_destination_state):
        destination_state = list(final_destination_state)
        destination_state[2] = 0.0
    elif behavior in {"LANE_KEEP", "GO_STRAIGHT"}:
        active_selected_lane_id = int(current_lane_id)
        lane_keep_destination = _destination_on_lane(
            target_lane_id=int(current_lane_id),
            target_distance_m=active_target_distance_m,
        )
        if lane_keep_destination is not None:
            destination_state = lane_keep_destination
        destination_state[2] = float(target_v_mps)
    elif behavior == "LANE_CHANGE_LEFT":
        if previous_applied_behavior == "LANE_CHANGE_LEFT":
            target_lane_id = int(active_selected_lane_id)
            lane_change_destination = _destination_on_lane(
                target_lane_id=int(active_selected_lane_id),
                target_distance_m=active_target_distance_m,
            )
        else:
            target_lane_id, lane_change_destination = _try_lane_change(raw_lane_delta=1)
        if lane_change_destination is not None:
            destination_state = lane_change_destination
        active_selected_lane_id = int(target_lane_id)
        destination_state[2] = float(target_v_mps)
    elif behavior == "LANE_CHANGE_RIGHT":
        if previous_applied_behavior == "LANE_CHANGE_RIGHT":
            target_lane_id = int(active_selected_lane_id)
            lane_change_destination = _destination_on_lane(
                target_lane_id=int(active_selected_lane_id),
                target_distance_m=active_target_distance_m,
            )
        else:
            target_lane_id, lane_change_destination = _try_lane_change(raw_lane_delta=-1)
        if lane_change_destination is not None:
            destination_state = lane_change_destination
        active_selected_lane_id = int(target_lane_id)
        destination_state[2] = float(target_v_mps)
    elif behavior == "STOP_AT_LINE":
        stop_distance_m = min(
            float(road_cfg.get("distance_to_signal_m", 10000.0)),
            float(road_cfg.get("distance_to_intersection_m", 10000.0)),
        )
        stop_destination = _destination_on_lane(
            target_lane_id=active_selected_lane_id,
            target_distance_m=max(1.0, stop_distance_m),
        )
        if stop_destination is not None:
            destination_state = stop_destination
        destination_state[2] = 0.0
    elif behavior == "EMERGENCY_BRAKE":
        min_acceleration_mps2 = float(mpc_constraints.get("min_acceleration_mps2", -3.0))
        braking_mps2 = max(1e-6, abs(min_acceleration_mps2))
        stopping_distance_m = (
            float(max(0.0, ego_snapshot.get("v", 0.0))) ** 2 / (2.0 * braking_mps2)
        )
        stop_destination = _destination_on_lane(
            target_lane_id=active_selected_lane_id,
            target_distance_m=max(1.0, stopping_distance_m),
        )
        if stop_destination is not None:
            destination_state = stop_destination
        destination_state[2] = 0.0
        max_velocity_override_mps = 0.0
    elif behavior == "TURN_LEFT":
        # Until CARLA/OpenDRIVE intersection-road routing is integrated, best
        # effort is to bias the temporary destination toward the left-adjacent
        # branch/lane at the current lookahead.
        target_lane_id, lane_change_destination = _try_lane_change(raw_lane_delta=1)
        if lane_change_destination is not None:
            destination_state = lane_change_destination
        active_selected_lane_id = int(target_lane_id)
        destination_state[2] = float(target_v_mps)
    elif behavior == "TURN_RIGHT":
        # Until CARLA/OpenDRIVE intersection-road routing is integrated, best
        # effort is to bias the temporary destination toward the right-adjacent
        # branch/lane at the current lookahead.
        target_lane_id, lane_change_destination = _try_lane_change(raw_lane_delta=-1)
        if lane_change_destination is not None:
            destination_state = lane_change_destination
        active_selected_lane_id = int(target_lane_id)
        destination_state[2] = float(target_v_mps)

    if (
        behavior in {"LANE_KEEP", "GO_STRAIGHT", "LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT", "TURN_LEFT", "TURN_RIGHT"}
        and not _same_destination_xy(destination_state, final_destination_state)
        and float(destination_state[2]) <= 1e-9
    ):
        # Recover from a previous emergency-stop V_ref=0 once a later
        # non-emergency behavior is accepted, unless the temporary destination
        # is already the final stop target.
        destination_state[2] = float(speed_limit_mps)

    if _same_destination_xy(destination_state, final_destination_state):
        destination_state = list(final_destination_state)
        destination_state[2] = 0.0

    return BehaviorExecutionResult(
        destination_state=_normalize_destination_state(
            destination_state=destination_state,
            fallback_heading_rad=float(destination_state[3]),
        ),
        selected_lane_id=int(active_selected_lane_id),
        max_velocity_override_mps=max_velocity_override_mps,
        applied_behavior=behavior,
        cav_broadcast=dict(decision.cav_broadcast),
    )

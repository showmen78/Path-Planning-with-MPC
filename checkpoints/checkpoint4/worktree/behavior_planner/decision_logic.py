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

from MPC.local_goal import build_destination_on_lane as build_mpc_destination_on_lane
from utility import AStarGlobalPlanner


ALLOWED_BEHAVIOR_DECISIONS = {
    "LANE_KEEP",
    "LANE_CHANGE_LEFT",
    "LANE_CHANGE_RIGHT",
    "STOP",
    "EMERGENCY_BRAKE",
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


def _default_target_speed_mps(
    speed_limit_mps: float,
) -> float:
    return float(max(0.0, float(speed_limit_mps)))


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
    local_lane_ids = list(range(1, max(1, int(lane_count)) + 1))

    def _clamp_lane_id_to_allowed(raw_lane_id: int | None) -> int:
        if len(local_lane_ids) == 0:
            return max(1, int(raw_lane_id or 1))
        lane_id = int(raw_lane_id if raw_lane_id is not None else local_lane_ids[0])
        if lane_id in local_lane_ids:
            return int(lane_id)
        return int(
            min(
                local_lane_ids,
                key=lambda candidate_lane_id: abs(int(candidate_lane_id) - int(lane_id)),
            )
        )

    active_selected_lane_id = _clamp_lane_id_to_allowed(
        selected_lane_id if selected_lane_id is not None else current_lane_id
    )
    default_target_v_mps = _default_target_speed_mps(
        speed_limit_mps=float(speed_limit_mps),
    )

    def _destination_on_lane(target_lane_id: int, target_distance_m: float) -> List[float] | None:
        return build_mpc_destination_on_lane(
            ego_snapshot=ego_snapshot,
            lane_center_waypoints=lane_center_waypoints,
            target_lane_id=int(target_lane_id),
            target_distance_m=float(target_distance_m),
            road_cfg=road_cfg,
        )

    def _candidate_lane_after_change(raw_lane_delta: int) -> int:
        if len(local_lane_ids) == 0:
            return int(active_selected_lane_id)
        anchor_lane_id = _clamp_lane_id_to_allowed(active_selected_lane_id)
        anchor_index = local_lane_ids.index(int(anchor_lane_id))
        lane_step = 1 if int(raw_lane_delta) > 0 else -1 if int(raw_lane_delta) < 0 else 0
        candidate_index = min(
            max(0, int(anchor_index) + int(lane_step)),
            len(local_lane_ids) - 1,
        )
        return int(local_lane_ids[int(candidate_index)])

    def _try_lane_change(raw_lane_delta: int) -> tuple[int, List[float] | None]:
        candidate_lane_id = _candidate_lane_after_change(raw_lane_delta=raw_lane_delta)
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
    else:
        destination_state[2] = float(default_target_v_mps)

    if _same_destination_xy(destination_state, final_destination_state):
        pass
    elif behavior == "LANE_KEEP":
        lane_keep_destination = _destination_on_lane(
            target_lane_id=int(active_selected_lane_id),
            target_distance_m=active_target_distance_m,
        )
        if lane_keep_destination is not None:
            destination_state = lane_keep_destination
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
    elif behavior == "STOP":
        stop_distance_m = float(
            road_cfg.get("distance_to_signal_m", road_cfg.get("distance_to_intersection_m", 10000.0))
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

    if (
        behavior in {"LANE_KEEP", "LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT"}
        and not _same_destination_xy(destination_state, final_destination_state)
        and float(destination_state[2]) <= 1e-9
    ):
        destination_state[2] = float(default_target_v_mps)

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

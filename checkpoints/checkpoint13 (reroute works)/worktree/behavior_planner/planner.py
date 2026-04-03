"""
Rule-based behavior planner.

Replaces the LLM-based planner.  Runs synchronously in the main loop
(reads cached thread results — no blocking CARLA or network calls).

Decision pipeline
-----------------
NORMAL mode
1. **Route-first preference** — keep following the global route's
   optimal lane whenever it is safe
2. **Front-blockage detour** — if the route lane loses safety because
   of an obstacle ahead, move one lane at a time to an adjacent safe lane
3. **Return-to-route** — once the route lane is safe again, move back
   toward it one lane at a time

INTERSECTION mode
1. **Route lane priority** — keep following the route-optimal lane
2. **One-step constraint** — at most one lane delta per decision
3. **No safety-driven lane preference override** — the global route
   stays in control while the blue dot is in intersection mode

Shared logic
1. **Lane-change state machine** — IDLE / CHANGING_LEFT / CHANGING_RIGHT
2. **Completion check** — lane change finishes when ego is in the target
   lane, laterally centred, and heading-aligned
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence

from .reroute import (
    CP_MESSAGE_PATH,
    load_lane_closure_messages,
    remove_cp_messages_by_id,
)

# --------------------------------------------------------------------- #
# Defaults                                                                #
# --------------------------------------------------------------------- #
DEFAULT_HYSTERESIS_DELTA = 0.15
DEFAULT_LATERAL_COMPLETE_M = 0.8
DEFAULT_HEADING_COMPLETE_RAD = math.radians(8.0)
DEFAULT_LANE_CHANGE_TARGET_SAFETY_THRESHOLD = 0.10
DEFAULT_MOVING_OBSTACLE_SPEED_THRESHOLD_MPS = 0.5
DEFAULT_COOPERATIVE_MESSAGE_CHECK_FREQUENCY_HZ = 1.0

# Lane-change states
_IDLE = "IDLE"
_CHANGING_LEFT = "CHANGING_LEFT"
_CHANGING_RIGHT = "CHANGING_RIGHT"

_DECISION_FOLLOW = "lane_follow"
_DECISION_CHANGE_LEFT = "lane_change_left"
_DECISION_CHANGE_RIGHT = "lane_change_right"
_DECISION_REROUTE = "reroute"

_MANEUVER_LEFT = "left"
_MANEUVER_RIGHT = "right"
_MANEUVER_STRAIGHT = "straight"


def normalize_behavior_decision(decision: str | None) -> str:
    normalized_name = (
        str(decision or "")
        .strip()
        .upper()
        .replace(" ", "_")
    )
    if normalized_name in {"LANE_CHANGE_LEFT", "CHANGE_LEFT"}:
        return _DECISION_CHANGE_LEFT
    if normalized_name in {"LANE_CHANGE_RIGHT", "CHANGE_RIGHT"}:
        return _DECISION_CHANGE_RIGHT
    if normalized_name in {"REROUTE", "RE_ROUTE"}:
        return _DECISION_REROUTE
    if normalized_name in {"LANE_KEEP", "KEEP_LANE", "KEEP"}:
        return _DECISION_FOLLOW
    return _DECISION_FOLLOW


def normalize_macro_maneuver(next_macro_maneuver: str | None) -> str:
    normalized_name = (
        str(next_macro_maneuver or "")
        .strip()
        .upper()
        .replace(" ", "_")
    )
    if normalized_name in {"LEFT", "LEFT_TURN", "LEFTTURN"}:
        return _MANEUVER_LEFT
    if normalized_name in {"RIGHT", "RIGHT_TURN", "RIGHTTURN"}:
        return _MANEUVER_RIGHT
    return _MANEUVER_STRAIGHT


def intersection_route_follow_maneuver(
    mode: str,
    next_macro_maneuver: str | None,
    decision: str,
    target_lane_id: int,
    available_lane_ids: Sequence[int],
    current_road_option: str | None = None,
) -> str:
    mode_name = str(mode or "NORMAL").strip().upper()
    del decision
    del target_lane_id
    del available_lane_ids

    maneuver_name = normalize_macro_maneuver(next_macro_maneuver)
    normalized_current_option = normalize_macro_maneuver(current_road_option)

    if mode_name != "INTERSECTION":
        return str(maneuver_name)
    if normalized_current_option in {_MANEUVER_LEFT, _MANEUVER_RIGHT}:
        return _DECISION_FOLLOW
    return str(maneuver_name)


def evaluate_intersection_obstacle_response(
    mode: str,
    front_obstacle_speed_mps: float | None,
    original_max_velocity_mps: float,
    moving_obstacle_speed_threshold_mps: float = DEFAULT_MOVING_OBSTACLE_SPEED_THRESHOLD_MPS,
    route_lane_safety_score: float | None = None,
    static_obstacle_replan_lane_safety_threshold: float = 0.5,
) -> Dict[str, Any]:
    response = {
        "speed_cap_mps": float(max(0.0, float(original_max_velocity_mps))),
        "follow_moving_obstacle": False,
        "request_static_obstacle_replan": False,
    }

    if str(mode or "NORMAL").strip().upper() != "INTERSECTION":
        return response
    if front_obstacle_speed_mps is None:
        return response

    obstacle_speed_mps = max(0.0, float(front_obstacle_speed_mps))
    if obstacle_speed_mps > max(0.0, float(moving_obstacle_speed_threshold_mps)):
        response["speed_cap_mps"] = min(
            float(response["speed_cap_mps"]),
            float(obstacle_speed_mps),
        )
        response["follow_moving_obstacle"] = True
        return response

    if (
        route_lane_safety_score is not None
        and float(route_lane_safety_score) < float(static_obstacle_replan_lane_safety_threshold)
    ):
        response["request_static_obstacle_replan"] = True
    return response


class RuleBasedBehaviorPlanner:
    """Stateful rule-based lane-selection planner."""

    def __init__(
        self,
        hysteresis_delta: float = DEFAULT_HYSTERESIS_DELTA,
        lateral_complete_m: float = DEFAULT_LATERAL_COMPLETE_M,
        heading_complete_rad: float = DEFAULT_HEADING_COMPLETE_RAD,
        lane_change_target_safety_threshold: float | None = None,
        intersection_lane_change_safety_threshold: float | None = None,
        cp_message_path: str | None = None,
        cooperative_message_check_frequency_hz: float = DEFAULT_COOPERATIVE_MESSAGE_CHECK_FREQUENCY_HZ,
    ) -> None:
        self._hysteresis = float(hysteresis_delta)
        self._lateral_complete = float(lateral_complete_m)
        self._heading_complete = float(heading_complete_rad)
        resolved_target_lane_safety_threshold = (
            lane_change_target_safety_threshold
            if lane_change_target_safety_threshold is not None
            else intersection_lane_change_safety_threshold
        )
        if resolved_target_lane_safety_threshold is None:
            resolved_target_lane_safety_threshold = (
                DEFAULT_LANE_CHANGE_TARGET_SAFETY_THRESHOLD
            )
        self._target_lane_safety_threshold = float(
            resolved_target_lane_safety_threshold
        )
        self._cp_message_path = None if cp_message_path is None else str(cp_message_path)
        self._cp_message_check_period_s = (
            0.0
            if float(cooperative_message_check_frequency_hz) <= 0.0
            else 1.0 / float(cooperative_message_check_frequency_hz)
        )
        self._last_cp_message_check_time_s: float | None = None
        self._processed_cp_message_ids: set[str] = set()

        # Persistent state across calls
        self._lc_state: str = _IDLE
        self._target_lane_id: int | None = None
        self._last_mode: str = "NORMAL"

    # ----------------------------------------------------------------- #
    # Public                                                              #
    # ----------------------------------------------------------------- #
    def update(
        self,
        lane_safety_scores: Dict[int, float],
        ego_lane_id: int,
        ego_lateral_offset_m: float = 0.0,
        ego_heading_error_rad: float = 0.0,
        mode: str = "NORMAL",
        route_optimal_lane_id: int | None = None,
        next_macro_maneuver: str | None = None,
        front_obstacle_distance_by_lane: Mapping[int, float] | None = None,
        current_time_s: float | None = None,
    ) -> Dict[str, Any]:
        """
        Run one planning cycle.

        Parameters
        ----------
        lane_safety_scores    : {lane_id: score}  score in [0, 1]
        ego_lane_id           : current internal lane id
        ego_lateral_offset_m  : signed lateral offset from lane centre
        ego_heading_error_rad : heading error w.r.t. lane direction

        mode                  : "NORMAL" | "INTERSECTION"
        route_optimal_lane_id : lane that the global route wants at the
                                 current planning position.
        next_macro_maneuver   : upcoming route maneuver used by the
                                 intersection-mode latch.
        front_obstacle_distance_by_lane :
                                 optional nearest front-obstacle distance
                                 for each lane. NORMAL mode uses this to
                                 confirm that a route-lane safety drop is
                                 caused by a blockage ahead.
        current_time_s       : optional simulation time used to rate-limit
                                 cooperative-message polling.

        Returns
        -------
        {"decision": "lane_follow" | "lane_change_left" | "lane_change_right" | "reroute",
         "target_lane_id": int,
         "lc_state": str}
        """
        if len(lane_safety_scores) == 0:
            available = [int(ego_lane_id)]
        else:
            available = sorted(lane_safety_scores.keys())
            if int(ego_lane_id) not in available and len(available) > 0:
                ego_lane_id = min(available, key=lambda lane_id: abs(int(lane_id) - int(ego_lane_id)))

        planner_mode = str(mode or "NORMAL").strip().upper()
        if str(planner_mode) != str(self._last_mode):
            self._lc_state = _IDLE
            self._target_lane_id = None
        self._last_mode = str(planner_mode)

        reroute_messages = self._poll_cooperative_messages(current_time_s=current_time_s)
        if len(reroute_messages) > 0:
            self._lc_state = _IDLE
            self._target_lane_id = None
            reroute_ids = [
                str(message.get("id", "")).strip()
                for message in list(reroute_messages)
                if str(message.get("id", "")).strip()
            ]
            print(
                "[BEHAVIOR] decision=reroute triggered by cooperative message ids="
                f"{reroute_ids}"
            )
            return self._make_result(
                _DECISION_REROUTE,
                int(ego_lane_id),
                reroute_messages=reroute_messages,
            )

        if len(lane_safety_scores) == 0:
            return self._make_result(_DECISION_FOLLOW, ego_lane_id)

        # -------------------------------------------------------------- #
        # Lane-change completion (checked first)                            #
        # -------------------------------------------------------------- #
        if self._lc_state != _IDLE and self._target_lane_id is not None:
            if self._lane_change_complete(
                ego_lane_id=ego_lane_id,
                target_lane_id=self._target_lane_id,
                lateral_offset_m=ego_lateral_offset_m,
                heading_error_rad=ego_heading_error_rad,
            ):
                self._lc_state = _IDLE
                self._target_lane_id = None

        # -------------------------------------------------------------- #
        # Do NOT interrupt ongoing lane change                              #
        # -------------------------------------------------------------- #
        if self._lc_state != _IDLE and self._target_lane_id is not None:
            if self._lc_state == _CHANGING_LEFT:
                return self._make_result(_DECISION_CHANGE_LEFT, self._target_lane_id)
            else:
                return self._make_result(_DECISION_CHANGE_RIGHT, self._target_lane_id)

        # -------------------------------------------------------------- #
        # INTERSECTION mode: route-optimal lane takes priority             #
        # -------------------------------------------------------------- #
        if planner_mode == "INTERSECTION":
            desired_lane_id = self._intersection_target_lane_id(
                ego_lane_id=int(ego_lane_id),
                route_optimal_lane_id=route_optimal_lane_id,
                next_macro_maneuver=next_macro_maneuver,
                available_lane_ids=available,
            )
            if desired_lane_id is None or int(desired_lane_id) == int(ego_lane_id):
                return self._make_result(_DECISION_FOLLOW, ego_lane_id)
            return self._start_one_step_lane_change(
                ego_lane_id=int(ego_lane_id),
                desired_lane_id=int(desired_lane_id),
                available_lane_ids=available,
                lane_safety_scores=lane_safety_scores,
                min_target_lane_safety=self._target_lane_safety_threshold,
            )

        # -------------------------------------------------------------- #
        # NORMAL mode: follow the route lane unless a front blockage      #
        # forces a temporary adjacent-lane detour.                        #
        # -------------------------------------------------------------- #
        route_lane_id = self._preferred_route_lane_id(
            route_optimal_lane_id=route_optimal_lane_id,
            available_lane_ids=available,
            fallback_lane_id=int(ego_lane_id),
        )
        route_lane_score = float(lane_safety_scores.get(int(route_lane_id), 0.0))
        current_lane_score = float(lane_safety_scores.get(int(ego_lane_id), 0.0))
        route_lane_has_front_obstacle = self._lane_has_front_obstacle(
            lane_id=int(route_lane_id),
            front_obstacle_distance_by_lane=front_obstacle_distance_by_lane,
        )
        current_lane_has_front_obstacle = self._lane_has_front_obstacle(
            lane_id=int(ego_lane_id),
            front_obstacle_distance_by_lane=front_obstacle_distance_by_lane,
        )

        if int(ego_lane_id) == int(route_lane_id):
            detour_lane_id = self._best_adjacent_safe_lane(
                reference_lane_id=int(ego_lane_id),
                available_lane_ids=available,
                lane_safety_scores=lane_safety_scores,
            )
            if detour_lane_id is None:
                return self._make_result(_DECISION_FOLLOW, ego_lane_id)

            detour_lane_score = float(
                lane_safety_scores.get(int(detour_lane_id), 0.0)
            )
            if (
                route_lane_has_front_obstacle
                and detour_lane_score > max(
                    route_lane_score + self._hysteresis,
                    self._target_lane_safety_threshold,
                )
            ):
                return self._start_one_step_lane_change(
                    ego_lane_id=int(ego_lane_id),
                    desired_lane_id=int(detour_lane_id),
                    available_lane_ids=available,
                    lane_safety_scores=lane_safety_scores,
                    min_target_lane_safety=self._target_lane_safety_threshold,
                )
            return self._make_result(_DECISION_FOLLOW, ego_lane_id)

        route_lane_safe_again = (
            route_lane_score > self._target_lane_safety_threshold
            and not route_lane_has_front_obstacle
        )
        if route_lane_safe_again:
            return self._start_one_step_lane_change(
                ego_lane_id=int(ego_lane_id),
                desired_lane_id=int(route_lane_id),
                available_lane_ids=available,
                lane_safety_scores=lane_safety_scores,
                min_target_lane_safety=self._target_lane_safety_threshold,
            )

        if current_lane_has_front_obstacle:
            excluded_lane_ids = {int(route_lane_id)} if route_lane_has_front_obstacle else set()
            escape_lane_id = self._best_adjacent_safe_lane(
                reference_lane_id=int(ego_lane_id),
                available_lane_ids=available,
                lane_safety_scores=lane_safety_scores,
                excluded_lane_ids=excluded_lane_ids,
            )
            if escape_lane_id is not None:
                escape_lane_score = float(
                    lane_safety_scores.get(int(escape_lane_id), 0.0)
                )
                if escape_lane_score > max(
                    current_lane_score + self._hysteresis,
                    self._target_lane_safety_threshold,
                ):
                    return self._start_one_step_lane_change(
                        ego_lane_id=int(ego_lane_id),
                        desired_lane_id=int(escape_lane_id),
                        available_lane_ids=available,
                        lane_safety_scores=lane_safety_scores,
                        min_target_lane_safety=self._target_lane_safety_threshold,
                    )

        return self._make_result(_DECISION_FOLLOW, ego_lane_id)

    # ----------------------------------------------------------------- #
    # Helpers                                                             #
    # ----------------------------------------------------------------- #
    def _poll_cooperative_messages(
        self,
        *,
        current_time_s: float | None,
    ) -> Sequence[Mapping[str, object]]:
        if not self._should_check_cooperative_messages(current_time_s=current_time_s):
            return []
        if not self._cp_message_path:
            return []
        if current_time_s is not None:
            self._last_cp_message_check_time_s = float(current_time_s)
        current_messages = load_lane_closure_messages(message_path=self._cp_message_path)
        current_message_ids = {
            str(message.get("id", "")).strip()
            for message in list(current_messages)
            if str(message.get("id", "")).strip()
        }
        self._processed_cp_message_ids.intersection_update(current_message_ids)

        new_messages = [
            dict(message)
            for message in list(current_messages)
            if str(message.get("id", "")).strip()
            and str(message.get("id", "")).strip() not in self._processed_cp_message_ids
        ]
        handled_message_ids = [
            str(message.get("id", "")).strip()
            for message in list(new_messages)
            if str(message.get("id", "")).strip()
        ]
        if len(handled_message_ids) > 0:
            remove_cp_messages_by_id(
                handled_message_ids,
                message_path=self._cp_message_path,
            )
        for message in new_messages:
            message_id = str(message.get("id", "")).strip()
            if message_id:
                self._processed_cp_message_ids.add(message_id)
        return new_messages

    def _should_check_cooperative_messages(
        self,
        *,
        current_time_s: float | None,
    ) -> bool:
        if current_time_s is None:
            return True
        if self._last_cp_message_check_time_s is None:
            return True
        if float(current_time_s) < float(self._last_cp_message_check_time_s):
            return True
        return (
            float(current_time_s) - float(self._last_cp_message_check_time_s)
        ) >= float(self._cp_message_check_period_s)

    def _intersection_target_lane_id(
        self,
        ego_lane_id: int,
        route_optimal_lane_id: int | None,
        next_macro_maneuver: str | None,
        available_lane_ids: Sequence[int],
    ) -> int | None:
        del next_macro_maneuver

        if len(available_lane_ids) == 0:
            return int(ego_lane_id)
        return int(
            self._preferred_route_lane_id(
                route_optimal_lane_id=route_optimal_lane_id,
                available_lane_ids=available_lane_ids,
                fallback_lane_id=int(ego_lane_id),
            )
        )

    @staticmethod
    def _preferred_route_lane_id(
        route_optimal_lane_id: int | None,
        available_lane_ids: Sequence[int],
        fallback_lane_id: int,
    ) -> int:
        if len(available_lane_ids) == 0:
            return int(fallback_lane_id)
        if route_optimal_lane_id is None:
            return int(fallback_lane_id)
        normalized_route_lane_id = int(route_optimal_lane_id)
        if normalized_route_lane_id in available_lane_ids:
            return int(normalized_route_lane_id)
        if normalized_route_lane_id <= 0:
            return int(fallback_lane_id)
        return int(
            min(
                available_lane_ids,
                key=lambda lane_id: abs(int(lane_id) - int(normalized_route_lane_id)),
            )
        )

    @staticmethod
    def _lane_has_front_obstacle(
        lane_id: int,
        front_obstacle_distance_by_lane: Mapping[int, float] | None,
    ) -> bool:
        if not isinstance(front_obstacle_distance_by_lane, Mapping):
            return False
        if int(lane_id) not in front_obstacle_distance_by_lane:
            return False
        try:
            return math.isfinite(
                float(front_obstacle_distance_by_lane.get(int(lane_id), float("inf")))
            )
        except Exception:
            return False

    def _best_adjacent_safe_lane(
        self,
        reference_lane_id: int,
        available_lane_ids: Sequence[int],
        lane_safety_scores: Mapping[int, float],
        excluded_lane_ids: Sequence[int] | None = None,
    ) -> int | None:
        excluded = {int(lane_id) for lane_id in list(excluded_lane_ids or [])}
        candidates = [
            int(lane_id)
            for lane_id in (int(reference_lane_id) - 1, int(reference_lane_id) + 1)
            if int(lane_id) in available_lane_ids and int(lane_id) not in excluded
        ]
        if len(candidates) == 0:
            return None

        safe_candidates = [
            lane_id
            for lane_id in candidates
            if float(lane_safety_scores.get(int(lane_id), 0.0))
            > float(self._target_lane_safety_threshold)
        ]
        if len(safe_candidates) == 0:
            return None

        return int(
            max(
                safe_candidates,
                key=lambda lane_id: float(lane_safety_scores.get(int(lane_id), 0.0)),
            )
        )

    def _start_one_step_lane_change(
        self,
        ego_lane_id: int,
        desired_lane_id: int,
        available_lane_ids: Sequence[int],
        lane_safety_scores: Mapping[int, float] | None = None,
        min_target_lane_safety: float | None = None,
    ) -> Dict[str, Any]:
        if int(desired_lane_id) > int(ego_lane_id):
            target_lane_id = int(ego_lane_id) + 1
            decision = _DECISION_CHANGE_LEFT
        elif int(desired_lane_id) < int(ego_lane_id):
            target_lane_id = int(ego_lane_id) - 1
            decision = _DECISION_CHANGE_RIGHT
        else:
            return self._make_result(_DECISION_FOLLOW, int(ego_lane_id))

        if int(target_lane_id) not in available_lane_ids:
            return self._make_result(_DECISION_FOLLOW, int(ego_lane_id))

        if min_target_lane_safety is not None:
            target_lane_safety = float(
                (lane_safety_scores or {}).get(int(target_lane_id), 0.0)
            )
            if not (float(target_lane_safety) > float(min_target_lane_safety)):
                return self._make_result(_DECISION_FOLLOW, int(ego_lane_id))

        return self._enter_lane_change_state(
            decision=str(decision),
            target_lane_id=int(target_lane_id),
        )

    def _enter_lane_change_state(self, decision: str, target_lane_id: int) -> Dict[str, Any]:
        normalized_decision = normalize_behavior_decision(decision)
        self._lc_state = (
            _CHANGING_LEFT if str(normalized_decision) == _DECISION_CHANGE_LEFT else _CHANGING_RIGHT
        )
        self._target_lane_id = int(target_lane_id)
        return self._make_result(str(normalized_decision), int(target_lane_id))

    def _lane_change_complete(
        self,
        ego_lane_id: int,
        target_lane_id: int,
        lateral_offset_m: float,
        heading_error_rad: float,
    ) -> bool:
        """Check if the ongoing lane change has finished."""
        if ego_lane_id != target_lane_id:
            return False
        if abs(lateral_offset_m) > self._lateral_complete:
            return False
        if abs(heading_error_rad) > self._heading_complete:
            return False
        return True

    def _make_result(
        self,
        decision: str,
        target_lane_id: int,
        reroute_messages: Sequence[Mapping[str, object]] | None = None,
    ) -> Dict[str, Any]:
        result = {
            "decision": str(normalize_behavior_decision(decision)),
            "target_lane_id": int(target_lane_id),
            "lc_state": str(self._lc_state),
        }
        if reroute_messages is not None:
            result["reroute_messages"] = [dict(message) for message in list(reroute_messages or [])]
        return result

    @property
    def lc_state(self) -> str:
        return str(self._lc_state)

    @property
    def target_lane_id(self) -> int | None:
        return self._target_lane_id

    def reset(self) -> None:
        """Reset lane-change state machine (e.g. on scenario restart)."""
        self._lc_state = _IDLE
        self._target_lane_id = None
        self._last_mode = "NORMAL"
        self._last_cp_message_check_time_s = None
        self._processed_cp_message_ids.clear()

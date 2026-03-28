"""
Rule-based behavior planner.

Replaces the LLM-based planner.  Runs synchronously in the main loop
(reads cached thread results — no blocking CARLA or network calls).

Decision pipeline
-----------------
NORMAL mode
1. **Candidate lane** — safest lane overall (highest safety score)
2. **Hysteresis** — only switch if the candidate is meaningfully better
3. **One-step constraint** — at most one lane delta per decision

INTERSECTION mode
1. **Turn preparation** — use the next route maneuver
2. **Left / Right** — move one lane at a time toward the leftmost /
   rightmost valid driving lane
3. **Straight** — fall back to NORMAL mode immediately
4. **After reaching the turn lane** — stay there until the mode returns
   to NORMAL

Shared logic
1. **Lane-change state machine** — IDLE / CHANGING_LEFT / CHANGING_RIGHT
2. **Completion check** — lane change finishes when ego is in the target
   lane, laterally centred, and heading-aligned
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence

# --------------------------------------------------------------------- #
# Defaults                                                                #
# --------------------------------------------------------------------- #
DEFAULT_HYSTERESIS_DELTA = 0.15
DEFAULT_LATERAL_COMPLETE_M = 0.8
DEFAULT_HEADING_COMPLETE_RAD = math.radians(8.0)

# Lane-change states
_IDLE = "IDLE"
_CHANGING_LEFT = "CHANGING_LEFT"
_CHANGING_RIGHT = "CHANGING_RIGHT"

_DECISION_FOLLOW = "lane_follow"
_DECISION_CHANGE_LEFT = "lane_change_left"
_DECISION_CHANGE_RIGHT = "lane_change_right"

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


class RuleBasedBehaviorPlanner:
    """Stateful rule-based lane-selection planner."""

    def __init__(
        self,
        hysteresis_delta: float = DEFAULT_HYSTERESIS_DELTA,
        lateral_complete_m: float = DEFAULT_LATERAL_COMPLETE_M,
        heading_complete_rad: float = DEFAULT_HEADING_COMPLETE_RAD,
    ) -> None:
        self._hysteresis = float(hysteresis_delta)
        self._lateral_complete = float(lateral_complete_m)
        self._heading_complete = float(heading_complete_rad)

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
        route_optimal_lane_id : lane that the global route wants for the
                                 upcoming maneuver. Used only in
                                 INTERSECTION mode.
        next_macro_maneuver   : upcoming route maneuver. In INTERSECTION
                                 mode, LEFT / RIGHT drive the blue dot
                                 toward the extreme turn lane. STRAIGHT
                                 falls back to NORMAL logic.

        Returns
        -------
        {"decision": "lane_follow" | "lane_change_left" | "lane_change_right",
         "target_lane_id": int,
         "lc_state": str}
        """
        if len(lane_safety_scores) == 0:
            return self._make_result(_DECISION_FOLLOW, ego_lane_id)

        available = sorted(lane_safety_scores.keys())
        if int(ego_lane_id) not in available and len(available) > 0:
            ego_lane_id = min(available, key=lambda lane_id: abs(int(lane_id) - int(ego_lane_id)))

        planner_mode = str(mode or "NORMAL").strip().upper()
        if str(planner_mode) != str(self._last_mode):
            self._lc_state = _IDLE
            self._target_lane_id = None
        self._last_mode = str(planner_mode)

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
            if desired_lane_id is None:
                planner_mode = "NORMAL"
            else:
                if int(desired_lane_id) == int(ego_lane_id):
                    return self._make_result(_DECISION_FOLLOW, ego_lane_id)
                return self._start_one_step_lane_change(
                    ego_lane_id=int(ego_lane_id),
                    desired_lane_id=int(desired_lane_id),
                    available_lane_ids=available,
                )

        # -------------------------------------------------------------- #
        # Candidate lane — safest overall                                   #
        # -------------------------------------------------------------- #
        i_star = max(available, key=lambda lid: lane_safety_scores.get(lid, 0.0))

        # -------------------------------------------------------------- #
        # Hysteresis — only change if significantly better                  #
        # -------------------------------------------------------------- #
        s_current = lane_safety_scores.get(ego_lane_id, 0.0)
        s_candidate = lane_safety_scores.get(i_star, 0.0)

        if s_candidate <= s_current + self._hysteresis:
            i_star = ego_lane_id  # stay

        # -------------------------------------------------------------- #
        # One lane step at a time                                           #
        # -------------------------------------------------------------- #
        if i_star > ego_lane_id:
            target = ego_lane_id + 1
            decision = _DECISION_CHANGE_LEFT
        elif i_star < ego_lane_id:
            target = ego_lane_id - 1
            decision = _DECISION_CHANGE_RIGHT
        else:
            return self._make_result(_DECISION_FOLLOW, ego_lane_id)

        if target not in available:
            return self._make_result(_DECISION_FOLLOW, ego_lane_id)

        return self._enter_lane_change_state(
            decision=str(decision),
            target_lane_id=int(target),
        )

    # ----------------------------------------------------------------- #
    # Helpers                                                             #
    # ----------------------------------------------------------------- #
    def _intersection_target_lane_id(
        self,
        ego_lane_id: int,
        route_optimal_lane_id: int | None,
        next_macro_maneuver: str | None,
        available_lane_ids: Sequence[int],
    ) -> int | None:
        del route_optimal_lane_id

        if len(available_lane_ids) == 0:
            return int(ego_lane_id)

        maneuver_name = normalize_macro_maneuver(next_macro_maneuver)
        if maneuver_name == _MANEUVER_LEFT:
            return int(max(available_lane_ids))
        if maneuver_name == _MANEUVER_RIGHT:
            return int(min(available_lane_ids))
        if maneuver_name == _MANEUVER_STRAIGHT:
            return None
        return int(ego_lane_id)

    def _start_one_step_lane_change(
        self,
        ego_lane_id: int,
        desired_lane_id: int,
        available_lane_ids: Sequence[int],
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

    def _make_result(self, decision: str, target_lane_id: int) -> Dict[str, Any]:
        return {
            "decision": str(normalize_behavior_decision(decision)),
            "target_lane_id": int(target_lane_id),
            "lc_state": str(self._lc_state),
        }

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

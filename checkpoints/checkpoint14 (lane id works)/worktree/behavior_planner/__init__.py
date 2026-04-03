"""Behavior-planner package exports."""

# -- Rule-based planner (new) ----------------------------------------- #
from .lane_safety import LaneSafetyScorer, LaneSafetyThread
from .planner import (
    evaluate_intersection_obstacle_response,
    RuleBasedBehaviorPlanner,
    intersection_route_follow_maneuver,
    normalize_behavior_decision,
    normalize_macro_maneuver,
)
from .temp_destination import (
    build_reference_samples,
    compute_ego_lane_offset,
    compute_temp_destination_mode,
    compute_temp_destination,
)
from .reroute import (
    CP_MESSAGE_PATH,
    ensure_cp_message_file_exists,
    lane_closure_messages,
    load_cp_messages,
    load_lane_closure_messages,
    pop_lane_closure_messages,
    remove_cp_messages_by_id,
    reroute_from_lane_closure_messages,
)
from .traffic_light_stop import (
    find_relevant_signal_context,
    find_stop_target_from_ego,
    normalize_signal_state,
    should_stop_for_signal,
)

# -- Legacy LLM-based planner (kept for reference / fallback) ---------- #
from .api_client import BehaviorPlannerAPIClient
from .decision_logic import (
    BehaviorExecutionResult,
    BehaviorPlannerDecision,
    apply_behavior_planner_decision,
    decision_from_mapping,
    parse_behavior_planner_response,
)
from .intention import (
    SurroundingVehicleSummary,
    compute_safe_zone_distance,
    infer_vehicle_intention,
    summarize_surrounding_vehicle,
)
from .prompt_builder import BehaviorPlannerPromptBuilder, build_behavior_planner_prompt

__all__ = [
    # Rule-based planner
    "LaneSafetyScorer",
    "LaneSafetyThread",
    "evaluate_intersection_obstacle_response",
    "RuleBasedBehaviorPlanner",
    "intersection_route_follow_maneuver",
    "normalize_behavior_decision",
    "normalize_macro_maneuver",
    "build_reference_samples",
    "compute_ego_lane_offset",
    "compute_temp_destination_mode",
    "compute_temp_destination",
    "CP_MESSAGE_PATH",
    "ensure_cp_message_file_exists",
    "lane_closure_messages",
    "load_cp_messages",
    "load_lane_closure_messages",
    "pop_lane_closure_messages",
    "remove_cp_messages_by_id",
    "reroute_from_lane_closure_messages",
    "find_relevant_signal_context",
    "find_stop_target_from_ego",
    "normalize_signal_state",
    "should_stop_for_signal",
    # Legacy
    "BehaviorPlannerAPIClient",
    "BehaviorExecutionResult",
    "BehaviorPlannerDecision",
    "apply_behavior_planner_decision",
    "decision_from_mapping",
    "parse_behavior_planner_response",
    "SurroundingVehicleSummary",
    "compute_safe_zone_distance",
    "infer_vehicle_intention",
    "summarize_surrounding_vehicle",
    "BehaviorPlannerPromptBuilder",
    "build_behavior_planner_prompt",
]

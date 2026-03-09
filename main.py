"""
Main entrypoint for MPC_custom.

Project execution flow (organized per PDF folder structure):
1. Load scenario configuration (`scenarios/<name>/scenario.yaml`).
2. Merge subsystem default YAMLs from vehicle_manager / road /
   utility / state_manager / MPC with scenario overrides.
3. Build the simulation objects and state manager.
4. Track and predict non-ego object trajectories.
5. Replan ego trajectory with MPC at a configurable frequency (default 2 Hz).
6. Follow the latest planned trajectory between replans using a PID trajectory
   tracker that outputs [acceleration, steering] and moves the ego through the
   vehicle kinematic model at dt = 0.05 s.
7. Render road, vehicles, destination, predicted obstacle paths, and ego path.

Run:
    python main.py scenario1
    python main.py scenario2
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence
import math
import os
import sys
import time
import warnings

# Hide pygame banner and ignore a known wheel warning that does not affect correctness.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
warnings.filterwarnings(
    "ignore",
    message=r".*avx2 capable but pygame was not built with support for it.*",
    category=RuntimeWarning,
)

import pygame

from MPC import MPC
from scenarios import load_scenario_by_name
from plot import SimulationPlotter
from state_manager import StateManager
from utility import Tracker, TrajectoryPIDController, deep_merge_dicts, load_yaml_file
from utility.rendering import (
    draw_destination,
    draw_dotted_trajectory,
    draw_hud_text,
    draw_predicted_object_trajectories,
    draw_world_scale,
    screen_to_world,
    world_to_screen,
)
from vehicle_manager import build_vehicles_from_config, compute_non_ego_control, find_ego_vehicle


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _load_subsystem_default_config() -> Dict[str, Any]:
    """
    Intent:
        Load and merge folder-level default YAML files.

    Why this exists:
        The user requested configurable variables in YAML files inside each
        folder. This loader assembles those defaults before scenario overrides
        are applied.
    """

    default_yaml_paths = [
        os.path.join(PROJECT_ROOT, "vehicle_manager", "vehicle_manager.yaml"),
        os.path.join(PROJECT_ROOT, "road", "road.yaml"),
        os.path.join(PROJECT_ROOT, "utility", "tracker.yaml"),
        os.path.join(PROJECT_ROOT, "utility", "pid_controller.yaml"),
        os.path.join(PROJECT_ROOT, "state_manager", "state_manager.yaml"),
        os.path.join(PROJECT_ROOT, "MPC", "mpc.yaml"),
    ]
    merged: Dict[str, Any] = {}
    for yaml_path in default_yaml_paths:
        if not os.path.exists(yaml_path):
            continue
        merged = deep_merge_dicts(merged, load_yaml_file(yaml_path))
    return merged


def _compose_runtime_config(scenario_config: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Intent:
        Create the final runtime config by merging subsystem defaults with the
        scenario-specific YAML.

    Priority:
        scenario config overrides subsystem defaults.
    """

    subsystem_defaults = _load_subsystem_default_config()
    merged = deep_merge_dicts(subsystem_defaults, dict(scenario_config))

    return merged


def _wrap_angle(angle_rad: float) -> float:
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def _is_inside_render_window(
    object_snapshot: Mapping[str, object],
    camera_center_world: Sequence[float],
    pixels_per_meter: float,
    screen_size_px: Sequence[float],
    margin_m: float = 0.5,
) -> bool:
    """
    Return True when the object's center lies inside the current rendered window.
    """

    if len(camera_center_world) < 2 or len(screen_size_px) < 2 or float(pixels_per_meter) <= 1e-9:
        return True

    x_m = float(object_snapshot.get("x", 0.0))
    y_m = float(object_snapshot.get("y", 0.0))
    camera_x_m = float(camera_center_world[0])
    camera_y_m = float(camera_center_world[1])
    half_width_m = 0.5 * float(screen_size_px[0]) / float(pixels_per_meter)
    half_height_m = 0.5 * float(screen_size_px[1]) / float(pixels_per_meter)
    margin_m = max(0.0, float(margin_m))

    return (
        (camera_x_m - half_width_m - margin_m) <= x_m <= (camera_x_m + half_width_m + margin_m)
        and (camera_y_m - half_height_m - margin_m) <= y_m <= (camera_y_m + half_height_m + margin_m)
    )


def _track_ego_with_pid(
    ego_vehicle,
    trajectory_pid_controller: TrajectoryPIDController,
    planned_states: Sequence[Sequence[float]],
    plan_cursor: int,
    sim_dt_s: float,
) -> int:
    """
    Intent:
        Track the currently planned MPC trajectory with a PID controller and
        move the ego vehicle through its kinematic model.

    Inputs:
        trajectory_pid_controller:
            `TrajectoryPIDController` instance used to convert the target
            trajectory state into [acceleration, steering] commands.
        planned_states:
            sequence of future MPC states [x, y, v, psi].
        plan_cursor:
            int index of the current trajectory target waypoint.
        sim_dt_s:
            float [s], simulation integration step used by `ego_vehicle.step`.

    Output:
        int, updated trajectory cursor after one PID tracking/control step.
    """

    if len(planned_states) == 0 or plan_cursor >= len(planned_states):
        ego_vehicle.set_control(0.0, 0.0)
        return int(plan_cursor)

    current_state = [float(v) for v in ego_vehicle.current_state]
    plan_cursor = trajectory_pid_controller.advance_target_index(
        current_state=current_state,
        planned_states=planned_states,
        plan_cursor=int(plan_cursor),
    )

    if plan_cursor >= len(planned_states):
        ego_vehicle.set_control(0.0, 0.0)
        return int(plan_cursor)

    lookahead_steps = max(0, int(trajectory_pid_controller.config.lookahead_steps))
    target_idx = min(int(plan_cursor) + lookahead_steps, len(planned_states) - 1)
    next_target_idx = min(target_idx + 1, len(planned_states) - 1)
    target_state = planned_states[target_idx]
    next_target_state = planned_states[next_target_idx] if next_target_idx != target_idx else None

    accel_cmd, steer_cmd, _pid_debug = trajectory_pid_controller.compute_control(
        current_state=current_state,
        target_state=target_state,
        next_target_state=next_target_state,
        limits={
            "min_velocity_mps": float(ego_vehicle.min_velocity_mps),
            "max_velocity_mps": float(ego_vehicle.max_velocity_mps),
            "max_acceleration_mps2": float(ego_vehicle.max_acceleration_mps2),
            "max_steer_rad": float(ego_vehicle.max_steer_rad),
        },
    )
    ego_vehicle.set_control(accel_cmd, steer_cmd)
    ego_vehicle.step(float(sim_dt_s))

    # Do not advance the cursor a second time in the same simulation step.
    # A double-advance can consume the planned trajectory too quickly, which
    # makes rolling-goal updates appear late and causes visible plan resets.
    return int(plan_cursor)


def _destination_for_mpc(
    destination_state: Sequence[float],
    max_velocity_mps: float,
    ego_state: Sequence[float] | None = None,
) -> List[float]:
    """
    Intent:
        Convert scenario destination state into the destination input passed to
        the MPC.

    Logic:
        - If destination explicitly provides non-zero v/psi, keep them.
        - If destination explicitly provides psi but uses v=0 as a placeholder,
          keep the heading and use the default terminal speed reference.
        - If destination omits v/psi (or uses [x,y,0,0] placeholders), compute
          a heading reference from the current ego position toward the target to
          avoid forcing psi_ref = 0 for arbitrary destination locations.
        - For missing speed, use a terminal speed reference of 0 m/s so the
          optimizer cost is consistent with the terminal-stop objective.
    """

    if len(destination_state) < 2:
        x_m, y_m = 0.0, 0.0
    else:
        x_m = float(destination_state[0])
        y_m = float(destination_state[1])

    has_explicit_terminal_state = len(destination_state) >= 4 and (
        abs(float(destination_state[2])) > 1e-9 or abs(float(destination_state[3])) > 1e-9
    )
    if has_explicit_terminal_state:
        return [x_m, y_m, float(destination_state[2]), float(destination_state[3])]

    _ = float(max_velocity_mps)  # kept in signature for backward compatibility
    v_ref = 0.0
    if ego_state is not None and len(ego_state) >= 4:
        ego_x_m = float(ego_state[0])
        ego_y_m = float(ego_state[1])
        dx_m = x_m - ego_x_m
        dy_m = y_m - ego_y_m
        if math.hypot(dx_m, dy_m) > 1e-9:
            psi_ref = math.atan2(dy_m, dx_m)
        else:
            psi_ref = float(ego_state[3])
    else:
        psi_ref = 0.0
    return [x_m, y_m, v_ref, psi_ref]


def run_simulation(config: Mapping[str, Any], scenario_handler: object, scenario_name: str = "scenario") -> None:
    """
    Intent:
        Run the pygame simulation loop with state tracking, obstacle prediction,
        and MPC-based trajectory generation.
    """

    config = _compose_runtime_config(config)

    sim_cfg = dict(config.get("simulation", {}))
    wnd_cfg = dict(config.get("window", {}))
    road_cfg = dict(config.get("road", {}))
    mpc_cfg = dict(config.get("mpc", {}))
    tracker_cfg = dict(config.get("tracker", {}))
    pid_controller_cfg = dict(config.get("pid_controller", {}))
    state_manager_cfg = dict(config.get("state_manager", {}))
    vehicle_manager_cfg = dict(config.get("vehicle_manager", {}))
    set_runtime_lookahead_fn = getattr(scenario_handler, "set_runtime_lookahead_waypoint_count", None)
    if callable(set_runtime_lookahead_fn):
        shared_local_goal_cfg = dict(mpc_cfg.get("local_goal", {}))
        set_runtime_lookahead_fn(shared_local_goal_cfg.get("lookahead_waypoint_count", None))

    # Enforce the PDF timing requirement: simulation dt = 0.05 s and MPC uses
    # the same discretization, while planning frequency is a separate variable.
    sim_dt_s = float(sim_cfg.get("dt_s", 0.05))
    mpc_cfg["plan_dt_s"] = float(sim_dt_s)
    mpc_cfg["horizon_s"] = float(mpc_cfg.get("horizon_s", 5.0))
    visualization_horizon_s = max(
        0.0,
        float(sim_cfg.get("visualization_horizon_s", mpc_cfg.get("visualization_horizon_s", 3.0))),
    )
    visualization_horizon_steps = max(1, int(round(visualization_horizon_s / max(1e-9, sim_dt_s)))) if visualization_horizon_s > 0.0 else 0

    destination_state = list(config.get("destination", [50.0, 0.0, 0.0, 0.0]))
    final_destination_state = list(destination_state)
    # Rolling/local destination is enabled again, but is updated only before
    # replanning (e.g., 2 Hz), not at every simulation step.

    pygame.init()
    pygame.display.set_caption(str(wnd_cfg.get("title", "MPC Custom")))
    screen = pygame.display.set_mode((int(wnd_cfg.get("width_px", 1500)), int(wnd_cfg.get("height_px", 900))))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    pixels_per_meter = float(wnd_cfg.get("pixels_per_meter", 14.0))

    show_world_scale = bool(sim_cfg.get("show_world_scale", True))
    world_scale_step_m = None
    if sim_cfg.get("world_scale_step_m", None) is not None:
        world_scale_step_m = max(0.0, float(sim_cfg.get("world_scale_step_m")))
    world_scale_min_tick_spacing_px = max(30, int(sim_cfg.get("world_scale_min_tick_spacing_px", 80)))
    world_scale_margin_px = max(4, int(sim_cfg.get("world_scale_margin_px", 14)))

    show_ego_trail = bool(sim_cfg.get("show_ego_trail", True))
    ego_trail_max_points = max(2, int(sim_cfg.get("ego_trail_max_points", 3000)))
    ego_trail_width_px = max(1, int(sim_cfg.get("ego_trail_width_px", 2)))
    ego_trail_color_rgb = tuple(sim_cfg.get("ego_trail_color_rgb", [255, 220, 40]))
    # Startup delay should be controllable from the scenario YAML. Keep support
    # for `mpc.delay` only as a fallback for older configs.
    scenario_start_delay_s = sim_cfg.get("start_delay_s", sim_cfg.get("start_delay", None))
    startup_delay_s = max(
        0.0,
        float(scenario_start_delay_s if scenario_start_delay_s is not None else mpc_cfg.get("delay", 0.0)),
    )
    target_fps = max(1, int(sim_cfg.get("target_fps", 60)))

    vehicles = build_vehicles_from_config(config=config, vehicle_manager_cfg=vehicle_manager_cfg)
    ego_vehicle = find_ego_vehicle(vehicles)
    vehicle_cfg_by_id = {str(item.get("vehicle_id")): dict(item) for item in list(config.get("vehicles", [])) if isinstance(item, Mapping)}
    mpc_cfg["wheelbase_m"] = float(ego_vehicle.wheelbase_m)
    mpc_cfg["ego_length_m"] = float(getattr(ego_vehicle.render_spec, "length_m", 4.5))
    mpc_cfg["ego_width_m"] = float(getattr(ego_vehicle.render_spec, "width_m", 2.0))
    mpc_planner = MPC(mpc_cfg=mpc_cfg, road_cfg=road_cfg)

    obstacle_tracker = Tracker(tracker_cfg=tracker_cfg.get("tracker", tracker_cfg))
    ego_trajectory_pid = TrajectoryPIDController(pid_cfg=pid_controller_cfg.get("pid_controller", pid_controller_cfg), dt_s=sim_dt_s)
    state_manager = StateManager(history_length=int(state_manager_cfg.get("history_length", 300)))

    planning_frequency_hz = float(mpc_planner.trajectory_generation_frequency_hz)
    planning_period_s = float(mpc_planner.trajectory_generation_period_s)
    tracker_horizon_s = float(mpc_planner.horizon_s)  # per user instruction: match horizon for future checking
    # Replan safety buffer: if the remaining planned points drop below this
    # threshold, trigger replan early so rolling temporary destination updates do
    # not lag and the ego does not run out of trajectory points.
    replan_buffer_steps = max(0, int(sim_cfg.get("replan_buffer_steps", mpc_cfg.get("replan_buffer_steps", 0))))

    simulation_time_s = 0.0
    next_replan_time_s = 0.0
    planned_trajectory_states: List[List[float]] = []
    planned_trajectory_is_optimized = False
    plan_cursor = 0
    obstacle_prediction_by_id: Dict[str, List[Dict[str, float]]] = {}
    lane_center_waypoints: List[Dict[str, object]] = []

    destination_reached_threshold_m = max(
        0.05,
        float(mpc_cfg.get("destination_reached_threshold_m", 0.5)),
    )
    base_mpc_max_velocity_mps = float(mpc_planner.constraints.max_velocity_mps)

    camera_center_world = (
        float(road_cfg.get("camera_center_x_m", 0.0)),
        float(road_cfg.get("center_y_m", 0.0)),
    )
    screen_center_px = (0.5 * float(screen.get_width()), 0.5 * float(screen.get_height()))

    road_total_width_m = float(road_cfg.get("lane_count", 1)) * float(road_cfg.get("lane_width_m", 4.0))
    plot_width_scale = max(0.2, float(sim_cfg.get("plot_width_scale", 0.6)))
    plot_height_scale = max(0.2, float(sim_cfg.get("plot_height_scale", 2.4)))
    plot_trajectory_height_scale = max(1.0, float(sim_cfg.get("plot_trajectory_height_scale", 1.0)))
    plotter = SimulationPlotter(
        output_dir=os.path.join(PROJECT_ROOT, "plot"),
        width_px=int(round(float(screen.get_width()) * plot_width_scale)),
        height_px=int(round(max(1.0, road_total_width_m) * float(pixels_per_meter) * plot_height_scale)),
        dpi=int(sim_cfg.get("plot_dpi", 100)),
        trajectory_height_scale=plot_trajectory_height_scale,
    )
    plot_cost_enabled = bool(mpc_cfg.get("plot_cost", True))
    plot_properties_enabled = bool(mpc_cfg.get("plot_properties", True))
    planned_profile_plot_cfg = dict(mpc_cfg.get("planned_profile_plot", mpc_cfg.get("planned_control_dump", {})))
    planned_profile_plot_enabled = bool(planned_profile_plot_cfg.get("enabled", True)) and bool(plot_properties_enabled)
    planned_profile_steps = max(1, int(planned_profile_plot_cfg.get("steps_to_plot", planned_profile_plot_cfg.get("steps_to_save", 5))))
    ego_log_time_s: List[float] = []
    ego_log_x_m: List[float] = []
    ego_log_y_m: List[float] = []
    ego_log_v_mps: List[float] = []
    ego_log_psi_rad: List[float] = []
    ego_log_a_mps2: List[float] = []
    ego_log_delta_rad: List[float] = []

    cost_log_time_s: List[float] = []
    cost_log_terms: Dict[str, List[float]] = {
        "Cost_ref": [],
        "Cost_LaneCenter": [],
        "Cost_Repulsive_Safe": [],
        "Cost_Repulsive_Collision": [],
        "Cost_Repulsive": [],
        "Cost_Control": [],
    }
    mpc_plan_log_x_m: List[float] = []
    mpc_planned_accel_by_step: List[List[float]] = [[] for _ in range(planned_profile_steps)]
    mpc_planned_steer_by_step: List[List[float]] = [[] for _ in range(planned_profile_steps)]
    mpc_planned_velocity_by_step: List[List[float]] = [[] for _ in range(planned_profile_steps)]
    mpc_planned_psi_by_step: List[List[float]] = [[] for _ in range(planned_profile_steps)]
    generated_plot_paths: List[str] = []
    seen_non_ego_vehicle_ids: set[str] = set()

    running = True
    try:
        if startup_delay_s > 0.0:
            start_wall_s = time.perf_counter()
            while running and (time.perf_counter() - start_wall_s) < startup_delay_s:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        running = False
                screen.fill(tuple(sim_cfg.get("background_color_rgb", [168, 192, 136])))
                draw_hud_text(screen, font, [f"Simulation starts in {max(0.0, startup_delay_s - (time.perf_counter() - start_wall_s)):.1f}s"], (16, 14))
                pygame.display.flip()
                clock.tick(target_fps)

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # Allow interactive update of the final destination.
                    dest_x, dest_y = screen_to_world(
                        x_px=float(event.pos[0]),
                        y_px=float(event.pos[1]),
                        camera_center_world=camera_center_world,
                        pixels_per_meter=pixels_per_meter,
                        screen_center_px=screen_center_px,
                    )
                    destination_state[0] = float(dest_x)
                    destination_state[1] = float(dest_y)
                    final_destination_state[0] = float(dest_x)
                    final_destination_state[1] = float(dest_y)
                    next_replan_time_s = simulation_time_s  # force immediate replan
                    planned_trajectory_states = []
                    plan_cursor = 0
                    ego_trajectory_pid.reset()

            # Snapshot current states before applying this step's ego/non-ego updates.
            ego_snapshot = ego_vehicle.to_snapshot()
            object_snapshots = [v.to_snapshot() for v in vehicles if str(v.vehicle_type).lower() != "ego"]

            # Scenario-specific camera behavior (optional hook).
            camera_center_fn = getattr(scenario_handler, "get_camera_center_world", None)
            if callable(camera_center_fn):
                camera_center_world = tuple(
                    camera_center_fn(
                        ego_snapshot=ego_snapshot,
                        current_camera_center_world=camera_center_world,
                    )
                )

            # Update tracker and predict future object trajectories (5 s horizon, dt=0.05 s).
            obstacle_tracker.update(obstacle_snapshots=object_snapshots, timestamp_s=simulation_time_s)
            obstacle_prediction_by_id = obstacle_tracker.predict(
                step_dt_s=float(mpc_planner.dt_s),
                horizon_s=float(tracker_horizon_s),
            )

            # Attach predicted trajectories to both vehicle objects and snapshots.
            object_future_by_id: Dict[str, List[List[float]]] = {}
            for object_id, predicted_samples in obstacle_prediction_by_id.items():
                future_states = []
                for sample in predicted_samples:
                    future_states.append([
                        float(sample.get("x", 0.0)),
                        float(sample.get("y", 0.0)),
                        float(sample.get("v", 0.0)),
                        float(sample.get("psi", 0.0)),
                    ])
                object_future_by_id[str(object_id)] = future_states

            for vehicle in vehicles:
                if str(vehicle.vehicle_type).lower() == "ego":
                    continue
                vehicle.set_future_trajectory(object_future_by_id.get(vehicle.vehicle_id, []))

            for snapshot in object_snapshots:
                object_id = str(snapshot.get("vehicle_id", ""))
                future_states = object_future_by_id.get(object_id, [])
                snapshot["future_trajectory"] = [list(state) for state in future_states]
                snapshot["predicted_trajectory"] = [list(state) for state in future_states]

            # Scenario-specific behavior planner hook (optional):
            # allow scenario to update final destination online.
            final_destination_fn = getattr(scenario_handler, "get_final_destination_state", None)
            if callable(final_destination_fn):
                updated_final_destination_state = list(
                    final_destination_fn(
                        ego_snapshot=ego_snapshot,
                        object_snapshots=object_snapshots,
                        current_final_destination_state=final_destination_state,
                        simulation_time_s=simulation_time_s,
                    )
                )
                if len(updated_final_destination_state) >= 2:
                    final_changed = (
                        math.hypot(
                            float(updated_final_destination_state[0]) - float(final_destination_state[0]),
                            float(updated_final_destination_state[1]) - float(final_destination_state[1]),
                        ) > 1e-6
                    )
                    final_destination_state = list(updated_final_destination_state)
                    if final_changed:
                        next_replan_time_s = simulation_time_s

            # Scenario-specific render override hook (optional):
            # allow scenario to set per-vehicle display colors dynamically
            # (e.g., traffic signal yellow->red at trigger time).
            vehicle_color_overrides_fn = getattr(scenario_handler, "get_vehicle_color_overrides", None)
            if callable(vehicle_color_overrides_fn):
                try:
                    color_overrides = dict(vehicle_color_overrides_fn(simulation_time_s=simulation_time_s))
                except TypeError:
                    color_overrides = dict(vehicle_color_overrides_fn())
                except Exception:
                    color_overrides = {}

                if len(color_overrides) > 0:
                    for vehicle in vehicles:
                        override_color = color_overrides.get(str(vehicle.vehicle_id))
                        if override_color is None or len(override_color) < 3:
                            continue
                        vehicle.render_spec.color_rgb = (
                            int(override_color[0]),
                            int(override_color[1]),
                            int(override_color[2]),
                        )

            # Destination check is always against the final destination.
            goal_dx = float(final_destination_state[0]) - float(ego_snapshot["x"])
            goal_dy = float(final_destination_state[1]) - float(ego_snapshot["y"])
            goal_distance_m = math.hypot(goal_dx, goal_dy)
            ego_reached_goal = goal_distance_m <= destination_reached_threshold_m

            remaining_plan_points = max(0, len(planned_trajectory_states) - int(plan_cursor))
            need_replan = (
                simulation_time_s + 1e-9 >= next_replan_time_s
                or len(planned_trajectory_states) == 0
                or plan_cursor >= len(planned_trajectory_states)
                or (replan_buffer_steps > 0 and remaining_plan_points <= replan_buffer_steps)
            )
            if need_replan:
                # Keep the planner's configured base speed bound intact here.
                # Any braking-distance cap for final-stop behavior is now applied
                # internally by MPC on a per-plan basis.
                mpc_planner.constraints.max_velocity_mps = float(base_mpc_max_velocity_mps)

                # Update rolling/local destination only at replan times.
                # Use the latest lane waypoints from the rendered road; if unavailable
                # (e.g., very first replan), try pulling from the scenario handler cache.
                local_waypoints = list(lane_center_waypoints)
                if len(local_waypoints) == 0 and hasattr(scenario_handler, "get_latest_lane_waypoints"):
                    local_waypoints = list(scenario_handler.get_latest_lane_waypoints())
                previous_destination_state = list(destination_state)

                if hasattr(scenario_handler, "get_step_destination_state"):
                    if len(local_waypoints) > 0:
                        destination_state = list(
                            scenario_handler.get_step_destination_state(
                                ego_snapshot=ego_snapshot,
                                lane_center_waypoints=local_waypoints,
                                obstacle_snapshots=object_snapshots,
                                final_destination_state=final_destination_state,
                                simulation_time_s=simulation_time_s,
                            )
                        )
                    else:
                        destination_state = list(final_destination_state)
                else:
                    destination_state = list(final_destination_state)

                mpc_destination_input = _destination_for_mpc(
                    destination_state=destination_state,
                    max_velocity_mps=float(mpc_planner.constraints.max_velocity_mps),
                    ego_state=list(ego_snapshot["current_state"]),
                )

                previous_plan = list(planned_trajectory_states)
                previous_plan_is_optimized = bool(planned_trajectory_is_optimized)
                previous_plan_cursor = int(plan_cursor)

                candidate_trajectory_states = mpc_planner.plan_trajectory(
                    current_state=list(ego_snapshot["current_state"]),
                    destination_state=mpc_destination_input,
                    object_snapshots=object_snapshots,
                    current_acceleration_mps2=float(ego_snapshot.get("acceleration_mps2", 0.0)),
                    current_steering_rad=float(ego_snapshot.get("steering_angle_rad", 0.0)),
                    lane_center_waypoints=local_waypoints,
                )
                solver_status = str(mpc_planner.get_runtime_status().get("solver_status", "")).lower()
                solved_ok = "solved" in solver_status

                if solved_ok:
                    planned_trajectory_states = list(candidate_trajectory_states)
                    planned_trajectory_is_optimized = True
                    plan_cursor = 0
                    ego_trajectory_pid.reset()
                else:
                    # When OSQP does not solve, keep previous trajectory only if
                    # its current start is still close to ego. Otherwise switch
                    # to MPC's deterministic fallback trajectory (built from
                    # current ego state), so trajectory does not visually freeze
                    # or disconnect from ego.
                    use_previous_plan = False
                    if previous_plan_cursor < len(previous_plan):
                        prev_state = previous_plan[previous_plan_cursor]
                        prev_start_gap_m = math.hypot(
                            float(prev_state[0]) - float(ego_snapshot["x"]),
                            float(prev_state[1]) - float(ego_snapshot["y"]),
                        )
                        use_previous_plan = prev_start_gap_m <= 2.0

                    if use_previous_plan and len(previous_plan) > 0 and len(destination_state) >= 2:
                        previous_plan_end = previous_plan[-1]
                        end_to_current_destination_gap_m = math.hypot(
                            float(previous_plan_end[0]) - float(destination_state[0]),
                            float(previous_plan_end[1]) - float(destination_state[1]),
                        )
                        # Reuse previous plan only when it still targets roughly
                        # the same destination. Otherwise, prefer the fresh
                        # candidate trajectory (QP solution or deterministic
                        # fallback) so trajectory endpoint does not detach from
                        # destination marker.
                        use_previous_plan = (
                            end_to_current_destination_gap_m
                            <= max(2.0, 2.0 * float(destination_reached_threshold_m))
                        )

                    if use_previous_plan:
                        planned_trajectory_states = previous_plan
                        planned_trajectory_is_optimized = bool(previous_plan_is_optimized)
                        plan_cursor = previous_plan_cursor
                        destination_state = previous_destination_state
                    else:
                        planned_trajectory_states = list(candidate_trajectory_states)
                        planned_trajectory_is_optimized = False
                        plan_cursor = 0
                        ego_trajectory_pid.reset()
                latest_cost_terms = mpc_planner.get_last_cost_terms()
                if len(latest_cost_terms) > 0:
                    cost_log_time_s.append(float(simulation_time_s))
                    for cost_key in cost_log_terms.keys():
                        cost_log_terms[cost_key].append(float(latest_cost_terms.get(cost_key, 0.0)))
                if planned_profile_plot_enabled:
                    planned_controls = mpc_planner.get_last_control_sequence(max_steps=planned_profile_steps)
                    mpc_plan_log_x_m.append(float(ego_snapshot["x"]))
                    for step_idx in range(planned_profile_steps):
                        accel_value = float("nan")
                        steer_value = float("nan")
                        velocity_value = float("nan")
                        psi_value = float("nan")
                        if step_idx < len(planned_controls):
                            accel_value = float(planned_controls[step_idx].get("acceleration_mps2", 0.0))
                            steer_value = float(planned_controls[step_idx].get("steering_angle_rad", 0.0))
                        if step_idx < len(candidate_trajectory_states) and len(candidate_trajectory_states[step_idx]) >= 4:
                            velocity_value = float(candidate_trajectory_states[step_idx][2])
                            psi_value = float(candidate_trajectory_states[step_idx][3])
                        mpc_planned_accel_by_step[step_idx].append(accel_value)
                        mpc_planned_steer_by_step[step_idx].append(steer_value)
                        mpc_planned_velocity_by_step[step_idx].append(velocity_value)
                        mpc_planned_psi_by_step[step_idx].append(psi_value)

                next_replan_time_s = simulation_time_s + planning_period_s

            # Track the planned trajectory with PID and move the ego through
            # the vehicle kinematic model for one simulation step.
            plan_cursor = _track_ego_with_pid(
                ego_vehicle=ego_vehicle,
                trajectory_pid_controller=ego_trajectory_pid,
                planned_states=planned_trajectory_states,
                plan_cursor=plan_cursor,
                sim_dt_s=float(sim_dt_s),
            )
            ego_vehicle.set_future_trajectory(planned_trajectory_states[plan_cursor:])

            # Advance non-ego objects with simple scenario-defined motion modes.
            non_ego_control_override_fn = getattr(scenario_handler, "get_non_ego_control_override", None)
            destroyed_non_ego_vehicle_ids: set[str] = set()
            screen_size_px = (float(screen.get_width()), float(screen.get_height()))
            for vehicle in vehicles:
                if str(vehicle.vehicle_type).lower() == "ego":
                    continue
                vehicle_cfg = vehicle_cfg_by_id.get(vehicle.vehicle_id, {})
                vehicle_snapshot = vehicle.to_snapshot()
                vehicle_id = str(vehicle.vehicle_id)
                if _is_inside_render_window(
                    object_snapshot=vehicle_snapshot,
                    camera_center_world=camera_center_world,
                    pixels_per_meter=float(pixels_per_meter),
                    screen_size_px=screen_size_px,
                ):
                    seen_non_ego_vehicle_ids.add(vehicle_id)
                elif vehicle_id in seen_non_ego_vehicle_ids:
                    destroyed_non_ego_vehicle_ids.add(vehicle_id)
                    continue

                control_override = None
                if callable(non_ego_control_override_fn):
                    control_override = non_ego_control_override_fn(
                        vehicle_snapshot=vehicle_snapshot,
                        vehicle_cfg=vehicle_cfg,
                        simulation_time_s=simulation_time_s,
                        camera_center_world=camera_center_world,
                        pixels_per_meter=float(pixels_per_meter),
                        screen_size_px=screen_size_px,
                    )
                    if control_override is not None:
                        control_override = dict(control_override)

                if control_override is not None and bool(control_override.get("destroy", False)):
                    destroyed_non_ego_vehicle_ids.add(str(vehicle.vehicle_id))
                    continue

                if control_override is not None and bool(control_override.get("freeze", False)):
                    vehicle.current_state[2] = 0.0
                    vehicle.set_control(0.0, 0.0)
                    vehicle.step(sim_dt_s)
                    continue

                accel_cmd = None if control_override is None else control_override.get("acceleration_mps2")
                steer_cmd = None if control_override is None else control_override.get("steering_angle_rad")
                if accel_cmd is None or steer_cmd is None:
                    default_accel_cmd, default_steer_cmd = compute_non_ego_control(
                        vehicle=vehicle,
                        vehicle_cfg=vehicle_cfg,
                        defaults_cfg={},
                        lane_center_waypoints=lane_center_waypoints,
                    )
                    if accel_cmd is None:
                        accel_cmd = default_accel_cmd
                    if steer_cmd is None:
                        steer_cmd = default_steer_cmd
                vehicle.set_control(accel_cmd, steer_cmd)
                vehicle.step(sim_dt_s)

            if len(destroyed_non_ego_vehicle_ids) > 0:
                seen_non_ego_vehicle_ids.difference_update(destroyed_non_ego_vehicle_ids)
                vehicles = [
                    vehicle
                    for vehicle in vehicles
                    if str(vehicle.vehicle_id) not in destroyed_non_ego_vehicle_ids
                ]

            simulation_time_s += sim_dt_s
            state_manager.refresh(vehicles=vehicles, timestamp_s=simulation_time_s)

            ego_log_time_s.append(float(simulation_time_s))
            ego_log_x_m.append(float(ego_vehicle.current_state[0]))
            ego_log_y_m.append(float(ego_vehicle.current_state[1]))
            ego_log_v_mps.append(float(ego_vehicle.current_state[2]))
            ego_log_psi_rad.append(float(ego_vehicle.current_state[3]))
            ego_log_a_mps2.append(float(ego_vehicle.acceleration_mps2))
            ego_log_delta_rad.append(float(ego_vehicle.steering_angle_rad))

            # --- Rendering ---
            screen.fill(tuple(sim_cfg.get("background_color_rgb", [168, 192, 136])))
            scenario_handler.draw_road(
                surface=screen,
                road_cfg=road_cfg,
                camera_center_world=camera_center_world,
                pixels_per_meter=pixels_per_meter,
                world_to_screen_fn=None,
            )
            lane_center_waypoints = list(scenario_handler.get_latest_lane_waypoints())

            draw_destination(
                surface=screen,
                destination_state=final_destination_state,
                camera_center_world=camera_center_world,
                pixels_per_meter=pixels_per_meter,
                fill_color_rgb=(220, 35, 35),
            )

            # Draw temporary/rolling destination with a distinct color so it is
            # visible separately from the final (red) destination marker.
            if len(destination_state) >= 2:
                temp_goal_dist_to_final_m = math.hypot(
                    float(destination_state[0]) - float(final_destination_state[0]),
                    float(destination_state[1]) - float(final_destination_state[1]),
                )
                if temp_goal_dist_to_final_m > 1e-3:
                    draw_destination(
                        surface=screen,
                        destination_state=destination_state,
                        camera_center_world=camera_center_world,
                        pixels_per_meter=pixels_per_meter,
                        fill_color_rgb=(30, 220, 250),
                        outline_color_rgb=(20, 20, 20),
                        radius_px=6,
                    )

            visible_prediction_by_object_id = {}
            if visualization_horizon_steps > 0:
                for object_id, predicted_samples in obstacle_prediction_by_id.items():
                    visible_prediction_by_object_id[str(object_id)] = list(predicted_samples[:visualization_horizon_steps])
            else:
                visible_prediction_by_object_id = {str(object_id): list(predicted_samples) for object_id, predicted_samples in obstacle_prediction_by_id.items()}

            draw_predicted_object_trajectories(
                surface=screen,
                prediction_by_object_id=visible_prediction_by_object_id,
                camera_center_world=camera_center_world,
                pixels_per_meter=pixels_per_meter,
                color_rgb=(220, 45, 45),
            )

            visible_ego_trajectory_states: List[List[float]] = []
            if bool(planned_trajectory_is_optimized) and plan_cursor < len(planned_trajectory_states):
                visible_ego_trajectory_states = planned_trajectory_states[plan_cursor:]
            if visualization_horizon_steps > 0:
                visible_ego_trajectory_states = visible_ego_trajectory_states[:visualization_horizon_steps]

            # Show remaining ego plan (from current cursor onward) as dotted line.
            draw_dotted_trajectory(
                surface=screen,
                trajectory_states=visible_ego_trajectory_states,
                camera_center_world=camera_center_world,
                pixels_per_meter=pixels_per_meter,
                color_rgb=(35, 210, 70),
                dot_spacing_px=12,
                dot_radius_px=3,
            )

            if show_ego_trail and len(ego_log_x_m) >= 2:
                start_idx = max(0, len(ego_log_x_m) - int(ego_trail_max_points))
                trail_points_px = [
                    world_to_screen(
                        x_m=float(x_m),
                        y_m=float(y_m),
                        camera_center_world=camera_center_world,
                        pixels_per_meter=pixels_per_meter,
                        screen_center_px=screen_center_px,
                    )
                    for x_m, y_m in zip(ego_log_x_m[start_idx:], ego_log_y_m[start_idx:])
                ]
                if len(trail_points_px) >= 2:
                    pygame.draw.lines(
                        screen,
                        tuple(int(v) for v in ego_trail_color_rgb),
                        False,
                        trail_points_px,
                        int(ego_trail_width_px),
                    )

            for vehicle in vehicles:
                vehicle.draw(
                    surface=screen,
                    pixels_per_meter=pixels_per_meter,
                    camera_center_world=camera_center_world,
                    screen_center_px=screen_center_px,
                )

            if show_world_scale:
                draw_world_scale(
                    surface=screen,
                    font=font,
                    camera_center_world=camera_center_world,
                    pixels_per_meter=pixels_per_meter,
                    step_m=world_scale_step_m,
                    min_tick_spacing_px=world_scale_min_tick_spacing_px,
                    margin_px=world_scale_margin_px,
                )

            ego_state = list(ego_vehicle.current_state)
            runtime_status = mpc_planner.get_runtime_status()
            hud_lines = [
                "ESC quit | LMB set destination (scenario1 only)",
                f"time={simulation_time_s:6.2f}s  sim_dt={sim_dt_s:.2f}s  replan={planning_frequency_hz:.2f} Hz ({planning_period_s:.2f}s)",
                f"ego state [x,y,v,psi]=[{ego_state[0]:.2f}, {ego_state[1]:.2f}, {ego_state[2]:.2f}, {ego_state[3]:.3f}]",
                f"destination_final=[{float(final_destination_state[0]):.2f}, {float(final_destination_state[1]):.2f}]  destination_temp=[{float(destination_state[0]):.2f}, {float(destination_state[1]):.2f}]",
                f"goal_dist={goal_distance_m:.2f}m  reached={ego_reached_goal}",
                f"planner status={runtime_status['solver_status']}  solve={float(runtime_status['solve_time_ms']):.1f}ms  horizon={runtime_status['horizon_s']:.1f}s",
                f"trajectory points total={len(planned_trajectory_states)} remaining={max(0, len(planned_trajectory_states)-plan_cursor)}",
                f"objects={len(object_snapshots)}  predicted_paths={len(obstacle_prediction_by_id)}  lane_waypoints={len(lane_center_waypoints)}",
            ]
            draw_hud_text(screen, font, hud_lines, (16, 14))

            pygame.display.flip()
            clock.tick(target_fps)
    finally:
        try:
            generated_plot_paths.extend(
                plotter.save_x_coordinate_plots(
                    scenario_name=str(scenario_name),
                    ego_time_s=ego_log_time_s,
                    x_m=ego_log_x_m,
                    y_m=ego_log_y_m,
                    velocity_mps=ego_log_v_mps,
                    accel_mps2=ego_log_a_mps2,
                    steer_rad=ego_log_delta_rad,
                    cost_time_s=cost_log_time_s,
                    cost_terms=cost_log_terms,
                    include_cost=plot_cost_enabled,
                    include_properties=plot_properties_enabled,
                )
            )
            if plot_properties_enabled:
                generated_plot_paths.extend(
                    plotter.save_mpc_plan_step_plots(
                        scenario_name=str(scenario_name),
                        replan_x_m=mpc_plan_log_x_m,
                        accel_by_step=mpc_planned_accel_by_step,
                        steer_by_step=mpc_planned_steer_by_step,
                        velocity_by_step=mpc_planned_velocity_by_step,
                        psi_by_step=mpc_planned_psi_by_step,
                    )
                )

            if len(generated_plot_paths) > 0:
                print("[PLOT] Saved plots:")
                for plot_path in generated_plot_paths:
                    print(f"[PLOT] {plot_path}")
            else:
                print("[PLOT] No plot data available to save.")
        except Exception as plot_exc:
            print(f"[WARN] Failed to save plots: {plot_exc}")
        pygame.quit()


def main() -> None:
    """
    Intent:
        Parse scenario name, load the scenario, and run the simulation.
    """

    scenario_name = "scenario1"
    if len(sys.argv) > 1:
        scenario_name = str(sys.argv[1]).strip() or scenario_name

    scenario_handler, scenario_config = load_scenario_by_name(scenario_name=scenario_name)
    run_simulation(config=scenario_config, scenario_handler=scenario_handler, scenario_name=scenario_name)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

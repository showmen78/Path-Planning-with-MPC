# MPC Planning Stack

This branch is a planning-only version of the project for CARLA integration work.

The old `pygame` simulator, scenario system, road rendering, plotting, and
vehicle-simulation runtime have been removed. What remains is the core planning
stack:

- LTV-MPC in [MPC/mpc.py](/home/umd-user/Desktop/Personal%20Docs/Research%20Topics/cp-x/MPC_custom%20%28potential%29%20%28exp%20potential%29%20/MPC/mpc.py)
- the behavior-planner package in [behavior_planner](/home/umd-user/Desktop/Personal%20Docs/Research%20Topics/cp-x/MPC_custom%20%28potential%29%20%28exp%20potential%29%20/behavior_planner)
- the motion tracker in [utility/tracker.py](/home/umd-user/Desktop/Personal%20Docs/Research%20Topics/cp-x/MPC_custom%20%28potential%29%20%28exp%20potential%29%20/utility/tracker.py)
- lightweight config helpers in [utility/config_loader.py](/home/umd-user/Desktop/Personal%20Docs/Research%20Topics/cp-x/MPC_custom%20%28potential%29%20%28exp%20potential%29%20/utility/config_loader.py)

## What Was Removed

- `pygame` windowing and HUD rendering
- scenario loading and scenario YAMLs
- simulated vehicle spawning and non-ego motion modes
- road drawing and waypoint generation for the old local simulator
- PID tracking layer used only for the old simulator execution loop
- plot generation and standalone visual demo scripts

## What Remains

## MPC

The planner state is:

$$
X_k = [x_k,\ y_k,\ v_k,\ \psi_k]
$$

The control input is:

$$
U_k = [a_k,\ \delta_k]
$$

The live MPC still uses:

- kinematic bicycle dynamics
- LTV linearization around a reference rollout
- OSQP for the QP solve
- the current super-ellipsoid repulsive potential

The main configuration is in [MPC/mpc.yaml](/home/umd-user/Desktop/Personal%20Docs/Research%20Topics/cp-x/MPC_custom%20%28potential%29%20%28exp%20potential%29%20/MPC/mpc.yaml).

Important active config areas:

- `mpc.behavior_planner_runtime.*`
- `mpc.local_goal.*`
- `mpc.final_stop_speed_cap.*`
- `mpc.reference_rollout.*`
- `mpc.cost.*`
- `mpc.constraints.*`
- `mpc.solver.*`

## Behavior Planner

The behavior planner is kept in [behavior_planner](/home/umd-user/Desktop/Personal%20Docs/Research%20Topics/cp-x/MPC_custom%20%28potential%29%20%28exp%20potential%29%20/behavior_planner).

Main files:

- [behavior_planner/prompt_builder.py](/home/umd-user/Desktop/Personal%20Docs/Research%20Topics/cp-x/MPC_custom%20%28potential%29%20%28exp%20potential%29%20/behavior_planner/prompt_builder.py)
  Builds the compact prompt from ego state, route safety, previous decision, and surrounding vehicles.
- [behavior_planner/global_planner.py](/home/umd-user/Desktop/Personal%20Docs/Research%20Topics/cp-x/MPC_custom%20%28potential%29%20%28exp%20potential%29%20/behavior_planner/global_planner.py)
  A* route support and lane-context inference over lane-center waypoints.
- [behavior_planner/intention.py](/home/umd-user/Desktop/Personal%20Docs/Research%20Topics/cp-x/MPC_custom%20%28potential%29%20%28exp%20potential%29%20/behavior_planner/intention.py)
  Surrounding-vehicle intention inference.
- [behavior_planner/decision_logic.py](/home/umd-user/Desktop/Personal%20Docs/Research%20Topics/cp-x/MPC_custom%20%28potential%29%20%28exp%20potential%29%20/behavior_planner/decision_logic.py)
  Runtime behavior execution logic that converts the LLM output into temporary-destination and speed overrides.
- [behavior_planner/api_client.py](/home/umd-user/Desktop/Personal%20Docs/Research%20Topics/cp-x/MPC_custom%20%28potential%29%20%28exp%20potential%29%20/behavior_planner/api_client.py)
  OpenAI Responses API client.
- [behavior_planner/system_instruction.txt](/home/umd-user/Desktop/Personal%20Docs/Research%20Topics/cp-x/MPC_custom%20%28potential%29%20%28exp%20potential%29%20/behavior_planner/system_instruction.txt)
  System instruction sent once to the model.

Current prompt shape:

```text
ID:[prompt_id]

Ego01:[x,y,v,psi,Llane]

ROUTE:[lane_safe_l,lane_safe_c,lane_safe_r]

PREV:[behavior]

V[id]:[x,y,v,psi,Llane,I]
```

Current response shape:

```json
{"id":"PROMPT_ID","behavior":"BEHAVIOR_NAME"}
```

## Rolling Temporary Destination

The temporary destination is still distance-based and lane-aware:

- distance comes from the dynamic lookahead logic in `mpc.local_goal.*`
- lane comes from the behavior-planner decision logic

Dynamic lookahead uses:

$$
L_{raw} = d_{min} + k_v V - k_c |\kappa|
$$

$$
L_d = clamp(L_{raw}, d_{min}, d_{max})
$$

with:

$$
\kappa = \frac{4A}{abc}
$$

The hard minimum distance is controlled by:

- `mpc.local_goal.dynamic_lookahead_min_distance_m`

This minimum is also used when the behavior planner rebuilds the temporary
destination, so the target does not collapse onto the ego vehicle at very low
speed.

## Repulsive Potential

The live obstacle cost is the super-ellipsoid safe/collision formulation in
[MPC/mpc.py](/home/umd-user/Desktop/Personal%20Docs/Research%20Topics/cp-x/MPC_custom%20%28potential%29%20%28exp%20potential%29%20/MPC/mpc.py).

Important current features:

- safe zone and collision zone
- dynamic inflation from relative motion
- longitudinal field cap:
  - `max_longitudinal_zone_length_m`
- lateral field cap to lane width:
  - `limit_lateral_zone_to_lane_width`
  - `max_lateral_zone_lane_fraction`
- local Taylor approximation for insertion into the QP

## Entry Points

[main.py](/home/umd-user/Desktop/Personal%20Docs/Research%20Topics/cp-x/MPC_custom%20%28potential%29%20%28exp%20potential%29%20/main.py) is now a planning-only helper entrypoint. It provides:

- `load_mpc_config()`
- `load_tracker_config()`
- `build_mpc_planner()`
- `build_tracker()`
- `build_behavior_planner_prompt_builder()`
- `build_behavior_planner_api_client()`

It no longer runs a simulator.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

or:

```bash
conda create -n mpc_custom python=3.10 -y
conda activate mpc_custom
pip install -r requirements.txt
```

If you want to use the behavior planner with the OpenAI API, create [/.env](/home/umd-user/Desktop/Personal%20Docs/Research%20Topics/cp-x/MPC_custom%20%28potential%29%20%28exp%20potential%29%20/.env):

```env
OPENAI_API_KEY=your_api_key_here
```

## Lightweight API Tests

- [test.py](/home/umd-user/Desktop/Personal%20Docs/Research%20Topics/cp-x/MPC_custom%20%28potential%29%20%28exp%20potential%29%20/test.py)
  sequential prompt/response latency loop
- [behavior_planner/check_latency.py](/home/umd-user/Desktop/Personal%20Docs/Research%20Topics/cp-x/MPC_custom%20%28potential%29%20%28exp%20potential%29%20/behavior_planner/check_latency.py)
  single-request latency check

## Dependencies

Current runtime dependencies in [requirements.txt](/home/umd-user/Desktop/Personal%20Docs/Research%20Topics/cp-x/MPC_custom%20%28potential%29%20%28exp%20potential%29%20/requirements.txt):

- `numpy`
- `scipy`
- `osqp`
- `pyyaml`
- `openai`

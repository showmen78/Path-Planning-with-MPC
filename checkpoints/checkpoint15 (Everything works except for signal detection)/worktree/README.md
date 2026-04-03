# MPC Planning Stack

This repo contains the CARLA planning stack built around:

- [main.py](/home/umd-user/Desktop/Personal%20Docs/Research%20Topics/cp-x/MPC_custom%20%28potential%29%20%28exp%20potential%29%20/main.py)
- [carla_scenario/runner.py](/home/umd-user/Desktop/Personal%20Docs/Research%20Topics/cp-x/MPC_custom%20%28potential%29%20%28exp%20potential%29%20/carla_scenario/runner.py)
- [MPC/mpc.py](/home/umd-user/Desktop/Personal%20Docs/Research%20Topics/cp-x/MPC_custom%20%28potential%29%20%28exp%20potential%29%20/MPC/mpc.py)
- [behavior_planner](/home/umd-user/Desktop/Personal%20Docs/Research%20Topics/cp-x/MPC_custom%20%28potential%29%20%28exp%20potential%29%20/behavior_planner)
- [utility/global_planner.py](/home/umd-user/Desktop/Personal%20Docs/Research%20Topics/cp-x/MPC_custom%20%28potential%29%20%28exp%20potential%29%20/utility/global_planner.py)

## Main Components

### MPC

The optimizer is implemented in [MPC/mpc.py](/home/umd-user/Desktop/Personal%20Docs/Research%20Topics/cp-x/MPC_custom%20%28potential%29%20%28exp%20potential%29%20/MPC/mpc.py). Runtime tuning lives in [MPC/mpc.yaml](/home/umd-user/Desktop/Personal%20Docs/Research%20Topics/cp-x/MPC_custom%20%28potential%29%20%28exp%20potential%29%20/MPC/mpc.yaml).

### Rule-Based Behavior Planner

The active behavior-planner path is fully rule-based:

- [behavior_planner/planner.py](/home/umd-user/Desktop/Personal%20Docs/Research%20Topics/cp-x/MPC_custom%20%28potential%29%20%28exp%20potential%29%20/behavior_planner/planner.py)
- [behavior_planner/temp_destination.py](/home/umd-user/Desktop/Personal%20Docs/Research%20Topics/cp-x/MPC_custom%20%28potential%29%20%28exp%20potential%29%20/behavior_planner/temp_destination.py)
- [behavior_planner/lane_safety.py](/home/umd-user/Desktop/Personal%20Docs/Research%20Topics/cp-x/MPC_custom%20%28potential%29%20%28exp%20potential%29%20/behavior_planner/lane_safety.py)
- [behavior_planner/traffic_light_stop.py](/home/umd-user/Desktop/Personal%20Docs/Research%20Topics/cp-x/MPC_custom%20%28potential%29%20%28exp%20potential%29%20/behavior_planner/traffic_light_stop.py)
- [behavior_planner/reroute.py](/home/umd-user/Desktop/Personal%20Docs/Research%20Topics/cp-x/MPC_custom%20%28potential%29%20%28exp%20potential%29%20/behavior_planner/reroute.py)

The behavior planner decides lane follow, lane change, stop, and reroute. The blue dot is produced from the route look-ahead plus the planner-selected lane.

### Global Route and Lane Model

Route generation and lane normalization live in:

- [utility/global_planner.py](/home/umd-user/Desktop/Personal%20Docs/Research%20Topics/cp-x/MPC_custom%20%28potential%29%20%28exp%20potential%29%20/utility/global_planner.py)
- [utility/carla_lane_graph.py](/home/umd-user/Desktop/Personal%20Docs/Research%20Topics/cp-x/MPC_custom%20%28potential%29%20%28exp%20potential%29%20/utility/carla_lane_graph.py)

## Running

Start a configured CARLA scenario with:

```bash
python main.py <scenario_name>
```

Available scenarios are discovered from [carla_scenario](/home/umd-user/Desktop/Personal%20Docs/Research%20Topics/cp-x/MPC_custom%20%28potential%29%20%28exp%20potential%29%20/carla_scenario).

## Dependencies

Install the runtime dependencies from [requirements.txt](/home/umd-user/Desktop/Personal%20Docs/Research%20Topics/cp-x/MPC_custom%20%28potential%29%20%28exp%20potential%29%20/requirements.txt):

```bash
pip install -r requirements.txt
```

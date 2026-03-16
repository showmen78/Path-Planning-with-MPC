# MPC for Path Planning

This project simulates an ego vehicle that plans a future trajectory with linear time-varying MPC and then tracks that planned trajectory with a PID controller in `pygame`.

The runtime path is:

1. Load a scenario YAML.
2. Merge it with subsystem defaults from `MPC/`, `road/`, `utility/`, `state_manager/`, and `vehicle_manager/`.
3. Build the road and vehicle objects.
4. Track and predict non-ego objects.
5. Build a rolling destination state for the ego.
6. Run MPC to plan a future state trajectory `[x, y, v, psi]`.
7. Track that planned trajectory with PID.
8. Render the scene and save plots at the end.

`super_ellipsoid.py` is kept as a reference/analysis script. The live obstacle potential used by MPC is implemented inside `MPC/mpc.py`, but it follows the same super-ellipsoid idea.

## Major Updates In The Current Project

- Added a new `behavior_planner/` package for high-level LLM-based lane-choice decisions.
- Added prompt generation from live scenario data:
  - prompt id
  - ego state
  - route safety flags
  - previous decision
  - surrounding-vehicle summaries with predicted intentions
- Added OpenAI API integration for the behavior planner:
  - separate behavior-planner loop in `main.py`
  - configurable frequency in `MPC/mpc.yaml`
  - `.env` / `OPENAI_API_KEY` support
  - request/response id matching so only the correct response is applied
- Added hardcoded runtime behavior-execution logic in code, independent of the text rules file.
- Added dynamic lookahead distance for the rolling temporary destination:
  - curvature-aware lookahead formula
  - hard minimum distance so the temporary target stays ahead of the ego
  - optional extra behavior-planner destination buffer
- Updated temporary-destination behavior so:
  - distance comes from the rolling-goal lookahead logic
  - lane is selected by the behavior planner
  - repeated same-direction lane-change outputs continue the same maneuver instead of requesting another extra lane jump
- Removed the old scenario-local temporary-destination lane-switch logic. Lane selection is now shared behavior-planner logic rather than per-scenario hardcoded overrides.
- Replaced the older obstacle-cost description with the current live super-ellipsoid exponential safe/collision potential used by MPC.
- Added field-size limits for the repulsive potential:
  - longitudinal cap with `max_longitudinal_zone_length_m`
  - lateral cap with `limit_lateral_zone_to_lane_width` and `max_lateral_zone_lane_fraction`
- Added potential-field visualization support in the `pygame` window.
- Added behavior-planner / route / decision debug information to the HUD.
- Added helper scripts for prompt/API latency checks and potential-field cost visualization.

## Installation

From the project root, create/activate a Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you use Conda instead:

```bash
conda create -n mpc_custom python=3.10 -y
conda activate mpc_custom
pip install -r requirements.txt
```

If you want the behavior planner to call the OpenAI API, add a project-local `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
```

`.env` is ignored by git.

## Running The Project

Run a scenario from the project root:

```bash
python main.py scenario4
python main.py VRU
python main.py red_light_violation_warning
python main.py red_light_violation_warning_2
python main.py workzone
python main.py "workzone with bp"
```

If you do not want the LLM loop during a run, set:

```yaml
mpc:
  behavior_planner_runtime:
    enabled: false
```

General form:

```bash
python main.py <scenario_name>
```

Available runtime scenarios in this workspace:

- `scenario4`
- `VRU`
- `red_light_violation_warning`
- `red_light_violation_warning_2`
- `workzone`
- `workzone with bp`

## Project Structure

- `main.py`
  Main simulation loop, config merge, planning cadence, PID tracking, rendering, and plot saving.
- `MPC/mpc.py`
  Live LTV-MPC implementation and obstacle potential cost.
- `MPC/mpc.yaml`
  Default MPC, rolling-goal, behavior-planner, and solver configuration.
- `behavior_planner/`
  Prompt creation, A* route support, intention inference, OpenAI API client, and behavior-execution logic.
- `behavior_planner/system_instruction.txt`
  The system instruction sent once to the LLM before prompt updates begin.
- `behavior_planner/behavior_planner_rules.txt`
  Human-readable rule document; the runtime logic is implemented directly in code.
- `utility/pid_controller.py`
  PID tracker that follows the MPC state trajectory.
- `utility/tracker.py`
  Polynomial history fitting and object prediction.
- `road/road_model.py`
  Straight/curved road generation and lane-center waypoint generation.
- `vehicle_manager/vehicle.py`
  Kinematic vehicle model used by ego and non-ego objects.
- `scenarios/`
  Scenario-specific road, destination, and object configuration.
- `plot/plotter.py`
  End-of-run plot generation.
- `super_ellipsoid.py`
  Standalone analysis/demo script for the super-ellipsoid safety field.
- `test.py`
  Standalone API test loop for repeated prompt/response latency checks.

## Kinematic Model

The planner state is:

$$
X_k = [x_k,\ y_k,\ v_k,\ \psi_k]
$$

The planner input is:

$$
U_k = [a_k,\ \delta_k]
$$

where:

- $x, y$: ego position in world coordinates
- $v$: longitudinal speed
- $\psi$: yaw / heading
- $a$: longitudinal acceleration
- $\delta$: steering angle

The nonlinear reference model is a kinematic bicycle model:

$$
\beta_k = \arctan\left(\frac{l_r}{L}\tan(\delta_k)\right)
$$

$$
x_{k+1} = x_k + \Delta t \cdot v_k \cos(\psi_k + \beta_k)
$$

$$
y_{k+1} = y_k + \Delta t \cdot v_k \sin(\psi_k + \beta_k)
$$

$$
v_{k+1} = v_k + \Delta t \cdot a_k
$$

$$
\psi_{k+1} = \psi_k + \Delta t \cdot \frac{v_k}{l_r}\sin(\beta_k)
$$

In the current project:

- `L = wheelbase_m`
- `l_r = wheelbase_m / 2`

So the CG is assumed centered between the axles.

The same basic kinematic model is used in simulation for vehicle motion in `vehicle_manager/vehicle.py`.

## Why This Is LTV-MPC

The project does not solve the full nonlinear MPC problem directly. Instead:

1. Build a guessed future trajectory with the nonlinear bicycle model.
2. Linearize the model around that guessed trajectory.
3. Approximate the obstacle cost locally around that guessed trajectory.
4. Solve the resulting convex QP with OSQP.

That is why the controller is linear time-varying MPC: the matrices change with stage $k$ because they are built around a nominal trajectory.

## Reference Rollout

The reference rollout is the nominal guessed trajectory used to linearize the nonlinear model and obstacle cost. It is not the final optimized trajectory.

At each stage, the rollout:

1. Chooses a target point.
   - Usually a lane-center waypoint on the destination lane.
   - Otherwise the active destination point directly.
2. Computes line-of-sight heading:

$$
\psi^{los}_k = atan2(y^{target}_k - y_k,\ x^{target}_k - x_k)
$$

3. Blends path heading and line-of-sight heading:

$$
\psi^{des}_k =
atan2(
(1-w)\sin(\psi^{path}_k) + w\sin(\psi^{los}_k),\ 
(1-w)\cos(\psi^{path}_k) + w\cos(\psi^{los}_k)
)
$$

4. Converts heading error into guessed steering:

$$
e^\psi_k = wrap(\psi^{des}_k - \psi_k)
$$

$$
\delta^{des}_k = clamp(k_h e^\psi_k,\ \delta_{min},\ \delta_{max})
$$

5. Chooses a guessed speed target.
   - Starts from the destination speed reference.
   - May be reduced by final-stop logic.
   - May be reduced by a lead-obstacle speed heuristic.
6. Converts speed error into guessed acceleration:

$$
a^{des}_k = clamp(k_v (v^{target}_k - v_k),\ a_{min},\ a_{max})
$$

7. Propagates the nonlinear bicycle model one step.

The previous solved MPC trajectory can also be reused as the next rollout seed if the current ego state is still close to it.

## Lane-Center Reference

Lane-center waypoints are generated by the road model and stored as centerline points for each lane.

MPC converts those waypoints into a stage-wise lane reference:

$$
r^{lane}_k = [x^{lane}_k,\ y^{lane}_k,\ \psi^{lane}_k]
$$

This lane-center reference is used for:

- rollout guidance
- lane-center-follow cost

The scenario-level `lookahead_waypoint_count` does not build the full lane-center reference by itself. It only decides where the temporary destination is placed ahead of the ego when distance-based local-goal selection is not available. MPC then uses the lane waypoints around that destination lane to build its own per-stage reference.

## Rolling Destination And Behavior Planner

The active temporary destination used by MPC is now built in two layers:

1. The rolling-goal layer selects how far ahead the target should be.
2. The behavior planner selects which lane that temporary destination should lie on.

### Dynamic lookahead

The current distance-based lookahead uses:

$$
L_{raw} = d_{min} + k_v V - k_c |\kappa|
$$

$$
L_d = clamp(L_{raw}, d_{min}, d_{max})
$$

with curvature estimated from three forward samples:

$$
\kappa = \frac{4A}{abc}
$$

The minimum lookahead distance is also used as a hard lower bound when the
behavior planner rebuilds the temporary destination, so the temporary target
does not collapse onto the ego at very low speed.

### Behavior-planner lane selection

When the behavior planner is enabled:

- the distance to the temporary destination comes from the rolling-goal layer
- the lane of the temporary destination comes from the latest accepted behavior-planner decision
- repeated `LANE_CHANGE_LEFT` or `LANE_CHANGE_RIGHT` outputs continue the same maneuver rather than requesting another additional lane jump

The current prompt sent to the LLM is compact and looks like:

```text
ID:[prompt_id]

Ego01:[x,y,v,psi,Llane]

ROUTE:[lane_safe_l,lane_safe_c,lane_safe_r]

PREV:[behavior]

V[id]:[x,y,v,psi,Llane,I]
```

The response format is:

```json
{"id":"PROMPT_ID","behavior":"BEHAVIOR_NAME"}
```

At runtime, `main.py`:

- sends the system instruction once before the scenario starts
- runs the behavior planner on a separate loop
- can call the OpenAI API at the configured frequency
- applies the latest valid matching-id response to update the temporary destination lane and speed overrides

## Potential Function

The live obstacle cost is based on a super-ellipsoid field.

### Relative pose

For each obstacle, the ego position is transformed into the obstacle frame:

$$
\begin{bmatrix}
x_{loc}\\
y_{loc}
\end{bmatrix}

=
R(-\psi_o)
\begin{bmatrix}
x_e - x_o\\
y_e - y_o
\end{bmatrix}
$$

### Shape inflation

The ego footprint is projected into the obstacle frame using heading difference:

$$
\Delta \psi = \psi_e - \psi_o
$$

$$
L^{proj}_e = |L_e\cos(\Delta\psi)| + |W_e\sin(\Delta\psi)|
$$

$$
W^{proj}_e = |L_e\sin(\Delta\psi)| + |W_e\cos(\Delta\psi)|
$$

Static base half-sizes are then:

$$
x_0 = \frac{L^{proj}_e + L_o}{2} + b_{long}
$$

$$
y_0 = \frac{W^{proj}_e + W_o}{2} + b_{lat}
$$

### Dynamic inflation

Using relative approach speeds, the model builds:

- a larger safe zone
- a tighter collision zone

Collision-zone half-sizes:

$$
x_c = x_0 + \frac{\Delta u^2}{2 a_{max}}
$$

$$
y_c = y_0 + \frac{\Delta v^2}{2 a_{max}}
$$

Safe-zone half-sizes:

$$
x_s = x_0 + \Delta u T_r + \frac{\Delta u^2}{2 a_{comfort}}
$$

$$
y_s = y_0 + \Delta v T_r + \frac{\Delta v^2}{2 a_{comfort}}
$$

The live project also applies geometric limits to keep the field size bounded:

$$
x_c \leftarrow \min\left(x_c,\ \frac{L_{max}}{2}\right), \qquad
x_s \leftarrow \min\left(x_s,\ \frac{L_{max}}{2}\right)
$$

$$
y_c \leftarrow \min\left(y_c,\ \frac{W_{lane} f_{lane}}{2}\right), \qquad
y_s \leftarrow \min\left(y_s,\ \frac{W_{lane} f_{lane}}{2}\right)
$$

where:

- $L_{max}$ = `max_longitudinal_zone_length_m`
- $W_{lane}$ = lane width from the road config
- $f_{lane}$ = `max_lateral_zone_lane_fraction`

### Super-ellipsoid normalized distances

$$
r_c = \left(\left|\frac{x_{loc}}{x_c}\right|^n + \left|\frac{y_{loc}}{y_c}\right|^n\right)^{1/n}
$$

$$
r_s = \left(\left|\frac{x_{loc}}{x_s}\right|^n + \left|\frac{y_{loc}}{y_s}\right|^n\right)^{1/n}
$$

where $n$ is the super-ellipsoid shape exponent.

### Stage obstacle cost

The current live obstacle cost is:

$$
J_{safe} = w_s \exp\left(-k_s (r_s - s_s)\right)
$$

$$
J_{collision} = w_c \exp\left(-k_c (r_c - s_c)\right)
$$

$$
J_{obs} = J_{safe} + J_{collision}
$$

Interpretation:

- both terms are always active
- smaller $r_s$ or $r_c$ produces larger cost
- $k_s, k_c$ control how sharply the cost changes with distance
- $s_s, s_c$ shift where the exponential reaches the reference level

The local obstacle cost is Taylor-expanded around the reference rollout so it can be inserted into the convex QP.

The same live super-ellipsoid safe/collision zones can also be visualized in
the `pygame` window through:

- `MPC/mpc.yaml -> mpc.cost.repulsive_potential.visualization.enabled`
- `grid_resolution`, `draw_margin_m`, `safe_zone_color_rgba`, and `collision_zone_color_rgba`

## Cost Function

The total stage cost is:

$$
J = J_{attractive} + J_{lane} + J_{repulsive} + J_{control}
$$

### Attractive destination cost

$$
J_{attractive,k} =
w_{att}
\left(
q_x (x_k - x_{ref})^2 +
q_y (y_k - y_{ref})^2 +
q_v (v_k - v_{ref})^2 +
q_\psi wrap(\psi_k - \psi_{ref})^2
\right)
$$

This pulls the ego toward the active destination state.

### Lane-center cost

For a lane reference point $(x^{lane}_k, y^{lane}_k, \psi^{lane}_k)$, the lateral error is:

$$
e^{lat}_k =
 -\sin(\psi^{lane}_k)(x_k - x^{lane}_k)
 +\cos(\psi^{lane}_k)(y_k - y^{lane}_k)
$$

The lane cost is:

$$
w_k = w_0 \alpha^{k-1}
$$

$$
J_{lane,k} = w_k \left((e^{lat}_k)^2 + q_{\psi,lane} wrap(\psi_k - \psi^{lane}_k)^2 \right)
$$

At the current default values, `q_psi` in the lane-center term may be zero, which means the lane term is acting as lateral position error only.

### Repulsive obstacle cost

$$
J_{repulsive,k} = \sum_i J_{obs,i,k}
$$

with the super-ellipsoid-based $J_{obs}$ above.

### Control smoothness cost

$$
J_{control,k} =
w_{ctrl}
\left(
q_a \left(\frac{a_k-a_{k-1}}{\Delta t}\right)^2 +
q_\delta \left(\frac{\delta_k-\delta_{k-1}}{\Delta t}\right)^2
\right)
$$

This penalizes control changes, not raw control magnitude.

## Hard Constraints

The live QP enforces:

### Initial-state equality

$$
X_0 = X_{current}
$$

### Linearized dynamics at each stage

$$
X_{k+1} = A_k X_k + B_k U_k + c_k
$$

### Speed bounds

$$
v_{min} \le v_k \le v_{max}
$$

### Acceleration bounds

$$
a_{min} \le a_k \le a_{max}
$$

### Steering bounds

$$
\delta_{min} \le \delta_k \le \delta_{max}
$$

### Jerk bounds

$$
|a_k - a_{k-1}| \le j_{max}\Delta t
$$

### Steering-rate bounds

$$
\dot\delta_{min}\Delta t \le \delta_k - \delta_{k-1} \le \dot\delta_{max}\Delta t
$$

### Optional terminal speed equality

If enabled by scenario constraint settings:

$$
v_N = v_{terminal}
$$

## What Is Not A Hard Constraint

The current live MPC does not enforce:

- hard collision avoidance constraints
- hard road-boundary constraints
- hard lane-membership constraints

Obstacle avoidance is currently handled by the super-ellipsoid repulsive cost, not by a hard feasibility constraint.

## Final Stop Logic

Stopping at the final destination currently comes from:

1. final destination state usually has $v_{ref}=0$
2. final-stop speed cap:

$$
v_{cap}(d)=\min\left(v_{max},\sqrt{2 a_{brake}\max(d-d_{buffer},0)}\right)
$$

3. optional terminal-speed equality if enabled in scenario constraints

The old no-overshoot stop-plane logic is not part of the current live project anymore.

## PID Tracking Layer

The MPC output is a future state trajectory:

$$
[x,\ y,\ v,\ \psi]
$$

That planned trajectory is then followed by the PID controller, which generates:

$$
[a,\ \delta]
$$

So the architecture is:

- MPC = trajectory planner
- PID = trajectory tracker

The PID controller is not part of the optimization objective. It is the execution layer used after MPC planning.

## Default Config Files

### `MPC/mpc.yaml`

Main planner settings:

- `horizon_s`, `plan_dt_s`, `trajectory_generation_frequency_hz`
  Horizon length, discretization, and replan cadence.
- `behavior_planner_runtime.*`
  Behavior-planner frequency, OpenAI API settings, lane-safety thresholds, and logging controls.
- `local_goal.*`
  Temporary-destination generation used by rolling-goal scenarios, including dynamic lookahead and behavior-planner destination buffer.
- `final_stop_speed_cap.*`
  Stop-goal speed limit near final destinations.
- `reference_rollout.*`
  Heuristic guessed-trajectory settings for linearization.
- `cost.attractive.*`
  Destination state tracking weights.
- `cost.lane_center_follow.*`
  Lane-center and optional lane-heading penalty.
- `cost.control.*`
  Smoothness penalty on acceleration and steering changes.
- `cost.repulsive_potential.*`
  Super-ellipsoid obstacle potential parameters, plus live visualization and
  field-size limits (`max_longitudinal_zone_length_m`,
  `limit_lateral_zone_to_lane_width`, `max_lateral_zone_lane_fraction`).
- `solver.*`
  OSQP iteration and tolerance settings.

See the inline comments in `MPC/mpc.yaml` for the role of each variable and what increasing or decreasing it does.

### `utility/pid_controller.yaml`

Tracker settings:

- `tracking.*`
  Waypoint advancement, lookahead, steering-rate limiting, and LOS/path blending.
- `longitudinal.*`
  Speed PID gains.
- `lateral_heading.*`
  Heading PID gains.
- `lateral_cross_track.*`
  Cross-track PID gains.

This file already includes a `note:` section that explains the effect of each parameter.

### `utility/tracker.yaml`

- `polynomial_degree`
- `fit_window_s`
- `max_history_points`
- `min_points_for_polyfit`

These control the prediction model used for surrounding vehicles.

### `road/road.yaml`

Default road geometry and drawing settings:

- lane count / width
- margins
- waypoint spacing
- drawing colors

### `vehicle_manager/vehicle_manager.yaml`

Global fallback geometry and per-type appearance defaults.

### `state_manager/state_manager.yaml`

Controls how much object history is stored.

## Scenario YAML Structure

Each scenario typically contains:

- `simulation`
  dt, fps, plot sizes, replan buffer, startup delay
- `window`
  render size and pixels-per-meter
- `road`
  straight/curved geometry and camera defaults
- `destination`
  final destination state
- `mpc`
  scenario-specific overrides, usually `local_goal` and `constraints`
- `tracker`
  prediction overrides
- `vehicles`
  ego and non-ego initial states, motion modes, and object geometry

Scenario-local temporary-destination lane switching has been removed. The lane
of the active temporary destination is now controlled by the shared
behavior-planner runtime, not by per-scenario hardcoded lane-change logic.

## Active Scenarios

- `scenario4`
  Curved-road rolling-goal scenario used for general MPC and behavior-planner tests.
- `VRU`
  Curved-road scenario with vulnerable-road-user style interactions.
- `red_light_violation_warning`
  Curved-road red-light proxy scenario.
- `red_light_violation_warning_2`
  Variant of the red-light scenario used for additional behavior and stability tests, including non-ego spin-in-place behavior hooks.
- `workzone`
  Workzone scenario used heavily for behavior-planner lane-selection tests.
- `workzone with bp`
  Workzone variant kept for additional behavior-planner experiments.

## Dependencies

See `requirements.txt`.

Current live dependencies are:

- `numpy`
- `scipy`
- `osqp`
- `pyyaml`
- `openai`
- `pygame`
- `matplotlib`

## Cleanup Notes

This repository intentionally keeps `super_ellipsoid.py` even though it is not in the live runtime path, because it documents and visualizes the safety-field idea used by the MPC obstacle cost.

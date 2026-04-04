# Motion Planning — LEGO Insertion Task

> Gabor's branch — developed on top of the existing SAC/DRL stack.
> All code lives under `crisp_drl/motion_planning/`, `crisp_drl/agents/shared/`, and `scripts/`.

---

## Overview

The motion planning stack handles two distinct phases of the LEGO brick insertion task:

| Phase | What happens | Code |
|---|---|---|
| **Free-space** | Move arm from home → above socket (large Cartesian moves) | `trajectory_planner.py`, `free_space_planner.py` |
| **Insertion search** | Fine XY search near the socket with Z contact maintained | `planners.py`, `sim_wrappers.py` |

The two phases are deliberately separated: free-space motion uses `Poly7Planner` / `TrajectoryFollower` (7th-order polynomial, zero jerk at endpoints); insertion search uses episode-level planners (`GoToGoalPlanner`, `SpiralPlanner`, `HybridPlanner`) that operate within the RL wrapper.

---

## File Map

```
crisp_drl/motion_planning/
├── trajectory_planner.py        # Free-space: TrajectoryPlanner + TrajectoryFollower + Poly7Planner
├── free_space_planner.py        # Free-space: FreeSpaceMotionPlanner (P-controller waypoint executor)
├── planners.py                  # Insertion: BasePlanner, GoToGoal, Oracle, Spiral, HybridPlanner
├── rl_policy.py                 # RLPolicyPlanner: loads SAC checkpoint as a BasePlanner
└── sim_wrappers.py              # Gym wrappers + create_simulated_env_lego() factory

crisp_drl/agents/shared/
└── insertion_wrapper_lego.py    # Real-robot wrapper: InsertionWrapperRealLEGO

crisp_drl/envs/
└── make_env.py                  # create_real_env_lego() — real robot factory (requires ROS)

scripts/
├── test_trajectory.py           # Demo: full pick-and-place pipeline (HOME→GRASP→INSERT→HOME)
├── test_motion_planner.py       # Benchmark: insertion planners, success rate, step count
└── benchmark_planners.py        # Benchmark: trajectory planners (raised_cosine vs poly7)
```

---

## Free-Space Motion Planning

### `Poly7Planner` + `TrajectoryFollower`  ← primary planner

`trajectory_planner.py` provides the main free-space motion stack:

**`Poly7Planner`** — 7th-order polynomial trajectory (kinematically optimal).
- Zero velocity, acceleration **and jerk** at both endpoints.
- `plan(start, goal)` → `List[Waypoint]` sampled from polynomial
- `t_min(dist)` → minimum time given acceleration limit
- Constructor: `Poly7Planner(a_limit=1.0, env_dt=0.066)`
- Used in `InsertionWrapperRealLEGO.go_to_waypoint()` for real robot free-space moves

**`TrajectoryFollower`** — env interaction only, portable to real robot.
- `follow_sampled(env, obs, waypoints)` → `(obs, FollowResult)`
  One `env.step()` per waypoint — executes the Poly7 velocity profile exactly.
- `follow(env, obs, waypoints)` → `(obs, FollowResult)`
  Arc-length parameterised polyline + **raised-cosine velocity profile** (alternative to Poly7).
- `go_to(env, obs, target_xyz, tolerance)` → `(obs, reached, n_steps)`
  Simple P-controller toward a single point; used for fine settling.
- Real-robot port: subclass and override `_step()` to call the robot's Cartesian controller.

**`TrajectoryPlanner`** — straight-line waypoint generator with adaptive spacing.
- `plan(start, goal, coarse_step_size, fine_zone)` → `List[Waypoint]`
- Fine (1 mm) near start/end, coarse (30 mm) in the middle — used with `follow()`.

### `FreeSpaceMotionPlanner`

`free_space_planner.py` — P-controller step loop for scripted waypoint sequences.
- Executes a list of `CartesianWaypoint` objects.
- Used inside `InsertionWrapperSimLEGO.reset()` for pre-insertion approach.
- Override `go_to_waypoint_3d()` to port to real robot.

---

## Insertion-Phase Planners

All live in `planners.py`. Shared interface:
```python
planner.reset(goal_position)   # once per episode — receives 3D goal [x, y, z]
planner.plan(obs)              # every step — returns 2D action [dx, dy] in metres
```

Actions are XY only (1 mm max step). Z is handled exclusively by the wrapper's force/impedance controller.

| Planner | Strategy | Use case |
|---|---|---|
| `OraclePlanner` | Uses `observation.perfect_action` directly | Upper-bound baseline |
| `GoToGoalPlanner` | P-controller straight toward known goal XY | Primary baseline; ~100% SR |
| `SpiralPlanner` | Archimedean spiral from goal centre; switches to P-controller on contact | No-vision search validation |
| `HybridPlanner` | `GoToGoalPlanner` until within `switch_distance`, then hands off to `rl_planner` | Combining motion planning + RL |
| `RLPolicyPlanner` | Loads SAC checkpoint (Actor + SharedEncoder) as a `BasePlanner` | Testing trained policies |

### `HybridPlanner`

```python
HybridPlanner(rl_planner=None, switch_distance=0.004)
```
- Coarse phase: `GoToGoalPlanner` (deterministic, fast convergence).
- Fine phase: `rl_planner` once `dist_xy <= switch_distance`. Switch is **one-way**.
- If `rl_planner=None`, fine phase falls back to `GoToGoalPlanner` (tests switching logic without a checkpoint).

### `RLPolicyPlanner`

```python
RLPolicyPlanner(checkpoint_dir)
```
- Loads `actor_state_dict.pth` + `shared_encoder_state_dict.pth` from `checkpoint_dir`.
- Formats raw obs dict → scaled flat tensor → `SharedEncoder` → `Actor` → 2D action.
- Observation scaling: errors/positions ×1000, F/T ×0.1.
- CPU-only (no CUDA required for inference).

---

## Gym Wrapper Stack

### Simulation

```
MujidEnv
  └── ActionTimeStampWrapper        records last action in obs
        └── LastObservationWrapper      adds error/velocity/previous keys
              └── InsertionWrapperSimLEGO
                    └── CustomTerminationWrapper   success/failure events
```

Factory: `create_simulated_env_lego()` in `crisp_drl/motion_planning/sim_wrappers.py`
(ROS-free — importable in sim environment without real robot dependencies)

### Real Robot

```
make_env("my_env_v4")              ROS/crisp_gym base env
  └── ActionTimeStampWrapper
        └── NoGripperActionWrapper
              └── LastObservationWrapper
                    └── SensorTareWrapper         zeros F/T on reset
                          └── InsertionWrapperRealLEGO
                                └── CLIWrapper
                                      └── DinoImageEncoderWrapper   wrist camera → DINO features
                                            └── ObservationFormatterWrapper  scales + flattens for policy
```

Factory: `create_real_env_lego()` in `crisp_drl/envs/make_env.py`

---

## `InsertionWrapperSimLEGO` (sim)

Mirrors `InsertionWrapperSiemens` structure. Reset does **all** motion planning; `step()` is purely RL + Z controller.

**`reset()` phases:**
- **⑥ Randomise** — sample grasp offset, goal XY noise, start position.
- **⑦ `go_to_waypoint()`** — GoToGoal loop to approach goal XY (not counted in `n_steps`).
- **⑧ Z contact** — tare F/T sensor, run Z impedance loop until target pressing depth.

**`step(action)`** — `action = [dx_rl, dy_rl]`:
- XY: RL action.
- Z: `z_impedance_controller_dz()` — maintains target pressing depth automatically.
- Safety box: if robot drifts outside `safety_box_radius` of goal, overrides with correcting action.

**Axis mapping:** Z (index 2) = insertion (pressing down), XY (indices 0, 1) = RL search plane.

---

## `InsertionWrapperRealLEGO` (real robot)

`crisp_drl/agents/shared/insertion_wrapper_lego.py`

Mirrors `InsertionWrapperSiemens` for real hardware. Replaces the Siemens I-controller with `Poly7Planner` for free-space motion.

**`reset()` phases:**
- **① Home** — `env.unwrapped.home()` to joint-space home via `LegoConfig.custom_home_position`.
- **② Randomise** — sample grasp X/Z offsets, compute goal and start positions.
- **③ Waypoints** — follow `waypoints_before_insertion` list via `go_to_waypoint()` (Poly7), then approach start position.
- **④ Contact** — tare F/T, run `z_force_controller_dz()` loop until `|Fz_error| < contact_force_threshold`.

**`go_to_waypoint(obs, position, distance_err, is_via)`**
- Uses `Poly7Planner.plan()` + `TrajectoryFollower.follow_sampled()`.
- `is_via=True`: transit point (looser tolerance); `is_via=False`: terminal point (precise settling).

**`z_force_controller_dz(obs)`** — adapted from Siemens `x_torque_controller_dx()`:
- Uses `sensors_bota_ft_sensor[2]` (Fz) directly instead of Y-torque via lever arm.
- Target: `ft_force_target` [N] (default −3.0 N = pressing down).

**`step(action)`** — same as sim wrapper: RL XY + Z force ctrl + safety box.

**Axis mapping:** same as sim — Z = insertion, XY = search plane.

---

## Env Factories

| Factory | Module | Robot | Description |
|---|---|---|---|
| `create_simulated_env_lego()` | `crisp_drl.motion_planning.sim_wrappers` | Sim | No ROS, no CUDA — raw obs dict |
| `create_real_env_lego()` | `crisp_drl.envs.make_env` | Real | Full stack with DINO + ObsFormatter |

```python
# Sim (importable without ROS)
from crisp_drl.motion_planning.sim_wrappers import create_simulated_env_lego
env = create_simulated_env_lego({"initial_keyframe": 2}, pe_accuracy=0.0015)

# Real robot
from crisp_drl.envs.make_env import create_real_env_lego
from crisp_drl.agents.shared.insertion_env_config import LegoConfig
from crisp_drl.agents.shared.algorithm_config import Config
env = create_real_env_lego(Config(), LegoConfig())
```

---

## Testing Scripts

### `test_trajectory.py` — Full pipeline demo

Runs the complete 9-state pick-and-place sequence using `TrajectoryPlanner` + `TrajectoryFollower`:
`HOME → ABOVE_A → AT_A → GRASP → ABOVE_A → ABOVE_B → AT_B → RELEASE → HOME`

```bash
cd real_world_drl
MUJOCO_GL=egl  pixi run -e sim python scripts/test_trajectory.py
MUJOCO_GL=glfw pixi run -e sim python scripts/test_trajectory.py --live_view
```

| Flag | Default | Description |
|---|---|---|
| `--live_view` | off | Open MuJoCo interactive viewer |
| `--approach_height` | `0.07` | Transit height above grasp/insert [m] |
| `--hold_s` | `2.0` | Hold duration for grasp/release [s] |
| `--n_runs` | `3` | Repeat pipeline N times |

---

### `test_motion_planner.py` — Insertion planner benchmark

Runs a chosen planner for N episodes and reports success rate + step statistics.

```bash
MUJOCO_GL=egl pixi run -e sim python scripts/test_motion_planner.py \
    --wrapper lego --planner go_to_goal --n_episodes 20
```

| Flag | Default | Description |
|---|---|---|
| `--wrapper` | `sim` | `sim` = `InsertionWrapperSim`, `lego` = `InsertionWrapperSimLEGO` |
| `--planner` | `go_to_goal` | `oracle`, `go_to_goal`, `spiral`, `hybrid` |
| `--n_episodes` | `100` | Number of episodes |
| `--pe_accuracy` | `0.0015` | Pose estimation accuracy [m] — sets randomisation range |
| `--eval_mode` | off | Tighter randomisation (eval distribution) |
| `--live_view` | off | Open MuJoCo viewer |
| `--verbose` | off | Print every episode (default: every 10) |
| `--approach_distance` | `None` | Wrapper approaches goal in `reset()` before RL starts |
| `--start_xy X Y` | random | Fix start position [m] |
| `--goal_xy X Y` | random | Fix goal position [m] |
| `--checkpoint PATH` | — | SAC checkpoint dir (hybrid planner, RL fine-phase) |
| `--switch_distance` | `0.004` | XY distance [m] to trigger GoToGoal→RL switch (hybrid) |

**Examples:**

```bash
# Baseline: ~100% SR expected
MUJOCO_GL=egl pixi run -e sim python scripts/test_motion_planner.py \
    --wrapper lego --planner go_to_goal --n_episodes 20 --verbose

# Fixed start/goal — controlled distance experiment
MUJOCO_GL=egl pixi run -e sim python scripts/test_motion_planner.py \
    --start_xy 0.615 0.005 --goal_xy 0.600 0.000 --n_episodes 20

# Hybrid planner with trained RL checkpoint
MUJOCO_GL=egl pixi run -e sim python scripts/test_motion_planner.py \
    --wrapper lego --planner hybrid \
    --checkpoint /path/to/checkpoint --switch_distance 0.004
```

---

### `benchmark_planners.py` — Trajectory planner comparison

Benchmarks `raised_cosine` vs `poly7` on the 6-move pick-and-place sequence. Moves chain continuously (no reset between moves).

```bash
MUJOCO_GL=egl  pixi run -e sim python scripts/benchmark_planners.py --planner all --a_limit 3.0
MUJOCO_GL=glfw pixi run -e sim python scripts/benchmark_planners.py --planner all --live_view --plot
```

| Flag | Default | Description |
|---|---|---|
| `--planner` | `all` | `raised_cosine`, `poly7`, or `all` |
| `--a_limit` | `2.0` | Acceleration limit for `Poly7Planner` [m/s²] |
| `--live_view` | off | Open MuJoCo viewer |
| `--plot` | off | Save velocity/acceleration plots to `benchmark_results.png` |

**Output table columns** (per move, per planner): `steps`, `wall_ms`, `final_err_mm`, `n_waypoints`.

**Move sequence** (`POS_A = [0.30, 0.20, 0.13]`, `POS_B = [0.60, 0.00, 0.14]`, approach height 70 mm):
```
HOME → ABOVE_A → AT_A → ABOVE_A → ABOVE_B → AT_B → HOME
```

---

## Environment Setup

All scripts require the `sim` pixi environment:

```bash
cd real_world_drl
pixi shell -e sim           # interactive shell
# or
pixi run -e sim python scripts/<script>.py
```

Headless rendering (WSL2 / no display):
```bash
MUJOCO_GL=egl python scripts/...
```

Interactive viewer (requires display):
```bash
MUJOCO_GL=glfw python scripts/... --live_view
```

---

## Success Condition

An episode is successful when the `moving_brick` is:
- Within **1 mm XY** of `[0.6, 0.0, 0.1198]` (socket centre), AND
- Z error > **0.8 mm** (brick is pressing into socket)

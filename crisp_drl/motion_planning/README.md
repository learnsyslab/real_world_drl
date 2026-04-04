# Motion Planning — LEGO Insertion Task (Sim)

> Gabor's branch — developed on top of the existing SAC/DRL stack.
> All code lives under `crisp_drl/motion_planning/` and `scripts/`.

---

## Overview

The motion planning stack handles two distinct phases of the LEGO brick insertion task:

| Phase | What happens | Code |
|---|---|---|
| **Free-space** | Move arm from home → above socket (large Cartesian moves) | `trajectory_planner.py`, `free_space_planner.py` |
| **Insertion search** | Fine XY search near the socket with Z contact maintained | `planners.py`, `sim_wrappers.py` |

The two phases are deliberately separated: free-space motion uses `TrajectoryPlanner` / `TrajectoryFollower` (smooth velocity profiles, millimetre-level step resolution); insertion search uses episode-level planners (`GoToGoalPlanner`, `SpiralPlanner`, `HybridPlanner`) that operate within the RL wrapper.

---

## File Map

```
crisp_drl/motion_planning/
├── trajectory_planner.py   # Free-space: TrajectoryPlanner + TrajectoryFollower + Poly7Planner
├── free_space_planner.py   # Free-space: FreeSpaceMotionPlanner (P-controller waypoint executor)
├── planners.py             # Insertion: BasePlanner, GoToGoal, Oracle, Spiral, HybridPlanner
├── sim_wrappers.py         # Gym wrappers: InsertionWrapperSim, InsertionWrapperSimLEGO, ...
└── rl_policy.py            # RLPolicyPlanner: loads SAC checkpoint as a BasePlanner

scripts/
├── test_trajectory.py      # Demo: full pick-and-place pipeline (HOME→GRASP→INSERT→HOME)
├── test_motion_planner.py  # Benchmark: insertion planners, success rate, step count
└── benchmark_planners.py   # Benchmark: trajectory planners (raised_cosine vs poly7)
```

---

## Free-Space Motion Planning

### `TrajectoryPlanner` + `TrajectoryFollower`

`trajectory_planner.py` provides two cooperating classes:

**`TrajectoryPlanner`** — pure geometry, no env dependency.
- `plan(start, goal, coarse_step_size, fine_zone)` → `List[Waypoint]`
- Generates a straight-line path with adaptive waypoint spacing:
  fine (1 mm) near start/end, coarse (30 mm) in the middle of long transits.

**`TrajectoryFollower`** — env interaction only.
- `follow(env, obs, waypoints)` → `(obs, FollowResult)`
  Arc-length parameterised polyline over all waypoints + **raised-cosine velocity profile** (smooth ease-in/ease-out, no jerk at waypoint boundaries).
- `follow_sampled(env, obs, waypoints)` → `(obs, FollowResult)`
  One `env.step()` per waypoint — used by `Poly7Planner`.
- `go_to(env, obs, target_xyz, tolerance)` → `(obs, reached, n_steps)`
  Simple P-controller toward a single point; used internally and in reset phases.

**`Poly7Planner`** — 7th-order polynomial trajectory.
- Zero velocity, acceleration **and jerk** at both endpoints — kinematically optimal.
- `plan(start, goal)` → `List[Waypoint]`  (sampled from polynomial)
- `t_min(dist)` → minimum time given acceleration limit.
- Constructor: `Poly7Planner(a_limit=2.0, env_dt=0.066)`

### `FreeSpaceMotionPlanner`

`free_space_planner.py` — simpler alternative for scripted waypoint sequences.
- Executes a list of `CartesianWaypoint` objects via a P-controller step loop.
- Used inside `InsertionWrapperSimLEGO.reset()` for the approach phase.
- Override `go_to_waypoint_3d()` to port to a real robot without changing callers.

---

## Insertion-Phase Planners

All live in `planners.py`. Shared interface:
```python
planner.reset(goal_position)   # once per episode — receives 3D goal [x, y, z]
planner.plan(obs)              # every step — returns 2D action [dx, dy] in metres
```

Actions are XY only (1 mm max step). Z is handled exclusively by the wrapper's impedance controller.

| Planner | Strategy | Use case |
|---|---|---|
| `OraclePlanner` | Uses `observation.perfect_action` directly — knows exact goal vector every step | Upper-bound baseline |
| `GoToGoalPlanner` | P-controller straight toward known goal XY | Primary baseline; should give ~100% SR |
| `SpiralPlanner` | Archimedean spiral from goal centre; switches to P-controller on contact | No-vision search validation |
| `HybridPlanner` | `GoToGoalPlanner` until within `switch_distance`, then hands off to `rl_planner` | Combining motion planning + RL |
| `RLPolicyPlanner` | Loads SAC checkpoint (Actor + SharedEncoder) as a `BasePlanner` | Testing trained policies |

### `HybridPlanner` details

```python
HybridPlanner(rl_planner=None, switch_distance=0.004)
```
- Coarse phase: `GoToGoalPlanner` (deterministic, fast convergence).
- Fine phase: `rl_planner` once `dist_xy <= switch_distance`. Switch is **one-way** — never returns to GoToGoal.
- If `rl_planner=None`, fine phase also uses `GoToGoalPlanner` (useful to test switching logic alone).

### `RLPolicyPlanner` details

```python
RLPolicyPlanner(checkpoint_dir)
```
- Loads `actor_state_dict.pth` + `shared_encoder_state_dict.pth` from `checkpoint_dir`.
- Formats raw obs dict → scaled flat tensor → `SharedEncoder` → `Actor` → 2D action.
- Observation scaling: errors/positions ×1000, F/T ×0.1.
- CPU-only (no CUDA required for inference).

---

## Gym Wrapper Stack

```
MujidEnv
  └── ActionTimeStampWrapper      records last action in obs
        └── LastObservationWrapper    adds error/velocity/previous keys
              └── InsertionWrapperSim / InsertionWrapperSimLEGO
                    └── CustomTerminationWrapper   success/failure events
```

### `InsertionWrapperSim` (original)

- `reset()`: samples random start/goal positions, teleports robot, optionally runs approach.
- `step(action)`: 2D RL action → 6D env action; Z pressing gated to when XY is close to goal.
- `approach_distance`: if set, runs `GoToGoalPlanner` loop inside `reset()` until within this distance.

### `InsertionWrapperSimLEGO` (Siemens-style)

Mirrors `InsertionWrapperSiemens` structure exactly. Reset does **all** motion planning; `step()` is purely RL + Z controller.

**`reset()` phases:**
- **⑥ Randomise** — sample grasp offset, goal XY noise, start position.
- **⑦ `go_to_waypoint()`** — approach goal XY (GoToGoal loop, not counted in `n_steps`).
- **⑧ Z contact** — tare F/T sensor, run Z impedance loop until target pressing depth.

**`step(action)`** — `action = [dx_rl, dy_rl]`:
- X, Y: RL action.
- Z: `z_impedance_controller_dz()` — maintains target pressing depth automatically.
- Safety box: if robot drifts outside `safety_box_radius` of goal, overrides with correcting action.

**Axis mapping** (LEGO-specific, no abstraction constants):
- Z (index 2) = insertion axis (pressing down).
- XY (indices 0, 1) = RL search plane.

---

## Testing Scripts

### `test_trajectory.py` — Full pipeline demo

Runs the complete 9-state pick-and-place sequence:
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
| `--approach_distance` | `None` | If set, wrapper approaches goal in `reset()` before RL starts |
| `--start_xy X Y` | random | Fix start position [m] |
| `--goal_xy X Y` | random | Fix goal position [m] |
| `--checkpoint PATH` | — | SAC checkpoint dir (hybrid planner, RL fine-phase) |
| `--switch_distance` | `0.004` | XY distance [m] to trigger GoToGoal→RL switch (hybrid) |

**Examples:**

```bash
# Baseline: 100% SR expected
MUJOCO_GL=egl pixi run -e sim python scripts/test_motion_planner.py \
    --wrapper lego --planner go_to_goal --n_episodes 20 --verbose

# Fixed start/goal — controlled distance experiment (15 mm apart)
MUJOCO_GL=egl pixi run -e sim python scripts/test_motion_planner.py \
    --start_xy 0.615 0.005 --goal_xy 0.600 0.000 --n_episodes 20

# Hybrid planner with trained RL checkpoint
MUJOCO_GL=egl pixi run -e sim python scripts/test_motion_planner.py \
    --wrapper lego --planner hybrid \
    --checkpoint /path/to/checkpoint --switch_distance 0.004
```

---

### `benchmark_planners.py` — Trajectory planner comparison

Benchmarks `raised_cosine` vs `poly7` on the 6-move pick-and-place sequence. Runs moves **continuously** (no reset between moves).

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

**Output table columns** (per move, per planner):

| Column | Meaning |
|---|---|
| `steps` | Number of `env.step()` calls |
| `ms` | Wall time in milliseconds |
| `err` | Final position error [mm] |
| `n_wp` | Number of waypoints generated |

**Move sequence** (`POS_A = [0.30, 0.20, 0.13]`, `POS_B = [0.60, 0.00, 0.14]`, approach height 70 mm):
```
HOME → ABOVE_A → AT_A → ABOVE_A → ABOVE_B → AT_B → HOME
```

**`--plot`** saves a grid of plots: arc length, speed [mm/step], and acceleration [mm/step²] for each move × each planner.

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

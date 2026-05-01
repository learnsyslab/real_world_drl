"""Bring-up checklist for InsertionWrapper3DoFRotZ on real hardware.

Tests
-----
T0  Unit tests — no robot needed (import + math checks)
T1  Smoke test  — env creation, reset, one zero-action step, obs/shape checks
T2  Pure rotation — [0,0,drz] moves yaw; XY/rx/ry stay put
T3  Pure XY      — [dx,0,0] advances X; yaw stays
T4  Angular safety box — action past limit gets clamped
T5  FT independence  — yaw rotation while FT Z-controller keeps contact
T6  Short data-collection dry run (N_ROLLOUTS=3, p=1.0)

Usage
-----
# Unit tests only (no robot):
python scripts/test_3dof_rz.py --tests T0

# Full hardware bring-up (robot must be on and homed):
python scripts/test_3dof_rz.py --tests T1 T2 T3 T4 T5

# With FT sensor disabled (for benchtop without sensor):
python scripts/test_3dof_rz.py --tests T1 T2 T3 T4 --no_ft_sensor
"""

import argparse
import shutil
import sys
import time
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RESULTS: dict[str, bool] = {}


def _ok(name: str, cond: bool, detail: str = "") -> bool:
    status = "PASS" if cond else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return cond


def _make_config():
    from crisp_drl.agents.shared.algorithm_config import Config
    config = Config()
    config.actor_output_dim = 3
    config.actor_nonvision_input_dim = 18
    config.max_action = np.array([0.00025, 0.00025, np.deg2rad(0.5)])
    return config


def _make_env(args=None):
    from crisp_drl.envs import make_env as make_env_mod
    config = _make_config()
    env = make_env_mod.create_real_env_v4_3dof_rz(config, args=args)
    return env, config


def _unwrap_insertion(env):
    """Walk the wrapper stack to reach InsertionWrapper3DoFRotZ."""
    from crisp_drl.agents.shared.insertion_wrapper import InsertionWrapper3DoFRotZ
    w = env
    while not isinstance(w, InsertionWrapper3DoFRotZ):
        w = w.env
    return w


# ---------------------------------------------------------------------------
# T0 — unit tests, no hardware
# ---------------------------------------------------------------------------

def test_T0_unit() -> bool:
    print("\n=== T0: Unit tests (no hardware) ===")
    ok = True

    # 1. crisp_gym math (no ROS dependency)
    try:
        from crisp_gym.envs.env_wrapper import LastObservationWrapper  # noqa
        ok &= _ok("crisp_gym.LastObservationWrapper importable", True)
    except ImportError as e:
        ok &= _ok("crisp_gym import", False, str(e))
        return ok

    # 2. Config dims consistent
    config = _make_config()
    ok &= _ok("actor_output_dim == 3", config.actor_output_dim == 3)
    ok &= _ok("actor_nonvision_input_dim == 18", config.actor_nonvision_input_dim == 18)
    ok &= _ok("max_action shape == (3,)", config.max_action.shape == (3,))

    # 3. formatter non-vision slice count = 18
    keys_ranges_scales = [
        ("observation.previous.action",          (0, 2),  1000.0),
        ("observation.previous.action",          (5, 6),  40.0),
        ("observation.velocity.cartesian",       (0, 2),  1000.0),
        ("observation.velocity.angular",         (2, 3),  40.0),
        ("observation.error.cartesian",          (0, 2),  1000.0),
        ("observation.error.angular",            (2, 3),  40.0),
        ("observation.previous.error.cartesian", (0, 2),  1000.0),
        ("observation.previous.error.angular",   (2, 3),  40.0),
        ("observation.state.sensors_bota_ft_sensor", (0, 6), 0.1),
    ]
    non_vision = sum(b - a for _, (a, b), _ in keys_ranges_scales)
    ok &= _ok("non-vision formatter dims == 18", non_vision == 18, str(non_vision))

    # 4. _yaw_error_to math via LastObservationWrapper (already imported above)

    def yaw_err(target_rz, current_rz):
        goal_rv = np.array([0.0, 0.0, float(target_rz)])
        curr_rv = np.array([0.0, 0.0, float(current_rz)])
        return float(LastObservationWrapper._relative_rotation_error(goal_rv, curr_rv)[2])

    err_pos = yaw_err(0.0, np.deg2rad(-30.0))
    ok &= _ok("yaw_err(target=0, curr=-30°) ≈ +30°",
              abs(err_pos - np.deg2rad(30.0)) < 1e-6,
              f"got {np.rad2deg(err_pos):.4f}°")

    err_neg = yaw_err(0.0, np.deg2rad(30.0))
    ok &= _ok("yaw_err(target=0, curr=+30°) ≈ -30°",
              abs(err_neg - np.deg2rad(-30.0)) < 1e-6,
              f"got {np.rad2deg(err_neg):.4f}°")

    # near-360° wrap-around: curr=179° → err ≈ -179° (shortest arc)
    err_wrap = yaw_err(0.0, np.deg2rad(179.0))
    ok &= _ok("yaw_err wraps: (target=0, curr=179°) has |err|≈179°",
              abs(abs(err_wrap) - np.deg2rad(179.0)) < 1e-5,
              f"got {np.rad2deg(err_wrap):.4f}°")

    return ok


# ---------------------------------------------------------------------------
# T1 — smoke test
# ---------------------------------------------------------------------------

def test_T1_smoke(env, config) -> bool:
    print("\n=== T1: Smoke test ===")
    ok = True

    obs, info = env.reset()

    ok &= _ok("action_space.shape == (3,)",
              env.action_space.shape == (3,), str(env.action_space.shape))

    expected_fmt = config.vision_head_input_dim + config.actor_nonvision_input_dim
    fmt_shape = obs["observation.formatted"].shape
    ok &= _ok(f"observation.formatted shape == ({expected_fmt},)",
              fmt_shape == (expected_fmt,), str(fmt_shape))

    ok &= _ok("observation.state.cartesian shape (6,)",
              obs["observation.state.cartesian"].shape == (6,))

    ok &= _ok("observation.error.angular shape (3,)",
              obs["observation.error.angular"].shape == (3,))

    ok &= _ok("observation.velocity.angular shape (3,)",
              obs["observation.velocity.angular"].shape == (3,))

    ok &= _ok("reset.goal_orientation.rotation_z in info",
              "reset.goal_orientation.rotation_z" in info)

    ok &= _ok("reset.start_orientation.rotation_z in info",
              "reset.start_orientation.rotation_z" in info)

    goal_rz = float(info.get("reset.goal_orientation.rotation_z", float("nan")))
    start_rz = float(info.get("reset.start_orientation.rotation_z", float("nan")))
    ok &= _ok("goal_rz ∈ [-1.5°, +1.5°]",
              abs(goal_rz) <= np.deg2rad(1.5) + 1e-9,
              f"{np.rad2deg(goal_rz):.3f}°")

    ok &= _ok("start_rz within safety box of goal_rz",
              abs(start_rz - goal_rz) <= np.deg2rad(3.0) + 1e-9,
              f"start={np.rad2deg(start_rz):.3f}° goal={np.rad2deg(goal_rz):.3f}°")

    # one zero step
    obs2, r, term, trunc, _ = env.step(np.zeros(3))
    ok &= _ok("step with zeros(3) succeeds", True)
    ok &= _ok("observation.formatted returned", "observation.formatted" in obs2)
    ok &= _ok("reward is float", isinstance(r, (float, np.floating)))

    return ok


# ---------------------------------------------------------------------------
# T2 — pure rotation
# ---------------------------------------------------------------------------

def test_T2_pure_rotation(env) -> bool:
    print("\n=== T2: Pure rotation ===")
    ok = True

    obs, info = env.reset()
    goal_rz = float(info["reset.goal_orientation.rotation_z"])

    rz0 = float(obs["observation.state.cartesian"][5])
    xy0 = obs["observation.state.cartesian"][:2].copy()

    # 5 small yaw steps (0.3°) toward goal — safely inside angular safety box
    rz_err = (goal_rz - rz0 + np.pi) % (2 * np.pi) - np.pi
    drz_step = np.sign(rz_err) * np.deg2rad(0.3) if abs(rz_err) > 1e-6 else np.deg2rad(0.3)

    rz_vals = [rz0]
    xy_drifts = []

    N = 5
    for _ in range(N):
        obs, *_ = env.step(np.array([0.0, 0.0, drz_step]))
        rz_vals.append(float(obs["observation.state.cartesian"][5]))
        xy_drifts.append(np.linalg.norm(obs["observation.state.cartesian"][:2] - xy0))

    rz_change = rz_vals[-1] - rz_vals[0]
    max_xy = max(xy_drifts) * 1e3  # mm

    ok &= _ok("yaw moved ≥ 0.3° total",
              abs(rz_change) >= np.deg2rad(0.3),
              f"Δrz={np.rad2deg(rz_change):.3f}°")

    ok &= _ok("XY drift < 0.2 mm",
              max(xy_drifts) < 2e-4,
              f"max={max_xy:.3f} mm")

    rx = float(obs["observation.state.cartesian"][3])
    ry = float(obs["observation.state.cartesian"][4])
    ok &= _ok("rx < 2° (ry same)", abs(rx) < np.deg2rad(2) and abs(ry) < np.deg2rad(2),
              f"rx={np.rad2deg(rx):.3f}° ry={np.rad2deg(ry):.3f}°")

    return ok


# ---------------------------------------------------------------------------
# T3 — pure XY
# ---------------------------------------------------------------------------

def test_T3_pure_xy(env) -> bool:
    print("\n=== T3: Pure XY ===")
    ok = True

    obs, info = env.reset()
    rz0 = float(obs["observation.state.cartesian"][5])
    x0 = float(obs["observation.state.cartesian"][0])

    # Tiny +X steps (0.05 mm each, well inside safety box)
    dx = 5e-5
    x_vals = [x0]
    rz_vals = [rz0]

    N = 5
    for _ in range(N):
        obs, *_ = env.step(np.array([dx, 0.0, 0.0]))
        x_vals.append(float(obs["observation.state.cartesian"][0]))
        rz_vals.append(float(obs["observation.state.cartesian"][5]))

    x_change = x_vals[-1] - x_vals[0]
    rz_drift = abs(rz_vals[-1] - rz_vals[0])

    ok &= _ok("X advanced in commanded direction",
              x_change > 0,
              f"Δx={x_change*1e3:.4f} mm")

    ok &= _ok("yaw drift < 0.1° (0.0017 rad)",
              rz_drift < np.deg2rad(0.1),
              f"Δrz={np.rad2deg(rz_drift):.4f}°")

    return ok


# ---------------------------------------------------------------------------
# T4 — angular safety box clamping
# ---------------------------------------------------------------------------

def test_T4_angular_safety_box(env) -> bool:
    print("\n=== T4: Angular safety box ===")
    ok = True

    obs, info = env.reset()
    iw = _unwrap_insertion(env)

    safety_radius = iw.safety_box_angular_radius
    safety_step = iw.safety_box_angular_step_size
    goal_rz = iw.goal_rotation_z

    print(f"  safety_radius={np.rad2deg(safety_radius):.2f}°  "
          f"safety_step={np.rad2deg(safety_step):.3f}°  "
          f"goal_rz={np.rad2deg(goal_rz):.2f}°")

    # Push away from goal until we leave the safety box
    current_rz = float(obs["observation.state.cartesian"][5])
    rz_err_init = (goal_rz - current_rz + np.pi) % (2 * np.pi) - np.pi
    # push in the direction away from goal
    push_dir = -np.sign(rz_err_init) if abs(rz_err_init) > 1e-6 else 1.0
    big_drz = push_dir * 1.0  # 1 rad >> safety_radius

    exited = False
    for i in range(30):
        obs, *_ = env.step(np.array([0.0, 0.0, big_drz]))
        current_rz = float(obs["observation.state.cartesian"][5])
        rz_err = (goal_rz - current_rz + np.pi) % (2 * np.pi) - np.pi
        if abs(rz_err) > safety_radius:
            exited = True
            break

    ok &= _ok("managed to exit angular safety box",
              exited,
              f"|rz_err|={np.rad2deg(abs(rz_err)):.3f}° vs radius={np.rad2deg(safety_radius):.3f}°")  # type: ignore[reportPossiblyUnboundVariable]

    if not exited:
        return ok

    # Now verify next step is clamped
    rz_before = float(obs["observation.state.cartesian"][5])
    obs2, *_ = env.step(np.array([0.0, 0.0, big_drz]))
    rz_after = float(obs2["observation.state.cartesian"][5])
    actual_drz = abs(rz_after - rz_before)

    ok &= _ok("drz clamped to ≤ safety_step * 1.1",
              actual_drz <= safety_step * 1.1,
              f"actual={np.rad2deg(actual_drz):.4f}° safety_step={np.rad2deg(safety_step):.4f}°")

    return ok


# ---------------------------------------------------------------------------
# T5 — FT independence during yaw rotation
# ---------------------------------------------------------------------------

def test_T5_ft_independence(env) -> bool:
    print("\n=== T5: FT independence ===")
    ok = True

    obs, info = env.reset()
    iw = _unwrap_insertion(env)

    if not iw.use_ft_controller:
        print("  [SKIP] FT controller disabled — T5 skipped")
        return True

    ft_target = iw.z_force_target
    goal_rz = iw.goal_rotation_z

    # Determine a drz toward goal (small, safe)
    current_rz = float(obs["observation.state.cartesian"][5])
    rz_err = (goal_rz - current_rz + np.pi) % (2 * np.pi) - np.pi
    drz = np.sign(rz_err) * np.deg2rad(0.3) if abs(rz_err) > 1e-6 else np.deg2rad(0.3)

    ft_readings = []
    rz_vals = [current_rz]

    N = 10
    for _ in range(N):
        obs, *_ = env.step(np.array([0.0, 0.0, drz]))
        ft_readings.append(float(obs["observation.state.sensors_bota_ft_sensor"][2]))
        rz_vals.append(float(obs["observation.state.cartesian"][5]))

    rz_total = abs(rz_vals[-1] - rz_vals[0])
    ft_dev_max = max(abs(f - ft_target) for f in ft_readings)
    ft_in_range = ft_dev_max < 1.5  # within 1.5 N

    ok &= _ok("yaw rotated during FT control",
              rz_total > np.deg2rad(0.2),
              f"Δrz={np.rad2deg(rz_total):.3f}°")

    ok &= _ok("FT-Z within ±1.5 N of target",
              ft_in_range,
              f"max_dev={ft_dev_max:.3f} N  "
              f"target={ft_target} N  "
              f"readings=[{min(ft_readings):.2f}, {max(ft_readings):.2f}] N")

    return ok


# ---------------------------------------------------------------------------
# T6 — short data-collection dry run
# ---------------------------------------------------------------------------

def test_T6_data_collection(no_ft: bool = False) -> bool:
    """3-episode dry run: create dataset, collect, assert shapes."""
    print("\n=== T6: Data collection dry run (3 rollouts, p=1.0) ===")
    ok = True

    N_TEST = 3
    p_test = 1.0  # always perfect action — no randomness

    try:
        from pathlib import Path
        import numpy as _np
        import shutil
        from crisp_drl.agents.shared.algorithm_config import Config
        from crisp_drl.envs import make_env as make_env_mod, make_rew
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        config = Config()
        config.actor_output_dim = 3
        config.actor_nonvision_input_dim = 18
        config.max_action = _np.array([0.00025, 0.00025, _np.deg2rad(0.5)])

        repo_id = "test_3dof_rz/dry_run"
        data_dir = Path("/tmp") / "test_3dof_rz_dry_run"
        if data_dir.exists():
            shutil.rmtree(data_dir)
        #data_dir.mkdir(parents=True, exist_ok=True)

        features = {
            "observation.state.cartesian": {"shape": (6,), "dtype": "float32"},
            "observation.state.joints": {"shape": (7,), "dtype": "float32"},
            "observation.velocity.cartesian": {"shape": (3,), "dtype": "float32"},
            "observation.velocity.angular": {"shape": (3,), "dtype": "float32"},
            "observation.error.cartesian": {"shape": (3,), "dtype": "float32"},
            "observation.error.angular": {"shape": (3,), "dtype": "float32"},
            "observation.previous.action": {"shape": (6,), "dtype": "float32"},
            "observation.previous.error.cartesian": {"shape": (3,), "dtype": "float32"},
            "observation.previous.error.angular": {"shape": (3,), "dtype": "float32"},
            "observation.state.sensors_bota_ft_sensor": {"shape": (6,), "dtype": "float32"},
            "action": {"shape": (3,), "dtype": "float32", "names": ["dx", "dy", "drz"]},
            "perfect_action": {"shape": (3,), "dtype": "float32", "names": ["dx", "dy", "drz"]},
            "grasp_delta": {"shape": (3,), "dtype": "float32"},
            "goal_position_delta": {"shape": (2,), "dtype": "float32"},
            "goal_rotation_z": {"shape": (1,), "dtype": "float32"},
            "start_rotation_z": {"shape": (1,), "dtype": "float32"},
            "reward": {"shape": (1,), "dtype": "float32"},
            "success": {"shape": (1,), "dtype": "bool"},
            "is_terminal": {"shape": (1,), "dtype": "bool"},
            "observation.features.wrist_camera": {
                "shape": (config.vision_head_input_dim,), "dtype": "float32"
            },
            "observation.formatted": {
                "shape": (config.vision_head_input_dim + config.actor_nonvision_input_dim,),
                "dtype": "float32",
            },
        }

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=config.control_frequency,
            features=features,
            robot_type="franka",
            root=data_dir,
            use_videos=True,
        )

        # simple arg namespace
        class _Args:
            no_ft_sensor = no_ft
            eval = False
            mp_backend = "quintic"

        env = make_env_mod.create_real_env_v4_3dof_rz(config, args=_Args())
        iw = _unwrap_insertion(env)

        reward_fn = make_rew.create_real_reward_fn(
            config,
            event_reward_map={
                "E_SUCCESS": 0.1 / (1 - config.gamma) * 3,
                "E_FAIL": -0.1 / (1 - config.gamma) / 2 * 3,
                "E_SAFETY_BOX_VIOLATION": 0.0,
                "E_ROLLOUT_UNUSABLE": -100,
                "E_CONTROLLER_ISSUE": -100,
                "E_TORQUE": -100,
            },
        )

        def _wrap_pi(a):
            return float((a + _np.pi) % (2 * _np.pi) - _np.pi)

        ideal_goal_rz = 0.0
        pe_ratio = _np.deg2rad(2) / 0.001
        pa_mag = 0.00025
        pa_mag_rz = pa_mag * pe_ratio

        for ep_i in range(N_TEST):
            obs, reset_info = env.reset()
            grasp_delta = reset_info["reset.grasped.delta_estimated"]
            goal_pos = config.goal_position_ground_truth[:2].copy()
            goal_pos[0] += grasp_delta[0]
            goal_position_delta = reset_info["reset.goal_position.offset"][:2]
            goal_rz = float(reset_info["reset.goal_orientation.rotation_z"])
            start_rz = float(reset_info.get("reset.start_orientation.rotation_z", 0.0))

            all_actions, all_rewards, all_perfect, all_obs, all_infos = [], [], [], [obs], [reset_info]

            while True:
                cart = obs["observation.state.cartesian"]
                pa_xy = goal_pos - cart[:2]
                n = _np.linalg.norm(pa_xy)
                pa_xy = pa_xy / n * pa_mag if n > 1e-9 else _np.zeros(2)
                pa_rz_raw = _wrap_pi(ideal_goal_rz - float(cart[5]))
                pa_rz = float(_np.sign(pa_rz_raw)) * min(abs(pa_rz_raw), pa_mag_rz) if abs(pa_rz_raw) > 1e-6 else 0.0
                perfect = _np.array([pa_xy[0], pa_xy[1], pa_rz], dtype=_np.float32)
                all_perfect.append(perfect)
                action = perfect.copy()
                all_actions.append(action)

                obs, reward, terminated, truncated, info = env.step(action)
                all_obs.append(obs)
                all_infos.append(info)
                all_rewards.append(float(reward))
                if terminated or truncated:
                    break

            # terminal perfect action
            cart = obs["observation.state.cartesian"]
            pa_xy = goal_pos - cart[:2]
            n = _np.linalg.norm(pa_xy)
            pa_xy = pa_xy / n * pa_mag if n > 1e-9 else _np.zeros(2)
            pa_rz_raw = _wrap_pi(ideal_goal_rz - float(cart[5]))
            pa_rz = float(_np.sign(pa_rz_raw)) * min(abs(pa_rz_raw), pa_mag_rz) if abs(pa_rz_raw) > 1e-6 else 0.0
            all_perfect.append(_np.array([pa_xy[0], pa_xy[1], pa_rz], dtype=_np.float32))

            all_actions, all_obs, all_rewards, all_infos = reward_fn(
                all_actions, all_obs, all_rewards, all_infos
            )

            last_events = [e[1] for e in all_infos[-1].get("custom_events", [])]
            if "E_ROLLOUT_UNUSABLE" in last_events or "E_CONTROLLER_ISSUE" in last_events:
                print(f"  Episode {ep_i}: unusable, skipping save")
                continue

            successful = "E_SUCCESS" in last_events
            all_actions.append(_np.full(3, _np.nan, dtype=_np.float32))
            all_rewards = [float("nan")] + all_rewards

            for frame_idx, obs_f in enumerate(all_obs):
                frame_data = {
                    "observation.state.cartesian": obs_f["observation.state.cartesian"].astype(_np.float32),
                    "observation.state.joints": obs_f["observation.state.joints"].astype(_np.float32),
                    "observation.velocity.cartesian": obs_f["observation.velocity.cartesian"].astype(_np.float32),
                    "observation.velocity.angular": obs_f["observation.velocity.angular"].astype(_np.float32),
                    "observation.error.cartesian": obs_f["observation.error.cartesian"].astype(_np.float32),
                    "observation.error.angular": obs_f["observation.error.angular"].astype(_np.float32),
                    "observation.previous.action": obs_f["observation.previous.action"].astype(_np.float32),
                    "observation.previous.error.cartesian": obs_f["observation.previous.error.cartesian"].astype(_np.float32),
                    "observation.previous.error.angular": obs_f["observation.previous.error.angular"].astype(_np.float32),
                    "observation.state.sensors_bota_ft_sensor": obs_f["observation.state.sensors_bota_ft_sensor"].astype(_np.float32),
                    "observation.features.wrist_camera": _np.array(obs_f["observation.features.wrist_camera"]).astype(_np.float32),
                    "observation.formatted": obs_f["observation.formatted"].cpu().numpy().astype(_np.float32),
                    "action": all_actions[frame_idx].astype(_np.float32),
                    "perfect_action": all_perfect[frame_idx].astype(_np.float32),
                    "reward": _np.array([all_rewards[frame_idx]], dtype=_np.float32),
                    "success": _np.array([successful and frame_idx == len(all_obs) - 1], dtype=bool),
                    "is_terminal": _np.array([terminated and frame_idx == len(all_obs) - 1], dtype=bool),
                    "grasp_delta": grasp_delta.astype(_np.float32),
                    "goal_position_delta": goal_position_delta.astype(_np.float32),
                    "goal_rotation_z": _np.array([goal_rz], dtype=_np.float32),
                    "start_rotation_z": _np.array([start_rz], dtype=_np.float32),
                }
                dataset.add_frame(frame=frame_data, task="collect_data",
                                  timestamp=frame_idx / config.control_frequency)
            dataset.save_episode()
            print(f"  Episode {ep_i}: length={len(all_obs)} success={successful}")

        dataset._wait_image_writer()
        dataset.stop_image_writer()

        # verify saved dataset
        loaded = LeRobotDataset(repo_id=repo_id, root=data_dir)
        sample = loaded[0]
        ok &= _ok("saved action.shape == (3,)",
                  tuple(sample["action"].shape) == (3,), str(sample["action"].shape))
        ok &= _ok("saved perfect_action.shape == (3,)",
                  tuple(sample["perfect_action"].shape) == (3,),
                  str(sample["perfect_action"].shape))
        ok &= _ok("angular obs fields not all-zero",
                  not _np.allclose(sample["observation.velocity.angular"].numpy(), 0),
                  "velocity.angular")

        env.reset(options={"last_reset": True})
        env.close()

    except Exception as e:
        ok &= _ok("T6 completed without exception", False, str(e))
        import traceback
        traceback.print_exc()

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--tests", nargs="+",
        choices=["T0", "T1", "T2", "T3", "T4", "T5", "T6"],
        default=["T0", "T1", "T2", "T3", "T4", "T5"],
        help="Which tests to run (default: T0-T5; T6 runs full data-collection loop).",
    )
    parser.add_argument("--no_ft_sensor", action="store_true",
                        help="Pass through to env factory (zero FT, no contact phase).")
    args = parser.parse_args()

    run = set(args.tests)
    results: dict[str, bool] = {}

    # T0 — no hardware
    if "T0" in run:
        results["T0"] = test_T0_unit()

    # T1–T5 share one env (create once, reset between tests)
    needs_env = run & {"T1", "T2", "T3", "T4", "T5"}
    env = config = None
    if needs_env:
        print("\nCreating env (connects to robot)…")

        class _Args:
            no_ft_sensor = args.no_ft_sensor
            eval = False
            mp_backend = "quintic"

        env, config = _make_env(_Args())

        try:
            if "T1" in run:
                results["T1"] = test_T1_smoke(env, config)
            if "T2" in run:
                input("\n[Press Enter to run T2 — pure rotation]")
                results["T2"] = test_T2_pure_rotation(env)
            if "T3" in run:
                input("\n[Press Enter to run T3 — pure XY]")
                results["T3"] = test_T3_pure_xy(env)
            if "T4" in run:
                input("\n[Press Enter to run T4 — angular safety box]")
                results["T4"] = test_T4_angular_safety_box(env)
            if "T5" in run:
                input("\n[Press Enter to run T5 — FT independence]")
                results["T5"] = test_T5_ft_independence(env)
        finally:
            if env is not None:
                try:
                    env.reset(options={"last_reset": True})
                    env.close()
                except Exception as e:
                    print(f"  [warn] env.close() raised: {e}")

    if "T6" in run:
        input("\n[Press Enter to run T6 — data-collection dry run (3 episodes)]")
        results["T6"] = test_T6_data_collection(no_ft=args.no_ft_sensor)

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    all_pass = True
    for t, passed in sorted(results.items()):
        status = "PASS" if passed else "FAIL"
        print(f"  {t}: {status}")
        if not passed:
            all_pass = False
    print("=" * 50)
    print("Overall:", "ALL PASS" if all_pass else "FAILURES — see above")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

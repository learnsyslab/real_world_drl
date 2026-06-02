"""One-shot PE calibration diagnostic for the Siemens insertion task.

Goal
----
Symptom: the PE visualization (camera-frame overlay) looks correct, but the
gripper does not actually reach the estimated object center. This script
isolates which step of the world-frame chain is biased.

It does NOT modify any code or config. It:

  1. Drives the gripper to ``SiemensConfig.grasp_position_ground_truth`` with
     ``env.unwrapped.move_to`` (controller-switching cartesian move).
  2. Lets the operator place / nudge the object so the gripper is centered
     on it (manual confirmation).
  3. Records the actual TCP cartesian ``tcp_at_grasp`` at that moment.
  4. Lifts and homes to ``custom_home_position_pe`` without moving the object.
  5. Runs ``wrapper.estimate_world_pose_once`` to get ``world_D_obj_live``.
  6. Computes the chain residual using the SAME formula
     ``InsertionWrapperSiemensPE`` uses at runtime:

         o_offset_cfg        = demo_R^T @ (grasp_pos_cfg - demo_t)
         predicted_grasp_cfg = world_D_obj_live[:3,3]
                              + world_D_obj_live[:3,:3] @ o_offset_cfg
         residual_world      = predicted_grasp_cfg - tcp_at_grasp   # mm/axis

     and the delta between live PE object center and configured demo center.
  7. Prints a self-consistent replacement pair (``_NEW``) ready to paste
     into ``SiemensConfig``.

Optionally (``--also_hover``) repeats the PE from the 2 cm hover used inside
``reset()`` to expose pose-dependent biases (extrinsic-calibration drift).

Interpretation guide is printed at the end.

Example
-------
    pixi run -e jazzy python scripts/diagnose_pe_calibration.py \\
        --pose_viz_dir pe_viz/pe_calib_$(date +%Y%m%d_%H%M%S) \\
        [--also_hover]
"""

import argparse
import logging
import os
import time

import numpy as np

from crisp_gym.envs.env_wrapper import (
    ActionTimeStampWrapper,
    LastObservationWrapper,
)
from crisp_gym.envs.manipulator_env import make_env
from crisp_drl.agents.shared.algorithm_config import Config
from crisp_drl.agents.shared.env_wrappers import (
    NoGripperActionWrapper,
    ZeroFTInjectorWrapper,
)
from crisp_drl.agents.shared.insertion_env_config import SiemensConfig
from crisp_drl.agents.shared.insertion_wrapper import SensorTareWrapper
from crisp_drl.agents.shared.insertion_wrapper_s import InsertionWrapperSiemensPE


def _find_pe_wrapper(env):
    """Walk the wrapper chain until an InsertionWrapperSiemensPE is found."""
    cur = env
    while cur is not None:
        if isinstance(cur, InsertionWrapperSiemensPE):
            return cur
        cur = getattr(cur, "env", None)
    raise RuntimeError("InsertionWrapperSiemensPE not found in wrapper chain")


def _build_diagnostic_env(args, alg_config, env_config):
    """Minimal wrapper chain that exposes InsertionWrapperSiemensPE.

    Skips the heavy Dino encoder / classifier / motion planner that
    ``create_real_env_s1_pe`` adds for training — none of those are needed for
    a PE calibration probe, and the classifier would require a checkpoint we
    don't have here.
    """
    no_ft = bool(getattr(args, "no_ft_sensor", False))
    env = make_env("my_env_v4_no_ft" if no_ft else "my_env_v4")
    print("[diag] env created, waiting until ready...")
    env.wait_until_ready()
    print("[diag] env ready.")

    env = ActionTimeStampWrapper(env)
    env = NoGripperActionWrapper(env)
    env = LastObservationWrapper(env)
    if no_ft:
        env = ZeroFTInjectorWrapper(env)
    else:
        env = SensorTareWrapper(
            env,
            sensor_key="observation.state.sensors_bota_ft_sensor",
            sensor_data_shape=(6,),
        )
    env = InsertionWrapperSiemensPE(
        env,
        alg_config=alg_config,
        env_config=env_config,
        safety_box_radius=0.003,
        safety_box_step_size=0.0004,
        step_limit=2 * env_config.episode_length,
        use_ft_controller=not no_ft,
        use_6dof_grasp=bool(getattr(args, "use_6dof_grasp", False)),
        pose_viz_dir=getattr(args, "pose_viz_dir", None),
    )
    return env


def _prompt(msg: str) -> None:
    try:
        input(f"[diag] {msg} (press Enter)... ")
    except EOFError:
        pass


def _settle(wrapper, n_steps: int = 5) -> dict:
    """Refresh observations WITHOUT going through the wrapper's policy step.

    ``InsertionWrapperSiemensPE.step`` runs FT/safety/goal logic that needs
    state seeded by ``reset()`` — which we deliberately never call here. So
    we step the inner env (the chain below the PE wrapper) directly with a
    zero action and stash the result on ``wrapper.obs`` so that the next
    ``estimate_world_pose_once`` call sees a valid observation.
    """
    obs = None
    for _ in range(n_steps):
        obs, *_ = wrapper.env.step(np.zeros(6))
    wrapper.obs = obs
    return obs


def _format_mm(v: np.ndarray, n_decimals: int = 2) -> str:
    return np.array2string(
        np.round(np.asarray(v, dtype=np.float64) * 1000.0, n_decimals),
        separator=", ",
    )


def _print_residual(label: str, tcp_at_grasp: np.ndarray, world_D_obj: np.ndarray, cfg):
    cfg_R = cfg.demo_w_D_w_o[:3, :3]
    cfg_t = cfg.demo_w_D_w_o[:3, 3]
    o_offset_cfg = cfg_R.T @ (cfg.grasp_position_ground_truth - cfg_t)
    predicted_grasp_cfg = world_D_obj[:3, 3] + world_D_obj[:3, :3] @ o_offset_cfg
    residual_world = predicted_grasp_cfg - tcp_at_grasp
    delta_obj_center = world_D_obj[:3, 3] - cfg_t

    print(f"\n========= residuals from {label} =========")
    print(f"  o_T_o_tcpgrasp (config)        [mm xyz]: {_format_mm(o_offset_cfg)}")
    print(f"  world PE object center (live)  [m  xyz]: {world_D_obj[:3, 3]}")
    print(f"  world predicted grasp target   [m  xyz]: {predicted_grasp_cfg}")
    print(f"  actual TCP at grasp moment     [m  xyz]: {tcp_at_grasp}")
    print(f"  RESIDUAL  predicted - actual   [mm xyz]: {_format_mm(residual_world)}")
    print(f"           ||residual||                  : {np.linalg.norm(residual_world) * 1000.0:.2f} mm")
    print(f"  Δ obj center vs. demo_w_D_w_o  [mm xyz]: {_format_mm(delta_obj_center)}")
    return residual_world, predicted_grasp_cfg


def _print_replacement_block(tcp_at_grasp: np.ndarray, world_D_obj: np.ndarray):
    R = world_D_obj[:3, :3]
    t = world_D_obj[:3, 3]
    # Sanity: by construction, the residual is zero when these replace cfg.
    print("\n========= self-consistent replacement pair =========")
    print("# Paste into crisp_drl/agents/shared/insertion_env_config.py SiemensConfig")
    print("# (replaces the current grasp_position_ground_truth + demo_w_D_w_o).")
    print()
    print("    grasp_position_ground_truth: np.ndarray = field(")
    print("        default_factory=lambda: np.array(")
    print(f"            [{tcp_at_grasp[0]:.8f}, {tcp_at_grasp[1]:.8f}, {tcp_at_grasp[2]:.8f}]")
    print("        )")
    print("    )")
    print()
    print("    demo_w_D_w_o: np.ndarray = field(")
    print("        default_factory=lambda: np.array(")
    print("            [")
    for row in range(3):
        print(
            f"                [{R[row, 0]:.8f}, {R[row, 1]:.8f}, "
            f"{R[row, 2]:.8f}, {t[row]:.8f}],"
        )
    print("            ]")
    print("        )")
    print("    )")
    # Re-derived sanity check:
    o_offset_new = R.T @ (tcp_at_grasp - t)
    predicted_new = t + R @ o_offset_new
    print()
    print(f"# sanity: with these values, predicted - actual = "
          f"{_format_mm(predicted_new - tcp_at_grasp)} mm (≈ 0 by construction)")


def _print_interpretation(residual_home: np.ndarray, residual_hover: np.ndarray | None):
    print("\n========= interpretation guide =========")
    n_home = float(np.linalg.norm(residual_home)) * 1000.0
    print(f"  ||residual @ home-PE||  = {n_home:.2f} mm")
    if residual_hover is not None:
        n_hover = float(np.linalg.norm(residual_hover)) * 1000.0
        delta = float(np.linalg.norm(residual_hover - residual_home)) * 1000.0
        print(f"  ||residual @ hover-PE|| = {n_hover:.2f} mm")
        print(f"  ||Δ residual (home→hover)|| = {delta:.2f} mm")

    print()
    if n_home < 5.0:
        print("  → Chain is healthy (<5 mm, at PE noise floor). Missed grasps are")
        print("    likely something else: gripper slip on contact, object moving")
        print("    under finger, or controller wind-up. PE / config are fine.")
    elif residual_hover is not None and (
        float(np.linalg.norm(residual_hover - residual_home)) * 1000.0 > 5.0
    ):
        print("  → Residual changes >5 mm between home-PE and hover-PE viewpoints.")
        print("    This points at TCP_R_CAM / TCP_T_TCP_CAM (extrinsics in")
        print("    crisp_drl/envs/pose_estimation_helper.py:40-61) being biased, not")
        print("    the demo pair. The replacement block above will NOT fix this —")
        print("    extrinsics need a separate fix (out of scope).")
    else:
        print("  → Demo pair (grasp_position_ground_truth, demo_w_D_w_o) is")
        print("    inconsistent. Paste the replacement block above into SiemensConfig,")
        print("    then re-run this script: ||residual|| should collapse to ~PE noise.")


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="One-shot PE calibration diagnostic for Siemens insertion.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--pose_viz_dir",
        type=str,
        default=None,
        help="If set, the InsertionWrapperSiemensPE will dump PE overlays here "
        "(same artifacts as a normal eval run).",
    )
    parser.add_argument(
        "--no_ft_sensor",
        action="store_true",
        help="Run without the BOTA FT sensor (matches eval CLI).",
    )
    parser.add_argument(
        "--use_6dof_grasp",
        action="store_true",
        help="Initialize the wrapper in 6-DoF mode (only matters for sanity prints).",
    )
    parser.add_argument(
        "--also_hover",
        action="store_true",
        help="After the home-PE measurement, lift the gripper to the same 2 cm "
        "object-frame hover used inside reset() and re-run PE. Reveals "
        "pose-dependent biases (extrinsic-calibration drift).",
    )
    parser.add_argument(
        "--lift_z",
        type=float,
        default=0.05,
        help="Meters to lift in +Z after the operator confirms the grasp pose "
        "(before homing to PE viewpoint). Default 0.05 m.",
    )
    args = parser.parse_args()
    args.eval = True  # so the wrapper takes the eval branch where applicable
    args.cli_training = False
    args.resume_training = None
    args.pre_train = None
    args.load_policy = None
    args.expert_buffer_path = None

    if args.pose_viz_dir is not None:
        os.makedirs(args.pose_viz_dir, exist_ok=True)

    alg_config = Config()
    env_config = SiemensConfig()
    print("[diag] grasp_position_ground_truth (cfg) =", env_config.grasp_position_ground_truth)
    print("[diag] demo_w_D_w_o[:,3] (cfg)          =", env_config.demo_w_D_w_o[:3, 3])

    env = _build_diagnostic_env(args, alg_config, env_config)
    wrapper = _find_pe_wrapper(env)

    # --- Step 1: home to PE viewpoint first, so move_to has a clean start. ---
    print("\n[diag] Step 1/5: opening gripper and homing to custom_home_position_pe...")
    env.unwrapped.gripper.open()
    time.sleep(1.0)
    env.unwrapped.home(home_config=env_config.custom_home_position_pe)

    # Pull a fresh obs so we have observation.state.cartesian.
    obs = _settle(wrapper, n_steps=2)

    # --- Step 2: drive to the configured GT grasp position. ---
    print("\n[diag] Step 2/5: moving to SiemensConfig.grasp_position_ground_truth via move_to...")
    grasp_pos_cfg = np.asarray(env_config.grasp_position_ground_truth, dtype=np.float64)
    print(f"        target XYZ = {grasp_pos_cfg}")
    try:
        env.unwrapped.move_to(position=grasp_pos_cfg, speed=0.03)
    except Exception as e:
        print(f"[diag] move_to to grasp_position_ground_truth FAILED: {e!r}")
        print("[diag] Aborting — fix the target or controller before re-running.")
        return

    obs = _settle(wrapper, n_steps=5)

    # --- Step 3: operator confirms the gripper is centered on the object. ---
    print("\n[diag] Step 3/5: place / nudge the object so the gripper would close")
    print("       cleanly on it at the current TCP pose. Do NOT move the robot.")
    print("       If the gripper looks off-center on the object, manually slide the")
    print("       object to center it under the fingers (the diagnostic measures")
    print("       residual relative to the OBJECT, not the cfg value).")
    _prompt("when the object is centered under the gripper")

    tcp_at_grasp = np.asarray(
        wrapper.obs["observation.state.cartesian"][:3], dtype=np.float64
    ).copy()
    print(f"[diag] tcp_at_grasp = {tcp_at_grasp}")
    drift_vs_cfg = tcp_at_grasp - grasp_pos_cfg
    print(f"[diag] drift from cfg target [mm xyz]: {_format_mm(drift_vs_cfg)}")

    # --- Step 4: lift and home to PE viewpoint without moving the object. ---
    print(f"\n[diag] Step 4/5: lifting +{args.lift_z * 1000:.0f} mm, then homing to custom_home_position_pe...")
    lift_target = tcp_at_grasp + np.array([0.0, 0.0, args.lift_z])
    try:
        env.unwrapped.move_to(position=lift_target, speed=0.03)
    except Exception as e:
        print(f"[diag] WARNING: move_to lift failed ({e!r}); proceeding to home() anyway.")
    env.unwrapped.home(home_config=env_config.custom_home_position_pe)
    _settle(wrapper, n_steps=2)

    # --- Step 5: PE at home viewpoint. ---
    print("\n[diag] Step 5/5: running PE at home viewpoint...")
    world_D_obj_home, details_home = wrapper.estimate_world_pose_once(return_details=True)
    wrapper.validate_world_pose_transform(world_D_obj_home, "diag-home")

    residual_home, _ = _print_residual(
        "home-PE viewpoint", tcp_at_grasp, world_D_obj_home, env_config
    )

    residual_hover = None
    if args.also_hover:
        # Reproduce the 2 cm object-frame hover used in reset() ([:1406-1408]).
        # The hover is 2 cm above the *configured* grasp offset rotated by the
        # live object rotation; we use the same expression as the wrapper.
        print("\n[diag] (--also_hover) moving to 2 cm object-frame hover above coarse grasp...")
        o_offset_cfg = env_config.demo_w_D_w_o[:3, :3].T @ (
            env_config.grasp_position_ground_truth - env_config.demo_w_D_w_o[:3, 3]
        )
        coarse_grasp_world = (
            world_D_obj_home[:3, 3] + world_D_obj_home[:3, :3] @ o_offset_cfg
        )
        coarse_hover_world = (
            coarse_grasp_world
            + world_D_obj_home[:3, :3] @ np.array([0.0, 0.0, 0.02])
        )
        try:
            env.unwrapped.move_to(position=coarse_hover_world, speed=0.03)
        except Exception as e:
            print(f"[diag] WARNING: move_to hover failed ({e!r}); skipping hover-PE.")
        else:
            _settle(wrapper, n_steps=2)
            world_D_obj_hover, _ = wrapper.estimate_world_pose_once(return_details=True)
            wrapper.validate_world_pose_transform(world_D_obj_hover, "diag-hover")
            residual_hover, _ = _print_residual(
                "hover-PE viewpoint", tcp_at_grasp, world_D_obj_hover, env_config
            )

    _print_replacement_block(tcp_at_grasp, world_D_obj_home)
    _print_interpretation(residual_home, residual_hover)

    # Bring the robot back to a safe pose so the operator isn't left at hover.
    print("\n[diag] returning to custom_home_position_pe...")
    try:
        env.unwrapped.home(home_config=env_config.custom_home_position_pe)
    except Exception as e:
        print(f"[diag] WARNING: final home() failed: {e!r}")

    print("[diag] done.")


if __name__ == "__main__":
    main()

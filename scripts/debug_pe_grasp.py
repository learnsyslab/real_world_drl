"""Interactive PE → grasp debug script.

Exercises only the grasp phase of the lego_3dof_rz_pe pipeline so we can sweep
calibration knobs (o_T_o_tcpgrasp_lavender shift, yaw offset, hand-eye X shift)
without running the full SAC eval loop.

Layout intentionally mirrors crisp_drl/scripts/calibrate_lego_2x4.py — same env,
same keyboard pattern, same PE helper. Math comes from
crisp_drl/agents/shared/insertion_wrapper_lego_3dof_rz_pe.py:1019, 1081, 1109-1112.

Workflow per attempt:
    h        home + open gripper
    t        servo to refined-PE hover above grasp_position_ground_truth
    p        run PE → print brick pose, applied o_T_o, commanded grasp_xy/yaw,
             and brick-frame projection of TCP (which axis carries the bias)
    g        execute hover → descend → close → lift  (skips close in --dry-run)
    o        open gripper
    +/-      nudge oToto-shift x by ±0.5 mm at runtime
    [ / ]    nudge yaw offset by ±1°
    q/Enter  quit
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import tifffile
from pynput import keyboard

from crisp_gym.envs.manipulator_env import make_env

# Adjust hand-eye BEFORE importing PoseEstimationHelper if requested
# (TCP_T_TCP_CAM is computed at module import time).
def _parse_args():
    p = argparse.ArgumentParser(description="PE → grasp calibration sandbox")
    p.add_argument(
        "--oToto-shift-mm",
        default="0,0,0",
        help='Brick-frame shift "x,y,z" (mm) added to o_T_o_tcpgrasp_lavender.',
    )
    p.add_argument("--grasp-z-offset", type=float, default=0.0)
    p.add_argument("--yaw-offset-deg", type=float, default=0.0)
    p.add_argument(
        "--force-yaw-deg",
        type=float,
        default=None,
        help="If set, bypass 180°-symmetric snap and use this yaw directly.",
    )
    p.add_argument(
        "--hand-eye-x-shift-mm",
        type=float,
        default=0.0,
        help="Add this mm to TCP_T_TCP_CAM[0] (gripper-X). Comment in "
             "pose_estimation_helper.py:48 notes 1.3 mm baked in already.",
    )
    p.add_argument("--dry-run", action="store_true", help="Hover only, don't close.")
    p.add_argument("--viz-dir", default=None)
    p.add_argument("--brick-color", default="lavender")
    p.add_argument(
        "--env-name", default="my_env_v4",
        help="manipulator_env name (matches calibrate_lego_2x4.py).",
    )
    return p.parse_args()


ARGS = _parse_args()

from crisp_drl.envs import pose_estimation_helper as _peh  # noqa: E402

if abs(ARGS.hand_eye_x_shift_mm) > 1e-9:
    _peh.TCP_T_TCP_CAM = _peh.TCP_T_TCP_CAM.copy()
    _peh.TCP_T_TCP_CAM[0] += ARGS.hand_eye_x_shift_mm / 1000.0
    print(f"[hand-eye] TCP_T_TCP_CAM[0] now = {_peh.TCP_T_TCP_CAM[0]:.6f} m "
          f"(shift {ARGS.hand_eye_x_shift_mm:+.2f} mm)")

from crisp_drl.envs.pose_estimation_helper import PoseEstimationHelper  # noqa: E402
from crisp_drl.agents.shared.insertion_env_config import LegoConfig2x4  # noqa: E402


# ---------------------------------------------------------------------- #
# Geometry helpers (mirror wrapper math at lines 319-332, 1019, 1109-1112)
# ---------------------------------------------------------------------- #
def _wrap(a: float) -> float:
    return ((a + np.pi) % (2 * np.pi)) - np.pi


def nearest_symmetric_yaw(brick_yaw: float, current_rz: float) -> float:
    y0 = _wrap(brick_yaw)
    y1 = _wrap(brick_yaw + np.pi)
    if abs(_wrap(y0 - current_rz)) <= abs(_wrap(y1 - current_rz)):
        return y0
    return y1


def yaw_from_world_pose(world_pose: np.ndarray) -> float:
    # Matches wrapper line 1019: SAM3 X = brick short axis → −90° offset.
    return np.arctan2(world_pose[1, 0], world_pose[0, 0]) - np.pi / 2


def parse_shift_mm(s: str) -> np.ndarray:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"--oToto-shift-mm expects 3 comma-separated values, got {s!r}")
    return np.array(parts) / 1000.0  # → metres


# ---------------------------------------------------------------------- #
# Env + PE setup                                                         #
# ---------------------------------------------------------------------- #
cfg = LegoConfig2x4()
GRASP_POS = np.array(cfg.grasp_position_ground_truth, dtype=np.float64)
DEMO_LAV = np.array(cfg.demo_grasped_pose_lavender, dtype=np.float64)
LAV_R = DEMO_LAV[:3, :3]
LAV_T = DEMO_LAV[:3, 3]
O_T_O_BASE = LAV_R.T @ (GRASP_POS - LAV_T)  # 3-vec, brick frame, meters
BRICK_LONG_M = 0.0318  # 2x4 ≈ 4×7.95 mm pitch
BRICK_SHORT_M = 0.0158  # 2x4 ≈ 2×7.95 mm pitch

print(f"[cfg] grasp_position_ground_truth = {GRASP_POS}")
print(f"[cfg] demo lav_t (world)           = {LAV_T}")
print(f"[cfg] o_T_o BASE  (brick frame mm) = {np.round(O_T_O_BASE * 1000, 3)}")

env = make_env(ARGS.env_name)
print(f"[env] created ({ARGS.env_name})")
env.wait_until_ready()
env.gripper.open()
env.reset()
print("[env] reset done — skipping joint home (will cartesian-servo to wide PE pose)")

pe_helper = PoseEstimationHelper(
    assumed_orientation=np.array([]),
    lock_orientation=False,
    brick_size=cfg.brick_size,
    use_tracker=False,
)
print("[pe] helper ready")

renderer = None
if ARGS.viz_dir is not None:
    os.makedirs(ARGS.viz_dir, exist_ok=True)
    try:
        from crisp_drl.envs.pose_visualizer import PoseOverlayRenderer
        mesh_path = None
        for cand in (
            f"/workspaces/isaac_ros-dev/lego_assets/lego_{cfg.brick_size}_{ARGS.brick_color}_up.obj",
            os.path.expanduser(
                f"~/workspaces/isaac_ros-dev/lego_assets/lego_{cfg.brick_size}_{ARGS.brick_color}_up.obj"
            ),
        ):
            if os.path.exists(cand):
                mesh_path = cand
                break
        renderer = PoseOverlayRenderer(
            camera_info_json_path="camera_parameters/realsense_d405_single.json",
            mesh_path=mesh_path,
            use_default_mesh_fallback=False,
        )
        print(f"[viz] renderer ready, mesh={mesh_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[viz] renderer unavailable ({exc}) — running without overlays.")


# ---------------------------------------------------------------------- #
# Runtime state                                                          #
# ---------------------------------------------------------------------- #
oToto_shift = parse_shift_mm(ARGS.oToto_shift_mm)
yaw_offset_rad = np.deg2rad(ARGS.yaw_offset_deg)
attempt_idx = 0
last_world_pose: np.ndarray | None = None
last_cam_pose: np.ndarray | None = None
last_grasp_yaw: float | None = None
last_grasp_xy: np.ndarray | None = None

HOVER_Z = GRASP_POS[2] + ARGS.grasp_z_offset + 0.015
GRASP_Z = GRASP_POS[2] + ARGS.grasp_z_offset
DRY_HOVER_Z = GRASP_Z + 0.005
WIDE_PE_XYZ = np.array(cfg.demo_goal_pose_estimation_euler[:3], dtype=np.float64)

print(
    f"[cfg] HOVER_Z={HOVER_Z:.4f}  GRASP_Z={GRASP_Z:.4f}  "
    f"oToto_shift_mm={oToto_shift * 1000}  yaw_offset_deg={np.rad2deg(yaw_offset_rad):.2f}"
)


# ---------------------------------------------------------------------- #
# Robot helpers                                                          #
# ---------------------------------------------------------------------- #
def step_zeros():
    obs, *_ = env.step(np.zeros(7))
    return obs


def current_state():
    return step_zeros()["observation.state.cartesian"]


def current_rz() -> float:
    return float(current_state()[5])


def servo_xyz(target, step_size=0.004, tol=0.001, max_steps=600):
    cur = current_state()[:3]
    n = 0
    while np.linalg.norm(target - cur) > tol and n < max_steps:
        delta = target - cur
        d = np.linalg.norm(delta)
        step = delta / d * min(step_size, d)
        obs, *_ = env.step(np.array([step[0], step[1], step[2], 0, 0, 0, 0]))
        cur = obs["observation.state.cartesian"][:3]
        n += 1
    res_mm = np.linalg.norm(target - cur) * 1000
    print(f"[servo] arrived at {cur}  residual={res_mm:.1f} mm steps={n}")


def servo_yaw(target_rz, step=np.deg2rad(0.5), tol=np.deg2rad(0.3), max_steps=400):
    cur = current_rz()
    n = 0
    while abs(_wrap(target_rz - cur)) > tol and n < max_steps:
        delta = _wrap(target_rz - cur)
        d = np.sign(delta) * min(step, abs(delta))
        obs, *_ = env.step(np.array([0, 0, 0, 0, 0, d, 0]))
        cur = float(obs["observation.state.cartesian"][5])
        n += 1
    print(f"[yaw] arrived rz={np.rad2deg(cur):.2f}°  steps={n}")


def gripper_open():
    env.step(np.array([0, 0, 0, 0, 0, 0, 0.2]))


def gripper_close():
    env.step(np.array([0, 0, 0, 0, 0, 0, -0.4]))


# ---------------------------------------------------------------------- #
# Core: PE → predicted grasp pose                                        #
# ---------------------------------------------------------------------- #
def run_pe() -> None:
    global last_world_pose, last_cam_pose, last_grasp_yaw, last_grasp_xy, attempt_idx
    obs = step_zeros()
    rgb = obs["observation.images.wrist_camera"]
    depth_raw = obs["observation.images.wrist_depth_camera"]
    depth_m = depth_raw.astype(np.float32) / 1000.0
    tcp_cart = obs["observation.state.cartesian"]

    print(f"\n[PE attempt {attempt_idx}] TCP at: {tcp_cart[:3]}")

    masks = pe_helper.segmenter.segment_lego(rgb, colors=("yellow", ARGS.brick_color))
    if ARGS.brick_color not in masks:
        print(f"[PE] ERROR: segmenter returned no '{ARGS.brick_color}' mask. keys={list(masks)}")
        return

    cam_pose = pe_helper.pose_estimator.estimate_lego(
        rgb, depth_m, masks[ARGS.brick_color], ARGS.brick_color
    )
    world_pose = pe_helper._compute_pose_in_world_frame(cam_pose, tcp_cart)
    # Flat 3DoF projection: keep Z = grasp_z baseline, zero out roll/pitch.
    yaw = yaw_from_world_pose(world_pose)
    refined_xy = world_pose[:3, 3][:2]
    print(f"[PE] brick centroid (world): xy = {refined_xy}  z = {world_pose[2, 3]:.4f}")
    print(f"[PE] brick yaw (raw, after −90°): {np.rad2deg(yaw):.2f}°")

    # Yaw choice
    if ARGS.force_yaw_deg is not None:
        grasp_yaw = np.deg2rad(ARGS.force_yaw_deg)
        print(f"[CAL] grasp_yaw FORCED = {np.rad2deg(grasp_yaw):.2f}°")
    else:
        cur_rz = current_rz()
        grasp_yaw = nearest_symmetric_yaw(yaw, cur_rz)
        print(f"[CAL] grasp_yaw (snap from cur_rz={np.rad2deg(cur_rz):.2f}°) = "
              f"{np.rad2deg(grasp_yaw):.2f}°")
    grasp_yaw = _wrap(grasp_yaw + yaw_offset_rad)

    # o_T_o + shift → world offset
    o_T_o_applied = O_T_O_BASE - oToto_shift  # shift in brick frame REDUCES applied offset
    c, s = np.cos(grasp_yaw), np.sin(grasp_yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    offset_world = Rz @ o_T_o_applied
    grasp_xy = refined_xy + offset_world[:2]

    print(f"[CAL] o_T_o BASE       (brick mm): {np.round(O_T_O_BASE * 1000, 3)}")
    print(f"[CAL] o_T_o shift       (brick mm): {np.round(oToto_shift * 1000, 3)}")
    print(f"[CAL] o_T_o APPLIED    (brick mm): {np.round(o_T_o_applied * 1000, 3)}")
    print(f"[CAL] offset_world_xy        (mm): {np.round(offset_world[:2] * 1000, 3)}")
    print(f"[CMD] grasp_xy (world): {grasp_xy}")
    print(f"[CMD] grasp_yaw       : {np.rad2deg(grasp_yaw):.2f}°")
    print(f"[CMD] grasp_z         : {GRASP_Z:.4f}  (hover {HOVER_Z:.4f})")

    # Brick-frame projection of commanded TCP for finger-side diagnosis.
    # Using PE world pose's R (3DoF projection: roll=pitch=0, yaw=grasp_yaw).
    Rz_brick = Rz  # consistent with grasp_yaw
    tcp_in_world = np.array([grasp_xy[0], grasp_xy[1], GRASP_Z])
    brick_center_world = np.array([refined_xy[0], refined_xy[1], world_pose[2, 3]])
    delta_world = tcp_in_world - brick_center_world
    delta_brick = Rz_brick.T @ delta_world  # vector in brick frame
    print(f"[CHK] TCP − brick (brick frame mm): "
          f"long={delta_brick[0]*1000:+.2f}  short={delta_brick[1]*1000:+.2f}  "
          f"z={delta_brick[2]*1000:+.2f}")
    print(f"[CHK] Expected (no bias): long≈0, short≈0, z≈{O_T_O_BASE[2]*1000:.1f}")
    print(f"[CHK] Brick half-dims (mm): long={BRICK_LONG_M*500:.1f}  short={BRICK_SHORT_M*500:.1f}")

    last_world_pose = world_pose
    last_cam_pose = cam_pose
    last_grasp_yaw = float(grasp_yaw)
    last_grasp_xy = grasp_xy.copy()

    # Optional viz: dump RGB + mask + cam pose for later overlay inspection.
    if ARGS.viz_dir is not None:
        path_rgb = os.path.join(ARGS.viz_dir, f"attempt_{attempt_idx:03d}_rgb.tiff")
        path_npz = os.path.join(ARGS.viz_dir, f"attempt_{attempt_idx:03d}.npz")
        tifffile.imwrite(path_rgb, rgb)
        np.savez(
            path_npz,
            rgb=rgb,
            depth=depth_raw,
            mask=masks[ARGS.brick_color],
            tcp_cart=tcp_cart,
            cam_pose=cam_pose,
            world_pose=world_pose,
            grasp_yaw=grasp_yaw,
            grasp_xy=grasp_xy,
            oToto_shift=oToto_shift,
            yaw_offset_rad=yaw_offset_rad,
            o_T_o_applied=o_T_o_applied,
        )
        if renderer is not None:
            png = os.path.join(ARGS.viz_dir, f"attempt_{attempt_idx:03d}_overlay.png")
            try:
                renderer.save_single(
                    rgb=rgb,
                    mask=masks[ARGS.brick_color],
                    pose_cam=cam_pose,
                    out_path=png,
                    label=f"{ARGS.brick_color} attempt {attempt_idx}",
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[viz] overlay save failed: {exc}")
        print(f"[viz] wrote {path_rgb}, {path_npz}")
    attempt_idx += 1


def run_grasp() -> None:
    if last_grasp_xy is None or last_grasp_yaw is None:
        print("[grasp] no PE result yet — press 'p' first.")
        return
    target_yaw = last_grasp_yaw
    print(f"[grasp] aligning yaw → {np.rad2deg(target_yaw):.2f}°")
    servo_yaw(target_yaw)

    hover = np.array([last_grasp_xy[0], last_grasp_xy[1], HOVER_Z])
    print(f"[grasp] hover {hover}")
    servo_xyz(hover)

    if ARGS.dry_run:
        descent = np.array([last_grasp_xy[0], last_grasp_xy[1], DRY_HOVER_Z])
        print(f"[grasp] DRY-RUN: descending only to {descent} (no close)")
        servo_xyz(descent, step_size=0.002)
        print("[grasp] DRY-RUN complete. Inspect finger gaps, then press 'h' or 't'.")
        return

    descent = np.array([last_grasp_xy[0], last_grasp_xy[1], GRASP_Z])
    print(f"[grasp] descend {descent}")
    servo_xyz(descent, step_size=0.002, tol=0.0005)
    print("[grasp] closing gripper")
    gripper_close()
    time.sleep(0.6)
    lift = np.array([last_grasp_xy[0], last_grasp_xy[1], GRASP_Z + 0.025])
    print(f"[grasp] lift {lift}")
    servo_xyz(lift, step_size=0.002)


def go_to_pe_hover():
    target = np.array([GRASP_POS[0], GRASP_POS[1], HOVER_Z])
    print(f"[hover] servo to refined-PE hover {target}")
    servo_xyz(target)


def go_to_wide_pe_pose():
    print(f"[wide] servo to wide-PE pose {WIDE_PE_XYZ}")
    # Slightly gentler step (3 mm) on the first big move from arbitrary start.
    servo_xyz(WIDE_PE_XYZ, step_size=0.003, tol=0.002, max_steps=800)


SAFE_LIFT_Z = max(WIDE_PE_XYZ[2], 0.12)  # cartesian-only safe Z (no joint home)


def safe_home():
    """Cartesian-only 'home': open gripper → lift Z → go to wide PE pose.

    Avoids env.home() entirely. The joint-space home would swing the EE through
    the workspace and can crash if the gripper is near the table or holding a
    brick. We instead lift straight up, then translate XY to the wide PE pose.
    """
    print("[home] opening gripper...")
    gripper_open()
    time.sleep(0.4)
    cart = current_state()[:3]
    if cart[2] < SAFE_LIFT_Z:
        lift_target = np.array([cart[0], cart[1], SAFE_LIFT_Z])
        print(f"[home] lifting cartesian {cart[2]:.3f} → {SAFE_LIFT_Z:.3f} m...")
        servo_xyz(lift_target, step_size=0.003, tol=0.002, max_steps=800)
    else:
        print(f"[home] already above SAFE_LIFT_Z ({cart[2]:.3f} ≥ {SAFE_LIFT_Z}); skip lift.")
    go_to_wide_pe_pose()
    print("[home] done (no joint home was executed).")


# ---------------------------------------------------------------------- #
# Post-grasp PE: measure where fingers actually held the brick           #
# ---------------------------------------------------------------------- #
def measure_post_grasp_bias() -> None:
    """Run PE on the currently-grasped (lifted) brick. The brick is now pinned
    where the fingers physically closed on it, so brick centroid − TCP gives
    the actual mechanical bias. Projected into gripper frame, this is the
    correction to add to oToto_shift to recenter the next grasp."""
    global oToto_shift
    obs = step_zeros()
    tcp = obs["observation.state.cartesian"]
    rgb = obs["observation.images.wrist_camera"]
    depth_m = obs["observation.images.wrist_depth_camera"].astype(np.float32) / 1000.0
    print(f"\n[measure] TCP now: xy={tcp[:2]}  rz={np.rad2deg(tcp[5]):.2f}°")

    masks = pe_helper.segmenter.segment_lego(rgb, colors=("yellow", ARGS.brick_color))
    if ARGS.brick_color not in masks:
        print(f"[measure] ERROR: no '{ARGS.brick_color}' mask in grasped view. "
              f"keys={list(masks)}. Is brick visible to wrist cam?")
        return
    cam_pose = pe_helper.pose_estimator.estimate_lego(
        rgb, depth_m, masks[ARGS.brick_color], ARGS.brick_color
    )
    world_pose = pe_helper._compute_pose_in_world_frame(cam_pose, tcp)
    brick_xy = world_pose[:3, 3][:2]
    delta_world = brick_xy - tcp[:2]
    # Rotate world delta into gripper frame (gripper rz = tcp[5]).
    rz = float(tcp[5])
    c, s = np.cos(rz), np.sin(rz)
    Rz_g = np.array([[c, -s], [s, c]])
    delta_gripper = Rz_g.T @ delta_world  # gripper frame
    print(f"[measure] brick world xy:    {brick_xy}")
    print(f"[measure] tcp world   xy:    {tcp[:2]}")
    print(f"[measure] delta (world mm):  {np.round(delta_world * 1000, 3)}")
    print(f"[measure] delta (gripper mm): "
          f"X={delta_gripper[0]*1000:+.3f}  Y={delta_gripper[1]*1000:+.3f}")
    print(f"[measure] sign convention: + X = brick is FORWARD of TCP "
          f"(gripper-X). To recenter, add this delta to oToto_shift.")
    new_shift = oToto_shift + np.array([delta_gripper[0], delta_gripper[1], 0.0])
    print(f"[measure] proposed new oToto_shift_mm: {np.round(new_shift * 1000, 3)}")
    print(f"[measure] press '+'/'-' or ','/'.' to accept manually, "
          f"or call again — small residuals iterate.")


def _print_paste_block(lav_R: np.ndarray, lav_t: np.ndarray, header: str) -> None:
    o_T_o = lav_R.T @ (GRASP_POS - lav_t)
    print(f"\n========= {header} =========")
    print(f"resulting o_T_o (brick mm): {np.round(o_T_o * 1000, 3)}")
    print()
    print("=== Paste into LegoConfig2x4 demo_grasped_pose_lavender "
          "(insertion_env_config.py:169-178) ===")
    print("    demo_grasped_pose_lavender: np.ndarray = field(")
    print("        default_factory=lambda: np.array(")
    print("            [")
    for i in range(3):
        r = lav_R[i]
        t = lav_t[i]
        print(f"                [{r[0]:.8f}, {r[1]:.8f}, {r[2]:.8f}, {t:.8f}],")
    print("                [0.0, 0.0, 0.0, 1.0],")
    print("            ]")
    print("        )")
    print("    )")
    print("================================\n")


def save_calibration_block() -> None:
    """Patch the original demo translation by current oToto_shift.
    Math: lav_t_new = lav_t + lav_R @ shift_brick, so the runtime-computed
    o_T_o = lav_R.T @ (grasp_pos - lav_t_new) = o_T_o_base - shift_brick.
    Use after iterating m + shift keys to convergence."""
    lav_t_new = LAV_T + LAV_R @ oToto_shift
    print(f"[save] using current oToto_shift_mm = {np.round(oToto_shift * 1000, 3)}")
    _print_paste_block(LAV_R, lav_t_new, "PASTE BLOCK (from oToto_shift)")


def capture_calibration_from_current_grasp() -> None:
    """One-shot: take the *current physical grasp* as the new demo.
    Run PE on the lifted brick, read live TCP, compute lav_R / lav_t such that
        o_T_o_runtime = lav_R.T @ (grasp_pos_FIXED - lav_t)
                      = brick_R.T @ (tcp_world - brick_t)
    i.e. replays this exact relative pose at future grasps.
    Use when the gripper is currently holding a brick you trust is centered
    (either after iterating m → ~0 residual, or after hand-placing the brick
    perfectly between the fingers before closing)."""
    obs = step_zeros()
    tcp = obs["observation.state.cartesian"]
    rgb = obs["observation.images.wrist_camera"]
    depth_m = obs["observation.images.wrist_depth_camera"].astype(np.float32) / 1000.0
    print(f"\n[capture] TCP now (world): {tcp[:3]}  rz={np.rad2deg(tcp[5]):.2f}°")

    masks = pe_helper.segmenter.segment_lego(rgb, colors=("yellow", ARGS.brick_color))
    if ARGS.brick_color not in masks:
        print(f"[capture] ERROR: no '{ARGS.brick_color}' mask. Is brick visible?")
        return
    cam_pose = pe_helper.pose_estimator.estimate_lego(
        rgb, depth_m, masks[ARGS.brick_color], ARGS.brick_color
    )
    brick_world = pe_helper._compute_pose_in_world_frame(cam_pose, tcp)
    brick_R = brick_world[:3, :3]
    brick_t = brick_world[:3, 3]

    tcp_xyz = np.array(tcp[:3], dtype=np.float64)
    lav_R_new = brick_R
    lav_t_new = brick_t + (GRASP_POS - tcp_xyz)

    sanity = lav_R_new.T @ (GRASP_POS - lav_t_new)
    expected = brick_R.T @ (tcp_xyz - brick_t)
    print(f"[capture] sanity check (should match within rounding):")
    print(f"          lav_R_new.T @ (grasp_pos - lav_t_new) = {np.round(sanity*1000, 3)} mm")
    print(f"          brick_R.T   @ (tcp - brick_t)          = {np.round(expected*1000, 3)} mm")
    _print_paste_block(lav_R_new, lav_t_new, "PASTE BLOCK (from live grasp)")


# ---------------------------------------------------------------------- #
# Keyboard handler                                                       #
# ---------------------------------------------------------------------- #
QUIT = False


def on_press(key):
    global QUIT, oToto_shift, yaw_offset_rad
    c = getattr(key, "char", None)
    if c is None:
        if key == keyboard.Key.enter:
            QUIT = True
        return
    if c == "h":
        safe_home()
    elif c == "t":
        go_to_pe_hover()
    elif c == "p":
        try:
            run_pe()
        except Exception as exc:  # noqa: BLE001
            print(f"[PE] FAILED: {exc}")
    elif c == "g":
        try:
            run_grasp()
        except Exception as exc:  # noqa: BLE001
            print(f"[grasp] FAILED: {exc}")
    elif c == "o":
        gripper_open()
        print("[key] gripper open")
    elif c == "+":
        oToto_shift = oToto_shift + np.array([0.0005, 0.0, 0.0])
        print(f"[key] oToto_shift_mm = {oToto_shift * 1000}")
    elif c == "-":
        oToto_shift = oToto_shift - np.array([0.0005, 0.0, 0.0])
        print(f"[key] oToto_shift_mm = {oToto_shift * 1000}")
    elif c == ".":
        oToto_shift = oToto_shift + np.array([0.0, 0.0005, 0.0])
        print(f"[key] oToto_shift_mm = {oToto_shift * 1000}")
    elif c == ",":
        oToto_shift = oToto_shift - np.array([0.0, 0.0005, 0.0])
        print(f"[key] oToto_shift_mm = {oToto_shift * 1000}")
    elif c == "[":
        yaw_offset_rad -= np.deg2rad(1.0)
        print(f"[key] yaw_offset = {np.rad2deg(yaw_offset_rad):+.2f}°")
    elif c == "]":
        yaw_offset_rad += np.deg2rad(1.0)
        print(f"[key] yaw_offset = {np.rad2deg(yaw_offset_rad):+.2f}°")
    elif c == "m":
        try:
            measure_post_grasp_bias()
        except Exception as exc:  # noqa: BLE001
            print(f"[measure] FAILED: {exc}")
    elif c == "s":
        save_calibration_block()
    elif c == "c":
        try:
            capture_calibration_from_current_grasp()
        except Exception as exc:  # noqa: BLE001
            print(f"[capture] FAILED: {exc}")
    elif c == "q":
        QUIT = True


# Startup: cartesian-servo to wide PE pose (no joint home — user request).
# Robot must already be in a Cartesian-reachable, table-clear pose.
print("[startup] going to wide PE pose (no joint home)")
try:
    go_to_wide_pe_pose()
except Exception as exc:  # noqa: BLE001
    print(f"[startup] wide-PE servo failed: {exc}. "
          "Robot may be in an unreachable configuration — jog it manually.")

listener = keyboard.Listener(on_press=on_press)
listener.start()

print(
    "\nKeys:\n"
    "  h = safe home (open gripper → cartesian lift → go to wide PE pose; NO joint home)\n"
    "  t = servo to refined-PE hover\n"
    "  p = run PE on table brick, print commanded grasp\n"
    "  g = execute grasp (hover → descend → close → lift; DRY: skip close)\n"
    "  m = MEASURE post-grasp bias via PE on lifted brick (auto-suggests shift)\n"
    "  s = SAVE: paste block from current oToto_shift (after iterating m)\n"
    "  c = CAPTURE: paste block from current physical grasp (one-shot)\n"
    "  o = gripper open\n"
    "  +/- = nudge oToto_shift X by ±0.5 mm  (gripper-X / brick long-ish)\n"
    "  ,/. = nudge oToto_shift Y by ±0.5 mm\n"
    "  [/] = nudge yaw_offset by ±1°\n"
    "  q / Enter = quit\n"
    "\nCalibration loop:\n"
    "  1) h → t → p → g    (real close, lifts ~25 mm)\n"
    "  2) m                 (PE on grasped brick, reports bias)\n"
    "  3) apply suggested shift via +/-/,/.\n"
    "  4) h → t → p → g → m  (iterate until |delta_gripper| < 0.5 mm)\n"
    "  5) s                 (paste block into insertion_env_config.py)\n"
)

try:
    while not QUIT:
        time.sleep(0.05)
except KeyboardInterrupt:
    pass
finally:
    listener.stop()
    env.close()
    print("\n[done] final oToto_shift_mm:", oToto_shift * 1000,
          " yaw_offset_deg:", np.rad2deg(yaw_offset_rad))
    sys.exit(0)

"""
Full SiemensConfigDemo calibration pipeline.

Extends crisp_drl/scripts/calibrate_siemens.py: every interactively captured
field is wired through to a JSON snapshot (crash-recoverable) and finally
code-generated into a fresh

    crisp_drl/agents/shared/siemens_config_demo.py

containing a `@dataclass SiemensConfigDemo` with the same shape as
`SiemensConfig`. After this script runs once, set
`SIEMENS_USE_DEMO_CONFIG=1` and re-run `scripts/collect_data_real_lerobot_s.py`
to pick up the calibrated layout without any source edits.

Workflow (Procedure B):
  1. Jog to desired FIRST-HOME joint config, press 'u'    -> custom_first_home_position
  2. Jog to general HOME, press 'h'                       -> custom_home_position
  3. Jog to PE-HOME viewpoint, press 'j'                  -> custom_home_position_pe
  4. Jog gripper down to grasp pose on the lid, press 'g' -> grasp_position_ground_truth
                                                            grasp_orientation_ground_truth_euler
  5. Lift back to PE-home viewpoint:
       'i'  capture RGB+depth frame
       'p'  run FoundationPose                            -> demo_w_D_w_o, demo_t_D_t_o
     Repeat at a few brick yaws to sanity-check <=2 mm drift.
  6. Jog to insertion target pose, press 'f'              -> goal_position_ground_truth
                                                            goal_orientation_ground_truth_euler
  7. Jog along post-grasp trajectory, press 'v' at each:  -> waypoints_after_grasp
       (rotations stored as delta from captured grasp Euler)
     'z' undoes the last waypoint.
  8. Jog to dropoff pose, press 'b'                       -> dropoff_point
  9. Press 'P' to write siemens_config_demo.py.

Robot is held in gravity-compensation mode (env "my_env_v3_grav_comp_xyz"),
so the operator moves the arm by hand for translation. Small angle nudges via
keyboard are still supported.

Rotation keys (small nudges):
  y/x -> +/-Ry   k/l -> +/-Rz   m/n -> +/-Rx
  Y   -> +3 deg about Y in one shot

Other:
  o/c   gripper open/close
  r     print TCP + joint state
  Enter quit
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
import sys
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile
from pynput import keyboard

from crisp_gym.envs.manipulator_env import make_env
from crisp_drl.agents.shared.insertion_env_config import SiemensConfig
from crisp_drl.envs.pose_estimation_helper import PoseEstimationHelper


_SCRIPT_DIR = Path(__file__).resolve().parent
_SNAPSHOT_PATH = _SCRIPT_DIR / "siemens_calib_snapshot.json"
_DEMO_CONFIG_PATH = (
    _SCRIPT_DIR.parent / "agents" / "shared" / "siemens_config_demo.py"
)
_TEST_IMAGES_DIR = Path("test_images")
_TEST_IMAGES_DIR.mkdir(exist_ok=True)

WAYPOINT_TOL_DEFAULT = 0.001
EXP_NAME = "siemens_calib_full"


# ---------------------------------------------------------------------------
# Snapshot persistence
# ---------------------------------------------------------------------------

def _empty_snapshot() -> dict:
    return {
        "first_home_joints": None,
        "home_joints": None,
        "home_pe_joints": None,
        "grasp_pos": None,
        "grasp_euler": None,
        "goal_pos": None,
        "goal_euler": None,
        "waypoints": [],
        "dropoff": None,
        "demo_w_D_w_o": None,
        "demo_t_D_t_o": None,
        # Shared viewpoint for the combined Siemens-then-Lego runner.
        # Captures the SAME physical TCP pose in both joint and Cartesian
        # representations so each task wrapper can be commanded to it via
        # its own (different) config field.
        "combined_pe_joints": None,
        "combined_pe_cartesian": None,
    }


def _to_jsonable(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x


def save_snapshot(snapshot: dict, path: Path = _SNAPSHOT_PATH) -> None:
    payload = {k: _to_jsonable(v) for k, v in snapshot.items()}
    payload["_saved_at"] = datetime.now().isoformat()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def load_snapshot(path: Path = _SNAPSHOT_PATH) -> dict | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        print(f"[snapshot] WARNING: failed to parse {path}: {e}")
        return None
    snap = _empty_snapshot()
    for k in snap:
        if k in data and data[k] is not None:
            snap[k] = data[k]
    return snap


def maybe_resume_snapshot() -> dict:
    existing = load_snapshot()
    if existing is None:
        return _empty_snapshot()
    print(f"[snapshot] Found existing snapshot at {_SNAPSHOT_PATH}")
    for k, v in existing.items():
        if v is None or (isinstance(v, list) and len(v) == 0):
            continue
        if isinstance(v, list) and k == "waypoints":
            print(f"  {k}: {len(v)} entries")
        else:
            print(f"  {k}: <set>")
    ans = input("[snapshot] Resume from snapshot? [y/N] ").strip().lower()
    if ans == "y":
        print("[snapshot] Resumed.")
        return existing
    print("[snapshot] Starting fresh; previous snapshot will be overwritten.")
    return _empty_snapshot()


# ---------------------------------------------------------------------------
# Code generation: snapshot + SiemensConfig defaults -> SiemensConfigDemo .py
# ---------------------------------------------------------------------------

# Map snapshot key -> SiemensConfig field name(s).
_SNAPSHOT_TO_FIELD = {
    "first_home_joints": "custom_first_home_position",
    "home_joints": "custom_home_position",
    "home_pe_joints": "custom_home_position_pe",
    "grasp_pos": "grasp_position_ground_truth",
    "grasp_euler": "grasp_orientation_ground_truth_euler",
    "goal_pos": "goal_position_ground_truth",
    "goal_euler": "goal_orientation_ground_truth_euler",
    "waypoints": "waypoints_after_grasp",
    "dropoff": "dropoff_point",
    "demo_w_D_w_o": "demo_w_D_w_o",
    "demo_t_D_t_o": "demo_t_D_t_o",
}


def _type_annotation(value) -> str:
    if isinstance(value, np.ndarray):
        return "np.ndarray"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, np.integer)):
        return "int"
    if isinstance(value, (float, np.floating)):
        return "float"
    if isinstance(value, list):
        return "list"
    if isinstance(value, str):
        return "str"
    raise TypeError(f"Unsupported field type: {type(value)!r}")


def _format_scalar(v) -> str:
    if isinstance(v, bool):
        return repr(v)
    if isinstance(v, (int, np.integer)):
        return repr(int(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    if isinstance(v, str):
        return repr(v)
    raise TypeError(f"Not a scalar: {type(v)!r}")


def _format_ndarray(arr: np.ndarray) -> str:
    arr = np.asarray(arr)
    if arr.size == 0:
        if arr.ndim == 2:
            return "np.array([[]])"
        return "np.array([])"
    body = np.array2string(
        arr,
        separator=", ",
        precision=8,
        suppress_small=False,
        threshold=int(1e9),
        max_line_width=120,
        floatmode="fixed",
    )
    return f"np.array({body})"


def _format_list_of_floats(xs) -> str:
    return "[" + ", ".join(f"{float(x):.8f}" for x in xs) + "]"


def _format_list_of_numerics(xs) -> str:
    """Preserve int vs float distinction (rl_axis_indices etc. need real ints)."""
    parts = []
    for v in xs:
        if isinstance(v, bool):
            parts.append(repr(v))
        elif isinstance(v, (int, np.integer)) and not isinstance(v, bool):
            parts.append(repr(int(v)))
        else:
            parts.append(f"{float(v):.8f}")
    return "[" + ", ".join(parts) + "]"


def _format_waypoints(wps) -> str:
    if not wps:
        return "[]"
    lines = ["["]
    for entry in wps:
        pose, tol = entry[0], entry[1]
        lines.append(f"        ({_format_list_of_floats(pose)}, {float(tol):.6f}),")
    lines.append("    ]")
    return "\n".join(lines)


def _format_generic_list(xs) -> str:
    flat_numeric = all(isinstance(v, (int, float, np.integer, np.floating)) for v in xs)
    if flat_numeric:
        return _format_list_of_numerics(xs)
    return repr(xs)


def _captured_value_for_field(field_name: str, snapshot: dict):
    """Return the captured value for a SiemensConfig field, or None if absent."""
    # 1:1 snapshot keys.
    if field_name == "custom_first_home_position" and snapshot["first_home_joints"]:
        return np.asarray(snapshot["first_home_joints"], dtype=np.float64)
    if field_name == "custom_home_position" and snapshot["home_joints"]:
        return np.asarray(snapshot["home_joints"], dtype=np.float64)
    if field_name == "custom_home_position_pe" and snapshot["home_pe_joints"]:
        return np.asarray(snapshot["home_pe_joints"], dtype=np.float64)
    if field_name == "grasp_position_ground_truth" and snapshot["grasp_pos"]:
        return np.asarray(snapshot["grasp_pos"], dtype=np.float64)
    if field_name == "grasp_orientation_ground_truth_euler" and snapshot["grasp_euler"]:
        return np.asarray(snapshot["grasp_euler"], dtype=np.float64)
    if field_name == "goal_position_ground_truth" and snapshot["goal_pos"]:
        return np.asarray(snapshot["goal_pos"], dtype=np.float64)
    if field_name == "goal_orientation_ground_truth_euler" and snapshot["goal_euler"]:
        return np.asarray(snapshot["goal_euler"], dtype=np.float64)
    if field_name == "waypoints_after_grasp" and snapshot["waypoints"]:
        return snapshot["waypoints"]  # list of [pose, tol]
    if field_name == "dropoff_point" and snapshot["dropoff"]:
        return list(snapshot["dropoff"])
    if field_name == "demo_w_D_w_o" and snapshot["demo_w_D_w_o"]:
        return np.asarray(snapshot["demo_w_D_w_o"], dtype=np.float64)
    if field_name == "demo_t_D_t_o" and snapshot["demo_t_D_t_o"]:
        return np.asarray(snapshot["demo_t_D_t_o"], dtype=np.float64)
    return None


def _validate_field_parity(rendered_names: set[str]) -> None:
    expected = {f.name for f in fields(SiemensConfig)}
    missing = expected - rendered_names
    extra = rendered_names - expected
    if missing or extra:
        raise RuntimeError(
            f"SiemensConfigDemo / SiemensConfig field mismatch.\n"
            f"  Missing: {sorted(missing)}\n"
            f"  Extra:   {sorted(extra)}"
        )


def _render_field_lines(name: str, value, ann: str) -> list[str]:
    """Render one SiemensConfig field as the lines that appear inside the
    @dataclass body. Returns a list without a trailing blank line."""
    if isinstance(value, np.ndarray):
        expr = _format_ndarray(value)
        return [
            f"    {name}: {ann} = field(",
            f"        default_factory=lambda: {expr}",
            "    )",
        ]
    if isinstance(value, list):
        if name in ("waypoints_after_grasp", "waypoints_after_rl_train"):
            expr = _format_waypoints(value)
        else:
            expr = _format_generic_list(value)
        return [
            f"    {name}: {ann} = field(",
            f"        default_factory=lambda: {expr}",
            "    )",
        ]
    return [f"    {name}: {ann} = {_format_scalar(value)}"]


def write_demo_config_py(
    snapshot: dict,
    out_path: Path = _DEMO_CONFIG_PATH,
    source_snapshot_path: Path = _SNAPSHOT_PATH,
) -> tuple[Path, list[str], list[str]]:
    """Render a SiemensConfigDemo .py file.

    Returns (path, captured_field_names, captured_paste_block_lines). The
    paste-block lines contain ONLY the captured fields, formatted exactly as
    they appear in source SiemensConfig, ready to copy-paste into
    crisp_drl/agents/shared/insertion_env_config.py.
    """
    defaults = SiemensConfig()
    captured_fields: list[str] = []
    body_lines: list[str] = []
    captured_paste_lines: list[str] = []
    rendered_names: set[str] = set()

    for f in fields(SiemensConfig):
        name = f.name
        rendered_names.add(name)

        cap_val = _captured_value_for_field(name, snapshot)
        was_captured = cap_val is not None
        if was_captured:
            captured_fields.append(name)
            value = cap_val
        else:
            value = getattr(defaults, name)

        ann = _type_annotation(value)
        field_lines = _render_field_lines(name, value, ann)
        body_lines.extend(field_lines)
        body_lines.append("")  # blank line between fields

        if was_captured:
            captured_paste_lines.extend(field_lines)
            captured_paste_lines.append("")

    _validate_field_parity(rendered_names)

    # Non-field class-level attributes on SiemensConfig (no annotation, so
    # dataclasses.fields() skips them). Emit them verbatim so SiemensConfigDemo
    # exposes the same attribute surface (e.g. insertion_forcetorque_index = 4).
    class_var_lines: list[str] = []
    for name, val in vars(SiemensConfig).items():
        if name.startswith("_") or name in rendered_names or callable(val):
            continue
        if isinstance(val, np.ndarray):
            class_var_lines.append(f"    {name} = {_format_ndarray(val)}")
        elif isinstance(val, list):
            class_var_lines.append(f"    {name} = {_format_generic_list(val)}")
        elif isinstance(val, (bool, int, float, str)):
            class_var_lines.append(f"    {name} = {_format_scalar(val)}")
        else:
            print(f"[codegen] WARNING: skipping unsupported class attr {name!r}: {type(val).__name__}")
        class_var_lines.append("")
    if class_var_lines:
        body_lines.append("    # --- non-field class attributes (mirror SiemensConfig) ---")
        body_lines.extend(class_var_lines)

    header = (
        '"""AUTO-GENERATED by crisp_drl/scripts/calibrate_siemens_full.py.\n\n'
        f"Generated:     {datetime.now().isoformat()}\n"
        f"Source snap:   {source_snapshot_path}\n"
        f"Captured fields: {captured_fields}\n\n"
        "Mirrors crisp_drl.agents.shared.insertion_env_config.SiemensConfig\n"
        'shape so collect_data_real_lerobot_s.py can swap import via\n'
        'SIEMENS_USE_DEMO_CONFIG=1.\n"""\n\n'
        "from dataclasses import dataclass, field\n"
        "import numpy as np\n\n\n"
        "@dataclass\n"
        "class SiemensConfigDemo:\n"
    )

    # Module-level constants for the combined Siemens+Lego orchestrator.
    # These are NOT fields on SiemensConfigDemo because the Lego config needs
    # them too and they're a viewpoint, not a Siemens-specific parameter.
    combined_lines: list[str] = ["", "", "# --- Combined Siemens+Lego shared first PE viewpoint ---"]
    cpj = snapshot.get("combined_pe_joints")
    cpc = snapshot.get("combined_pe_cartesian")
    if cpj is not None:
        combined_lines.append(
            "COMBINED_FIRST_PE_POSITION: np.ndarray = "
            + _format_ndarray(np.asarray(cpj, dtype=np.float64))
        )
    else:
        combined_lines.append("COMBINED_FIRST_PE_POSITION: np.ndarray | None = None")
    if cpc is not None:
        combined_lines.append(
            "COMBINED_FIRST_PE_CARTESIAN: np.ndarray = "
            + _format_ndarray(np.asarray(cpc, dtype=np.float64))
        )
    else:
        combined_lines.append("COMBINED_FIRST_PE_CARTESIAN: np.ndarray | None = None")

    file_text = (
        header
        + "\n".join(body_lines).rstrip()
        + "\n"
        + "\n".join(combined_lines)
        + "\n"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = out_path.with_suffix(out_path.suffix + f".{stamp}.bak")
        shutil.copy2(out_path, backup)
        print(f"[codegen] Existing {out_path.name} backed up to {backup.name}")

    out_path.write_text(file_text)
    print(f"[codegen] Wrote {out_path}")

    return out_path, captured_fields, captured_paste_lines


def smoke_test_demo_config(out_path: Path = _DEMO_CONFIG_PATH) -> bool:
    """Import the freshly generated file in a subprocess; report success."""
    cmd = [
        sys.executable,
        "-c",
        (
            "from crisp_drl.agents.shared.siemens_config_demo import SiemensConfigDemo;"
            "c = SiemensConfigDemo();"
            "print('grasp:', c.grasp_position_ground_truth);"
            "print('w_D_w_o shape:', c.demo_w_D_w_o.shape);"
        ),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[smoke] FAILED:\n--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}")
        return False
    print(f"[smoke] OK\n{res.stdout}")
    return True


def archive_snapshot(path: Path = _SNAPSHOT_PATH) -> None:
    if not path.exists():
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived = path.with_name(f"{path.stem}.{stamp}.json")
    path.rename(archived)
    print(f"[snapshot] Archived to {archived.name}")


# ---------------------------------------------------------------------------
# Robot env + keyboard handlers
# ---------------------------------------------------------------------------

snapshot: dict = maybe_resume_snapshot()

env = make_env("my_env_v3_grav_comp_xyz")
print("Env created (gravity-compensation mode -- move the arm by hand).")
env.wait_until_ready()
print("Env ready.")
env.gripper.open()
env.home()
env.reset()
print("Env reset.")

_cfg = SiemensConfig()
GRASP_POS_REF = np.array(_cfg.grasp_position_ground_truth, dtype=np.float64)
print(f"[calib] GRASP_POS_REF = {GRASP_POS_REF}  (Procedure-B reference)")

rot_deg = 4.0
rot_deg_z = 0.25
rot_deg_x = 0.25

i_demo_img = 0
last_pe_tcp = None
last_color_path = None
last_depth_path = None


def _read_obs():
    obs, *_ = env.step(np.zeros(7))
    return obs


def _print_captured_summary():
    print("\n[snapshot] Current captures:")
    for k, v in snapshot.items():
        if v is None or (isinstance(v, list) and len(v) == 0):
            mark = "."
        else:
            mark = "*"
        if isinstance(v, list) and k == "waypoints":
            print(f"  [{mark}] {k}: {len(v)} entries")
        else:
            print(f"  [{mark}] {k}")
    print()


def run_pe():
    global snapshot
    if last_color_path is None or last_pe_tcp is None:
        print("[PE] No image - press 'i' first.")
        return

    print(f"[PE] Loading {last_color_path} ...")
    image = tifffile.imread(last_color_path)
    depth = tifffile.imread(last_depth_path)
    depth_m = depth.astype(np.float32) / 1000.0

    pe = PoseEstimationHelper(
        assumed_orientation=np.array([]),
        lock_orientation=False,
        brick_size="2x2",
        use_tracker=False,
    )

    print("[PE] Running SAM3 segmentation ('black cover with circular grille') ...")
    mask = pe.segmenter.segment_siemens(image)
    print(f"[PE] Mask coverage: {int(mask.sum() / 255)} px")

    print("[PE] Running FoundationPose (requires Isaac ROS) ...")
    cam_pose = pe.pose_estimator.estimate_siemens(image, depth_m, mask)
    print(f"[PE] Camera-frame pose:\n{cam_pose}")

    world_pose = pe._compute_pose_in_world_frame(cam_pose, last_pe_tcp)
    tcp_pose = pe._compute_pose_in_tcp_frame(cam_pose)
    print(f"[PE] World-frame pose:\n{world_pose}")

    R_w = world_pose[:3, :3]
    t_w = world_pose[:3, 3]
    o_T_o = R_w.T @ (GRASP_POS_REF - t_w)
    print(f"[PE] grasp_pos - obj_t (world):  {np.round((GRASP_POS_REF - t_w) * 1000, 2)} mm")
    print(f"[PE] o_T_o_tcpgrasp (obj frame): {np.round(o_T_o * 1000, 2)} mm")

    snapshot["demo_w_D_w_o"] = world_pose[:3, :].tolist()
    snapshot["demo_t_D_t_o"] = tcp_pose[:3, :].tolist()
    save_snapshot(snapshot)
    print("[PE] Captured demo_w_D_w_o + demo_t_D_t_o.")


def _capture_joints(key_label: str, snap_key: str):
    obs = _read_obs()
    joints = np.asarray(obs.get("observation.state.joints"), dtype=np.float64)
    if joints.ndim != 1 or joints.size < 7:
        print(f"[{key_label}] joints obs unavailable - cannot capture.")
        return
    snapshot[snap_key] = joints[:7].tolist()
    save_snapshot(snapshot)
    print(f"[{key_label}] {snap_key} <- {np.round(joints[:7], 5)}")


def _capture_pose(key_label: str, pos_key: str, euler_key: str):
    obs = _read_obs()
    tcp = np.asarray(obs["observation.state.cartesian"], dtype=np.float64)
    snapshot[pos_key] = tcp[:3].tolist()
    snapshot[euler_key] = tcp[3:6].tolist()
    save_snapshot(snapshot)
    print(f"[{key_label}] {pos_key}   <- {np.round(tcp[:3], 5)}")
    print(f"[{key_label}] {euler_key} <- {np.round(tcp[3:6], 5)}")


def _capture_waypoint():
    obs = _read_obs()
    tcp = np.asarray(obs["observation.state.cartesian"], dtype=np.float64)
    # Prefer the freshly-captured grasp Euler (press 'g'), otherwise fall
    # back to the source SiemensConfig default. This lets you capture
    # waypoints alone for a layout where the grasp orientation is unchanged.
    if snapshot["grasp_euler"] is not None:
        grasp_euler = np.asarray(snapshot["grasp_euler"], dtype=np.float64)
        ref_source = "snapshot (press 'g')"
    else:
        grasp_euler = np.asarray(
            SiemensConfig().grasp_orientation_ground_truth_euler, dtype=np.float64
        )
        ref_source = "SiemensConfig.grasp_orientation_ground_truth_euler default"
    rel_rot = (tcp[3:6] - grasp_euler).tolist()
    pose = tcp[:3].tolist() + rel_rot
    snapshot["waypoints"].append([pose, WAYPOINT_TOL_DEFAULT])
    save_snapshot(snapshot)
    print(
        f"[v] waypoint #{len(snapshot['waypoints'])}: pos={np.round(tcp[:3], 5)} "
        f"rel_rot(rad)={np.round(rel_rot, 5)} tol={WAYPOINT_TOL_DEFAULT}  "
        f"(rotation ref = {ref_source})"
    )


def _undo_waypoint():
    if not snapshot["waypoints"]:
        print("[z] no waypoint to undo")
        return
    dropped = snapshot["waypoints"].pop()
    save_snapshot(snapshot)
    print(f"[z] popped waypoint: {dropped}")


def _capture_dropoff():
    obs = _read_obs()
    tcp = np.asarray(obs["observation.state.cartesian"], dtype=np.float64)
    snapshot["dropoff"] = tcp[:3].tolist()
    save_snapshot(snapshot)
    print(f"[b] dropoff_point <- {np.round(tcp[:3], 5)}")


def _capture_combined_pe():
    """Capture the shared first-PE viewpoint for the Siemens+Lego runner.

    Records BOTH the 7-joint config and the Cartesian TCP pose so each task
    wrapper can be commanded to the same physical pose via its own config
    field (Siemens uses joints, Lego uses Cartesian).
    """
    obs = _read_obs()
    tcp = np.asarray(obs["observation.state.cartesian"], dtype=np.float64)
    joints = np.asarray(obs.get("observation.state.joints"), dtype=np.float64)
    if joints.ndim != 1 or joints.size < 7:
        print("[C] joints obs unavailable - cannot capture.")
        return
    snapshot["combined_pe_joints"] = joints[:7].tolist()
    snapshot["combined_pe_cartesian"] = tcp[:6].tolist()
    save_snapshot(snapshot)
    print(f"[C] combined_pe_joints    <- {np.round(joints[:7], 5)}")
    print(f"[C] combined_pe_cartesian <- {np.round(tcp[:6], 5)}")


def _emit_demo_config():
    print("\n[P] Writing SiemensConfigDemo ...")
    out_path, captured, paste_lines = write_demo_config_py(snapshot)
    print(f"[P] Captured fields ({len(captured)}): {captured}")
    not_captured = [
        _SNAPSHOT_TO_FIELD[k]
        for k, v in snapshot.items()
        if k in _SNAPSHOT_TO_FIELD and (v is None or (isinstance(v, list) and len(v) == 0))
    ]
    if not_captured:
        print(f"[P] WARNING: not captured (using SiemensConfig defaults): {not_captured}")

    ok = smoke_test_demo_config(out_path)
    if not ok:
        print("[P] Smoke test FAILED. Snapshot kept in place; "
              "fix and re-press 'P' once resolved.")
        return

    archive_snapshot()

    # Focused paste-block for the captured fields only. Copy/paste these
    # directly over the matching fields in crisp_drl/agents/shared/insertion_env_config.py
    # SiemensConfig. Field names line up 1:1 with the source class.
    if paste_lines:
        print("\n" + "=" * 72)
        print("[P] Paste-block for SiemensConfig (only captured fields below).")
        print("    Replace the matching fields in")
        print("    crisp_drl/agents/shared/insertion_env_config.py SiemensConfig.")
        print("=" * 72)
        for line in paste_lines:
            print(line)
        print("=" * 72)

    print("[P] Done. To use:")
    print("    SIEMENS_USE_DEMO_CONFIG=1 python scripts/collect_data_real_lerobot_s.py")


def on_press(key):
    global i_demo_img, last_pe_tcp, last_color_path, last_depth_path
    if not hasattr(key, "char") or key.char is None:
        return
    c = key.char

    if c == "o":
        env.step(np.array([0, 0, 0, 0, 0, 0, 0.2]))
        print("gripper open")
    elif c == "c":
        env.step(np.array([0, 0, 0, 0, 0, 0, -0.2]))
        print("gripper close")
    elif c == "y":
        env.step(np.array([0, 0, 0, 0, np.deg2rad(rot_deg), 0, 0]))
    elif c == "x":
        env.step(np.array([0, 0, 0, 0, np.deg2rad(-rot_deg), 0, 0]))
    elif c == "Y":
        env.step(np.array([0, 0, 0, 0, np.deg2rad(3.0), 0, 0]))
        print("[rot] applied +3 deg about Y")
    elif c == "k":
        env.step(np.array([0, 0, 0, 0, 0, np.deg2rad(rot_deg_z), 0]))
    elif c == "l":
        env.step(np.array([0, 0, 0, 0, 0, np.deg2rad(-rot_deg_z), 0]))
    elif c == "m":
        env.step(np.array([0, 0, 0, np.deg2rad(rot_deg_x), 0, 0, 0]))
    elif c == "n":
        env.step(np.array([0, 0, 0, np.deg2rad(-rot_deg_x), 0, 0, 0]))
    elif c == "r":
        obs = _read_obs()
        tcp = np.asarray(obs["observation.state.cartesian"], dtype=np.float64)
        joints = np.asarray(obs.get("observation.state.joints"), dtype=np.float64)
        print(f"[state] cartesian (x y z rx ry rz) = {tcp}")
        if joints.ndim == 1 and joints.size >= 7:
            print(f"[state] joints                     = {joints[:7]}")

    elif c == "u":
        _capture_joints("u", "first_home_joints")
    elif c == "h":
        _capture_joints("h", "home_joints")
    elif c == "j":
        _capture_joints("j", "home_pe_joints")
    elif c == "g":
        _capture_pose("g", "grasp_pos", "grasp_euler")
    elif c == "f":
        _capture_pose("f", "goal_pos", "goal_euler")
    elif c == "v":
        _capture_waypoint()
    elif c == "z":
        _undo_waypoint()
    elif c == "b":
        _capture_dropoff()
    elif c == "C":
        _capture_combined_pe()
    elif c == "S":
        _print_captured_summary()

    elif c == "i":
        obs = _read_obs()
        tcp = obs["observation.state.cartesian"]
        last_pe_tcp = np.asarray(tcp, dtype=np.float64).copy()
        color_path = str(_TEST_IMAGES_DIR / f"demo_img_color_{i_demo_img}_{EXP_NAME}.tiff")
        depth_path = str(_TEST_IMAGES_DIR / f"demo_img_depth_{i_demo_img}_{EXP_NAME}.tiff")
        tifffile.imwrite(color_path, obs["observation.images.wrist_camera"])
        tifffile.imwrite(depth_path, obs["observation.images.wrist_depth_camera"])
        last_color_path = color_path
        last_depth_path = depth_path
        print(f"Saved {color_path}  PE-TCP: {tcp}")
        i_demo_img += 1
    elif c == "p":
        run_pe()
    elif c == "P":
        _emit_demo_config()


listener = keyboard.Listener(on_press=on_press)
listener.start()

print(
    "\nControls (robot is in gravity-compensation mode -- move it by hand):\n"
    "  y/x/k/l/m/n = small rotate    Y = +3 deg about Y (one-shot)\n"
    "  o/c = gripper open/close       r = print state\n"
    "  Capture (autosaves to snapshot JSON):\n"
    "    u/h/j = first/home/PE-home joints\n"
    "    g/f   = grasp / goal pose (pos + euler)\n"
    "    v     = append waypoint (rel rot from grasp); z to undo\n"
    "    b     = dropoff point\n"
    "    C     = combined first-PE viewpoint (joints + cartesian)\n"
    "    i+p   = capture image then run PE -> demo_w_D_w_o + demo_t_D_t_o\n"
    "    S     = show capture status\n"
    "  P = write SiemensConfigDemo .py + smoke test\n"
    "  Enter = quit\n"
)
_print_captured_summary()

input()

listener.stop()
env.close()

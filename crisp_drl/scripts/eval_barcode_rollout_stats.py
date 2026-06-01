from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_poses(
    path: Path,
    stage: str,
    pose_index: int,
    pose_key: str = "poses_world",
    pose_is_matrix: bool = False,
) -> np.ndarray:
    poses: list[np.ndarray] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            if record.get("stage") != stage:
                continue

            if pose_is_matrix:
                pose = record.get(pose_key)
                if pose is None:
                    continue
                poses.append(np.asarray(pose, dtype=np.float64))
                continue

            pose_list = record.get(pose_key, [])
            if len(pose_list) <= pose_index:
                continue

            poses.append(np.asarray(pose_list[pose_index], dtype=np.float64))

    if not poses:
        raise RuntimeError(
            f"No poses found in {path} for stage={stage!r} and pose_index={pose_index}"
        )

    return np.stack(poses, axis=0)


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
        "range": float(values.max() - values.min()),
    }


def euler_xyz_to_rotation_matrix(euler_xyz_rad: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = euler_xyz_rad
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)

    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cx, -sx],
            [0.0, sx, cx],
        ],
        dtype=np.float64,
    )
    ry = np.array(
        [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ],
        dtype=np.float64,
    )
    rz = np.array(
        [
            [cz, -sz, 0.0],
            [sz, cz, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return rz @ ry @ rx


def rotation_error_deg(poses: np.ndarray, reference_rotation: np.ndarray) -> np.ndarray:
    rotations = poses[:, :3, :3]
    relative = np.einsum("ij,njk->nik", reference_rotation.T, rotations)
    cos_theta = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved barcode rollout poses and summarize position and rotation error."
        )
    )
    parser.add_argument(
        "--exp-name",
        required=True,
        help="Experiment name used to derive the rollout JSONL path",
    )
    parser.add_argument(
        "--assumed-euler-xyz-deg",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("ROLL", "PITCH", "YAW"),
        help=(
            "Assumed object orientation in Euler-XYZ angles (degrees). "
            "Rotation error is measured as the geodesic angle between each pose "
            "rotation matrix and this reference orientation."
        ),
    )
    parser.add_argument(
        "--stage",
        default="final_2",
        help="Stage name to filter rollout records (default: final_2)",
    )
    parser.add_argument(
        "--grasp-stage",
        default="grasp_1",
        help=(
            "Stage name to filter grasp pose records (default: grasp_1). "
            "If no matching grasp records exist, the grasp summary is skipped."
        ),
    )
    args = parser.parse_args()

    path = Path("rollout_data") / "tests" / f"{args.exp_name}.txt"
    stage = args.stage
    pose_index = 0
    poses = load_poses(path, stage, pose_index)
    reference_euler_deg = np.asarray(args.assumed_euler_xyz_deg, dtype=np.float64)
    reference_rotation = euler_xyz_to_rotation_matrix(np.deg2rad(reference_euler_deg))
    grasp_stage = args.grasp_stage
    grasp_poses: np.ndarray | None
    try:
        grasp_poses = load_poses(
            path,
            grasp_stage,
            pose_index,
            pose_key="grasp_pose_world",
            pose_is_matrix=True,
        )
    except RuntimeError:
        grasp_poses = None

    translation = poses[:, :3, 3]
    rotation_err = rotation_error_deg(poses, reference_rotation)
    translation_span_xyz = translation.max(axis=0) - translation.min(axis=0)
    translation_span_norm = float(np.linalg.norm(translation_span_xyz))
    grasp_position = None
    if grasp_poses is not None:
        grasp_translation = grasp_poses[:, :3, 3]
        grasp_position = {
            axis: summarize(grasp_translation[:, idx])
            for idx, axis in enumerate(("x", "y", "z"))
        }

    result = {
        "input": str(path),
        "stage": stage,
        "pose_index": pose_index,
        "count": int(len(poses)),
        "assumed_euler_xyz_deg": reference_euler_deg.tolist(),
        "position": {
            axis: summarize(translation[:, idx])
            for idx, axis in enumerate(("x", "y", "z"))
        },
        "translation_span_xyz": translation_span_xyz.tolist(),
        "translation_span_norm_m": translation_span_norm,
        "rotation_error_deg": summarize(rotation_err),
    }
    if grasp_position is not None:
        result["grasp_stage"] = grasp_stage
        result["grasp_position"] = grasp_position

    print(f"Input: {result['input']}")
    print(f"Stage: {result['stage']}")
    print(f"Pose index: {result['pose_index']}")
    print(f"Count: {result['count']}")
    print(
        "Assumed Euler XYZ (deg): ["
        + ", ".join(f"{value:.3f}" for value in reference_euler_deg)
        + "]"
    )
    print("POSITION (m)")
    for axis in ("x", "y", "z"):
        stats = result["position"][axis]
        print(
            f"  {axis}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
            f"min={stats['min']:.6f}, max={stats['max']:.6f}, range={stats['range']:.6f}"
        )
    print(
        "  span_xyz=["
        + ", ".join(f"{value:.6f}" for value in translation_span_xyz)
        + f"], span_norm={translation_span_norm:.6f} m"
    )
    if grasp_position is not None:
        print("GRASP POSITION (m)")
        print(f"  stage={grasp_stage}")
        for axis in ("x", "y", "z"):
            stats = grasp_position[axis]
            print(
                f"  {axis}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
                f"min={stats['min']:.6f}, max={stats['max']:.6f}, range={stats['range']:.6f}"
            )
    print("ROTATION ERROR (deg)")
    rot_stats = result["rotation_error_deg"]
    print(
        f"  mean={rot_stats['mean']:.6f}, std={rot_stats['std']:.6f}, "
        f"min={rot_stats['min']:.6f}, max={rot_stats['max']:.6f}, range={rot_stats['range']:.6f}"
    )


if __name__ == "__main__":
    main()

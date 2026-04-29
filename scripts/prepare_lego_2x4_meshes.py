"""One-shot: convert raw 2x4 LEGO OBJs to FoundationPose-friendly *_up.obj.

Raw input (in /home/gabor/workspaces/isaac_ros-dev/lego_assets/):
    lego_2x4_lavender.obj, lego_2x4_purple.obj
        units: millimetres
        origin: at one corner
        height axis: +Y (studs point +Y)
        extent: ~15.8 (X, width) x 11.55 (Y, height) x 31.8 (Z, length)

Output convention (matches existing lego_2x2_lavender_up.obj):
    units: metres
    origin: centroid of axis-aligned bbox
    height axis: +Z (studs point +Z)
    extent for 2x4: ~0.0318 (X, length) x 0.0158 (Y, width) x 0.01155 (Z, height)

Transform:  new = R @ (old / 1000) - centroid
            R = [[0,0,1],[1,0,0],[0,1,0]]   # old (X,Y,Z) -> new (Z,X,Y)
                so new_x=old_length, new_y=old_width, new_z=old_height (studs +Z)
"""

import os
import sys

import numpy as np

ASSETS_DIR = "/home/gabor/workspaces/isaac_ros-dev/lego_assets"
INPUTS = ("lego_2x4_lavender.obj", "lego_2x4_purple.obj")
OUTPUT_SUFFIX = "_up.obj"

# old -> new permutation: new_x=old_z, new_y=old_x, new_z=old_y
R = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
SCALE = 1e-3  # mm -> m


def transform_vertices(verts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotated = (R @ (verts * SCALE).T).T
    centroid = 0.5 * (rotated.min(axis=0) + rotated.max(axis=0))
    centered = rotated - centroid
    return centered, centroid


def transform_normals(normals: np.ndarray) -> np.ndarray:
    return (R @ normals.T).T


def process(in_path: str, out_path: str, mtl_basename: str) -> None:
    if not os.path.isfile(in_path):
        sys.exit(f"missing input: {in_path}")

    with open(in_path) as f:
        lines = f.readlines()

    v_idx, vn_idx, v_rows, vn_rows = [], [], [], []
    for i, line in enumerate(lines):
        if line.startswith("v "):
            _, x, y, z = line.split()[:4]
            v_idx.append(i)
            v_rows.append((float(x), float(y), float(z)))
        elif line.startswith("vn "):
            _, x, y, z = line.split()[:4]
            vn_idx.append(i)
            vn_rows.append((float(x), float(y), float(z)))

    v_arr = np.asarray(v_rows, dtype=np.float64)
    print(f"  raw verts n={len(v_arr)}")
    print(f"  raw min   = {v_arr.min(axis=0)}")
    print(f"  raw max   = {v_arr.max(axis=0)}")
    print(f"  raw extent= {v_arr.max(axis=0) - v_arr.min(axis=0)}")

    v_new, centroid = transform_vertices(v_arr)
    print(f"  centroid (post-rot, pre-center, m) = {centroid}")
    print(f"  new min   = {v_new.min(axis=0)}")
    print(f"  new max   = {v_new.max(axis=0)}")
    print(f"  new extent= {v_new.max(axis=0) - v_new.min(axis=0)}")

    if vn_rows:
        vn_arr = np.asarray(vn_rows, dtype=np.float64)
        vn_new = transform_normals(vn_arr)
    else:
        vn_new = None

    # write back: replace v / vn lines, swap mtllib to local mtl
    out_lines = list(lines)
    for i, vec in zip(v_idx, v_new):
        out_lines[i] = f"v {vec[0]:.6f} {vec[1]:.6f} {vec[2]:.6f}\n"
    if vn_new is not None:
        for i, vec in zip(vn_idx, vn_new):
            out_lines[i] = f"vn {vec[0]:.6f} {vec[1]:.6f} {vec[2]:.6f}\n"

    for i, line in enumerate(out_lines):
        if line.startswith("mtllib "):
            out_lines[i] = f"mtllib {mtl_basename}\n"
            break

    with open(out_path, "w") as f:
        f.writelines(out_lines)
    print(f"  wrote {out_path}")


def main() -> int:
    for name in INPUTS:
        in_path = os.path.join(ASSETS_DIR, name)
        out_name = name.replace(".obj", OUTPUT_SUFFIX)
        out_path = os.path.join(ASSETS_DIR, out_name)
        # mtl name must match the existing mtl that ships next to the raw obj
        mtl_basename = name.replace(".obj", ".mtl")

        print(f"--- {name} -> {out_name} ---")
        process(in_path, out_path, mtl_basename)
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

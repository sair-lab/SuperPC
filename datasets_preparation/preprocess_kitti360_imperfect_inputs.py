import argparse
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm

if __package__ is None or __package__ == "":
    # Allow direct execution: python datasets/preprocess_kitti360_imperfect_inputs.py
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from dataset.dataset import (
    _farthest_point_sampling_np,
    _farthest_point_sampling_np_fast,
    _remove_random_local_areas,
    _resample_to_fixed_count,
)


def _get_kitti360_preprocessed_paths(ply_path, num_in, num_out, seed):
    """Generate paths for imperfect input and GT point clouds for KITTI-360."""
    folder = os.path.dirname(ply_path)
    stem = os.path.splitext(os.path.basename(ply_path))[0]
    # Keep full stem (including "_submap_xyzrgb") to match training loader naming.
    tag = f"n{int(num_in)}_m{int(num_out)}_s{int(seed)}"
    input_npy = os.path.join(folder, f"{stem}__imperfect_{tag}.npy")
    gt_npy = os.path.join(folder, f"{stem}__gt_{tag}.npy")
    return input_npy, gt_npy


def _list_kitti360_records(root):
    """
    List all KITTI-360 submap records from the submaps directory.
    
    Expected structure:
    root/
        2013_05_28_drive_0000_sync/
            image_00/
                0000000080/
                    0000000080_rgb.png
                    0000000080_submap_xyzrgb.ply
                    metadata.json
                0000000090/
                    ...
        2013_05_28_drive_0002_sync/
            ...
    """
    import glob
    import json

    records = []
    
    # Find all drives
    drive_dirs = sorted(glob.glob(os.path.join(root, "2013_05_28_drive_*_sync")))
    
    for drive_dir in drive_dirs:
        if not os.path.isdir(drive_dir):
            continue
        
        # Get drive name
        drive_name = os.path.basename(drive_dir)
        
        # Look for image_00 subdirectory (could also have image_01, image_02, etc. but we'll use image_00)
        camera_dir = os.path.join(drive_dir, "image_00")
        if not os.path.isdir(camera_dir):
            continue
        
        # Find all frame folders
        frame_dirs = sorted(glob.glob(os.path.join(camera_dir, "*")))
        
        for frame_dir in frame_dirs:
            if not os.path.isdir(frame_dir):
                continue
            
            # Look for metadata.json
            metadata_path = os.path.join(frame_dir, "metadata.json")
            if not os.path.isfile(metadata_path):
                continue
            
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                
                # Extract file paths from metadata
                rgb_path = meta.get("rgb_image", "")
                ply_path = meta.get("submap_ply", "")
                
                # Resolve to absolute paths if needed
                if not os.path.isabs(rgb_path):
                    rgb_path = os.path.join(frame_dir, rgb_path)
                if not os.path.isabs(ply_path):
                    ply_path = os.path.join(frame_dir, ply_path)
                
                # Verify files exist
                if not (os.path.isfile(rgb_path) and os.path.isfile(ply_path)):
                    continue
                
                # Extract intrinsics if available
                intrinsics = None
                if "intrinsics" in meta:
                    intr = meta["intrinsics"]
                    if isinstance(intr, dict) and "K" in intr:
                        intrinsics = np.asarray(intr["K"], dtype=np.float32).reshape(3, 3)
                
                records.append({
                    "metadata_path": metadata_path,
                    "drive": drive_name,
                    "frame": int(meta.get("frame", -1)),
                    "rgb_path": rgb_path,
                    "ply_path": ply_path,
                    "intrinsics": intrinsics,
                })
            except Exception as e:
                # Skip frames with invalid metadata
                continue
    
    return records


def _sample_raw_pair(points, args, seed):
    """Sample input and ground truth point clouds from raw points."""
    rng = np.random.RandomState(int(seed) % (2**31 - 1))

    num_in = int(args.num_points)
    num_out = int(args.target_num_points)
    if num_out <= 0:
        raise ValueError("target_num_points must be > 0")
    if num_out > points.shape[0]:
        raise ValueError(f"target_num_points={num_out} exceeds available points={points.shape[0]}")

    fps_backend = getattr(args, "fps_backend", "numpy")

    gt_idx = _farthest_point_sampling_np(points, num_out, rng)
    gt_pts = points[gt_idx].astype(np.float32)

    in_idx = _farthest_point_sampling_np_fast(gt_pts, num_in, rng, backend=fps_backend)
    input_pts = gt_pts[in_idx].copy()

    if args.input_occlusion_ratio is not None:
        occ_ratio = float(args.input_occlusion_ratio)
    else:
        occ_min = float(min(args.input_occlusion_ratio_min, args.input_occlusion_ratio_max))
        occ_max = float(max(args.input_occlusion_ratio_min, args.input_occlusion_ratio_max))
        occ_ratio = float(rng.uniform(occ_min, occ_max))

    input_pts = _remove_random_local_areas(
        input_pts,
        rng,
        occlusion_ratio=occ_ratio,
        num_areas=int(args.kitti360_num_occlusion_areas),
    )
    input_pts = _resample_to_fixed_count(input_pts, num_in, rng)

    std_min = float(min(args.input_noise_std_min, args.input_noise_std_max))
    std_max = float(max(args.input_noise_std_min, args.input_noise_std_max))
    sigma = rng.uniform(std_min, std_max)
    input_pts = input_pts + sigma * rng.randn(*input_pts.shape).astype(np.float32)

    return input_pts.astype(np.float32), gt_pts.astype(np.float32)


def _save_ply(points_xyz, ply_path):
    """Save point cloud to PLY file."""
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float32))
    o3d.io.write_point_cloud(ply_path, pc, write_ascii=False, compressed=False)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute KITTI-360 imperfect input point clouds and cache beside GT .ply files"
    )
    parser.add_argument("--kitti360_root", default="/data_sair/kitti360_maps/submaps", type=str)
    parser.add_argument("--seed", default=21, type=int, help="base seed; sample seed is seed + dataset index")
    parser.add_argument("--num_points", default=10520, type=int)
    parser.add_argument("--target_num_points", default=46080, type=int)
    parser.add_argument("--input_noise_std_min", default=0.005, type=float)
    parser.add_argument("--input_noise_std_max", default=0.02, type=float)
    parser.add_argument("--input_occlusion_ratio_min", default=0.1, type=float)
    parser.add_argument("--input_occlusion_ratio_max", default=0.25, type=float)
    parser.add_argument("--input_occlusion_ratio", default=None, type=float)
    parser.add_argument("--kitti360_num_occlusion_areas", default=3, type=int)
    parser.add_argument("--max_samples", default=0, type=int, help="process at most this many samples (0 means all)")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing cache files")
    parser.add_argument("--save_ply", action="store_true", help="also save imperfect input as .ply for visualization")
    parser.add_argument(
        "--fps_backend",
        default="auto",
        choices=["auto", "numpy", "torch-cpu", "torch-cuda"],
        help="backend for sparse FPS (auto uses CUDA when available)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.kitti360_root):
        raise FileNotFoundError(f"KITTI-360 root not found: {args.kitti360_root}")

    records = _list_kitti360_records(args.kitti360_root)
    if int(args.max_samples) > 0:
        records = records[: int(args.max_samples)]
    if len(records) == 0:
        raise RuntimeError("No KITTI-360 records found.")

    saved = 0
    skipped = 0
    failed = 0

    for idx, rec in enumerate(tqdm(records, desc="preprocess KITTI-360")):
        ply_path = rec["ply_path"]
        sample_seed = int(args.seed) + idx
        input_npy_path, gt_npy_path = _get_kitti360_preprocessed_paths(
            ply_path,
            args.num_points,
            args.target_num_points,
            sample_seed,
        )
        input_ply_path = input_npy_path.replace(".npy", ".ply")

        if (not args.overwrite) and os.path.isfile(input_npy_path) and os.path.isfile(gt_npy_path):
            if (not args.save_ply) or os.path.isfile(input_ply_path):
                skipped += 1
                continue

        try:
            gt_pc = o3d.io.read_point_cloud(ply_path)
            points = np.asarray(gt_pc.points, dtype=np.float32)
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(f"Invalid GT points shape: {points.shape} in {ply_path}")

            input_pts, gt_pts = _sample_raw_pair(points, args, sample_seed)

            np.save(input_npy_path, input_pts)
            np.save(gt_npy_path, gt_pts)
            if args.save_ply:
                _save_ply(input_pts, input_ply_path)

            saved += 1
        except Exception as exc:
            failed += 1
            print(f"[WARN] Failed for {ply_path}: {exc}")

    print("Done.")
    print(f"Records: {len(records)}")
    print(f"Saved: {saved}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()

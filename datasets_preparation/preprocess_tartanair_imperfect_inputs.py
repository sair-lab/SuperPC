import argparse
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm

if __package__ is None or __package__ == "":
    # Allow direct execution: python datasets/preprocess_tartanair_imperfect_inputs.py
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from dataset.dataset import (
    TARTANAIR_TEST_SCENES,
    TARTANAIR_TRAIN_SCENES,
    _farthest_point_sampling_np,
    _farthest_point_sampling_np_fast,
    _get_tartanair_preprocessed_paths,
    _list_tartanair_records,
    _remove_random_local_areas,
    _resample_to_fixed_count,
    _split_tartanair_train_val,
)


def _build_records(args):
    split = args.split.lower()
    if split == "test":
        return _list_tartanair_records(args.tartanair_root, TARTANAIR_TEST_SCENES)

    train_records = _list_tartanair_records(args.tartanair_root, TARTANAIR_TRAIN_SCENES)
    train_records, val_records = _split_tartanair_train_val(
        train_records,
        val_ratio=float(args.val_ratio),
        split_seed=int(args.split_seed),
    )
    if split == "train":
        return train_records
    if split == "val":
        return val_records
    if split == "all":
        return train_records + val_records + _list_tartanair_records(args.tartanair_root, TARTANAIR_TEST_SCENES)
    raise ValueError(f"Unsupported split: {args.split}")


def _sample_raw_pair(points, args, seed):
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
        num_areas=int(args.num_occlusion_areas),
    )
    input_pts = _resample_to_fixed_count(input_pts, num_in, rng)

    std_min = float(min(args.input_noise_std_min, args.input_noise_std_max))
    std_max = float(max(args.input_noise_std_min, args.input_noise_std_max))
    sigma = rng.uniform(std_min, std_max)
    input_pts = input_pts + sigma * rng.randn(*input_pts.shape).astype(np.float32)

    return input_pts.astype(np.float32), gt_pts.astype(np.float32)


def _save_ply(points_xyz, ply_path):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float32))
    o3d.io.write_point_cloud(ply_path, pc, write_ascii=False, compressed=False)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute Tartanair imperfect input point clouds and cache beside GT .ply files"
    )
    parser.add_argument("--tartanair_root", default="/data_sair/tartanair_maps", type=str)
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "all"], type=str)
    parser.add_argument("--val_ratio", default=0.1, type=float)
    parser.add_argument("--split_seed", default=21, type=int)
    parser.add_argument("--seed", default=21, type=int, help="base seed; sample seed is seed + dataset index")
    parser.add_argument("--num_points", default=10520, type=int)
    parser.add_argument("--target_num_points", default=46080, type=int)
    parser.add_argument("--input_noise_std_min", default=0.005, type=float)
    parser.add_argument("--input_noise_std_max", default=0.02, type=float)
    parser.add_argument("--input_occlusion_ratio_min", default=0.1, type=float)
    parser.add_argument("--input_occlusion_ratio_max", default=0.25, type=float)
    parser.add_argument("--input_occlusion_ratio", default=None, type=float)
    parser.add_argument("--num_occlusion_areas", default=3, type=int)
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

    if not os.path.isdir(args.tartanair_root):
        raise FileNotFoundError(f"Tartanair root not found: {args.tartanair_root}")

    records = _build_records(args)
    if int(args.max_samples) > 0:
        records = records[: int(args.max_samples)]
    if len(records) == 0:
        raise RuntimeError("No Tartanair records found for selected split.")

    saved = 0
    skipped = 0
    failed = 0

    for idx, rec in enumerate(tqdm(records, desc=f"preprocess {args.split}")):
        ply_path = rec["ply_path"]
        sample_seed = int(args.seed) + idx
        input_npy_path, gt_npy_path = _get_tartanair_preprocessed_paths(
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

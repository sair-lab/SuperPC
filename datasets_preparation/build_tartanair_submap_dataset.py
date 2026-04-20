import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import open3d as o3d


def log_message(args, message):
    print(message, flush=True)
    log_fh = getattr(args, "_log_fh", None)
    if log_fh is not None:
        log_fh.write(message + "\n")
        log_fh.flush()


def log_once(args, key, message):
    if not hasattr(args, "_log_once_keys"):
        args._log_once_keys = set()
    if key in args._log_once_keys:
        return
    args._log_once_keys.add(key)
    log_message(args, message)


def depth_to_points(depth, fx, fy, cx, cy, max_depth):
    h, w = depth.shape
    v, u = np.indices(depth.shape)
    valid = np.isfinite(depth) & (depth > 0.0) & (depth < max_depth)

    z = depth[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    return xyz, valid


def compute_view_metrics(depth, valid, xyz, max_depth):
    h, w = depth.shape
    z = depth[valid]

    valid_ratio = float(valid.mean()) if valid.size else 0.0

    # Spatial spread across image bins.
    grid_h, grid_w = 12, 16
    bh = max(h // grid_h, 1)
    bw = max(w // grid_w, 1)
    covered = 0
    for gy in range(grid_h):
        ys = gy * bh
        ye = h if gy == grid_h - 1 else min((gy + 1) * bh, h)
        for gx in range(grid_w):
            xs = gx * bw
            xe = w if gx == grid_w - 1 else min((gx + 1) * bw, w)
            if np.any(valid[ys:ye, xs:xe]):
                covered += 1
    coverage = covered / float(grid_h * grid_w)

    z_std = float(np.std(z)) if z.size else 0.0
    z_std_norm = z_std / max(max_depth, 1e-6)

    dx = np.abs(depth[:, 1:] - depth[:, :-1])
    dy = np.abs(depth[1:, :] - depth[:-1, :])
    dx_valid = valid[:, 1:] & valid[:, :-1]
    dy_valid = valid[1:, :] & valid[:-1, :]
    grad_vals = []
    if np.any(dx_valid):
        grad_vals.append(float(np.mean(dx[dx_valid])))
    if np.any(dy_valid):
        grad_vals.append(float(np.mean(dy[dy_valid])))
    depth_grad_norm = (float(np.mean(grad_vals)) if grad_vals else 0.0) / max(max_depth, 1e-6)

    # Reject wall-like views via anisotropy in 3D covariance.
    if xyz.shape[0] > 6000:
        idx = np.linspace(0, xyz.shape[0] - 1, 6000, dtype=np.int64)
        pts = xyz[idx]
    else:
        pts = xyz

    if pts.shape[0] >= 10:
        centered = pts - pts.mean(axis=0, keepdims=True)
        cov = (centered.T @ centered) / max(pts.shape[0] - 1, 1)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.clip(eigvals, 1e-12, None)
        planar_ratio = float(eigvals[0] / eigvals.sum())
        linear_ratio = float(eigvals[1] / eigvals[2])
    else:
        planar_ratio = 0.0
        linear_ratio = 0.0

    return {
        "valid_ratio": valid_ratio,
        "coverage": coverage,
        "depth_std_norm": z_std_norm,
        "depth_grad_norm": depth_grad_norm,
        "planar_ratio": planar_ratio,
        "linear_ratio": linear_ratio,
    }


def farthest_point_sampling(points, n_samples):
    # Deterministic FPS in O(N * n_samples) for dataset preparation.
    n_points = points.shape[0]
    if n_points < n_samples:
        raise ValueError(f"Not enough points for FPS: {n_points} < {n_samples}")

    selected = np.empty((n_samples,), dtype=np.int64)
    distances = np.full((n_points,), np.inf, dtype=np.float64)

    first_idx = n_points // 2
    selected[0] = first_idx
    last_point = points[first_idx]

    for i in range(1, n_samples):
        d = np.sum((points - last_point) ** 2, axis=1)
        distances = np.minimum(distances, d)
        next_idx = int(np.argmax(distances))
        selected[i] = next_idx
        last_point = points[next_idx]

    return selected


def fps_downsample_xyzrgb_open3d(xyzrgb_full, n_samples):
    n_points = xyzrgb_full.shape[0]
    if n_points < n_samples:
        raise ValueError(f"Not enough points for FPS: {n_points} < {n_samples}")
    if n_points == n_samples:
        return xyzrgb_full

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzrgb_full[:, :3].astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector((xyzrgb_full[:, 3:6] / 255.0).astype(np.float64))

    # Open3D runs FPS in optimized C++, much faster than Python loops.
    pcd_down = pcd.farthest_point_down_sample(n_samples)

    xyz_down = np.asarray(pcd_down.points, dtype=np.float32)
    rgb_down = (np.asarray(pcd_down.colors, dtype=np.float32) * 255.0).clip(0.0, 255.0)
    xyzrgb_down = np.concatenate([xyz_down, rgb_down], axis=1)

    # Safety fallback if library behavior differs in edge cases.
    if xyzrgb_down.shape[0] != n_samples:
        idx = farthest_point_sampling(xyzrgb_full[:, :3], n_samples)
        xyzrgb_down = xyzrgb_full[idx]

    return xyzrgb_down


def fps_downsample_xyzrgb_torch(xyzrgb_full, n_samples, device):
    # Optional GPU FPS using PyTorch3D. Falls back to caller on import/runtime errors.
    import torch
    from pytorch3d.ops import sample_farthest_points

    n_points = xyzrgb_full.shape[0]
    if n_points < n_samples:
        raise ValueError(f"Not enough points for FPS: {n_points} < {n_samples}")
    if n_points == n_samples:
        return xyzrgb_full

    xyz = torch.from_numpy(xyzrgb_full[:, :3].astype(np.float32)).unsqueeze(0).to(device)
    sampled_xyz, sampled_idx = sample_farthest_points(xyz, K=n_samples)
    idx_np = sampled_idx.squeeze(0).detach().cpu().numpy().astype(np.int64)
    return xyzrgb_full[idx_np]


def fps_downsample_xyzrgb_torch_native(xyzrgb_full, n_samples, device):
    # Pure torch FPS (no PyTorch3D dependency). Works on CUDA if torch has CUDA.
    import torch

    n_points = xyzrgb_full.shape[0]
    if n_points < n_samples:
        raise ValueError(f"Not enough points for FPS: {n_points} < {n_samples}")
    if n_points == n_samples:
        return xyzrgb_full

    pts = torch.from_numpy(xyzrgb_full[:, :3].astype(np.float32)).to(device)
    idx = torch.empty((n_samples,), dtype=torch.long, device=device)
    min_dist = torch.full((n_points,), float("inf"), device=device)

    first = n_points // 2
    idx[0] = first
    last = pts[first]

    for i in range(1, n_samples):
        d = torch.sum((pts - last) ** 2, dim=1)
        min_dist = torch.minimum(min_dist, d)
        next_idx = torch.argmax(min_dist)
        idx[i] = next_idx
        last = pts[next_idx]

    idx_np = idx.detach().cpu().numpy().astype(np.int64)
    return xyzrgb_full[idx_np]


def fps_downsample_xyzrgb(xyzrgb_full, n_samples, args):
    if args.fps_backend == "torch" and not getattr(args, "_torch_fps_disabled", False):
        try:
            xyzrgb_down = fps_downsample_xyzrgb_torch(xyzrgb_full, n_samples, args.fps_device)
            log_once(args, "fps_backend_torch", f"[fps] backend=torch device={args.fps_device}")
            return xyzrgb_down
        except Exception as ex:
            log_once(
                args,
                "fps_torch_fallback",
                f"[fps] torch backend unavailable ({type(ex).__name__}: {ex}). Fallback to open3d.",
            )
            args._torch_fps_disabled = True
    elif args.fps_backend == "torch" and getattr(args, "_torch_fps_disabled", False):
        log_once(args, "fps_torch_disabled", "[fps] torch backend disabled for this run; using open3d.")

    if args.fps_backend == "torch_native" and not getattr(args, "_torch_native_fps_disabled", False):
        try:
            xyzrgb_down = fps_downsample_xyzrgb_torch_native(xyzrgb_full, n_samples, args.fps_device)
            log_once(args, "fps_backend_torch_native", f"[fps] backend=torch_native device={args.fps_device}")
            return xyzrgb_down
        except Exception as ex:
            log_once(
                args,
                "fps_torch_native_fallback",
                f"[fps] torch_native backend unavailable ({type(ex).__name__}: {ex}). Fallback to open3d.",
            )
            args._torch_native_fps_disabled = True
    elif args.fps_backend == "torch_native" and getattr(args, "_torch_native_fps_disabled", False):
        log_once(args, "fps_torch_native_disabled", "[fps] torch_native backend disabled for this run; using open3d.")

    xyzrgb_down = fps_downsample_xyzrgb_open3d(xyzrgb_full, n_samples)
    log_once(args, "fps_backend_open3d", "[fps] backend=open3d")
    return xyzrgb_down


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def save_pair(out_dir, frame, rgb_src, xyzrgb_full, xyzrgb_down, intrinsics, metrics, meta_extra):
    ensure_dir(out_dir)

    frame_stem = f"{frame:06d}"
    rgb_dst = out_dir / f"{frame_stem}_rgb_left.png"
    ply_down = out_dir / f"{frame_stem}_submap_xyzrgb.ply"
    meta_path = out_dir / "metadata.json"

    shutil.copy2(rgb_src, rgb_dst)

    pcd_down = o3d.geometry.PointCloud()
    pcd_down.points = o3d.utility.Vector3dVector(xyzrgb_down[:, :3].astype(np.float64))
    pcd_down.colors = o3d.utility.Vector3dVector((xyzrgb_down[:, 3:6] / 255.0).astype(np.float64))
    o3d.io.write_point_cloud(str(ply_down), pcd_down)

    metadata = {
        "frame": frame,
        "rgb_image": str(rgb_dst),
        "submap_ply": str(ply_down),
        "num_points_original": int(xyzrgb_full.shape[0]),
        "num_points_saved": int(xyzrgb_down.shape[0]),
        "intrinsics": intrinsics,
        "quality_metrics": metrics,
    }
    metadata.update(meta_extra)

    with open(meta_path, "w", encoding="ascii") as f:
        json.dump(metadata, f, indent=2)


def process_sequence(seq_root, out_root, args, summary, seq_index=None, total_sequences=None):
    depth_dir = seq_root / "depth_left"
    rgb_dir = seq_root / "image_left"
    pose_file = seq_root / "pose_left.txt"

    if not depth_dir.exists() or not rgb_dir.exists() or not pose_file.exists():
        return

    depth_files = sorted(depth_dir.glob("*_left_depth.npy"))
    if not depth_files:
        return

    selected_frames = []
    last_selected = -10**9
    skip_reasons = {
        "frame_gap": 0,
        "already_exists": 0,
        "bad_depth_shape": 0,
        "too_few_points": 0,
        "rgb_missing": 0,
        "bad_rgb_shape": 0,
    }

    # seq_root is .../<scene>/<difficulty>/<trajectory>
    scene = seq_root.parents[1].name
    difficulty = seq_root.parents[0].name
    trajectory = seq_root.name

    candidate_depth_files = depth_files[:: max(args.candidate_stride, 1)]
    candidate_total = len(candidate_depth_files)

    if seq_index is not None and total_sequences is not None:
        seq_remaining_after_current = max(total_sequences - seq_index, 0)
        log_message(
            args,
            f"[sequence-start] {scene}/{difficulty}/{trajectory} "
            f"seq_processed={seq_index - 1}/{total_sequences} "
            f"seq_remaining={seq_remaining_after_current + 1} "
            f"candidates={candidate_total}",
        )
    else:
        log_message(args, f"[sequence-start] {scene}/{difficulty}/{trajectory} candidates={candidate_total}")

    for i, depth_file in enumerate(candidate_depth_files, start=1):
        frame = int(depth_file.name.split("_left_depth.npy")[0])

        if i == 1 or i % max(args.log_every_n_candidates, 1) == 0:
            cand_remaining = candidate_total - i
            log_message(
                args,
                f"[sequence-progress] {scene}/{difficulty}/{trajectory} "
                f"cand_processed={i}/{candidate_total} cand_remaining={cand_remaining} "
                f"selected={len(selected_frames)}",
            )

        # Rule 1: at least 10 frames apart.
        if frame - last_selected < args.min_frame_gap:
            skip_reasons["frame_gap"] += 1
            continue

        if args.max_pairs_per_sequence > 0 and len(selected_frames) >= args.max_pairs_per_sequence:
            break

        out_dir = out_root / scene / difficulty / trajectory / f"{frame:06d}"
        if (out_dir / "metadata.json").exists():
            skip_reasons["already_exists"] += 1
            selected_frames.append(frame)
            last_selected = frame
            continue

        depth = np.load(depth_file).astype(np.float32)
        if depth.shape != (args.height, args.width):
            skip_reasons["bad_depth_shape"] += 1
            continue

        xyz, valid = depth_to_points(depth, args.fx, args.fy, args.cx, args.cy, args.max_depth)

        # Rule 2: original cloud must have at least 300000 points.
        if xyz.shape[0] < args.min_original_points:
            skip_reasons["too_few_points"] += 1
            continue

        metrics = compute_view_metrics(depth, valid, xyz, args.max_depth)

        rgb_path = rgb_dir / f"{frame:06d}_left.png"
        if not rgb_path.exists():
            skip_reasons["rgb_missing"] += 1
            continue

        rgb = np.asarray(o3d.io.read_image(str(rgb_path)))
        if rgb.shape[:2] != (args.height, args.width):
            skip_reasons["bad_rgb_shape"] += 1
            continue

        rgb_valid = rgb[valid]
        xyzrgb_full = np.concatenate([xyz, rgb_valid[:, :3].astype(np.float32)], axis=1)

        # Rule 4: evenly downsample to exactly 46080 points (FPS).
        xyzrgb_down = fps_downsample_xyzrgb(xyzrgb_full, args.downsample_points, args)

        save_pair(
            out_dir=out_dir,
            frame=frame,
            rgb_src=rgb_path,
            xyzrgb_full=xyzrgb_full,
            xyzrgb_down=xyzrgb_down,
            intrinsics={
                "width": args.width,
                "height": args.height,
                "fx": args.fx,
                "fy": args.fy,
                "cx": args.cx,
                "cy": args.cy,
                "max_depth": args.max_depth,
                "K": [[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy], [0.0, 0.0, 1.0]],
            },
            metrics=metrics,
            meta_extra={
                "scene": scene,
                "difficulty": difficulty,
                "trajectory": trajectory,
                "pose_left_file": str(pose_file),
                "note": "Submap points are in left-camera frame and can be backprojected with intrinsics K.",
            },
        )

        selected_frames.append(frame)
        last_selected = frame

    log_message(
        args,
        f"[sequence-done] {scene}/{difficulty}/{trajectory} "
        f"cand_processed={candidate_total}/{candidate_total} cand_remaining=0 "
        f"selected={len(selected_frames)} "
        f"skip_gap={skip_reasons['frame_gap']} "
        f"skip_exists={skip_reasons['already_exists']} "
        f"skip_points={skip_reasons['too_few_points']} "
        f"skip_depth_shape={skip_reasons['bad_depth_shape']} "
        f"skip_rgb_missing={skip_reasons['rgb_missing']} "
        f"skip_rgb_shape={skip_reasons['bad_rgb_shape']}",
    )

    summary[f"{scene}/{difficulty}/{trajectory}"] = {
        "selected_count": len(selected_frames),
        "selected_frames": selected_frames,
        "skip_reasons": skip_reasons,
    }


def collect_sequences(dataset_root, include_scenes, exclude_scenes, difficulty):
    scenes = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    sequences = []
    for scene_dir in scenes:
        if include_scenes and scene_dir.name not in include_scenes:
            continue
        if exclude_scenes and scene_dir.name in exclude_scenes:
            continue
        diff_dir = scene_dir / difficulty
        if not diff_dir.exists() or not diff_dir.is_dir():
            continue
        traj_dirs = sorted([p for p in diff_dir.iterdir() if p.is_dir() and p.name.startswith("P")])
        sequences.extend(traj_dirs)
    return sequences


def main():
    parser = argparse.ArgumentParser(description="Build TartanAir submap+RGB paired dataset with quality constraints")
    parser.add_argument("--dataset_root", default="/data_sairpro/TartanAir")
    parser.add_argument("--output_root", default="/data_sair/tartanair_maps")
    parser.add_argument("--difficulty", default="Easy")
    parser.add_argument("--include_scenes", nargs="*", default=[])
    parser.add_argument(
        "--exclude_scenes",
        nargs="*",
        default=["gascola", "ocean", "seasonsforest", "seasonsforest_winter"],
        help="Scenes to skip when building dataset",
    )
    parser.add_argument("--fx", type=float, default=320.0)
    parser.add_argument("--fy", type=float, default=320.0)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=240.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--max_depth", type=float, default=8.0)

    # User rules.
    parser.add_argument("--min_frame_gap", type=int, default=10)
    parser.add_argument("--min_original_points", type=int, default=200000)
    parser.add_argument("--downsample_points", type=int, default=46080)

    parser.add_argument("--max_sequences", type=int, default=0, help="0 means all")
    parser.add_argument("--candidate_stride", type=int, default=5, help="Only evaluate every N-th frame")
    parser.add_argument("--max_pairs_per_sequence", type=int, default=0, help="0 means no per-sequence limit")
    parser.add_argument("--log_every_n_candidates", type=int, default=50, help="Log progress every N candidate frames")
    parser.add_argument("--log_file", default=None, help="Path to persistent log file")
    parser.add_argument(
        "--fps_backend",
        choices=["open3d", "torch", "torch_native"],
        default="open3d",
        help="FPS backend",
    )
    parser.add_argument("--fps_device", default="cuda", help="Device for torch FPS backend")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    ensure_dir(output_root)

    if args.log_file is None:
        args.log_file = str(output_root / "build_tartanair_submap_dataset.log")

    log_path = Path(args.log_file)
    ensure_dir(log_path.parent)
    args._log_fh = open(log_path, "a", encoding="ascii")
    log_message(args, f"\n[{datetime.now().isoformat(timespec='seconds')}] build start")
    log_message(args, f"[config] dataset_root={dataset_root}")
    log_message(args, f"[config] output_root={output_root}")
    log_message(args, f"[config] log_file={log_path}")
    log_message(args, f"[config] fps_backend={args.fps_backend}")
    if args.fps_backend == "torch":
        log_message(args, f"[config] fps_device={args.fps_device}")

    sequences = collect_sequences(
        dataset_root,
        set(args.include_scenes),
        set(args.exclude_scenes),
        args.difficulty,
    )
    if args.max_sequences > 0:
        sequences = sequences[: args.max_sequences]

    summary = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "difficulty": args.difficulty,
        "exclude_scenes": args.exclude_scenes,
        "rules": {
            "min_frame_gap": args.min_frame_gap,
            "min_original_points": args.min_original_points,
            "downsample_points": args.downsample_points,
            "max_depth": args.max_depth,
        },
        "sequences_total": len(sequences),
        "scenes_total": len({seq.parents[1].name for seq in sequences}),
        "sequences": {},
    }

    total_sequences = len(sequences)
    for seq_index, seq_root in enumerate(sequences, start=1):
        scene_processed = len({seq.parents[1].name for seq in sequences[: seq_index - 1]})
        scene_remaining = len({seq.parents[1].name for seq in sequences[seq_index - 1 :]})
        log_message(
            args,
            f"[overall] seq_processed={seq_index - 1}/{total_sequences} "
            f"seq_remaining={total_sequences - (seq_index - 1)} "
            f"scene_processed={scene_processed}/{summary['scenes_total']} "
            f"scene_remaining={scene_remaining} now={seq_root}",
        )
        process_sequence(
            seq_root,
            output_root,
            args,
            summary["sequences"],
            seq_index=seq_index,
            total_sequences=total_sequences,
        )
        log_message(
            args,
            f"[overall] seq_processed={seq_index}/{total_sequences} "
            f"seq_remaining={total_sequences - seq_index} "
            f"scene_processed={len({seq.parents[1].name for seq in sequences[:seq_index]})}/{summary['scenes_total']} "
            f"scene_remaining={len({seq.parents[1].name for seq in sequences[seq_index:]})}",
        )

    summary_path = output_root / "dataset_summary.json"
    with open(summary_path, "w", encoding="ascii") as f:
        json.dump(summary, f, indent=2)

    selected_total = sum(v["selected_count"] for v in summary["sequences"].values())
    log_message(args, f"Processed sequences: {len(sequences)}")
    log_message(args, f"Selected pairs: {selected_total}")
    log_message(args, f"Summary: {summary_path}")
    log_message(args, f"[{datetime.now().isoformat(timespec='seconds')}] build end")
    args._log_fh.close()


if __name__ == "__main__":
    main()

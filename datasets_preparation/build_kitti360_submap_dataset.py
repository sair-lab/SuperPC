import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d

from generate_kitti360_pair import (
    find_covering_tile,
    frustum_clip_mask,
    fps_downsample_xyzrgb_torch_native,
    load_cam0_poses,
    load_perspective_intrinsics,
    load_world_points,
    parse_frame_ranges,
    transform_world_to_camera,
)


def log_message(args, message):
    print(message, flush=True)
    log_fh = getattr(args, "_log_fh", None)
    if log_fh is not None:
        log_fh.write(message + "\n")
        log_fh.flush()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def collect_drives(maps_root: Path):
    drives = []
    for p in sorted(maps_root.iterdir()):
        if not p.is_dir():
            continue
        if not p.name.endswith("_sync"):
            continue
        if (p / "static").exists():
            drives.append(p.name)
    return drives


def process_drive(drive, intrinsics, args, summary):
    kitti_root = Path(args.kitti_root)
    maps_root = Path(args.kitti_maps_root)
    out_root = Path(args.output_root)

    cam0_pose_file = kitti_root / "data_poses" / drive / "cam0_to_world.txt"
    image_dir = kitti_root / "data_2d_raw" / drive / args.camera / "data_rect"
    static_dir = maps_root / drive / "static"
    dynamic_dir = maps_root / drive / "dynamic"

    if not cam0_pose_file.exists() or not image_dir.exists() or not static_dir.exists():
        log_message(args, f"[drive-skip] {drive} missing required files/directories")
        summary[drive] = {"selected_count": 0, "selected_frames": [], "skip_reasons": {"missing_inputs": 1}}
        return

    poses = load_cam0_poses(cam0_pose_file)
    pose_frames = sorted(poses.keys())
    static_ranges = parse_frame_ranges(static_dir)
    dynamic_ranges = parse_frame_ranges(dynamic_dir) if args.include_dynamic and dynamic_dir.exists() else []

    if not pose_frames or not static_ranges:
        log_message(args, f"[drive-skip] {drive} no poses or no static map tiles")
        summary[drive] = {"selected_count": 0, "selected_frames": [], "skip_reasons": {"missing_pose_or_tile": 1}}
        return

    selected_frames = []
    last_selected = -10**9

    skip_reasons = {
        "frame_gap": 0,
        "image_missing": 0,
        "tile_missing": 0,
        "too_few_points": 0,
        "already_exists": 0,
    }

    tile_cache = {}
    drive_end_frame = pose_frames[-1]
    max_allowed_frame = drive_end_frame - args.end_margin_frames
    candidate_frames = [f for f in pose_frames if args.start_frame <= f <= max_allowed_frame]
    candidate_frames = candidate_frames[:: max(args.candidate_stride, 1)]
    if args.max_candidate_frames > 0:
        candidate_frames = candidate_frames[: args.max_candidate_frames]

    log_message(
        args,
        f"[drive-start] {drive} candidates={len(candidate_frames)} "
        f"drive_end_frame={drive_end_frame} max_allowed_frame={max_allowed_frame}",
    )

    for i, frame in enumerate(candidate_frames, start=1):
        if i == 1 or i % max(args.log_every_n_candidates, 1) == 0:
            log_message(
                args,
                f"[drive-progress] {drive} cand_processed={i}/{len(candidate_frames)} selected={len(selected_frames)}",
            )

        if frame - last_selected < args.min_frame_gap:
            skip_reasons["frame_gap"] += 1
            continue

        if args.max_pairs_per_drive > 0 and len(selected_frames) >= args.max_pairs_per_drive:
            break

        out_dir = out_root / drive / args.camera / f"{frame:010d}"
        out_rgb = out_dir / f"{frame:010d}_rgb.png"
        out_ply = out_dir / f"{frame:010d}_submap_xyzrgb.ply"
        out_meta = out_dir / "metadata.json"

        if out_meta.exists():
            skip_reasons["already_exists"] += 1
            selected_frames.append(frame)
            last_selected = frame
            continue

        rgb_path = image_dir / f"{frame:010d}.png"
        if not rgb_path.exists():
            skip_reasons["image_missing"] += 1
            continue

        tile_info = find_covering_tile(frame, static_ranges)
        if tile_info is None:
            skip_reasons["tile_missing"] += 1
            continue

        tile_start, tile_end, static_ply = tile_info
        dynamic_ply = None
        if dynamic_ranges:
            dyn_tile_info = find_covering_tile(frame, dynamic_ranges)
            if dyn_tile_info is not None:
                dynamic_ply = dyn_tile_info[2]

        cache_key = (str(static_ply), str(dynamic_ply) if dynamic_ply else "")
        if cache_key not in tile_cache:
            tile_cache[cache_key] = load_world_points(static_ply, dynamic_ply)
        xyz_world, rgb_world = tile_cache[cache_key]

        rgb_img = np.asarray(o3d.io.read_image(str(rgb_path)))
        if rgb_img.ndim < 2:
            skip_reasons["image_missing"] += 1
            continue
        img_h, img_w = rgb_img.shape[:2]

        t_cam_to_world = poses[frame]
        xyz_cam, t_world_to_cam = transform_world_to_camera(xyz_world, t_cam_to_world)

        frustum_valid = frustum_clip_mask(
            xyz_cam,
            fx=intrinsics["fx"],
            fy=intrinsics["fy"],
            cx=intrinsics["cx"],
            cy=intrinsics["cy"],
            width=img_w,
            height=img_h,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        )
        dist = np.linalg.norm(xyz_cam, axis=1)
        valid = frustum_valid & (dist < args.max_radius)

        xyz_cam = xyz_cam[valid]
        rgb = rgb_world[valid]
        xyzrgb_full = np.concatenate([xyz_cam.astype(np.float32), rgb.astype(np.float32)], axis=1)

        if xyzrgb_full.shape[0] < args.downsample_points:
            skip_reasons["too_few_points"] += 1
            continue

        xyzrgb_down = fps_downsample_xyzrgb_torch_native(
            xyzrgb_full,
            n_samples=args.downsample_points,
            device=args.fps_device,
        )

        ensure_dir(out_dir)
        shutil.copy2(rgb_path, out_rgb)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyzrgb_down[:, :3].astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector((xyzrgb_down[:, 3:6] / 255.0).astype(np.float64))
        o3d.io.write_point_cloud(str(out_ply), pcd)

        metadata = {
            "dataset": "KITTI-360",
            "drive": drive,
            "camera": args.camera,
            "frame": frame,
            "rgb_image": str(out_rgb),
            "submap_ply": str(out_ply),
            "num_points_original": int(xyzrgb_full.shape[0]),
            "num_points_saved": int(xyzrgb_down.shape[0]),
            "tile": {
                "start_frame": tile_start,
                "end_frame": tile_end,
                "static_ply": str(static_ply),
                "dynamic_ply": str(dynamic_ply) if dynamic_ply else None,
            },
            "filters": {
                "min_depth": args.min_depth,
                "max_depth": args.max_depth,
                "max_radius": args.max_radius,
                "downsample_points": args.downsample_points,
                "fps_device": args.fps_device,
                "frustum_clip": {
                    "enabled": True,
                    "image_width": int(img_w),
                    "image_height": int(img_h),
                },
            },
            "intrinsics": intrinsics,
            "extrinsics": {
                "T_cam0_to_world": t_cam_to_world.tolist(),
                "T_world_to_cam0": t_world_to_cam.tolist(),
            },
            "notes": "Submap points are transformed from global map (world) into camera frame with frustum clipping.",
        }

        with open(out_meta, "w", encoding="ascii") as f:
            json.dump(metadata, f, indent=2)

        selected_frames.append(frame)
        last_selected = frame

    log_message(
        args,
        f"[drive-done] {drive} selected={len(selected_frames)} "
        f"skip_gap={skip_reasons['frame_gap']} "
        f"skip_image_missing={skip_reasons['image_missing']} "
        f"skip_tile_missing={skip_reasons['tile_missing']} "
        f"skip_too_few_points={skip_reasons['too_few_points']} "
        f"skip_exists={skip_reasons['already_exists']}",
    )

    summary[drive] = {
        "selected_count": len(selected_frames),
        "selected_frames": selected_frames,
        "skip_reasons": skip_reasons,
    }


def main():
    parser = argparse.ArgumentParser(description="Build KITTI-360 submap+RGB paired dataset")
    parser.add_argument("--kitti_root", default="/data_sair/kitti360")
    parser.add_argument("--kitti_maps_root", default="/data_sair/kitti360_maps/data_3d_semantics/train")
    parser.add_argument("--output_root", default="/data_sair/kitti360_maps/submaps")
    parser.add_argument("--camera", choices=["image_00", "image_01"], default="image_00")

    parser.add_argument("--start_frame", type=int, default=80, help="Only consider frames >= this id per drive")
    parser.add_argument(
        "--end_margin_frames",
        type=int,
        default=80,
        help="Exclude frames within this many frames from the end of each drive",
    )
    parser.add_argument("--min_frame_gap", type=int, default=80)
    parser.add_argument("--candidate_stride", type=int, default=1)
    parser.add_argument("--max_pairs_per_drive", type=int, default=0, help="0 means no cap")
    parser.add_argument("--downsample_points", type=int, default=46080)
    parser.add_argument("--fps_device", default="cuda")

    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=30.0)
    parser.add_argument("--max_radius", type=float, default=80.0)
    parser.add_argument("--include_dynamic", action="store_true")

    parser.add_argument("--include_drives", nargs="*", default=[])
    parser.add_argument(
        "--exclude_drives",
        nargs="*",
        default=[],
        help="Drives to skip",
    )
    parser.add_argument("--max_drives", type=int, default=0, help="0 means all")
    parser.add_argument("--max_candidate_frames", type=int, default=0, help="0 means all")

    parser.add_argument("--log_every_n_candidates", type=int, default=100)
    parser.add_argument("--log_file", default=None)

    args = parser.parse_args()

    output_root = Path(args.output_root)
    ensure_dir(output_root)

    if args.log_file is None:
        args.log_file = str(output_root / "build_kitti360_submap_dataset.log")

    log_path = Path(args.log_file)
    ensure_dir(log_path.parent)
    args._log_fh = open(log_path, "a", encoding="ascii")

    log_message(args, f"\n[{datetime.now().isoformat(timespec='seconds')}] build start")
    log_message(args, f"[config] kitti_root={args.kitti_root}")
    log_message(args, f"[config] kitti_maps_root={args.kitti_maps_root}")
    log_message(args, f"[config] output_root={args.output_root}")
    log_message(args, f"[config] camera={args.camera}")
    log_message(args, f"[config] start_frame={args.start_frame}")
    log_message(args, f"[config] end_margin_frames={args.end_margin_frames}")
    log_message(args, f"[config] exclude_drives={args.exclude_drives}")
    log_message(args, f"[config] min_frame_gap={args.min_frame_gap}")
    log_message(args, f"[config] downsample_points={args.downsample_points}")
    log_message(args, f"[config] fps_device={args.fps_device}")
    log_message(args, f"[config] log_file={args.log_file}")

    perspective_file = Path(args.kitti_root) / "calibration" / "perspective.txt"
    intrinsics = load_perspective_intrinsics(perspective_file, args.camera)

    all_drives = collect_drives(Path(args.kitti_maps_root))
    drives = []
    include_set = set(args.include_drives)
    exclude_set = set(args.exclude_drives)
    for d in all_drives:
        if include_set and d not in include_set:
            continue
        if d in exclude_set:
            continue
        drives.append(d)

    if args.max_drives > 0:
        drives = drives[: args.max_drives]

    summary = {
        "kitti_root": args.kitti_root,
        "kitti_maps_root": args.kitti_maps_root,
        "output_root": args.output_root,
        "camera": args.camera,
        "rules": {
            "start_frame": args.start_frame,
            "end_margin_frames": args.end_margin_frames,
            "min_frame_gap": args.min_frame_gap,
            "downsample_points": args.downsample_points,
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
            "max_radius": args.max_radius,
        },
        "drives_total": len(drives),
        "drives": {},
    }

    for i, drive in enumerate(drives, start=1):
        log_message(args, f"[overall] drive_processed={i-1}/{len(drives)} drive_remaining={len(drives)-(i-1)} now={drive}")
        process_drive(drive, intrinsics, args, summary["drives"])
        log_message(args, f"[overall] drive_processed={i}/{len(drives)} drive_remaining={len(drives)-i}")

    summary_path = output_root / "dataset_summary.json"
    with open(summary_path, "w", encoding="ascii") as f:
        json.dump(summary, f, indent=2)

    selected_total = sum(v["selected_count"] for v in summary["drives"].values())
    log_message(args, f"Processed drives: {len(drives)}")
    log_message(args, f"Selected pairs: {selected_total}")
    log_message(args, f"Summary: {summary_path}")
    log_message(args, f"[{datetime.now().isoformat(timespec='seconds')}] build end")

    args._log_fh.close()


if __name__ == "__main__":
    main()

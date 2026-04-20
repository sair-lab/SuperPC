import argparse
import json
import shutil
from bisect import bisect_left
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_frame_ranges(map_dir: Path):
    ranges = []
    for ply_path in sorted(map_dir.glob("*.ply")):
        stem = ply_path.stem
        parts = stem.split("_")
        if len(parts) != 2:
            continue
        try:
            start = int(parts[0])
            end = int(parts[1])
        except ValueError:
            continue
        ranges.append((start, end, ply_path))
    return ranges


def load_cam0_poses(cam0_pose_file: Path):
    poses = {}
    with open(cam0_pose_file, "r", encoding="ascii") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = line.split()
            if len(vals) != 17:
                continue
            frame = int(vals[0])
            mat = np.array([float(x) for x in vals[1:]], dtype=np.float64).reshape(4, 4)
            poses[frame] = mat
    if not poses:
        raise RuntimeError(f"No poses parsed from {cam0_pose_file}")
    return poses


def load_perspective_intrinsics(perspective_file: Path, camera: str):
    key = "P_rect_00" if camera == "image_00" else "P_rect_01"
    with open(perspective_file, "r", encoding="ascii") as f:
        for line in f:
            line = line.strip()
            if not line.startswith(key + ":"):
                continue
            vals = [float(v) for v in line.split(":", 1)[1].split()]
            if len(vals) != 12:
                raise RuntimeError(f"Unexpected {key} length in {perspective_file}")
            p = np.array(vals, dtype=np.float64).reshape(3, 4)
            fx = float(p[0, 0])
            fy = float(p[1, 1])
            cx = float(p[0, 2])
            cy = float(p[1, 2])
            return {
                "key": key,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                "P_rect": p.tolist(),
            }
    raise RuntimeError(f"Could not find {key} in {perspective_file}")


def pick_frame(args, poses, static_ranges):
    pose_frames = sorted(poses.keys())
    if args.frame >= 0:
        return args.frame

    if not static_ranges:
        raise RuntimeError("No static map tiles found")

    start, end, _ = static_ranges[0]
    for frame in pose_frames:
        if start <= frame <= end:
            return frame
    return pose_frames[0]


def resolve_pose_frame(requested_frame, pose_frames, fallback_mode, max_gap):
    if requested_frame in pose_frames:
        return requested_frame, 0

    if fallback_mode == "error":
        raise ValueError(
            f"Frame {requested_frame} is missing from cam0_to_world poses. "
            f"Use one of the available pose frames or set --pose_fallback nearest."
        )

    idx = bisect_left(pose_frames, requested_frame)
    candidates = []
    if idx > 0:
        candidates.append(pose_frames[idx - 1])
    if idx < len(pose_frames):
        candidates.append(pose_frames[idx])
    if not candidates:
        raise RuntimeError("No pose frames available for fallback")

    nearest = min(candidates, key=lambda f: abs(f - requested_frame))
    gap = abs(nearest - requested_frame)
    if max_gap >= 0 and gap > max_gap:
        raise ValueError(
            f"Requested frame {requested_frame} has no nearby pose within max gap {max_gap}. "
            f"Nearest pose frame is {nearest} (gap={gap})."
        )
    return nearest, gap


def find_covering_tile(frame, ranges):
    for start, end, ply_path in ranges:
        if start <= frame <= end:
            return start, end, ply_path
    return None


def load_world_points(static_ply: Path, dynamic_ply: Path | None):
    pcd_static = o3d.io.read_point_cloud(str(static_ply))
    xyz_s = np.asarray(pcd_static.points, dtype=np.float32)
    rgb_s = (np.asarray(pcd_static.colors, dtype=np.float32) * 255.0).clip(0.0, 255.0)

    if dynamic_ply is None or not dynamic_ply.exists():
        return xyz_s, rgb_s

    pcd_dyn = o3d.io.read_point_cloud(str(dynamic_ply))
    xyz_d = np.asarray(pcd_dyn.points, dtype=np.float32)
    rgb_d = (np.asarray(pcd_dyn.colors, dtype=np.float32) * 255.0).clip(0.0, 255.0)

    if xyz_d.size == 0:
        return xyz_s, rgb_s

    xyz = np.concatenate([xyz_s, xyz_d], axis=0)
    rgb = np.concatenate([rgb_s, rgb_d], axis=0)
    return xyz, rgb


def transform_world_to_camera(xyz_world: np.ndarray, t_cam_to_world: np.ndarray):
    t_world_to_cam = np.linalg.inv(t_cam_to_world)
    xyz_h = np.concatenate([xyz_world.astype(np.float64), np.ones((xyz_world.shape[0], 1), dtype=np.float64)], axis=1)
    xyz_cam_h = (t_world_to_cam @ xyz_h.T).T
    return xyz_cam_h[:, :3].astype(np.float32), t_world_to_cam


def frustum_clip_mask(xyz_cam, fx, fy, cx, cy, width, height, min_depth, max_depth):
    z = xyz_cam[:, 2]
    depth_ok = (z > min_depth) & (z < max_depth)
    if not np.any(depth_ok):
        return depth_ok

    x = xyz_cam[:, 0]
    y = xyz_cam[:, 1]
    z_safe = np.where(depth_ok, z, 1.0)
    u = fx * x / z_safe + cx
    v = fy * y / z_safe + cy
    pix_ok = (u >= 0.0) & (u < float(width)) & (v >= 0.0) & (v < float(height))
    return depth_ok & pix_ok


def fps_downsample_xyzrgb_torch_native(xyzrgb_full, n_samples, device):
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


def main():
    parser = argparse.ArgumentParser(description="Generate one KITTI-360 submap + RGB pair from global map tiles.")
    parser.add_argument("--kitti_root", default="/data_sair/kitti360")
    parser.add_argument("--kitti_maps_root", default="/data_sair/kitti360_maps/data_3d_semantics/train")
    parser.add_argument("--drive", default="2013_05_28_drive_0000_sync")
    parser.add_argument("--camera", choices=["image_00", "image_01"], default="image_00")
    parser.add_argument("--frame", type=int, default=-1, help="Frame id in drive. Use -1 to auto-pick.")
    parser.add_argument(
        "--pose_fallback",
        choices=["error", "nearest"],
        default="nearest",
        help="Behavior when requested frame has no pose in cam0_to_world.txt",
    )
    parser.add_argument(
        "--max_pose_frame_gap",
        type=int,
        default=50,
        help="Maximum allowed |requested_frame - pose_frame| for nearest fallback (-1 disables).",
    )
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=30.0)
    parser.add_argument("--max_radius", type=float, default=80.0, help="Keep points within this camera-frame distance.")
    parser.add_argument("--downsample_points", type=int, default=46080, help="Exact output point count after FPS.")
    parser.add_argument("--fps_device", default="cuda", help="Device for torch FPS (for example cuda or cpu).")
    parser.add_argument("--include_dynamic", action="store_true", help="Merge dynamic tile points with static points.")
    parser.add_argument("--output_root", default="./kitti360_example_pair")
    args = parser.parse_args()

    kitti_root = Path(args.kitti_root)
    maps_root = Path(args.kitti_maps_root)
    drive = args.drive

    perspective_file = kitti_root / "calibration" / "perspective.txt"
    cam0_pose_file = kitti_root / "data_poses" / drive / "cam0_to_world.txt"
    image_dir = kitti_root / "data_2d_raw" / drive / args.camera / "data_rect"

    static_dir = maps_root / drive / "static"
    dynamic_dir = maps_root / drive / "dynamic"

    if not perspective_file.exists():
        raise FileNotFoundError(f"Missing calibration file: {perspective_file}")
    if not cam0_pose_file.exists():
        raise FileNotFoundError(f"Missing pose file: {cam0_pose_file}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image folder: {image_dir}")
    if not static_dir.exists():
        raise FileNotFoundError(f"Missing static map folder: {static_dir}")

    intrinsics = load_perspective_intrinsics(perspective_file, args.camera)
    poses = load_cam0_poses(cam0_pose_file)

    static_ranges = parse_frame_ranges(static_dir)
    dynamic_ranges = parse_frame_ranges(dynamic_dir) if dynamic_dir.exists() else []

    pose_frames = sorted(poses.keys())
    frame = pick_frame(args, poses, static_ranges)
    requested_frame = frame
    pose_gap = 0

    if args.frame >= 0:
        requested_frame = args.frame
        frame, pose_gap = resolve_pose_frame(
            requested_frame,
            pose_frames,
            fallback_mode=args.pose_fallback,
            max_gap=args.max_pose_frame_gap,
        )

    rgb_path = image_dir / f"{frame:010d}.png"
    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB image not found for frame {frame}: {rgb_path}")

    rgb_img = np.asarray(o3d.io.read_image(str(rgb_path)))
    if rgb_img.ndim < 2:
        raise RuntimeError(f"Invalid RGB image shape for {rgb_path}: {rgb_img.shape}")
    img_h, img_w = rgb_img.shape[:2]

    tile_info = find_covering_tile(frame, static_ranges)
    if tile_info is None:
        raise RuntimeError(f"No static map tile covers frame {frame}")
    tile_start, tile_end, static_ply = tile_info

    dynamic_ply = None
    if args.include_dynamic:
        dyn_tile_info = find_covering_tile(frame, dynamic_ranges)
        if dyn_tile_info is not None:
            dynamic_ply = dyn_tile_info[2]

    xyz_world, rgb = load_world_points(static_ply, dynamic_ply)

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
    rgb = rgb[valid]

    xyzrgb_full = np.concatenate([xyz_cam.astype(np.float32), rgb.astype(np.float32)], axis=1)
    if xyzrgb_full.shape[0] < args.downsample_points:
        raise ValueError(
            f"Not enough valid frustum points for exact FPS: {xyzrgb_full.shape[0]} < {args.downsample_points}. "
            f"Try increasing --max_depth/--max_radius or using --include_dynamic."
        )

    xyzrgb = fps_downsample_xyzrgb_torch_native(
        xyzrgb_full,
        n_samples=args.downsample_points,
        device=args.fps_device,
    )

    out_dir = Path(args.output_root) / f"{drive}_{frame:010d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_rgb = out_dir / f"{frame:010d}_rgb.png"
    out_ply = out_dir / f"{frame:010d}_submap_xyzrgb.ply"
    out_meta = out_dir / "metadata.json"

    shutil.copy2(rgb_path, out_rgb)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzrgb[:, :3].astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector((xyzrgb[:, 3:6] / 255.0).astype(np.float64))
    o3d.io.write_point_cloud(str(out_ply), pcd)

    metadata = {
        "dataset": "KITTI-360",
        "drive": drive,
        "camera": args.camera,
        "frame": frame,
        "requested_frame": requested_frame,
        "pose_frame_used": frame,
        "pose_frame_gap": int(pose_gap),
        "rgb_image": str(out_rgb),
        "submap_ply": str(out_ply),
        "num_points_original": int(xyzrgb_full.shape[0]),
        "num_points_saved": int(xyzrgb.shape[0]),
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
        "notes": "Submap points are transformed from global map (world) into the selected camera frame.",
    }

    with open(out_meta, "w", encoding="ascii") as f:
        json.dump(metadata, f, indent=2)

    print(f"Frame: {frame}")
    if requested_frame != frame:
        print(f"Requested frame: {requested_frame} -> using nearest pose frame: {frame} (gap={pose_gap})")
    print(f"RGB: {out_rgb}")
    print(f"Submap: {out_ply}")
    print(f"Metadata: {out_meta}")
    print(f"Points original: {xyzrgb_full.shape[0]}")
    print(f"Points saved: {xyzrgb.shape[0]}")


if __name__ == "__main__":
    main()

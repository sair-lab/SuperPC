import torch
import torch.utils.data as data
from dataset.utils import *
import random
import transforms3d
import copy
import math
import glob
import open3d
import os
import json
import time
from PIL import Image
import torch.nn.functional as F


SHAPENET_13_CAT_IDS = {
    "watercraft": "04530566",
    "rifle": "04090263",
    "display": "03211117",
    "lamp": "03636649",
    "speaker": "03691459",
    "cabinet": "02933112",
    "chair": "03001627",
    "bench": "02828884",
    "car": "02958343",
    "airplane": "02691156",
    "sofa": "04256520",
    "table": "04379243",
    "phone": "04401088",
}
SHAPENET_TEST_CATS = {"airplane", "sofa", "table"}
SHAPENET_TEST_CAT_IDS = {SHAPENET_13_CAT_IDS[name] for name in SHAPENET_TEST_CATS}
SHAPENET_TRAINVAL_CAT_IDS = {
    cat_id
    for name, cat_id in SHAPENET_13_CAT_IDS.items()
    if name not in SHAPENET_TEST_CATS
}

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

# Canonical intrinsics used as robust fallbacks when metadata or intrinsics files are missing.
TARTANAIR_CANONICAL_INTRINSICS = {
    "width": 640.0,
    "height": 480.0,
    "fx": 320.0,
    "fy": 320.0,
    "cx": 320.0,
    "cy": 240.0,
}

KITTI360_CANONICAL_INTRINSICS = {
    "width": 1408.0,
    "height": 376.0,
    "fx": 552.554261,
    "fy": 552.554261,
    "cx": 682.049453,
    "cy": 238.769549,
}

TARTANAIR_TRAIN_SCENES = {
    "abandonedfactory",
    "amusement",
    "carwelding",
    "endofworld",
    "japanesealley",
    "neighborhood",
    "office",
    "office2",
    "seasidetown",
    "soulcity",
}
TARTANAIR_TEST_SCENES = {
    "hospital",
    "abandonedfactory_night",
    "oldtown",
}
TARTANAIR_DISCARD_SCENES = {"westerndesert"}


def _get_tartanair_preprocessed_paths(ply_path, num_in, num_out, seed):
    folder = os.path.dirname(ply_path)
    stem = os.path.splitext(os.path.basename(ply_path))[0]
    tag = f"n{int(num_in)}_m{int(num_out)}_s{int(seed)}"
    input_npy = os.path.join(folder, f"{stem}__imperfect_{tag}.npy")
    gt_npy = os.path.join(folder, f"{stem}__gt_{tag}.npy")
    return input_npy, gt_npy


def _get_kitti_legacy_preprocessed_paths(ply_path, num_in, num_out, seed):
    """Compatibility path for early KITTI-360 preprocess output names.

    Legacy preprocessing removed "_submap_xyzrgb" from the stem, while training
    expects the full stem. We support both styles for cache loading.
    """
    folder = os.path.dirname(ply_path)
    stem = os.path.splitext(os.path.basename(ply_path))[0]
    if stem.endswith("_submap_xyzrgb"):
        stem = stem[: -len("_submap_xyzrgb")]
    tag = f"n{int(num_in)}_m{int(num_out)}_s{int(seed)}"
    input_npy = os.path.join(folder, f"{stem}__imperfect_{tag}.npy")
    gt_npy = os.path.join(folder, f"{stem}__gt_{tag}.npy")
    return input_npy, gt_npy


def _find_any_seed_preprocessed_pair(ply_path, num_in, num_out):
    """Find a cache pair for a sample when exact seed-tag match is unavailable."""
    folder = os.path.dirname(ply_path)
    stem = os.path.splitext(os.path.basename(ply_path))[0]
    stems = [stem]
    if stem.endswith("_submap_xyzrgb"):
        stems.append(stem[: -len("_submap_xyzrgb")])

    tag_prefix = f"n{int(num_in)}_m{int(num_out)}_s"
    for s in stems:
        pattern = os.path.join(folder, f"{s}__imperfect_{tag_prefix}*.npy")
        for input_path in sorted(glob.glob(pattern)):
            gt_path = input_path.replace("__imperfect_", "__gt_")
            if os.path.isfile(gt_path):
                return input_path, gt_path
    return None, None


def _find_matching_file(folder, stem, exts):
    if folder is None:
        return None
    for ext in exts:
        candidate = os.path.join(folder, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def _as_intrinsics_matrix(value):
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return None

    if arr.ndim == 2 and arr.shape == (3, 3):
        return arr.copy()
    if arr.ndim == 2 and arr.shape == (3, 4):
        return arr[:, :3].copy()
    if arr.size == 9:
        return arr.reshape(3, 3).copy()
    if arr.size == 12:
        return arr.reshape(3, 4)[:, :3].copy()
    return None


def _intrinsics_from_meta(meta):
    if not isinstance(meta, dict):
        return None

    def _parse_block(block):
        if block is None:
            return None
        if isinstance(block, dict):
            for key in ["K", "intrinsics", "camera_intrinsics", "P", "projection"]:
                k = _as_intrinsics_matrix(block.get(key, None))
                if k is not None:
                    return k
            fx = block.get("fx", None)
            fy = block.get("fy", None)
            cx = block.get("cx", None)
            cy = block.get("cy", None)
            if None not in (fx, fy, cx, cy):
                k = np.eye(3, dtype=np.float32)
                k[0, 0] = float(fx)
                k[1, 1] = float(fy)
                k[0, 2] = float(cx)
                k[1, 2] = float(cy)
                return k
        return _as_intrinsics_matrix(block)

    for key in ["intrinsics", "camera_intrinsics", "K", "P", "projection", "calib"]:
        k = _parse_block(meta.get(key, None))
        if k is not None:
            return k

    fx = meta.get("fx", None)
    fy = meta.get("fy", None)
    cx = meta.get("cx", None)
    cy = meta.get("cy", None)
    if None not in (fx, fy, cx, cy):
        k = np.eye(3, dtype=np.float32)
        k[0, 0] = float(fx)
        k[1, 1] = float(fy)
        k[0, 2] = float(cx)
        k[1, 2] = float(cy)
        return k

    return None


def _load_intrinsics_matrix(path):
    if path is None or (not os.path.isfile(path)):
        return None

    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
        return _as_intrinsics_matrix(arr)
    if ext == ".npz":
        data = np.load(path)
        for key in ["K", "intrinsics", "arr_0"]:
            if key in data:
                k = _as_intrinsics_matrix(data[key])
                if k is not None:
                    return k
        for key in data.keys():
            k = _as_intrinsics_matrix(data[key])
            if k is not None:
                return k
        return None
    if ext == ".txt":
        return _as_intrinsics_matrix(np.loadtxt(path, dtype=np.float32))
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            k = _intrinsics_from_meta(payload)
            if k is not None:
                return k
        return _as_intrinsics_matrix(payload)
    return None


def _default_intrinsics(width, height):
    k = np.eye(3, dtype=np.float32)
    k[0, 0] = float(width)
    k[1, 1] = float(height)
    k[0, 2] = float(width - 1) / 2.0
    k[1, 2] = float(height - 1) / 2.0
    return k


def _canonical_intrinsics_to_image_size(canonical, width, height):
    """Scale canonical intrinsics into the current image size."""
    base_w = float(canonical["width"])
    base_h = float(canonical["height"])
    out_w = float(width if width and width > 0 else base_w)
    out_h = float(height if height and height > 0 else base_h)

    sx = out_w / max(base_w, 1e-8)
    sy = out_h / max(base_h, 1e-8)

    k = np.eye(3, dtype=np.float32)
    k[0, 0] = float(canonical["fx"]) * sx
    k[1, 1] = float(canonical["fy"]) * sy
    k[0, 2] = float(canonical["cx"]) * sx
    k[1, 2] = float(canonical["cy"]) * sy
    return k


def _dataset_fallback_intrinsics(use_tartanair, use_kitti360, width, height):
    if use_tartanair:
        return _canonical_intrinsics_to_image_size(
            TARTANAIR_CANONICAL_INTRINSICS,
            width,
            height,
        )
    if use_kitti360:
        return _canonical_intrinsics_to_image_size(
            KITTI360_CANONICAL_INTRINSICS,
            width,
            height,
        )
    return None


def _load_and_normalize_image(image_path, out_h, out_w):
    if image_path is None or (not os.path.isfile(image_path)):
        img = torch.zeros((3, out_h, out_w), dtype=torch.float32)
        return img, out_h, out_w

    pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = pil.size
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    img = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    if img.shape[1] != out_h or img.shape[2] != out_w:
        img = F.interpolate(img.unsqueeze(0), size=(out_h, out_w), mode="bilinear", align_corners=False).squeeze(0)

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    img = (img - mean) / std
    return img, orig_h, orig_w


def _get_cat_id_from_name(path):
    base = os.path.basename(path)
    return base.split("-")[0]


def _list_shapenet_files(root, allowed_cat_ids):
    files = sorted(glob.glob(os.path.join(root, "*.npy")))
    return [p for p in files if _get_cat_id_from_name(p) in allowed_cat_ids]


def _resolve_path_from_meta(raw_path, base_dir):
    p = raw_path if os.path.isabs(raw_path) else os.path.join(base_dir, raw_path)
    return os.path.abspath(p)


def _intrinsics_from_tartanair_meta(meta):
    return _intrinsics_from_meta(meta)


def _list_tartanair_records(root, allowed_scenes):
    records = []
    for scene in sorted(allowed_scenes):
        if scene in TARTANAIR_DISCARD_SCENES:
            continue
        scene_dir = os.path.join(root, scene)
        if not os.path.isdir(scene_dir):
            continue
        metadata_files = sorted(glob.glob(os.path.join(scene_dir, "*", "*", "*", "metadata.json")))
        for mp in metadata_files:
            with open(mp, "r", encoding="utf-8") as f:
                meta = json.load(f)
            frame_dir = os.path.dirname(mp)
            rgb_path = _resolve_path_from_meta(meta.get("rgb_image", ""), frame_dir)
            ply_path = _resolve_path_from_meta(meta.get("submap_ply", ""), frame_dir)
            if (not os.path.isfile(rgb_path)) or (not os.path.isfile(ply_path)):
                continue
            records.append(
                {
                    "metadata_path": mp,
                    "scene": meta.get("scene", scene),
                    "difficulty": meta.get("difficulty", "unknown"),
                    "trajectory": meta.get("trajectory", "unknown"),
                    "frame": int(meta.get("frame", -1)),
                    "rgb_path": rgb_path,
                    "ply_path": ply_path,
                    "intrinsics": _intrinsics_from_tartanair_meta(meta),
                }
            )
    return records


def _list_kitti360_records(root):
    records = []
    drive_dirs = sorted(glob.glob(os.path.join(root, "2013_05_28_drive_*_sync")))

    for drive_dir in drive_dirs:
        if not os.path.isdir(drive_dir):
            continue

        drive_name = os.path.basename(drive_dir)
        camera_dir = os.path.join(drive_dir, "image_00")
        if not os.path.isdir(camera_dir):
            continue

        frame_dirs = sorted(glob.glob(os.path.join(camera_dir, "*")))
        for frame_dir in frame_dirs:
            if not os.path.isdir(frame_dir):
                continue

            metadata_path = os.path.join(frame_dir, "metadata.json")
            if not os.path.isfile(metadata_path):
                continue

            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                continue

            rgb_path = _resolve_path_from_meta(meta.get("rgb_image", ""), frame_dir)
            ply_path = _resolve_path_from_meta(meta.get("submap_ply", ""), frame_dir)
            if (not os.path.isfile(rgb_path)) or (not os.path.isfile(ply_path)):
                continue

            intrinsics = _intrinsics_from_meta(meta)

            records.append(
                {
                    "metadata_path": metadata_path,
                    "scene": drive_name,
                    "drive": drive_name,
                    "frame": int(meta.get("frame", -1)),
                    "rgb_path": rgb_path,
                    "ply_path": ply_path,
                    "intrinsics": intrinsics,
                }
            )

    return records


def _split_tartanair_train_val(records, val_ratio, split_seed):
    by_scene = {}
    for rec in records:
        scene = rec.get("scene", "unknown")
        by_scene.setdefault(scene, []).append(rec)

    train_records = []
    val_records = []
    rng = np.random.RandomState(int(split_seed))

    for scene in sorted(by_scene.keys()):
        scene_records = by_scene[scene]
        n = len(scene_records)
        if n <= 1:
            train_records.extend(scene_records)
            continue

        idx = np.arange(n)
        rng.shuffle(idx)
        split_at = max(1, int(n * (1.0 - float(val_ratio))))
        split_at = min(split_at, n - 1)
        train_idx = idx[:split_at]
        val_idx = idx[split_at:]

        train_records.extend([scene_records[i] for i in train_idx])
        val_records.extend([scene_records[i] for i in val_idx])

    train_records = sorted(train_records, key=lambda x: x.get("metadata_path", ""))
    val_records = sorted(val_records, key=lambda x: x.get("metadata_path", ""))
    return train_records, val_records


def _split_shapenet_train_val(root, val_ratio, split_seed):
    train_files, val_files = [], []
    rng = np.random.RandomState(split_seed)

    for cat_id in sorted(SHAPENET_TRAINVAL_CAT_IDS):
        cat_files = sorted(glob.glob(os.path.join(root, f"{cat_id}-*.npy")))
        if len(cat_files) == 0:
            continue
        indices = np.arange(len(cat_files))
        rng.shuffle(indices)
        split_at = max(1, int(len(cat_files) * (1.0 - val_ratio)))
        train_idx = indices[:split_at]
        val_idx = indices[split_at:]
        if len(val_idx) == 0:
            val_idx = train_idx[-1:]
            train_idx = train_idx[:-1]
        train_files.extend([cat_files[i] for i in train_idx])
        val_files.extend([cat_files[i] for i in val_idx])

    return sorted(train_files), sorted(val_files)


def _select_single_sample(items, sample_index):
    if len(items) == 0:
        raise RuntimeError("Cannot select a single sample from an empty split.")
    resolved_index = int(sample_index) % len(items)
    return [items[resolved_index]], resolved_index


def _normalize_pair_from_input(input_pts, gt_pts):
    input_centroid = np.mean(input_pts, axis=0, keepdims=True)
    input_pts = input_pts - input_centroid
    input_furthest_distance = np.max(np.sqrt(np.sum(input_pts ** 2, axis=-1)), keepdims=True)
    input_furthest_distance = max(input_furthest_distance, 1e-8)
    input_pts = input_pts / input_furthest_distance
    gt_pts = (gt_pts - input_centroid) / input_furthest_distance
    return input_pts.astype(np.float32), gt_pts.astype(np.float32)


def _remove_patch_by_random_center(points, rng, occlusion_ratio=0.5):
    n = int(points.shape[0])
    if n <= 1:
        return points

    ratio = float(np.clip(occlusion_ratio, 0.0, 0.99))
    remove_count = int(round(n * ratio))
    remove_count = min(max(remove_count, 0), n - 1)
    if remove_count == 0:
        return points

    center_idx = int(rng.randint(0, n))
    center = points[center_idx]
    d2 = np.sum((points - center[None, :]) ** 2, axis=1)
    order = np.argsort(d2)
    keep_idx = order[remove_count:]
    return points[keep_idx]


def _remove_random_local_areas(points, rng, occlusion_ratio=0.5, num_areas=1):
    n = int(points.shape[0])
    if n <= 1:
        return points

    ratio = float(np.clip(occlusion_ratio, 0.0, 0.99))
    remove_count = int(round(n * ratio))
    remove_count = min(max(remove_count, 0), n - 1)
    if remove_count == 0:
        return points

    num_areas = max(1, int(num_areas))
    remove_mask = np.zeros(n, dtype=bool)
    base = remove_count // num_areas
    rem = remove_count % num_areas

    for area_id in range(num_areas):
        quota = base + (1 if area_id < rem else 0)
        if quota <= 0:
            continue
        available = np.where(~remove_mask)[0]
        if available.size <= 1:
            break
        center_idx = int(available[rng.randint(0, available.size)])
        center = points[center_idx]
        d2 = np.sum((points - center[None, :]) ** 2, axis=1)
        local_order = np.argsort(d2[available])
        take = min(quota, available.size - 1)
        remove_idx = available[local_order[:take]]
        remove_mask[remove_idx] = True

    keep_idx = np.where(~remove_mask)[0]
    return points[keep_idx]


def _resample_to_fixed_count(points, target_n, rng):
    n = int(points.shape[0])
    target_n = int(target_n)
    if n == target_n:
        return points
    if n > target_n:
        idx = rng.choice(n, size=target_n, replace=False)
        return points[idx]
    # n < target_n: sample with replacement to keep batch-collation tensor shapes fixed.
    extra = rng.choice(n, size=(target_n - n), replace=True)
    return np.concatenate([points, points[extra]], axis=0)


def _farthest_point_sampling_np(points, num_samples, rng):
    return _farthest_point_sampling_np_fast(points, num_samples, rng, backend="numpy")


def _farthest_point_sampling_torch(points, num_samples, start_idx, device):
    pts_t = torch.as_tensor(np.ascontiguousarray(points), dtype=torch.float32, device=device)
    n = int(pts_t.shape[0])
    selected = torch.empty((num_samples,), dtype=torch.long, device=device)
    selected[0] = int(start_idx)

    min_dist2 = torch.full((n,), float("inf"), dtype=torch.float32, device=device)
    last = pts_t[selected[0]]

    with torch.no_grad():
        for i in range(1, num_samples):
            d2 = torch.sum((pts_t - last.unsqueeze(0)) ** 2, dim=1)
            min_dist2 = torch.minimum(min_dist2, d2)
            selected[i] = torch.argmax(min_dist2)
            last = pts_t[selected[i]]

    return selected.detach().cpu().numpy().astype(np.int64)


def _farthest_point_sampling_np_fast(points, num_samples, rng, backend="auto"):
    n = int(points.shape[0])
    if num_samples > n:
        raise ValueError(f"Requested {num_samples} samples, but only {n} points are available.")
    if num_samples == n:
        return np.arange(n, dtype=np.int64)

    backend = str(backend).lower()
    start_idx = int(rng.randint(0, n))

    if backend == "auto":
        if torch.cuda.is_available() and n >= 8192 and num_samples >= 2048:
            backend = "torch-cuda"
        elif n >= 8192 and num_samples >= 2048:
            backend = "torch-cpu"
        else:
            backend = "numpy"

    if backend in ("torch", "torch-cuda", "torch-cpu"):
        device = "cuda" if backend in ("torch", "torch-cuda") else "cpu"
        if device == "cuda" and (not torch.cuda.is_available()):
            device = "cpu"
        try:
            return _farthest_point_sampling_torch(points, num_samples, start_idx, device=device)
        except Exception:
            # Fall through to numpy implementation for robustness.
            pass

    if backend != "numpy":
        raise ValueError(f"Unsupported FPS backend: {backend}")

    selected = np.empty(num_samples, dtype=np.int64)
    selected[0] = start_idx
    min_dist2 = np.full(n, np.inf, dtype=np.float64)

    for i in range(1, num_samples):
        last = points[selected[i - 1]]
        d2 = np.sum((points - last[None, :]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, d2)
        selected[i] = int(np.argmax(min_dist2))

    return selected


def _sample_shapenet_pair(
    points,
    num_in,
    num_out,
    rng,
    add_input_noise,
    noise_std_min=0.0025,
    noise_std_max=0.01,
    occlusion_ratio_min=0.05,
    occlusion_ratio_max=0.15,
    occlusion_ratio_fixed=None,
    num_occlusion_areas=0,
    num_occlusion_areas_min=2,
    num_occlusion_areas_max=6,
    return_timing=False,
):
    n = points.shape[0]
    if num_out > n:
        raise ValueError(f"Requested num_out={num_out}, but sample has only {n} points.")

    prep_timing = None

    # Use FPS-style downsampling for more even spatial coverage.
    gt_idx = _farthest_point_sampling_np(points, num_out, rng)
    gt_pts = points[gt_idx].astype(np.float32)

    sparse_start = time.perf_counter() if return_timing else None
    in_idx = _farthest_point_sampling_np(gt_pts, num_in, rng)
    input_pts = gt_pts[in_idx].copy()
    sparse_sec = (time.perf_counter() - sparse_start) if return_timing else 0.0

    if occlusion_ratio_fixed is not None:
        occ_ratio = float(occlusion_ratio_fixed)
    else:
        occ_min = float(min(occlusion_ratio_min, occlusion_ratio_max))
        occ_max = float(max(occlusion_ratio_min, occlusion_ratio_max))
        occ_ratio = float(rng.uniform(occ_min, occ_max))

    incompletion_start = time.perf_counter() if return_timing else None
    if int(num_occlusion_areas) > 0:
        resolved_num_areas = int(num_occlusion_areas)
    else:
        area_min = int(min(num_occlusion_areas_min, num_occlusion_areas_max))
        area_max = int(max(num_occlusion_areas_min, num_occlusion_areas_max))
        resolved_num_areas = int(rng.randint(area_min, area_max + 1))
    input_pts = _remove_random_local_areas(
        input_pts,
        rng,
        occlusion_ratio=occ_ratio,
        num_areas=resolved_num_areas,
    )
    input_pts = _resample_to_fixed_count(input_pts, num_in, rng)
    incompletion_sec = (time.perf_counter() - incompletion_start) if return_timing else 0.0

    noisy_sec = 0.0
    if add_input_noise:
        noisy_start = time.perf_counter() if return_timing else None
        std_min = float(min(noise_std_min, noise_std_max))
        std_max = float(max(noise_std_min, noise_std_max))
        sigma = rng.uniform(std_min, std_max)
        input_pts = input_pts + sigma * rng.randn(*input_pts.shape).astype(np.float32)
        noisy_sec = (time.perf_counter() - noisy_start) if return_timing else 0.0

    input_norm, gt_norm = _normalize_pair_from_input(input_pts, gt_pts)

    if return_timing:
        prep_timing = np.asarray([sparse_sec, incompletion_sec, noisy_sec], dtype=np.float32)
        return input_norm, gt_norm, prep_timing

    return input_norm, gt_norm

def augment_cloud(input, gt, input_rand=None, pc_augm_scale=1.2, pc_augm_rot=True, 
                    pc_rot_scale=90, pc_augm_mirror_prob=0.5, 
                    translation_magnitude=0.1, pc_augm_jitter=False):
    """" Augmentation on XYZ and jittering of everything """
    # Ps is a list of point clouds

    M = transforms3d.zooms.zfdir2mat(1) # M is 3*3 identity matrix
    # scale
    if pc_augm_scale > 1:
        s = random.uniform(1/pc_augm_scale, pc_augm_scale)
        M = np.dot(transforms3d.zooms.zfdir2mat(s), M)

    # rotation
    if pc_augm_rot:
        scale = pc_rot_scale # we assume the scale is given in degrees
        # should range from 0 to 180
        if scale > 0:
            angle = random.uniform(-math.pi, math.pi) * scale / 180.0
            M = np.dot(transforms3d.axangles.axangle2mat([0,1,0], angle), M) 
    
    # mirror
    if pc_augm_mirror_prob > 0: # mirroring x&z, not y
        if random.random() < pc_augm_mirror_prob/2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [1,0,0]), M)
        if random.random() < pc_augm_mirror_prob/2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0,0,1]), M)

    # translation
    translation_sigma = translation_magnitude
    translation_sigma = max(pc_augm_scale, 1) * translation_sigma
    if translation_sigma > 0:
        noise = np.random.normal(scale=translation_sigma, size=(1, 3))
        # noise = noise.astype(Ps[0].dtype)
        
    input[:,:3] = np.dot(input[:,:3], M.T)
    gt[:,:3] = np.dot(gt[:,:3], M.T)
    if input_rand is not None:
        input_rand[:,:3] = np.dot(input_rand[:,:3], M.T)

    if translation_sigma > 0:
        input[:,:3] = input[:,:3] + noise
        gt[:,:3] = gt[:,:3] + noise
        if input_rand is not None:
            input_rand[:,:3] = input_rand[:,:3] + noise

    if pc_augm_jitter:
        sigma = 0.02
        input = input + sigma * np.random.randn(*input.shape).astype(np.float32)
        # gt = gt + np.clip(sigma * np.random.randn(*gt.shape), -1*clip, clip).astype(np.float32)
        if input_rand is not None:
            input_rand = input_rand + sigma * np.random.randn(*input.shape).astype(np.float32)

    return input, gt, input_rand



class PUDataset(data.Dataset):
    def __init__(self, args):
        super(PUDataset, self).__init__()

        self.args = args
        self.use_shapenet = args.dataset == 'shapenet'
        self.use_tartanair = args.dataset == 'tartanair'
        self.use_kitti360 = args.dataset == 'kitti360'
        self.use_map_dataset = self.use_tartanair or self.use_kitti360
        self.use_vision_conditioning = bool(getattr(args, 'use_vision_conditioning', False))
        self.overfit_single_sample = bool(getattr(args, 'overfit_single_sample', False))
        self.overfit_single_sample_index = int(getattr(args, 'overfit_single_sample_index', 0))
        self.vision_image_dir = getattr(args, 'vision_image_dir', None)
        self.vision_intrinsics_dir = getattr(args, 'vision_intrinsics_dir', None)
        self.vision_intrinsics_path = getattr(args, 'vision_intrinsics_path', None)
        self.vision_img_height = int(getattr(args, 'vision_img_height', 518))
        self.vision_img_width = int(getattr(args, 'vision_img_width', 518))
        self.return_prep_timing = bool(getattr(args, 'debug_timing', False))
        self.global_intrinsics = _load_intrinsics_matrix(self.vision_intrinsics_path)
        self.visual_records = []
        self.non_shapenet_images = []
        self.tartanair_records = []
        self.tartanair_use_preprocessed_input = bool(
            getattr(
                args,
                'use_preprocessed_input',
                getattr(
                    args,
                    'tartanair_use_preprocessed_input',
                    getattr(args, 'kitti360_use_preprocessed_input', False),
                ),
            )
        )
        self._warned_missing_tartanair_cache = False
        self._warned_dataset_fallback_intrinsics = False
        self._warned_default_fallback_intrinsics = False

        if self.use_shapenet:
            root = args.shapenet_pc_path
            if not os.path.isdir(root):
                raise FileNotFoundError(f"ShapeNet path not found: {root}")
            split_seed = int(args.shapenet_split_seed)
            train_files, _ = _split_shapenet_train_val(root, args.shapenet_val_ratio, split_seed)
            test_files = _list_shapenet_files(root, SHAPENET_TEST_CAT_IDS)
            if bool(getattr(args, 'shapenet_overfit_test_split', False)):
                if len(test_files) == 0:
                    raise RuntimeError("ShapeNet overfit mode is on, but no test-split files were found.")
                self.file_paths = test_files
            else:
                if len(train_files) == 0:
                    raise RuntimeError("No ShapeNet training files found for the selected categories.")
                self.file_paths = train_files
            if self.overfit_single_sample:
                self.file_paths, resolved_idx = _select_single_sample(
                    self.file_paths,
                    self.overfit_single_sample_index,
                )
                print(
                    "[PUDataset] overfit_single_sample=true -> using ShapeNet sample "
                    f"index {resolved_idx}"
                )
            self.num_out_points = int(getattr(args, 'target_num_points', 0) or 0)
            if self.num_out_points <= 0:
                self.num_out_points = int(args.num_points * args.up_rate)
            if self.num_out_points > args.shapenet_points_per_shape:
                raise ValueError(
                    f"target_num_points ({self.num_out_points}) exceeds shapenet_points_per_shape "
                    f"({args.shapenet_points_per_shape})."
                )
            if self.use_vision_conditioning:
                for p in self.file_paths:
                    stem = os.path.splitext(os.path.basename(p))[0]
                    image_path = _find_matching_file(self.vision_image_dir, stem, IMAGE_EXTS)
                    intr_path = _find_matching_file(self.vision_intrinsics_dir, stem, (".npy", ".npz", ".txt", ".json"))
                    self.visual_records.append((image_path, intr_path))
        elif self.use_map_dataset:
            if self.use_tartanair:
                root = getattr(args, 'tartanair_root', '/data_sair/tartanair_maps')
            else:
                root = getattr(args, 'kitti360_root', '/data_sair/kitti360_maps/submaps')

            if not os.path.isdir(root):
                dataset_name = 'Tartanair' if self.use_tartanair else 'KITTI-360'
                raise FileNotFoundError(f"{dataset_name} root path not found: {root}")

            overfit_mode = bool(getattr(args, 'overfit_test_split', getattr(args, 'tartanair_overfit_test_split', False)))

            if self.use_tartanair:
                if overfit_mode:
                    self.tartanair_records = _list_tartanair_records(root, TARTANAIR_TEST_SCENES)
                else:
                    all_train_records = _list_tartanair_records(root, TARTANAIR_TRAIN_SCENES)
                    train_records, _ = _split_tartanair_train_val(
                        all_train_records,
                        val_ratio=float(getattr(args, 'val_ratio', getattr(args, 'tartanair_val_ratio', 0.1))),
                        split_seed=int(getattr(args, 'split_seed', getattr(args, 'tartanair_split_seed', 21))),
                    )
                    self.tartanair_records = train_records
            else:
                all_records = _list_kitti360_records(root)
                train_records, val_records = _split_tartanair_train_val(
                    all_records,
                    val_ratio=float(getattr(args, 'val_ratio', getattr(args, 'tartanair_val_ratio', 0.1))),
                    split_seed=int(getattr(args, 'split_seed', getattr(args, 'tartanair_split_seed', 21))),
                )
                self.tartanair_records = val_records if overfit_mode else train_records

            if len(self.tartanair_records) == 0:
                if self.use_tartanair:
                    split_name = 'test scenes' if overfit_mode else 'train scenes'
                    raise RuntimeError(f"No Tartanair training samples found for selected {split_name}.")
                split_name = 'validation split' if overfit_mode else 'training split'
                raise RuntimeError(f"No KITTI-360 training samples found for selected {split_name}.")

            if self.overfit_single_sample:
                self.tartanair_records, resolved_idx = _select_single_sample(
                    self.tartanair_records,
                    self.overfit_single_sample_index,
                )
                dataset_name = 'Tartanair' if self.use_tartanair else 'KITTI-360'
                print(
                    f"[PUDataset] overfit_single_sample=true -> using {dataset_name} sample "
                    f"index {resolved_idx}"
                )

            self.num_out_points = int(getattr(args, 'target_num_points', 0) or 0)
            if self.num_out_points <= 0:
                self.num_out_points = int(args.num_points * args.up_rate)

            if self.use_vision_conditioning:
                for rec in self.tartanair_records:
                    self.visual_records.append((rec['rgb_path'], rec.get('intrinsics', None)))
        else:
            # input and gt: (b, n, 3) radius: (b, 1)
            self.input_data, self.gt_data, self.radius_data = load_h5_data(args)
            if self.use_vision_conditioning and self.vision_image_dir is not None:
                all_images = []
                for ext in IMAGE_EXTS:
                    all_images.extend(glob.glob(os.path.join(self.vision_image_dir, f"*{ext}")))
                self.non_shapenet_images = sorted(all_images)

    def _build_vision_tensors(self, index):
        img_h = self.vision_img_height
        img_w = self.vision_img_width

        image_path = None
        intrinsics_override = None
        intr_path = None
        if self.use_shapenet and len(self.visual_records) > 0:
            image_path, intr_path = self.visual_records[index]
        elif self.use_map_dataset and len(self.visual_records) > 0:
            image_path, intrinsics_override = self.visual_records[index]
        elif (not self.use_shapenet) and len(self.non_shapenet_images) == len(self):
            image_path = self.non_shapenet_images[index]

        image_tensor, orig_h, orig_w = _load_and_normalize_image(image_path, img_h, img_w)
        intrinsics = self.global_intrinsics
        if intrinsics is None and intrinsics_override is not None:
            intrinsics = intrinsics_override
        if intrinsics is None:
            intrinsics = _load_intrinsics_matrix(intr_path)
        if intrinsics is None:
            intrinsics = _dataset_fallback_intrinsics(
                use_tartanair=self.use_tartanair,
                use_kitti360=self.use_kitti360,
                width=orig_w,
                height=orig_h,
            )
            if intrinsics is not None and (not self._warned_dataset_fallback_intrinsics):
                dataset_name = 'Tartanair' if self.use_tartanair else ('KITTI-360' if self.use_kitti360 else 'dataset')
                print(
                    "[PUDataset] WARNING: Using hardcoded fallback intrinsics for "
                    f"{dataset_name} sample index {index}. "
                    f"image_path={image_path}, intrinsics_path={intr_path}"
                )
                self._warned_dataset_fallback_intrinsics = True
        if intrinsics is None:
            intrinsics = _default_intrinsics(orig_w if orig_w > 0 else img_w, orig_h if orig_h > 0 else img_h)
            if not self._warned_default_fallback_intrinsics:
                print(
                    "[PUDataset] WARNING: Falling back to generic default intrinsics. "
                    f"sample index={index}, image_path={image_path}, intrinsics_path={intr_path}"
                )
                self._warned_default_fallback_intrinsics = True
        intrinsics = intrinsics.copy().astype(np.float32)

        if orig_w > 0 and orig_h > 0:
            sx = float(img_w) / float(orig_w)
            sy = float(img_h) / float(orig_h)
            intrinsics[0, 0] *= sx
            intrinsics[0, 2] *= sx
            intrinsics[1, 1] *= sy
            intrinsics[1, 2] *= sy

        return image_tensor, torch.from_numpy(intrinsics)

    
    def __len__(self):
        if self.use_shapenet:
            return len(self.file_paths)
        if self.use_map_dataset:
            return len(self.tartanair_records)
        return self.input_data.shape[0]

    def __getitem__(self, index):
        if self.use_shapenet:
            path = self.file_paths[index]
            points = np.load(path).astype(np.float32)
            rng = np.random.RandomState((int(self.args.seed) + index) % (2**31 - 1))
            input, gt = _sample_shapenet_pair(
                points,
                num_in=self.args.num_points,
                num_out=self.num_out_points,
                rng=rng,
                add_input_noise=True,
                noise_std_min=float(getattr(self.args, 'input_noise_std_min', 0.0025)),
                noise_std_max=float(getattr(self.args, 'input_noise_std_max', 0.01)),
                occlusion_ratio_min=float(getattr(self.args, 'input_occlusion_ratio_min', 0.05)),
                occlusion_ratio_max=float(getattr(self.args, 'input_occlusion_ratio_max', 0.15)),
                occlusion_ratio_fixed=getattr(self.args, 'input_occlusion_ratio', None),
                num_occlusion_areas=int(getattr(self.args, 'num_occlusion_areas', getattr(self.args, 'tartanair_num_occlusion_areas', getattr(self.args, 'kitti360_num_occlusion_areas', 0)))),
                num_occlusion_areas_min=int(getattr(self.args, 'num_occlusion_areas_min', getattr(self.args, 'tartanair_num_occlusion_areas_min', getattr(self.args, 'kitti360_num_occlusion_areas_min', 2)))),
                num_occlusion_areas_max=int(getattr(self.args, 'num_occlusion_areas_max', getattr(self.args, 'tartanair_num_occlusion_areas_max', getattr(self.args, 'kitti360_num_occlusion_areas_max', 6)))),
            )
            radius = np.ones((1,), dtype=np.float32)
        elif self.use_map_dataset:
            rec = self.tartanair_records[index]
            rng = np.random.RandomState((int(self.args.seed) + index) % (2**31 - 1))
            if self.tartanair_use_preprocessed_input:
                cache_input_path, cache_gt_path = _get_tartanair_preprocessed_paths(
                    rec['ply_path'],
                    self.args.num_points,
                    self.num_out_points,
                    int(self.args.seed) + index,
                )

                # KITTI-360 compatibility: support legacy cache names and any-seed match.
                if (not self.use_tartanair) and (not (os.path.isfile(cache_input_path) and os.path.isfile(cache_gt_path))):
                    legacy_input, legacy_gt = _get_kitti_legacy_preprocessed_paths(
                        rec['ply_path'],
                        self.args.num_points,
                        self.num_out_points,
                        int(self.args.seed) + index,
                    )
                    if os.path.isfile(legacy_input) and os.path.isfile(legacy_gt):
                        cache_input_path, cache_gt_path = legacy_input, legacy_gt
                    else:
                        any_input, any_gt = _find_any_seed_preprocessed_pair(
                            rec['ply_path'],
                            self.args.num_points,
                            self.num_out_points,
                        )
                        if any_input is not None and any_gt is not None:
                            cache_input_path, cache_gt_path = any_input, any_gt

                if os.path.isfile(cache_input_path) and os.path.isfile(cache_gt_path):
                    input = np.load(cache_input_path).astype(np.float32)
                    gt = np.load(cache_gt_path).astype(np.float32)
                    if input.ndim != 2 or gt.ndim != 2 or input.shape[1] != 3 or gt.shape[1] != 3:
                        dataset_name = 'Tartanair' if self.use_tartanair else 'KITTI-360'
                        raise ValueError(
                            f"Invalid cached shape for {dataset_name} sample: input={input.shape}, gt={gt.shape}"
                        )
                    input, gt = _normalize_pair_from_input(input, gt)
                    prep_timing = np.zeros((3,), dtype=np.float32)
                else:
                    if not self._warned_missing_tartanair_cache:
                        print(
                            "[PUDataset] use_preprocessed_input=true but cache file is missing; "
                            "falling back to online generation. "
                            f"Example missing: {cache_input_path}"
                        )
                        self._warned_missing_tartanair_cache = True

                    pc = open3d.io.read_point_cloud(rec['ply_path'])
                    points = np.asarray(pc.points, dtype=np.float32)
                    if self.return_prep_timing:
                        input, gt, prep_timing = _sample_shapenet_pair(
                            points,
                            num_in=self.args.num_points,
                            num_out=self.num_out_points,
                            rng=rng,
                            add_input_noise=True,
                            noise_std_min=float(getattr(self.args, 'input_noise_std_min', 0.0025)),
                            noise_std_max=float(getattr(self.args, 'input_noise_std_max', 0.01)),
                            occlusion_ratio_min=float(getattr(self.args, 'input_occlusion_ratio_min', 0.05)),
                            occlusion_ratio_max=float(getattr(self.args, 'input_occlusion_ratio_max', 0.15)),
                            occlusion_ratio_fixed=getattr(self.args, 'input_occlusion_ratio', None),
                            num_occlusion_areas=int(getattr(self.args, 'num_occlusion_areas', getattr(self.args, 'tartanair_num_occlusion_areas', getattr(self.args, 'kitti360_num_occlusion_areas', 0)))),
                            num_occlusion_areas_min=int(getattr(self.args, 'num_occlusion_areas_min', getattr(self.args, 'tartanair_num_occlusion_areas_min', getattr(self.args, 'kitti360_num_occlusion_areas_min', 2)))),
                            num_occlusion_areas_max=int(getattr(self.args, 'num_occlusion_areas_max', getattr(self.args, 'tartanair_num_occlusion_areas_max', getattr(self.args, 'kitti360_num_occlusion_areas_max', 6)))),
                            return_timing=True,
                        )
                    else:
                        input, gt = _sample_shapenet_pair(
                            points,
                            num_in=self.args.num_points,
                            num_out=self.num_out_points,
                            rng=rng,
                            add_input_noise=True,
                            noise_std_min=float(getattr(self.args, 'input_noise_std_min', 0.0025)),
                            noise_std_max=float(getattr(self.args, 'input_noise_std_max', 0.01)),
                            occlusion_ratio_min=float(getattr(self.args, 'input_occlusion_ratio_min', 0.05)),
                            occlusion_ratio_max=float(getattr(self.args, 'input_occlusion_ratio_max', 0.15)),
                            occlusion_ratio_fixed=getattr(self.args, 'input_occlusion_ratio', None),
                            num_occlusion_areas=int(getattr(self.args, 'num_occlusion_areas', getattr(self.args, 'tartanair_num_occlusion_areas', getattr(self.args, 'kitti360_num_occlusion_areas', 0)))),
                            num_occlusion_areas_min=int(getattr(self.args, 'num_occlusion_areas_min', getattr(self.args, 'tartanair_num_occlusion_areas_min', getattr(self.args, 'kitti360_num_occlusion_areas_min', 2)))),
                            num_occlusion_areas_max=int(getattr(self.args, 'num_occlusion_areas_max', getattr(self.args, 'tartanair_num_occlusion_areas_max', getattr(self.args, 'kitti360_num_occlusion_areas_max', 6)))),
                        )
            else:
                pc = open3d.io.read_point_cloud(rec['ply_path'])
                points = np.asarray(pc.points, dtype=np.float32)
                if self.return_prep_timing:
                    input, gt, prep_timing = _sample_shapenet_pair(
                        points,
                        num_in=self.args.num_points,
                        num_out=self.num_out_points,
                        rng=rng,
                        add_input_noise=True,
                        noise_std_min=float(getattr(self.args, 'input_noise_std_min', 0.0025)),
                        noise_std_max=float(getattr(self.args, 'input_noise_std_max', 0.01)),
                        occlusion_ratio_min=float(getattr(self.args, 'input_occlusion_ratio_min', 0.05)),
                        occlusion_ratio_max=float(getattr(self.args, 'input_occlusion_ratio_max', 0.15)),
                        occlusion_ratio_fixed=getattr(self.args, 'input_occlusion_ratio', None),
                        num_occlusion_areas=int(getattr(self.args, 'num_occlusion_areas', getattr(self.args, 'tartanair_num_occlusion_areas', getattr(self.args, 'kitti360_num_occlusion_areas', 0)))),
                        num_occlusion_areas_min=int(getattr(self.args, 'num_occlusion_areas_min', getattr(self.args, 'tartanair_num_occlusion_areas_min', getattr(self.args, 'kitti360_num_occlusion_areas_min', 2)))),
                        num_occlusion_areas_max=int(getattr(self.args, 'num_occlusion_areas_max', getattr(self.args, 'tartanair_num_occlusion_areas_max', getattr(self.args, 'kitti360_num_occlusion_areas_max', 6)))),
                        return_timing=True,
                    )
                else:
                    input, gt = _sample_shapenet_pair(
                        points,
                        num_in=self.args.num_points,
                        num_out=self.num_out_points,
                        rng=rng,
                        add_input_noise=True,
                        noise_std_min=float(getattr(self.args, 'input_noise_std_min', 0.0025)),
                        noise_std_max=float(getattr(self.args, 'input_noise_std_max', 0.01)),
                        occlusion_ratio_min=float(getattr(self.args, 'input_occlusion_ratio_min', 0.05)),
                        occlusion_ratio_max=float(getattr(self.args, 'input_occlusion_ratio_max', 0.15)),
                        occlusion_ratio_fixed=getattr(self.args, 'input_occlusion_ratio', None),
                        num_occlusion_areas=int(getattr(self.args, 'num_occlusion_areas', getattr(self.args, 'tartanair_num_occlusion_areas', getattr(self.args, 'kitti360_num_occlusion_areas', 0)))),
                        num_occlusion_areas_min=int(getattr(self.args, 'num_occlusion_areas_min', getattr(self.args, 'tartanair_num_occlusion_areas_min', getattr(self.args, 'kitti360_num_occlusion_areas_min', 2)))),
                        num_occlusion_areas_max=int(getattr(self.args, 'num_occlusion_areas_max', getattr(self.args, 'tartanair_num_occlusion_areas_max', getattr(self.args, 'kitti360_num_occlusion_areas_max', 6)))),
                    )
            radius = np.ones((1,), dtype=np.float32)
        else:
            # (n, 3)
            radius = self.radius_data[index]
            input = copy.deepcopy(self.input_data[index])
            gt = copy.deepcopy(self.gt_data[index])
        # radius = radius * scale
        # augmentation
        # sample_lst = np.random.choice(input.shape[0], input.shape[0], replace=False)
        # input = input[sample_lst, :]

        # In single-sample overfit mode we want a deterministic target/input pair.
        # Applying random geometric augmentation here changes the geometry each step
        # (while the paired image/intrinsics stay fixed), which hurts overfit behavior.
        if not self.overfit_single_sample:
            input, gt, _ = augment_cloud(input, gt, None, pc_augm_jitter=False)
        # to tensor
        input = torch.from_numpy(input)
        gt = torch.from_numpy(gt)
        radius = torch.from_numpy(radius)

        if self.use_vision_conditioning:
            image_tensor, intrinsics = self._build_vision_tensors(index)
            if self.use_map_dataset and self.return_prep_timing:
                return input, gt, radius, image_tensor, intrinsics, torch.from_numpy(prep_timing)
            return input, gt, radius, image_tensor, intrinsics

        if self.use_map_dataset and self.return_prep_timing:
            return input, gt, radius, torch.from_numpy(prep_timing)

        return input, gt, radius
    

class PUDataset_test(data.Dataset):
    def __init__(self, args):
        super(PUDataset_test, self).__init__()

        self.args = args
        self.use_shapenet = args.dataset == 'shapenet'
        self.use_tartanair = args.dataset == 'tartanair'
        self.use_kitti360 = args.dataset == 'kitti360'
        self.use_map_dataset = self.use_tartanair or self.use_kitti360
        self.use_vision_conditioning = bool(getattr(args, 'use_vision_conditioning', False))
        self.overfit_single_sample = bool(getattr(args, 'overfit_single_sample', False))
        self.overfit_single_sample_index = int(getattr(args, 'overfit_single_sample_index', 0))
        self.vision_image_dir = getattr(args, 'vision_image_dir', None)
        self.vision_intrinsics_dir = getattr(args, 'vision_intrinsics_dir', None)
        self.vision_intrinsics_path = getattr(args, 'vision_intrinsics_path', None)
        self.vision_img_height = int(getattr(args, 'vision_img_height', 518))
        self.vision_img_width = int(getattr(args, 'vision_img_width', 518))
        self.global_intrinsics = _load_intrinsics_matrix(self.vision_intrinsics_path)
        self.visual_records = []
        self.non_shapenet_images = []
        self._warned_dataset_fallback_intrinsics = False
        self._warned_default_fallback_intrinsics = False
        self.effective_eval_split = 'default'
        self.tartanair_records = []

        if self.use_shapenet:
            root = args.shapenet_pc_path
            if not os.path.isdir(root):
                raise FileNotFoundError(f"ShapeNet path not found: {root}")
            split_seed = int(args.shapenet_split_seed)
            train_files, val_files = _split_shapenet_train_val(root, args.shapenet_val_ratio, split_seed)
            test_files = _list_shapenet_files(root, SHAPENET_TEST_CAT_IDS)

            # In overfit mode, keep evaluation on the test split to match the training source.
            overfit_mode = bool(getattr(args, 'shapenet_overfit_test_split', False))
            if self.overfit_single_sample:
                source_split = 'test' if overfit_mode else 'train'
                train_source_files = test_files if overfit_mode else train_files
                self.file_paths, resolved_idx = _select_single_sample(
                    train_source_files,
                    self.overfit_single_sample_index,
                )
                self.effective_eval_split = 'single'
                print(
                    "[PUDataset_test] overfit_single_sample=true -> evaluating on ShapeNet "
                    f"{source_split} sample index {resolved_idx}"
                )
            else:
                effective_eval_split = 'test' if overfit_mode else args.shapenet_eval_split
                self.effective_eval_split = effective_eval_split
                if effective_eval_split == 'val':
                    self.file_paths = val_files
                else:
                    self.file_paths = test_files
                    if overfit_mode:
                        max_eval = int(getattr(args, 'shapenet_overfit_eval_max_samples', 0) or 0)
                        if max_eval <= 0:
                            max_eval = len(val_files)
                        if max_eval > 0 and len(self.file_paths) > max_eval:
                            rng = np.random.RandomState(split_seed + 999)
                            indices = np.arange(len(self.file_paths))
                            rng.shuffle(indices)
                            selected = np.sort(indices[:max_eval])
                            self.file_paths = [self.file_paths[i] for i in selected]

            if len(self.file_paths) == 0:
                raise RuntimeError(
                    f"No ShapeNet files found for split '{self.effective_eval_split}'."
                )

            self.num_out_points = int(getattr(args, 'target_num_points', 0) or 0)
            if self.num_out_points <= 0:
                self.num_out_points = int(args.num_points * args.up_rate)
            if self.num_out_points > args.shapenet_points_per_shape:
                raise ValueError(
                    f"target_num_points ({self.num_out_points}) exceeds shapenet_points_per_shape "
                    f"({args.shapenet_points_per_shape})."
                )
            if self.use_vision_conditioning:
                for p in self.file_paths:
                    stem = os.path.splitext(os.path.basename(p))[0]
                    image_path = _find_matching_file(self.vision_image_dir, stem, IMAGE_EXTS)
                    intr_path = _find_matching_file(self.vision_intrinsics_dir, stem, (".npy", ".npz", ".txt", ".json"))
                    self.visual_records.append((image_path, intr_path))
            return

        if self.use_map_dataset:
            if self.use_tartanair:
                root = getattr(args, 'tartanair_root', '/data_sair/tartanair_maps')
            else:
                root = getattr(args, 'kitti360_root', '/data_sair/kitti360_maps/submaps')

            if not os.path.isdir(root):
                dataset_name = 'Tartanair' if self.use_tartanair else 'KITTI-360'
                raise FileNotFoundError(f"{dataset_name} root path not found: {root}")

            overfit_mode = bool(getattr(args, 'overfit_test_split', getattr(args, 'tartanair_overfit_test_split', False)))
            effective_eval_split = 'test' if overfit_mode else getattr(args, 'eval_split', getattr(args, 'tartanair_eval_split', 'val'))
            self.effective_eval_split = effective_eval_split

            if self.use_tartanair:
                if self.overfit_single_sample:
                    if overfit_mode:
                        source_split = 'test'
                        source_records = _list_tartanair_records(root, TARTANAIR_TEST_SCENES)
                    else:
                        source_split = 'train'
                        all_train_records = _list_tartanair_records(root, TARTANAIR_TRAIN_SCENES)
                        source_records, _ = _split_tartanair_train_val(
                            all_train_records,
                            val_ratio=float(getattr(args, 'val_ratio', getattr(args, 'tartanair_val_ratio', 0.1))),
                            split_seed=int(getattr(args, 'split_seed', getattr(args, 'tartanair_split_seed', 21))),
                        )
                    self.tartanair_records, resolved_idx = _select_single_sample(
                        source_records,
                        self.overfit_single_sample_index,
                    )
                    self.effective_eval_split = 'single'
                    print(
                        "[PUDataset_test] overfit_single_sample=true -> evaluating on Tartanair "
                        f"{source_split} sample index {resolved_idx}"
                    )
                elif effective_eval_split == 'val':
                    all_train_records = _list_tartanair_records(root, TARTANAIR_TRAIN_SCENES)
                    _, val_records = _split_tartanair_train_val(
                        all_train_records,
                        val_ratio=float(getattr(args, 'val_ratio', getattr(args, 'tartanair_val_ratio', 0.1))),
                        split_seed=int(getattr(args, 'split_seed', getattr(args, 'tartanair_split_seed', 21))),
                    )
                    self.tartanair_records = val_records
                else:
                    self.tartanair_records = _list_tartanair_records(root, TARTANAIR_TEST_SCENES)
                    if overfit_mode:
                        all_train_records = _list_tartanair_records(root, TARTANAIR_TRAIN_SCENES)
                        _, val_records = _split_tartanair_train_val(
                            all_train_records,
                            val_ratio=float(getattr(args, 'val_ratio', getattr(args, 'tartanair_val_ratio', 0.1))),
                            split_seed=int(getattr(args, 'split_seed', getattr(args, 'tartanair_split_seed', 21))),
                        )
                        max_eval = int(getattr(args, 'overfit_eval_max_samples', getattr(args, 'tartanair_overfit_eval_max_samples', 0)) or 0)
                        if max_eval <= 0:
                            max_eval = len(val_records)
                        if max_eval > 0 and len(self.tartanair_records) > max_eval:
                            rng = np.random.RandomState(int(getattr(args, 'split_seed', getattr(args, 'tartanair_split_seed', 21))) + 999)
                            indices = np.arange(len(self.tartanair_records))
                            rng.shuffle(indices)
                            selected = np.sort(indices[:max_eval])
                            self.tartanair_records = [self.tartanair_records[i] for i in selected]
            else:
                all_records = _list_kitti360_records(root)
                train_records, val_records = _split_tartanair_train_val(
                    all_records,
                    val_ratio=float(getattr(args, 'val_ratio', getattr(args, 'tartanair_val_ratio', 0.1))),
                    split_seed=int(getattr(args, 'split_seed', getattr(args, 'tartanair_split_seed', 21))),
                )
                if self.overfit_single_sample:
                    source_split = 'val' if overfit_mode else 'train'
                    train_source_records = val_records if overfit_mode else train_records
                    self.tartanair_records, resolved_idx = _select_single_sample(
                        train_source_records,
                        self.overfit_single_sample_index,
                    )
                    self.effective_eval_split = 'single'
                    print(
                        "[PUDataset_test] overfit_single_sample=true -> evaluating on KITTI-360 "
                        f"{source_split} sample index {resolved_idx}"
                    )
                else:
                    self.tartanair_records = val_records

                    max_eval = int(getattr(args, 'overfit_eval_max_samples', getattr(args, 'tartanair_overfit_eval_max_samples', 0)) or 0)
                    if max_eval > 0 and len(self.tartanair_records) > max_eval:
                        rng = np.random.RandomState(int(getattr(args, 'split_seed', getattr(args, 'tartanair_split_seed', 21))) + 999)
                        indices = np.arange(len(self.tartanair_records))
                        rng.shuffle(indices)
                        selected = np.sort(indices[:max_eval])
                        self.tartanair_records = [self.tartanair_records[i] for i in selected]

            if len(self.tartanair_records) == 0:
                dataset_name = 'Tartanair' if self.use_tartanair else 'KITTI-360'
                raise RuntimeError(f"No {dataset_name} files found for split '{effective_eval_split}'.")

            self.num_out_points = int(getattr(args, 'target_num_points', 0) or 0)
            if self.num_out_points <= 0:
                self.num_out_points = int(args.num_points * args.up_rate)
            if self.use_vision_conditioning:
                for rec in self.tartanair_records:
                    self.visual_records.append((rec['rgb_path'], rec.get('intrinsics', None)))
            return

        self.input_path = "/data/point_cloud/PUGAN/test_pc_v2/input_2048_4X/input_2048"
        self.gt_path = "/data/point_cloud/PUGAN/test_pc_v2/input_2048_4X/gt_8192"

        # ---- input ----
        plys = glob.glob(os.path.join(self.input_path, "*.xyz"))
        input_data = []
        for ply in plys:
            pc = open3d.io.read_point_cloud(ply)
            points = np.asarray(pc.points, dtype=np.float32)
            input_data.append(points)
        self.input_data = np.stack(input_data, axis=0)
        # ---- input ----

        # ---- gt ----
        plys = glob.glob(os.path.join(self.gt_path, "*.xyz"))
        gt_data = []
        for ply in plys:
            pc = open3d.io.read_point_cloud(ply)
            points = np.asarray(pc.points, dtype=np.float32)
            gt_data.append(points)
        self.gt_data = np.stack(gt_data, axis=0)
        # ---- gt ----

        # ---- name ----
        self.plys = [ply.split("/")[-1][:-4] for ply in plys]
        # ---- name ----

    def _build_vision_tensors(self, index):
        img_h = self.vision_img_height
        img_w = self.vision_img_width

        image_path = None
        intrinsics_override = None
        intr_path = None
        if self.use_shapenet and len(self.visual_records) > 0:
            image_path, intr_path = self.visual_records[index]
        elif self.use_map_dataset and len(self.visual_records) > 0:
            image_path, intrinsics_override = self.visual_records[index]
        elif (not self.use_shapenet) and len(self.non_shapenet_images) == len(self):
            image_path = self.non_shapenet_images[index]

        image_tensor, orig_h, orig_w = _load_and_normalize_image(image_path, img_h, img_w)
        intrinsics = self.global_intrinsics
        if intrinsics is None and intrinsics_override is not None:
            intrinsics = intrinsics_override
        if intrinsics is None:
            intrinsics = _load_intrinsics_matrix(intr_path)
        if intrinsics is None:
            intrinsics = _dataset_fallback_intrinsics(
                use_tartanair=self.use_tartanair,
                use_kitti360=self.use_kitti360,
                width=orig_w,
                height=orig_h,
            )
            if intrinsics is not None and (not self._warned_dataset_fallback_intrinsics):
                dataset_name = 'Tartanair' if self.use_tartanair else ('KITTI-360' if self.use_kitti360 else 'dataset')
                print(
                    "[PUDataset_test] WARNING: Using hardcoded fallback intrinsics for "
                    f"{dataset_name} sample index {index}. "
                    f"image_path={image_path}, intrinsics_path={intr_path}"
                )
                self._warned_dataset_fallback_intrinsics = True
        if intrinsics is None:
            intrinsics = _default_intrinsics(orig_w if orig_w > 0 else img_w, orig_h if orig_h > 0 else img_h)
            if not self._warned_default_fallback_intrinsics:
                print(
                    "[PUDataset_test] WARNING: Falling back to generic default intrinsics. "
                    f"sample index={index}, image_path={image_path}, intrinsics_path={intr_path}"
                )
                self._warned_default_fallback_intrinsics = True
        intrinsics = intrinsics.copy().astype(np.float32)

        if orig_w > 0 and orig_h > 0:
            sx = float(img_w) / float(orig_w)
            sy = float(img_h) / float(orig_h)
            intrinsics[0, 0] *= sx
            intrinsics[0, 2] *= sx
            intrinsics[1, 1] *= sy
            intrinsics[1, 2] *= sy

        return image_tensor, torch.from_numpy(intrinsics)

    def __len__(self):
        if self.use_shapenet:
            return len(self.file_paths)
        if self.use_map_dataset:
            return len(self.tartanair_records)
        return self.input_data.shape[0]

    def __getitem__(self, index):
        if self.use_shapenet:
            path = self.file_paths[index]
            points = np.load(path).astype(np.float32)
            rng = np.random.RandomState((int(self.args.seed) + 100000 + index) % (2**31 - 1))
            input, gt = _sample_shapenet_pair(
                points,
                num_in=self.args.num_points,
                num_out=self.num_out_points,
                rng=rng,
                add_input_noise=False,
            )
            input = torch.from_numpy(input)
            gt = torch.from_numpy(gt)
            if self.use_vision_conditioning:
                image_tensor, intrinsics = self._build_vision_tensors(index)
                return input, gt, image_tensor, intrinsics
            return input, gt

        if self.use_map_dataset:
            rec = self.tartanair_records[index]
            pc = open3d.io.read_point_cloud(rec['ply_path'])
            points = np.asarray(pc.points, dtype=np.float32)
            rng = np.random.RandomState((int(self.args.seed) + 100000 + index) % (2**31 - 1))
            input, gt = _sample_shapenet_pair(
                points,
                num_in=self.args.num_points,
                num_out=self.num_out_points,
                rng=rng,
                add_input_noise=False,
                occlusion_ratio_min=float(getattr(self.args, 'input_occlusion_ratio_min', 0.05)),
                occlusion_ratio_max=float(getattr(self.args, 'input_occlusion_ratio_max', 0.15)),
                occlusion_ratio_fixed=getattr(self.args, 'input_occlusion_ratio', None),
                num_occlusion_areas=int(getattr(self.args, 'num_occlusion_areas', getattr(self.args, 'tartanair_num_occlusion_areas', getattr(self.args, 'kitti360_num_occlusion_areas', 0)))),
                num_occlusion_areas_min=int(getattr(self.args, 'num_occlusion_areas_min', getattr(self.args, 'tartanair_num_occlusion_areas_min', getattr(self.args, 'kitti360_num_occlusion_areas_min', 2)))),
                num_occlusion_areas_max=int(getattr(self.args, 'num_occlusion_areas_max', getattr(self.args, 'tartanair_num_occlusion_areas_max', getattr(self.args, 'kitti360_num_occlusion_areas_max', 6)))),
            )
            input = torch.from_numpy(input)
            gt = torch.from_numpy(gt)
            if self.use_vision_conditioning:
                image_tensor, intrinsics = self._build_vision_tensors(index)
                return input, gt, image_tensor, intrinsics
            return input, gt

        # (n, 3)
        input = copy.deepcopy(self.input_data[index])
        gt = copy.deepcopy(self.gt_data[index])
        # radius = radius * scale
        # to tensor
        input = torch.from_numpy(input)
        gt = torch.from_numpy(gt)

        return input, gt

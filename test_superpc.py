import argparse
import json
import os
import shutil

import numpy as np
import open3d
import torch
from einops import rearrange
from tqdm import tqdm

from args.superpc_args import parse_pc_args
from args.utils import str2bool
from dataset.dataset import (
    TARTANAIR_TEST_SCENES,
    TARTANAIR_TRAIN_SCENES,
    SHAPENET_TEST_CAT_IDS,
    _dataset_fallback_intrinsics,
    _default_intrinsics,
    _find_matching_file,
    _list_kitti360_records,
    _list_shapenet_files,
    _list_tartanair_records,
    _load_and_normalize_image,
    _intrinsics_from_meta,
    _load_intrinsics_matrix,
    _normalize_pair_from_input,
    _remove_patch_by_random_center,
    _resample_to_fixed_count,
    _split_tartanair_train_val,
)
from models.diffusion import SUPERPC, SUPERPC_w_attn
from models.utils import hybrid_initialization, input_scout_fill_initialization, midpoint_interpolate
from test_depth_anything_overlap import resolve_render_assets
from emd_assignment import emd_module
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist, density_aware_chamfer_distance

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
cd_module = chamfer_3DDist()


def parse_args():
    parser = argparse.ArgumentParser(description="Test SuperPC on ShapeNet, Tartanair, or KITTI-360.")
    parser.add_argument("--dataset", default="shapenet", type=str, choices=["shapenet", "tartanair", "kitti360"], help="dataset to test")
    parser.add_argument("--shapenet_pc_path", default="/extra_ws/data/datasets/ShapeNet55-34/shapenet_pc", type=str)
    parser.add_argument("--tartanair_root", default="/data_sair/tartanair_maps", type=str)
    parser.add_argument("--kitti360_root", default="/data_sair/kitti360_maps/submaps", type=str)
    parser.add_argument("--eval_split", default="test", type=str, choices=["train", "val", "test", "all"], help="split for tartanair/kitti360")
    parser.add_argument("--val_ratio", default=0.1, type=float, help="val split ratio for tartanair/kitti360 train/val split")
    parser.add_argument("--split_seed", default=21, type=int, help="seed used for tartanair/kitti360 train/val split")
    parser.add_argument("--max_samples", default=0, type=int, help="if > 0, cap number of tested samples")
    parser.add_argument("--sample_index", default=None, type=int, help="optional sample index in discovered list for tartanair/kitti360")
    parser.add_argument("--metadata_path", default=None, type=str, help="optional metadata.json path for tartanair/kitti360 single sample")
    parser.add_argument("--render_root", default="/extra_ws/data/datasets/shapenet/render_rgb_v2_13cat/image", type=str)

    parser.add_argument("--cat-id", default=None, type=str, help="single-sample category id")
    parser.add_argument("--model-id", default=None, type=str, help="single-sample model id")
    parser.add_argument("--inference_only", default=False, type=str2bool, help="run without GT using --input_pc_path")
    parser.add_argument("--input_pc_path", default=None, type=str, help="path to input point cloud (.npy/.xyz/.txt) for inference-only mode")
    parser.add_argument("--mode", default="easy", type=str, help="render mode for single/full split")
    parser.add_argument("--view-index", default=0, type=int)

    parser.add_argument("--model", default="superpc_w_attn", type=str, help="superpc or superpc_w_attn")
    parser.add_argument("--ckpt_path", required=True, type=str, help="path to trained checkpoint .pth")

    parser.add_argument("--num_points", default=2048, type=int, help="input sparse point count")
    parser.add_argument("--target_num_points", default=8192, type=int, help="target output point count")

    parser.add_argument("--input_noise_std_min", default=0.0025, type=float)
    parser.add_argument("--input_noise_std_max", default=0.01, type=float)
    parser.add_argument("--input_occlusion_ratio_min", default=0.25, type=float)
    parser.add_argument("--input_occlusion_ratio_max", default=0.5, type=float)
    parser.add_argument("--input_occlusion_ratio", default=None, type=float)
    parser.add_argument("--num_occlusion_areas", default=3, type=int)

    parser.add_argument("--use_vision_conditioning", default=True, type=str2bool)
    parser.add_argument("--vision_pretrained_id", default="depth-anything/Depth-Anything-V2-Small-hf", type=str)
    parser.add_argument("--vision_cache_dir", default=None, type=str)
    parser.add_argument("--vision_image_dir", default=None, type=str)
    parser.add_argument("--vision_image_path", default=None, type=str, help="direct RGB image path for inference-only mode")
    parser.add_argument("--vision_intrinsics_dir", default=None, type=str)
    parser.add_argument("--vision_intrinsics_path", default=None, type=str)
    parser.add_argument("--vision_img_height", default=224, type=int)
    parser.add_argument("--vision_img_width", default=224, type=int)
    parser.add_argument("--vision_attn_d_model", default=128, type=int)
    parser.add_argument("--vision_attn_heads", default=4, type=int)

    parser.add_argument("--sampling_steps", default=5, type=int)
    parser.add_argument("--seed", default=21, type=int)
    parser.add_argument("--save_dir", default="output/superpc_test", type=str)
    parser.add_argument("--save_pc", default=True, type=str2bool)
    parser.add_argument("--downsample_method", default="fps", type=str, help="random or fps")
    parser.add_argument("--use_hybrid_initialization", default=False, type=str2bool, help="use hybrid midpoint+scout initialization at inference")
    parser.add_argument("--use_input_scout_fill", default=False, type=str2bool, help="skip midpoint interpolation and fill input to target count with scout points only")
    parser.add_argument("--hybrid_scout_ratio", default=0.3, type=float, help="scout point ratio when hybrid initialization is enabled")
    parser.add_argument(
        "--metric_align_mode",
        default="full_emd",
        type=str,
        choices=["full_emd", "patch_emd", "nn"],
        help="point correspondence mode for MSE/SSE metrics",
    )
    parser.add_argument(
        "--metric_align_anchor",
        default="seed",
        type=str,
        choices=["seed", "generated"],
        help="anchor used to compute EMD/patch-EMD correspondence before MSE/SSE",
    )
    parser.add_argument("--metric_patch_emd_patch_size", default=1024, type=int, help="patch size when metric_align_mode=patch_emd")
    return parser.parse_args()


def farthest_point_sampling(points, num_samples, rng):
    n = points.shape[0]
    if num_samples > n:
        raise ValueError(f"Requested {num_samples} samples, but have only {n} points.")

    selected_indices = [int(rng.randint(n))]
    distances = np.full(n, np.inf, dtype=np.float64)

    while len(selected_indices) < num_samples:
        last_selected = selected_indices[-1]
        dists = np.sqrt(np.sum((points - points[last_selected]) ** 2, axis=1))
        distances = np.minimum(distances, dists)
        next_idx = int(np.argmax(distances))
        selected_indices.append(next_idx)

    return np.asarray(selected_indices, dtype=np.int64)


def sample_shapenet_pair_fps(points, num_in, num_out, rng, model_args):
    n = points.shape[0]
    if num_out > n:
        raise ValueError(f"Requested num_out={num_out}, but sample has only {n} points.")

    gt_idx = farthest_point_sampling(points, num_out, rng)
    gt_pts = points[gt_idx].astype(np.float32)

    in_idx = farthest_point_sampling(gt_pts, num_in, rng)
    input_pts = gt_pts[in_idx].copy()

    occ_fixed = getattr(model_args, "input_occlusion_ratio", None)
    if occ_fixed is not None:
        occ_ratio = float(occ_fixed)
    else:
        occ_min = float(min(model_args.input_occlusion_ratio_min, model_args.input_occlusion_ratio_max))
        occ_max = float(max(model_args.input_occlusion_ratio_min, model_args.input_occlusion_ratio_max))
        occ_ratio = float(rng.uniform(occ_min, occ_max))

    input_raw = _remove_patch_by_random_center(input_pts, rng, occlusion_ratio=occ_ratio)

    std_min = float(min(model_args.input_noise_std_min, model_args.input_noise_std_max))
    std_max = float(max(model_args.input_noise_std_min, model_args.input_noise_std_max))
    sigma = float(rng.uniform(std_min, std_max))
    input_raw = input_raw + sigma * rng.randn(*input_raw.shape).astype(np.float32)

    input_model = _resample_to_fixed_count(input_raw, int(num_in), rng)
    input_model, gt_pts = _normalize_pair_from_input(input_model, gt_pts)
    return input_raw.astype(np.float32), input_model.astype(np.float32), gt_pts.astype(np.float32)


def sample_shapenet_pair_random(points, num_in, num_out, rng, model_args):
    n = points.shape[0]
    if num_out > n:
        raise ValueError(f"Requested num_out={num_out}, but sample has only {n} points.")

    gt_idx = rng.choice(n, size=num_out, replace=False)
    gt_pts = points[gt_idx].astype(np.float32)

    in_idx = rng.choice(num_out, size=num_in, replace=False)
    input_pts = gt_pts[in_idx].copy()

    occ_fixed = getattr(model_args, "input_occlusion_ratio", None)
    if occ_fixed is not None:
        occ_ratio = float(occ_fixed)
    else:
        occ_min = float(min(model_args.input_occlusion_ratio_min, model_args.input_occlusion_ratio_max))
        occ_max = float(max(model_args.input_occlusion_ratio_min, model_args.input_occlusion_ratio_max))
        occ_ratio = float(rng.uniform(occ_min, occ_max))

    input_raw = _remove_patch_by_random_center(input_pts, rng, occlusion_ratio=occ_ratio)

    std_min = float(min(model_args.input_noise_std_min, model_args.input_noise_std_max))
    std_max = float(max(model_args.input_noise_std_min, model_args.input_noise_std_max))
    sigma = float(rng.uniform(std_min, std_max))
    input_raw = input_raw + sigma * rng.randn(*input_raw.shape).astype(np.float32)

    input_model = _resample_to_fixed_count(input_raw, int(num_in), rng)
    input_model, gt_pts = _normalize_pair_from_input(input_model, gt_pts)
    return input_raw.astype(np.float32), input_model.astype(np.float32), gt_pts.astype(np.float32)


def build_model_args(test_args):
    model_args = parse_pc_args([])
    keys = [
        "dataset",
        "shapenet_pc_path",
        "tartanair_root",
        "kitti360_root",
        "eval_split",
        "val_ratio",
        "split_seed",
        "num_points",
        "target_num_points",
        "up_rate",
        "input_noise_std_min",
        "input_noise_std_max",
        "input_occlusion_ratio_min",
        "input_occlusion_ratio_max",
        "input_occlusion_ratio",
        "num_occlusion_areas",
        "use_input_scout_fill",
        "use_hybrid_initialization",
        "hybrid_scout_ratio",
        "midpoint_downsample_mode",
        "midpoint_hybrid_fps_ratio",
        "use_vision_conditioning",
        "vision_pretrained_id",
        "vision_cache_dir",
        "vision_image_dir",
        "vision_intrinsics_dir",
        "vision_intrinsics_path",
        "vision_img_height",
        "vision_img_width",
        "vision_attn_d_model",
        "vision_attn_heads",
    ]
    for k in keys:
        if hasattr(test_args, k):
            setattr(model_args, k, getattr(test_args, k))

    # Legacy fallback only; target_num_points is primary.
    if getattr(model_args, "target_num_points", 0) > 0 and getattr(model_args, "num_points", 0) > 0:
        model_args.up_rate = max(1, int(round(float(model_args.target_num_points) / float(model_args.num_points))))
    return model_args


def load_model(model_args, model_name, ckpt_path, device):
    if model_name == "superpc":
        model = SUPERPC(model_args).to(device)
    elif model_name == "superpc_w_attn":
        model = SUPERPC_w_attn(model_args).to(device)
    else:
        raise ValueError("--model must be 'superpc' or 'superpc_w_attn'")

    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        # Backward compatibility with older torch versions.
        state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def get_alignment_clean(aligner):
    @torch.no_grad()
    def align(noisy, clean):
        noisy = noisy.clone().transpose(1, 2).contiguous()
        clean = clean.clone().transpose(1, 2).contiguous()
        _dis, alignment = aligner(noisy, clean, 0.01, 100)
        return alignment.detach()

    return align


def _align_clean_to_noisy_with_patch_emd(noisy, clean, emd_align, patch_size):
    """Approximate global EMD by running EMD on sorted local chunks."""
    patch_size = int(patch_size)
    b, c, n = noisy.shape

    if patch_size <= 0 or n <= patch_size:
        align_idxs = emd_align(noisy, clean).detach().long()
        align_idxs = align_idxs.unsqueeze(1).expand(-1, c, -1)
        return torch.gather(clean, -1, align_idxs)

    proj_dir = torch.tensor([0.303, 0.505, 0.808], device=noisy.device, dtype=noisy.dtype)
    proj_dir = proj_dir / torch.clamp(torch.norm(proj_dir), min=1e-12)

    noisy_key = (noisy * proj_dir.view(1, 3, 1)).sum(dim=1)
    clean_key = (clean * proj_dir.view(1, 3, 1)).sum(dim=1)
    noisy_sort_idx = torch.argsort(noisy_key, dim=1)
    clean_sort_idx = torch.argsort(clean_key, dim=1)

    noisy_sorted = torch.gather(noisy, 2, noisy_sort_idx.unsqueeze(1).expand(-1, c, -1))
    clean_sorted = torch.gather(clean, 2, clean_sort_idx.unsqueeze(1).expand(-1, c, -1))

    num_chunks = (n + patch_size - 1) // patch_size
    n_padded = num_chunks * patch_size
    pad = n_padded - n
    if pad > 0:
        noisy_pad = noisy_sorted[:, :, -1:].expand(-1, -1, pad)
        clean_pad = clean_sorted[:, :, -1:].expand(-1, -1, pad)
        noisy_sorted = torch.cat([noisy_sorted, noisy_pad], dim=2)
        clean_sorted = torch.cat([clean_sorted, clean_pad], dim=2)

    noisy_patch = noisy_sorted.view(b, c, num_chunks, patch_size).permute(0, 2, 1, 3).contiguous()
    clean_patch = clean_sorted.view(b, c, num_chunks, patch_size).permute(0, 2, 1, 3).contiguous()
    noisy_patch = noisy_patch.view(b * num_chunks, c, patch_size)
    clean_patch = clean_patch.view(b * num_chunks, c, patch_size)

    patch_align = emd_align(noisy_patch, clean_patch).detach().long()
    patch_align = patch_align.unsqueeze(1).expand(-1, c, -1)
    clean_patch_aligned = torch.gather(clean_patch, -1, patch_align)

    clean_sorted_aligned = (
        clean_patch_aligned.view(b, num_chunks, c, patch_size)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(b, c, n_padded)
    )
    clean_sorted_aligned = clean_sorted_aligned[:, :, :n]

    inv_noisy_sort_idx = torch.empty_like(noisy_sort_idx)
    inv_noisy_sort_idx.scatter_(
        1,
        noisy_sort_idx,
        torch.arange(n, device=noisy.device, dtype=noisy_sort_idx.dtype).view(1, -1).expand(b, -1),
    )
    aligned_clean = torch.gather(
        clean_sorted_aligned,
        2,
        inv_noisy_sort_idx.unsqueeze(1).expand(-1, c, -1),
    )
    return aligned_clean


def compute_pc_metrics(
    generated_np,
    gt_np,
    device,
    metric_align_mode="full_emd",
    emd_align_fn=None,
    metric_patch_emd_patch_size=1024,
    align_reference_np=None,
    align_anchor_name="generated",
):
    """Compute simple point-cloud metrics for terminal diagnostics.

        Returns a dict with:
      - mse_nn_g2t: mean squared NN distance generated->gt
            - sse_nn_g2t: summed squared NN distance generated->gt (training-log style scale)
            - mse_aligned / sse_aligned: MSE/SSE after EMD or patch-EMD alignment (if enabled)
            - align_anchor: reference used to compute EMD/patch-EMD correspondence
      - cd_l1: symmetric Chamfer-L1
    - cd_l2: symmetric Chamfer-L2 (same Chamfer3D kernel/reduction as train test-epoch when CUDA is available)
            - dcd: density-aware chamfer distance
            - sse_direct: summed squared error with direct index matching (only if same shape)
    """
    generated = torch.from_numpy(generated_np).float().to(device)
    gt = torch.from_numpy(gt_np).float().to(device)

    if generated.ndim != 2 or gt.ndim != 2 or generated.shape[1] != 3 or gt.shape[1] != 3:
        raise ValueError(f"Invalid shape for metric computation: generated={generated.shape}, gt={gt.shape}")

    with torch.no_grad():
        dmat = torch.cdist(generated.unsqueeze(0), gt.unsqueeze(0), p=2).squeeze(0)
        min_g2t, _ = torch.min(dmat, dim=1)
        min_t2g, _ = torch.min(dmat, dim=0)

        sq_g2t = min_g2t ** 2
        mse_nn_g2t = torch.mean(sq_g2t).item()
        sse_nn_g2t = torch.sum(sq_g2t).item()
        cd_l1 = 0.5 * (torch.mean(min_g2t) + torch.mean(min_t2g)).item()
        cd_l2 = 0.5 * (torch.mean(min_g2t ** 2) + torch.mean(min_t2g ** 2)).item()
        if device.type == "cuda":
            generated_bnc = generated.unsqueeze(0).contiguous()
            gt_bnc = gt.unsqueeze(0).contiguous()
            cd_p, cd_t, _, _ = cd_module(generated_bnc, gt_bnc)
            # Keep identical reduction to train test-epoch: mean((cd_p + cd_t) / 2).
            cd_l2 = ((cd_p + cd_t) / 2.0).mean().item()
            cd_l1 = (0.5 * (torch.sqrt(torch.clamp(cd_p, min=0.0)).mean() + torch.sqrt(torch.clamp(cd_t, min=0.0)).mean())).item()

        sse_direct = None
        if generated.shape == gt.shape:
            # Direct index-wise SSE can be useful for debugging deterministic pair generation.
            sse_direct = torch.sum((generated - gt) ** 2).item()

        dcd = None
        try:
            dcd = density_aware_chamfer_distance(
                generated.unsqueeze(0).contiguous(),
                gt.unsqueeze(0).contiguous(),
            ).item()
        except Exception:
            dcd = None

        mse_aligned = None
        sse_aligned = None
        align_used = "nn"
        align_anchor = "generated"
        if generated.shape[0] == gt.shape[0] and metric_align_mode in ("full_emd", "patch_emd") and emd_align_fn is not None:
            align_ref = generated
            if align_reference_np is not None:
                align_reference = torch.from_numpy(align_reference_np).float().to(device)
                if align_reference.shape == gt.shape:
                    align_ref = align_reference
                    align_anchor = str(align_anchor_name)
                else:
                    align_anchor = "generated"
            else:
                align_anchor = "generated"

            noisy = align_ref.transpose(0, 1).unsqueeze(0).contiguous()
            clean = gt.transpose(0, 1).unsqueeze(0).contiguous()
            if metric_align_mode == "patch_emd":
                clean_aligned = _align_clean_to_noisy_with_patch_emd(
                    noisy=noisy,
                    clean=clean,
                    emd_align=emd_align_fn,
                    patch_size=int(metric_patch_emd_patch_size),
                )
                align_used = "patch_emd"
            else:
                align_idxs = emd_align_fn(noisy, clean).detach().long()
                align_idxs = align_idxs.unsqueeze(1).expand(-1, 3, -1)
                clean_aligned = torch.gather(clean, -1, align_idxs)
                align_used = "full_emd"

            clean_aligned = clean_aligned.squeeze(0).transpose(0, 1).contiguous()
            sq = (generated - clean_aligned) ** 2
            sse_aligned = torch.sum(sq).item()
            mse_aligned = torch.mean(sq).item()

    return {
        "mse_nn_g2t": mse_nn_g2t,
        "sse_nn_g2t": sse_nn_g2t,
        "mse_aligned": mse_aligned,
        "sse_aligned": sse_aligned,
        "align_used": align_used,
        "align_anchor": align_anchor,
        "cd_l1": cd_l1,
        "cd_l2": cd_l2,
        "dcd": dcd,
        "sse_direct": sse_direct,
    }


def load_vision_tensors_for_sample(model_args, cat_id, model_id):
    if not bool(getattr(model_args, "use_vision_conditioning", False)):
        return None, None, None

    render_root = model_args.vision_image_dir
    if render_root is None:
        return None, None, None

    image_path, _metadata_path, _resolved_mode = resolve_render_assets(
        render_root=render_root,
        cat_id=cat_id,
        model_id=model_id,
        preferred_mode=getattr(model_args, "mode", "easy"),
        view_index=int(getattr(model_args, "view_index", 0)),
    )

    img_h = int(getattr(model_args, "vision_img_height", 224))
    img_w = int(getattr(model_args, "vision_img_width", 224))
    image_tensor, orig_h, orig_w = _load_and_normalize_image(image_path, img_h, img_w)

    intrinsics = _load_intrinsics_matrix(getattr(model_args, "vision_intrinsics_path", None))
    if intrinsics is None and getattr(model_args, "vision_intrinsics_dir", None) is not None:
        stem = f"{cat_id}-{model_id}"
        intr_path = _find_matching_file(model_args.vision_intrinsics_dir, stem, (".npy", ".npz", ".txt", ".json"))
        intrinsics = _load_intrinsics_matrix(intr_path)
    if intrinsics is None:
        intrinsics = _default_intrinsics(orig_w if orig_w > 0 else img_w, orig_h if orig_h > 0 else img_h)

    intrinsics = intrinsics.copy().astype(np.float32)
    if orig_w > 0 and orig_h > 0:
        sx = float(img_w) / float(orig_w)
        sy = float(img_h) / float(orig_h)
        intrinsics[0, 0] *= sx
        intrinsics[0, 2] *= sx
        intrinsics[1, 1] *= sy
        intrinsics[1, 2] *= sy

    return image_tensor, torch.from_numpy(intrinsics), image_path


def sample_input_and_gt(pc_path, model_args, sample_seed):
    points = np.load(pc_path).astype(np.float32)
    return sample_input_and_gt_from_points(points, model_args, sample_seed)


def sample_input_and_gt_from_points(points, model_args, sample_seed):
    rng = np.random.RandomState(sample_seed)
    method = str(getattr(model_args, "downsample_method", "fps")).lower()
    if method == "fps":
        input_raw, input_model, gt_pts = sample_shapenet_pair_fps(
            points,
            num_in=int(model_args.num_points),
            num_out=int(model_args.target_num_points),
            rng=rng,
            model_args=model_args,
        )
    elif method == "random":
        input_raw, input_model, gt_pts = sample_shapenet_pair_random(
            points,
            num_in=int(model_args.num_points),
            num_out=int(model_args.target_num_points),
            rng=rng,
            model_args=model_args,
        )
    else:
        raise ValueError("--downsample_method must be 'fps' or 'random'")

    return input_raw, input_model, gt_pts


def _load_points_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        pts = np.load(path)
    elif ext in (".xyz", ".txt"):
        pts = np.loadtxt(path)
    else:
        raise ValueError(f"Unsupported point cloud extension: {ext}. Use .npy, .xyz, or .txt")

    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim == 1:
        if pts.size % 3 != 0:
            raise ValueError(f"Invalid point cloud shape from {path}: expected Nx3 values")
        pts = pts.reshape(-1, 3)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"Invalid point cloud shape from {path}: got {pts.shape}, expected Nx3")
    return pts[:, :3].astype(np.float32)


def _normalize_from_input_only(input_pts):
    center = np.mean(input_pts, axis=0, keepdims=True).astype(np.float32)
    shifted = input_pts - center
    scale = float(np.max(np.linalg.norm(shifted, axis=1)))
    if scale < 1e-8:
        scale = 1.0
    normalized = shifted / scale
    return normalized.astype(np.float32), center.astype(np.float32), scale


def prepare_inference_only_input(input_pc_path, model_args, sample_seed):
    points = _load_points_any(input_pc_path)
    if points.shape[0] < 2:
        raise ValueError(f"Input point cloud must have at least 2 points, got {points.shape[0]}")

    rng = np.random.RandomState(sample_seed)
    input_raw = points.astype(np.float32)
    input_model = _resample_to_fixed_count(input_raw, int(model_args.num_points), rng).astype(np.float32)
    input_model_norm, center, scale = _normalize_from_input_only(input_model)
    return input_raw, input_model_norm, center, scale


def load_vision_tensors_from_image_path(model_args, image_path):
    if image_path is None:
        return None, None, None
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Vision image not found: {image_path}")

    img_h = int(getattr(model_args, "vision_img_height", 224))
    img_w = int(getattr(model_args, "vision_img_width", 224))
    image_tensor, orig_h, orig_w = _load_and_normalize_image(image_path, img_h, img_w)

    intrinsics = _load_intrinsics_matrix(getattr(model_args, "vision_intrinsics_path", None))
    if intrinsics is None:
        intrinsics = _default_intrinsics(orig_w if orig_w > 0 else img_w, orig_h if orig_h > 0 else img_h)

    intrinsics = intrinsics.copy().astype(np.float32)
    if orig_w > 0 and orig_h > 0:
        sx = float(img_w) / float(orig_w)
        sy = float(img_h) / float(orig_h)
        intrinsics[0, 0] *= sx
        intrinsics[0, 2] *= sx
        intrinsics[1, 1] *= sy
        intrinsics[1, 2] *= sy

    return image_tensor, torch.from_numpy(intrinsics), image_path


def run_sampling(model, model_args, input_seed_np, image_tensor, intrinsics, device, steps):
    input_seed = torch.from_numpy(input_seed_np).float().to(device)
    input_seed = rearrange(input_seed, "n c -> c n").unsqueeze(0).contiguous()

    if bool(getattr(model_args, "use_input_scout_fill", False)):
        mid_pts = input_scout_fill_initialization(model_args, input_seed)
    elif bool(getattr(model_args, "use_hybrid_initialization", False)):
        mid_pts = hybrid_initialization(model_args, input_seed)
    else:
        mid_pts = midpoint_interpolate(model_args, input_seed)
    updated = mid_pts.clone()

    img = None
    k = None
    if image_tensor is not None and intrinsics is not None:
        img = image_tensor.unsqueeze(0).float().to(device)
        k = intrinsics.unsqueeze(0).float().to(device)

    with torch.no_grad():
        for i in range(steps):
            alpha = torch.ones(1, device=device) * (float(i) / float(max(steps, 1)))
            pred = model(updated, mid_pts, alpha, image_tensor=img, intrinsics=k)
            updated = updated + (1.0 / float(max(steps, 1))) * pred

    updated = updated.clamp(-1, 1)
    out_np = rearrange(updated.squeeze(0), "c n -> n c").detach().cpu().numpy()
    mid_np = rearrange(mid_pts.squeeze(0), "c n -> n c").detach().cpu().numpy()
    return out_np, mid_np


def save_xyz(path, points):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, points.astype(np.float32), fmt="%.6f")


def parse_cat_model_from_path(pc_path):
    stem = os.path.splitext(os.path.basename(pc_path))[0]
    cat_id, model_id = stem.split("-", 1)
    return cat_id, model_id


def _list_map_records(test_args):
    if test_args.dataset == "tartanair":
        root = test_args.tartanair_root
        split = str(test_args.eval_split).lower()
        if split == "test":
            return _list_tartanair_records(root, TARTANAIR_TEST_SCENES)
        if split == "all":
            train = _list_tartanair_records(root, TARTANAIR_TRAIN_SCENES)
            test = _list_tartanair_records(root, TARTANAIR_TEST_SCENES)
            return sorted(train + test, key=lambda x: x.get("metadata_path", ""))

        train_records = _list_tartanair_records(root, TARTANAIR_TRAIN_SCENES)
        train_records, val_records = _split_tartanair_train_val(
            train_records,
            val_ratio=float(test_args.val_ratio),
            split_seed=int(test_args.split_seed),
        )
        if split == "train":
            return train_records
        if split == "val":
            return val_records
        raise ValueError("For tartanair, --eval_split must be one of train/val/test/all")

    if test_args.dataset == "kitti360":
        root = test_args.kitti360_root
        all_records = _list_kitti360_records(root)
        train_records, val_records = _split_tartanair_train_val(
            all_records,
            val_ratio=float(test_args.val_ratio),
            split_seed=int(test_args.split_seed),
        )
        split = str(test_args.eval_split).lower()
        if split in ("test", "val"):
            return val_records
        if split == "train":
            return train_records
        if split == "all":
            return all_records
        raise ValueError("For kitti360, --eval_split must be one of train/val/test/all")

    return []


def _resolve_record_subset(records, test_args):
    selected = records
    if test_args.metadata_path is not None:
        meta_abs = os.path.abspath(test_args.metadata_path)
        selected = [r for r in records if os.path.abspath(r.get("metadata_path", "")) == meta_abs]
        if len(selected) == 0:
            selected = [_build_record_from_metadata_path(test_args.metadata_path, test_args.dataset)]

    if test_args.sample_index is not None:
        sample_index = int(test_args.sample_index)
        if sample_index < 0 or sample_index >= len(selected):
            raise IndexError(f"sample_index out of range: {sample_index}, available={len(selected)}")
        selected = [selected[sample_index]]

    max_samples = int(test_args.max_samples)
    if max_samples > 0 and len(selected) > max_samples:
        selected = selected[:max_samples]
    return selected


def _build_record_from_metadata_path(metadata_path, dataset_name):
    meta_abs = os.path.abspath(metadata_path)
    if not os.path.isfile(meta_abs):
        raise FileNotFoundError(f"metadata_path does not exist: {metadata_path}")

    with open(meta_abs, "r", encoding="utf-8") as f:
        meta = json.load(f)

    frame_dir = os.path.dirname(meta_abs)
    rgb_raw = str(meta.get("rgb_image", "")).strip()
    ply_raw = str(meta.get("submap_ply", "")).strip()

    rgb_path = rgb_raw if os.path.isabs(rgb_raw) else os.path.abspath(os.path.join(frame_dir, rgb_raw))
    ply_path = ply_raw if os.path.isabs(ply_raw) else os.path.abspath(os.path.join(frame_dir, ply_raw))

    if not os.path.isfile(rgb_path):
        raise FileNotFoundError(f"rgb_image path from metadata is missing: {rgb_path}")
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"submap_ply path from metadata is missing: {ply_path}")

    scene_default = "kitti360" if dataset_name == "kitti360" else "tartanair"
    return {
        "metadata_path": meta_abs,
        "scene": meta.get("scene", scene_default),
        "frame": int(meta.get("frame", -1)),
        "rgb_path": rgb_path,
        "ply_path": ply_path,
        "intrinsics": _intrinsics_from_meta(meta),
    }


def get_samples_to_run(test_args):
    if test_args.dataset == "shapenet":
        if test_args.cat_id and test_args.model_id:
            pc_path = os.path.join(test_args.shapenet_pc_path, f"{test_args.cat_id}-{test_args.model_id}.npy")
            if not os.path.isfile(pc_path):
                raise FileNotFoundError(f"Point cloud not found: {pc_path}")
            return [{"dataset": "shapenet", "pc_path": pc_path}]

        sample_paths = _list_shapenet_files(test_args.shapenet_pc_path, SHAPENET_TEST_CAT_IDS)
        max_samples = int(test_args.max_samples)
        if max_samples > 0:
            sample_paths = sample_paths[:max_samples]
        return [{"dataset": "shapenet", "pc_path": p} for p in sample_paths]

    if test_args.dataset in ("tartanair", "kitti360"):
        records = _list_map_records(test_args)
        records = _resolve_record_subset(records, test_args)
        return [{"dataset": test_args.dataset, "record": r} for r in records]

    raise ValueError(f"Unsupported dataset: {test_args.dataset}")


def load_vision_tensors_for_record(model_args, record):
    if not bool(getattr(model_args, "use_vision_conditioning", False)):
        return None, None, None

    image_path = record.get("rgb_path", None)
    if image_path is None:
        return None, None, None

    img_h = int(getattr(model_args, "vision_img_height", 224))
    img_w = int(getattr(model_args, "vision_img_width", 224))
    image_tensor, orig_h, orig_w = _load_and_normalize_image(image_path, img_h, img_w)

    intrinsics = _load_intrinsics_matrix(getattr(model_args, "vision_intrinsics_path", None))
    if intrinsics is None and getattr(model_args, "vision_intrinsics_dir", None) is not None:
        stem = os.path.splitext(os.path.basename(record.get("metadata_path", "")))[0]
        intr_path = _find_matching_file(model_args.vision_intrinsics_dir, stem, (".npy", ".npz", ".txt", ".json"))
        intrinsics = _load_intrinsics_matrix(intr_path)
    if intrinsics is None:
        intrinsics = record.get("intrinsics", None)
    if intrinsics is None:
        intrinsics = _dataset_fallback_intrinsics(
            use_tartanair=(str(getattr(model_args, "dataset", "")) == "tartanair"),
            use_kitti360=(str(getattr(model_args, "dataset", "")) == "kitti360"),
            width=(orig_w if orig_w > 0 else img_w),
            height=(orig_h if orig_h > 0 else img_h),
        )
    if intrinsics is None:
        intrinsics = _default_intrinsics(orig_w if orig_w > 0 else img_w, orig_h if orig_h > 0 else img_h)

    intrinsics = np.asarray(intrinsics, dtype=np.float32).copy()
    if orig_w > 0 and orig_h > 0:
        sx = float(img_w) / float(orig_w)
        sy = float(img_h) / float(orig_h)
        intrinsics[0, 0] *= sx
        intrinsics[0, 2] *= sx
        intrinsics[1, 1] *= sy
        intrinsics[1, 2] *= sy

    return image_tensor, torch.from_numpy(intrinsics), image_path


def sample_id_from_record(dataset_name, rec, idx):
    scene = str(rec.get("scene", "scene"))
    frame = int(rec.get("frame", -1))
    if frame >= 0:
        return f"{dataset_name}-{scene}-f{frame:06d}-{idx:06d}"
    meta_stem = os.path.splitext(os.path.basename(rec.get("metadata_path", "sample")))[0]
    return f"{dataset_name}-{scene}-{meta_stem}-{idx:06d}"


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_args = build_model_args(args)
    model_args.downsample_method = args.downsample_method
    # expose single-sample render controls to model args for vision loader helper.
    model_args.mode = args.mode
    model_args.view_index = args.view_index

    model = load_model(model_args, args.model, args.ckpt_path, device)

    emd_align_fn = None
    if args.metric_align_mode in ("full_emd", "patch_emd"):
        if device.type != "cuda":
            print("[WARN] EMD alignment metric requested but CUDA is unavailable. Falling back to NN metrics.")
        else:
            try:
                aligner = emd_module.emdModule()
                emd_align_fn = get_alignment_clean(aligner)
            except Exception as exc:
                print(f"[WARN] Failed to initialize EMD aligner for metrics ({exc}). Falling back to NN metrics.")

    if bool(args.inference_only):
        if args.input_pc_path is None:
            raise ValueError("--inference_only true requires --input_pc_path")

        os.makedirs(args.save_dir, exist_ok=True)
        seed_i = int(args.seed)
        sample_name = os.path.splitext(os.path.basename(args.input_pc_path))[0]

        input_raw_np, input_model_np, center_np, scale = prepare_inference_only_input(
            args.input_pc_path,
            model_args,
            seed_i,
        )

        image_tensor, intrinsics, image_path = (None, None, None)
        if bool(args.use_vision_conditioning):
            try:
                if args.vision_image_path is not None:
                    image_tensor, intrinsics, image_path = load_vision_tensors_from_image_path(model_args, args.vision_image_path)
                elif args.cat_id and args.model_id:
                    image_tensor, intrinsics, image_path = load_vision_tensors_for_sample(model_args, args.cat_id, args.model_id)
                else:
                    print("[WARN] --use_vision_conditioning is true but no vision image source was provided. Running without image condition.")
            except Exception as exc:
                print(f"[WARN] Vision condition load failed: {exc}")

        generated_np, seeded_np = run_sampling(
            model,
            model_args,
            input_seed_np=input_model_np,
            image_tensor=image_tensor,
            intrinsics=intrinsics,
            device=device,
            steps=int(args.sampling_steps),
        )
        generated_np = (generated_np * float(scale) + center_np).astype(np.float32)

        print(
            f"[INFO] {sample_name} | input_raw={input_raw_np.shape[0]} pts, "
            f"generated={generated_np.shape[0]} pts (inference_only=true)"
        )

        if bool(args.save_pc):
            sample_dir = os.path.join(args.save_dir, sample_name)
            os.makedirs(sample_dir, exist_ok=True)
            save_xyz(os.path.join(sample_dir, "input_imperfect.xyz"), input_raw_np)
            save_xyz(os.path.join(sample_dir, "input_model.xyz"), input_model_np)
            if bool(getattr(model_args, "use_hybrid_initialization", False)) or bool(getattr(model_args, "use_input_scout_fill", False)):
                save_xyz(os.path.join(sample_dir, "input_model_with_scout.xyz"), seeded_np)
            save_xyz(os.path.join(sample_dir, "generated.xyz"), generated_np)
            if image_path is not None and os.path.isfile(image_path):
                dst = os.path.join(sample_dir, "image" + os.path.splitext(image_path)[1].lower())
                shutil.copy2(image_path, dst)
        return

    samples = get_samples_to_run(args)
    if len(samples) == 0:
        raise RuntimeError(f"No test samples found for dataset={args.dataset}")

    os.makedirs(args.save_dir, exist_ok=True)

    iterator = tqdm(samples, desc=f"Testing ({args.dataset})") if len(samples) > 1 else samples
    for idx, sample in enumerate(iterator):
        seed_i = int(args.seed) + idx

        sample_name = None
        image_tensor, intrinsics, image_path = (None, None, None)

        if sample["dataset"] == "shapenet":
            pc_path = sample["pc_path"]
            cat_id, model_id = parse_cat_model_from_path(pc_path)
            sample_name = f"{cat_id}-{model_id}"
            input_raw_np, input_model_np, gt_np = sample_input_and_gt(pc_path, model_args, seed_i)

            if bool(args.use_vision_conditioning):
                try:
                    image_tensor, intrinsics, image_path = load_vision_tensors_for_sample(model_args, cat_id, model_id)
                except Exception as exc:
                    print(f"[WARN] Vision condition load failed for {sample_name}: {exc}")
        else:
            rec = sample["record"]
            sample_name = sample_id_from_record(sample["dataset"], rec, idx)

            pc = open3d.io.read_point_cloud(rec["ply_path"])
            points = np.asarray(pc.points, dtype=np.float32)
            input_raw_np, input_model_np, gt_np = sample_input_and_gt_from_points(points, model_args, seed_i)

            if bool(args.use_vision_conditioning):
                try:
                    image_tensor, intrinsics, image_path = load_vision_tensors_for_record(model_args, rec)
                except Exception as exc:
                    print(f"[WARN] Vision condition load failed for {sample_name}: {exc}")


        generated_np, seeded_np = run_sampling(
            model,
            model_args,
            input_seed_np=input_model_np,
            image_tensor=image_tensor,
            intrinsics=intrinsics,
            device=device,
            steps=int(args.sampling_steps),
        )

        print(
            f"[INFO] {sample_name} | input_raw={input_raw_np.shape[0]} pts, "
            f"input_model={input_model_np.shape[0]} pts, "
            f"gt={gt_np.shape[0]} pts, generated={generated_np.shape[0]} pts"
        )

        metrics = compute_pc_metrics(
            generated_np,
            gt_np,
            device,
            metric_align_mode=args.metric_align_mode,
            emd_align_fn=emd_align_fn,
            metric_patch_emd_patch_size=int(args.metric_patch_emd_patch_size),
            align_reference_np=(seeded_np if str(args.metric_align_anchor).lower() == "seed" else None),
            align_anchor_name=("seed_input" if str(args.metric_align_anchor).lower() == "seed" else "generated"),
        )
        print(
            f"[METRIC] {sample_name} | "
            f"MSE(gen->gt,NN)={metrics['mse_nn_g2t']:.6e}, "
            f"SSE(gen->gt,NN)={metrics['sse_nn_g2t']:.6e}, "
            f"CD_L1={metrics['cd_l1']:.6e}, CD_L2={metrics['cd_l2']:.6e}"
            + (f", DCD={metrics['dcd']:.6e}" if metrics['dcd'] is not None else "")
        )
        if metrics["mse_aligned"] is not None and metrics["sse_aligned"] is not None:
            print(
                f"[METRIC] {sample_name} | "
                f"MSE(gen->gt,{metrics['align_used']}@{metrics['align_anchor']})={metrics['mse_aligned']:.6e}, "
                f"SSE(gen->gt,{metrics['align_used']}@{metrics['align_anchor']})={metrics['sse_aligned']:.6e}"
            )
        if metrics["sse_direct"] is not None:
            print(f"[METRIC] {sample_name} | SSE(gen->gt,direct-index)={metrics['sse_direct']:.6e}")

        if bool(args.save_pc):
            sample_dir = os.path.join(args.save_dir, sample_name)
            os.makedirs(sample_dir, exist_ok=True)
            save_xyz(os.path.join(sample_dir, "input_imperfect.xyz"), input_raw_np)
            save_xyz(os.path.join(sample_dir, "input_model.xyz"), input_model_np)
            if bool(getattr(model_args, "use_hybrid_initialization", False)) or bool(getattr(model_args, "use_input_scout_fill", False)):
                save_xyz(os.path.join(sample_dir, "input_model_with_scout.xyz"), seeded_np)
            save_xyz(os.path.join(sample_dir, "gt.xyz"), gt_np)
            save_xyz(os.path.join(sample_dir, "generated.xyz"), generated_np)
            if image_path is not None and os.path.isfile(image_path):
                dst = os.path.join(sample_dir, "image" + os.path.splitext(image_path)[1].lower())
                shutil.copy2(image_path, dst)


if __name__ == "__main__":
    main()

import torch
import math
from einops import rearrange
from models.pointops.functions import pointops
import logging
import os
import numpy as np
import random
from torch.autograd import grad
from einops import rearrange, repeat
from sklearn.neighbors import NearestNeighbors



def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return points_grad


def set_seed(seed):
    seed = int(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def index_points(pts, idx):
    """
    Input:
        pts: input points data, [B, C, N]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, C, S, [K]]
    """
    batch_size = idx.shape[0]
    sample_num = idx.shape[1]
    fdim = pts.shape[1]
    reshape = False
    if len(idx.shape) == 3:
        reshape = True
        idx = idx.reshape(batch_size, -1)
    # (b, c, (s k))
    res = torch.gather(pts, 2, idx[:, None].repeat(1, fdim, 1))
    if reshape:
        res = rearrange(res, 'b c (s k) -> b c s k', s=sample_num)

    return res


def FPS(pts, fps_pts_num):
    # input: (b, 3, n)

    # (b, n, 3)
    pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
    # (b, fps_pts_num)
    sample_idx = pointops.furthestsampling(pts_trans, fps_pts_num).long()
    # (b, 3, fps_pts_num)
    sample_pts = index_points(pts, sample_idx)

    return sample_pts


def _random_sample_points(pts, sample_num):
    # pts: (b, 3, n)
    n = int(pts.shape[-1])
    sample_num = int(sample_num)
    if sample_num <= 0:
        return pts[:, :, :0]
    if sample_num >= n:
        return pts

    idx = torch.randint(0, n, (pts.shape[0], sample_num), device=pts.device)
    return index_points(pts, idx)


def _hybrid_downsample_after_midpoint(candidates, target_num, fps_ratio):
    # candidates: (b, 3, n), output: (b, 3, target_num)
    n = int(candidates.shape[-1])
    target_num = int(target_num)
    if target_num <= 0:
        return candidates[:, :, :0]
    if target_num >= n:
        return candidates

    fps_ratio = float(max(0.0, min(1.0, fps_ratio)))
    fps_num = int(round(target_num * fps_ratio))
    fps_num = min(max(fps_num, 0), target_num)
    rand_num = target_num - fps_num

    parts = []
    if rand_num > 0:
        parts.append(_random_sample_points(candidates, rand_num))
    if fps_num > 0:
        parts.append(FPS(candidates, fps_num))

    if len(parts) == 1:
        out = parts[0]
    else:
        out = torch.cat(parts, dim=2)

    out_n = int(out.shape[-1])
    if out_n > target_num:
        out = _random_sample_points(out, target_num)
    elif out_n < target_num:
        pad = _random_sample_points(candidates, target_num - out_n)
        out = torch.cat([out, pad], dim=2)

    return out.contiguous()


def get_knn_pts(k, pts, center_pts, return_idx=False):
    # input: (b, 3, n)

    # (b, n, 3)
    pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
    # (b, m, 3)
    center_pts_trans = rearrange(center_pts, 'b c m -> b m c').contiguous()
    # (b, m, k)
    knn_idx = pointops.knnquery_heap(k, pts_trans, center_pts_trans).long()
    # (b, 3, m, k)
    knn_pts = index_points(pts, knn_idx)

    if return_idx == False:
        return knn_pts
    else:
        return knn_pts, knn_idx


def midpoint_interpolate(args, sparse_pts):
    # sparse_pts: (b, 3, n)

    pts_num = sparse_pts.shape[-1]
    target_num_points = int(getattr(args, 'target_num_points', 0) or 0)
    if target_num_points <= 0:
        target_num_points = int(pts_num * args.up_rate)
    up_pts_num = target_num_points

    ratio = max(1, int(math.ceil(float(up_pts_num) / float(max(pts_num, 1)))))
    k = max(2, int(2 * ratio))
    # (b, 3, n, k)
    knn_pts = get_knn_pts(k, sparse_pts, sparse_pts)
    # (b, 3, n, k)
    repeat_pts = repeat(sparse_pts, 'b c n -> b c n k', k=k)
    # (b, 3, n, k)
    mid_pts = (knn_pts + repeat_pts) / 2.0
    # (b, 3, (n k))
    mid_pts = rearrange(mid_pts, 'b c n k -> b c (n k)').contiguous()
    # note that interpolated_pts already contain sparse_pts
    interpolated_pts = mid_pts

    mode = str(getattr(args, 'midpoint_downsample_mode', 'fps')).strip().lower()
    if mode == 'hybrid':
        fps_ratio = float(getattr(args, 'midpoint_hybrid_fps_ratio', 0.25))
        interpolated_pts = _hybrid_downsample_after_midpoint(interpolated_pts, up_pts_num, fps_ratio)
    else:
        # default: pure FPS for best coverage quality
        interpolated_pts = FPS(interpolated_pts, up_pts_num)

    return interpolated_pts


def hybrid_initialization(args, sparse_pts):
    # sparse_pts: (b, 3, n). Returns (b, 3, target_num_points).
    target_num_points = int(getattr(args, 'target_num_points', 0) or 0)
    if target_num_points <= 0:
        target_num_points = int(sparse_pts.shape[-1] * args.up_rate)

    scout_ratio = float(getattr(args, 'hybrid_scout_ratio', 0.3))
    scout_ratio = max(0.0, min(1.0, scout_ratio))

    scout_num = int(round(target_num_points * scout_ratio))
    scout_num = min(max(scout_num, 0), target_num_points)
    structure_num = max(0, target_num_points - scout_num)

    # Keep geometry-faithful seeds from midpoint interpolation.
    midpoint_pts = midpoint_interpolate(args, sparse_pts)
    if structure_num > 0:
        if structure_num == int(midpoint_pts.shape[-1]):
            structure_pts = midpoint_pts
        else:
            structure_pts = FPS(midpoint_pts, structure_num)
    else:
        structure_pts = midpoint_pts[:, :, :0]

    if scout_num > 0:
        bb_min = sparse_pts.min(dim=2, keepdim=True)[0]
        bb_max = sparse_pts.max(dim=2, keepdim=True)[0]

        rand_u = torch.rand(
            sparse_pts.shape[0],
            sparse_pts.shape[1],
            scout_num,
            device=sparse_pts.device,
            dtype=sparse_pts.dtype,
        )
        scout_pts = bb_min + rand_u * (bb_max - bb_min)
        out = torch.cat([structure_pts, scout_pts], dim=2)
    else:
        out = structure_pts

    out_n = int(out.shape[-1])
    if out_n > target_num_points:
        out = FPS(out, target_num_points)
    elif out_n < target_num_points:
        pad = midpoint_pts if int(midpoint_pts.shape[-1]) > 0 else sparse_pts
        need = target_num_points - out_n
        if need >= int(pad.shape[-1]):
            pad_pts = FPS(pad, int(pad.shape[-1]))
            repeat_times = int(math.ceil(float(need) / float(max(1, pad_pts.shape[-1]))))
            pad_pts = pad_pts.repeat(1, 1, repeat_times)[:, :, :need]
        else:
            pad_pts = FPS(pad, need)
        out = torch.cat([out, pad_pts], dim=2)

    return out.contiguous()


def input_scout_fill_initialization(args, sparse_pts):
    # sparse_pts: (b, 3, n). Returns (b, 3, target_num_points).
    target_num_points = int(getattr(args, 'target_num_points', 0) or 0)
    if target_num_points <= 0:
        target_num_points = int(sparse_pts.shape[-1] * args.up_rate)

    input_num = int(sparse_pts.shape[-1])
    if input_num >= target_num_points:
        return FPS(sparse_pts, target_num_points).contiguous()

    scout_num = target_num_points - input_num
    bb_min = sparse_pts.min(dim=2, keepdim=True)[0]
    bb_max = sparse_pts.max(dim=2, keepdim=True)[0]

    rand_u = torch.rand(
        sparse_pts.shape[0],
        sparse_pts.shape[1],
        scout_num,
        device=sparse_pts.device,
        dtype=sparse_pts.dtype,
    )
    scout_pts = bb_min + rand_u * (bb_max - bb_min)
    out = torch.cat([sparse_pts, scout_pts], dim=2)
    return out.contiguous()



def midpoint_interpolate_v2(up_rate, sparse_pts):
    # sparse_pts: (b, 3, n)

    pts_num = sparse_pts.shape[-1]
    up_pts_num = int(pts_num * up_rate)
    k = int(2 * up_rate)
    # (b, 3, n, k)
    knn_pts = get_knn_pts(k, sparse_pts, sparse_pts)
    # (b, 3, n, k)
    repeat_pts = repeat(sparse_pts, 'b c n -> b c n k', k=k)
    # (b, 3, n, k)
    mid_pts = (knn_pts + repeat_pts) / 2.0
    # (b, 3, (n k))
    mid_pts = rearrange(mid_pts, 'b c n k -> b c (n k)')
    # note that interpolated_pts already contain sparse_pts
    interpolated_pts = mid_pts
    # fps: (b, 3, up_pts_num)
    interpolated_pts = FPS(interpolated_pts, up_pts_num)

    return interpolated_pts


def get_p2p_loss(args, pred_p2p, sample_pts, gt_pts):
    # input: (b, c, n)

    # (b, 3, n)
    knn_pts = get_knn_pts(1, gt_pts, sample_pts).squeeze(-1)
    # (b, 1, n)
    gt_p2p = torch.norm(knn_pts - sample_pts, p=2, dim=1, keepdim=True)
    # gt_p2p = knn_pts - sample_pts
    # (b, 1, n)
    if args.use_smooth_loss == True:
        if args.truncate_distance == True:
            loss = torch.nn.SmoothL1Loss(reduction='none', beta=args.beta)(torch.clamp(pred_p2p, max=args.max_dist), torch.clamp(gt_p2p, max=args.max_dist))
        else:
            loss = torch.nn.SmoothL1Loss(reduction='none', beta=args.beta)(pred_p2p, gt_p2p)
    else:
        if args.truncate_distance == True:
            loss = torch.nn.L1Loss(reduction='none')(torch.clamp(pred_p2p, max=args.max_dist), torch.clamp(gt_p2p, max=args.max_dist))
        else:
            loss = torch.nn.L1Loss(reduction='none')(pred_p2p, gt_p2p)
    # (b, 1, n) -> (b, n) -> (b) -> scalar
    loss = loss.squeeze(1).sum(dim=-1).mean()

    return loss

def self_loss(args, pred_mnf, random_mnf, random_pts):
    # compute grad
    random_grad = gradient(random_pts, random_mnf)

    # manifold loss
    mnfld_loss = (pred_mnf.abs()).mean()

    # eikonal loss
    grad_loss = ((random_grad.norm(2, dim=1) - 1) ** 2).mean()

    loss = mnfld_loss + args.grad_lambda * grad_loss

    return loss, mnfld_loss, grad_loss

def self_loss_v2(args, pred_mnf, pred_pts):
    # compute grad
    pred_grad = gradient(pred_pts, pred_mnf)

    # manifold loss
    mnfld_loss = (pred_grad.abs()).sum()

    return mnfld_loss

def normalize_point_cloud(input, centroid=None, furthest_distance=None):
    # input: (b, 3, n) tensor

    if centroid is None:
        # (b, 3, 1)
        centroid = torch.mean(input, dim=-1, keepdim=True)
    # (b, 3, n)
    input = input - centroid
    if furthest_distance is None:
        # (b, 3, n) -> (b, 1, n) -> (b, 1, 1)
        furthest_distance = torch.max(torch.norm(input, p=2, dim=1, keepdim=True), dim=-1, keepdim=True)[0]
    input = input / furthest_distance

    return input, centroid, furthest_distance


def add_noise(pts, sigma, clamp):
    # input: (b, 3, n)

    assert (clamp > 0)
    jittered_data = torch.clamp(sigma * torch.randn_like(pts), -1 * clamp, clamp).cuda()
    jittered_data += pts

    return jittered_data


# generate patch for test
def extract_knn_patch(k, pts, center_pts):
    # input : (b, 3, n)

    # (n, 3)
    pts_trans = rearrange(pts.squeeze(0), 'c n -> n c').contiguous()
    pts_np = pts_trans.detach().cpu().numpy()
    # (m, 3)
    center_pts_trans = rearrange(center_pts.squeeze(0), 'c m -> m c').contiguous()
    center_pts_np = center_pts_trans.detach().cpu().numpy()
    knn_search = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn_search.fit(pts_np)
    # (m, k)
    knn_idx = knn_search.kneighbors(center_pts_np, return_distance=False)
    # (m, k, 3)
    patches = np.take(pts_np, knn_idx, axis=0)
    patches = torch.from_numpy(patches).float().cuda()
    # (m, 3, k)
    patches = rearrange(patches, 'm k c -> m c k').contiguous()

    return patches


def get_logger(name, log_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')
    # output to console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    # output to log file
    log_name = name + '_log.txt'
    file_handler = logging.FileHandler(os.path.join(log_dir, log_name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_query_points(input_pts, args):
    query_pts = input_pts + (torch.randn_like(input_pts) * args.local_sigma)

    return query_pts

def get_random_points(input_pts, args):
    query_pts = input_pts.repeat(1, 1, args.up_rate-1)
    query_pts = query_pts + (torch.randn_like(query_pts) * args.local_sigma)
    query_pts = torch.cat((input_pts, query_pts), dim=2)

    return query_pts

def get_mid_points(input_pts, args):
    batch_size, dim, pts_num = input_pts.shape
    pts = input_pts + (torch.randn_like(input_pts) * args.local_sigma)
    # sample_global = (torch.rand(batch_size, 3, pts_num//8, device=input_pts.device) * (args.global_sigma*2)) - args.global_sigma
    # pts = torch.cat([pts, sample_global], dim=2)

    return pts



def reset_model_args(train_args, model_args):
    for arg in vars(train_args):
        setattr(model_args, arg, getattr(train_args, arg))

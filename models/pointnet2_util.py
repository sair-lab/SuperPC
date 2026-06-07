import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from models.utils import *
from models.pointops.functions import pointops
from models.vision_condition import VisionCrossAttention

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


# def index_points(points, idx):
#     """

#     Input:
#         points: input points data, [B, N, C]
#         idx: sample index data, [B, S]
#     Return:
#         new_points:, indexed points data, [B, S, C]
#     """
#     device = points.device
#     B = points.shape[0]
#     view_shape = list(idx.shape)
#     view_shape[1:] = [1] * (len(view_shape) - 1)
#     repeat_shape = list(idx.shape)
#     repeat_shape[0] = 1
#     batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
#     new_points = points[batch_indices, idx, :]
#     return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = FPS(xyz.permute(0, 2, 1), S)
        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = pointops.ballquery(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz.permute(0, 2, 1), group_idx)
            grouped_xyz = grouped_xyz.permute(0, 2, 3, 1)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points.permute(0, 2, 1), group_idx)
                grouped_points = grouped_points.permute(0, 2, 3, 1)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            # print('grouped_points before=============', grouped_points.shape)
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            # print('points=============', points.shape)
            # print('new_xyz=============', new_xyz.shape)
            # print('grouped_points after=============', grouped_points.shape)
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat
    

class PointNetSetAbstractionMsgAttn(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsgAttn, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()

        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)
        
        for i in range(len(mlp_list)):
            last_channel = in_channel + 3
            self.attn_blocks.append(AttentionModule(C_in1=mlp_list[i][-1], C_in2=last_channel, C1=mlp_list[i][-1], C2=mlp_list[i][-1], C_out=mlp_list[i][-1]))


    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = FPS(xyz.permute(0, 2, 1), S)
        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = pointops.ballquery(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz.permute(0, 2, 1), group_idx)
            grouped_xyz = grouped_xyz.permute(0, 2, 3, 1)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points.permute(0, 2, 1), group_idx)
                grouped_points = grouped_points.permute(0, 2, 3, 1)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            grouped_points_before = grouped_points
            # print('grouped_points before=============', grouped_points_before.shape)
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            # print('new_points=============', i, new_points.shape, grouped_points_before.shape, grouped_points.shape)
            new_points = self.attn_blocks[i](new_points, grouped_points_before, grouped_points)
            # print('grouped_points after=============', grouped_points.shape)
            # print('new_points=============', new_points.shape)
            
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        # points_all_concat = torch.cat(points_list_all, dim=1)
        return new_xyz, new_points_concat
    

# class PointNetSetAbstractionMsg(nn.Module):
#     def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
#         super(PointNetSetAbstractionMsg, self).__init__()
#         self.npoint = npoint
#         self.radius_list = radius_list
#         self.nsample_list = nsample_list
#         self.conv_blocks = nn.ModuleList()
#         self.bn_blocks = nn.ModuleList()
#         for i in range(len(mlp_list)):
#             convs = nn.ModuleList()
#             bns = nn.ModuleList()
#             last_channel = in_channel + 3
#             for out_channel in mlp_list[i]:
#                 convs.append(nn.Conv2d(last_channel, out_channel, 1))
#                 bns.append(nn.BatchNorm2d(out_channel))
#                 last_channel = out_channel
#             self.conv_blocks.append(convs)
#             self.bn_blocks.append(bns)

#     def forward(self, xyz, points):
#         """
#         Input:
#             xyz: input points position data, [B, C, N]
#             points: input points data, [B, D, N]
#         Return:
#             new_xyz: sampled points position data, [B, C, S]
#             new_points_concat: sample points feature data, [B, D', S]
#         """
#         xyz = xyz.permute(0, 2, 1)
#         if points is not None:
#             points = points.permute(0, 2, 1)

#         B, N, C = xyz.shape
#         S = self.npoint
#         new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
#         new_points_list = []
#         for i, radius in enumerate(self.radius_list):
#             K = self.nsample_list[i]
#             group_idx = query_ball_point(radius, K, xyz, new_xyz)
#             grouped_xyz = index_points(xyz, group_idx)
#             grouped_xyz -= new_xyz.view(B, S, 1, C)
#             if points is not None:
#                 grouped_points = index_points(points, group_idx)
#                 grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
#             else:
#                 grouped_points = grouped_xyz

#             grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
#             for j in range(len(self.conv_blocks[i])):
#                 conv = self.conv_blocks[i][j]
#                 bn = self.bn_blocks[i][j]
#                 grouped_points =  F.relu(bn(conv(grouped_points)))
#             new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
#             new_points_list.append(new_points)

#         new_xyz = new_xyz.permute(0, 2, 1)
#         new_points_concat = torch.cat(new_points_list, dim=1)
#         return new_xyz, new_points_concat



class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp, vision_cfg=None):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

        self.vision_cross_attn = None
        if vision_cfg is not None:
            self.vision_cross_attn = VisionCrossAttention(
                d_point=mlp[-1],
                d_img=vision_cfg['d_img'],
                d_model=vision_cfg['d_model'],
                n_heads=vision_cfg['n_heads'],
            )

    def forward(self, xyz1, xyz2, points1, points2, img_features=None, intrinsics=None):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            a = index_points(points2.permute(0, 2, 1), idx)
            interpolated_points = torch.sum(a.permute(0, 2, 3, 1) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = bn(conv(new_points))
            if i == len(self.mlp_convs) - 1 and self.vision_cross_attn is not None and img_features is not None and intrinsics is not None:
                new_points = self.vision_cross_attn(new_points, xyz1.permute(0, 2, 1).contiguous(), img_features.contiguous(), intrinsics)
            new_points = F.relu(new_points)
        return new_points
    

class PointNetFeaturePropagation_v2(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation_v2, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            a = index_points(points2.permute(0, 2, 1), idx)
            interpolated_points = torch.sum(a.permute(0, 2, 3, 1) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points
    

# class PointNetFeaturePropagation(nn.Module):
#     def __init__(self, in_channel, mlp):
#         super(PointNetFeaturePropagation, self).__init__()
#         self.mlp_convs = nn.ModuleList()
#         self.mlp_bns = nn.ModuleList()
#         last_channel = in_channel
#         for out_channel in mlp:
#             self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
#             self.mlp_bns.append(nn.BatchNorm1d(out_channel))
#             last_channel = out_channel

#     def forward(self, xyz1, xyz2, points1, points2):
#         """
#         Input:
#             xyz1: input points position data, [B, C, N]
#             xyz2: sampled input points position data, [B, C, S]
#             points1: input points data, [B, D, N]
#             points2: input points data, [B, D, S]
#         Return:
#             new_points: upsampled points data, [B, D', N]
#         """
#         xyz1 = xyz1.permute(0, 2, 1)
#         xyz2 = xyz2.permute(0, 2, 1)

#         points2 = points2.permute(0, 2, 1)
#         B, N, C = xyz1.shape
#         _, S, _ = xyz2.shape

#         if S == 1:
#             interpolated_points = points2.repeat(1, N, 1)
#         else:
#             dists = square_distance(xyz1, xyz2)
#             dists, idx = dists.sort(dim=-1)
#             dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

#             dist_recip = 1.0 / (dists + 1e-8)
#             norm = torch.sum(dist_recip, dim=2, keepdim=True)
#             weight = dist_recip / norm
#             interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

#         if points1 is not None:
#             points1 = points1.permute(0, 2, 1)
#             new_points = torch.cat([points1, interpolated_points], dim=-1)
#         else:
#             new_points = interpolated_points

#         new_points = new_points.permute(0, 2, 1)
#         for i, conv in enumerate(self.mlp_convs):
#             bn = self.mlp_bns[i]
#             new_points = F.relu(bn(conv(new_points)))
#         return new_points




class MyGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels):
        super(MyGroupNorm, self).__init__()
        self.num_channels = num_channels - num_channels % num_groups
        self.num_groups = num_groups
        self.group_norm = nn.GroupNorm(self.num_groups, self.num_channels)
    def forward(self, x):
        # x is of shape BCHW
        if x.shape[1] == self.num_channels:
            out = self.group_norm(x)
        else:
            # some times we may attach position info to the end of feature in the channel dimension
            # we do not need to normalize them
            x0 = x[:,0:self.num_channels,:,:]
            res = x[:,self.num_channels:,:,:]
            x0_out = self.group_norm(x0)
            out = torch.cat([x0_out, res], dim=1)
        return out
    
def count_to_mask(count, K):
    # counts is of shape (B, npoint)
    # its value range from 0 to K-1
    # return a mask of shape (B, npoint, K)
    mask = torch.arange(K, device=count.device, dtype=count.dtype)
    B, npoint = count.size()
    mask = mask.repeat(B, npoint).view(B, npoint,-1) # shape (B, npoint, K)
    mask = mask < count.unsqueeze(-1)
    return mask

class AttentionModule(nn.Module):
    def __init__(self, C_in1, C_in2, C1, C2, C_out, attention_bn=True, transform_grouped_feat_out=True, last_activation=True):
        super(AttentionModule, self).__init__()
        C1 = max(C1, 32)
        C2 = max(C2, 32)
        self.feat_conv = nn.Conv2d(C_in1, C1, kernel_size=1, bias=False)
        self.grouped_feat_conv = nn.Conv2d(C_in2, C2, kernel_size=1, bias=False)

        inter_C = min(C1+C2, C_out)
        if attention_bn:
            self.weight_conv = nn.Sequential(
                        nn.ReLU(),
                        MyGroupNorm(min(32, C1+C2), C1+C2),
                        nn.Conv2d(C1+C2, inter_C, kernel_size=1, bias=False),
                        nn.ReLU(),
                        MyGroupNorm(min(32, inter_C), inter_C),
                        nn.Conv2d(inter_C, C_out,kernel_size=1, bias=False))
        else:
            self.weight_conv = nn.Sequential(
                        nn.ReLU(),
                        nn.Conv2d(C1+C2, inter_C, kernel_size=1, bias=False),
                        nn.ReLU(),
                        nn.Conv2d(inter_C, C_out,kernel_size=1, bias=False))

        self.transform_grouped_feat_out = transform_grouped_feat_out
        if transform_grouped_feat_out:
            self.feat_out_conv = [nn.Conv2d(C_out, C_out, kernel_size=1, bias=False)]
            if last_activation:
                if attention_bn:
                    self.feat_out_conv.append(MyGroupNorm(min(32, C_out), C_out))
                self.feat_out_conv.append(nn.ReLU())

            self.feat_out_conv = nn.Sequential(*self.feat_out_conv)


    def forward(self, feat, grouped_feat, grouped_feat_out):
        # feat (B,C_in1,N), acts like query
        # grouped_feat (B,C_in2,N,K), acts like key
        # grouped_feat_out (B,C_out,N,K) # acts like value
        grouped_feat = grouped_feat.permute(0, 1, 3, 2).contiguous()
        grouped_feat_out = grouped_feat_out.permute(0, 1, 3, 2).contiguous()
        K = grouped_feat.shape[-1]
        feat1 = self.feat_conv(feat.unsqueeze(-1)) # (B,C1,N,1)
        feat1 = feat1.repeat(1, 1, 1, K) # (B,C1,N,K)

        grouped_feat1 = self.grouped_feat_conv(grouped_feat) # (B,C2,N,K)

        total_feat = torch.cat([feat1, grouped_feat1], dim=1) # (B,C1+C2,N,K)
        scores = self.weight_conv(total_feat) # (B,C_out,N,K)

        weight = F.softmax(scores, dim=-1).contiguous() # (B,C_out,N,K)

        if self.transform_grouped_feat_out:
            grouped_feat_out = self.feat_out_conv(grouped_feat_out) # B,C_out,N,K
        out = grouped_feat_out * weight # B,C_out,N,K
        out = out.sum(dim=-1) # B,C_out,N
        return out
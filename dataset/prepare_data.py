import os
import open3d as o3d
import numpy as np
import argparse
import torch
from tqdm import tqdm
from einops import rearrange
import open3d
import copy
import os
import glob


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

def add_possion_noise(pts, sigma, clamp, rate=3.0):
    # input: (b, 3, n)

    assert (clamp > 0)
    poisson_distribution = torch.distributions.Poisson(rate)
    jittered_data = torch.clamp(sigma * poisson_distribution.sample(pts.shape), -1 * clamp, clamp).cuda()
    jittered_data += pts

    return jittered_data

def add_laplace_noise(pts, sigma, clamp,loc=0.0,scale=1.0):
    # input: (b, 3, n)

    assert (clamp > 0)
    laplace_distribution = torch.distributions.Laplace(loc=loc, scale=scale)
    jittered_data = torch.clamp(sigma * laplace_distribution.sample(pts.shape), -1 * clamp, clamp).cuda()
    jittered_data += pts

    return jittered_data

def add_gaussian_noise(pts, sigma, clamp):
    # input: (b, 3, n)

    assert (clamp > 0)
    jittered_data = torch.clamp(sigma * torch.randn_like(pts), -1 * clamp, clamp).cuda()
    jittered_data += pts

    return jittered_data

def add_random_noise(pts, sigma, clamp):
    # input: (b, 3, n)

    assert (clamp > 0)
    jittered_data = torch.clamp(sigma * torch.rand_like(pts), -1 * clamp, clamp).cuda()
    jittered_data += pts

    return jittered_data


if __name__ == '__main__':


    mesh_dir="/mnt/SG10T/DataSet/PUGAN/test/mesh"
    save_dir = "/mnt/SG10T/DataSet/PUGAN/temp"
    input_pts_num = 2048
    R=4


    parser = argparse.ArgumentParser(description='PU-GAN Test Data Generation Arguments')
    parser.add_argument('--input_pts_num', default=input_pts_num, type=int, help='the input points number')
    parser.add_argument('--R', default=R, type=int, help='ground truth for up rate')
    parser.add_argument('--jitter_max', default=0.03, type=float, help="jitter max")
    parser.add_argument('--mesh_dir', default=mesh_dir, type=str, help='input mesh dir')
    parser.add_argument('--save_dir', default=save_dir, type=str, help='output point cloud dir')
    args = parser.parse_args()

    gt_pts_num = args.input_pts_num * args.R

    print(f"---- points : {input_pts_num}, R : {R}----")

    dir_name = 'input_' + str(args.input_pts_num)
    if gt_pts_num % args.input_pts_num == 0:
        up_rate = gt_pts_num / args.input_pts_num
        dir_name += '_' + str(int(up_rate)) + 'X'
    else:
        up_rate = gt_pts_num / args.input_pts_num
        dir_name += '_' + str(up_rate) + 'X'
    if args.noise_level != 0:
        dir_name += f'_{args.noise_type}_' + str(args.noise_level)
    input_save_dir = os.path.join(args.save_dir, dir_name, 'input_' + str(args.input_pts_num))
    if not os.path.exists(input_save_dir):
        os.makedirs(input_save_dir)
    gt_save_dir = os.path.join(args.save_dir, dir_name, 'gt_' + str(gt_pts_num))
    if not os.path.exists(gt_save_dir):
        os.makedirs(gt_save_dir)
    mesh_path = glob.glob(os.path.join(args.mesh_dir, '*.off'))
    for i, path in tqdm(enumerate(mesh_path), desc='Processing'):
        pcd_name = path.split('/')[-1].replace(".off", ".xyz")
        mesh = o3d.io.read_triangle_mesh(path)
        # input pcd
        # input_pcd = mesh.sample_points_poisson_disk(args.input_pts_num)
        input_pcd = mesh.sample_points_poisson_disk(args.input_pts_num)
        input_pts = np.array(input_pcd.points)

        input_save_path = os.path.join(input_save_dir, pcd_name)
        np.savetxt(input_save_path, input_pts, fmt='%.6f')

        # gt pcd
        gt_pcd = mesh.sample_points_poisson_disk(gt_pts_num)
        gt_pts = np.array(gt_pcd.points)
        gt_save_path = os.path.join(gt_save_dir, pcd_name)
        np.savetxt(gt_save_path, gt_pts, fmt='%.6f')


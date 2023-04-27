import torch
import numpy as np
import math
import random
import numbers
import random
from itertools import repeat

class Rotate(object):
    r"""Divide point clouds into several patches
    Args:
        pts_clouds (``torch.Tensor``): Batched point clouds. 
            Shape (B, N, 6)
        num_patches (``int``): Number of patches of each point cloud.
    Returns:
        ``torch.Tensor``: Batched point clouds patches.
    """

    def __init__(self, num_patches):
        self.num_patches = num_patches

    def naive_divide(self, pts_clouds):
        pts_xyzs = torch.cat((pts_points.unsqueeze(0), pts_points.unsqueeze(0)), 0)
        # Plot the clustered point cloud in 3D space
        num_patches = self
        pts_batched = torch.empty(pts_xyzs.size()[0]*num_patches, int(pts_xyzs.size()[1]/num_patches), pts_xyzs.size()[2])
        for i in range(pts_xyzs.size()[0]):
            # Medium value and rough patches
            pts_xyz = pts_xyzs[i].squeeze(0)
            part1 = pts_xyz[pts_xyz[:, 0] < pts_xyz[:, 0].sort()[0][int(pts_xyz.size()[0]/2)]]
            part2 = pts_xyz[pts_xyz[:, 0] > pts_xyz[:, 0].sort()[0][int(pts_xyz.size()[0]/2)]]
            midx = pts_xyz[pts_xyz[:, 0] == pts_xyz[:, 0].sort()[0][int(pts_xyz.size()[0]/2)]]

            # Refined patches
            num1 = int(pts_xyz.size()[0]/2 - part1.size()[0])
            num2 = int(pts_xyz.size()[0]/2 - part2.size()[0])
            part1 = torch.cat((part1, midx[:num1]), 0)
            part2 = torch.cat((part2, midx[-num2:]), 0)

            # Batched point clouds
            pts_batched[i] = part1.unsqueeze(0)
            pts_batched[i+1] = part1.unsqueeze(0)


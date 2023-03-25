# import os
# import random
# from copy import copy
# import torch
# from torch.utils.data import Dataset
# import numpy as np
# import h5py
# from tqdm.auto import tqdm

# def get_pcInfo(path, cate_synsetids):
#     basename = os.path.basename(path)
#     dsetname = basename[:basename.rfind('.')]
#     stats_dir = os.path.join(os.path.dirname(path), dsetname + '_stats')
#     os.makedirs(stats_dir, exist_ok=True)


#     stats_save_path = '/home/jared/SAIR_Lab/Super-Map/Super-Map-Fusion-Head-Point-Based-Model_twoBranchsModel/data/shapenet_overfit_flip_stats/stats_02801938.pt'
    
#     if os.path.exists(stats_save_path):
#         stats = torch.load(stats_save_path)
#         return stats

#     with h5py.File(path, 'r') as f:
#         pointclouds = []
#         for synsetid in cate_synsetids:
#             for split in ('train', 'val', 'test'):
#                 pointclouds.append(torch.from_numpy(f[synsetid][split][...]))

#     all_points = torch.cat(pointclouds, dim=0) # (B, N, 6)
#     B, N, _ = all_points.size()
#     mean = all_points.view(B*N, -1).mean(dim=0) # (1, 6)
#     std = all_points.view(-1).std(dim=0)        # (1, )

#     stats = {'mean': mean, 'std': std}
#     torch.save(stats, stats_save_path)
#     return stats


# path = '/home/jared/SAIR_Lab/Super-Map/Super-Map-Fusion-Head-Point-Based-Model_twoBranchsModel/data/tartanair_allEnvs.hdf5'
# cate_synsetids = ['02801938']

# stats_test = get_pcInfo(path, cate_synsetids)


import os
import random
from copy import copy
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
from tqdm.auto import tqdm

def get_pcInfo(path, cate_synsetids):
    # basename = os.path.basename(path)
    # dsetname = basename[:basename.rfind('.')]
    # stats_dir = os.path.join(os.path.dirname(path), dsetname + '_stats')
    # os.makedirs(stats_dir, exist_ok=True)


    # stats_save_path = '/home/jared/SAIR_Lab/Super-Map/Super-Map-Fusion-Head-Point-Based-Model_twoBranchsModel/data/shapenet_OVERFIT_stats/stats_02801938.pt'
    
    # if os.path.exists(stats_save_path):
    #     stats = torch.load(stats_save_path)
    #     return stats

    with h5py.File(path, 'r') as f:
        means = torch.empty((0, 6), dtype=torch.float64)
        stds = torch.empty((0, 1), dtype=torch.float64)
        nums = []
        for synsetid in cate_synsetids:
            for split in ('train', 'val', 'test'):
                pointclouds = torch.from_numpy(f[synsetid][split][0:2])
                B, N, _ = pointclouds.size()
                max = pointclouds.view(B*N, -1).max(dim=0)
                mean = pointclouds.view(B*N, -1).mean(dim=0) # (1, 6)
                std = pointclouds.view(-1).std(dim=0)        # (1, )
                mean = N*torch.reshape(mean, (1, 6))
                std = N*torch.reshape(std, (1, 1))
                nums.append(N)
                means = torch.cat((means, mean), 0)
                stds = torch.cat((stds, std), 0)

    all_means = torch.sum(means, dim=0)/sum(nums) # (B, N, 6)
    all_stds = torch.sum(stds, dim=0)/sum(nums) # (B, N, 6)

    

    stats = {'mean': mean, 'std': std}
    # torch.save(stats, stats_save_path)
    return stats


path = '/home/jared/SAIR_Lab/Super-Map/Super-Map-Fusion-Head-Point-Based-Model_twoBranchsModel/data/tartanair_allEnvs.hdf5'
cate_synsetids = ['02801938']

stats_test = get_pcInfo(path, cate_synsetids)
print(stats_test)
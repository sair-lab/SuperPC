import RGBD2PointCloud as RGBD2PtsCloud
import numpy as np
import torch


# Briefly view the processed point cloud data
# frame = 328 # seasonsforest_winter: 328 - tree
# frame = 198 # hospital - P000: 198 - room with sofa
frame = 266 # hospital - P000: 286 - long way
rgbd2PtsCloud = RGBD2PtsCloud.RGBD2PtsCloud(frame)
pcd_dense, pcd_sparse = rgbd2PtsCloud.convert6()
# rgbd2PtsCloud.view_image()
# rgbd2PtsCloud.view_PtsCloud()
# print(pcd_sparse)
# Save pcd file
# o3d.io.write_point_cloud('./PtsDataFunc/P000/train/frame' + str(frame) + '.pcd', pcd_sparse, write_ascii=True, compressed=False, print_progress=False)

pts_sparse_xyz = list(pcd_sparse.points)
# print(pts_sparse_xyz[0])
print('RGB max:')
print(np.max(np.array(pcd_sparse.colors)))
print('RGB min:')
print(np.min(np.array(pcd_sparse.colors)))
print('')
print('XYZ max:')
print(np.max(np.array(pcd_sparse.points)))
print('XYZ max:')
print(np.min(np.array(pcd_sparse.points)))

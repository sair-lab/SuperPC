import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as img
import open3d as o3d
import RGBD2PointCloud as RGBD2PtsCloud
import h5py


# region generate RGB point cloud dataset
# read point clouds
train_data = []
val_data = []
train_colors_data = []
val_colors_data = []
frame_num1 = 360
for frame in np.linspace(100, 100+frame_num1-1, frame_num1):
# for frame in np.linspace(136, 136+frame_num1-1, frame_num1):
    rgbd2PtsCloud = RGBD2PtsCloud.RGBD2PtsCloud(int(frame))
    pcd_dense, _ = rgbd2PtsCloud.convert6()
    # save coordinates data
    pts_dense = list(pcd_dense.points)
    train_data.append(pts_dense)
    # save RGB data
    pts_colors_dense = list(pcd_dense.colors)
    train_colors_data.append(pts_colors_dense)

frame_num2 = 36
for frame in np.linspace(100+frame_num1, 100+frame_num1+frame_num2-1, frame_num2):
# for frame in np.linspace(186, 186+frame_num2-1, frame_num2):
    rgbd2PtsCloud = RGBD2PtsCloud.RGBD2PtsCloud(int(frame))
    pcd_sparse, _ = rgbd2PtsCloud.convert6()
    # save coordinates data
    pts_sparse = list(pcd_sparse.points)
    val_data.append(pts_sparse)
    # save RGB data
    pts_colors_sparse = list(pcd_sparse.colors)
    val_colors_data.append(pts_colors_sparse)

# Rearrange point cloud data size
train_data_ori = np.array(train_data) # data in wrong shape
val_data_ori = np.array(val_data) # val data in wrong shape
train_colors_data_ori = np.array(train_colors_data) # data in wrong shape
val_colors_data_ori = np.array(val_colors_data) # val data in wrong shape
# Reshape
len_train = train_data_ori.shape[1]
len_val = val_data_ori.shape[1]
# train_data = train_data_ori.reshape((len_train, frame_num, 3))
# val_data = val_data_ori.reshape((len_val, frame_num, 3))
# train_colors_data = train_colors_data_ori.reshape((len_train, frame_num, 3))
# val_colors_data = val_colors_data_ori.reshape((len_val, frame_num, 3))
train_data = train_data_ori.reshape((frame_num1, len_train, 3))
val_data = val_data_ori.reshape((frame_num2, len_val, 3))
train_colors_data = train_colors_data_ori.reshape((frame_num1, len_train, 3))
val_colors_data = val_colors_data_ori.reshape((frame_num2, len_val, 3))
# print(train_data.shape)
# print(val_data.shape)
# Concatenate XYZ abd RGB together
train_XYZRGB_data = np.concatenate((train_data, train_colors_data), axis=2)
val_XYZRGB_data = np.concatenate((val_data, val_colors_data), axis=2)
print(train_XYZRGB_data.shape)
print(val_XYZRGB_data.shape)
# print(val_data_ori[0][0])
# print(val_data[:][0][0])



# SAVE data into hdf5 file
f = h5py.File("./data/shapenet_overfit_flip.hdf5", "r+")
# 1. For XYZ pts only 
print(list(f.keys()))
hospital_dataset = f['02773838']
# (1) for 'train' data
del f['02773838']['train'] # Delete the old 'train' point cloud data
f['02773838'].create_dataset('train', data=train_data) # Add the new 'train' point cloud data
# (2) for 'val' data
del f['02773838']['val'] # Delete the old 'val' point cloud data
f['02773838'].create_dataset('val', data=val_data) # Add the new 'val' point cloud data
print(list(hospital_dataset.keys()))
# (3) for 'test' data
del f['02773838']['test'] # Delete the old 'test' point cloud data
f['02773838'].create_dataset('test', data=val_data) # Add the new 'test' point cloud data
print(list(hospital_dataset.keys()))

# 2. For XYZRGB pts 
print(list(f.keys()))
hospital_XYZRGB_dataset = f['02801938']
# (1) for 'train' data
del f['02801938']['train'] # Delete the old 'train' point cloud data
f['02801938'].create_dataset('train', data=train_XYZRGB_data) # Add the new 'train' point cloud data
# (2) for 'val' data
del f['02801938']['val'] # Delete the old 'val' point cloud data
f['02801938'].create_dataset('val', data=val_XYZRGB_data) # Add the new 'val' point cloud data
print(list(hospital_XYZRGB_dataset.keys()))
# (3) for 'test' data
del f['02801938']['test'] # Delete the old 'test' point cloud data
f['02801938'].create_dataset('test', data=val_XYZRGB_data) # Add the new 'test' point cloud data
print(list(hospital_XYZRGB_dataset.keys()))


# endregion

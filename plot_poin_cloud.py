import numpy as np
import open3d as o3d


# data = np.load('/home/jared/Downloads/bunny/data/bun045.ply')

# path = os.getcwd() + '/results/AE_Ours_hospitalRGB_1673560411'
path = './results/Unet_TwoBranch_room'
data1 = np.load(path + '/out.npy')
# depth_raw = o3d.geometry.Image((data))
frame11 = data1[0, :]
print(data1.shape)
print(frame11.shape)


# Form the Generated point cloud
pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(frame11[:, :3])
pcd1.colors = o3d.utility.Vector3dVector(frame11[:, 3:])


# Form the Reference point cloud
data2 = np.load(path + '/ref.npy')
frame21 = data2[0, :]
pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(frame21[:, :3])
pcd2.colors = o3d.utility.Vector3dVector(frame21[:, 3:])


# Form the Input point cloud
PtsNum_ori = data2.shape[1]
input_num_points = int(data2.shape[1]/10)
pcd_sameNum_list = list(np.linspace(0, PtsNum_ori-1, input_num_points).round().astype(int))
ref_input = frame21[pcd_sameNum_list, :]
pcd3 = o3d.geometry.PointCloud()
pcd3.points = o3d.utility.Vector3dVector(ref_input[:, :3])
pcd3.colors = o3d.utility.Vector3dVector(ref_input[:, 3:])


# Visualization
vis_out = o3d.visualization.VisualizerWithEditing()
vis_out.create_window(window_name='Output Point Cloud', width=1200, height=1000, left=0, top=0)
vis_out.add_geometry(pcd1)

vis_ref = o3d.visualization.VisualizerWithEditing()
vis_ref.create_window(window_name='Reference Point Cloud', width=1200, height=1000, left=1500, top=0)
vis_ref.add_geometry(pcd2)

vis_inp = o3d.visualization.VisualizerWithEditing()
vis_inp.create_window(window_name='Reference Point Cloud', width=1200, height=1000, left=1500, top=1000)
vis_inp.add_geometry(pcd3)

while True:
    vis_out.update_geometry(pcd1)
    if not vis_out.poll_events():
        break
    vis_out.update_renderer()

    vis_ref.update_geometry(pcd2)
    if not vis_ref.poll_events():
        break
    vis_ref.update_renderer()

    vis_inp.update_geometry(pcd3)
    if not vis_inp.poll_events():
        break
    vis_inp.update_renderer()

vis_out.destroy_window()
vis_ref.destroy_window()
vis_inp.destroy_window()





# data3 = np.load('/home/jared/SAIR_Lab/Super-Map/diffusion-point-cloud-main/results/GEN_Ours_hospital_1673237938/out.npy')
# frame31 = data3[:, 0]
# print(frame31.shape)


# # import matplotlib
# # matplotlib.use('WebAgg')
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import proj3d

# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(x, y, z)
# plt.show()

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
pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(frame11[:, :3])
pcd1.colors = o3d.utility.Vector3dVector(frame11[:, 3:])


data2 = np.load(path + '/ref.npy')
frame21 = data2[0, :]
# print(frame21.shape)
pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(frame21[:, :3])
pcd2.colors = o3d.utility.Vector3dVector(frame21[:, 3:])


vis_out = o3d.visualization.VisualizerWithEditing()
vis_out.create_window(window_name='Output Point Cloud', width=1200, height=1000, left=0, top=0)
vis_out.add_geometry(pcd1)

vis_ref = o3d.visualization.VisualizerWithEditing()
vis_ref.create_window(window_name='Reference Point Cloud', width=1200, height=1000, left=1500, top=0)
vis_ref.add_geometry(pcd2)
while True:
    vis_out.update_geometry(pcd1)
    if not vis_out.poll_events():
        break
    vis_out.update_renderer()

    vis_ref.update_geometry(pcd2)
    if not vis_ref.poll_events():
        break
    vis_ref.update_renderer()

vis_out.destroy_window()
vis_ref.destroy_window()





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

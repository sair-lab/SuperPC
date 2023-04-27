import RGBD2PointCloud as RGBD2PtsCloud
import numpy as np
import torch
import open3d as o3d


# ------------------------------------------------- FPS - Functions used to divide point clouds -------------------------------------------------
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



def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


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


# # Read the point cloud from the dataset
# if __name__ == "__main__":
#     # Read the point cloud from the dataset
#     frame = 266 # hospital - P000: 286 - long way
#     rgbd2PtsCloud = RGBD2PtsCloud.RGBD2PtsCloud(frame)
#     pcd_dense, pcd_sparse = rgbd2PtsCloud.convert6()
#     pts_xyz = np.array(pcd_sparse.points)
#     pts_colors = np.array(pcd_sparse.colors)
#     pts_xyz = torch.from_numpy(pts_xyz)
#     pts_colors = torch.from_numpy(pts_colors)
#     pts_points = torch.concat([pts_xyz, pts_colors], 1)
#     pts_xyz = pts_xyz.unsqueeze(0)
#     pts_colors = pts_colors.unsqueeze(0)
#     pts_points = pts_points.unsqueeze(0)
    

#     divide_num = 256
#     new_xyz, new_points = sample_and_group(int(pts_points.size()[1]/divide_num), 64, divide_num, pts_xyz.float(), pts_colors.float())
#     # new_xyz: sampled points position data, [B, npoint, C]
#     # new_points: sampled points data, [B, npoint, nsample, C+D]
#     # new_points = new_points.permute(0, 2, 1, 3) # [B, nsample, npoint, C+D]
#     new_points = new_points.flatten(0, 1) # [B*nsample, npoint, C+D]
#     points = new_points[:, :, :3]
#     colors = new_points[:, :, 3:]
    
#     part1 = new_points[0]
#     pcd1 = o3d.geometry.PointCloud()
#     pcd1.points = o3d.utility.Vector3dVector(part1[:, :3])
#     pcd1.colors = o3d.utility.Vector3dVector(part1[:, 3:])

#     part2 = new_points[1]
#     pcd2 = o3d.geometry.PointCloud()
#     pcd2.points = o3d.utility.Vector3dVector(part2[:, :3])
#     pcd2.colors = o3d.utility.Vector3dVector(part2[:, 3:])


#     # Visualization
#     vis_out = o3d.visualization.VisualizerWithEditing()
#     vis_out.create_window(window_name='Part1', width=1200, height=1000, left=0, top=0)
#     vis_out.add_geometry(pcd1)

#     vis_ref = o3d.visualization.VisualizerWithEditing()
#     vis_ref.create_window(window_name='Part2', width=1200, height=1000, left=1500, top=0)
#     vis_ref.add_geometry(pcd2)

#     vis_inp = o3d.visualization.VisualizerWithEditing()
#     vis_inp.create_window(window_name='Original Point Cloud', width=1200, height=1000, left=1500, top=1000)
#     vis_inp.add_geometry(pcd_sparse)

#     while True:
#         vis_out.update_geometry(pcd1)
#         if not vis_out.poll_events():
#             break
#         vis_out.update_renderer()

#         vis_ref.update_geometry(pcd2)
#         if not vis_ref.poll_events():
#             break
#         vis_ref.update_renderer()

#         vis_inp.update_geometry(pcd_sparse)
#         if not vis_inp.poll_events():
#             break
#         vis_inp.update_renderer()

#     vis_out.destroy_window()
#     vis_ref.destroy_window()
#     vis_inp.destroy_window()



# Read the point cloud from the result
if __name__ == "__main__":

    # Read the point cloud from the result
    path = './results/Unet_TwoBranch_room'
    # path = '/home/jared/SAIR_Lab/Super-Map/Super-Map-Fusion-Head-Point-Based-Model/results/1800000iters_one_branch'

    data1 = np.load(path + '/out.npy')
    # depth_raw = o3d.geometry.Image((data))
    pcd_sparse = data1[0, :]
    pts_xyz = pcd_sparse[:, :3]
    pts_colors = pcd_sparse[:, 3:]
    pts_xyz = torch.from_numpy(pts_xyz)
    pts_colors = torch.from_numpy(pts_colors)

    pts_points = torch.concat([pts_xyz, pts_colors], 1)
    pts_xyz = pts_points
    pcd3 = o3d.geometry.PointCloud()
    pcd3.points = o3d.utility.Vector3dVector(pts_xyz[:, :3])
    pcd3.colors = o3d.utility.Vector3dVector(pts_xyz[:, 3:])
    # pts_xyz = pts_xyz.unsqueeze(0)
    # pts_colors = pts_colors.unsqueeze(0)
    # pts_points = pts_points.unsqueeze(0)
    

    part1 = pts_xyz[pts_xyz[:, 0] < pts_xyz[:, 0].sort()[0][int(pts_xyz.size()[0]/2)]]
    part2 = pts_xyz[pts_xyz[:, 0] > pts_xyz[:, 0].sort()[0][int(pts_xyz.size()[0]/2)]]
    midx = pts_xyz[pts_xyz[:, 0] == pts_xyz[:, 0].sort()[0][int(pts_xyz.size()[0]/2)]]

    num1 = int(pts_xyz.size()[0]/2 - part1.size()[0])
    num2 = int(pts_xyz.size()[0]/2 - part2.size()[0])
    part1 = torch.cat((part1, midx[:num1]), 0)
    part2 = torch.cat((part2, midx[-num2:]), 0)
    

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(part1[:, :3])
    pcd1.colors = o3d.utility.Vector3dVector(part1[:, 3:])


    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(part2[:, :3])
    pcd2.colors = o3d.utility.Vector3dVector(part2[:, 3:])


    # Visualization
    vis_out = o3d.visualization.VisualizerWithEditing()
    vis_out.create_window(window_name='Part1', width=1200, height=1000, left=0, top=0)
    vis_out.add_geometry(pcd1)

    vis_ref = o3d.visualization.VisualizerWithEditing()
    vis_ref.create_window(window_name='Part2', width=1200, height=1000, left=1500, top=0)
    vis_ref.add_geometry(pcd2)

    vis_inp = o3d.visualization.VisualizerWithEditing()
    vis_inp.create_window(window_name='Original Point Cloud', width=1200, height=1000, left=1500, top=1000)
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





# # ------------------------------------------------- Naive divide - Functions used to divide point clouds -------------------------------------------------
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
# from sklearn.cluster import KMeans
# from sklearn.cluster import DBSCAN
# from sklearn.neighbors import NearestNeighbors


# if __name__ == "__main__":
    
#     frame = 198 # hospital - P000: 286 - long way; hospital - P000: 198 - room with sofa
#     rgbd2PtsCloud = RGBD2PtsCloud.RGBD2PtsCloud(frame)
#     pcd_dense, pcd_sparse = rgbd2PtsCloud.convert6(2)
#     pts_xyz = np.array(pcd_sparse.points)
#     pts_colors = np.array(pcd_sparse.colors)
#     pts_points = np.column_stack((pts_xyz, pts_colors))
#     pts_xyz = torch.from_numpy(pts_xyz)
#     pts_colors = torch.from_numpy(pts_colors)
#     pts_points = torch.concat([pts_xyz, pts_colors], 1)



#     # ---------------------------------------- Model Implementation ----------------------------------------
#     pts_xyzs = torch.cat((pts_points.unsqueeze(0), pts_points.unsqueeze(0)), 0)
#     # Plot the clustered point cloud in 3D space
#     num_patches = 2
#     pts_batched = torch.empty(pts_xyzs.size()[0]*num_patches, int(pts_xyzs.size()[1]/num_patches), pts_xyzs.size()[2])
#     for i in range(pts_xyzs.size()[0]):
#         # Medium value and rough patches
#         pts_xyz = pts_xyzs[i].squeeze(0)
#         part1 = pts_xyz[pts_xyz[:, 0] < pts_xyz[:, 0].sort()[0][int(pts_xyz.size()[0]/2)]]
#         part2 = pts_xyz[pts_xyz[:, 0] > pts_xyz[:, 0].sort()[0][int(pts_xyz.size()[0]/2)]]
#         midx = pts_xyz[pts_xyz[:, 0] == pts_xyz[:, 0].sort()[0][int(pts_xyz.size()[0]/2)]]

#         # Refined patches
#         num1 = int(pts_xyz.size()[0]/2 - part1.size()[0])
#         num2 = int(pts_xyz.size()[0]/2 - part2.size()[0])
#         part1 = torch.cat((part1, midx[:num1]), 0)
#         part2 = torch.cat((part2, midx[-num2:]), 0)

#         # Batched point clouds
#         pts_batched[i] = part1.unsqueeze(0)
#         pts_batched[i+1] = part1.unsqueeze(0)
    
#     # ---------------------------------------- Model Implementation ----------------------------------------



#     # # Plot the clustered point cloud in 2D space
#     # kmeans = KMeans(n_clusters=2).fit(pts_xyz[:,:2])
#     # plt.scatter(pts_xyz[:, 0], pts_xyz[:, 1], c=kmeans.labels_, s=0.1)
#     # plt.show() 


#     # Plot the clustered point cloud in 3D space
#     lables = torch.zeros(pts_xyz.size()[0])
#     x = pts_xyz[0].sort()[1]
#     part1 = pts_xyz[pts_xyz[:, 0] < pts_xyz[:, 0].sort()[0][int(pts_xyz.size()[0]/2)]]
#     part2 = pts_xyz[pts_xyz[:, 0] > pts_xyz[:, 0].sort()[0][int(pts_xyz.size()[0]/2)]]
#     midx = pts_xyz[pts_xyz[:, 0] == pts_xyz[:, 0].sort()[0][int(pts_xyz.size()[0]/2)]]

#     num1 = int(pts_xyz.size()[0]/2 - part1.size()[0])
#     num2 = int(pts_xyz.size()[0]/2 - part2.size()[0])
#     part1 = torch.cat((part1, midx[:num1]), 0)
#     part2 = torch.cat((part2, midx[-num2:]), 0)


#     # for i in range(n_clusters):
#     #     index = (kmeans.labels_ == i)
    
#     lables[torch.logical_and(pts_xyz[:, 0] < pts_xyz[:, 0].sort()[0][int(pts_xyz.size()[0]/2)-1], pts_xyz[:, 1] < pts_xyz[:, 1].sort()[0][int(pts_xyz.size()[0]/2)-1])] = 1
#     lables[torch.logical_and(pts_xyz[:, 0] < pts_xyz[:, 0].sort()[0][int(pts_xyz.size()[0]/2)-1], pts_xyz[:, 1] > pts_xyz[:, 1].sort()[0][int(pts_xyz.size()[0]/2)-1])] = 2
#     lables[torch.logical_and(pts_xyz[:, 0] > pts_xyz[:, 0].sort()[0][int(pts_xyz.size()[0]/2)-1], pts_xyz[:, 1] < pts_xyz[:, 1].sort()[0][int(pts_xyz.size()[0]/2)-1])] = 3


#     ax = plt.axes(projection='3d')
#     ax.scatter(pts_xyz[:, 0], pts_xyz[:, 1], pts_xyz[:, 2], c=lables, s=0.1)
#     plt.show() 


#     # # Plot the WCSS (Within-Cluster Sum of Square; sum of squared distance between each point and the centroid in a cluster) value vs number of clusters
#     # wcss = [] 
#     # for i in range(1, 20):
#     #     kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
#     #     kmeans.fit(pts_xyz)
#     #     wcss.append(kmeans.inertia_)

#     # plt.plot(range(1, 20), wcss)
#     # plt.xlabel('Number of Clusters')    
#     # plt.ylabel('WCSS') 
#     # plt.title('K-Means Evaluation - WCSS vs Number of Clusters') 
#     # plt.show()


#      # knn = NearestNeighbors(n_neighbors=5).fit(pts_xyz)
#     # ax = plt.axes(projection='3d')
#     # ax.scatter(pts_xyz[:, 0], pts_xyz[:, 1], pts_xyz[:, 2], c=knn.classes_, s=0.1)
#     # plt.show() 









# # ------------------------------------------------- K-Means - Functions used to divide point clouds -------------------------------------------------
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
# from sklearn.cluster import KMeans
# from sklearn.cluster import DBSCAN


# if __name__ == "__main__":
    
#     frame = 198 # hospital - P000: 286 - long way; hospital - P000: 198 - room with sofa
#     rgbd2PtsCloud = RGBD2PtsCloud.RGBD2PtsCloud(frame)
#     pcd_dense, pcd_sparse = rgbd2PtsCloud.convert6(2)
#     pts_xyz = np.array(pcd_sparse.points)
#     pts_colors = np.array(pcd_sparse.colors)
#     pts_points = np.column_stack((pts_xyz, pts_colors))
#     pts_xyz = torch.from_numpy(pts_xyz)
#     pts_colors = torch.from_numpy(pts_colors)
#     pts_points = torch.concat([pts_xyz, pts_colors], 1)


#     # # Plot the clustered point cloud in 2D space
#     # kmeans = KMeans(n_clusters=2).fit(pts_xyz[:,:2])
#     # plt.scatter(pts_xyz[:, 0], pts_xyz[:, 1], c=kmeans.labels_, s=0.1)
#     # plt.show() 


#     # Plot the clustered point cloud in 3D space
#     n_clusters = 6
#     kmeans = KMeans(n_clusters=n_clusters).fit(pts_xyz)
#     # for i in range(n_clusters):
#     #     index = (kmeans.labels_ == i)



#     ax = plt.axes(projection='3d')
#     ax.scatter(pts_xyz[:, 0], pts_xyz[:, 1], pts_xyz[:, 2], c=kmeans.labels_, s=0.1)
#     plt.show() 


#     # # Plot the WCSS (Within-Cluster Sum of Square; sum of squared distance between each point and the centroid in a cluster) value vs number of clusters
#     # wcss = [] 
#     # for i in range(1, 20):
#     #     kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
#     #     kmeans.fit(pts_xyz)
#     #     wcss.append(kmeans.inertia_)

#     # plt.plot(range(1, 20), wcss)
#     # plt.xlabel('Number of Clusters')    
#     # plt.ylabel('WCSS') 
#     # plt.title('K-Means Evaluation - WCSS vs Number of Clusters') 
#     # plt.show()


#      # knn = NearestNeighbors(n_neighbors=5).fit(pts_xyz)
#     # ax = plt.axes(projection='3d')
#     # ax.scatter(pts_xyz[:, 0], pts_xyz[:, 1], pts_xyz[:, 2], c=knn.classes_, s=0.1)
#     # plt.show() 







# # ------------------------------------------------- KNN - Functions used to divide point clouds -------------------------------------------------
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
# from sklearn.cluster import KMeans
# from sklearn.cluster import DBSCAN
# from sklearn.neighbors import NearestNeighbors


# if __name__ == "__main__":
    
#     frame = 198 # hospital - P000: 286 - long way; hospital - P000: 198 - room with sofa
#     rgbd2PtsCloud = RGBD2PtsCloud.RGBD2PtsCloud(frame)
#     pcd_dense, pcd_sparse = rgbd2PtsCloud.convert6(2)
#     pts_xyz = np.array(pcd_sparse.points)
#     pts_colors = np.array(pcd_sparse.colors)
#     pts_points = np.column_stack((pts_xyz, pts_colors))
#     pts_xyz = torch.from_numpy(pts_xyz)
#     pts_colors = torch.from_numpy(pts_colors)
#     pts_points = torch.concat([pts_xyz, pts_colors], 1)




#     # Plot the clustered point cloud in 3D space
#     # knn = NearestNeighbors(n_neighbors=5).fit(pts_xyz)
#     # ax = plt.axes(projection='3d')
#     # ax.scatter(pts_xyz[:, 0], pts_xyz[:, 1], pts_xyz[:, 2], c=knn.classes_, s=0.1)
#     # plt.show() 


#     # # Plot the WCSS (Within-Cluster Sum of Square; sum of squared distance between each point and the centroid in a cluster) value vs number of clusters
#     # wcss = [] 
#     # for i in range(1, 20):
#     #     kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
#     #     kmeans.fit(pts_xyz)
#     #     wcss.append(kmeans.inertia_)

#     # plt.plot(range(1, 20), wcss)
#     # plt.xlabel('Number of Clusters')    
#     # plt.ylabel('WCSS') 
#     # plt.title('K-Means Evaluation - WCSS vs Number of Clusters') 
#     # plt.show()


    
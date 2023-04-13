import RGBD2PointCloud as RGBD2PtsCloud
import numpy as np
import torch


# Briefly view the processed point cloud data
# frame = 328 # seasonsforest_winter: 328 - tree
frame = 198 # hospital - P000: 198 - room with sofa
# frame = 266 # hospital - P000: 286 - long way
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



# used to test the chamfer distance memory cost
def distChamfer(a, b):
    # numbers of the partitions of the point cloud
    n = 4  # /4 for 50k Pts; 
    _, num_points_all, _ = a.size()
    x_list = torch.split(a, int(np.ceil(num_points_all/n)), dim=1) # /4 for 50000 pts
    y_list = torch.split(b, int(np.ceil(num_points_all/n)), dim=1)


    # generate the initial blank tensor to store the distances between two portions of the two point clouds
    d_ref_all_com = torch.tensor([]).to('cuda').float()
    d_gen_com_all = 10000 * torch.ones(2, num_points_all).to('cuda').float()
    for i in range(n):
        # generate the initial blank tensor to store the distances between two portions of the two point clouds
        d_gen_all = torch.tensor([]).to('cuda').float()
        d_ref_com = 10000 * torch.ones(2, int(num_points_all/n)).to('cuda').float()

        # Borrow from https://github.com/ThibaultGROUEIX/AtlasNet
        for j in range(n):
            x = x_list[i]
            y = y_list[j]
            _, num_points, _ = x.size()
            xx = torch.bmm(x, x.transpose(2, 1))
            yy = torch.bmm(y, y.transpose(2, 1))
            zz = torch.bmm(x, y.transpose(2, 1)) # Source of exceeding the max GPU memory 
            diag_ind = torch.arange(0, num_points).to(a).long()
            rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
            ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
            P_ij = (rx.transpose(2, 1) + ry - 2 * zz)
            d_gen_ij, d_ref_ij = P_ij.min(1)[0], P_ij.min(2)[0]
            d_gen_all = torch.cat((d_gen_all, d_gen_ij), 1)
            d_ref_com = torch.minimum(d_ref_com, d_ref_ij)
        
        # compare and find the smaller minimum distacese for each set in the generated point cloud
        d_gen_com_all = torch.minimum(d_gen_com_all, d_gen_all)
        # concat the minimum distances for each points in the reference point cloud
        d_ref_all_com = torch.cat((d_ref_all_com, d_ref_com), 1)
        


    return d_gen_com_all, d_ref_all_com
import RGBD2PointCloud as RGBD2PtsCloud


# Briefly view the processed point cloud data
frame = 478
rgbd2PtsCloud = RGBD2PtsCloud.RGBD2PtsCloud(frame)
pcd_dense, pcd_sparse = rgbd2PtsCloud.convert6()
rgbd2PtsCloud.view_image()
rgbd2PtsCloud.view_PtsCloud()
print(pcd_sparse)
# Save pcd file
# o3d.io.write_point_cloud('./PtsDataFunc/P000/train/frame' + str(frame) + '.pcd', pcd_sparse, write_ascii=True, compressed=False, print_progress=False)

pts_sparse_xyz = list(pcd_sparse.points)
print(pts_sparse_xyz[0])

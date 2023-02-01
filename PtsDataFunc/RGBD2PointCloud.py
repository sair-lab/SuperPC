import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as img
import open3d as o3d


class RGBD2PtsCloud(object):
    r'''
    Convert RGBD image to point cloud

    Args:
        frame: the frame of the image
    Returns:
        pcd_dense: saved pcd_dense files (dense and sparse).
    '''
    def __init__(self, frame):
        self.w = 640
        self.h = 480
        self.fx = 320
        self.fy = 320
        self.cx = 320
        self.cy = 240
        self.pix2metre = 80.0
        self.frame = frame

    def convert6(self):
        # Read a depth image (D) and RGB image (RGB)
        img_depth_L = np.load('./PtsDataFunc/P000/depth_left/000' + str(self.frame) + '_left_depth.npy')
        depth_raw = o3d.geometry.Image((img_depth_L))
        color_raw = o3d.io.read_image('./PtsDataFunc/P000/image_left/000' + str(self.frame) + '_left.png')
        # region Convert RGBD to Point Cloud (Left camera)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw, depth_scale=1, depth_trunc=10, convert_rgb_to_intensity=False)

        # Set camera intrinsic
        intrinsic = o3d.camera.PinholeCameraIntrinsic(self.w, self.h, self.fx, self.fy, self.cx, self.cy)
        # Convert RGBD to Point Cloud
        pcd_dense = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        # Flip it, otherwise the pointcloud will be upside down
        pcd_dense.transform([[1,  0,  0, 0], 
                             [0, -1,  0, 0], 
                             [0,  0, -1, 0], 
                             [0,  0,  0, 1]])

        # Select points to make sure the input point clouds have the same number
        PtsNum_ori = len(list(pcd_dense.points)) # the original points number
        PtsNum_tar = 208000 + 1 # the target points number (288000 as an example)
        pcd_sameNum_list = np.linspace(0, PtsNum_ori, PtsNum_tar).round().astype(int)
        pcd_dense = pcd_dense.select_by_index(pcd_sameNum_list)
        # Point Cloud downsample
        pcd_sparse = pcd_dense.uniform_down_sample(2)
        # Extra sparse
        # pcd_sparse = pcd_dense.uniform_down_sample(200)
        pcd_dense = pcd_sparse # make pcd_ dense = pcd_sparse
        # pcd_sparse = pcd_sparse.uniform_down_sample(8)
        # endregion
        self.PtsDim = 6
        return pcd_dense, pcd_sparse
    
    def convert4(self):
        # Read a depth image (D) and RGB image (RGB)
        img_depth_L = np.load('./PtsDataFunc/P000/depth_left/000' + str(self.frame) + '_left_depth.npy')
        depth_raw = o3d.geometry.Image((img_depth_L))
        color_raw = o3d.io.read_image('./PtsDataFunc/P000/image_left/000' + str(self.frame) + '_left.png')
        # region Convert RGBD to Point Cloud (Left camera)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw, depth_scale=1.0, depth_trunc=20, convert_rgb_to_intensity=True)

        # Set camera intrinsic
        intrinsic = o3d.camera.PinholeCameraIntrinsic(self.w, self.h, self.fx, self.fy, self.cx, self.cy)
        # Convert RGBD to Point Cloud
        pcd_dense = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        # Flip it, otherwise the pointcloud will be upside down
        pcd_dense.transform([[1,  0,  0, 0], 
                             [0, -1,  0, 0], 
                             [0,  0, -1, 0], 
                             [0,  0,  0, 1]])

        # Point Cloud downsample
        pcd_sparse = pcd_dense.uniform_down_sample(50)
        # endregion
        self.PtsDim = 4
        return pcd_dense, pcd_sparse

    def view_image(self):
        # View images
        # Read a depth image (D) and RGB image (RGB)
        img_depth_L = np.load('./PtsDataFunc/P000/depth_left/000' + str(self.frame) + '_left_depth.npy')
        depth_raw = o3d.geometry.Image((img_depth_L))
        color_raw = o3d.io.read_image('./PtsDataFunc/P000/image_left/000' + str(self.frame) + '_left.png')
        # Convert RGBD to Point Cloud (Left camera)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw, depth_scale=1000.0, depth_trunc=1000, convert_rgb_to_intensity=False)
        print(rgbd_image)
        plt.subplot(1, 2, 1)
        plt.title('Image')
        plt.imshow(rgbd_image.color)
        plt.subplot(1, 2, 2)
        plt.title('Depth Image')
        plt.imshow(rgbd_image.depth)
        plt.show()
        return None

    def view_PtsCloud(self):
        if self.PtsDim == 6:
            pcd_dense, pcd_sparse = self.convert6()
        if self.PtsDim == 4:
            pcd_dense, pcd_sparse = self.convert4()

        vis_dense = o3d.visualization.VisualizerWithEditing()
        vis_dense.create_window(window_name='Dense Point Cloud', width=1500, height=1350, left=0, top=0)
        vis_dense.add_geometry(pcd_dense)

        vis_sparse = o3d.visualization.VisualizerWithEditing()
        vis_sparse.create_window(window_name='Sparse Point Cloud', width=1500, height=1350, left=1500, top=0)
        vis_sparse.add_geometry(pcd_sparse)

        while True:
            vis_dense.update_geometry(pcd_dense)
            if not vis_dense.poll_events():
                break
            vis_dense.update_renderer()

            vis_sparse.update_geometry(pcd_dense)
            if not vis_sparse.poll_events():
                break
            vis_sparse.update_renderer()

        vis_dense.destroy_window()
        vis_sparse.destroy_window()

        print(pcd_dense)
        print(pcd_sparse)
        return None


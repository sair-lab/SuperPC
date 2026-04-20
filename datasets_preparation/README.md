# Dataset Preparation Guide

This document describes how to prepare datasets for SuperPC.

## ShapeNet Dataset Preparation

### 1. Download ground-truth point clouds
Download the ShapeNet55/34 point clouds from the PoinTr dataset guide:
https://github.com/yuxumin/PoinTr/blob/master/DATASET.md

Only the ground-truth point clouds are required (each point cloud has 8192 points).

### 2. Download rendered RGB images
Download rendered images from:
https://drive.google.com/file/d/1_QivYLFFhVDvb_S3-ga5TBSouQ1x9K7m/view?usp=drive_link

These images were rendered with Blender based on the code from:
https://github.com/Xharlie/ShapenetRender_more_variation

### 3. Arrange folders
Expected structure:

```text
SuperPC/datasets/ShapeNet/
├── render_rgb_v2_13cat/
│   └── image/
│       ├── 02691156/
│       ├── 02818832/
│       └── ...
└── shapenet_pc/
		├── 02691156-1a04e3eab45ca15dd86060f189eb133.npy
		├── 02691156-1a6ad7a24bb89733f412783097373bdc.npy
		└── ...
```

## TartanAir Dataset Preparation

### 1. Download the original TartanAir dataset
Follow the official TartanAir download instructions:
https://github.com/castacks/tartanair_tools

### 2. Place or link it under the SuperPC datasets folder
Expected structure:

```text
SuperPC/datasets/TartanAir/
├── abandonedfactory/
│   └── Easy/
│       ├── P000/
│       ├── P001/
│       └── ...
└── ...
```

## KITTI-360 Dataset Preparation

### 1. Download the original KITTI-360 dataset
From the KITTI-360 download page, you only need:

- Accumulated Point Clouds for Train and Val (12G)
- Perspective Images for Train and Val (128G)
- Vehicle Poses (8.9M)
- Calibrations (3K)

Download page:
https://www.cvlibs.net/datasets/kitti-360/download.php

### 2. Place or link it under the SuperPC datasets folder
Expected structure:

```text
SuperPC/datasets/KITTI360/
├── calibration/
│   ├── calib_cam_to_pose.txt
│   ├── calib_cam_to_velo.txt
│   ├── calib_sick_to_velo.txt
│   └── ...
├── data_2d_raw/
│   └── 2013_05_28_drive_0000_sync/
│       └── image_00/
│           ├── data_rect/
│           └── timestamps.txt
├── data_3d_semantics/
│   └── 2013_05_28_drive_0000_sync/
│       ├── dynamic/
│       │   ├── 0000000002_0000000385.ply
│       │   └── ...
│       └── static/
└── data_poses/
		└── 2013_05_28_drive_0000_sync/
				├── cam0_to_world.txt
				└── poses.txt
```

### 3. Build the KITTI-360 submap dataset
Script:
[build_kitti360_submap_dataset.py](build_kitti360_submap_dataset.py)

Default behavior:

- Output root: `/data_sair/kitti360_maps/submaps`
- Candidate frames start from frame `80` in each drive (`--start_frame 80`)
- Candidate frames stop `80` frames before each drive end (`--end_margin_frames 80`)
- Minimum spacing between selected frames: `80` (`--min_frame_gap 80`)
- Exact saved points per submap: `46080` (`--downsample_points 46080`)

Example:

```bash
conda run -n rgbd_map python build_kitti360_submap_dataset.py \
	--kitti_root /data_sair/kitti360 \
	--kitti_maps_root /path/to/kitti360_maps/data_3d_semantics/train \
	--output_root /path/to/kitti360_maps/submaps \
	--camera image_00 \
	--start_frame 80 \
	--end_margin_frames 80 \
	--min_frame_gap 10 \
	--downsample_points 46080 \
	--fps_device cuda
```

To skip specific drives, use `--exclude_drives`:

```bash
conda run -n rgbd_map python build_kitti360_submap_dataset.py \
	--exclude_drives 2013_05_28_drive_0002_sync
```

Output layout:

```text
/path/to/kitti360_maps/submaps/<drive>/<camera>/<frame>/
```

Each frame folder includes:

- `<frame>_rgb.png`
- `<frame>_submap_xyzrgb.ply`
- `metadata.json`

The script also writes:

- `build_kitti360_submap_dataset.log`
- `dataset_summary.json`
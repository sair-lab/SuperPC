# SuperPC: A Single Diffusion Model for Unified Point Cloud Processing

[[Paper](https://arxiv.org/abs/2503.14558)] [[Code](https://github.com/sair-lab/SuperPC)] [[Homepage](https://sairlab.org/superpc/)] 

The official code repository for the CVPR 2025 paper "SuperPC: A Single Diffusion Model for Unified Point Cloud Processing".

### News
1. Added dataset preparation code and instructions
2. Uploaded the pretrained weights
3. Added evaluation and single-sample testing code

### Future Plans
- [ ] Add detailed training pipeline
- [ ] Possibly add prepared datasets if permission is granted by the original dataset providers
- [ ] Add the flow-matching based training pipeline and pretrained weights

## Installation

Run the setup script:
```
./setup_env.sh
```


## Datasets Preparation
Follow the [detailed dataset preparation guide](datasets_preparation/README.md) to download the three raw datasets and prepare ground-truth point clouds and images.
1. [ShapeNet dataset](https://shapenet.org/)
2. [TartanAir dataset](https://github.com/castacks/tartanair_tools) - [separate sub-dataset list](https://github.com/castacks/tartanair_tools/blob/master/download_training_zipfiles.txt)
3. [KITTI-360 dataset](https://www.cvlibs.net/datasets/kitti-360/user_login.php)


## Model Zoo
Download all pretrained weights from:
https://drive.google.com/drive/folders/1FrQtm8LBVrbdRT4Xs87rIZpJ9nYaTqcG?usp=drive_link




## Evaluation and Testing (Combined Tasks)

### 1. ShapeNet Dataset
Use `test_superpc.py` for single-sample or split-based testing with optional output saving.

[Single sample] (assuming you have already prepared the ShapeNet dataset):

airplane: /02691156/10aa040f470500c6a66ef8df4909ded9


```bash
python test_superpc.py \
  --dataset shapenet \
  --cat-id 02691156 \
  --model-id 10155655850468db78d106ce0a280f87 \
  --mode easy \
  --ckpt_path /path/to/your_checkpoint.pth \
  --use_vision_conditioning true \
  --vision_image_dir /path/to/shapenet/render/images \
  --num_points 2048 \
  --target_num_points 8192 \
  --use_input_scout_fill true \
  --seed 21 \
  --sampling_steps 25 \
  --save_pc true
```

[Whole ShapeNet test split] (omit `--cat-id` and `--model-id`):

```bash
python test_superpc.py \
  --dataset shapenet \
  --ckpt_path /path/to/your_checkpoint.pth \
  --use_vision_conditioning true \
  --vision_image_dir /path/to/shapenet/render/images \
  --num_points 2048 \
  --target_num_points 8192 \
  --use_input_scout_fill true \
  --seed 21 \
  --sampling_steps 25 \
  --save_pc true
```


### 2. TartanAir Dataset
Use `test_superpc.py` for single-sample or split-based testing with optional output saving.

[Single sample] (assuming you have already prepared the TartanAir dataset):

Single TartanAir sample by metadata path:

```bash
python test_superpc.py \
  --dataset tartanair \
  --tartanair_root /data_sair/tartanair_maps \
  --metadata_path /path/to/tartanair_maps/hospital/Easy/P000/000000/metadata.json \
  --ckpt_path /path/to/your_checkpoint.pth \
  --use_vision_conditioning true \
  --num_points 11520 \
  --target_num_points 46080 \
  --seed 21 \
  --sampling_steps 25 \
  --save_pc true
```

TartanAir split testing (`--eval_split`: `train`, `val`, `test`, or `all`):

```bash
python test_superpc.py \
  --dataset tartanair \
  --tartanair_root /data_sair/tartanair_maps \
  --eval_split test \
  --ckpt_path /path/to/your_checkpoint.pth \
  --use_vision_conditioning true \
  --num_points 11520 \
  --target_num_points 46080 \
  --seed 21 \
  --sampling_steps 25 \
  --save_pc true
```

### 3. KITTI-360 Dataset
Use `test_superpc.py` for single-sample or split-based testing with optional output saving.

[Single sample] (assuming you have already prepared the KITTI-360 dataset):

Single KITTI-360 sample by metadata path:

```bash
python test_superpc.py \
  --dataset kitti360 \
  --kitti360_root /path/to/kitti360_maps/submaps \
  --metadata_path /path/to/kitti360_maps/submaps/2013_05_28_drive_0000_sync/image_00/0000000090/metadata.json \
  --ckpt_path /path/to/your_checkpoint.pth \
  --use_vision_conditioning true \
  --num_points 11520 \
  --target_num_points 46080 \
  --use_input_scout_fill true \
  --seed 21 \
  --sampling_steps 25 \
  --save_pc true
```

KITTI-360 split testing (`--eval_split=test` maps to the val split because KITTI-360 currently uses train/val partitioning):

```bash
python test_superpc.py \
  --dataset kitti360 \
  --kitti360_root /path/to/kitti360_maps/submaps \
  --eval_split test \
  --ckpt_path /path/to/your_checkpoint.pth \
  --use_vision_conditioning true \
  --num_points 11520 \
  --target_num_points 46080 \
  --seed 21 \
  --sampling_steps 25 \
  --save_pc true
```


## Acknowledgement
The codebase is built upon: [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine), [PUFM](https://github.com/Holmes-Alan/PUFM), [DiffusionPC](https://github.com/luost26/diffusion-point-cloud), and [DCD](https://github.com/wutong16/Density_aware_Chamfer_Distance). We appreciate the authors' excellent work.


## Reference

```
@inproceedings{du2025superpc,
  title={SuperPC: a single diffusion model for point cloud completion, upsampling, denoising, and colorization},
  author={Du, Yi and Zhao, Zhipeng and Su, Shaoshu and Golluri, Sharath and Zheng, Haoze and Yao, Runmao and Wang, Chen},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={16953--16964},
  year={2025}
}
```

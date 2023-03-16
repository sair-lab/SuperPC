
# #-------------------------------------------Two-branches Model:ResNet50 + PointNet----------------------------------------
# import os
# import time
# import argparse
# import torch
# import torch.utils.tensorboard
# from torch.nn.utils import clip_grad_norm_
# from tqdm.auto import tqdm
# import wandb

# from utils.dataset import *
# from utils.datasetImg import *
# from utils.misc import *
# from utils.data import *
# from models.autoencoder import *
# from evaluation import EMD_CD


# # Arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('--ckpt', type=str, default='/home/jared/SAIR_Lab/Super-Map/Super-Map-Fusion-Head-Point-Based-Model_twoBranchsModel/logs_ae/AE_2023_03_02_06h01m_TwoBranch-resume232k_latentDim2048_inputDownsample2/ckpt_0.047040_248000.pt')
# parser.add_argument('--categories', type=str_list, default=['hospitalRGB'])
# parser.add_argument('--save_dir', type=str, default='./metrics_test')
# parser.add_argument('--device', type=str, default='cuda')
# # wandb config
# parser.add_argument('--run_name', type=str, default='AE_2023_03_02_06h01m_TwoBranch-resume232k_latentDim2048_inputDownsample2')
# parser.add_argument('--project_name', type=str, default='Metrics-Test')
# # Datasets and loaders
# parser.add_argument('--input_downsample', type=int, default=2)
# parser.add_argument('--dataset_path', type=str, default='/home/jared/SAIR_Lab/Super-Map/Super-Map-Fusion-Head-Point-Based-Model/data/shapenet_oneTraj_50000pts.hdf5')
# parser.add_argument('--datasetImg_path', type=str, default='./PtsDataFunc/imagedata_small')
# parser.add_argument('--scale_mode', type=str, default='shape_unit')
# parser.add_argument('--seed', type=int, default=2020)
# # parser.add_argument('--batch_size', type=int, default=128) # orignial
# # parser.add_argument('--batch_size', type=int, default=8) # poits /40; 30 frames for training; 10 frames for testing
# parser.add_argument('--batch_size', type=int, default=1) # poits /20; 30 frames for training; 10 frames for testing
# parser.add_argument('--num_inspect_batches', type=int, default=35)
# parser.add_argument('--num_inspect_pointclouds', type=int, default=35)
# args = parser.parse_args()

# # Logging
# save_dir = os.path.join(args.save_dir, args.run_name)
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# logger = get_logger('test', save_dir)
# writer = torch.utils.tensorboard.SummaryWriter(save_dir)
# ckpt_mgr = CheckpointManager(save_dir)
# for k, v in vars(args).items():
#     logger.info('[ARGS::%s] %s' % (k, repr(v)))


# seed_all(args.seed)
# # Datasets and loaders
# logger.info('Loading datasets...')
# test_dset = ShapeNetCore(
#     path=args.dataset_path,
#     cates=args.categories,
#     split='test',
#     scale_mode=args.scale_mode
# )
# test_loader = DataLoader(test_dset, batch_size=args.batch_size, num_workers=0)


# # Datasets and loaders (Images)
# test_dset_img = ImageNetCore(
#     path = args.datasetImg_path,
#     split='test',
# )
# test_iter_img = get_data_iterator(DataLoader(
#     test_dset_img,
#     batch_size=args.batch_size,
#     num_workers=0,
# ))

# # ckpt_listRE = ['ckpt_0.084076_233000.pt', 'ckpt_0.073407_236000.pt', 'ckpt_0.056538_241000.pt', 'ckpt_0.047040_248000.pt', 'ckpt_0.068755_257000.pt', 'ckpt_0.052599_268000.pt', 'ckpt_0.069663_281000.pt', 'ckpt_0.058261_296000.pt', 'ckpt_0.089857_305022.pt']
# # ckpt_list = ['']

# ckpt = torch.load(args.ckpt)
# # Model
# logger.info('Loading model...')
# model = AutoEncoder(ckpt['args']).to(args.device)
# model.load_state_dict(ckpt['state_dict'])


# # start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project = args.project_name,
#     name = args.run_name,

#     # track hyperparameters and run metadata
#     config = {
#     "architecture": "TwoBranch-SkipConnection",
#     "dataset": "TartanAir",
#     }
# )


# sum_n = 0
# sum_chamfer = 0
# for i, batch in enumerate(tqdm(test_loader, desc='Inspect')):
#     # Load point cloud
#     x = batch['pointcloud'].to(args.device).float()
#     shift = batch['shift'].to(args.device)
#     scale = batch['scale'].to(args.device)
#     # Downsample the GT to the input point cloud
#     PtsNum_ori = x.size(dim=1)
#     input_num_points = int(x.size(dim=1)/args.input_downsample)
#     pcd_sameNum_list = list(np.linspace(0, PtsNum_ori-1, input_num_points).round().astype(int))
#     x_input = x[:, pcd_sameNum_list, :]

#     # Load image
#     batch_img = next(test_iter_img)
#     img = batch_img['image'].to(args.device).float()

#     model.eval()
#     code = model.encode(x_input, img)
#     recons = model.decode(code, x.size(1), flexibility=args.flexibility).detach()
#     # Remap the generated pointcloud xyz and RGB to original map
#     recons = recons * scale + shift
#     vertices = recons[:args.num_inspect_pointclouds, :, :3]
#     colors = torch.round(255*recons[:args.num_inspect_pointclouds, :, 3:]).type(torch.int)

#     sum_n += x.size(0)
#     if i >= args.num_inspect_batches:
#         break   # Inspect only 5 batch


#     logger.info('Start computing metrics...')
#     metrics = EMD_CD(recons.to(args.device), x.to(args.device), batch_size=args.batch_size)
#     cd, emd = metrics['MMD-CD'].item(), metrics['MMD-EMD'].item()
#     logger.info('CD:  %.12f' % cd)

#     writer.add_mesh('val/pointcloud', vertices, colors, global_step=i)
#     writer.add_scalar('val/cd', cd, i)
#     writer.flush()

#     # wandb save point cloud
#     points = torch.Tensor.numpy(torch.Tensor.cpu(torch.cat((-vertices, colors), dim=2)))
#     wandb.log({"point_scene": wandb.Object3D(points[0])})
#     wandb.log({"scene": i,"val/cd-loss": cd})





#-------------------------------------------Two-branches Model:ResNet50 + PointNet----------------------------------------
import os
import time
import argparse
import torch
from tqdm.auto import tqdm

from utils.dataset import *
from utils.datasetImg import *
from utils.misc import *
from utils.data import *
from models.autoencoder import *
from evaluation import EMD_CD


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='/home/jared/SAIR_Lab/Super-Map/Super-Map-Fusion-Head-Point-Based-Model_twoBranchsModel/logs_ae/AE_2023_03_02_06h01m_TwoBranch-resume232k_latentDim2048_inputDownsample2/ckpt_0.047040_248000.pt')
parser.add_argument('--categories', type=str_list, default=['hospitalRGB'])
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--device', type=str, default='cuda')
# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='/home/jared/SAIR_Lab/Super-Map/Super-Map-Fusion-Head-Point-Based-Model/data/shapenet_oneTraj_50000pts.hdf5')
parser.add_argument('--datasetImg_path', type=str, default='./PtsDataFunc/imagedata_small')
# parser.add_argument('--batch_size', type=int, default=128) # orignial
# parser.add_argument('--batch_size', type=int, default=8) # poits /40; 30 frames for training; 10 frames for testing
parser.add_argument('--batch_size', type=int, default=1) # poits /20; 30 frames for training; 10 frames for testing
args = parser.parse_args()

# Logging
save_dir = os.path.join(args.save_dir, 'AE_Ours_%s_%d' % ('_'.join(args.categories), int(time.time())) )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logger = get_logger('test', save_dir)
for k, v in vars(args).items():
    logger.info('[ARGS::%s] %s' % (k, repr(v)))

# Checkpoint
ckpt = torch.load(args.ckpt)
seed_all(ckpt['args'].seed)

# Datasets and loaders
logger.info('Loading datasets...')
test_dset = ShapeNetCore(
    path=args.dataset_path,
    cates=args.categories,
    split='test',
    scale_mode=ckpt['args'].scale_mode
)
test_loader = DataLoader(test_dset, batch_size=args.batch_size, num_workers=0)

# Datasets and loaders (Images)
test_dset_img = ImageNetCore(
    path = args.datasetImg_path,
    split='test',
)
test_iter_img = get_data_iterator(DataLoader(
    test_dset_img,
    batch_size=args.batch_size,
    num_workers=0,
))


# Model
logger.info('Loading model...')
model = AutoEncoder(ckpt['args']).to(args.device)
model.load_state_dict(ckpt['state_dict'])

all_ref = []
all_recons = []
for j in range(10):
    for i, batch in enumerate(tqdm(test_loader)):
        # Load image
        batch_img = next(test_iter_img)
        img = batch_img['image'].to(args.device).float()
        # Load point cloud
        ref = batch['pointcloud'].to(args.device).float()
        # Downsampling the GT to input point cloud
        PtsNum_ori = ref.size(dim=1)
        input_num_points = int(ref.size(dim=1)/2)
        pcd_sameNum_list = list(np.linspace(0, PtsNum_ori-1, input_num_points).round().astype(int))
        ref_input = ref[:, pcd_sameNum_list, :]
        # ref = ref[]
        shift = batch['shift'].to(args.device)
        scale = batch['scale'].to(args.device)
        model.eval()
        with torch.no_grad():
            code = model.encode(ref, img)
            recons = model.decode(code, ref.size(1), flexibility=ckpt['args'].flexibility).detach()

        ref = ref * scale + shift
        recons = recons * scale + shift

        if i >= 1:
                break   # Inspect only 5 batch

    all_ref.append(ref.detach().cpu())
    all_recons.append(recons.detach().cpu())

all_ref = torch.cat(all_ref, dim=0)
all_recons = torch.cat(all_recons, dim=0)

logger.info('Saving point clouds...')
np.save(os.path.join(save_dir, 'ref.npy'), all_ref.numpy())
np.save(os.path.join(save_dir, 'out.npy'), all_recons.numpy())

# logger.info('Start computing metrics...')
# metrics = EMD_CD(all_recons.to(args.device), all_ref.to(args.device), batch_size=args.batch_size)
# cd, emd = metrics['MMD-CD'].item(), metrics['MMD-EMD'].item()
# logger.info('CD:  %.12f' % cd)
# logger.info('EMD: %.12f' % emd)

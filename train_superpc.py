import os
import math
import argparse
import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
import wandb

from utils.dataset import *
from utils.datasetImg import *
from utils.misc import *
from utils.data import *
from models.vae_gaussian import *
from models.vae_flow import *
from models.flow import add_spectral_norm, spectral_norm_power_iteration
from evaluation import *

import datetime


# Arguments
parser = argparse.ArgumentParser()
# Model arguments
parser.add_argument('--model', type=str, default='flow', choices=['flow', 'gaussian'])
parser.add_argument('--latent_dim', type=int, default=256)
parser.add_argument('--num_steps', type=int, default=100)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.02)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--flexibility', type=float, default=0.0)
parser.add_argument('--truncate_std', type=float, default=2.0)
parser.add_argument('--latent_flow_depth', type=int, default=14)
parser.add_argument('--latent_flow_hidden_dim', type=int, default=256)
parser.add_argument('--num_samples', type=int, default=4)
parser.add_argument('--sample_num_points', type=int, default=2048)
parser.add_argument('--kl_weight', type=float, default=0.001)
parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--resume_iters', type=int, default=0)

# Datasets and loaders
parser.add_argument('--input_downsample', type=int, default=2)
# parser.add_argument('--dataset_path', type=str, default='/home/jared/SAIR_Lab/Super-Map/Super-Map-Fusion-Head-Point-Based-Model_twoBranchsModel/data/tartanair_allEnvs.hdf5')
parser.add_argument('--dataset_path', type=str, default='/home/jared/SAIR_Lab/Super-Map/Super-Map-Fusion-Head-Point-Based-Model/data/shapenet_oneTraj_50000pts.hdf5')
# parser.add_argument('--datasetImg_path', type=str, default='/home/jared/Large_datasets/TartanAir/data_image')
parser.add_argument('--datasetImg_path', type=str, default='/home/jared/SAIR_Lab/Super-Map/Super-Map-Fusion-Head-Point-Based-Model_twoBranchsModel/PtsDataFunc/imagedata_small')
parser.add_argument('--categories', type=str_list, default=['hospitalRGB'])
parser.add_argument('--scale_mode', type=str, default='shape_unit')
# parser.add_argument('--train_batch_size', type=int, default=128) # original
# parser.add_argument('--val_batch_size', type=int, default=32) # original
# parser.add_argument('--train_batch_size', type=int, default=32) # poits /40; 30 frames for training; 10 frames for testing
# parser.add_argument('--val_batch_size', type=int, default=8)# poits /40; 30 frames for training; 10 frames for testing
parser.add_argument('--train_batch_size', type=int, default=4) # poits /20; 30 frames for training; 10 frames for testing
parser.add_argument('--val_batch_size', type=int, default=1)# poits /20; 30 frames for training; 10 frames for testing

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-4)
parser.add_argument('--sched_start_epoch', type=int, default=200*THOUSAND)
parser.add_argument('--sched_end_epoch', type=int, default=400*THOUSAND)

# wandb config
parser.add_argument('--run_name', type=str, default='Two-Branch')
parser.add_argument('--project_name', type=str, default='GeneratorTest-Super-Map-Project-LargeDataset')


# Training
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs_gen')
parser.add_argument('--device', type=str, default='cuda')
# parser.add_argument('--max_iters', type=int, default=float('inf'))
parser.add_argument('--max_iters', type=int, default=18000000)
parser.add_argument('--val_freq', type=int, default=1000)
parser.add_argument('--test_freq', type=int, default=30*THOUSAND)
parser.add_argument('--test_size', type=int, default=6)
parser.add_argument('--tag', type=str, default=None)
parser.add_argument('--num_val_batches', type=int, default=-1)
parser.add_argument('--num_inspect_batches', type=int, default=1)
parser.add_argument('--num_inspect_pointclouds', type=int, default=1)
args = parser.parse_args()
seed_all(args.seed)

# Logging
if args.logging:
    log_dir = get_new_log_dir(args, args.log_root, prefix='GEN_', postfix='_' + args.tag if args.tag is not None else '')
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
    log_hyperparams(writer, args)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)

# Datasets and loaders
logger.info('Loading datasets...')

train_dset = ShapeNetCore(
    path=args.dataset_path,
    cates=args.categories,
    split='train',
    scale_mode=args.scale_mode,
)
val_dset = ShapeNetCore(
    path=args.dataset_path,
    cates=args.categories,
    split='val',
    scale_mode=args.scale_mode,
)
train_iter = get_data_iterator(DataLoader(
    train_dset,
    batch_size=args.train_batch_size,
    num_workers=0,
))
val_loader = DataLoader(val_dset, batch_size=args.val_batch_size, num_workers=0)

# Datasets and loaders (Images)
train_dset_img = ImageNetCore(
    path = args.datasetImg_path,
    split='train',
)
val_dset_img = ImageNetCore(
    path = args.datasetImg_path,
    split='val',
)
train_iter_img = get_data_iterator(DataLoader(
    train_dset_img,
    batch_size=args.train_batch_size,
    num_workers=0,
))
val_iter_img = get_data_iterator(DataLoader(
    val_dset_img,
    batch_size=args.val_batch_size,
    num_workers=0,
))
val_loader_img = DataLoader(val_dset_img, batch_size=args.val_batch_size, num_workers=0)


# Model
logger.info('Building model...')
if args.model == 'gaussian':
    model = GaussianVAE(args).to(args.device)
elif args.model == 'flow':
    model = FlowVAE(args).to(args.device)
logger.info(repr(model))
if args.spectral_norm:
    add_spectral_norm(model, logger=logger)

# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), 
    lr=args.lr, 
    weight_decay=args.weight_decay
)
scheduler = get_linear_scheduler(
    optimizer,
    start_epoch=args.sched_start_epoch,
    end_epoch=args.sched_end_epoch,
    start_lr=args.lr,
    end_lr=args.end_lr
)

# Train, validate and test
def train(it):
    # Load point cloud data
    batch = next(train_iter)
    x = batch['pointcloud'].to(args.device).float()

    # Load image
    batch_img = next(train_iter_img)
    img = batch_img['image'].to(args.device).float()

    # Reset grad and model state
    optimizer.zero_grad()
    model.train()
    if args.spectral_norm:
        spectral_norm_power_iteration(model, n_power_iterations=1)

    # Forward
    kl_weight = args.kl_weight
    loss = model.get_loss(x, img, kl_weight=kl_weight, writer=writer, it=it)

    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()

    logger.info('[Train] Iter %04d | Loss %.6f | Grad %.4f | KLWeight %.4f' % (
        it, loss.item(), orig_grad_norm, kl_weight
    ))
    writer.add_scalar('train/loss', loss, it)
    writer.add_scalar('train/kl_weight', kl_weight, it)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    writer.add_scalar('train/grad_norm', orig_grad_norm, it)
    writer.flush()
    # wandb save
    wandb.log({"iters": it,"train-loss": loss, "train-lr": optimizer.param_groups[0]['lr'], "train-grad_norm": orig_grad_norm, "train-kl_weight": kl_weight})


# def validate_inspect(it):
#     z = torch.randn([args.num_samples, args.latent_dim]).to(args.device)
#     x = model.sample(z, args.sample_num_points, flexibility=args.flexibility) #, truncate_std=args.truncate_std)
#     writer.add_mesh('val/pointcloud', x, global_step=it)
#     writer.flush()
#     logger.info('[Inspect] Generating samples...')

# def test(it):
#     ref_pcs = []
#     for i, data in enumerate(val_dset):
#         if i >= args.test_size:
#             break
#         ref_pcs.append(data['pointcloud'].unsqueeze(0))
#     ref_pcs = torch.cat(ref_pcs, dim=0)

#     gen_pcs = []
#     for i in tqdm(range(0, math.ceil(args.test_size / args.val_batch_size)), 'Generate'):
#         with torch.no_grad():
#             z = torch.randn([args.val_batch_size, args.latent_dim]).to(args.device)
#             x = model.sample(z, args.sample_num_points, flexibility=args.flexibility)
#             gen_pcs.append(x.detach().cpu())
#     gen_pcs = torch.cat(gen_pcs, dim=0)[:args.test_size]

#     # Denormalize point clouds, all shapes have zero mean.
#     # [WARNING]: Do NOT denormalize!
#     # ref_pcs *= val_dset.stats['std']
#     # gen_pcs *= val_dset.stats['std']

#     with torch.no_grad():
#         results = compute_all_metrics(gen_pcs.to(args.device), ref_pcs.to(args.device), args.val_batch_size)
#         results = {k:v.item() for k, v in results.items()}
#         jsd = jsd_between_point_cloud_sets(gen_pcs.cpu().numpy(), ref_pcs.cpu().numpy())
#         results['jsd'] = jsd

#     # CD related metrics
#     writer.add_scalar('test/Coverage_CD', results['lgan_cov-CD'], global_step=it)
#     writer.add_scalar('test/MMD_CD', results['lgan_mmd-CD'], global_step=it)
#     writer.add_scalar('test/1NN_CD', results['1-NN-CD-acc'], global_step=it)
#     # EMD related metrics
#     # writer.add_scalar('test/Coverage_EMD', results['lgan_cov-EMD'], global_step=it)
#     # writer.add_scalar('test/MMD_EMD', results['lgan_mmd-EMD'], global_step=it)
#     # writer.add_scalar('test/1NN_EMD', results['1-NN-EMD-acc'], global_step=it)
#     # JSD
#     writer.add_scalar('test/JSD', results['jsd'], global_step=it)

#     # logger.info('[Test] Coverage  | CD %.6f | EMD %.6f' % (results['lgan_cov-CD'], results['lgan_cov-EMD']))
#     # logger.info('[Test] MinMatDis | CD %.6f | EMD %.6f' % (results['lgan_mmd-CD'], results['lgan_mmd-EMD']))
#     # logger.info('[Test] 1NN-Accur | CD %.6f | EMD %.6f' % (results['1-NN-CD-acc'], results['1-NN-EMD-acc']))
#     logger.info('[Test] Coverage  | CD %.6f | EMD n/a' % (results['lgan_cov-CD'], ))
#     logger.info('[Test] MinMatDis | CD %.6f | EMD n/a' % (results['lgan_mmd-CD'], ))
#     logger.info('[Test] 1NN-Accur | CD %.6f | EMD n/a' % (results['1-NN-CD-acc'], ))
#     logger.info('[Test] JsnShnDis | %.6f ' % (results['jsd']))


def validate_loss(it):

    all_refs = []
    all_recons = []
    for i, batch in enumerate(tqdm(val_loader, desc='Validate')):
        # iterate the batch of the images
        batch_img = next(val_iter_img)
        if args.num_val_batches > 0 and i >= args.num_val_batches:
            break
        # Load image
        ref_img = batch_img['image'].to(args.device).float()
        # Load point cloud
        ref = batch['pointcloud'].to(args.device).float()
        # Downsampling the GT to input point cloud
        PtsNum_ori = ref.size(dim=1)
        input_num_points = int(ref.size(dim=1)/args.input_downsample)
        pcd_sameNum_list = list(np.linspace(0, PtsNum_ori-1, input_num_points).round().astype(int))
        ref_input = ref[:, pcd_sameNum_list, :]

        shift = batch['shift'].to(args.device)
        scale = batch['scale'].to(args.device)
        with torch.no_grad():
            model.eval()
            code = model.encode(ref_input, ref_img)
            recons = model.decode(code, ref.size(1), flexibility=args.flexibility)
        all_refs.append(ref * scale + shift)
        all_recons.append(recons * scale + shift)

    all_refs = torch.cat(all_refs, dim=0)
    all_recons = torch.cat(all_recons, dim=0)
    metrics = EMD_CD(all_recons, all_refs, batch_size=args.val_batch_size)
    cd, emd = metrics['MMD-CD'].item(), metrics['MMD-EMD'].item()
    
    logger.info('[Val] Iter %04d | CD %.6f | EMD %.6f  ' % (it, cd, emd))
    writer.add_scalar('val/cd', cd, it)
    writer.add_scalar('val/emd', emd, it)
    writer.flush()

    # wandb save
    wandb.log({"iters": it,"val/cd-loss": cd})

    return cd

def validate_inspect(it):
    sum_n = 0
    sum_chamfer = 0
    for i, batch in enumerate(tqdm(val_loader, desc='Inspect')):
        # Load point cloud
        x = batch['pointcloud'].to(args.device).float()
        shift = batch['shift'].to(args.device)
        scale = batch['scale'].to(args.device)
        # Downsample the GT to the input point cloud
        PtsNum_ori = x.size(dim=1)
        input_num_points = int(x.size(dim=1)/args.input_downsample)
        pcd_sameNum_list = list(np.linspace(0, PtsNum_ori-1, input_num_points).round().astype(int))
        x_input = x[:, pcd_sameNum_list, :]

        # Load image
        batch_img = next(val_iter_img)
        img = batch_img['image'].to(args.device).float()

        model.eval()
        code = model.encode(x_input, img)
        recons = model.decode(code, x.size(1), flexibility=args.flexibility).detach()
        # Remap the generated pointcloud xyz and RGB to original map
        recons = recons * scale + shift
        vertices = recons[:args.num_inspect_pointclouds, :, :3]
        colors = torch.round(255*recons[:args.num_inspect_pointclouds, :, 3:]).type(torch.int)

        sum_n += x.size(0)
        if i >= args.num_inspect_batches:
            break   # Inspect only 5 batch

    writer.add_mesh('val/pointcloud', vertices, colors, global_step=it)
    writer.flush()

    # wandb save point cloud
    points = torch.Tensor.numpy(torch.Tensor.cpu(torch.cat((-vertices, colors), dim=2)))
    wandb.log({"point_scene": wandb.Object3D(points[0])})



# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project = args.project_name,
    name = args.run_name + '-latenDim' + str(args.latent_dim) + '-inputDownsample' + str(args.input_downsample) + '_' + datetime.datetime.now().strftime("%Y_%m_%d_%Hh%Mm"),
    
    # track hyperparameters and run metadata
    config = {
    "latent_dim": args.latent_dim,
    "architecture": "TwoBranch-SkipConnection",
    "dataset": "TartanAir",
    "max_iters": args.max_iters,
    "train_batch_size": args.train_batch_size,
    "val_batch_size": args.val_batch_size,
    }
)

# # For saving datas to analyze
# train_loss = []
# timeSt = datetime.datetime.now().strftime("%Y_%m_%d_%Hh%Mm")
# path = './plot_save/' + timeSt
# os.makedirs(path)
# Set the point cloud saving iteration inspection point
iters_inspect = (np.linspace(1, 36, 36).astype(int)**2)*1000
# Main loop
logger.info('Start training...')
try:
    if args.resume is not None:
        it = 1 + args.resume_iters
    else:
        it = 1  
    while it <= args.max_iters:
        train(it)
        if it % args.val_freq == 0 or it == args.max_iters:
            with torch.no_grad():
                cd_loss = validate_loss(it)
                if any(it == iters_inspect):
                    validate_inspect(it)
            opt_states = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            if any(it == iters_inspect):
                ckpt_mgr.save(model, args, cd_loss, opt_states, step=it)
            
            # # Save plot of iter vs loss
            # plt.plot(np.array(train_loss))
            # plt.xlabel('iter')
            # plt.ylabel('training loss')
            # file_name ='./plot_save/' + timeSt + '/' + str(it) + '_train_loss.png'
            # plt.savefig(file_name)
            # # plt.show()
        it += 1



except KeyboardInterrupt:
    # Save the point cloud generated by the latest model
    validate_inspect(it)
    cd_loss = validate_loss(it)
    ckpt_mgr.save(model, args, cd_loss, opt_states, step=it)
    logger.info('Terminating...')




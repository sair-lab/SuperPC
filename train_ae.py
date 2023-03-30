
#-------------------------------------------Two-branches Model:ResNet50 + PointNet----------------------------------------
import os
import argparse
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
import wandb

from utils.dataset import *
from utils.datasetImg import *
from utils.misc import *
from utils.data import *
from utils.transform import *
from models.autoencoder import *
from evaluation import EMD_CD

# To save plots for analyze
import matplotlib.pyplot as plt
import datetime
import os


# Arguments
parser = argparse.ArgumentParser()
# Model arguments
parser.add_argument('--latent_dim', type=int, default=2048)
parser.add_argument('--num_steps', type=int, default=200)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.05)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--flexibility', type=float, default=0.0)
parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--resume_iters', type=int, default=0)

# Datasets and loaders
parser.add_argument('--input_downsample', type=int, default=2)
# parser.add_argument('--dataset_path', type=str, default='/home/jared/SAIR_Lab/Super-Map/Super-Map-Fusion-Head-Point-Based-Model_twoBranchsModel/data/tartanair_allEnvs.hdf5') # Tartanair allEnvs
parser.add_argument('--dataset_path', type=str, default='/home/jared/SAIR_Lab/Super-Map/Super-Map-Fusion-Head-Point-Based-Model/data/shapenet_oneTraj_50000pts.hdf5')
# parser.add_argument('--datasetImg_path', type=str, default='/home/jared/Large_datasets/TartanAir/data_image') # Tartanair allEnvs
parser.add_argument('--datasetImg_path', type=str, default='/home/jared/SAIR_Lab/Super-Map/Super-Map-Fusion-Head-Point-Based-Model_twoBranchsModel/PtsDataFunc/imagedata_small')
parser.add_argument('--categories', type=str_list, default=['hospitalRGB'])
parser.add_argument('--scale_mode', type=str, default='shape_unit')
# parser.add_argument('--train_batch_size', type=int, default=128) # original
# parser.add_argument('--val_batch_size', type=int, default=32) # original
# parser.add_argument('--train_batch_size', type=int, default=32) # poits /40; 30 frames for training; 10 frames for testing
# parser.add_argument('--val_batch_size', type=int, default=8)# poits /40; 30 frames for training; 10 frames for testing
parser.add_argument('--train_batch_size', type=int, default=2) # poits /20; 30 frames for training; 10 frames for testing
parser.add_argument('--val_batch_size', type=int, default=1)# poits /20; 30 frames for training; 10 frames for testing
parser.add_argument('--rotate', type=eval, default=False, choices=[True, False])

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-4)
parser.add_argument('--sched_start_epoch', type=int, default=150*THOUSAND)
parser.add_argument('--sched_end_epoch', type=int, default=300*THOUSAND)

# wandb config
parser.add_argument('--run_name', type=str, default='Attention-TwoBranch')
parser.add_argument('--project_name', type=str, default='Super-Map-Project-SmallDataset')

# Training
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs_ae')
parser.add_argument('--device', type=str, default='cuda')
# parser.add_argument('--max_iters', type=int, default=float('inf'))
parser.add_argument('--max_iters', type=int, default=18000000)
parser.add_argument('--val_freq', type=float, default=1000)
parser.add_argument('--tag', type=str, default=None)
parser.add_argument('--num_val_batches', type=int, default=-1)
parser.add_argument('--num_inspect_batches', type=int, default=1)
parser.add_argument('--num_inspect_pointclouds', type=int, default=1)
args = parser.parse_args()
seed_all(args.seed)

# Logging
if args.logging:
    # log_dir = get_new_log_dir(args.log_root, prefix='AE_', postfix='_' + args.tag if args.tag is not None else '')
    log_dir = get_new_log_dir(args, args.log_root, prefix='AE_', postfix='_' + args.tag if args.tag is not None else '')
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)

# Datasets and loaders
transform = None
if args.rotate:
    transform = RandomRotate(180, ['pointcloud'], axis=1)
logger.info('Transform: %s' % repr(transform))
logger.info('Loading datasets...')
train_dset = ShapeNetCore(
    path=args.dataset_path,
    cates=args.categories,
    split='train',
    scale_mode=args.scale_mode,
    transform=transform,
)
val_dset = ShapeNetCore(
    path=args.dataset_path,
    cates=args.categories,
    split='val',
    scale_mode=args.scale_mode,
    transform=transform,
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
    transform=transform,
)
val_dset_img = ImageNetCore(
    path = args.datasetImg_path,
    split='val',
    transform=transform,
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
if args.resume is not None:
    logger.info('Resuming from checkpoint...')
    ckpt = torch.load(args.resume)
    model = AutoEncoder(ckpt['args']).to(args.device)
    model.load_state_dict(ckpt['state_dict'])
else:
    model = AutoEncoder(args).to(args.device)
logger.info(repr(model))


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

# Train, validate 
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

    # Forward
    loss = model.get_loss(x, img)

    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()

    logger.info('[Train] Iter %04d | Loss %.6f | Grad %.4f ' % (it, loss.item(), orig_grad_norm))
    writer.add_scalar('train/loss', loss, it)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    writer.add_scalar('train/grad_norm', orig_grad_norm, it)
    writer.flush()

    # wandb save
    wandb.log({"iters": it,"train-loss": loss, "train-lr": optimizer.param_groups[0]['lr'], "train-grad_norm": orig_grad_norm})


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
            code, fmap_skips = model.encode(ref_input, ref_img)
            recons = model.decode(code, fmap_skips, ref.size(1), flexibility=args.flexibility)
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
        code, fmap_skips = model.encode(x_input, img)
        recons = model.decode(code, fmap_skips, x.size(1), flexibility=args.flexibility).detach()
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
# timeSt = datetime.datetime.now().strftime("%Y_%m_%d_%Hh%Mm")
# path = './plot_save/' + timeSt
# os.makedirs(path)
# Set the point cloud saving iteration inspection point
iters_inspect = (np.linspace(1, 36, 36).astype(int)**2)*1000 + args.resume_iters
# Main loop
logger.info('Start training...')
try:
    if args.resume is not None:
        it = 1 + args.resume_iters
    else:
        it = 1  
    train_loss = []
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

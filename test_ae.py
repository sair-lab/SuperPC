
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
parser.add_argument('--ckpt', type=str, default='./pretrained/AE_airplane.pt')
parser.add_argument('--categories', type=str_list, default=['hospitalRGB'])
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--device', type=str, default='cuda')
# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='./data/shapenet_overfit_flip.hdf5')
# parser.add_argument('--batch_size', type=int, default=128) # orignial
# parser.add_argument('--batch_size', type=int, default=8) # poits /40; 30 frames for training; 10 frames for testing
parser.add_argument('--batch_size', type=int, default=2) # poits /20; 30 frames for training; 10 frames for testing
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
for i, batch in enumerate(tqdm(test_loader)):
    # Load image
    batch_img = next(test_iter_img)
    img = batch_img['image'].to(args.device).float()
    # Load point cloud
    ref = batch['pointcloud'].to(args.device).float()
    shift = batch['shift'].to(args.device)
    scale = batch['scale'].to(args.device)
    model.eval()
    with torch.no_grad():
        code = model.encode(ref, img)
        recons = model.decode(code, 2*ref.size(1), flexibility=ckpt['args'].flexibility).detach()

    ref = ref * scale + shift
    recons = recons * scale + shift

    all_ref.append(ref.detach().cpu())
    all_recons.append(recons.detach().cpu())

all_ref = torch.cat(all_ref, dim=0)
all_recons = torch.cat(all_recons, dim=0)

logger.info('Saving point clouds...')
np.save(os.path.join(save_dir, 'ref.npy'), all_ref.numpy())
np.save(os.path.join(save_dir, 'out.npy'), all_recons.numpy())

logger.info('Start computing metrics...')
metrics = EMD_CD(all_recons.to(args.device), all_ref.to(args.device), batch_size=args.batch_size)
cd, emd = metrics['MMD-CD'].item(), metrics['MMD-EMD'].item()
logger.info('CD:  %.12f' % cd)
logger.info('EMD: %.12f' % emd)

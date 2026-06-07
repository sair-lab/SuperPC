import argparse
from args.utils import str2bool


def parse_tartanair_args(cli_args=None):
    parser = argparse.ArgumentParser(description='Tartanair Model Arguments')
    # seed
    parser.add_argument('--seed', default=21, type=int, help='seed')
    # optimizer
    parser.add_argument('--optim', default='adam', type=str, help='optimizer, adam or sgd')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')
    # lr scheduler
    parser.add_argument('--lr_decay_step', default=20, type=int, help='learning rate decay step size')
    parser.add_argument('--gamma', default=0.5, type=float, help='gamma for scheduler_steplr')
    # dataset
    parser.add_argument('--dataset', default='tartanair', type=str, help='tartanair')
    parser.add_argument('--tartanair_root', default='/data_sair/tartanair_maps', type=str, help='Tartanair maps root directory')
    parser.add_argument('--gpu_ids', default='0', type=str, help='comma-separated visible GPU indices for DataParallel, e.g. 0,1,2')
    parser.add_argument('--val_ratio', default=0.1, type=float, help='validation ratio within Tartanair train scenes')
    parser.add_argument('--split_seed', default=21, type=int, help='seed for Tartanair train/val split')
    parser.add_argument('--eval_split', default='val', type=str, help='evaluation split for training: val or test')
    parser.add_argument('--overfit_test_split', default=False, type=str2bool, help='if true, train Tartanair on test scenes for overfitting diagnostics')
    parser.add_argument('--overfit_eval_max_samples', default=0, type=int, help='when overfit mode is true, cap eval test samples (0 means auto cap to val split size)')
    parser.add_argument('--overfit_single_sample', default=False, type=str2bool, help='if true, train/eval on exactly one sample from the selected split')
    parser.add_argument('--overfit_single_sample_index', default=0, type=int, help='sample index used when overfit_single_sample=true (wraps around split size)')
    parser.add_argument('--num_points', default=11520, type=int, help='the points number of each input patch before occlusion')
    parser.add_argument('--target_num_points', default=46080, type=int, help='target output point count for training/eval/test')
    parser.add_argument('--up_rate', default=4, type=int, help='upsampling rate')
    parser.add_argument('--skip_rate', default=1, type=int, help='used for dataset')
    parser.add_argument('--use_random_input', default=False, type=str2bool, help='whether use random sampling for input generation')
    parser.add_argument('--jitter_sigma', type=float, default=0.01, help='jitter augmentation')
    parser.add_argument('--jitter_max', type=float, default=0.03, help='jitter augmentation')
    parser.add_argument('--input_noise_std_min', default=0.005, type=float, help='minimum Gaussian std for input noise (normalized units)')
    parser.add_argument('--input_noise_std_max', default=0.02, type=float, help='maximum Gaussian std for input noise (normalized units)')
    parser.add_argument('--midpoint_downsample_mode', default='fps', choices=['fps', 'hybrid'], type=str, help='downsampling after midpoint candidate generation: fps (quality) or hybrid (faster random+small-fps)')
    parser.add_argument('--midpoint_hybrid_fps_ratio', default=0.25, type=float, help='FPS fraction in hybrid midpoint downsampling (0.0-1.0)')
    parser.add_argument('--use_hybrid_initialization', default=False, type=str2bool, help='enable hybrid midpoint + scout-point initialization')
    parser.add_argument('--use_input_scout_fill', default=False, type=str2bool, help='skip midpoint interpolation and fill input to target count with scout points only')
    parser.add_argument('--use_patch_emd', default=True, type=str2bool, help='enable patch-wise EMD alignment during training')
    parser.add_argument('--patch_emd_patch_size', default=1024, type=int, help='points per patch for patch-wise EMD alignment')
    parser.add_argument('--diffusion_steps', default=1000, type=int, help='number of DDPM training timesteps')
    parser.add_argument('--diffusion_beta_schedule', default='cosine', choices=['cosine', 'linear'], type=str, help='DDPM beta schedule')
    parser.add_argument('--prediction_type', default='epsilon', choices=['epsilon'], type=str, help='diffusion prediction target')
    parser.add_argument('--ddim_eta', default=0.0, type=float, help='DDIM eta; 0 is deterministic')
    parser.add_argument('--sampling_steps', default=50, type=int, help='DDIM sampling steps for eval/test')
    parser.add_argument('--sparse_cond_k', default=16, type=int, help='sparse conditioning nearest neighbors per noisy target point')
    parser.add_argument('--sparse_cond_channels', default=64, type=int, help='sparse conditioning feature channels')
    parser.add_argument('--hybrid_scout_ratio', default=0.3, type=float, help='fraction of scout points sampled in bbox for hybrid initialization')
    parser.add_argument('--hybrid_bbox_padding', default=0.1, type=float, help='deprecated no-op (scout points are sampled in tight observed bbox)')
    parser.add_argument('--input_occlusion_ratio_min', default=0.1, type=float, help='minimum fraction of sparse input points removed around random local centers')
    parser.add_argument('--input_occlusion_ratio_max', default=0.25, type=float, help='maximum fraction of sparse input points removed around random local centers')
    parser.add_argument('--input_occlusion_ratio', default=None, type=float, help='optional fixed occlusion ratio override')
    parser.add_argument('--num_occlusion_areas', default=3, type=int, help='number of random local regions removed for incompletion')
    parser.add_argument(
        '--use_preprocessed_input',
        '--tartanair_use_preprocessed_input',
        dest='use_preprocessed_input',
        default=False,
        type=str2bool,
        help='load offline-generated imperfect input/gt caches when available',
    )
    parser.add_argument('--patch_rate', default=3, type=int, help='used for patch generation')
    # train
    parser.add_argument('--epochs', default=200, type=int, help='training epochs')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='workers number')
    parser.add_argument('--print_rate', default=1, type=int, help='loss print frequency in each epoch')
    parser.add_argument('--save_rate', default=10, type=int, help='testing/evaluation frequency')
    parser.add_argument('--ckpt_save_interval', default=None, type=int, help='periodic checkpoint save interval in epochs (default: same as save_rate)')
    parser.add_argument('--use_wandb', default=True, type=str2bool, help='enable wandb logging')
    parser.add_argument('--wandb_project', default='SuperPC_tartanair', type=str, help='wandb project name')
    parser.add_argument('--wandb_entity', default=None, type=str, help='wandb entity/user/team')
    parser.add_argument('--wandb_run_name', default='tartanair_with_image_run_128d', type=str, help='wandb run name')
    parser.add_argument('--debug_timing', default=True, type=str2bool, help='log detailed training timing breakdown')
    parser.add_argument('--timing_log_rate', default=100, type=int, help='steps between timing debug logs')

    # vision conditioning
    parser.add_argument('--use_vision_conditioning', default=True, type=str2bool, help='enable Depth Anything V2 image conditioning')
    parser.add_argument('--vision_pretrained_id', default='depth-anything/Depth-Anything-V2-Small-hf', type=str, help='hf id for depth model')
    parser.add_argument('--vision_cache_dir', default=None, type=str, help='optional local cache for hf models')
    parser.add_argument('--vision_image_dir', default=None, type=str, help='directory with per-sample RGB images')
    parser.add_argument('--vision_intrinsics_dir', default=None, type=str, help='directory with per-sample intrinsics (.npy/.npz/.txt/.json)')
    parser.add_argument('--vision_intrinsics_path', default=None, type=str, help='single global intrinsics file')
    parser.add_argument('--vision_img_height', default=224, type=int, help='input image height for depth model')
    parser.add_argument('--vision_img_width', default=224, type=int, help='input image width for depth model')
    parser.add_argument('--vision_attn_d_model', default=128, type=int, help='cross-attention latent dimension')
    parser.add_argument('--vision_attn_heads', default=4, type=int, help='cross-attention number of heads')

    args = parser.parse_args(cli_args)
    return args

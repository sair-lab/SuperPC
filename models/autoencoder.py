# import torch
# from torch.nn import Module

# from .encoders import *
# from .diffusion import *


# class AutoEncoder(Module):

#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         self.encoder = PointNetEncoder(zdim=args.latent_dim)
#         self.diffusion = DiffusionPoint(
#             net = PointwiseNet(point_dim=6, context_dim=args.latent_dim, residual=args.residual),
#             var_sched = VarianceSchedule(
#                 num_steps=args.num_steps,
#                 beta_1=args.beta_1,
#                 beta_T=args.beta_T,
#                 mode=args.sched_mode
#             )
#         )

#     def encode(self, x):
#         """
#         Args:
#             x:  Point clouds to be encoded, (B, N, d).
#         """
#         code, _ = self.encoder(x)
#         return code

#     def decode(self, code, num_points, flexibility=0.0, ret_traj=False):
#         return self.diffusion.sample(num_points, code, flexibility=flexibility, ret_traj=ret_traj)

#     def get_loss(self, x):
#         code = self.encode(x)
#         loss = self.diffusion.get_loss(x, code)
#         return loss


#-------------------------------------------Two-branches Model:ResNet50 + PointNet----------------------------------------
import torch
from torch.nn import Module

from .encoders import *
from .diffusion import *


class AutoEncoder(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoder(zdim=args.latent_dim, input_downsample=args.input_downsample)
        self.diffusion = DiffusionPoint(
            net = PointwiseNet(point_dim=6, context_dim=args.latent_dim, residual=args.residual),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )

    def encode(self, x, img):
        """
        Args:
            x:  Point clouds to be encoded, (B, N, d).
            img: Images to be encoded, (B, H, W)
        """
        code, _, fmap_skips = self.encoder(x, img)
        return code, fmap_skips

    def decode(self, code, fmap_skips, num_points, flexibility=0.0, ret_traj=False):
        return self.diffusion.sample(num_points, code, fmap_skips, flexibility=flexibility, ret_traj=ret_traj)

    def get_loss(self, x, img):
        PtsNum_ori = x.size(dim=1)
        input_num_points = int(x.size(dim=1)/self.args.input_downsample)
        pcd_sameNum_list = list(np.linspace(0, PtsNum_ori-1, input_num_points).round().astype(int))
        x_input = x[:, pcd_sameNum_list, :]
       
        code, fmap_skips = self.encode(x_input, img)
        # code = self.encode(x, img)
        loss = self.diffusion.get_loss(x, code, fmap_skips)
        return loss

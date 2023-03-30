import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np

from .common import *
from .encoders import *


class VarianceSchedule(Module):

    def __init__(self, num_steps, beta_1, beta_T, mode='linear'):
        super().__init__()
        assert mode in ('linear', )
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas

# # Original PointwiseNet
# class PointwiseNet(Module):

#     def __init__(self, point_dim, context_dim, residual):
#         super().__init__()
#         self.act = F.leaky_relu
#         self.residual = residual
#         self.layers = ModuleList([
#             ConcatSquashLinear(6, 256, context_dim+3),
#             ConcatSquashLinear(256, 512, context_dim+3),
#             ConcatSquashLinear(512, 2048, context_dim+3),
#             ConcatSquashLinear(2048, 512, context_dim+3),
#             ConcatSquashLinear(512, 256, context_dim+3),
#             ConcatSquashLinear(256, 6, context_dim+3)
#         ])

#     def forward(self, x, beta, context):
#         """
#         Args:
#             x:  Point clouds at some timestep t, (B, N, d).
#             beta:     Time. (B, ).
#             context:  Shape latents. (B, F).
#         """
#         batch_size = x.size(0)
#         beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
#         context = context.view(batch_size, 1, -1)   # (B, 1, F)
 
#         time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
#         ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

#         out = x
#         for i, layer in enumerate(self.layers):
#             out = layer(ctx=ctx_emb, x=out)
#             if i < len(self.layers) - 1:
#                 out = self.act(out)

#         if self.residual:
#             return x + out
#         else:
#             return out

# Attention-based skip-connection PointwiseNet
class PointwiseNet(Module):

    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual 
        # PointNet++ layers
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 6 + 3, [32, 64, 256], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 256 + 3, [256, 256, 512], False)
        self.sa3 = PointNetSetAbstraction(16, 0.8, 32, 512 + 3, [512, 512, 2048], False)
        self.fp3 = PointNetFeaturePropagation(2048+512, [512, 512])
        self.fp2 = PointNetFeaturePropagation(512+256, [512, 256])
        self.fp1 = PointNetFeaturePropagation(256, [128, 6])
        # self.conv1 = nn.Conv1d(128, 128, 1)
        self.conv1 = nn.Conv1d(6, 6, 1)
        self.bn1 = nn.BatchNorm1d(6)
        
        #concateSquashLinear Layers
        self.layers = ModuleList([
            ConcatSquashLinear(6, 256, context_dim+3, pointNet2Layer = self.sa1),
            ConcatSquashLinear(256, 512, context_dim+3, pointNet2Layer = self.sa2),
            ConcatSquashLinear(512, 2048, context_dim+3, pointNet2Layer = self.sa3),
            ConcatSquashLinearUp(2048, 512, context_dim+3, pointNet2Layer = self.fp3),
            ConcatSquashLinearUp(512, 256, context_dim+3, pointNet2Layer = self.fp2),
            ConcatSquashLinearUp(256, 6, context_dim+3, pointNet2Layer = self.fp1)
        ])
        

        # Fully-connected layers for the feature map concatate
        self.layers_fc = ModuleList([
            torch.nn.Linear(512, 256),
            # torch.nn.Linear(1024, 512),
            # torch.nn.Linear(4096, 2048),
            # torch.nn.Linear(1024, 512),
            # torch.nn.Linear(512, 256)
        ])


    def forward(self, x, beta, context, fmap_skips):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F).
        """
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)
 
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        fmap_skips.append(fmap_skips[0])
        out = x
        out_xyz = x[:, :, :3]
        out_list = [None]
        out_xyz_list = [out_xyz]
        for i, layer in enumerate(self.layers):
            if 0 <= i <= 2:
                out_xyz, out = layer(ctx=ctx_emb, out_xyz=out_xyz, x=out)
                out_xyz_list.append(out_xyz)
            else:
                out_xyz = out_xyz_list[-(i-2)]
                out_xyz_skip = out_xyz_list[-(i-2+1)]
                out_skip = out_list[-(i-2+1)]
                out = layer(ctx=ctx_emb, out_xyz_skip=out_xyz_skip, out_xyz=out_xyz, x_skip=out_skip, x=out)

            # Skip-connection
            if i <= 1-1: 
                out = torch.cat((out, fmap_skips[i]), dim=-1)
                fc_layer = self.layers_fc[i]
                out = fc_layer(out)
            # Leaky-relu
            if i < len(self.layers) - 1:
                out = self.act(out)

            if 0 <= i <= 2:
                out_list.append(out)
        
        out = out.transpose(1, 2)
        out = self.bn1(self.conv1(out))
        out = out.transpose(1, 2)

        if self.residual:
            return x + out
        else:
            return out




class DiffusionPoint(Module):

    def __init__(self, net, var_sched:VarianceSchedule):
        super().__init__()
        self.net = net
        self.var_sched = var_sched

    def get_loss(self, x_0, context, fmap_skips, t=None):
        """
        Args:
            x_0:  Input point cloud, (B, N, d).
            context:  Shape latent, (B, F).
        """
        batch_size, _, point_dim = x_0.size()
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)   # (B, 1, 1)

        e_rand = torch.randn_like(x_0)  # (B, N, d)
        e_theta = self.net(c0 * x_0 + c1 * e_rand, beta=beta, context=context, fmap_skips=fmap_skips)

        loss = F.mse_loss(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')
        return loss

    def sample(self, num_points, context, fmap_skips, point_dim=6, flexibility=0.0, ret_traj=False):
        batch_size = context.size(0)
        x_T = torch.randn([batch_size, num_points, point_dim]).to(context.device)
        traj = {self.var_sched.num_steps: x_T}
        for t in range(self.var_sched.num_steps, 0, -1):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]
            beta = self.var_sched.betas[[t]*batch_size]
            e_theta = self.net(x_t, beta=beta, context=context, fmap_skips=fmap_skips)
            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]
        
        if ret_traj:
            return traj
        else:
            return traj[0]


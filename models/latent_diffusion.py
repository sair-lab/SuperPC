import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ddpm import DDPMScheduler
from models.pointnet2_util import PointNetSetAbstractionMsg
from models.utils import get_knn_pts, index_points
from models.vision_condition import FrozenDepthAnythingV2


def _time_embedding(t, dim):
    t = t.float().view(-1, 1)
    half = max(1, dim // 2)
    freqs = torch.exp(
        torch.linspace(
            math.log(1.0),
            math.log(1000.0),
            half,
            device=t.device,
            dtype=t.dtype,
        )
    ).view(1, -1)
    emb = torch.cat([torch.sin(t * freqs), torch.cos(t * freqs)], dim=1)
    if emb.shape[1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[1]))
    return emb[:, :dim]


def _make_folding_grid(points_per_token):
    side = int(math.ceil(math.sqrt(int(points_per_token))))
    lin = torch.linspace(-1.0, 1.0, side)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=0)
    return grid[:, : int(points_per_token)].contiguous()


def _make_1d_norm(channels, norm_type="batch"):
    norm_type = str(norm_type).lower()
    channels = int(channels)
    if norm_type == "batch":
        return nn.BatchNorm1d(channels)
    if norm_type == "group":
        groups = min(8, channels)
        while groups > 1 and channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if norm_type in ("none", "identity"):
        return nn.Identity()
    raise ValueError(f"Unsupported 1D norm type: {norm_type}")


def _make_2d_norm(channels, norm_type="batch"):
    norm_type = str(norm_type).lower()
    channels = int(channels)
    if norm_type == "batch":
        return nn.BatchNorm2d(channels)
    if norm_type == "group":
        groups = min(8, channels)
        while groups > 1 and channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if norm_type in ("none", "identity"):
        return nn.Identity()
    raise ValueError(f"Unsupported 2D norm type: {norm_type}")


class LatentPointAutoEncoder(nn.Module):
    """Point-token autoencoder for dense point clouds.

    Input/output point clouds use BNC layout. Latents use BCT layout, where the
    first 3 channels are token xyz and the remaining channels are learned token
    features.
    """

    def __init__(
        self,
        latent_tokens=512,
        latent_channels=96,
        target_num_points=46080,
        ae_variant="current",
        norm_type="batch",
    ):
        super().__init__()
        self.latent_tokens = int(latent_tokens)
        self.latent_channels = int(latent_channels)
        self.target_num_points = int(target_num_points)
        self.latent_dim = self.latent_channels + 3
        self.points_per_token = int(math.ceil(float(self.target_num_points) / float(self.latent_tokens)))
        self.ae_variant = str(ae_variant).lower()
        if self.ae_variant == "current_gn":
            self.ae_variant = "current"
            norm_type = "group"
        if self.ae_variant not in ("current", "query_v2", "query_v2_ms"):
            raise ValueError(f"Unsupported AE variant: {ae_variant}")
        self.norm_type = str(norm_type).lower()

        branch = max(16, self.latent_channels // 2)
        self.encoder_sa = PointNetSetAbstractionMsg(
            self.latent_tokens,
            [0.05, 0.1, 0.2],
            [16, 32, 64],
            0,
            [
                [branch // 2, branch // 2, branch],
                [branch // 2, branch, branch],
                [branch, branch, self.latent_channels - 2 * branch if self.latent_channels > 2 * branch else branch],
            ],
        )
        enc_out = branch + branch + (self.latent_channels - 2 * branch if self.latent_channels > 2 * branch else branch)
        self.encoder_proj = nn.Sequential(
            nn.Conv1d(enc_out, self.latent_channels, 1),
            _make_1d_norm(self.latent_channels, self.norm_type),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.latent_channels, self.latent_channels, 1),
        )

        self.query_support_tokens = min(2048, max(1024, self.latent_tokens * 2))
        query_dim = max(96, self.latent_channels)
        support_branch = max(32, query_dim // 2)
        self.query_support_sa = PointNetSetAbstractionMsg(
            self.query_support_tokens,
            [0.05, 0.1, 0.2],
            [16, 32, 64],
            0,
            [[32, support_branch], [32, support_branch], [64, query_dim]],
        )
        support_out = support_branch + support_branch + query_dim
        self.query_support_proj = nn.Sequential(
            nn.Conv1d(support_out + 3, query_dim, 1),
            _make_1d_norm(query_dim, self.norm_type),
            nn.ReLU(inplace=True),
            nn.Conv1d(query_dim, query_dim, 1),
        )
        self.latent_queries = nn.Parameter(torch.randn(1, self.latent_tokens, query_dim) * 0.02)
        self.query_attn = nn.MultiheadAttention(query_dim, num_heads=4, batch_first=True)
        self.query_norm1 = nn.LayerNorm(query_dim)
        self.query_ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(query_dim * 2, query_dim),
        )
        self.query_norm2 = nn.LayerNorm(query_dim)
        self.query_anchor = nn.Linear(query_dim, 3)
        self.query_feat = nn.Linear(query_dim, self.latent_channels)

        self.register_buffer("folding_grid", _make_folding_grid(self.points_per_token), persistent=False)
        dec_in = self.latent_dim + 2
        dec_hidden = (384, 384, 192) if self.ae_variant == "query_v2_ms" else (256, 256, 128)
        self.decoder = nn.Sequential(
            nn.Conv2d(dec_in, dec_hidden[0], 1),
            _make_2d_norm(dec_hidden[0], self.norm_type),
            nn.ReLU(inplace=True),
            nn.Conv2d(dec_hidden[0], dec_hidden[1], 1),
            _make_2d_norm(dec_hidden[1], self.norm_type),
            nn.ReLU(inplace=True),
            nn.Conv2d(dec_hidden[1], dec_hidden[2], 1),
            _make_2d_norm(dec_hidden[2], self.norm_type),
            nn.ReLU(inplace=True),
            nn.Conv2d(dec_hidden[2], 3, 1),
        )

    def encode(self, points_bnc):
        if points_bnc.ndim != 3 or points_bnc.shape[-1] != 3:
            raise ValueError(f"Expected points shape (B,N,3), got {tuple(points_bnc.shape)}")
        if self.ae_variant in ("query_v2", "query_v2_ms"):
            return self._encode_query_v2(points_bnc)
        xyz = points_bnc.transpose(1, 2).contiguous()
        token_xyz, token_feat = self.encoder_sa(xyz, None)
        token_feat = self.encoder_proj(token_feat)
        return torch.cat([token_xyz, token_feat], dim=1).contiguous()

    def _encode_query_v2(self, points_bnc):
        xyz = points_bnc.transpose(1, 2).contiguous()
        support_xyz, support_feat = self.query_support_sa(xyz, None)
        support = self.query_support_proj(torch.cat([support_xyz, support_feat], dim=1))
        support = support.transpose(1, 2).contiguous()
        queries = self.latent_queries.expand(points_bnc.shape[0], -1, -1)
        attn_out, _ = self.query_attn(queries, support, support, need_weights=False)
        queries = self.query_norm1(queries + attn_out)
        queries = self.query_norm2(queries + self.query_ffn(queries))
        token_xyz = torch.tanh(self.query_anchor(queries)).transpose(1, 2).contiguous()
        token_feat = self.query_feat(queries).transpose(1, 2).contiguous()
        return torch.cat([token_xyz, token_feat], dim=1).contiguous()

    def decode(self, latent_bct):
        if latent_bct.ndim != 3 or latent_bct.shape[1] != self.latent_dim:
            raise ValueError(f"Expected latent shape (B,{self.latent_dim},T), got {tuple(latent_bct.shape)}")
        bsz, _channels, tokens = latent_bct.shape
        if tokens != self.latent_tokens:
            raise ValueError(f"Expected {self.latent_tokens} latent tokens, got {tokens}")

        grid = self.folding_grid.to(device=latent_bct.device, dtype=latent_bct.dtype)
        grid = grid.view(1, 2, 1, self.points_per_token).expand(bsz, -1, tokens, -1)
        latent = latent_bct.unsqueeze(-1).expand(-1, -1, -1, self.points_per_token)
        dec_in = torch.cat([latent, grid], dim=1)
        offsets = self.decoder(dec_in)
        token_xyz = latent_bct[:, :3, :].unsqueeze(-1)
        points = token_xyz + offsets
        points = points.permute(0, 2, 3, 1).contiguous().view(bsz, tokens * self.points_per_token, 3)
        return points[:, : self.target_num_points, :].contiguous()

    def forward(self, points_bnc):
        latent = self.encode(points_bnc)
        decoded = self.decode(latent)
        return decoded, latent


class ImperfectConditionEncoder(nn.Module):
    def __init__(self, condition_tokens=512, condition_channels=128, norm_type="batch"):
        super().__init__()
        self.condition_tokens = int(condition_tokens)
        self.condition_channels = int(condition_channels)
        self.norm_type = str(norm_type).lower()
        self.sa = PointNetSetAbstractionMsg(
            self.condition_tokens,
            [0.05, 0.1, 0.2],
            [16, 32, 64],
            0,
            [[32, 32, 64], [32, 64, 64], [64, 64, 128]],
        )
        self.proj = nn.Sequential(
            nn.Conv1d(256, self.condition_channels, 1),
            _make_1d_norm(self.condition_channels, self.norm_type),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.condition_channels, self.condition_channels, 1),
        )
        self.global_proj = nn.Sequential(
            nn.Linear(self.condition_channels, self.condition_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.condition_channels, self.condition_channels),
        )

    def forward(self, imperfect_bnc):
        if imperfect_bnc.ndim != 3 or imperfect_bnc.shape[-1] != 3:
            raise ValueError(f"Expected imperfect shape (B,N,3), got {tuple(imperfect_bnc.shape)}")
        xyz = imperfect_bnc.transpose(1, 2).contiguous()
        cond_xyz, cond_feat = self.sa(xyz, None)
        cond_feat = self.proj(cond_feat)
        cond_global = self.global_proj(torch.max(cond_feat, dim=2)[0])
        return cond_xyz.contiguous(), cond_feat.contiguous(), cond_global.contiguous()


class SimpleImageGlobalEncoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(
            nn.Linear(128, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, image_tensor):
        feat = self.net(image_tensor.float()).flatten(1)
        return self.proj(feat)


class DepthAnythingGlobalEncoder(nn.Module):
    def __init__(self, out_channels, pretrained_id="depth-anything/Depth-Anything-V2-Small-hf", cache_dir=None):
        super().__init__()
        if cache_dir in ("", "none", "None"):
            cache_dir = None
        self.backbone = FrozenDepthAnythingV2(pretrained_id=pretrained_id, cache_dir=cache_dir)
        self.proj = nn.Sequential(
            nn.LazyLinear(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, image_tensor):
        feat = self.backbone(image_tensor.float())
        feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        return self.proj(feat)


class CachedImageGlobalEncoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LazyLinear(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, image_tensor):
        feat = image_tensor.float()
        if feat.ndim == 4:
            feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        elif feat.ndim > 2:
            feat = feat.flatten(1)
        return self.proj(feat)


class CachedImageSpatialTokenEncoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LazyLinear(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, image_tensor):
        feat = image_tensor.float()
        if feat.ndim == 4:
            feat = feat.flatten(2).transpose(1, 2).contiguous()
        elif feat.ndim == 2:
            feat = feat.unsqueeze(1)
        elif feat.ndim != 3:
            feat = feat.flatten(1).unsqueeze(1)
        return self.proj(feat)


def _make_image_global_encoder(
    image_encoder_type,
    out_channels,
    vision_pretrained_id="depth-anything/Depth-Anything-V2-Small-hf",
    vision_cache_dir=None,
):
    image_encoder_type = str(image_encoder_type).lower()
    if image_encoder_type == "simple_cnn":
        return SimpleImageGlobalEncoder(out_channels)
    if image_encoder_type == "depth_anything_global":
        return DepthAnythingGlobalEncoder(
            out_channels,
            pretrained_id=vision_pretrained_id,
            cache_dir=vision_cache_dir,
        )
    if image_encoder_type == "depth_anything_cached":
        return CachedImageGlobalEncoder(out_channels)
    if image_encoder_type == "depth_anything_spatial_cached":
        return CachedImageSpatialTokenEncoder(out_channels)
    raise ValueError(f"Unsupported image_encoder_type: {image_encoder_type}")


class LatentDenoiser(nn.Module):
    def __init__(
        self,
        latent_dim,
        latent_tokens=512,
        condition_encoder=None,
        condition_tokens=512,
        condition_channels=128,
        hidden_channels=256,
        knn_k=16,
        time_dim=64,
        use_condition_prior=False,
        use_image_conditioning=False,
        image_encoder_type="simple_cnn",
        vision_pretrained_id="depth-anything/Depth-Anything-V2-Small-hf",
        vision_cache_dir=None,
        norm_type="batch",
        image_token_dropout=0.0,
        image_condition_dropout_prob=0.0,
        geometry_condition_dropout_prob=0.0,
        use_image_pseudo_cloud=False,
        condition_fusion_mode="geom_only",
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.latent_tokens = int(latent_tokens)
        self.knn_k = int(knn_k)
        self.time_dim = int(time_dim)
        self.use_condition_prior = bool(use_condition_prior)
        self.use_image_conditioning = bool(use_image_conditioning)
        self.image_encoder_type = str(image_encoder_type).lower()
        self.use_spatial_image_conditioning = self.use_image_conditioning and self.image_encoder_type == "depth_anything_spatial_cached"
        self.norm_type = str(norm_type).lower()
        self.image_token_dropout = float(image_token_dropout)
        self.image_condition_dropout_prob = float(image_condition_dropout_prob)
        self.geometry_condition_dropout_prob = float(geometry_condition_dropout_prob)
        self.use_image_pseudo_cloud = bool(use_image_pseudo_cloud)
        self.condition_fusion_mode = str(condition_fusion_mode).lower()
        self.condition_encoder = condition_encoder or ImperfectConditionEncoder(
            condition_tokens=condition_tokens,
            condition_channels=condition_channels,
            norm_type=self.norm_type,
        )
        self.pseudo_condition_encoder = (
            ImperfectConditionEncoder(
                condition_tokens=condition_tokens,
                condition_channels=condition_channels,
                norm_type=self.norm_type,
            )
            if self.use_image_pseudo_cloud and self.condition_fusion_mode == "geom_plus_pseudo_tokens"
            else None
        )
        self.time_proj = nn.Sequential(
            nn.Linear(self.time_dim, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.image_encoder = (
            _make_image_global_encoder(
                self.image_encoder_type,
                condition_channels,
                vision_pretrained_id=vision_pretrained_id,
                vision_cache_dir=vision_cache_dir,
            )
            if self.use_image_conditioning
            else None
        )
        if self.use_spatial_image_conditioning:
            heads = 4
            while heads > 1 and condition_channels % heads != 0:
                heads -= 1
            self.image_query_proj = nn.Conv1d(self.latent_dim, condition_channels, 1)
            self.projection_point_head = nn.Sequential(
                nn.Conv1d(self.latent_dim, condition_channels, 1),
                _make_1d_norm(condition_channels, self.norm_type),
                nn.ReLU(inplace=True),
                nn.Conv1d(condition_channels, condition_channels, 1),
            )
            self.image_cross_attn = nn.MultiheadAttention(condition_channels, num_heads=heads, batch_first=True)
            self.image_attn_norm = nn.LayerNorm(condition_channels)
            self.image_gate = nn.Sequential(
                nn.Conv1d(self.latent_dim + condition_channels, condition_channels, 1),
                nn.Sigmoid(),
            )
            self.prior_image_query = nn.Parameter(torch.zeros(1, condition_channels, self.latent_tokens))
            self.prior_image_proj = nn.Conv1d(condition_channels, hidden_channels, 1)
            self.prior_image_gate = nn.Sequential(
                nn.Conv1d(condition_channels, hidden_channels, 1),
                nn.Sigmoid(),
            )
        else:
            self.image_query_proj = None
            self.projection_point_head = None
            self.image_cross_attn = None
            self.image_attn_norm = None
            self.image_gate = None
            self.prior_image_query = None
            self.prior_image_proj = None
            self.prior_image_gate = None
        in_channels = self.latent_dim + condition_channels + condition_channels + 3 + 1 + hidden_channels
        if self.use_spatial_image_conditioning:
            in_channels += condition_channels
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 1),
            _make_1d_norm(hidden_channels, self.norm_type),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, hidden_channels, 1),
            _make_1d_norm(hidden_channels, self.norm_type),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, hidden_channels, 1),
            _make_1d_norm(hidden_channels, self.norm_type),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, self.latent_dim, 1),
        )
        if self.use_condition_prior:
            self.prior_token_embed = nn.Parameter(torch.zeros(1, hidden_channels, self.latent_tokens))
            self.prior_global = nn.Sequential(
                nn.Linear(condition_channels, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(inplace=True),
            )
            self.prior_out = nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels, 1),
                _make_1d_norm(hidden_channels, self.norm_type),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_channels, self.latent_dim, 1),
            )
        else:
            self.prior_token_embed = None
            self.prior_global = None
            self.prior_out = None

    def _maybe_drop_image_tensor(self, image_tensor, batch_size, device):
        if image_tensor is None:
            return None
        if self.training and self.image_condition_dropout_prob > 0.0:
            keep = (
                torch.rand(int(batch_size), 1, device=device)
                >= self.image_condition_dropout_prob
            ).float()
            view_shape = [int(batch_size)] + [1] * (image_tensor.ndim - 1)
            image_tensor = image_tensor * keep.view(*view_shape).to(image_tensor.device)
        return image_tensor

    def _attend_image_tokens(self, query_btc, image_tensor):
        if not self.use_spatial_image_conditioning or image_tensor is None:
            return None
        image_tokens = self.encode_image_tokens(image_tensor.to(query_btc.device))
        if self.training and self.image_token_dropout > 0.0:
            image_tokens = F.dropout(image_tokens, p=self.image_token_dropout, training=True)
        attn_out, _ = self.image_cross_attn(query_btc, image_tokens, image_tokens, need_weights=False)
        return self.image_attn_norm(query_btc + attn_out)

    def encode_image_tokens(self, image_tensor):
        if not self.use_spatial_image_conditioning or image_tensor is None:
            return None
        return self.image_encoder(image_tensor.float())

    def project_latent_tokens_for_alignment(self, latent_bct):
        if self.projection_point_head is None:
            return None
        return self.projection_point_head(latent_bct).transpose(1, 2).contiguous()

    def _encode_condition(self, imperfect_bnc, image_tensor=None, pseudo_cloud_bnc=None):
        condition_input = imperfect_bnc
        if self.use_image_pseudo_cloud and pseudo_cloud_bnc is not None:
            pseudo_cloud_bnc = pseudo_cloud_bnc.to(imperfect_bnc.device, imperfect_bnc.dtype)
            if self.condition_fusion_mode == "pseudo_only":
                condition_input = pseudo_cloud_bnc
            elif self.condition_fusion_mode == "geom_plus_pseudo":
                condition_input = torch.cat([imperfect_bnc, pseudo_cloud_bnc], dim=1)
        cond_xyz, cond_feat, cond_global = self.condition_encoder(condition_input)
        if (
            self.use_image_pseudo_cloud
            and pseudo_cloud_bnc is not None
            and self.condition_fusion_mode == "geom_plus_pseudo_tokens"
            and self.pseudo_condition_encoder is not None
        ):
            pseudo_xyz, pseudo_feat, pseudo_global = self.pseudo_condition_encoder(pseudo_cloud_bnc)
            cond_xyz = torch.cat([cond_xyz, pseudo_xyz], dim=2)
            cond_feat = torch.cat([cond_feat, pseudo_feat], dim=2)
            cond_global = cond_global + pseudo_global
        if self.training and self.geometry_condition_dropout_prob > 0.0:
            keep = (
                torch.rand(cond_global.shape[0], 1, device=cond_global.device)
                >= self.geometry_condition_dropout_prob
            ).float()
            cond_feat = cond_feat * keep.view(-1, 1, 1)
            cond_global = cond_global * keep
            cond_xyz = cond_xyz * keep.view(-1, 1, 1)
        if self.use_image_conditioning and (not self.use_spatial_image_conditioning) and image_tensor is not None:
            cond_global = cond_global + self.image_encoder(image_tensor.to(imperfect_bnc.device))
        return cond_xyz, cond_feat, cond_global

    def forward(self, z_t, imperfect_bnc, t, image_tensor=None, pseudo_cloud_bnc=None, return_residual=False):
        if z_t.ndim != 3 or z_t.shape[1] != self.latent_dim:
            raise ValueError(f"Expected latent shape (B,{self.latent_dim},T), got {tuple(z_t.shape)}")
        if z_t.shape[-1] != self.latent_tokens:
            raise ValueError(f"Expected {self.latent_tokens} latent tokens, got {z_t.shape[-1]}")
        image_tensor = self._maybe_drop_image_tensor(image_tensor, z_t.shape[0], z_t.device)
        cond_xyz, cond_feat, cond_global = self._encode_condition(
            imperfect_bnc,
            image_tensor=image_tensor,
            pseudo_cloud_bnc=pseudo_cloud_bnc,
        )
        z_xyz = z_t[:, :3, :]
        k = min(max(1, self.knn_k), int(cond_xyz.shape[-1]))
        knn_idx = get_knn_pts(k, cond_xyz, z_xyz, return_idx=True)[1].long()
        knn_feat = index_points(cond_feat, knn_idx).max(dim=-1)[0]
        knn_xyz = index_points(cond_xyz, knn_idx)
        rel = knn_xyz - z_xyz.unsqueeze(-1)
        rel_mean = rel.mean(dim=-1)
        rel_dist = torch.norm(rel, p=2, dim=1).mean(dim=-1).unsqueeze(1)

        time_feat = self.time_proj(_time_embedding(t, self.time_dim)).unsqueeze(-1).expand(-1, -1, z_t.shape[-1])
        global_feat = cond_global.unsqueeze(-1).expand(-1, -1, z_t.shape[-1])
        feat_parts = [z_t, knn_feat, global_feat, rel_mean, rel_dist, time_feat]
        if self.use_spatial_image_conditioning:
            image_query = self.image_query_proj(z_t).transpose(1, 2).contiguous()
            image_feat = self._attend_image_tokens(image_query, image_tensor)
            if image_feat is None:
                image_feat = z_t.new_zeros(z_t.shape[0], z_t.shape[-1], knn_feat.shape[1])
            image_feat_bct = image_feat.transpose(1, 2).contiguous()
            image_gate = self.image_gate(torch.cat([z_t, image_feat_bct], dim=1))
            feat_parts.append(image_feat_bct * image_gate)
        feat = torch.cat(feat_parts, dim=1)
        residual = self.net(feat)
        if return_residual or (not self.use_condition_prior):
            return residual

        prior_feat = self.prior_global(cond_global).unsqueeze(-1) + self.prior_token_embed
        if self.use_spatial_image_conditioning:
            prior_query = self.prior_image_query.expand(z_t.shape[0], -1, -1).transpose(1, 2).contiguous()
            prior_image = self._attend_image_tokens(prior_query, image_tensor)
            if prior_image is not None:
                prior_image_bct = prior_image.transpose(1, 2).contiguous()
                prior_image_proj = self.prior_image_proj(prior_image_bct)
                prior_feat = prior_feat + self.prior_image_gate(prior_image_bct) * prior_image_proj
        prior = self.prior_out(prior_feat)
        return prior + residual

    def condition_prior(self, imperfect_bnc, image_tensor=None, pseudo_cloud_bnc=None):
        if not self.use_condition_prior:
            raise RuntimeError("condition_prior requires use_condition_prior=True")
        image_tensor = self._maybe_drop_image_tensor(image_tensor, imperfect_bnc.shape[0], imperfect_bnc.device)
        _cond_xyz, _cond_feat, cond_global = self._encode_condition(
            imperfect_bnc,
            image_tensor=image_tensor,
            pseudo_cloud_bnc=pseudo_cloud_bnc,
        )
        prior_feat = self.prior_global(cond_global).unsqueeze(-1) + self.prior_token_embed
        if self.use_spatial_image_conditioning:
            prior_query = self.prior_image_query.expand(imperfect_bnc.shape[0], -1, -1).transpose(1, 2).contiguous()
            prior_image = self._attend_image_tokens(prior_query, image_tensor)
            if prior_image is not None:
                prior_image_bct = prior_image.transpose(1, 2).contiguous()
                prior_image_proj = self.prior_image_proj(prior_image_bct)
                prior_feat = prior_feat + self.prior_image_gate(prior_image_bct) * prior_image_proj
        return self.prior_out(prior_feat)


@torch.no_grad()
def latent_ddim_sample(
    denoiser,
    scheduler,
    imperfect_bnc,
    shape,
    sampling_steps=50,
    eta=0.0,
    clip_denoised=False,
    prediction_type="x0",
    image_tensor=None,
    pseudo_cloud_bnc=None,
    return_residual=False,
):
    device = imperfect_bnc.device
    scheduler.to(device)
    steps = max(1, int(sampling_steps))
    eta = float(eta)
    times = torch.linspace(scheduler.num_steps - 1, 0, steps, device=device).long()
    z = torch.randn(shape, device=device, dtype=imperfect_bnc.dtype)

    for i, t_value in enumerate(times):
        t = torch.full((shape[0],), int(t_value.item()), device=device, dtype=torch.long)
        model_t = scheduler.model_time(t)
        pred = denoiser(
            z,
            imperfect_bnc,
            model_t,
            image_tensor=image_tensor,
            pseudo_cloud_bnc=pseudo_cloud_bnc,
            return_residual=return_residual,
        )
        pred_type = str(prediction_type).lower()
        if pred_type == "epsilon":
            eps = pred
            z0 = scheduler.predict_x0_from_eps(z, t, eps)
        elif pred_type == "x0":
            z0 = pred
            sqrt_ab = scheduler._extract(scheduler.sqrt_alphas_cumprod, t, z.shape)
            sqrt_omab = scheduler._extract(scheduler.sqrt_one_minus_alphas_cumprod, t, z.shape)
            eps = (z - sqrt_ab * z0) / torch.clamp(sqrt_omab, min=1e-8)
        else:
            raise ValueError(f"Unsupported latent prediction_type: {prediction_type}")
        if clip_denoised:
            z0 = z0.clamp(-1.0, 1.0)
        if i == len(times) - 1:
            z = z0
            continue

        prev_t_value = int(times[i + 1].item())
        alpha_bar_t = scheduler._extract(scheduler.alphas_cumprod, t, z.shape)
        prev_t = torch.full((shape[0],), prev_t_value, device=device, dtype=torch.long)
        alpha_bar_prev = scheduler._extract(scheduler.alphas_cumprod, prev_t, z.shape)
        sigma = eta * torch.sqrt(
            torch.clamp((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t), min=0.0)
            * torch.clamp(1.0 - (alpha_bar_t / alpha_bar_prev), min=0.0)
        )
        direction_scale = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma ** 2, min=0.0))
        noise = torch.randn_like(z) if eta > 0.0 else torch.zeros_like(z)
        z = torch.sqrt(alpha_bar_prev) * z0 + direction_scale * eps + sigma * noise
    return z

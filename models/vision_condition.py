import torch
import torch.nn as nn


class FrozenDepthAnythingV2(nn.Module):
    def __init__(self, pretrained_id, cache_dir=None):
        super().__init__()
        try:
            from transformers import AutoModelForDepthEstimation
        except ImportError as exc:
            raise ImportError(
                "transformers is required for vision conditioning. "
                "Install it with: pip install transformers"
            ) from exc

        self.model = AutoModelForDepthEstimation.from_pretrained(pretrained_id, cache_dir=cache_dir)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @staticmethod
    def _pick_last_spatial_feature(outputs):
        for key in ["decoder_hidden_states", "hidden_states", "feature_maps"]:
            value = getattr(outputs, key, None)
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                for item in reversed(value):
                    if torch.is_tensor(item) and item.dim() == 4:
                        return item
            elif torch.is_tensor(value) and value.dim() == 4:
                return value

        pred_depth = getattr(outputs, "predicted_depth", None)
        if torch.is_tensor(pred_depth):
            if pred_depth.dim() == 3:
                return pred_depth.unsqueeze(1)
            if pred_depth.dim() == 4:
                return pred_depth

        raise RuntimeError("Could not find a 2D spatial feature map from Depth Anything output.")

    def forward(self, image_tensor):
        with torch.no_grad():
            outputs = self.model(pixel_values=image_tensor, output_hidden_states=True, return_dict=True)
            feat = self._pick_last_spatial_feature(outputs)
        return feat.contiguous()


def project_points_to_2d(points, intrinsics, img_shape):
    # points: [B, N, 3], intrinsics: [B, 3, 3], img_shape: (H, W)
    h, w = img_shape
    xyz = points
    z = xyz[..., 2].clamp(min=1e-6)

    u = (intrinsics[:, None, 0, 0] * xyz[..., 0] + intrinsics[:, None, 0, 2]) / z
    v = (intrinsics[:, None, 1, 1] * xyz[..., 1] + intrinsics[:, None, 1, 2]) / z

    u = 2.0 * (u / max(w - 1, 1)) - 1.0
    v = 2.0 * (v / max(h - 1, 1)) - 1.0

    uv = torch.stack([u, v], dim=-1)
    return uv.clamp(-1.0, 1.0)


class VisionCrossAttention(nn.Module):
    def __init__(self, d_point, d_img, d_model, n_heads):
        super().__init__()
        self.q_proj = nn.Linear(d_point, d_model)
        if d_img is None:
            self.k_proj = nn.LazyLinear(d_model)
            self.v_proj = nn.LazyLinear(d_model)
        else:
            self.k_proj = nn.Linear(d_img, d_model)
            self.v_proj = nn.Linear(d_img, d_model)
        self.uv_pe = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.out_proj = nn.Linear(d_model, d_point)

    def forward(self, point_features, xyz, img_features, intrinsics):
        # point_features: [B, C, N], xyz: [B, 3, N], img_features: [B, C_img, H, W], intrinsics: [B, 3, 3]
        b, _, n = point_features.shape
        _, c_img, h, w = img_features.shape

        pt = point_features.permute(0, 2, 1).contiguous()
        xyz_bn3 = xyz.permute(0, 2, 1).contiguous()

        uv = project_points_to_2d(xyz_bn3, intrinsics, (h, w))
        q = self.q_proj(pt) + self.uv_pe(uv)

        img = img_features.reshape(b, c_img, h * w).permute(0, 2, 1).contiguous()
        k = self.k_proj(img)
        v = self.v_proj(img)

        attn_out, _ = self.attn(q, k, v, need_weights=False)
        out = self.out_proj(attn_out)
        out = out.permute(0, 2, 1).contiguous()
        return point_features + out

import math

import torch


def _cosine_beta_schedule(num_steps, s=0.008):
    steps = int(num_steps)
    x = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-8, 0.999).float()


def _linear_beta_schedule(num_steps, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, int(num_steps), dtype=torch.float32)


class DDPMScheduler:
    def __init__(self, num_steps=1000, beta_schedule="cosine"):
        self.num_steps = int(num_steps)
        if self.num_steps <= 1:
            raise ValueError("num_steps must be > 1")

        schedule = str(beta_schedule).strip().lower()
        if schedule == "cosine":
            betas = _cosine_beta_schedule(self.num_steps)
        elif schedule == "linear":
            betas = _linear_beta_schedule(self.num_steps)
        else:
            raise ValueError(f"Unsupported diffusion beta schedule: {beta_schedule}")

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def to(self, device):
        for name, value in vars(self).items():
            if torch.is_tensor(value):
                setattr(self, name, value.to(device))
        return self

    def _extract(self, values, timesteps, x_shape):
        out = values.gather(0, timesteps.long())
        return out.view(timesteps.shape[0], *((1,) * (len(x_shape) - 1)))

    def model_time(self, timesteps):
        return timesteps.float() / float(max(1, self.num_steps - 1))

    def q_sample(self, x0, timesteps, noise):
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod, timesteps, x0.shape)
        sqrt_omab = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x0.shape)
        return sqrt_ab * x0 + sqrt_omab * noise

    def predict_x0_from_eps(self, xt, timesteps, eps):
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod, timesteps, xt.shape)
        sqrt_omab = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, xt.shape)
        return (xt - sqrt_omab * eps) / torch.clamp(sqrt_ab, min=1e-8)

    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        sparse_cond,
        shape,
        image_tensor=None,
        intrinsics=None,
        sampling_steps=50,
        eta=0.0,
        clip_denoised=True,
    ):
        device = sparse_cond.device
        self.to(device)
        steps = max(1, int(sampling_steps))
        eta = float(eta)
        times = torch.linspace(self.num_steps - 1, 0, steps, device=device).long()
        x = torch.randn(shape, device=device, dtype=sparse_cond.dtype)

        sample_model = model
        if isinstance(model, torch.nn.DataParallel) and shape[0] < len(model.device_ids):
            sample_model = model.module

        for i, t_value in enumerate(times):
            t = torch.full((shape[0],), int(t_value.item()), device=device, dtype=torch.long)
            model_t = self.model_time(t)
            if image_tensor is not None and intrinsics is not None:
                eps = sample_model(x, sparse_cond, model_t, image_tensor=image_tensor, intrinsics=intrinsics)
            else:
                eps = sample_model(x, sparse_cond, model_t)
            x0 = self.predict_x0_from_eps(x, t, eps)
            if clip_denoised:
                x0 = x0.clamp(-1.0, 1.0)

            if i == len(times) - 1:
                x = x0
                continue

            prev_t_value = int(times[i + 1].item())
            alpha_bar_t = self._extract(self.alphas_cumprod, t, x.shape)
            prev_t = torch.full((shape[0],), prev_t_value, device=device, dtype=torch.long)
            alpha_bar_prev = self._extract(self.alphas_cumprod, prev_t, x.shape)

            sigma = eta * torch.sqrt(
                torch.clamp((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t), min=0.0)
                * torch.clamp(1.0 - (alpha_bar_t / alpha_bar_prev), min=0.0)
            )
            direction_scale = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma ** 2, min=0.0))
            noise = torch.randn_like(x) if eta > 0.0 else torch.zeros_like(x)
            x = torch.sqrt(alpha_bar_prev) * x0 + direction_scale * eps + sigma * noise

        return x.clamp(-1.0, 1.0) if clip_denoised else x

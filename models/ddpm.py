import torch
import torch.nn as nn
from tqdm import tqdm
class DDPM(nn.Module):
    def __init__(self, network, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cuda'):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.device = device
        self.network = network.to(device)
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    def forward(self, x_start):
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=self.device).long()
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise=noise)
        predicted_noise = self.network(x_noisy, t)
        loss = nn.functional.mse_loss(predicted_noise, noise)
        return loss
    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self._extract(torch.sqrt(1.0 / self.alphas), t, x.shape)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.network(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract(self.betas, t, x.shape) 
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    @torch.no_grad()
    def sample(self, image_size, batch_size=16, channels=3):
        shape = (batch_size, channels, image_size, image_size)
        img = torch.randn(shape, device=self.device)
        print("Generating images...")
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(img, t, i)
        return img
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

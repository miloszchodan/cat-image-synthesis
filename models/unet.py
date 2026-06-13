import torch
import torch.nn as nn
import math
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        if up:
            self.conv1 = nn.Conv2d(2 * in_channels, out_channels, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.transform = nn.Conv2d(out_channels, out_channels, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bnorm1 = nn.GroupNorm(8, out_channels)
        self.bnorm2 = nn.GroupNorm(8, out_channels)
        self.relu = nn.ReLU()
    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)
class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()
            size = x.shape[-1]
            x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
            x_ln = self.ln(x)
            attention_value, _ = self.mha(x_ln, x_ln, x_ln)
            attention_value = attention_value + x
            attention_value = self.ff_self(attention_value) + attention_value
            return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)
class UNet(nn.Module):
    def __init__(self, image_channels=3, down_channels=(64, 128, 256, 512), time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)
        self.downs = nn.ModuleList([
            Block(down_channels[0], down_channels[1], time_emb_dim), 
            Block(down_channels[1], down_channels[2], time_emb_dim), 
            Block(down_channels[2], down_channels[3], time_emb_dim), 
        ])
        self.attns_down = nn.ModuleList([
            nn.Identity(), 
            SelfAttention(down_channels[2]),
            SelfAttention(down_channels[3]),
        ])
        self.bot1 = nn.Conv2d(down_channels[3], down_channels[3], 3, padding=1)
        self.bot2 = SelfAttention(down_channels[3])
        self.bot3 = nn.Conv2d(down_channels[3], down_channels[3], 3, padding=1)
        self.ups = nn.ModuleList([
            Block(down_channels[3], down_channels[2], time_emb_dim, up=True), 
            Block(down_channels[2], down_channels[1], time_emb_dim, up=True), 
            Block(down_channels[1], down_channels[0], time_emb_dim, up=True), 
        ])
        self.attns_up = nn.ModuleList([
            SelfAttention(down_channels[2]),
            nn.Identity(), 
            nn.Identity(), 
        ])
        self.output = nn.Conv2d(down_channels[0], image_channels, 1)
    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        residuals = []
        for down, attn in zip(self.downs, self.attns_down):
            x = down(x, t)
            x = attn(x)
            residuals.append(x)
        x = self.bot1(x)
        x = self.bot2(x)
        x = self.bot3(x)
        for up, attn in zip(self.ups, self.attns_up):
            res_x = residuals.pop()
            x = torch.cat((x, res_x), dim=1)
            x = up(x, t)
            x = attn(x)
        return self.output(x)

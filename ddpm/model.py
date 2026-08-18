"""
DDPM U-Net, matching the architecture used by Ho et al. (2020) and reused by
SecMI (Duan et al. 2023) / PIA (Kong et al. 2023) / SimA, via w86763777/pytorch-ddpm:
  - Sinusoidal timestep embedding -> 2-layer MLP (Swish activation)
  - ResBlocks: GroupNorm -> Swish -> Conv3x3, with timestep embedding injected
    additively after the first conv; GroupNorm -> Swish -> Dropout -> Conv3x3;
    residual (1x1 conv if channel count changes)
  - Self-attention blocks at a chosen resolution (16x16 for CIFAR-10 at 32x32 input)
  - Standard encoder / bottleneck / decoder with skip connections (U-Net)

Default hyperparameters below match the standard CIFAR-10 DDPM config (Ho et al.
2020, replicated in pytorch-ddpm/config/CIFAR10.txt):
  base channels = 128, channel multipliers = [1, 2, 2, 2], 2 ResBlocks per stage,
  attention at 16x16 resolution, dropout = 0.1.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by a 2-layer MLP."""

    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb, freeze=True),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        return self.timembedding(t)


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        # named 'main' (not 'conv') to match pytorch-ddpm/SecMI/PIA's actual
        # state_dict keys, confirmed via checkpoint_loader's diagnose_checkpoint
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)

    def forward(self, x, temb):
        return self.main(x)


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)

    def forward(self, x, temb):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.main(x)


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1)
        self.proj = nn.Conv2d(in_ch, in_ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q, k, v = self.proj_q(h), self.proj_k(h), self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (C ** (-0.5))
        assert w.shape == (B, H * W, H * W)
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert h.shape == (B, H * W, C)
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        self.attn = AttnBlock(out_ch) if attn else nn.Identity()

    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)
        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class UNet(nn.Module):
    """Predicts epsilon (the noise) given a noised image x_t and timestep t."""

    def __init__(self, T=1000, ch=128, ch_mult=(1, 2, 2, 2), attn=(1,),
                 num_res_blocks=2, dropout=0.1, in_ch=3):
        super().__init__()
        assert all(i < len(ch_mult) for i in attn), "attn index out of bound"
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)

        self.head = nn.Conv2d(in_ch, ch, kernel_size=3, stride=1, padding=1)

        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channels for skip connections
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(
                    ResBlock(now_ch, out_ch, tdim, dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(
                    ResBlock(chs.pop() + now_ch, out_ch, tdim, dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, in_ch, 3, stride=1, padding=1),
        )

    def forward(self, x, t):
        """x: (B, in_ch, H, W); t: (B,) integer timesteps in [0, T)."""
        temb = self.time_embedding(t)

        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)

        for layer in self.middleblocks:
            h = layer(h, temb)

        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)

        h = self.tail(h)
        assert len(hs) == 0
        return h
# Implementation inspired by the V-JEPA paper (arXiv:2404.08471); all code original.
"""Vision Transformer backbone with 3D tubelet patch embedding for video."""

import torch
import torch.nn as nn


class PatchEmbed3D(nn.Module):
    """Embed (B, 3, T, H, W) video into (B, N, D) tokens via Conv3d tubelets."""

    def __init__(self, tubelet_t=2, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        self.tubelet_t = tubelet_t
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tubelet_t, patch_size, patch_size),
            stride=(tubelet_t, patch_size, patch_size),
        )

    def grid_size(self, T, H, W):
        assert T % self.tubelet_t == 0 and H % self.patch_size == 0 and W % self.patch_size == 0
        return T // self.tubelet_t, H // self.patch_size, W // self.patch_size

    def forward(self, x):
        # x: (B, 3, T, H, W)
        x = self.proj(x)  # (B, D, t, h, w)
        return x.flatten(2).transpose(1, 2)  # (B, N, D)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, d)
        q, k, v = qkv.unbind(0)
        x = nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MLP(nn.Module):
    def __init__(self, dim, ratio=4.0):
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """ViT encoder for video: patch embed -> (optional mask) -> pos embed -> blocks."""

    def __init__(self, img_size=224, num_frames=16, tubelet_t=2, patch_size=16,
                 depth=12, embed_dim=384, num_heads=6, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbed3D(tubelet_t, patch_size, 3, embed_dim)
        gt, gh, gw = self.patch_embed.grid_size(num_frames, img_size, img_size)
        self.grid = (gt, gh, gw)
        self.num_tokens = gt * gh * gw
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, video, visible_idx=None):
        """video: (B, 3, T, H, W). If visible_idx given (B, V), encode only visible tokens."""
        x = self.patch_embed(video)  # (B, N, D)
        if visible_idx is not None:
            x = torch.gather(x, 1, visible_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
            pos = torch.gather(
                self.pos_embed.expand(x.shape[0], -1, -1), 1,
                visible_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]),
            )
            x = x + pos
        else:
            x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

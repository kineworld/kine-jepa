# Implementation inspired by the V-JEPA paper (arXiv:2404.08471); all code original.
"""KineOne-WM-Latent: online encoder + EMA target encoder + transformer predictor.

The Python package name is retained for compatibility with historical
KINE-EXP-001 checkpoints.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit import VisionTransformer, Block


class Predictor(nn.Module):
    """Transform encoded visible tokens + learnable mask tokens into target space."""

    def __init__(self, num_tokens, depth=6, dim=384, num_heads=12):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList([Block(dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, visible_feats, vis_idx, mask_idx):
        B, V, D = visible_feats.shape
        M = mask_idx.shape[1]
        pos = self.pos_embed.expand(B, -1, -1)
        vis = visible_feats + torch.gather(pos, 1, vis_idx.unsqueeze(-1).expand(B, V, D))
        masks = self.mask_token.expand(B, M, D) + torch.gather(
            pos, 1, mask_idx.unsqueeze(-1).expand(B, M, D)
        )
        idx = torch.cat([vis_idx, mask_idx], dim=1)  # (B, N) scrambled order
        tokens = torch.cat([vis, masks], dim=1)
        order = torch.argsort(idx, dim=1)           # restore canonical token order
        tokens = torch.gather(tokens, 1, order.unsqueeze(-1).expand(B, -1, D))
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        return torch.gather(tokens, 1, mask_idx.unsqueeze(-1).expand(B, M, D))


class KineJEPA(nn.Module):
    def __init__(self, img_size=224, num_frames=16, tubelet_t=2, patch_size=16,
                 enc_depth=12, enc_dim=384, enc_heads=6,
                 pred_depth=6, pred_dim=384, pred_heads=12,
                 base_momentum=0.996, final_momentum=1.0):
        super().__init__()
        self.encoder = VisionTransformer(
            img_size=img_size, num_frames=num_frames, tubelet_t=tubelet_t,
            patch_size=patch_size, depth=enc_depth, embed_dim=enc_dim, num_heads=enc_heads,
        )
        self.target = copy.deepcopy(self.encoder)
        for p in self.target.parameters():
            p.requires_grad = False
        self.predictor = Predictor(
            self.encoder.num_tokens, depth=pred_depth, dim=pred_dim, num_heads=pred_heads,
        )
        self.base_momentum = base_momentum
        self.final_momentum = final_momentum

    @property
    def grid(self):
        return self.encoder.grid

    def momentum(self, step, total_steps):
        """Cosine ramp from base to final momentum."""
        frac = min(1.0, step / max(1, total_steps))
        return self.final_momentum - (self.final_momentum - self.base_momentum) * \
            (1 + math.cos(math.pi * frac)) / 2

    @torch.no_grad()
    def update_target(self, momentum):
        for pt, po in zip(self.target.parameters(), self.encoder.parameters()):
            pt.data.mul_(momentum).add_(po.data, alpha=1.0 - momentum)

    def forward(self, video, vis_idx, mask_idx):
        """video: (B, 3, T, H, W); returns loss and the effective mask ratio."""
        with torch.no_grad():
            target_feats = self.target(video)  # full sequence, no mask
            target = torch.gather(
                target_feats, 1,
                mask_idx.unsqueeze(-1).expand(-1, -1, target_feats.shape[-1]),
            )
            target = F.normalize(target, dim=-1)

        visible = self.encoder(video, visible_idx=vis_idx)
        pred = self.predictor(visible, vis_idx, mask_idx)
        loss = F.l1_loss(pred, target)
        return loss, mask_idx.shape[1] / (mask_idx.shape[1] + vis_idx.shape[1])

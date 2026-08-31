# Intervention conditioning for KINE-EXP-002. Original code.
"""Add a discrete do(x) token to visible features before the predictor.

Intervention ids:
  0 empty / no intervention
  1 remove_support
  2 break_contact
  3 random (used only by negative control B)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_INTERVENTIONS = 4


class InterventionHead(nn.Module):
    def __init__(self, dim: int, num_interventions: int = NUM_INTERVENTIONS):
        super().__init__()
        self.embed = nn.Embedding(num_interventions, dim)
        self.proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        nn.init.trunc_normal_(self.embed.weight, std=0.02)

    def forward(self, visible_feats: torch.Tensor, intervention_id: torch.Tensor) -> torch.Tensor:
        cond = self.embed(intervention_id).unsqueeze(1)
        return self.proj(visible_feats + cond)


class CausalKineJEPA(nn.Module):
    def __init__(self, base, train_last_n: int = 2):
        super().__init__()
        self.base = base
        for p in self.base.encoder.parameters():
            p.requires_grad = False
        for p in self.base.target.parameters():
            p.requires_grad = False
        dim = self.base.predictor.mask_token.shape[-1]
        self.head = InterventionHead(dim)
        blocks = list(self.base.predictor.blocks)
        freeze_until = max(0, len(blocks) - train_last_n)
        for i, blk in enumerate(blocks):
            requires = i >= freeze_until
            for p in blk.parameters():
                p.requires_grad = requires
        for p in self.base.predictor.norm.parameters():
            p.requires_grad = True

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, video, vis_idx, mask_idx, intervention_id):
        with torch.no_grad():
            target_feats = self.base.target(video)
            target = torch.gather(
                target_feats, 1,
                mask_idx.unsqueeze(-1).expand(-1, -1, target_feats.shape[-1]),
            )
            target = F.normalize(target, dim=-1)
            visible = self.base.encoder(video, visible_idx=vis_idx)
        visible = self.head(visible, intervention_id)
        pred = self.base.predictor(visible, vis_idx, mask_idx)
        loss = F.l1_loss(pred, target)
        return loss, mask_idx.shape[1] / (mask_idx.shape[1] + vis_idx.shape[1])

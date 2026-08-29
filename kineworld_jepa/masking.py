# Implementation inspired by the V-JEPA paper (arXiv:2404.08471); all code original.
"""Spatiotemporal multi-block masking for video tokens."""

import random

import torch


class MultiBlockMask3D:
    """Sample a union of random 3D blocks over a (t, h, w) token grid.

    Returns per-sample boolean masks; masked tokens are the prediction targets.
    """

    def __init__(self, grid, mask_ratio=0.75, min_block_scale=0.2, max_block_scale=1.0,
                 aspect_range=(0.75, 1.5), max_blocks=12):
        self.gt, self.gh, self.gw = grid
        self.num_tokens = self.gt * self.gh * self.gw
        self.mask_ratio = mask_ratio
        self.min_block_scale = min_block_scale
        self.max_block_scale = max_block_scale
        self.aspect_range = aspect_range
        self.max_blocks = max_blocks

    def _sample_block_dims(self):
        log_lo, log_hi = self.aspect_range
        # temporal vs spatial aspect
        ratio_t = random.uniform(log_lo, log_hi)
        ratio_s = random.uniform(log_lo, log_hi)
        # target block volume as a fraction of the whole grid
        scale = random.uniform(self.min_block_scale, self.max_block_scale)
        vol = self.num_tokens * scale * self.mask_ratio / self.max_blocks * 3.0
        bt = max(1, int(round((vol * ratio_t) ** (1 / 3))))
        bs = max(1, int(round((vol * ratio_s / ratio_t) ** (1 / 3))))
        bh = bw = bs
        bt = min(bt, self.gt)
        bh = min(bh, self.gh)
        bw = min(bw, self.gw)
        return bt, bh, bw

    def sample_one(self, mask_ratio=None):
        ratio = self.mask_ratio if mask_ratio is None else mask_ratio
        target = int(round(self.num_tokens * ratio))
        mask = torch.zeros(self.gt, self.gh, self.gw, dtype=torch.bool)
        for _ in range(self.max_blocks):
            if mask.sum().item() >= target:
                break
            bt, bh, bw = self._sample_block_dims()
            t0 = random.randint(0, self.gt - bt)
            h0 = random.randint(0, self.gh - bh)
            w0 = random.randint(0, self.gw - bw)
            mask[t0:t0 + bt, h0:h0 + bh, w0:w0 + bw] = True
        flat = mask.flatten()
        n_masked = int(flat.sum().item())
        if n_masked < target:  # top up with random tokens so the ratio is honored
            unmasked = (~flat).nonzero(as_tuple=True)[0].tolist()
            random.shuffle(unmasked)
            for i in unmasked[: target - n_masked]:
                flat[i] = True
        return flat

    def sample_batch(self, batch_size, mask_ratio=None, device="cpu"):
        masks = torch.stack([self.sample_one(mask_ratio) for _ in range(batch_size)])  # (B, N)
        mask_idx = []
        vis_idx = []
        for m in masks:
            mask_idx.append(m.nonzero(as_tuple=True)[0])
            vis_idx.append((~m).nonzero(as_tuple=True)[0])
        mask_idx = torch.stack(mask_idx).to(device)  # (B, M)
        vis_idx = torch.stack(vis_idx).to(device)    # (B, V)
        return vis_idx, mask_idx

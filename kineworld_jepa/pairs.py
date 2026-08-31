"""On-the-fly control/intervene video pairs for EXP-002. No OpenCV."""
from __future__ import annotations
import torch
from torch.utils.data import Dataset


def make_clip(falling: bool, frames=16, size=64, seed=0):
    g = torch.Generator().manual_seed(int(seed))
    video = 0.05 * torch.randn(3, frames, size, size, generator=g)
    bar_y = size // 2
    x0, w, h = size // 3, max(4, size // 8), max(4, size // 8)
    for t in range(frames):
        if falling and t >= frames // 2:
            y = min(size - h - 1, bar_y - h + int((t - frames // 2) * (size / max(frames, 1)) * 1.8))
            draw_bar = False
        else:
            y = max(0, bar_y - h)
            draw_bar = True
        video[:, t, y:y + h, x0:x0 + w] = 1.0
        if draw_bar:
            video[:, t, bar_y:min(size, bar_y + 3), size // 5: 4 * size // 5] = 0.7
    return video


class PairedInterventionDataset(Dataset):
    """Each item is (video, intervention_id). Even idx = control (id 0), odd = intervene (id 1)."""

    def __init__(self, n_pairs=64, num_frames=16, size=64):
        self.n_pairs = n_pairs
        self.num_frames = num_frames
        self.size = size

    def __len__(self):
        return self.n_pairs * 2

    def __getitem__(self, idx):
        pair = idx // 2
        falling = idx % 2 == 1
        video = make_clip(falling, self.num_frames, self.size, seed=pair)
        iid = 1 if falling else 0
        return video, torch.tensor(iid, dtype=torch.long)

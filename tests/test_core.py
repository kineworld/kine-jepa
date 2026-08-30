"""Unit tests for KINE-JEPA core modules. Run: python tests/test_core.py (no pytest needed)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from kineworld_jepa.train import cosine_schedule, mask_schedule
from kineworld_jepa.masking import MultiBlockMask3D
from kineworld_jepa.jepa import KineJEPA
from kineworld_jepa.dataset import SyntheticVideoDataset


def test_cosine_schedule():
    total, peak, mn, warm = 1000, 3e-4, 3e-6, 100
    assert cosine_schedule(0, total, peak, mn, warm) == 0.0
    assert abs(cosine_schedule(warm, total, peak, mn, warm) - peak) < 1e-9
    assert abs(cosine_schedule(total, total, peak, mn, warm) - mn) < 1e-9
    prev = peak
    for s in range(warm + 1, total + 1, 37):
        v = cosine_schedule(s, total, peak, mn, warm)
        assert v <= prev + 1e-12
        prev = v


def test_mask_schedule():
    assert abs(mask_schedule(0, 1000) - 0.9) < 1e-9
    assert abs(mask_schedule(1000, 1000) - 0.75) < 1e-9
    for step in range(0, 1001, 97):
        r = mask_schedule(step, 1000)
        assert 0.75 - 1e-9 <= r <= 0.9 + 1e-9


def test_mask_exact_count_and_partition():
    grid = (8, 14, 14)
    n = 8 * 14 * 14
    masker = MultiBlockMask3D(grid)
    for ratio in (0.75, 0.85, 0.9):
        vis, mask = masker.sample_batch(4, ratio)
        target = int(round(n * ratio))
        assert mask.shape == (4, target), (mask.shape, target)
        assert vis.shape == (4, n - target)
        for b in range(4):
            both = torch.cat([vis[b], mask[b]]).sort().values
            assert torch.equal(both, torch.arange(n))


def small_model():
    return KineJEPA(
        img_size=64, num_frames=4, tubelet_t=2, patch_size=16,
        enc_depth=1, enc_dim=64, enc_heads=4,
        pred_depth=1, pred_dim=64, pred_heads=4,
    )


def test_forward_loss_and_ratio():
    torch.manual_seed(0)
    model = small_model()
    video = torch.randn(2, 3, 4, 64, 64)
    masker = MultiBlockMask3D(model.grid)
    vis, mask = masker.sample_batch(2, 0.75)
    loss, ratio = model(video, vis, mask)
    assert torch.isfinite(loss)
    assert abs(ratio - 0.75) < 1e-6


def test_target_encoder_updates():
    torch.manual_seed(0)
    model = small_model()
    enc_p = next(iter(model.encoder.parameters()))
    tgt_p = next(iter(model.target.parameters()))
    before = tgt_p.detach().clone()
    with torch.no_grad():
        enc_p.add_(1.0)
    model.update_target(0.5)
    assert not torch.equal(before, tgt_p.detach())
    assert torch.allclose(tgt_p.detach(), before + 0.5, atol=1e-6)


def test_synthetic_dataset():
    ds = SyntheticVideoDataset(length=4, num_frames=4, size=64)
    x = ds[0]
    assert x.shape == (3, 4, 64, 64)
    assert torch.isfinite(x).all()


ALL = [
    test_cosine_schedule,
    test_mask_schedule,
    test_mask_exact_count_and_partition,
    test_forward_loss_and_ratio,
    test_target_encoder_updates,
    test_synthetic_dataset,
]

if __name__ == "__main__":
    for fn in ALL:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"all {len(ALL)} tests passed")

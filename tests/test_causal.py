import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from kineworld_jepa.causal import CausalKineJEPA, InterventionHead
from kineworld_jepa.jepa import KineJEPA
from kineworld_jepa.masking import MultiBlockMask3D

def small_base():
    return KineJEPA(img_size=64, num_frames=4, tubelet_t=2, patch_size=16, enc_depth=1, enc_dim=64, enc_heads=4, pred_depth=2, pred_dim=64, pred_heads=4)

def test_head_changes_features():
    head = InterventionHead(32)
    x = torch.randn(2, 5, 32)
    y0 = head(x, torch.zeros(2, dtype=torch.long))
    y1 = head(x, torch.ones(2, dtype=torch.long))
    assert y0.shape == x.shape
    assert not torch.allclose(y0, y1)

def test_encoder_frozen():
    wrap = CausalKineJEPA(small_base(), train_last_n=1)
    assert all(not p.requires_grad for p in wrap.base.encoder.parameters())
    assert any(p.requires_grad for p in wrap.head.parameters())
    last = list(wrap.base.predictor.blocks)[-1]
    first = list(wrap.base.predictor.blocks)[0]
    assert any(p.requires_grad for p in last.parameters())
    assert all(not p.requires_grad for p in first.parameters())

def test_causal_forward():
    torch.manual_seed(0)
    wrap = CausalKineJEPA(small_base(), train_last_n=1)
    video = torch.randn(2, 3, 4, 64, 64)
    masker = MultiBlockMask3D(wrap.base.grid)
    vis, mask = masker.sample_batch(2, 0.75)
    loss, ratio = wrap(video, vis, mask, torch.tensor([0, 1]))
    assert torch.isfinite(loss)
    loss.backward()
    assert sum(p.grad.abs().sum().item() for p in wrap.head.parameters() if p.grad is not None) > 0
    assert [p.grad for p in wrap.base.encoder.parameters() if p.grad is not None] == []

ALL = [test_head_changes_features, test_encoder_frozen, test_causal_forward]
if __name__ == "__main__":
    for fn in ALL:
        fn(); print("PASS", fn.__name__)
    print("all", len(ALL), "tests passed")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from kineworld_jepa.pairs import PairedInterventionDataset, make_clip

def test_shapes_and_labels():
    ds = PairedInterventionDataset(n_pairs=3, num_frames=8, size=32)
    assert len(ds) == 6
    v0, i0 = ds[0]
    v1, i1 = ds[1]
    assert v0.shape == (3, 8, 32, 32)
    assert int(i0) == 0 and int(i1) == 1
    assert not torch.allclose(v0, v1)

def test_same_seed_control_differs_from_fall():
    a = make_clip(False, 8, 32, seed=7)
    b = make_clip(True, 8, 32, seed=7)
    assert (a - b).abs().mean() > 0.01

if __name__ == "__main__":
    test_shapes_and_labels(); test_same_seed_control_differs_from_fall()
    print("PASS test_pairs")

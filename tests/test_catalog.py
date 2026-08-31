import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kineworld_jepa.catalog import CatalogDataset

def test_synthetic_catalog(tmp_path=None):
    from pathlib import Path as P
    p = P("/tmp/cat.json")
    p.write_text(json.dumps({"pairs": [
        {"id": "syn-0000", "source": "synthetic", "control": {"seed": 1}, "intervene": {"seed": 1}},
        {"id": "win-0000", "source": "missing.mp4", "pre": [0, 16], "post": [17, 33]},
    ]}))
    ds = CatalogDataset(p, video_dir="/tmp", num_frames=8, size=32)
    assert len(ds) == 4
    v, iid = ds[0]
    assert v.shape == (3, 8, 32, 32)
    assert int(iid) == 0
    v2, iid2 = ds[1]
    assert int(iid2) == 1
    # missing video falls back to synthetic, still returns a tensor
    v3, _ = ds[2]
    assert v3.shape == (3, 8, 32, 32)

if __name__ == "__main__":
    test_synthetic_catalog()
    print("PASS test_catalog")

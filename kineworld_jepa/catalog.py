"""Load a datapipe pairs.json catalog.

Synthetic rows are generated in-process. Event-window rows need a video dir
and OpenCV; missing files are skipped instead of crashing training.
"""
from __future__ import annotations
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .pairs import make_clip

VIDEO_EXT = (".mp4", ".mkv", ".webm")


def _read_window(path, start, end, size):
    import cv2
    cap = cv2.VideoCapture(str(path))
    frames = []
    for i in range(int(start), int(end)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        frame = cv2.resize(frame[y0:y0 + side, x0:x0 + side], (size, size))
        frames.append(torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0)
    cap.release()
    if not frames:
        return None
    return torch.stack(frames, dim=1)  # (3, T, H, W)


class CatalogDataset(Dataset):
    def __init__(self, catalog_path, video_dir=None, num_frames=16, size=64):
        payload = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
        self.rows = payload.get("pairs", payload if isinstance(payload, list) else [])
        self.video_dir = Path(video_dir) if video_dir else None
        self.num_frames = num_frames
        self.size = size
        self.items = []
        for row in self.rows:
            if row.get("source") == "synthetic" or "control" in row:
                self.items.append(("syn", row, 0))
                self.items.append(("syn", row, 1))
            elif "pre" in row and "post" in row:
                self.items.append(("win", row, 0))
                self.items.append(("win", row, 1))
        if not self.items:
            raise ValueError(f"no usable pairs in {catalog_path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        kind, row, falling = self.items[idx]
        if kind == "syn":
            seed = int(row.get("control", row.get("intervene", {})).get("seed", hash(row["id"]) % 10_000))
            video = make_clip(bool(falling), self.num_frames, self.size, seed=seed)
            return video, torch.tensor(1 if falling else 0, dtype=torch.long)
        path = None
        if self.video_dir is not None:
            cand = self.video_dir / row["source"]
            if cand.is_file():
                path = cand
        if path is None:
            video = make_clip(bool(falling), self.num_frames, self.size, seed=abs(hash(row["id"])) % 10_000)
            return video, torch.tensor(1 if falling else 0, dtype=torch.long)
        start, end = row["post" if falling else "pre"]
        video = _read_window(path, start, end, self.size)
        if video is None:
            video = make_clip(bool(falling), self.num_frames, self.size, seed=abs(hash(row["id"])) % 10_000)
        return video, torch.tensor(1 if falling else 0, dtype=torch.long)

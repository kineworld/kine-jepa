# Implementation inspired by the V-JEPA paper (arXiv:2404.08471); all code original.
"""Video dataset over kine-datapipe clip output, plus a synthetic set for smoke tests."""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class VideoClipDataset(Dataset):
    """Reads scene-cut clips (mp4) and yields uniformly sampled frame stacks."""

    def __init__(self, clips_dir, num_frames=16, size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        clips_dir = Path(clips_dir)
        self.clips = sorted(
            p for p in clips_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        ) if clips_dir.is_dir() else []
        if not self.clips:
            raise FileNotFoundError(f"no video clips found in {clips_dir}")
        self.num_frames = num_frames
        self.size = size
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        import cv2  # imported lazily so the module loads without OpenCV in tests

        for attempt in range(5):
            path = self.clips[idx]
            cap = cv2.VideoCapture(str(path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total >= self.num_frames:
                frames = self._sample_frames(cap, total)
                cap.release()
                if frames is not None:
                    return self._to_tensor(frames)
            cap.release()
            idx = random.randrange(len(self.clips))
        raise RuntimeError("could not decode a usable clip after retries")

    def _sample_frames(self, cap, total):
        import cv2

        positions = np.linspace(0, total - 1, self.num_frames).astype(int)
        frames = []
        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
            ok, frame = cap.read()
            if not ok:
                return None
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            side = min(h, w)
            y0, x0 = (h - side) // 2, (w - side) // 2
            frame = frame[y0:y0 + side, x0:x0 + side]
            frame = cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        return np.stack(frames)  # (T, H, W, 3) uint8

    def _to_tensor(self, frames):
        x = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0  # (3, T, H, W)
        return (x - self.mean) / self.std


class SyntheticVideoDataset(Dataset):
    """Moving-gradient noise videos; only for smoke-testing the training loop."""

    def __init__(self, length=64, num_frames=16, size=224):
        self.length = length
        self.num_frames = num_frames
        self.size = size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        t = torch.linspace(0, 2 * np.pi, self.num_frames)
        phase = torch.rand(1) * 2 * np.pi
        ramp = (torch.sin(t + phase) + 1) / 2  # (T,)
        x = ramp.view(1, self.num_frames, 1, 1).expand(3, -1, self.size, self.size)
        noise = torch.rand(3, self.num_frames, self.size, self.size) * 0.1
        return x + noise - 0.5

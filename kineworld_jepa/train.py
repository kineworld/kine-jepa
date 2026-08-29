# Implementation inspired by the V-JEPA paper (arXiv:2404.08471); all code original.
"""Single-GPU training loop for KINE-JEPA with jsonl experiment logging."""

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import VideoClipDataset, SyntheticVideoDataset
from .jepa import KineJEPA
from .masking import MultiBlockMask3D


def cosine_schedule(step, total, start, end, warmup=0):
    if step < warmup:
        return start + (end - start) * step / max(1, warmup)
    frac = min(1.0, (step - warmup) / max(1, total - warmup))
    return end + (start - end) * (1 + math.cos(math.pi * frac)) / 2


def mask_schedule(step, total, start=0.9, end=0.75):
    frac = min(1.0, step / max(1, total))
    return end + (start - end) * (1 + math.cos(math.pi * frac)) / 2


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default=None, help="kine-datapipe clips dir")
    ap.add_argument("--smoke", action="store_true", help="synthetic data, few steps")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-6)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--enc-depth", type=int, default=12)
    ap.add_argument("--pred-depth", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--exp-name", type=str, default="KINE-EXP-001")
    ap.add_argument("--resume", type=str, default=None)
    return ap.parse_args()


def main():
    args = build_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(__file__).resolve().parent.parent / "experiments" / args.exp_name / f"run-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    (run_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    model = KineJEPA(
        img_size=args.img_size, num_frames=args.num_frames,
        enc_depth=args.enc_depth, pred_depth=args.pred_depth,
    ).to(device)
    masker = MultiBlockMask3D(model.grid)

    if args.smoke or not args.data_dir:
        dataset = SyntheticVideoDataset(num_frames=args.num_frames, size=args.img_size)
        print(f"[data] synthetic set ({len(dataset)} samples)")
    else:
        dataset = VideoClipDataset(args.data_dir, num_frames=args.num_frames, size=args.img_size)
        print(f"[data] {len(dataset)} clips from {args.data_dir}")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, persistent_workers=args.workers > 0,
    )
    data_iter = iter(loader)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        print(f"[resume] step {start_step} from {args.resume}")

    n_params = sum(p.numel() for p in model.encoder.parameters()) / 1e6
    print(f"[model] encoder ViT-S {n_params:.1f}M params | device={device} | amp_bf16={use_amp}")

    metrics_f = open(metrics_path, "a", encoding="utf-8")
    t0 = time.time()
    log_acc, log_n = 0.0, 0

    for step in range(start_step, args.steps):
        try:
            video = next(data_iter)[0]
        except StopIteration:
            data_iter = iter(loader)
            video = next(data_iter)[0]
        video = video.to(device, non_blocking=True)

        ratio = mask_schedule(step, args.steps)
        vis_idx, mask_idx = masker.sample_batch(video.shape[0], ratio, device=device)

        lr = cosine_schedule(step, args.steps, args.lr, args.min_lr, args.warmup_steps)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss, eff_ratio = model(video, vis_idx, mask_idx)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        model.update_target(model.momentum(step, args.steps))

        log_acc += loss.item()
        log_n += 1
        if (step + 1) % args.log_every == 0 or step == start_step:
            mem = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else 0
            rec = {
                "step": step + 1,
                "loss": round(log_acc / log_n, 6),
                "lr": lr,
                "mask_ratio": round(float(eff_ratio), 4),
                "ema_momentum": round(model.momentum(step, args.steps), 6),
                "peak_gpu_mem_mb": round(mem, 1),
                "wall_s": round(time.time() - t0, 1),
            }
            metrics_f.write(json.dumps(rec) + "\n")
            metrics_f.flush()
            print(f"step {rec['step']:>6} | loss {rec['loss']:.4f} | mask {rec['mask_ratio']:.2f} "
                  f"| lr {lr:.2e} | mem {mem:.0f}MB")
            log_acc, log_n = 0.0, 0

        if (step + 1) % args.ckpt_every == 0:
            ckpt_path = run_dir / f"ckpt-step{step + 1}.pt"
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "step": step, "config": vars(args)}, ckpt_path)
            print(f"[ckpt] {ckpt_path.name}")

    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "step": args.steps - 1, "config": vars(args)}, run_dir / "ckpt-final.pt")
    (run_dir / "summary.md").write_text(
        f"# {args.exp_name} run-{stamp}\n\n"
        f"- steps: {args.steps} | wall: {time.time() - t0:.0f}s\n"
        f"- data: {'synthetic' if args.smoke or not args.data_dir else args.data_dir}\n"
        f"- metrics: metrics.jsonl\n",
        encoding="utf-8",
    )
    metrics_f.close()
    print(f"[done] run dir: {run_dir}")


if __name__ == "__main__":
    main()

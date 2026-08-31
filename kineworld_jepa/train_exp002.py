"""KINE-EXP-002 trainer: freeze encoder, train on paired do() clips.

  python -m kineworld_jepa.train_exp002 --tiny --steps 8 --batch-size 2
  python -m kineworld_jepa.train_exp002 --resume PATH --arm C --steps 2000
"""
from __future__ import annotations
import argparse, json, time
from datetime import datetime
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .causal import CausalKineJEPA, NUM_INTERVENTIONS
from .jepa import KineJEPA
from .masking import MultiBlockMask3D
from .pairs import PairedInterventionDataset
from .train import cosine_schedule, mask_schedule

def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--arm", choices=["A", "B", "C"], default="C")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument("--n-pairs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--enc-depth", type=int, default=12)
    ap.add_argument("--pred-depth", type=int, default=6)
    ap.add_argument("--log-every", type=int, default=20)
    return ap.parse_args()

def relabel(arm, iid):
    if arm == "A":
        return torch.zeros_like(iid)
    if arm == "B":
        return torch.randint(0, NUM_INTERVENTIONS, iid.shape, device=iid.device)
    return iid

def main():
    args = build_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(__file__).resolve().parent.parent / "experiments" / "KINE-EXP-002" / f"{args.arm}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    tiny = args.tiny or (args.smoke and device.type == "cpu")
    if tiny:
        frames, size = 4, 64
        base = KineJEPA(img_size=size, num_frames=frames, tubelet_t=2, patch_size=16,
                        enc_depth=1, enc_dim=64, enc_heads=4, pred_depth=2, pred_dim=64, pred_heads=4)
    else:
        frames, size = args.num_frames, args.img_size
        base = KineJEPA(img_size=size, num_frames=frames,
                        enc_depth=args.enc_depth, pred_depth=args.pred_depth)
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        base.load_state_dict(ckpt["model"], strict=False)
        print(f"[exp002] loaded {args.resume}")

    model = CausalKineJEPA(base).to(device)
    masker = MultiBlockMask3D(model.base.grid)
    ds = PairedInterventionDataset(n_pairs=args.n_pairs, num_frames=frames, size=size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    data_iter = iter(loader)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr)
    print(f"[exp002] arm={args.arm} pairs={args.n_pairs} trainable={sum(p.numel() for p in model.trainable_parameters())} device={device}")

    metrics = open(run_dir / "metrics.jsonl", "a", encoding="utf-8")
    t0 = time.time(); acc = 0.0; n = 0
    for step in range(args.steps):
        try:
            video, iid = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            video, iid = next(data_iter)
        video = video.to(device)
        iid = relabel(args.arm, iid.to(device))
        vis, mask = masker.sample_batch(video.shape[0], mask_schedule(step, args.steps), device=device)
        for g in opt.param_groups:
            g["lr"] = cosine_schedule(step, args.steps, args.lr, args.lr * 0.05, warmup=20)
        opt.zero_grad(set_to_none=True)
        loss, ratio = model(video, vis, mask, iid)
        loss.backward(); opt.step()
        acc += float(loss.item()); n += 1
        if (step + 1) % args.log_every == 0 or step == 0:
            rec = {"step": step + 1, "loss": round(acc / max(n, 1), 6), "arm": args.arm,
                   "mask_ratio": round(float(ratio), 4), "wall_s": round(time.time() - t0, 1)}
            metrics.write(json.dumps(rec) + "\n"); metrics.flush()
            print(f"step {rec['step']:>5} | {args.arm} | loss {rec['loss']:.4f}")
            acc, n = 0.0, 0
    torch.save({"model": model.state_dict(), "arm": args.arm, "step": args.steps, "config": vars(args)}, run_dir / "ckpt-final.pt")
    metrics.close()
    print(f"[done] {run_dir}")

if __name__ == "__main__":
    main()

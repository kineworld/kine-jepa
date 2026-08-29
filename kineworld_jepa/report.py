# Implementation inspired by the V-JEPA paper (arXiv:2404.08471); all code original.
"""Generate the KINE-EXP-001 technical report + loss curves from a run's metrics.jsonl."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(run_dir: Path):
    recs = []
    with open(run_dir / "metrics.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return sorted(recs, key=lambda r: r["step"])


def plot_curves(recs, run_dir: Path):
    steps = [r["step"] for r in recs]
    loss = [r["loss"] for r in recs]
    mask = [r["mask_ratio"] for r in recs]
    lr = [r["lr"] for r in recs]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(steps, loss, color="#38d9a9", lw=1.5)
    axes[0].set_yscale("log")
    axes[0].set_title("L1 prediction loss (log scale)")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, mask, color="#4dabf7", lw=1.5)
    axes[1].set_title("Encoder mask ratio (annealed)")
    axes[1].set_xlabel("step"); axes[1].set_ylabel("mask ratio")
    axes[1].grid(alpha=0.3)

    axes[2].plot(steps, lr, color="#f7b84d", lw=1.5)
    axes[2].set_title("Learning rate (cosine, 500-step warmup)")
    axes[2].set_xlabel("step"); axes[2].set_ylabel("lr")
    axes[2].grid(alpha=0.3)

    fig.suptitle("KINE-EXP-001 · single RTX 5070 Ti · bf16", fontsize=12)
    fig.tight_layout()
    out = run_dir / "loss_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def write_report(recs, run_dir: Path, png: Path):
    first, last = recs[0], recs[-1]
    wall = last.get("wall_s", 0)
    steps_done = last["step"]
    mem = max(r.get("peak_gpu_mem_mb", 0) for r in recs)
    sps = steps_done / wall if wall else 0

    lines = [
        "# KINE-EXP-001 技术报告（自动生成）",
        "",
        "> 单卡世界模型预训练 · 由 `report.py` 从 `metrics.jsonl` 生成。",
        "",
        "## 关键数字",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 已完成步数 | {steps_done} |",
        f"| 损失（首 → 末） | {first['loss']:.4f} → {last['loss']:.4f} |",
        f"| 峰值显存 | {mem:.0f} MB |",
        f"| 吞吐 | {sps:.2f} 步/秒 |",
        f"| 用时 | {wall/3600:.2f} 小时 |",
        f"| 末步学习率 | {last['lr']:.2e} |",
        f"| 末步掩码比例 | {last['mask_ratio']:.3f} |",
        "",
        "## 损失曲线",
        "",
        f"![loss curve]({png.name})",
        "",
        "## 复现",
        "",
        "```bash",
        ".venv/Scripts/python -m kineworld_jepa.train --data-dir ../kine-datapipe/data/clips \\",
        "    --steps 25000 --batch-size 8 --seed 42",
        "```",
        "",
        "硬件：单张 NVIDIA GeForce RTX 5070 Ti Laptop GPU（12GB）。",
        "实现为 clean-room（仅借鉴 JEPA 论文思想，未复制其代码或权重）。",
    ]
    out = run_dir / "REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, default=None,
                    help="specific run dir; defaults to the latest under experiments/KINE-EXP-001")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent.parent / "experiments" / "KINE-EXP-001"
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        runs = sorted(d for d in base.iterdir() if d.is_dir() and (d / "metrics.jsonl").exists())
        if not runs:
            raise SystemExit("no run with metrics.jsonl found")
        run_dir = runs[-1]

    recs = load_metrics(run_dir)
    if len(recs) < 2:
        raise SystemExit(f"too few records in {run_dir}/metrics.jsonl")
    png = plot_curves(recs, run_dir)
    report = write_report(recs, run_dir, png)
    print(f"run: {run_dir.name}")
    print(f"  steps: {recs[-1]['step']}  loss: {recs[0]['loss']:.4f} -> {recs[-1]['loss']:.4f}")
    print(f"  curve: {png}")
    print(f"  report: {report}")


if __name__ == "__main__":
    main()

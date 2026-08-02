"""
plot_lora_loss.py — LoRA Training Loss 曲線可視化
輸出：lora_training_loss.png（適合放入簡報）
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path

# Register Chinese font
_font_candidates = [
    r"C:\Windows\Fonts\msjh.ttc",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simsun.ttc",
]
for _fp in _font_candidates:
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
        _prop = fm.FontProperties(fname=_fp)
        matplotlib.rcParams["font.family"] = _prop.get_name()
        break

PROJECT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT / "docs" / "generated" / "training"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = str(OUTPUT_DIR / "lora_training_loss.png")

MODELS = {
    "human_8692 (Male Mask LoRA)": os.path.join(
        PROJECT, r"models\ethan_mask_lora\logs\text2image-fine-tune"
    ),
    "human_5805 (Female Mask LoRA)": os.path.join(
        PROJECT, r"models\yourname_mask_lora\logs\text2image-fine-tune"
    ),
}

COLORS = ["#2563eb", "#dc2626"]   # blue / red


def load_loss(log_dir, smooth_window=30, max_points=800):
    ea = event_accumulator.EventAccumulator(log_dir, size_guidance={"scalars": 0})
    ea.Reload()
    events = ea.Scalars("train_loss")
    steps  = np.array([e.step  for e in events], dtype=float)
    losses = np.array([e.value for e in events], dtype=float)

    # 若資料點太多先均勻取樣，再做滑動平均
    if len(steps) > max_points:
        idx    = np.round(np.linspace(0, len(steps) - 1, max_points)).astype(int)
        steps  = steps[idx]
        losses = losses[idx]

    # 滑動平均平滑曲線
    kernel  = np.ones(smooth_window) / smooth_window
    padded  = np.pad(losses, (smooth_window // 2, smooth_window // 2), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")[: len(losses)]

    return steps, losses, smoothed


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor("#ffffff")
fig.suptitle("LoRA Training Loss Curves", fontsize=15, fontweight="bold",
             color="#1e293b", y=1.02)

for ax, (name, log_dir), color in zip(axes, MODELS.items(), COLORS):
    steps, raw, smooth = load_loss(log_dir)

    # 原始曲線（半透明）
    ax.plot(steps, raw, color=color, alpha=0.18, linewidth=0.8)
    # 平滑曲線
    ax.plot(steps, smooth, color=color, linewidth=2.2, label="Train Loss (smoothed)")

    # 標記最終 loss
    final_loss = smooth[-1]
    ax.scatter(steps[-1], final_loss, color=color, s=60, zorder=5)
    ax.annotate(f"Final: {final_loss:.4f}",
                xy=(steps[-1], final_loss),
                xytext=(-60, 12), textcoords="offset points",
                fontsize=9, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2))

    # 標記起始 loss
    init_loss = smooth[0]
    ax.scatter(steps[0], init_loss, color="#64748b", s=40, zorder=5)
    ax.annotate(f"Init: {init_loss:.4f}",
                xy=(steps[0], init_loss),
                xytext=(10, 10), textcoords="offset points",
                fontsize=9, color="#64748b")

    ax.set_title(name, fontsize=12, fontweight="bold", color="#1e293b", pad=10)
    ax.set_xlabel("Training Steps", fontsize=10, color="#475569")
    ax.set_ylabel("Loss", fontsize=10, color="#475569")
    ax.set_facecolor("#f8fafc")
    ax.grid(True, color="#e2e8f0", linewidth=0.8, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cbd5e1")
    ax.tick_params(colors="#64748b", labelsize=9)
    ax.set_ylim(bottom=0)

    # 降幅標註
    drop_pct = (init_loss - final_loss) / init_loss * 100
    ax.text(0.97, 0.97, f"Loss -{drop_pct:.1f}%",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=10, fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=color, alpha=0.9))

plt.tight_layout(pad=2.0)
plt.savefig(OUTPUT, dpi=180, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print(f"[DONE] Saved: {OUTPUT}")

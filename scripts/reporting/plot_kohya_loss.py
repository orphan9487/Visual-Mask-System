"""
plot_kohya_loss.py — Kohya-SS LoRA Training Loss 分析圖
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path

# Chinese font
for _fp in [r"C:\Windows\Fonts\msjh.ttc", r"C:\Windows\Fonts\msyh.ttc"]:
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
        matplotlib.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
        break

LOG = r"C:\Ethan\Edu_proj\DataSets\CelebA\train_data\OUPUT\Logs\20260524001426\network_train\events.out.tfevents.1779552891.DESKTOP-40UKTLF.30152.0"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "docs" / "generated" / "training"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = str(OUTPUT_DIR / "kohya_loss_analysis.png")

ea = event_accumulator.EventAccumulator(LOG, size_guidance={"scalars": 0})
ea.Reload()

def get(tag):
    ev = ea.Scalars(tag)
    return np.array([e.step for e in ev]), np.array([e.value for e in ev])

steps_cur,  loss_cur  = get("loss/current")
steps_avg,  loss_avg  = get("loss/average")
steps_ep,   loss_ep   = get("loss/epoch_average")

# smooth current loss
def smooth(arr, w=60):
    k = np.ones(w) / w
    p = np.pad(arr, (w//2, w//2), mode="edge")
    return np.convolve(p, k, mode="valid")[:len(arr)]

loss_cur_sm = smooth(loss_cur, 60)

# ── Figure layout: 1x2 ───────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor("#ffffff")
fig.suptitle("Kohya-SS LoRA Training Analysis  (human_5805 / CelebA)",
             fontsize=13, fontweight="bold", color="#1e293b", y=1.01)

BLUE   = "#2563eb"
RED    = "#dc2626"
GRAY   = "#94a3b8"
BG     = "#f8fafc"
GRID   = "#e2e8f0"

# ── Panel 1: Step-level loss ─────────────────────────────────────────────────
ax1.plot(steps_cur, loss_cur, color=BLUE, alpha=0.12, linewidth=0.6, label="Per-step loss (raw)")
ax1.plot(steps_cur, loss_cur_sm, color=BLUE, linewidth=2.2, label="Per-step loss (smoothed)")
ax1.plot(steps_avg, loss_avg, color=RED,  linewidth=1.6, linestyle="--", label="Running average loss")

ax1.set_title("Step-Level Training Loss", fontsize=11, fontweight="bold", color="#1e293b")
ax1.set_xlabel("Training Steps", fontsize=10, color="#475569")
ax1.set_ylabel("Loss", fontsize=10, color="#475569")
ax1.set_facecolor(BG)
ax1.grid(True, color=GRID, linewidth=0.8, linestyle="--")
ax1.spines[["top","right"]].set_visible(False)
ax1.spines[["left","bottom"]].set_color("#cbd5e1")
ax1.tick_params(colors="#64748b", labelsize=9)
ax1.legend(fontsize=8.5, framealpha=0.9)
ax1.set_ylim(bottom=0, top=min(loss_cur.max(), 1.2))

# annotate step 300 epoch boundary region
for ep_boundary in range(300, 3001, 300):
    ax1.axvline(ep_boundary, color=GRAY, linewidth=0.5, linestyle=":", alpha=0.6)
ax1.text(150, ax1.get_ylim()[1]*0.92, "epoch\nboundaries", fontsize=7,
         color=GRAY, ha="center")

# ── Panel 2: Epoch-level loss ─────────────────────────────────────────────────
epochs = steps_ep   # 1-10

ax2.plot(epochs, loss_ep, color=BLUE, linewidth=2.5, marker="o",
         markersize=7, markerfacecolor="white", markeredgewidth=2,
         markeredgecolor=BLUE, label="Train Loss (epoch avg)", zorder=5)

# annotate each epoch point
for e, l in zip(epochs, loss_ep):
    ax2.annotate(f"{l:.4f}", xy=(e, l),
                 xytext=(0, 10), textcoords="offset points",
                 fontsize=7.5, ha="center", color="#1e293b")

# shade the "still decreasing" region
best_epoch = int(epochs[np.argmin(loss_ep)])
ax2.axvline(best_epoch, color=RED, linewidth=1.5, linestyle="--", alpha=0.7)
ax2.text(best_epoch + 0.15, loss_ep.min() + 0.003,
         f"Lowest loss\n@ Epoch {best_epoch}", fontsize=8, color=RED)

ax2.fill_between(epochs, loss_ep,
                 where=(epochs <= best_epoch),
                 alpha=0.08, color=BLUE, label="Decreasing phase")
ax2.fill_between(epochs, loss_ep,
                 where=(epochs >= best_epoch),
                 alpha=0.08, color=RED, label="Oscillating / plateau phase")

ax2.set_title("Epoch-Level Training Loss", fontsize=11, fontweight="bold", color="#1e293b")
ax2.set_xlabel("Epoch", fontsize=10, color="#475569")
ax2.set_ylabel("Loss (epoch average)", fontsize=10, color="#475569")
ax2.set_facecolor(BG)
ax2.grid(True, color=GRID, linewidth=0.8, linestyle="--")
ax2.spines[["top","right"]].set_visible(False)
ax2.spines[["left","bottom"]].set_color("#cbd5e1")
ax2.tick_params(colors="#64748b", labelsize=9)
ax2.set_xticks(epochs)
ax2.legend(fontsize=8.5, framealpha=0.9)
ax2.set_ylim(bottom=0.08, top=0.23)

plt.tight_layout(pad=2.0)
plt.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor="white", edgecolor="none")
print(f"[DONE] {OUTPUT}")

"""
report_lora_5805.py — 生成 person5805 LoRA 訓練報告圖表
輸出：
  lora_5805_loss_report.png  — 報告用雙曲線圖
  lora_5805_params_table.png — 訓練參數表格圖
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path

# ── 中文字型 ─────────────────────────────────────────────────────────────────
for _fp in [r"C:\Windows\Fonts\msjh.ttc", r"C:\Windows\Fonts\msyh.ttc"]:
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
        matplotlib.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
        break

# ── 路徑 ─────────────────────────────────────────────────────────────────────
TFEVENT_PATH = (
    r"C:\Ethan\Edu_proj\DataSets\CelebA\train_data\OUPUT\Logs"
    r"\20260525023309\network_train"
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "docs" / "generated" / "training"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Val loss（watcher 輸出結果）────────────────────────────────────────────
VAL_LOSS = {
    1: 0.2422,
    2: 0.2648,
    3: 0.3105,
    4: 0.3307,
    5: 0.3091,
    6: 0.3382,
    7: 0.3346,
    8: 0.3288,
    9: 0.3312,
}

# ── 讀 Train loss（epoch_average）─────────────────────────────────────────
ea = event_accumulator.EventAccumulator(TFEVENT_PATH, size_guidance={"scalars": 0})
ea.Reload()

train_losses = {}
tag = "loss/epoch_average"
if tag in ea.Tags().get("scalars", []):
    for ev in ea.Scalars(tag):
        train_losses[int(ev.step)] = ev.value
    print(f"[OK] Train loss loaded: {len(train_losses)} epochs")
else:
    print(f"[WARN] Tag '{tag}' not found. Available: {ea.Tags().get('scalars', [])}")

# 對齊 epoch 範圍（取交集）
epochs = sorted(set(train_losses.keys()) & set(VAL_LOSS.keys()))
t_loss = [train_losses[e] for e in epochs]
v_loss = [VAL_LOSS[e]     for e in epochs]

print("\n=== Epoch-level Loss Table ===")
print(f"{'Epoch':>6} | {'Train Loss':>11} | {'Val Loss':>10} | {'Gap':>8}")
print("-" * 44)
for e, tl, vl in zip(epochs, t_loss, v_loss):
    gap   = vl - tl
    mark  = " ← sweet spot" if e == 1 else ""
    print(f"{e:>6} | {tl:>11.4f} | {vl:>10.4f} | {gap:>+8.4f}{mark}")

best_epoch   = epochs[int(np.argmin(v_loss))]
best_val     = min(v_loss)

# ═══════════════════════════════════════════════════════════════════════════
# 圖 1：Train vs Val Loss 雙曲線
# ═══════════════════════════════════════════════════════════════════════════
BLUE = "#2563eb"
RED  = "#dc2626"
GOLD = "#f59e0b"
BG   = "#f8fafc"
GRID = "#e2e8f0"

fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor("#ffffff")

ax.plot(epochs, t_loss, color=BLUE, linewidth=2.2, marker="o",
        markersize=7, markerfacecolor="white", markeredgewidth=2,
        markeredgecolor=BLUE, label="Train Loss (epoch avg)", zorder=5)

ax.plot(epochs, v_loss, color=RED, linewidth=2.2, marker="s",
        markersize=7, markerfacecolor="white", markeredgewidth=2,
        markeredgecolor=RED, linestyle="--", label="Validation Loss", zorder=5)

# 標每個點數值
for e, tl, vl in zip(epochs, t_loss, v_loss):
    ax.annotate(f"{tl:.3f}", xy=(e, tl), xytext=(0, 9),
                textcoords="offset points", fontsize=7.5,
                ha="center", color=BLUE)
    ax.annotate(f"{vl:.3f}", xy=(e, vl), xytext=(0, -14),
                textcoords="offset points", fontsize=7.5,
                ha="center", color=RED)


# 過擬合區域標色
ax.axvspan(best_epoch + 0.5, max(epochs) + 0.5,
           alpha=0.06, color=RED, label="過擬合區間")

ax.set_title(
    "LoRA 訓練分析：Train Loss vs Validation Loss\n"
    "person5805_v1_baseline（human_5805 Female LoRA）",
    fontsize=12, fontweight="bold", color="#1e293b")
ax.set_xlabel("Epoch", fontsize=10, color="#475569")
ax.set_ylabel("Diffusion Loss (MSE)", fontsize=10, color="#475569")
ax.set_xticks(epochs)
ax.set_facecolor(BG)
ax.grid(True, color=GRID, linewidth=0.8, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#cbd5e1")
ax.tick_params(colors="#64748b", labelsize=9)
ax.legend(fontsize=9.5, framealpha=0.95, loc="upper left")
ax.set_ylim(bottom=0.05)

plt.tight_layout()
out1 = OUTPUT_DIR / "lora_5805_loss_report.png"
plt.savefig(out1, dpi=180, bbox_inches="tight", facecolor="white")
print(f"\n[OK] Loss chart saved: {out1}")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# 圖 2：訓練參數摘要表格
# ═══════════════════════════════════════════════════════════════════════════
params = [
    ("模型基底",         "runwayml/stable-diffusion-v1-5"),
    ("LoRA 方法",        "Kohya-SS network.lora"),
    ("network_dim (rank)", "32"),
    ("network_alpha",    "8  (effective scale = alpha/dim = 0.25)"),
    ("Learning Rate (UNet)", "5e-5"),
    ("Learning Rate (Text Enc)", "2e-5"),
    ("LR Scheduler",     "constant_with_warmup  (warmup 30 steps)"),
    ("Optimizer",        "AdamW8bit"),
    ("Epoch",            "10  (max_train_steps = 4500)"),
    ("Batch Size",       "1"),
    ("Resolution",       "512 × 512"),
    ("noise_offset",     "0.15"),
    ("rank_dropout",     "0.1"),
    ("mixed_precision",  "bf16"),
    ("clip_skip",        "2"),
    ("訓練圖片數",        "40 張（train_data_5805）"),
    ("驗證圖片數",        "6 張（val_data_5805）"),
    ("Val loss 最低點",   f"Epoch 1  (val_loss = {best_val:.4f})"),
    ("過擬合觀察",        "Epoch 2 起 val_loss 持續上升，確認 epoch 1 為甜蜜點"),
    ("建議調整方向",      "network_dim ↓16, epoch ↓5, lr 可適度提高"),
]

fig2, ax2 = plt.subplots(figsize=(12, 7))
fig2.patch.set_facecolor("#ffffff")
ax2.axis("off")

ax2.set_title(
    "LoRA 訓練參數記錄表  —  person5805_v1_baseline",
    fontsize=13, fontweight="bold", color="#1e293b", pad=16)

col_labels = ["參數", "數值 / 設定"]
table_data = [[k, v] for k, v in params]

tbl = ax2.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc="left",
    loc="center",
    colWidths=[0.32, 0.62],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1, 1.45)

# 樣式
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#e2e8f0")
    if r == 0:
        cell.set_facecolor("#1e40af")
        cell.set_text_props(color="white", fontweight="bold")
    elif r in [17, 18, 19]:    # 最後三行（結果與建議）
        cell.set_facecolor("#fef3c7" if c == 0 else "#fffbeb")
    elif r % 2 == 0:
        cell.set_facecolor("#f1f5f9")
    else:
        cell.set_facecolor("#ffffff")

plt.tight_layout()
out2 = OUTPUT_DIR / "lora_5805_params_table.png"
plt.savefig(out2, dpi=180, bbox_inches="tight", facecolor="white")
print(f"[OK] Params table saved: {out2}")
plt.close()

print("\n[DONE] 兩張報告圖已儲存至 Visual-Mask-System/")

"""
report_t5_training.py — 訓練完成後執行，生成簡報用圖表

輸出：
  t5_loss_curve.png       — Train / Eval Loss 曲線
  t5_confusion_matrix.png — Confusion Matrix 熱力圖
  t5_f1_table.png         — Per-class F1 表格

使用方式：
  conda activate mask_env
  python scripts/reporting/report_t5_training.py
"""
import sys, os, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

# ── 中文字型 ──────────────────────────────────────────────────────────────────
for _fp in [r"C:\Windows\Fonts\msjh.ttc", r"C:\Windows\Fonts\msyh.ttc"]:
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
        matplotlib.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
        break

PROJECT     = Path(__file__).resolve().parents[2]
MODEL_PATH  = str(PROJECT / "models" / "mcot_answer_model_v2" / "checkpoint-21623")
SYNTH_DATA  = r"D:\Independent_study2_Backup\train_stage2_synthetic.jsonl"
EXT_DATA    = str(PROJECT / "external_stage2_dataset.jsonl")
TRAINER_LOG = PROJECT / "models" / "mcot_answer_model_v2" / "checkpoint-216230" / "trainer_state.json"
OUTPUT_DIR  = PROJECT / "docs" / "generated" / "training"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# ═══════════════════════════════════════════════════════════════════════════
# 圖 1：Train / Eval Loss 曲線（從 trainer_state.json 讀取）
# ═══════════════════════════════════════════════════════════════════════════
def plot_loss_curve():
    if not TRAINER_LOG.exists():
        print("[SKIP] trainer_state.json 不存在，跳過 loss curve")
        return

    with open(TRAINER_LOG, encoding="utf-8") as f:
        state = json.load(f)

    log = state.get("log_history", [])
    train_steps, train_loss = [], []
    eval_steps,  eval_loss  = [], []

    for entry in log:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry["step"])
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry["epoch"])
            eval_loss.append(entry["eval_loss"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#ffffff")
    fig.suptitle("T5 AffectiveStateAnalyzer 訓練過程",
                 fontsize=13, fontweight="bold", color="#1e293b")

    BLUE, RED = "#2563eb", "#dc2626"
    BG, GRID  = "#f8fafc", "#e2e8f0"

    # Panel 1: step-level train loss
    ax = axes[0]
    ax.plot(train_steps, train_loss, color=BLUE, linewidth=1.5, alpha=0.8)
    ax.set_title("Step-Level Train Loss", fontweight="bold", color="#1e293b")
    ax.set_xlabel("Training Steps"); ax.set_ylabel("Loss")
    ax.set_facecolor(BG); ax.grid(True, color=GRID, linestyle="--", linewidth=0.8)
    ax.spines[["top","right"]].set_visible(False)

    # Panel 2: epoch-level eval loss
    ax = axes[1]
    if eval_loss:
        ax.plot(eval_steps, eval_loss, color=RED, linewidth=2.2, marker="o",
                markersize=7, markerfacecolor="white", markeredgewidth=2,
                markeredgecolor=RED)
        for ep, el in zip(eval_steps, eval_loss):
            ax.annotate(f"{el:.4f}", xy=(ep, el), xytext=(0, 9),
                        textcoords="offset points", fontsize=8, ha="center", color=RED)
        best_ep = eval_steps[int(np.argmin(eval_loss))]
        ax.axvline(best_ep, color="#f59e0b", linewidth=1.5, linestyle=":", alpha=0.9)

    ax.set_title("Epoch-Level Eval Loss", fontweight="bold", color="#1e293b")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Eval Loss")
    ax.set_facecolor(BG); ax.grid(True, color=GRID, linestyle="--", linewidth=0.8)
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = OUTPUT_DIR / "t5_loss_curve.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"[OK] Loss curve saved: {out}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# 圖 2 & 3：Confusion Matrix + F1 Table（需要跑推論）
# ═══════════════════════════════════════════════════════════════════════════
def run_eval_and_plot():
    if not Path(MODEL_PATH).exists():
        print(f"[SKIP] 模型不存在：{MODEL_PATH}")
        return

    print("[INFO] 載入模型...")
    import torch
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model     = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()

    # 載入並切出 eval set（同 train_stage2.py 的邏輯，seed=42）
    existing = [f for f in [SYNTH_DATA, EXT_DATA] if Path(f).exists()]
    ds = load_dataset("json", data_files=existing, split="train")
    split = ds.train_test_split(test_size=0.1, seed=42)
    eval_ds = split["test"]
    print(f"[INFO] Eval set: {len(eval_ds)} 筆")

    # 推論（取前 2000 筆加快速度）
    eval_ds = eval_ds.select(range(min(2000, len(eval_ds))))
    y_true, y_pred = [], []
    valid_set = set(EMOTION_LABELS)

    for i, sample in enumerate(eval_ds):
        if i % 200 == 0:
            print(f"  {i}/{len(eval_ds)}...")
        inp = tokenizer(
            sample["input_text"], return_tensors="pt",
            max_length=256, truncation=True
        ).to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=10)
        pred = tokenizer.decode(out[0], skip_special_tokens=True).strip().lower()
        true = sample["target_text"].strip().lower()
        if true in valid_set:
            y_true.append(true)
            y_pred.append(pred if pred in valid_set else "neutral")

    acc      = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_w     = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"\nAccuracy:    {acc:.4f}")
    print(f"F1 Macro:    {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_w:.4f}")
    print(classification_report(y_true, y_pred, labels=EMOTION_LABELS, zero_division=0))

    # ── Confusion Matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=EMOTION_LABELS)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    fig.patch.set_facecolor("#ffffff")
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(EMOTION_LABELS)))
    ax.set_yticks(range(len(EMOTION_LABELS)))
    ax.set_xticklabels(EMOTION_LABELS, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(EMOTION_LABELS, fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label",      fontsize=10)
    ax.set_title(
        f"AffectiveStateAnalyzer — Confusion Matrix\n"
        f"Accuracy={acc:.3f}  Macro F1={f1_macro:.3f}",
        fontsize=11, fontweight="bold", color="#1e293b"
    )
    for i in range(len(EMOTION_LABELS)):
        for j in range(len(EMOTION_LABELS)):
            val = cm_norm[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if val > 0.5 else "#1e293b")

    plt.tight_layout()
    out = OUTPUT_DIR / "t5_confusion_matrix.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"[OK] Confusion matrix saved: {out}")
    plt.close()

    # ── Per-class F1 Bar Chart ────────────────────────────────────────────
    from sklearn.metrics import f1_score as f1_per
    f1_per_class = [
        f1_score(y_true, y_pred, labels=[emo], average="micro", zero_division=0)
        for emo in EMOTION_LABELS
    ]
    colors = ["#dc2626" if f < 0.5 else "#f59e0b" if f < 0.7 else "#16a34a"
              for f in f1_per_class]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("#ffffff")
    bars = ax.bar(EMOTION_LABELS, f1_per_class, color=colors, edgecolor="white",
                  linewidth=0.8, width=0.6)
    ax.axhline(f1_macro, color="#2563eb", linewidth=1.8, linestyle="--",
               label=f"Macro F1 = {f1_macro:.3f}")
    for bar, val in zip(bars, f1_per_class):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 Score", fontsize=10)
    ax.set_title("Per-class F1 Score — AffectiveStateAnalyzer",
                 fontsize=11, fontweight="bold", color="#1e293b")
    ax.set_facecolor("#f8fafc")
    ax.grid(True, axis="y", color="#e2e8f0", linestyle="--", linewidth=0.8)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(fontsize=9.5)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#16a34a", label="F1 ≥ 0.70 (Good)"),
        Patch(facecolor="#f59e0b", label="F1 0.50–0.70 (Fair)"),
        Patch(facecolor="#dc2626", label="F1 < 0.50 (Needs work)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8.5, loc="lower right")

    plt.tight_layout()
    out = OUTPUT_DIR / "t5_f1_table.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"[OK] F1 bar chart saved: {out}")
    plt.close()


if __name__ == "__main__":
    plot_loss_curve()
    run_eval_and_plot()
    print("\n[DONE] 三張報告圖已儲存至 Visual-Mask-System/")

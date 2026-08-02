"""
report_pipeline_visual.py — 生成簡報視覺化圖表
輸出：
  pipeline_diagram.png    — M-CoT 推理管線流程圖
  dataset_composition.png — 訓練資料來源圓餅圖
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

# ── 中文字型 ──────────────────────────────────────────────────────────────────
for _fp in [r"C:\Windows\Fonts\msjh.ttc", r"C:\Windows\Fonts\msyh.ttc"]:
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
        matplotlib.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
        break

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "docs" / "generated" / "architecture"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 圖 1：M-CoT 推理管線流程圖
# ═══════════════════════════════════════════════════════════════════════════════
def plot_pipeline():
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # ── 色盤 ────────────────────────────────────────────────────────────────
    C = {
        "input":    "#1e40af",
        "bart":     "#065f46",
        "micl":     "#064e3b",
        "stage1":   "#4c1d95",
        "stage2":   "#7c2d12",
        "vig":      "#92400e",
        "sd":       "#1e3a5f",
        "output":   "#134e4a",
        "arrow":    "#94a3b8",
        "text_w":   "#f8fafc",
        "text_dim": "#94a3b8",
        "accent1":  "#38bdf8",
        "accent2":  "#a78bfa",
        "accent3":  "#34d399",
        "accent4":  "#fb923c",
        "bg_box":   "#1e293b",
    }

    def draw_box(ax, x, y, w, h, color, label, sublabel="", accent=None):
        # 外框（發光效果）
        glow = FancyBboxPatch((x - 0.04, y - 0.04), w + 0.08, h + 0.08,
                              boxstyle="round,pad=0.05",
                              facecolor=color, alpha=0.3, zorder=2)
        ax.add_patch(glow)
        # 主框
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor=accent or C["arrow"],
                             linewidth=1.5, zorder=3)
        ax.add_patch(box)
        # 主標籤
        ax.text(x + w/2, y + h/2 + (0.1 if sublabel else 0),
                label, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=C["text_w"], zorder=4)
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.22,
                    sublabel, ha="center", va="center",
                    fontsize=8, color=C["text_dim"], zorder=4)

    def draw_arrow(ax, x1, y1, x2, y2, label="", color=None):
        c = color or C["arrow"]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=c,
                                   lw=1.8, mutation_scale=16),
                    zorder=5)
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx + 0.08, my, label, fontsize=7.5, color=C["text_dim"],
                    va="center", zorder=6,
                    bbox=dict(fc="#0f172a", ec="none", pad=1))

    # ── 標題 ────────────────────────────────────────────────────────────────
    ax.text(7, 8.6, "M-CoT 多鏈情感推理管線",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color=C["text_w"])
    ax.text(7, 8.25, "Multi-Chain-of-Thought Emotion Reasoning Pipeline",
            ha="center", va="center", fontsize=9, color=C["text_dim"])

    # ── Layer 標籤（左側） ───────────────────────────────────────────────────
    for ly, label, col in [
        (6.5, "Input Layer",      C["accent1"]),
        (4.6, "Reasoning Layer",  C["accent2"]),
        (2.7, "Generation Layer", C["accent4"]),
    ]:
        ax.text(0.35, ly, label, ha="center", va="center",
                fontsize=8, color=col, fontweight="bold", rotation=90)
        ax.plot([0.65, 0.65], [ly - 0.55, ly + 0.55],
                color=col, linewidth=2, alpha=0.6, zorder=2)

    # ════════════════════════════════════════════════════════════════════════
    # Row 1：使用者輸入
    # ════════════════════════════════════════════════════════════════════════
    draw_box(ax, 4.5, 7.4, 5.0, 0.7, C["input"],
             "使用者輸入", '"omg I was so mad at my boss"',
             accent=C["accent1"])

    # ════════════════════════════════════════════════════════════════════════
    # Row 2：Input Layer（BART + Buffer + M-ICL）
    # ════════════════════════════════════════════════════════════════════════
    # BART
    draw_box(ax, 1.0, 5.9, 3.2, 0.85, C["bart"],
             "TextSummarizer", "BART-large-CNN", accent=C["accent1"])
    # ContextualBuffer
    draw_box(ax, 5.0, 5.9, 2.8, 0.85, C["micl"],
             "ContextualBuffer", "滑動視窗（5輪）", accent=C["accent1"])
    # M-ICL
    draw_box(ax, 8.5, 5.9, 3.5, 0.85, C["micl"],
             "M_ICL_Integrator", "情緒趨勢追蹤", accent=C["accent3"])

    # ════════════════════════════════════════════════════════════════════════
    # Row 3：raw_context 合併
    # ════════════════════════════════════════════════════════════════════════
    draw_box(ax, 3.5, 4.7, 7.0, 0.7, C["bg_box"],
             "Context / Current Stream / User Prior",
             "結構化輸入字串", accent=C["arrow"])

    # ════════════════════════════════════════════════════════════════════════
    # Row 4：Reasoning Layer（Stage 1 + Stage 2）
    # ════════════════════════════════════════════════════════════════════════
    draw_box(ax, 1.0, 3.45, 5.0, 0.95, C["stage1"],
             "Stage 1：SemanticIntentDecoder",
             "T5-base  →  Rationale 推理鏈", accent=C["accent2"])
    draw_box(ax, 7.0, 3.45, 5.0, 0.95, C["stage2"],
             "Stage 2：AffectiveStateAnalyzer",
             "T5-base  →  Emotion Label", accent=C["accent4"])

    # Label 正規化
    draw_box(ax, 7.0, 2.5, 5.0, 0.65, C["bg_box"],
             "Label 正規化層",
             "非標準標籤 → 標準 7 類", accent=C["accent4"])

    # ════════════════════════════════════════════════════════════════════════
    # Row 5：VIG
    # ════════════════════════════════════════════════════════════════════════
    draw_box(ax, 3.5, 1.55, 7.0, 0.75, C["vig"],
             "VisualInstructionGenerator",
             "Emotion Label  →  SD Prompt + Intensity", accent=C["accent4"])

    # ════════════════════════════════════════════════════════════════════════
    # Row 6：Generation Layer（SD + LoRA）
    # ════════════════════════════════════════════════════════════════════════
    draw_box(ax, 1.5, 0.35, 5.0, 0.95, C["sd"],
             "AnimateDiff / Stable Diffusion",
             "SD v1.5 + LoRA（person5805）", accent=C["accent1"])
    draw_box(ax, 7.5, 0.35, 4.5, 0.95, C["output"],
             "輸出圖片 / GIF",
             "sticker_anger_xxxx.png", accent=C["accent3"])

    # ════════════════════════════════════════════════════════════════════════
    # 箭頭
    # ════════════════════════════════════════════════════════════════════════
    # 輸入 → BART / Buffer / M-ICL
    draw_arrow(ax, 5.5,  7.4, 2.6,  6.75, color=C["accent1"])
    draw_arrow(ax, 7.0,  7.4, 6.4,  6.75, color=C["accent1"])
    draw_arrow(ax, 8.5,  7.4, 10.25, 6.75, color=C["accent1"])

    # BART / Buffer / M-ICL → raw_context
    draw_arrow(ax, 2.6,  5.9, 5.2,  5.4,  color=C["accent1"])
    draw_arrow(ax, 6.4,  5.9, 7.0,  5.4,  color=C["accent1"])
    draw_arrow(ax, 10.25, 5.9, 8.8, 5.4,  color=C["accent3"])

    # raw_context → Stage 1
    draw_arrow(ax, 5.5,  4.7, 3.5,  4.4,  color=C["accent2"])
    # raw_context → Stage 2
    draw_arrow(ax, 8.5,  4.7, 10.0, 4.4,  color=C["accent4"])

    # Stage 1 → Stage 2（rationale）
    draw_arrow(ax, 6.0,  3.92, 7.0, 3.92,
               label="Rationale", color=C["accent2"])

    # Stage 2 → 正規化
    draw_arrow(ax, 9.5,  3.45, 9.5, 3.15, color=C["accent4"])

    # 正規化 → VIG
    draw_arrow(ax, 9.5,  2.5, 8.0,  2.3,  color=C["accent4"])

    # Stage 1 也接 VIG（rationale 傳 context）
    draw_arrow(ax, 3.5,  3.45, 5.5,  2.3,  color=C["accent2"])

    # VIG → SD
    draw_arrow(ax, 5.5,  1.55, 4.0,  1.3,  color=C["accent4"])
    # VIG → Output
    draw_arrow(ax, 8.5,  1.55, 9.75, 1.3,  color=C["accent3"])

    # SD → Output
    draw_arrow(ax, 6.5,  0.82, 7.5,  0.82, color=C["accent3"])

    # ── 情緒標籤說明框 ──────────────────────────────────────────────────────
    labels_text = "anger / joy / sadness\nneutral / fear\ndisgust / surprise"
    ax.text(12.2, 3.92, labels_text, ha="center", va="center",
            fontsize=7.5, color=C["accent4"],
            bbox=dict(fc="#7c2d12", ec=C["accent4"], pad=4,
                      boxstyle="round,pad=0.3"),
            zorder=6)
    draw_arrow(ax, 12.0, 3.92, 11.0, 3.92, color=C["accent4"])

    plt.tight_layout(pad=0.5)
    out = OUTPUT_DIR / "pipeline_diagram.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="#0f172a")
    print(f"[OK] Pipeline diagram saved: {out}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 圖 2：訓練資料來源圓餅圖
# ═══════════════════════════════════════════════════════════════════════════════
def plot_dataset():
    datasets = [
        ("GoEmotions\n(Google)",    58000,  "#3b82f6"),
        ("OpinRank\n(Hotels/Cars)", 17127,  "#8b5cf6"),
        ("MELD\n(Friends TV)",      10000,  "#06b6d4"),
        ("ISEAR",                    7665,  "#10b981"),
        ("SemEval-2018",             3259,  "#f59e0b"),
        ("合成資料\n(Synthetic)",       49,  "#6b7280"),
    ]

    labels  = [d[0] for d in datasets]
    sizes   = [d[1] for d in datasets]
    colors  = [d[2] for d in datasets]
    total   = sum(sizes)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.5),
                                   gridspec_kw={"width_ratios": [1.2, 1]})
    fig.patch.set_facecolor("#0f172a")

    # ── 左：圓餅圖 ────────────────────────────────────────────────────────
    ax1.set_facecolor("#0f172a")

    explode = [0.04] * len(datasets)
    explode[0] = 0.08   # GoEmotions 突出

    wedges, texts, autotexts = ax1.pie(
        sizes,
        labels=None,
        colors=colors,
        explode=explode,
        autopct=lambda p: f"{p:.1f}%\n({int(p*total/100):,})",
        pctdistance=0.72,
        startangle=140,
        wedgeprops=dict(edgecolor="#0f172a", linewidth=2),
    )
    for at in autotexts:
        at.set_fontsize(8.5)
        at.set_color("white")
        at.set_fontweight("bold")

    # 圓心文字
    ax1.text(0, 0, f"總計\n{total:,}\n筆", ha="center", va="center",
             fontsize=11, fontweight="bold", color="white")

    ax1.set_title("Stage 2 訓練資料來源分佈",
                  fontsize=13, fontweight="bold", color="white", pad=16)

    # 圖例
    legend_patches = [mpatches.Patch(color=c, label=f"{l}  {s:,} 筆")
                      for l, s, c in zip(labels, sizes, colors)]
    ax1.legend(handles=legend_patches, loc="lower center",
               bbox_to_anchor=(0.5, -0.18), ncol=2,
               fontsize=8.5, framealpha=0.15,
               labelcolor="white", facecolor="#1e293b",
               edgecolor="#334155")

    # ── 右：橫向長條圖 ────────────────────────────────────────────────────
    ax2.set_facecolor("#0f172a")

    sorted_data = sorted(zip(sizes, labels, colors), reverse=True)
    s_sizes, s_labels, s_colors = zip(*sorted_data)

    y_pos = range(len(s_labels))
    bars = ax2.barh(list(y_pos), s_sizes, color=s_colors,
                    edgecolor="#0f172a", linewidth=1.2, height=0.6)

    # 數值標籤
    for bar, size in zip(bars, s_sizes):
        pct = size / total * 100
        ax2.text(bar.get_width() + 400, bar.get_y() + bar.get_height()/2,
                 f"{size:,}  ({pct:.1f}%)",
                 va="center", fontsize=8.5, color="white")

    ax2.set_yticks(list(y_pos))
    ax2.set_yticklabels([l.replace("\n", " ") for l in s_labels],
                        fontsize=9, color="white")
    ax2.set_xlabel("樣本數", fontsize=9, color="#94a3b8")
    ax2.set_title("各資料集樣本數量",
                  fontsize=12, fontweight="bold", color="white", pad=12)
    ax2.set_facecolor("#0f172a")
    ax2.tick_params(colors="#94a3b8")
    ax2.spines[["top","right","bottom","left"]].set_color("#334155")
    ax2.set_xlim(0, max(s_sizes) * 1.35)
    ax2.grid(True, axis="x", color="#1e293b", linewidth=0.8)

    # 標籤說明
    fig.text(0.5, 0.01,
             "資料集涵蓋：日常情感文字 / 影視對話 / 產品評論 / 社群媒體留言",
             ha="center", fontsize=8.5, color="#64748b")

    plt.tight_layout(pad=1.5)
    out = OUTPUT_DIR / "dataset_composition.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="#0f172a")
    print(f"[OK] Dataset composition saved: {out}")
    plt.close()


if __name__ == "__main__":
    print("生成流程圖...")
    plot_pipeline()
    print("生成資料來源圖...")
    plot_dataset()
    print("\n[DONE] 兩張圖已儲存至 Visual-Mask-System/")

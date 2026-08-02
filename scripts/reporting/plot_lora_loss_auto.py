"""
plot_lora_loss_auto.py
自動抓最新一次 Kohya-SS 訓練的 TensorBoard log，輸出 loss 分析圖。
同時讀取 val_watcher_v2 的 val loss 疊加在 epoch-level 圖表上。

用法：
  conda activate mask_env
  python scripts/reporting/plot_lora_loss_auto.py                          # 自動抓最新 log
  python scripts/reporting/plot_lora_loss_auto.py --name 5805_v2          # 圖表標題
  python scripts/reporting/plot_lora_loss_auto.py --log 20260527xxxxxx    # 指定特定 log 資料夾名稱
  python scripts/reporting/plot_lora_loss_auto.py --no-val                # 不疊加 val loss

輸出：
  docs/generated/training/lora_loss_<name>.png
"""

import sys, os, io, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path

# ── 中文字型 ─────────────────────────────────────────────────────────────────
for _fp in [r"C:\Windows\Fonts\msjh.ttc", r"C:\Windows\Fonts\msyh.ttc"]:
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
        matplotlib.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
        break

LOG_BASE    = r"C:\Ethan\Edu_proj\DataSets\CelebA\train_data\OUPUT\Logs"
VAL_LOG_DIR = r"C:\Ethan\Edu_proj\DataSets\CelebA\train_data\OUPUT\Logs\val_watcher_v2"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "docs" / "generated" / "training"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_latest_log(base: str, specific: str | None = None) -> str:
    """回傳最新（或指定）log 的 network_train 資料夾路徑。"""
    # 只篩選時間戳資料夾（純數字），排除 val_watcher* 等子目錄
    entries = sorted(
        e for e in os.listdir(base)
        if os.path.isdir(os.path.join(base, e)) and e[:4].isdigit()
    )
    if specific:
        entries = [e for e in entries if specific in e]
        if not entries:
            raise FileNotFoundError(f"找不到含 '{specific}' 的 log 資料夾")
    for name in reversed(entries):              # 從最新往前找
        net = os.path.join(base, name, "network_train")
        if os.path.isdir(net):
            return net, name
    raise FileNotFoundError("LOG_BASE 裡找不到任何 network_train 資料夾")


def load_tb(log_path: str):
    ea = event_accumulator.EventAccumulator(log_path, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])

    def get(tag):
        if tag not in tags:
            return np.array([]), np.array([])
        ev = ea.Scalars(tag)
        return np.array([e.step for e in ev]), np.array([e.value for e in ev])

    return {
        "cur":  get("loss/current"),
        "avg":  get("loss/average"),
        "ep":   get("loss/epoch_average"),
    }


def load_val_loss(val_log_dir: str):
    """從 val_watcher TensorBoard log 讀取 val loss（按 epoch）。"""
    if not os.path.isdir(val_log_dir):
        print(f"[WARN] Val log dir not found: {val_log_dir}")
        return np.array([]), np.array([])
    ea = event_accumulator.EventAccumulator(val_log_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if "loss/val_loss" not in tags:
        print(f"[WARN] 'loss/val_loss' tag not found in {val_log_dir}")
        return np.array([]), np.array([])
    ev = ea.Scalars("loss/val_loss")
    steps  = np.array([e.step  for e in ev])
    values = np.array([e.value for e in ev])
    # 排序（以防 watcher 不按順序寫入）
    idx = np.argsort(steps)
    return steps[idx], values[idx]


def smooth(arr, w=60):
    if len(arr) < w:
        return arr
    k = np.ones(w) / w
    p = np.pad(arr, (w // 2, w // 2), mode="edge")
    return np.convolve(p, k, mode="valid")[: len(arr)]


def plot(data: dict, val_steps: np.ndarray, val_loss: np.ndarray,
         title: str, out_path: str):
    steps_cur, loss_cur = data["cur"]
    steps_avg, loss_avg = data["avg"]
    steps_ep,  loss_ep  = data["ep"]

    BLUE   = "#2563eb"
    RED    = "#dc2626"
    GREEN  = "#16a34a"
    GRAY   = "#94a3b8"
    BG     = "#f8fafc"
    GRID   = "#e2e8f0"

    has_val = len(val_steps) > 0

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#ffffff")
    fig.suptitle(f"Kohya-SS LoRA Training Analysis  —  {title}",
                 fontsize=13, fontweight="bold", color="#1e293b", y=1.01)

    # ── 左圖：step-level ──────────────────────────────────────────────────────
    ax = axes[0]
    if len(loss_cur):
        loss_sm = smooth(loss_cur, 60)
        ax.plot(steps_cur, loss_cur, color=BLUE, alpha=0.12, linewidth=0.6,
                label="Per-step (raw)")
        ax.plot(steps_cur, loss_sm, color=BLUE, linewidth=2.2,
                label="Per-step (smoothed)")
    if len(loss_avg):
        ax.plot(steps_avg, loss_avg, color=RED, linewidth=1.6,
                linestyle="--", label="Running average")

    # epoch 分隔線（每 N steps）
    if len(steps_ep) >= 2:
        ep_interval = int(steps_ep[0]) if len(steps_ep) == 1 else int(steps_ep[1] - steps_ep[0])
        for boundary in range(ep_interval, int(steps_cur[-1]) + 1 if len(steps_cur) else 1, ep_interval):
            ax.axvline(boundary, color=GRAY, linewidth=0.5, linestyle=":", alpha=0.6)

    ax.set_title("Step-Level Training Loss", fontsize=11, fontweight="bold", color="#1e293b")
    ax.set_xlabel("Training Steps", fontsize=10, color="#475569")
    ax.set_ylabel("Loss", fontsize=10, color="#475569")
    ax.set_facecolor(BG); ax.grid(True, color=GRID, linewidth=0.8, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="#64748b", labelsize=9)
    ax.legend(fontsize=8.5, framealpha=0.9)
    if len(loss_cur):
        ax.set_ylim(bottom=0, top=min(float(loss_cur.max()), 1.2))

    # ── 右圖：epoch-level（Train + Val）────────────────────────────────────────
    ax = axes[1]

    # ── Train loss 曲線 ──
    if len(loss_ep):
        ax.plot(steps_ep, loss_ep, color=BLUE, linewidth=2.5, marker="o",
                markersize=7, markerfacecolor="white", markeredgewidth=2,
                markeredgecolor=BLUE, label="Train Loss (epoch avg)", zorder=5)

        for e, l in zip(steps_ep, loss_ep):
            ax.annotate(f"{l:.4f}", xy=(e, l), xytext=(0, 10),
                        textcoords="offset points", fontsize=7.5,
                        ha="center", color="#1e293b")

        ax.set_xticks(steps_ep)

    # ── Val loss 曲線 ──
    if has_val:
        ax.plot(val_steps, val_loss, color=GREEN, linewidth=2.5, marker="s",
                markersize=7, markerfacecolor="white", markeredgewidth=2,
                markeredgecolor=GREEN, label="Val Loss (diffusion MSE)", zorder=5)

        for e, l in zip(val_steps, val_loss):
            ax.annotate(f"{l:.4f}", xy=(e, l), xytext=(0, -14),
                        textcoords="offset points", fontsize=7.5,
                        ha="center", color="#166534")

        # best marker 已移除

        # 動態 y 範圍：包含 train + val 兩條曲線
        all_vals = np.concatenate([loss_ep, val_loss]) if len(loss_ep) else val_loss
        ax.set_ylim(bottom=max(0, float(all_vals.min()) - 0.04),
                    top=float(all_vals.max()) + 0.04)
    elif len(loss_ep):
        ax.set_ylim(bottom=max(0, float(loss_ep.min()) - 0.03),
                    top=float(loss_ep.max()) + 0.03)

    ax.set_title("Epoch-Level: Train vs Val Loss", fontsize=11, fontweight="bold", color="#1e293b")
    ax.set_xlabel("Epoch", fontsize=10, color="#475569")
    ax.set_ylabel("Loss", fontsize=10, color="#475569")
    ax.set_facecolor(BG); ax.grid(True, color=GRID, linewidth=0.8, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="#64748b", labelsize=9)
    ax.legend(fontsize=8.5, framealpha=0.9)

    plt.tight_layout(pad=2.0)
    plt.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"[OK] 圖表已儲存：{out_path}")
    plt.close()


def print_table(data: dict, val_steps: np.ndarray, val_loss: np.ndarray):
    steps_ep, loss_ep = data["ep"]
    has_val = len(val_steps) > 0

    if not len(loss_ep):
        print("[WARN] 沒有 epoch_average 資料")
        return

    # 建立 val loss 字典（epoch → val_loss）
    val_dict = dict(zip(val_steps.astype(int), val_loss)) if has_val else {}

    header = f"{'Epoch':>6} | {'Train Loss':>11}"
    sep    = "-" * 22
    if has_val:
        header += f" | {'Val Loss':>10}"
        sep    += "-" * 14

    print(f"\n=== Train vs Val Loss per Epoch ===")
    print(header)
    print(sep)

    best_train = int(steps_ep[np.argmin(loss_ep)])
    best_val   = int(val_steps[np.argmin(val_loss)]) if has_val else None

    for e, l in zip(steps_ep, loss_ep):
        ep = int(e)
        mark_train = " ←T" if ep == best_train else ""
        row = f"{ep:>6} | {l:>11.4f}{mark_train}"
        if has_val:
            vl = val_dict.get(ep, float("nan"))
            mark_val = " ←V" if ep == best_val else ""
            row += f" | {vl:>10.4f}{mark_val}"
        print(row)

    print(f"\n最低 Train Loss：Epoch {best_train}（{loss_ep.min():.4f}）")
    if has_val:
        print(f"最低 Val Loss：  Epoch {best_val}（{val_loss.min():.4f}）")
        if best_val != best_train:
            print(f"  ★ 建議使用 Epoch {best_val} 的 checkpoint（以 val loss 判斷）")
        else:
            print(f"  ★ Train 與 Val 最低點一致，Epoch {best_val} 是最佳 checkpoint。")
    else:
        print("（尚無 val loss 資料，無法確認是否過擬合）")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",   default="", help="圖表標題（如 5805_v2）")
    parser.add_argument("--log",    default=None, help="指定 log 資料夾名稱（部分匹配）")
    parser.add_argument("--no-val", action="store_true", help="不載入 val loss")
    args = parser.parse_args()

    log_path, folder_name = find_latest_log(LOG_BASE, args.log)
    label = args.name or folder_name
    print(f"[INFO] Train log：{log_path}")

    data = load_tb(log_path)

    val_steps, val_loss = (np.array([]), np.array([])) if args.no_val \
                          else load_val_loss(VAL_LOG_DIR)

    if len(val_steps):
        print(f"[INFO] Val log：{VAL_LOG_DIR}（{len(val_steps)} epochs）")
    else:
        print("[INFO] Val loss 尚無資料，僅顯示 train loss。")

    print_table(data, val_steps, val_loss)

    out = OUTPUT_DIR / f"lora_loss_{label}.png"
    plot(data, val_steps, val_loss, label, out)


if __name__ == "__main__":
    main()

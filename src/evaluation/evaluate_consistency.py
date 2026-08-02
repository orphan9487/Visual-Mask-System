# -*- coding: utf-8 -*-
"""
生成一致性評估 (Generation Consistency Evaluation) — 三層驗證架構「第二層」

對應計畫書驗證方案第二層「生成一致性（純技術跑分，適合消費級 GPU）」：
    1. 身分一致性 (Identity Consistency)   : CLIP-I / DINO
    2. 時序連貫  (Temporal Coherence)      : CLIP 幀間一致性 + warping error
    3. 語意對齊  (Text-Image Alignment)    : CLIP-T

本腳本刻意設計成「可分級執行」：
  - reference-free 指標（CLIP-T、時序連貫、身分自一致性）只需要現有生成結果，
    在任何機器（含 CPU）都能立即跑出數字 → 直接寫進計畫書 Preliminary Results。
  - reference-based 指標（生成圖 vs 該人物真圖的 CLIP-I / DINO / ArcFace）需要
    真人參照圖，透過 --reference_dir 提供；沒有提供時自動略過。

依賴：torch, transformers, pillow, numpy；warping error 額外需要 opencv-python(cv2)。
CLIP 用 openai/clip-vit-base-patch32、DINO 用 facebook/dinov2-small（首次執行會自 HF 下載並快取）。

用法：
    # 只跑 reference-free（現況即可）
    python src/evaluation/evaluate_consistency.py

    # 加上與真人參照圖的比對（回到 GPU 機、有真圖時）
    python src/evaluation/evaluate_consistency.py --reference_dir data/reference/person8692
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence

import torch


# --------------------------------------------------------------------------- #
# 設定區
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DINO_MODEL_ID = "facebook/dinov2-small"

# 檔名情緒標籤 → 用於 CLIP-T 的自然語言 prompt。
# CLIP-T 衡量「生成圖是否真的長得像它宣稱的情緒」，是本系統核心主張的直接量測。
EMOTION_PROMPTS = {
    "angry": "a photo of an angry person, furrowed brows, furious, frowning expression",
    "happy": "a photo of a happy person, smiling, cheerful expression",
    "sad": "a photo of a sad person, sorrowful, downcast expression",
    "surprised": "a photo of a surprised person, wide eyes, shocked expression",
    "neutral": "a photo of a person with a calm, neutral expression",
}

# 為了對照，CLIP-T 同時對「所有情緒 prompt」計算相似度，回報 argmax 是否命中，
# 這樣就能得到一個「情緒可辨識率 (emotion recognizability)」，比單一 CLIP-T 分數更有說服力。
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# 影像 / GIF 載入工具
# --------------------------------------------------------------------------- #
def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_gif_frames(path: Path):
    """讀取 GIF 的所有幀，回傳 list[PIL.Image](RGB)。"""
    frames = []
    with Image.open(path) as im:
        for frame in ImageSequence.Iterator(im):
            frames.append(frame.convert("RGB").copy())
    return frames


def parse_emotion_from_name(name: str) -> str:
    """從檔名解析情緒標籤，例如 result_angry_532002.gif → angry。"""
    lower = name.lower()
    for emo in EMOTION_PROMPTS:
        if emo in lower:
            return emo
    return "neutral"


# --------------------------------------------------------------------------- #
# 特徵抽取器 (CLIP + DINO)
# --------------------------------------------------------------------------- #
class FeatureExtractor:
    """封裝 CLIP（影像+文字）與 DINO（純影像）的 L2-normalized embedding。"""

    def __init__(self, load_dino: bool = True):
        from transformers import CLIPModel, CLIPProcessor

        print(f"[load] CLIP  : {CLIP_MODEL_ID}  (device={DEVICE})")
        self.clip = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(DEVICE).eval()
        self.clip_proc = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

        self.dino = None
        self.dino_proc = None
        if load_dino:
            try:
                from transformers import AutoModel, AutoImageProcessor

                print(f"[load] DINO  : {DINO_MODEL_ID}")
                self.dino = AutoModel.from_pretrained(DINO_MODEL_ID).to(DEVICE).eval()
                self.dino_proc = AutoImageProcessor.from_pretrained(DINO_MODEL_ID)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] DINO 載入失敗，將略過 DINO 指標：{e}")

    @staticmethod
    def _as_feature_tensor(out):
        """不同 transformers 版本可能回傳 tensor 或 ModelOutput，統一取出特徵 tensor。"""
        if isinstance(out, torch.Tensor):
            return out
        for attr in ("pooler_output", "image_embeds", "text_embeds", "last_hidden_state"):
            val = getattr(out, attr, None)
            if val is not None:
                return val
        raise TypeError(f"無法從 {type(out)} 取出特徵 tensor")

    @torch.inference_mode()
    def clip_image(self, img: Image.Image) -> np.ndarray:
        inputs = self.clip_proc(images=img, return_tensors="pt").to(DEVICE)
        feat = self._as_feature_tensor(self.clip.get_image_features(**inputs))
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).float().cpu().numpy()

    @torch.inference_mode()
    def clip_text(self, text: str) -> np.ndarray:
        inputs = self.clip_proc(
            text=[text], return_tensors="pt", padding=True, truncation=True
        ).to(DEVICE)
        feat = self._as_feature_tensor(self.clip.get_text_features(**inputs))
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).float().cpu().numpy()

    @torch.inference_mode()
    def dino_image(self, img: Image.Image):
        if self.dino is None:
            return None
        inputs = self.dino_proc(images=img, return_tensors="pt").to(DEVICE)
        out = self.dino(**inputs)
        # 用 CLS token 作為全域影像表徵
        feat = out.last_hidden_state[:, 0, :]
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).float().cpu().numpy()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # 皆已 L2-normalized


def mean_pairwise_cosine(vecs) -> float:
    """一組向量兩兩 cosine 的平均（衡量內聚一致性）。"""
    n = len(vecs)
    if n < 2:
        return float("nan")
    sims = [cosine(vecs[i], vecs[j]) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(sims))


# --------------------------------------------------------------------------- #
# 指標 1：CLIP-T（語意/情緒對齊）
# --------------------------------------------------------------------------- #
def eval_clip_t(fx: FeatureExtractor, image_paths):
    """
    對每張圖：
      - clip_t_target : 與「宣稱情緒」prompt 的相似度
      - emotion_hit   : 在所有情緒 prompt 中，argmax 是否 == 宣稱情緒（情緒可辨識率）
    """
    text_feats = {emo: fx.clip_text(p) for emo, p in EMOTION_PROMPTS.items()}
    per_image = []
    hits = 0
    for path in image_paths:
        emo = parse_emotion_from_name(path.name)
        img_feat = fx.clip_image(load_image(path))
        sims = {e: cosine(img_feat, tf) for e, tf in text_feats.items()}
        pred = max(sims, key=sims.get)
        hit = int(pred == emo)
        hits += hit
        per_image.append(
            {
                "file": path.name,
                "claimed_emotion": emo,
                "clip_t_target": round(sims[emo], 4),
                "predicted_emotion": pred,
                "emotion_hit": hit,
            }
        )
    n = len(per_image)
    return {
        "n_images": n,
        "clip_t_mean": round(float(np.mean([r["clip_t_target"] for r in per_image])), 4) if n else None,
        "emotion_recognizability": round(hits / n, 4) if n else None,
        "per_image": per_image,
    }


# --------------------------------------------------------------------------- #
# 指標 2：身分自一致性（intra-identity cohesion，reference-free）
# --------------------------------------------------------------------------- #
def eval_identity_self_consistency(fx: FeatureExtractor, image_paths):
    """
    同一個視覺面具(person8692)在多次生成間，身分是否穩定？
    以 CLIP-I / DINO 對這組圖做「兩兩平均 cosine」衡量內聚度。
    註：這是 self-consistency（非 vs 真人 ground-truth）。gold-standard 版本見 eval_reference_based。
    """
    clip_vecs = [fx.clip_image(load_image(p)) for p in image_paths]
    result = {
        "n_images": len(image_paths),
        "clip_i_self_consistency": round(mean_pairwise_cosine(clip_vecs), 4),
    }
    if fx.dino is not None:
        dino_vecs = [fx.dino_image(load_image(p)) for p in image_paths]
        result["dino_self_consistency"] = round(mean_pairwise_cosine(dino_vecs), 4)
    return result


# --------------------------------------------------------------------------- #
# 指標 3：時序連貫（CLIP 幀間一致性 + warping error）
# --------------------------------------------------------------------------- #
def eval_temporal_clip(fx: FeatureExtractor, frames):
    """相鄰幀 CLIP-I cosine 的平均，越高代表時序越平滑。"""
    feats = [fx.clip_image(f) for f in frames]
    sims = [cosine(feats[i], feats[i + 1]) for i in range(len(feats) - 1)]
    return {
        "n_frames": len(frames),
        "clip_frame_consistency": round(float(np.mean(sims)), 4) if sims else None,
    }


def eval_warping_error(frames):
    """
    以光流(Farneback)把 frame_t 對齊到 frame_{t+1}，計算殘差 MSE(0-1 尺度)。
    越低代表運動越連貫、閃爍越少。需要 opencv-python。
    """
    try:
        import cv2
    except Exception:  # noqa: BLE001
        return {"warping_error": None, "note": "cv2 未安裝，略過 warping error"}

    grays = [cv2.cvtColor(np.array(f), cv2.COLOR_RGB2GRAY) for f in frames]
    rgbs = [np.array(f).astype(np.float32) / 255.0 for f in frames]
    errs = []
    for i in range(len(frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            grays[i], grays[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        h, w = grays[i].shape
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)
        warped = cv2.remap(rgbs[i], map_x, map_y, cv2.INTER_LINEAR)
        errs.append(float(np.mean((warped - rgbs[i + 1]) ** 2)))
    return {"warping_error": round(float(np.mean(errs)), 6) if errs else None}


# --------------------------------------------------------------------------- #
# 指標 4：reference-based（生成 vs 真人，需 --reference_dir）
# --------------------------------------------------------------------------- #
def eval_reference_based(fx: FeatureExtractor, gen_paths, ref_dir: Path):
    ref_paths = sorted(
        p for p in ref_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not ref_paths:
        return {"note": f"參照資料夾 {ref_dir} 內無影像，略過。"}

    ref_clip = [fx.clip_image(load_image(p)) for p in ref_paths]
    ref_clip_mean = np.mean(ref_clip, axis=0)
    ref_clip_mean = ref_clip_mean / np.linalg.norm(ref_clip_mean)

    ref_dino_mean = None
    if fx.dino is not None:
        ref_dino = [fx.dino_image(load_image(p)) for p in ref_paths]
        ref_dino_mean = np.mean(ref_dino, axis=0)
        ref_dino_mean = ref_dino_mean / np.linalg.norm(ref_dino_mean)

    per_image = []
    for p in gen_paths:
        img = load_image(p)
        row = {"file": p.name, "clip_i_vs_ref": round(cosine(fx.clip_image(img), ref_clip_mean), 4)}
        if ref_dino_mean is not None:
            row["dino_vs_ref"] = round(cosine(fx.dino_image(img), ref_dino_mean), 4)
        per_image.append(row)

    out = {
        "reference_dir": str(ref_dir),
        "n_reference_images": len(ref_paths),
        "clip_i_vs_ref_mean": round(float(np.mean([r["clip_i_vs_ref"] for r in per_image])), 4),
        "per_image": per_image,
    }
    if ref_dino_mean is not None:
        out["dino_vs_ref_mean"] = round(
            float(np.mean([r["dino_vs_ref"] for r in per_image])), 4
        )
    return out


# --------------------------------------------------------------------------- #
# 資產探索
# --------------------------------------------------------------------------- #
def discover_assets(root: Path):
    """從專案根目錄找出生成的靜態圖與 GIF。"""
    images = sorted(
        p for p in root.glob("*.png")
        if p.name.startswith(("initial_", "sticker_", "result_"))
    )
    gifs = sorted(root.glob("*.gif"))
    return images, gifs


# --------------------------------------------------------------------------- #
# Markdown 報告
# --------------------------------------------------------------------------- #
def write_markdown(report: dict, path: Path):
    L = []
    L.append("# 生成一致性評估報告（三層驗證 · 第二層）\n")
    L.append(f"- 產生時間：{report['timestamp']}")
    L.append(f"- 裝置：`{report['device']}`")
    L.append(f"- CLIP：`{CLIP_MODEL_ID}` / DINO：`{DINO_MODEL_ID}`\n")

    ct = report.get("clip_t")
    if ct:
        L.append("## 1. 語意/情緒對齊 (CLIP-T)\n")
        L.append(f"- 平均 CLIP-T（對宣稱情緒）：**{ct['clip_t_mean']}**")
        L.append(f"- 情緒可辨識率（argmax 命中率）：**{ct['emotion_recognizability']}**  "
                 f"（n={ct['n_images']}）\n")
        L.append("| 檔案 | 宣稱情緒 | CLIP-T | 預測情緒 | 命中 |")
        L.append("|---|---|---|---|---|")
        for r in ct["per_image"]:
            L.append(f"| {r['file']} | {r['claimed_emotion']} | {r['clip_t_target']} "
                     f"| {r['predicted_emotion']} | {'✓' if r['emotion_hit'] else '✗'} |")
        L.append("")

    idc = report.get("identity_self_consistency")
    if idc:
        L.append("## 2. 身分自一致性 (CLIP-I / DINO，同一面具多次生成的內聚度)\n")
        L.append(f"- CLIP-I 自一致性：**{idc.get('clip_i_self_consistency')}**")
        if "dino_self_consistency" in idc:
            L.append(f"- DINO 自一致性：**{idc.get('dino_self_consistency')}**")
        L.append(f"- 樣本數：{idc['n_images']}")
        L.append("- 註：此為 self-consistency（同組生成互比），非 vs 真人 ground-truth。\n")

    tc = report.get("temporal")
    if tc:
        L.append("## 3. 時序連貫 (Temporal Coherence)\n")
        L.append("| GIF | 幀數 | CLIP 幀間一致性 | Warping Error |")
        L.append("|---|---|---|---|")
        for r in tc:
            L.append(f"| {r['file']} | {r.get('n_frames')} | "
                     f"{r.get('clip_frame_consistency')} | {r.get('warping_error')} |")
        L.append("")

    rb = report.get("reference_based")
    if rb and "note" not in rb:
        L.append("## 4. 身分一致性 vs 真人參照 (gold standard)\n")
        L.append(f"- 參照圖數：{rb['n_reference_images']}（{rb['reference_dir']}）")
        L.append(f"- CLIP-I vs 真人（平均）：**{rb['clip_i_vs_ref_mean']}**")
        if "dino_vs_ref_mean" in rb:
            L.append(f"- DINO vs 真人（平均）：**{rb['dino_vs_ref_mean']}**")
        L.append("")

    L.append("---")
    L.append("### 指標解讀速記")
    L.append("- **CLIP-T / 情緒可辨識率**：生成圖是否真的傳達出目標情緒（越高越好）。")
    L.append("- **身分自一致性**：同一面具重複生成是否維持同一人（越高越穩定）。")
    L.append("- **CLIP 幀間一致性**：影片相鄰幀是否平滑（越高越好）。")
    L.append("- **Warping Error**：光流對齊後殘差，衡量閃爍/抖動（越低越好）。")
    path.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT),
                    help="放置生成結果(png/gif)的根目錄，預設為專案根目錄")
    ap.add_argument("--reference_dir", default=None,
                    help="真人參照圖資料夾（提供才會跑 gold-standard 身分一致性）")
    ap.add_argument("--out_dir", default=str(PROJECT_ROOT / "evaluation_results"))
    ap.add_argument("--no_dino", action="store_true", help="略過 DINO（省下載/加速）")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images, gifs = discover_assets(root)
    print(f"[assets] 靜態圖 {len(images)} 張、GIF {len(gifs)} 段")
    if not images and not gifs:
        print("[error] 找不到任何生成結果（initial_/sticker_/result_*.png 或 *.gif）。")
        return

    fx = FeatureExtractor(load_dino=not args.no_dino)

    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": DEVICE,
        "assets": {"images": [p.name for p in images], "gifs": [p.name for p in gifs]},
    }

    # 1. CLIP-T
    if images:
        print("[run] CLIP-T ...")
        report["clip_t"] = eval_clip_t(fx, images)

    # 2. 身分自一致性
    if len(images) >= 2:
        print("[run] 身分自一致性 ...")
        report["identity_self_consistency"] = eval_identity_self_consistency(fx, images)

    # 3. 時序連貫
    if gifs:
        print("[run] 時序連貫 ...")
        temporal = []
        for g in gifs:
            frames = load_gif_frames(g)
            row = {"file": g.name}
            row.update(eval_temporal_clip(fx, frames))
            row.update(eval_warping_error(frames))
            temporal.append(row)
            print(f"    {g.name}: frames={row['n_frames']} "
                  f"clip={row['clip_frame_consistency']} warp={row.get('warping_error')}")
        report["temporal"] = temporal

    # 4. reference-based（可選）
    if args.reference_dir:
        ref_dir = Path(args.reference_dir)
        if ref_dir.is_dir():
            print(f"[run] 與真人參照比對 ({ref_dir}) ...")
            report["reference_based"] = eval_reference_based(fx, images, ref_dir)
        else:
            print(f"[warn] reference_dir 不存在：{ref_dir}")

    # 輸出
    json_path = out_dir / "consistency_report.json"
    md_path = out_dir / "consistency_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(f"\n[done] JSON → {json_path}")
    print(f"[done] 報告 → {md_path}")


if __name__ == "__main__":
    main()

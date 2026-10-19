# -*- coding: utf-8 -*-
"""
第一層驗證：ERC 理解能力評估 harness。

對應計畫書 E 節第一層「理解能力驗證」：
  - 指標：Weighted-F1（主，因情緒類別嚴重不平衡）、Macro-F1、Accuracy。
  - 消融：--ablation context 比較「有/無 Contextual Buffer」；
          --ablation stage  比較「兩階段 M-CoT vs 單次直接輸出」。
  - 另回報「幻覺率」(無法解析出合法標籤的比例)，量化小模型幻覺（F 節困難點）。

資料來源（--dataset）：
  - builtin : 內建小型雙語煙霧測試集（驗證 pipeline 是否跑通，非正式結果）。
  - csv     : 自備 CSV，需 --data_path，欄位見 load_csv()。
  - meld    : 嘗試自 HF 載入 MELD 文字版（Poria et al., 2019）。

用法：
  python src/evaluation/eval_erc.py --model qwen2.5-1.5b --dataset builtin
  python src/evaluation/eval_erc.py --model qwen2.5-3b --quant 4bit --dataset meld --limit 200 --ablation context
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.reasoning.backbone import Backbone, BackboneConfig
from src.reasoning.erc_engine import ERCEngine
from src.reasoning.labels import CANONICAL_EMOTIONS, normalize_label


# --------------------------------------------------------------------------- #
# 資料載入：統一格式 = list[conversation]，conversation = list[utterance dict]
#   utterance = {"speaker": str, "text": str, "emotion": canonical-label}
# --------------------------------------------------------------------------- #
def load_builtin():
    """內建小型雙語 ERC 煙霧測試集（含一個反諷例）。僅供 pipeline 驗證。"""
    return [
        [
            {"speaker": "A", "text": "I studied all week for this exam.", "emotion": "neutral"},
            {"speaker": "B", "text": "So how did it go?", "emotion": "neutral"},
            {"speaker": "A", "text": "I failed. Completely failed.", "emotion": "sadness"},
            {"speaker": "B", "text": "Oh no, I'm so sorry to hear that.", "emotion": "sadness"},
        ],
        [
            {"speaker": "A", "text": "Guess what, I got the internship!", "emotion": "joy"},
            {"speaker": "B", "text": "That's amazing, congratulations!", "emotion": "joy"},
            {"speaker": "A", "text": "Wait, there's a spider on your shoulder!", "emotion": "fear"},
            {"speaker": "B", "text": "What?! Get it off, get it off!", "emotion": "fear"},
        ],
        [
            {"speaker": "A", "text": "You broke my laptop again.", "emotion": "anger"},
            {"speaker": "B", "text": "Oh great, yeah, I just LOVE cleaning up your messes.", "emotion": "anger"},  # 反諷：字面 love 實為憤怒
            {"speaker": "A", "text": "這也太扯了吧，怎麼會這樣。", "emotion": "surprise"},
            {"speaker": "B", "text": "這鮭魚壽司放到現在，聞起來好噁心。", "emotion": "disgust"},
        ],
    ]


def load_csv(path: str, dataset_name: str = "meld"):
    """
    CSV 欄位（大小寫不拘）：dialogue_id, speaker, utterance/text, emotion
    以 dialogue_id 分組成對話，emotion 依 dataset_name 映射到 canonical。
    """
    import csv
    convs = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = {c.lower(): c for c in reader.fieldnames}
        did_c = cols.get("dialogue_id") or cols.get("dialogueid") or cols.get("conv_id")
        spk_c = cols.get("speaker")
        txt_c = cols.get("utterance") or cols.get("text") or cols.get("sentence")
        emo_c = cols.get("emotion") or cols.get("label")
        for row in reader:
            did = row[did_c] if did_c else "0"
            emo = normalize_label(row[emo_c], dataset_name) if emo_c else None
            convs.setdefault(did, []).append({
                "speaker": row[spk_c] if spk_c else "Speaker",
                "text": row[txt_c],
                "emotion": emo,
            })
    return list(convs.values())


def load_meld(split="test"):
    """
    自 HF 下載 MELD 文字版 CSV（ajyy/MELD_audio 的 {split}.csv，只取 CSV 不取音檔），
    依 Dialogue_ID 分組、按原始列序（＝Utterance_ID 序）保留對話順序。
    """
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download("ajyy/MELD_audio", f"{split}.csv", repo_type="dataset")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"無法自 HF 下載 MELD {split}.csv：{e}。"
            "可改用 --dataset csv --data_path <MELD_csv>（欄位 Dialogue_ID/Speaker/Utterance/Emotion）。"
        ) from e
    print(f"[meld] 使用 ajyy/MELD_audio {split}.csv → {path}")
    return load_csv(path, "meld")


def load_zh_dialogue(split="test", test_ratio=0.15, seed=42):
    """
    繁體中文（台灣）單句情緒資料集 Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset。
    無官方 train/test 切分，故以固定 seed 決定性切分，確保 format(train) 與 eval(test) 一致。
    單句無語境 → 每句視為一段「只有一句的對話」。標籤依 zh_dialogue 映射到 canonical，
    無法對應者（關切/疑問）normalize 回 None，後續自動過濾。
    """
    import csv
    import random
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset",
                           "data.csv", repo_type="dataset")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    random.Random(seed).shuffle(rows)
    n_test = int(len(rows) * test_ratio)
    rows = rows[n_test:] if split == "train" else rows[:n_test]
    print(f"[zh] Johnson8187 繁中資料集 split={split}：{len(rows)} 句")
    # 每句 = 一段單句對話
    return [[{"speaker": "Speaker",
              "text": r["text"],
              "emotion": normalize_label(r["emotion"], "zh_dialogue")}]
            for r in rows]


def load_cped(split="test", to_traditional=True):
    """
    CPED（scutcyr/CPED）：對話式簡體中文情緒資料集，含 Dialogue_ID/Utterance_ID/Speaker。
    - 自 GitHub 下載 {train,valid,test}_split.csv（首次下載後快取到 data/cped/）。
    - OpenCC s2twp 簡→繁（台灣用語）。
    - 依 Dialogue_ID 分組、Utterance_ID 排序 → 保留對話語境（這正是 ERC 重點）。
    - Emotion 依 labels 的 "cped" 映射到 canonical；曖昧類別回 None → 自動過濾。
    """
    import csv
    import urllib.request

    fname = {"train": "train_split.csv", "valid": "valid_split.csv",
             "test": "test_split.csv"}[split]
    cache_dir = PROJECT_ROOT / "data" / "cped"
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / fname
    if not local.exists():
        url = f"https://raw.githubusercontent.com/scutcyr/CPED/main/data/CPED/{fname}"
        print(f"[cped] 下載 {url} …")
        req = urllib.request.Request(url, headers={"User-Agent": "vms"})
        local.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    rows = list(csv.DictReader(open(local, encoding="utf-8-sig")))

    convert = (lambda s: s)
    if to_traditional:
        from opencc import OpenCC
        cc = OpenCC("s2twp")
        convert = cc.convert

    # 依 Dialogue_ID 分組，保留原始列序（＝Utterance_ID 序）
    convs = {}
    for r in rows:
        did = r.get("Dialogue_ID", "0")
        convs.setdefault(did, []).append({
            "speaker": convert(r.get("Speaker", "Speaker")),
            "text": convert(r.get("Utterance", "")),
            "emotion": normalize_label(r.get("Emotion"), "cped"),
        })
    print(f"[cped] split={split}：{len(convs)} 段對話、{len(rows)} 句"
          f"（已{'轉繁' if to_traditional else '保持簡體'}）")
    return list(convs.values())


# --------------------------------------------------------------------------- #
# 評估
# --------------------------------------------------------------------------- #
def run_eval(engine, conversations, limit=None):
    """對每句用其前文當歷史，逐句預測。回傳 (y_true, y_pred, records)。"""
    y_true, y_pred, records = [], [], []
    count = 0
    for conv in conversations:
        history = []
        for utt in conv:
            gold = utt.get("emotion")
            if gold in CANONICAL_EMOTIONS:  # 只評估能對應到 canonical 的句子
                res = engine.predict(utt["text"], history=history)
                # 無法解析的預測記為 "invalid"，仍納入計分（等同答錯，反映幻覺代價）
                pred = res.predicted if res.parse_ok else "invalid"
                y_true.append(gold)
                y_pred.append(pred)
                records.append({
                    "text": utt["text"], "gold": gold, "pred": pred,
                    "parse_ok": res.parse_ok, "rationale": res.rationale[:200],
                    "raw_label": res.raw_label_output[:120],
                })
                count += 1
            history.append({"role": utt.get("speaker", "Speaker"), "content": utt["text"]})
            if limit and count >= limit:
                break
        if limit and count >= limit:
            break
    return y_true, y_pred, records


def compute_metrics(y_true, y_pred):
    from sklearn.metrics import f1_score, accuracy_score, classification_report
    labels = CANONICAL_EMOTIONS
    return {
        "n": len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0), 4),
        "macro_f1": round(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4),
        "report": classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True),
    }


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-1.5b")
    ap.add_argument("--quant", default=None, choices=[None, "4bit", "8bit"])
    ap.add_argument("--dataset", default="builtin", choices=["builtin", "csv", "meld", "zh", "cped"])
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None, help="最多評估幾句（省時）")
    ap.add_argument("--ablation", default=None, choices=[None, "context", "stage"],
                    help="context: 比較有無對話歷史；stage: 比較兩階段 vs 單次")
    ap.add_argument("--lora", default=None, help="LoRA adapter 路徑（評估微調後模型）")
    ap.add_argument("--single", action="store_true",
                    help="用單次直出標籤（不做兩階段）。評估 LoRA 微調結果時必用，"
                         "因訓練樣本即單次格式，train/test 才一致")
    ap.add_argument("--out_dir", default=str(PROJECT_ROOT / "evaluation_results"))
    args = ap.parse_args()

    # 載入資料
    if args.dataset == "builtin":
        conversations = load_builtin()
    elif args.dataset == "csv":
        conversations = load_csv(args.data_path, "meld")
    elif args.dataset == "zh":
        conversations = load_zh_dialogue(args.split)
    elif args.dataset == "cped":
        conversations = load_cped(args.split)
    else:
        conversations = load_meld(args.split)
    n_utt = sum(len(c) for c in conversations)
    print(f"[data] {len(conversations)} 段對話，共 {n_utt} 句")

    # 載入模型（可選 LoRA adapter）
    bk = Backbone(BackboneConfig(model=args.model, quantization=args.quant,
                                 lora_path=args.lora))

    two_stage = not args.single   # --single 時單次直出，與 LoRA 訓練格式一致

    # 決定要跑哪些設定（消融）
    if args.ablation == "context":
        settings = [("context_on", dict(use_context=True, two_stage=two_stage)),
                    ("context_off", dict(use_context=False, two_stage=two_stage))]
    elif args.ablation == "stage":
        settings = [("two_stage_mcot", dict(use_context=True, two_stage=True)),
                    ("single_pass", dict(use_context=True, two_stage=False))]
    else:
        settings = [("default", dict(use_context=True, two_stage=two_stage))]

    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model, "quant": args.quant,
        "dataset": args.dataset, "limit": args.limit,
        "settings": {},
    }

    for name, kw in settings:
        print(f"\n===== 設定：{name} ({kw}) =====")
        engine = ERCEngine(bk, **kw)
        y_true, y_pred, records = run_eval(engine, conversations, limit=args.limit)
        metrics = compute_metrics(y_true, y_pred)
        metrics["hallucination_rate"] = round(engine.stats.hallucination_rate, 4)
        report["settings"][name] = {"metrics": metrics, "records": records}
        print(f"  Weighted-F1={metrics['weighted_f1']}  Macro-F1={metrics['macro_f1']}  "
              f"Acc={metrics['accuracy']}  幻覺率={metrics['hallucination_rate']}  (n={metrics['n']})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}_{args.dataset}" + (f"_{args.ablation}" if args.ablation else "") \
          + ("_lora" if args.lora else "")   # 避免 before/after 報告互相覆蓋
    out = out_dir / f"erc_report_{tag}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = out_dir / f"erc_report_{tag}.md"
    write_markdown(report, md)
    print(f"\n[done] JSON → {out}")
    print(f"[done] 報告 → {md}")


def write_markdown(report: dict, path: Path):
    """把評估結果寫成 proposal-ready 的 Markdown 表格。"""
    L = [f"# 第一層驗證：ERC 理解能力評估\n",
         f"- 時間：{report['timestamp']}",
         f"- 模型：`{report['model']}`（量化：{report['quant']}）",
         f"- 資料集：{report['dataset']}　評估句數上限：{report['limit']}\n",
         "## 結果總表\n",
         "| 設定 | n | Weighted-F1 | Macro-F1 | Accuracy | 幻覺率 |",
         "|---|---|---|---|---|---|"]
    for name, s in report["settings"].items():
        m = s["metrics"]
        L.append(f"| {name} | {m['n']} | {m['weighted_f1']} | {m['macro_f1']} "
                 f"| {m['accuracy']} | {m['hallucination_rate']} |")

    # 消融差異摘要
    names = list(report["settings"].keys())
    if len(names) == 2:
        a, b = names
        ma, mb = report["settings"][a]["metrics"], report["settings"][b]["metrics"]
        d = round(ma["weighted_f1"] - mb["weighted_f1"], 4)
        L.append(f"\n**消融差異**：{a} 相對 {b} 的 Weighted-F1 變化 = **{d:+}**")

    # 各類別 F1（取第一個設定）
    first = report["settings"][names[0]]["metrics"]["report"]
    L.append("\n## 各情緒類別表現（設定：%s）\n" % names[0])
    L.append("| 情緒 | precision | recall | f1 | support |")
    L.append("|---|---|---|---|---|")
    for emo in CANONICAL_EMOTIONS:
        if emo in first:
            r = first[emo]
            L.append(f"| {emo} | {r['precision']:.3f} | {r['recall']:.3f} "
                     f"| {r['f1-score']:.3f} | {int(r['support'])} |")
    L.append("\n> 註：Weighted-F1 為主指標（情緒類別不平衡）；幻覺率＝模型輸出無法解析為合法標籤的比例。")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()

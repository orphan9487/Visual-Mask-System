# -*- coding: utf-8 -*-
"""
真實台灣聊天測試集：語料整理 → 雙人標註一致度 → 合併 gold。

為什麼要有這份測試集：模型所有分數都量在 CPED（大陸劇台詞、簡轉繁）上，
產品實際面對的是台灣人在聊天室打字。這份小測試集回答兩件事：
  1. 模型在真實聊天上表現如何（領域落差有多大）。
  2. 人類彼此的一致度（Cohen's κ）＝這個任務的天花板，用來判斷還值不值得調模型。

標籤用競賽 ERC 的 7 類 canonical（與部署模型同一套，才能直接比較）。
src/annotation/schema.py 的產品細層 schema 是另一套標籤空間，這裡不混用。

流程：
  1) 整理語料（聊天室錄製檔、純文字逐字稿，或 ptt_crawler 抓的論壇推文），發話者匿名成 A/B/C、遮蔽電話/email/網址：
     python -m src.annotation.chat_testset build \\
         --record data/chat_testset/recordings.jsonl --txt data/chat_testset/raw/*.txt \\
         --out data/chat_testset/items.jsonl
  2) 兩位標註者各自用 src/annotation/chat_annotator.html 標註，各匯出一個 labels_*.json
  3) 算一致度，並輸出分歧清單供兩人討論裁決（填 final 欄）：
     python -m src.annotation.chat_testset agree data/chat_testset/labels_a.json \\
         data/chat_testset/labels_b.json --adjudicate data/chat_testset/adjudicate.jsonl
  4) 合併 gold（一致者直接採用 + 裁決結果）：
     python -m src.annotation.chat_testset gold data/chat_testset/labels_a.json \\
         data/chat_testset/labels_b.json --adjudicate data/chat_testset/adjudicate.jsonl \\
         --out data/chat_testset/gold.jsonl
  5) 評估模型：python -m src.evaluation.eval_chat_testset（見該檔說明）

語料屬個資：只收錄參與者知情同意的對話，檔案一律放 data/chat_testset/（已在 .gitignore）。
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from src.reasoning.labels import CANONICAL_EMOTIONS as LABELS

# --------------------------------------------------------------------------- #
# 1. 語料整理
# --------------------------------------------------------------------------- #

_PII = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[email]"),
    (re.compile(r"https?://\S+|www\.\S+"), "[網址]"),
    (re.compile(r"(?:\+?886[-\s]?|0)9\d{2}[-\s]?\d{3}[-\s]?\d{3}"), "[電話]"),
    (re.compile(r"\b[A-Z][12]\d{8}\b"), "[身分證]"),
]
_TIME_PREFIX = re.compile(r"^\s*(?:上午|下午)?\s*\d{1,2}:\d{2}(?::\d{2})?\s*")
_SPLIT = re.compile(r"^(?P<name>[^:：\t]{1,20}?)\s*[:：\t]\s*(?P<text>.*)$")


def mask_pii(text: str) -> str:
    for pat, repl in _PII:
        text = pat.sub(repl, text)
    return text


def parse_transcript(raw: str) -> list[list[dict]]:
    """純文字逐字稿 → 對話清單。每行「名字：內容」（可帶 HH:MM 前綴、冒號或 tab 分隔）；
    空行或 --- 分隔不同對話；不符格式的行視為上一則訊息的換行續寫。"""
    convs, cur = [], []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s == "---":
            if cur:
                convs.append(cur)
                cur = []
            continue
        m = _SPLIT.match(_TIME_PREFIX.sub("", s))
        if m and m.group("text").strip():
            cur.append({"speaker": m.group("name").strip(), "text": m.group("text").strip()})
        elif cur:
            cur[-1]["text"] += " " + s
    if cur:
        convs.append(cur)
    return convs


def split_recording(rows: list[dict], gap_minutes: float) -> list[list[dict]]:
    """聊天室錄製檔 → 對話清單：依 session 分組，同 session 內相隔超過 gap 分鐘就切開。"""
    by_session = defaultdict(list)
    for r in rows:
        by_session[r["session"]].append(r)
    convs = []
    for sess in sorted(by_session):
        cur, last_ts = [], None
        for r in sorted(by_session[sess], key=lambda r: r["ts"]):
            if cur and last_ts is not None and r["ts"] - last_ts > gap_minutes * 60:
                convs.append(cur)
                cur = []
            cur.append({"speaker": r["speaker"], "text": r["text"]})
            last_ts = r["ts"]
        if cur:
            convs.append(cur)
    return convs


def anonymize(conv: list[dict]) -> list[dict]:
    """同一段對話內把發話者依出現順序換成 A、B、C…（標註者與模型都只看到代號）。
    內文提到其他參與者的名字／帳號（如「樓上 beef68 說的」）也換成代號。"""
    names: dict[str, str] = {}
    for u in conv:
        if u["speaker"] not in names:
            i = len(names)
            names[u["speaker"]] = chr(ord("A") + i) if i < 26 else f"P{i + 1}"
    # 長名字先換，避免「小明明」被「小明」先吃掉一部分；一字名太容易誤傷一般用字，不換
    mentions = sorted((n for n in names if len(n) >= 2), key=len, reverse=True)
    out = []
    for u in conv:
        text = mask_pii(u["text"])
        for n in mentions:
            text = text.replace(n, names[n])
        out.append({**u, "speaker": names[u["speaker"]], "text": text})
    return out


def build_items(convs: list[list[dict]], min_turns: int = 2, keep_names: bool = False,
                sources: list[str] | None = None) -> list[dict]:
    """對話清單 → 待標註題目。turn 帶 context_only=True 者只當語境（如論壇文章本文），
    不列為標註題；min_turns 只計可標註的句數。sources 為各對話的來源標記（如 ptt/Boy-Girl）。"""
    items = []
    sources = sources or [None] * len(convs)
    kept = [(c, src) for c, src in zip(convs, sources)
            if sum(not u.get("context_only") for u in c) >= min_turns]
    for ci, (conv, src) in enumerate(kept, 1):
        conv = [{**u, "text": mask_pii(u["text"])} for u in conv]             if keep_names else anonymize(conv)
        for ti, u in enumerate(conv):
            it = {"id": f"c{ci:03d}-t{ti:03d}", "conv": f"c{ci:03d}", "turn": ti,
                  "speaker": u["speaker"], "text": u["text"]}
            if u.get("context_only"):
                it["context_only"] = True
            if src:
                it["source"] = src
            items.append(it)
    return items


def load_items(path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def history_for(items: list[dict]) -> dict[str, list[dict]]:
    """item id → 同一段對話中「這句之前」的全部發言（{role, content}，與聊天室語境同格式）。"""
    by_conv = defaultdict(list)
    for it in items:
        by_conv[it["conv"]].append(it)
    hist = {}
    for conv in by_conv.values():
        conv.sort(key=lambda it: it["turn"])
        for i, it in enumerate(conv):
            hist[it["id"]] = [{"role": p["speaker"], "content": p["text"]} for p in conv[:i]]
    return hist


# --------------------------------------------------------------------------- #
# 2. 標註檔與一致度
# --------------------------------------------------------------------------- #


def expand_paths(patterns: list[str]) -> list[str]:
    """展開萬用字元（PowerShell 不會替你展開 labels_*.json）。"""
    out = []
    for p in patterns:
        out += sorted(glob.glob(p)) or [p]
    return out


def load_labels(path) -> tuple[str, dict[str, dict]]:
    """標註網頁匯出的 JSON → (標註者名稱, {item id: {label, sarcasm, unsure}})。"""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    labels = {k: v for k, v in d["labels"].items() if v.get("label") in LABELS}
    return d.get("annotator") or Path(path).stem, labels


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in LABELS) / (n * n)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def bootstrap_ci(fn, n: int, iters: int = 1000, seed: int = 0) -> tuple[float, float]:
    """對「題目」重抽樣的 95% 信賴區間；fn(idx) 回傳該樣本的統計量。"""
    rng = np.random.default_rng(seed)
    vals = [fn(rng.integers(0, n, n)) for _ in range(iters)]
    vals = [v for v in vals if v == v]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (
        float("nan"), float("nan"))


def interpret_kappa(k: float) -> str:
    """Landis & Koch (1977) 分級。"""
    if k != k:
        return "無法計算"
    for th, name in [(0.8, "幾乎完全一致"), (0.6, "高度一致"), (0.4, "中度一致"),
                     (0.2, "尚可"), (0.0, "輕微")]:
        if k > th:
            return name
    return "比隨機還差"


def confusion(a: list[str], b: list[str]) -> list[list[int]]:
    idx = {k: i for i, k in enumerate(LABELS)}
    m = [[0] * len(LABELS) for _ in LABELS]
    for x, y in zip(a, b):
        m[idx[x]][idx[y]] += 1
    return m


def print_confusion(m, row_name: str, col_name: str) -> None:
    short = [k[:4] for k in LABELS]
    print(f"  列={row_name}  欄={col_name}")
    print("        " + " ".join(f"{s:>5}" for s in short))
    for s, row in zip(short, m):
        print(f"  {s:>5} " + " ".join(f"{v:>5}" for v in row))


def agreement_report(named: list[tuple[str, dict]], items: dict | None = None) -> dict:
    """兩兩 Cohen's κ（含 bootstrap CI）、觀察一致率、各類一致情形。"""
    report = {"pairs": []}
    for i in range(len(named)):
        for j in range(i + 1, len(named)):
            (na, la), (nb, lb) = named[i], named[j]
            common = sorted(set(la) & set(lb))
            a = [la[k]["label"] for k in common]
            b = [lb[k]["label"] for k in common]
            k = cohen_kappa(a, b)
            lo, hi = bootstrap_ci(lambda idx: cohen_kappa([a[t] for t in idx], [b[t] for t in idx]),
                                  len(common))
            po = sum(x == y for x, y in zip(a, b)) / len(common) if common else float("nan")
            print(f"\n[{na} vs {nb}] 共同標註 {len(common)} 則")
            print(f"  觀察一致率 {po:.3f}   Cohen's κ = {k:.3f} [{lo:.3f}–{hi:.3f}]（{interpret_kappa(k)}）")
            # 各類：以「兩人任一標了這類」為分母的一致比例（Jaccard 式），看哪類最難
            print("  各類一致（兩人都標 / 任一人標）：")
            for lab in LABELS:
                both = sum(x == lab and y == lab for x, y in zip(a, b))
                either = sum(x == lab or y == lab for x, y in zip(a, b))
                if either:
                    print(f"    {lab:<9}{both:>4} / {either:<4} = {both / either:.2f}")
            print_confusion(confusion(a, b), na, nb)
            report["pairs"].append({"a": na, "b": nb, "n": len(common), "observed": po,
                                    "kappa": k, "kappa_ci": [lo, hi]})
    if len(named) > 2:
        from src.annotation.iaa import fleiss_kappa
        ids = sorted(set.union(*(set(l) for _, l in named)))
        ratings = [[l[k]["label"] for _, l in named if k in l] for k in ids]
        fk = fleiss_kappa(ratings, LABELS)
        print(f"\n[全體] Fleiss' κ = {fk:.3f}（{interpret_kappa(fk)}）")
        report["fleiss"] = fk
    return report


def write_adjudication(named: list[tuple[str, dict]], items: list[dict], out) -> int:
    """把有分歧的題目寫成 JSONL，final 欄留空給兩人討論後填（填 7 類英文標籤之一）。"""
    by_id = {it["id"]: it for it in items}
    hist = history_for(items)
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for iid in sorted(set.union(*(set(l) for _, l in named))):
            labs = {name: l[iid]["label"] for name, l in named if iid in l}
            if len(labs) < 2 or len(set(labs.values())) == 1:
                continue
            it = by_id.get(iid, {})
            ctx = [f"{h['role']}: {h['content']}" for h in hist.get(iid, [])[-6:]]
            f.write(json.dumps({"id": iid, "context": ctx,
                                "utterance": f"{it.get('speaker', '?')}: {it.get('text', '')}",
                                "labels": labs, "final": "", "sarcasm": None, "note": ""},
                               ensure_ascii=False) + "\n")
            n += 1
    return n


def merge_gold(named: list[tuple[str, dict]], adjudicated: dict[str, dict]) -> tuple[list[dict], Counter]:
    """全員一致 → 直接採用；有分歧 → 用裁決檔的 final；沒裁決的排除（並計數回報）。
    反諷旗標：至少一半標註者勾選，或裁決時明確指定。"""
    gold, stats = [], Counter()
    for iid in sorted(set.union(*(set(l) for _, l in named))):
        votes = [l[iid] for _, l in named if iid in l]
        if len(votes) < 2:
            stats["只有一人標"] += 1
            continue
        sarcasm = sum(bool(v.get("sarcasm")) for v in votes) * 2 >= len(votes)
        labs = {v["label"] for v in votes}
        if len(labs) == 1:
            gold.append({"id": iid, "label": votes[0]["label"], "sarcasm": sarcasm,
                         "source": "agree"})
            stats["一致"] += 1
            continue
        adj = adjudicated.get(iid, {})
        if adj.get("final") in LABELS:
            if adj.get("sarcasm") is not None:
                sarcasm = bool(adj["sarcasm"])
            gold.append({"id": iid, "label": adj["final"], "sarcasm": sarcasm,
                         "source": "adjudicated"})
            stats["裁決"] += 1
        else:
            stats["分歧未裁決（排除）"] += 1
    return gold, stats


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _cmd_build(args) -> None:
    convs = []
    if args.record:
        rows = load_items(args.record)
        convs += split_recording(rows, args.gap_minutes)
        print(f"[build] 錄製檔 {len(rows)} 則 → {len(convs)} 段對話")
    sources = [None] * len(convs)
    for path in args.forum or []:
        for art in load_items(path):
            # 推／噓標記會洩漏情緒，不交給標註者與模型；網址也不帶進題目檔
            convs.append([{"speaker": t["speaker"], "text": t["text"],
                           **({"context_only": True} if t.get("context_only") else {})}
                          for t in art["turns"]])
            sources.append(art.get("source"))
        print(f"[build] 論壇檔 {path}：累計 {len(convs)} 段對話")
    for pattern in args.txt or []:
        for path in sorted(glob.glob(pattern)):
            got = parse_transcript(Path(path).read_text(encoding="utf-8"))
            print(f"[build] {path}: {len(got)} 段對話")
            convs += got
            sources += [None] * len(got)
    items = build_items(convs, args.min_turns, args.keep_names, sources)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    n_convs = len({it["conv"] for it in items})
    n_task = sum(not it.get("context_only") for it in items)
    print(f"[build] {n_convs} 段對話、{n_task} 則待標註（另 {len(items) - n_task} 則只當語境）→ {args.out}")
    print("[build] 發給標註者前請自行瀏覽一遍，刪掉不該外流的內容（自動遮蔽只抓電話/email/網址/身分證）")


def _cmd_agree(args) -> None:
    named = [load_labels(p) for p in args.labels]
    report = agreement_report(named)
    if args.adjudicate:
        items = load_items(args.items)
        n = write_adjudication(named, items, args.adjudicate)
        print(f"\n[agree] {n} 則有分歧 → {args.adjudicate}（兩人討論後在 final 欄填標籤）")
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _cmd_gold(args) -> None:
    named = [load_labels(p) for p in args.labels]
    adjudicated = {}
    if args.adjudicate and Path(args.adjudicate).exists():
        adjudicated = {r["id"]: r for r in load_items(args.adjudicate)}
    gold, stats = merge_gold(named, adjudicated)
    with open(args.out, "w", encoding="utf-8") as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print(f"[gold] {len(gold)} 則 → {args.out}　組成：{dict(stats)}")
    print(f"[gold] 標籤分布：{dict(Counter(g['label'] for g in gold))}"
          f"　反諷 {sum(g['sarcasm'] for g in gold)} 則")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="整理語料成待標註題目")
    b.add_argument("--record", help="聊天室錄製檔（VMS_CHAT_RECORD 產生的 JSONL）")
    b.add_argument("--txt", nargs="*", help="純文字逐字稿（可用萬用字元）")
    b.add_argument("--forum", nargs="*", help="論壇爬蟲輸出（ptt_crawler 產生的 JSONL）")
    b.add_argument("--gap_minutes", type=float, default=30, help="錄製檔中相隔多久算新對話")
    b.add_argument("--min_turns", type=int, default=2, help="少於幾句的對話不收")
    b.add_argument("--keep_names", action="store_true", help="保留原始名字（預設匿名成 A/B/C）")
    b.add_argument("--out", default="data/chat_testset/items.jsonl")
    b.set_defaults(fn=_cmd_build)

    a = sub.add_parser("agree", help="算標註者一致度、輸出分歧清單")
    a.add_argument("labels", nargs="+")
    a.add_argument("--items", default="data/chat_testset/items.jsonl")
    a.add_argument("--adjudicate", help="分歧清單輸出路徑")
    a.add_argument("--out", help="一致度報告 JSON")
    a.set_defaults(fn=_cmd_agree)

    g = sub.add_parser("gold", help="合併成 gold 標籤")
    g.add_argument("labels", nargs="+")
    g.add_argument("--adjudicate", help="填好 final 的分歧清單")
    g.add_argument("--out", default="data/chat_testset/gold.jsonl")
    g.set_defaults(fn=_cmd_gold)

    args = ap.parse_args(argv)
    if args.cmd in ("agree", "gold"):
        args.labels = expand_paths(args.labels)
    if args.cmd in ("agree", "gold") and len(args.labels) < 2:
        ap.error("至少需要兩位標註者的標註檔")
    args.fn(args)


if __name__ == "__main__":
    main()

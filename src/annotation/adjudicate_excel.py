# -*- coding: utf-8 -*-
"""
分歧清單 ⇄ Excel：讓標註者在 Excel 裡討論、填最終標籤，再寫回 adjudicate.jsonl。

  export：adjudicate.jsonl（＋各標註者的 labels 檔，取反諷／不確定勾選）→ .xlsx
          「最終標籤」「是否反諷」為下拉選單；前文只取最後幾句，過長的原文截斷
  import：填好的 .xlsx → 把最終標籤、反諷、備註寫回 adjudicate.jsonl（原檔先備份成 .bak）

用法：
    python -m src.annotation.adjudicate_excel export data/chat_testset/forum_adjudicate.jsonl \\
        --labels data/chat_testset/forum_labels_*.json --out data/chat_testset/forum_裁決表.xlsx
    python -m src.annotation.adjudicate_excel import data/chat_testset/forum_裁決表.xlsx \\
        --adjudicate data/chat_testset/forum_adjudicate.jsonl
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from src.annotation.chat_testset import expand_paths, load_items, load_labels
from src.reasoning.labels import CANONICAL_EMOTIONS as L

ZH = {"neutral": "中性", "joy": "開心", "sadness": "悲傷", "anger": "憤怒",
      "surprise": "驚訝", "fear": "恐懼", "disgust": "厭惡"}
FROM_ZH = {v: k for k, v in ZH.items()}
YES, NO = "是", "否"
HEAD = ["序號", "題號", "前文（最後幾句）", "這一句"]
TAIL = ["最終標籤", "是否反諷", "備註"]


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "…"


def export(args) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    rows = load_items(args.adjudicate)
    named = [load_labels(p) for p in expand_paths(args.labels or [])]
    names = list(rows[0]["labels"]) if rows else []
    wb = Workbook()
    ws = wb.active
    ws.title = "裁決"
    cols = HEAD + [f"{n} 標籤" for n in names] + [f"{n} 勾選" for n in names] + TAIL
    ws.append(cols)
    flags = {n: lab for n, lab in named}
    for k, r in enumerate(rows, 1):
        ctx = r.get("context", [])[-args.context_turns:]
        ctx_text = "\n".join(_clip(c, args.clip) for c in ctx) or "（無前文）"
        marks = []
        for n in names:
            v = flags.get(n, {}).get(r["id"], {})
            marks.append("、".join(t for t, on in (("反諷", v.get("sarcasm")), ("不確定", v.get("unsure"))) if on))
        final = ZH.get(r.get("final") or "", "")
        sarc = "" if r.get("sarcasm") is None else (YES if r["sarcasm"] else NO)
        ws.append([k, r["id"], ctx_text, r.get("utterance", "")]
                  + [ZH.get(r["labels"].get(n, ""), r["labels"].get(n, "")) for n in names]
                  + marks + [final, sarc, r.get("note", "")])

    widths = [6, 12, 60, 40] + [10] * len(names) + [12] * len(names) + [12, 10, 30]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = w
    head_fill = PatternFill("solid", fgColor="1B1A18")
    edit_fill = PatternFill("solid", fgColor="E5F4E4")
    for c in ws[1]:
        c.font = Font(bold=True, color="F4F1EA")
        c.fill = head_fill
        c.alignment = Alignment(vertical="center", wrap_text=True)
    n_rows = len(rows) + 1
    first_edit = len(cols) - len(TAIL) + 1
    for row in ws.iter_rows(min_row=2, max_row=n_rows):
        for c in row:
            c.alignment = Alignment(vertical="top", wrap_text=True)
        for c in row[first_edit - 1:]:
            c.fill = edit_fill
    letter = lambda i: ws.cell(1, i).column_letter
    dv_label = DataValidation(type="list", formula1='"' + ",".join(ZH[k] for k in L) + '"', allow_blank=True)
    dv_sarc = DataValidation(type="list", formula1=f'"{YES},{NO}"', allow_blank=True)
    ws.add_data_validation(dv_label)
    ws.add_data_validation(dv_sarc)
    dv_label.add(f"{letter(first_edit)}2:{letter(first_edit)}{n_rows}")
    dv_sarc.add(f"{letter(first_edit + 1)}2:{letter(first_edit + 1)}{n_rows}")
    ws.freeze_panes = "E2"

    info = wb.create_sheet("說明")
    for line in [
        "裁決方式",
        "1. 兩位標註者一起逐題討論，依《標註準則》決定最終標籤；不要參考模型的答案。",
        "2. 只填綠色欄位：「最終標籤」（下拉選單）、「是否反諷」（是／否）、「備註」（可寫討論理由）。",
        "3. 真的無法決定的題目，「最終標籤」留空並在備註寫原因；留空的題目不會納入標準答案。",
        "4. 「勾選」欄是標註當時勾的反諷／不太確定，供討論參考。",
        "5. 不要增刪列、不要改題號；填完存檔後交回即可。",
    ]:
        info.append([line])
    info.column_dimensions["A"].width = 90
    info["A1"].font = Font(bold=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    wb.save(args.out)
    print(f"[export] {len(rows)} 則分歧 → {args.out}")


def import_(args) -> None:
    from openpyxl import load_workbook

    ws = load_workbook(args.xlsx, data_only=True)["裁決"]
    header = [c.value for c in ws[1]]
    col = {name: header.index(name) for name in ["題號"] + TAIL}
    filled = {}
    problems = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        iid = row[col["題號"]]
        if not iid:
            continue
        lab = (row[col["最終標籤"]] or "").strip()
        sarc = (row[col["是否反諷"]] or "").strip()
        if lab and lab not in FROM_ZH:
            problems.append(f"{iid}：最終標籤「{lab}」不在 7 類中")
            continue
        filled[iid] = {"final": FROM_ZH.get(lab, ""),
                       "sarcasm": None if not sarc else sarc == YES,
                       "note": str(row[col["備註"]] or "")}
    if problems:
        sys.exit("[import] 有無法辨識的標籤，請修正後再匯入：\n  " + "\n  ".join(problems))

    rows = load_items(args.adjudicate)
    shutil.copyfile(args.adjudicate, args.adjudicate + ".bak")
    done = 0
    with open(args.adjudicate, "w", encoding="utf-8") as f:
        for r in rows:
            if r["id"] in filled:
                r.update(filled[r["id"]])
            done += bool(r.get("final"))
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[import] 已裁決 {done}／{len(rows)} 則 → {args.adjudicate}（原檔備份為 .bak）")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="分歧清單 ⇄ Excel")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("adjudicate")
    e.add_argument("--labels", nargs="*", help="各標註者的 labels 檔（帶出反諷／不確定勾選）")
    e.add_argument("--out", required=True)
    e.add_argument("--context_turns", type=int, default=4)
    e.add_argument("--clip", type=int, default=120, help="每句前文最多顯示幾字")
    e.set_defaults(fn=export)
    i = sub.add_parser("import")
    i.add_argument("xlsx")
    i.add_argument("--adjudicate", required=True)
    i.set_defaults(fn=import_)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()

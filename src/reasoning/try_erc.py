# -*- coding: utf-8 -*-
"""
ERC 推理層獨立測試器（完全不經過 LINE / FastAPI）。

用途：直接在終端機測試情緒判讀，觀察兩階段推理鏈、語境影響、速度。
這是「把推理層單獨拉出來測」最直接的方式。

模式一：互動式 REPL（逐句輸入，會累積成對話語境）
    .venv\\Scripts\\python -m src.reasoning.try_erc --model breeze2-3b --quant fp16
    指令：
      直接打字            = 當作新的一句，會用先前輸入當語境
      /reset             = 清空對話語境
      /nocontext         = 切換：是否餵入語境（用來現場對照）
      /quit              = 離開

模式二：單句/單組（給 --text，可選 --history 用 || 分隔前文）
    .venv\\Scripts\\python -m src.reasoning.try_erc --text "真好啊" --history "你把我的資料庫刪光了"
"""

import argparse
import sys
import time
from pathlib import Path

# Windows 主控台預設 cp950，避免非該編碼字元導致輸出崩潰。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.reasoning.backbone import Backbone, BackboneConfig
from src.reasoning.erc_engine import ERCEngine
from src.reasoning.labels import EMOTION_ZH


def _norm_quant(q):
    q = (q or "").strip().lower()
    return None if q in ("", "none", "fp16", "float16", "no") else q


def show(res, elapsed):
    zh = EMOTION_ZH.get(res.predicted, res.predicted) if res.predicted else "(無法解析)"
    print(f"\n  情緒判讀 : {res.predicted}  ({zh})   [{elapsed:.2f}s, "
          f"parse_ok={res.parse_ok}, context={res.used_context}]")
    if res.rationale:
        print(f"  推理鏈   : {res.rationale.strip()}")
    print(f"  原始標籤輸出: {res.raw_label_output.strip()!r}\n")


def build_engine(args):
    bk = Backbone(BackboneConfig(model=args.model, quantization=_norm_quant(args.quant)))
    return ERCEngine(bk, use_context=not args.no_context, two_stage=not args.single)


def run_once(args):
    engine = build_engine(args)
    history = [{"role": "Speaker", "content": h}
               for h in (args.history.split("||") if args.history else []) if h.strip()]
    if history:
        print("前文語境：")
        for h in history:
            print(f"  - {h['content']}")
    t = time.time()
    res = engine.predict(args.text, history=history)
    show(res, time.time() - t)


def run_repl(args):
    engine = build_engine(args)
    history = []
    print("=" * 60)
    print(f"ERC 互動測試器  模型={args.model} 量化={_norm_quant(args.quant)} "
          f"兩階段={not args.single} 用語境={engine.use_context}")
    print("指令：/reset 清語境  /nocontext 切換語境  /quit 離開")
    print("=" * 60)
    while True:
        try:
            line = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再見")
            break
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/reset":
            history = []
            print("  (已清空語境)")
            continue
        if line == "/nocontext":
            engine.use_context = not engine.use_context
            print(f"  (用語境 = {engine.use_context})")
            continue

        t = time.time()
        res = engine.predict(line, history=history)
        show(res, time.time() - t)
        history.append({"role": "You", "content": line})  # 這句成為之後的語境


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="breeze2-3b")
    ap.add_argument("--quant", default="fp16", help="4bit / 8bit / fp16(不量化)")
    ap.add_argument("--single", action="store_true", help="用單次直出(對照)，而非兩階段")
    ap.add_argument("--no_context", action="store_true", help="不餵入語境")
    ap.add_argument("--text", default=None, help="單句模式：要測的句子")
    ap.add_argument("--history", default=None, help="單句模式：前文，用 || 分隔")
    args = ap.parse_args()

    if args.text:
        run_once(args)
    else:
        run_repl(args)


if __name__ == "__main__":
    main()

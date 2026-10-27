# -*- coding: utf-8 -*-
"""
PTT 公開看板 → 真實聊天測試集的「論壇補充子集」（主要用來測反諷、酸文）。

為什麼抓推文：推文短、你來我往、常帶情緒和反話，形式最接近聊天；
文章本身（標題＋內文）只當語境（context_only），不標註。

資料處理原則（寫在程式裡，不靠人記得）：
  - 只抓不需登入、不需年齡確認的公開看板；遇到 over18 驗證頁就跳過該板，不繞過。
  - 每次請求至少間隔 --delay 秒（預設 1.5），User-Agent 表明是研究用途。
  - 推／噓／→ 標記不交給標註者與模型（等於洩漏情緒線索），只留在原始檔供分析。
  - 推文的 IP 與時間不保存；帳號在 build 階段匿名成 A/B/C，內文提到的帳號也一併替換。
  - 原始檔含網址與帳號，只存在 data/chat_testset/（已 gitignore），不公開散布。

用法：
    .venv\\Scripts\\python -m src.annotation.ptt_crawler --boards Boy-Girl WomenTalk marriage \\
        --articles_per_board 6 --max_turns 15 --out data/chat_testset/forum_raw.jsonl
    .venv\\Scripts\\python -m src.annotation.chat_testset build \\
        --forum data/chat_testset/forum_raw.jsonl --out data/chat_testset/forum_items.jsonl
"""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

BASE = "https://www.ptt.cc"
USER_AGENT = "Mozilla/5.0 (compatible; VMS-ERC-research/0.1; academic emotion-annotation study)"
DEFAULT_BOARDS = ["Boy-Girl", "WomenTalk", "marriage", "Tech_Job", "joke"]

_ARTICLE_LINK = re.compile(r'<div class="title">\s*<a href="(/bbs/[\w-]+/M\.\d+\.A\.[0-9A-F]+\.html)">([^<]*)</a>')
_PREV_PAGE = re.compile(r'href="/bbs/[\w-]+/index(\d+)\.html">&lsaquo; 上頁')
_PUSH = re.compile(
    r'<div class="push"><span class="[^"]*push-tag">([^<]*)</span>'
    r'<span class="[^"]*push-userid">([^<]*)</span>'
    r'<span class="[^"]*push-content">([^<]*)</span>'
    r'<span class="push-ipdatetime">([^<]*)</span>')
_META = re.compile(r'<span class="article-meta-tag">(\S+)</span><span class="article-meta-value">([^<]*)</span>')
_TAGS = re.compile(r"<[^>]+>")
# 只表示「推／噓」本身、沒有內容可判斷的推文（如「推」「+1」「噓」）→ 不收
_FILLER = re.compile(r"^(推+|噓+|\+1|1樓|頭香|先推|朝聖|[.。…]+)$")
_SKIP_TITLES = re.compile(r"\[(公告|協尋|版務|徵求)\]|^\(本文已被刪除\)")


class Fetcher:
    """限速抓取：任兩次請求至少間隔 delay 秒（含些微隨機，避免固定節奏）。"""

    def __init__(self, delay: float):
        self.delay = delay
        self._last = 0.0
        self.n_requests = 0

    def get(self, path: str) -> str:
        wait = self.delay + random.uniform(0, 0.5) - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(BASE + path, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        self._last = time.time()
        self.n_requests += 1
        return body


def is_age_gated(page: str) -> bool:
    """需年齡確認的看板（如 Gossiping）頁面會帶 over18 檢查；一般看板完全沒有這個字。"""
    return "over18" in page


def parse_index(page: str) -> tuple[list[tuple[str, str]], int | None]:
    links = [(href, html.unescape(t).strip()) for href, t in _ARTICLE_LINK.findall(page)]
    prev = _PREV_PAGE.search(page)
    return links, int(prev.group(1)) if prev else None


def clean_body(raw_html: str) -> str:
    """文章內文：去掉 meta、引言（: 開頭與 ※ 引述）、簽名檔與發信站資訊。"""
    main = raw_html.split('id="main-content"', 1)[-1]
    main = main.split("※ 發信站", 1)[0]
    main = re.sub(r'<div class="article-metaline[^"]*">.*?</div>', "", main, flags=re.S)
    text = html.unescape(_TAGS.sub("", main.split(">", 1)[-1]))
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if s == "--":                      # 簽名檔分隔線
            break
        if not s or s.startswith((":", "※ 引述", "※")):
            continue
        lines.append(s)
    return " ".join(lines)


def parse_article(page: str, max_body_chars: int) -> dict | None:
    meta = dict(_META.findall(page))
    title = html.unescape(meta.get("標題", "")).strip()
    author = html.unescape(meta.get("作者", "")).split(" ")[0].strip()
    if not title or not author:
        return None
    body = clean_body(page)
    if len(body) > max_body_chars:
        body = body[:max_body_chars] + "…"
    turns = [{"speaker": author, "text": f"【{title}】{body}", "context_only": True}]
    # 同一人同一分鐘的連續推文是被 PTT 斷行的一則留言 → 直接接起來（不加空白，常斷在字中間）
    last_key = None
    for tag, uid, content, ipdt in _PUSH.findall(page):
        uid = uid.strip()
        text = html.unescape(content).lstrip(":").strip()
        stamp = re.search(r"\d{2}/\d{2} \d{2}:\d{2}", ipdt)
        key = (uid, stamp.group(0) if stamp else None)
        if not text or _FILLER.match(text):
            last_key = None
            continue
        if key == last_key and turns[-1]["speaker"] == uid:
            turns[-1]["text"] += text
            continue
        turns.append({"speaker": uid, "text": text, "push_tag": tag.strip()})
        last_key = key
    return {"title": title, "turns": turns}


def crawl(boards, articles_per_board, max_turns, min_turns, max_body_chars, delay,
          max_pages, seed):
    fetch = Fetcher(delay)
    rng = random.Random(seed)
    out = []
    for board in boards:
        try:
            page = fetch.get(f"/bbs/{board}/index.html")
        except Exception as exc:
            print(f"[ptt] {board}: 讀取失敗（{exc}），跳過")
            continue
        if is_age_gated(page):
            print(f"[ptt] {board}: 需要年齡確認，依規則跳過")
            continue
        # 先收集候選文章（往前翻 max_pages 頁），再隨機抽，避免只抓到同一天的熱門串
        candidates, prev = [], None
        for p in range(max_pages):
            links, prev = parse_index(page)
            candidates += [(h, t) for h, t in links if not _SKIP_TITLES.search(t)]
            if prev is None or p == max_pages - 1:
                break
            page = fetch.get(f"/bbs/{board}/index{prev}.html")
        rng.shuffle(candidates)
        got = 0
        for href, title in candidates:
            if got >= articles_per_board:
                break
            try:
                art = parse_article(fetch.get(href), max_body_chars)
            except Exception as exc:
                print(f"[ptt]   讀取 {href} 失敗（{exc}）")
                continue
            if art is None or len(art["turns"]) - 1 < min_turns:
                continue
            art["turns"] = art["turns"][:1 + max_turns]
            out.append({"source": f"ptt/{board}", "url": BASE + href, **art})
            got += 1
            print(f"[ptt]   {board} {got}/{articles_per_board}：{title[:30]}（{len(art['turns']) - 1} 則推文）")
    print(f"[ptt] 共 {len(out)} 篇、{sum(len(a['turns']) - 1 for a in out)} 則推文；"
          f"發出 {fetch.n_requests} 次請求")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", nargs="+", default=DEFAULT_BOARDS)
    ap.add_argument("--articles_per_board", type=int, default=6)
    ap.add_argument("--max_turns", type=int, default=15, help="每篇最多取幾則推文（多的捨棄）")
    ap.add_argument("--min_turns", type=int, default=5, help="推文少於此數的文章不收")
    ap.add_argument("--max_body_chars", type=int, default=300, help="內文語境最多幾字")
    ap.add_argument("--max_pages", type=int, default=5, help="每板往前翻幾頁收候選文章")
    ap.add_argument("--delay", type=float, default=1.5, help="請求間隔秒數（不得低於 1）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/chat_testset/forum_raw.jsonl")
    args = ap.parse_args(argv)
    if args.delay < 1:
        ap.error("--delay 不得低於 1 秒")
    arts = crawl(args.boards, args.articles_per_board, args.max_turns, args.min_turns,
                 args.max_body_chars, args.delay, args.max_pages, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for a in arts:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    print(f"[ptt] → {args.out}")


if __name__ == "__main__":
    main()

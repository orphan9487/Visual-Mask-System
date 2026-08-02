"""
build_external_dataset.py — 外部情感資料集前處理腳本

將以下資料集轉換為 Stage 2（情緒標籤推論）的 M-CoT 訓練格式，
並輸出為 JSONL，可直接合入 train_stage2.py。

支援的資料集：
  A. SentimentDictionaries (nproellochs/GitHub) ← 自動下載
  B. Sentiment140 (Kaggle)                      ← 需手動放置
  C. Multi-Domain Sentiment (Amazon reviews)    ← 需手動放置
  D. OpinRank (Hotel/Car reviews)               ← 需手動放置
  E. Blog Authorship Corpus                     ← 需手動放置（無標籤，略過）

輸出：
  external_stage2_dataset.jsonl   ← 可直接加入 train_stage2.py 的 DATA_FILES

使用方式：
  conda activate mask_env
  python scripts/training/build_external_dataset.py

資料集下載說明（手動部分）：
  Sentiment140:
    https://www.kaggle.com/datasets/kazanova/sentiment140
    下載後解壓，將 training.1600000.processed.noemoticon.csv 放到：
    data/external/sentiment140/training.csv

  Multi-Domain Sentiment:
    https://www.cs.jhu.edu/~mdredze/datasets/sentiment/
    下載 processed_acl.tar.gz，解壓後放到：
    data/external/multidomain/

  OpinRank:
    https://archive.ics.uci.edu/ml/datasets/opinrank+review+dataset
    解壓後放到：
    data/external/opinrank/
"""

import csv
import json
import os
import random
import re
import time
from pathlib import Path

import requests

# ── 路徑設定 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parents[2]
DATA_DIR      = PROJECT_ROOT / "data" / "external"
OUTPUT_FILE   = PROJECT_ROOT / "external_stage2_dataset.jsonl"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── 情緒標籤映射 ──────────────────────────────────────────────────────────────
# 外部資料集只有 positive / negative / 星等，映射到系統的細粒度情緒標籤
# 使用隨機取樣增加多樣性

# 對齊 AffectiveStateAnalyzer 的標準 7 種情感標籤
POSITIVE_EMOTIONS   = ["joy"]
MILD_POS_EMOTIONS   = ["joy", "neutral"]
NEUTRAL_EMOTIONS    = ["neutral"]
MILD_NEG_EMOTIONS   = ["sadness"]
NEGATIVE_EMOTIONS   = ["sadness", "anger"]
STRONG_NEG_EMOTIONS = ["anger", "disgust"]

def stars_to_emotions(stars: float) -> list[str]:
    """星等 (1–5) → 情緒標籤候選清單（對齊標準 7 類）"""
    if stars >= 4.5:
        return POSITIVE_EMOTIONS       # joy
    elif stars >= 3.5:
        return MILD_POS_EMOTIONS       # joy / neutral
    elif stars >= 2.5:
        return NEUTRAL_EMOTIONS        # neutral
    elif stars >= 1.5:
        return MILD_NEG_EMOTIONS       # sadness
    else:
        return STRONG_NEG_EMOTIONS     # anger / disgust

def sentiment_to_emotions(polarity: int) -> list[str]:
    """Sentiment140 極性 (0=負, 4=正) → 情緒標籤候選（對齊標準 7 類）"""
    return POSITIVE_EMOTIONS if polarity == 4 else NEGATIVE_EMOTIONS

# ── M-CoT 格式化函式 ──────────────────────────────────────────────────────────
def make_stage2_example(
    text: str,
    emotion: str,
    context: str = "Initial state.",
    sentiment_words: list[str] | None = None,
    domain: str = "general",
) -> dict:
    """
    將文字 + 情緒標籤轉換為 Stage 2 的 M-CoT 輸入格式。

    格式對齊訓練資料：
      input_text: emotion inference: Context: ... Current Stream: ... User Prior: ... Rationale: ...
      target_text: {emotion}
    """
    # 清理文字
    text = text.strip().replace("\n", " ")[:300]  # 限制長度

    # 用 sentiment_words 組裝更豐富的 rationale
    if sentiment_words:
        key_words = ", ".join(sentiment_words[:5])
        rationale = (
            f"[Context Analysis]: Text contains sentiment-bearing words: {key_words}. "
            f"[Sentiment Analysis]: Overall {emotion} tone detected in {domain} context. "
            f"[Final Inference]: The expression reflects {emotion} sentiment."
        )
    else:
        sentiment_type = (
            "positive" if emotion in POSITIVE_EMOTIONS + MILD_POS_EMOTIONS
            else "negative" if emotion in NEGATIVE_EMOTIONS + STRONG_NEG_EMOTIONS
            else "neutral"
        )
        rationale = (
            f"[Context Analysis]: Text exhibits {sentiment_type} sentiment in {domain} context. "
            f"[Sentiment Analysis]: Language and tone indicate {emotion} emotional state. "
            f"[Final Inference]: The speaker conveys {emotion} emotion."
        )

    input_text = (
        f"emotion inference: "
        f"Context: {context}. "
        f"Current Stream: {text}. "
        f"User Prior: A general user. "
        f"Rationale: {rationale}"
    )
    return {"input_text": input_text, "target_text": emotion}


# ══════════════════════════════════════════════════════════════════════════════
# A. SentimentDictionaries (nproellochs/SentimentDictionaries)
#    ← 自動從 GitHub 下載
# ══════════════════════════════════════════════════════════════════════════════

GITHUB_RAW = "https://raw.githubusercontent.com/nproellochs/SentimentDictionaries/master"
DICT_FILES = {
    "imdb": f"{GITHUB_RAW}/DictionaryIMDB.csv",
    "8k":   f"{GITHUB_RAW}/Dictionary8K.csv",
}

def download_sentiment_dictionaries() -> dict[str, set]:
    """
    下載 SentimentDictionaries 並回傳：
      { "positive": {word, ...}, "negative": {word, ...} }
    兩個 CSV 合併。
    """
    pos_words, neg_words = set(), set()
    dict_dir = DATA_DIR / "SentimentDictionaries"
    dict_dir.mkdir(exist_ok=True)

    for name, url in DICT_FILES.items():
        local = dict_dir / f"Dictionary{name.upper()}.csv"
        if not local.exists():
            print(f"  ⬇️  下載 {url} ...")
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                local.write_bytes(r.content)
                time.sleep(0.5)
            except Exception as e:
                print(f"  ⚠️  下載失敗：{e}")
                continue

        with open(local, encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 欄位名稱：word / type (Positive / Negative)
                word = (row.get("word") or row.get("Word") or "").strip().lower()
                label = (row.get("type") or row.get("Type") or "").strip()
                if not word:
                    continue
                if "pos" in label.lower():
                    pos_words.add(word)
                elif "neg" in label.lower():
                    neg_words.add(word)

    print(f"  ✅ SentimentDictionaries：正面詞 {len(pos_words)}，負面詞 {len(neg_words)}")
    return {"positive": pos_words, "negative": neg_words}


def get_sentiment_words(text: str, lexicon: dict[str, set]) -> list[str]:
    """從文字中抽出出現在詞典裡的情感詞。"""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    hits  = [w for w in words if w in lexicon["positive"] or w in lexicon["negative"]]
    return list(dict.fromkeys(hits))[:8]   # 去重，最多取 8 個


# ══════════════════════════════════════════════════════════════════════════════
# B. Sentiment140
#    資料放置：data/external/sentiment140/training.csv
#    格式：polarity(0/4), id, date, query, user, text
# ══════════════════════════════════════════════════════════════════════════════

def process_sentiment140(lexicon: dict, max_samples: int = 50000) -> list[dict]:
    path = DATA_DIR / "sentiment140" / "training.csv"
    if not path.exists():
        print("  ⚠️  Sentiment140 未找到，跳過。")
        print(f"      請下載後放至：{path}")
        return []

    records, count = [], 0
    with open(path, encoding="latin-1") as f:
        reader = csv.reader(f)
        rows = list(reader)

    random.shuffle(rows)  # 打亂確保多樣性
    for row in rows:
        if count >= max_samples:
            break
        if len(row) < 6:
            continue
        polarity, text = int(row[0]), row[5].strip()
        if not text or len(text) < 10:
            continue

        emotions   = sentiment_to_emotions(polarity)
        emotion    = random.choice(emotions)
        sent_words = get_sentiment_words(text, lexicon)

        records.append(make_stage2_example(
            text=text, emotion=emotion,
            sentiment_words=sent_words, domain="social media"
        ))
        count += 1

    print(f"  ✅ Sentiment140：{len(records)} 筆")
    return records


# ══════════════════════════════════════════════════════════════════════════════
# C. Multi-Domain Sentiment Dataset（Amazon 多領域評論）
#    資料放置：data/external/multidomain/
#    格式：.review 文字檔，每則評論包含 <rating> 和 <review_text>
# ══════════════════════════════════════════════════════════════════════════════

def process_multidomain(lexicon: dict, max_per_domain: int = 5000) -> list[dict]:
    base = DATA_DIR / "multidomain"
    if not base.exists():
        print("  ⚠️  Multi-Domain Sentiment 未找到，跳過。")
        print(f"      請下載後放至：{base}")
        return []

    review_files = list(base.glob("**/*.review"))
    if not review_files:
        review_files = list(base.glob("**/*.txt"))

    if not review_files:
        print("  ⚠️  找不到 .review 檔案，跳過。")
        return []

    records = []
    for rfile in review_files:
        domain = rfile.parent.name
        content = rfile.read_text(encoding="utf-8", errors="ignore")

        # 解析每則評論（<review>...</review> 格式）
        reviews = re.findall(r"<review>(.*?)</review>", content, re.DOTALL)
        if not reviews:
            # 嘗試無標籤格式（每行一則）
            reviews = [line.strip() for line in content.splitlines() if len(line.strip()) > 20]

        domain_records = []
        for review in reviews:
            if len(domain_records) >= max_per_domain:
                break

            # 嘗試取出星等
            rating_match = re.search(r"<rating>\s*([\d.]+)\s*</rating>", review)
            stars = float(rating_match.group(1)) if rating_match else None

            # 取出評論文字
            text_match = re.search(r"<review_text>(.*?)</review_text>", review, re.DOTALL)
            text = text_match.group(1).strip() if text_match else review[:300]

            if not text or len(text) < 15:
                continue

            emotions   = stars_to_emotions(stars) if stars else random.choice([POSITIVE_EMOTIONS, NEGATIVE_EMOTIONS])
            emotion    = random.choice(emotions)
            sent_words = get_sentiment_words(text, lexicon)

            domain_records.append(make_stage2_example(
                text=text, emotion=emotion,
                sentiment_words=sent_words, domain=f"Amazon/{domain}"
            ))

        records.extend(domain_records)
        print(f"    {domain}: {len(domain_records)} 筆")

    print(f"  ✅ Multi-Domain Sentiment：共 {len(records)} 筆")
    return records


# ══════════════════════════════════════════════════════════════════════════════
# D. OpinRank（TripAdvisor 飯店 / Edmunds 汽車評論）
#    資料放置：data/external/opinrank/
#    子目錄結構：hotels/<city>/<hotel>.txt  |  cars/<year>/<model>.txt
# ══════════════════════════════════════════════════════════════════════════════

def process_opinrank(lexicon: dict, max_samples: int = 20000) -> list[dict]:
    """
    OpinRank 實際格式：
      - 評論檔：無副檔名，XML-like <DOC><TEXT>...</TEXT></DOC>
      - 星等：各 category 下的 CSV 檔（overall_rating 欄位）
        cars:   0~10 分
        hotels: 0~5  分
    """
    base = DATA_DIR / "opinrank"
    if not base.exists():
        print("  ⚠️  OpinRank 未找到，跳過。")
        return []

    # ── Step 1：從 CSV 建立 doc_id → rating 對照表 ──────────────────────────
    rating_map: dict[str, float] = {}   # doc_id → normalized 0~5 rating
    for csv_path in base.glob("**/*.csv"):
        try:
            with open(csv_path, encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    doc_id = (row.get("doc_id") or row.get("docid") or "").strip()
                    rating_raw = (row.get("overall_rating") or
                                  row.get("overall_ratingsource") or "").strip()
                    if not doc_id or not rating_raw:
                        continue
                    try:
                        rating = float(rating_raw)
                        # cars 用 0-10，hotels 用 0-5，統一轉成 0-5
                        if rating > 5:
                            rating = rating / 2.0
                        rating_map[doc_id] = rating
                    except ValueError:
                        continue
        except Exception:
            continue
    print(f"    rating_map: {len(rating_map)} 筆 CSV 評分")

    # ── Step 2：找所有無副檔名的評論檔 ─────────────────────────────────────
    review_files = [
        p for p in base.rglob("*")
        if p.is_file() and p.suffix == "" and p.stat().st_size > 100
    ]
    random.shuffle(review_files)
    print(f"    review files: {len(review_files)} 個")

    records, count = [], 0
    for fpath in review_files:
        if count >= max_samples:
            break

        domain = "hotel" if "hotel" in str(fpath).lower() else "car"
        doc_id = fpath.stem   # 檔名就是 doc_id（如 2007_acura_mdx）

        # 從 rating_map 取星等，找不到就用中等正面
        raw_rating = rating_map.get(doc_id)
        emotions   = stars_to_emotions(raw_rating) if raw_rating else MILD_POS_EMOTIONS

        try:
            content = fpath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        # 解析每個 <DOC>...</DOC> 區塊
        docs = re.findall(r"<DOC>(.*?)</DOC>", content, re.DOTALL)
        for doc in docs:
            if count >= max_samples:
                break

            # 優先取 <TEXT>，其次取 <FAVORITE>
            text_match = re.search(r"<TEXT>(.*?)</TEXT>", doc, re.DOTALL)
            if not text_match:
                continue
            text = text_match.group(1).strip().replace("\n", " ")
            if len(text) < 20:
                continue

            emotion    = random.choice(emotions)
            sent_words = get_sentiment_words(text, lexicon)

            records.append(make_stage2_example(
                text=text, emotion=emotion,
                sentiment_words=sent_words, domain=domain
            ))
            count += 1

    print(f"  ✅ OpinRank：{len(records)} 筆")
    return records


# ══════════════════════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  外部情感資料集前處理")
    print("=" * 60)

    all_records = []

    # A. 下載 SentimentDictionaries → 作為情感詞典（非直接訓練資料）
    print("\n[A] 下載 SentimentDictionaries 詞典...")
    lexicon = download_sentiment_dictionaries()

    # B. Sentiment140
    print("\n[B] 處理 Sentiment140...")
    all_records += process_sentiment140(lexicon, max_samples=50000)

    # C. Multi-Domain Sentiment
    print("\n[C] 處理 Multi-Domain Sentiment Dataset...")
    all_records += process_multidomain(lexicon, max_per_domain=5000)

    # D. OpinRank
    print("\n[D] 處理 OpinRank...")
    all_records += process_opinrank(lexicon, max_samples=20000)

    if not all_records:
        print("\n⚠️  沒有任何資料被處理，請確認資料集已放置到正確路徑。")
        return

    # 打亂後寫出 JSONL
    random.shuffle(all_records)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"✅ 完成！共 {len(all_records)} 筆訓練資料")
    print(f"   輸出：{OUTPUT_FILE}")
    print(f"\n接下來，在 train_stage2.py 的 DATA_FILES 加入：")
    print(f'   str(PROJECT_ROOT / "external_stage2_dataset.jsonl")')
    print("=" * 60)


if __name__ == "__main__":
    random.seed(42)
    main()

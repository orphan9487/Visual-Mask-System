"""
train_stage2.py — Stage 2 繼續訓練腳本
AffectiveStateAnalyzer（情緒推論 → emotion label）

改良點（相較於原始 train_stage2.py）：
  1. 從現有 checkpoint 繼續訓練
  2. 自動從 MySQL training_data 表匯出使用者反饋資料，與合成資料合併
  3. 學習率降為 1e-5
  4. 加入 eval split 觀察 loss 曲線

使用方式：
  conda activate mask_env
  python scripts/training/train_stage2.py
"""

import json
import numpy as np
from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ── 路徑設定 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parents[2]
BACKUP_DIR      = Path(r"D:\Independent_study2_Backup")

CHECKPOINT_PATH = str(PROJECT_ROOT / "models" / "mcot_answer_model")

DATA_FILES = [
    str(BACKUP_DIR / "train_stage2_synthetic.jsonl"),
    # 外部情感資料集（執行 build_external_dataset.py 後自動生成）
    # 若檔案存在則自動納入，不存在時跳過
    str(PROJECT_ROOT / "external_stage2_dataset.jsonl"),
]

OUTPUT_DIR = str(PROJECT_ROOT / "models" / "mcot_answer_model_v2")

# ── 訓練超參數 ────────────────────────────────────────────────────────────────
LEARNING_RATE   = 1e-5
EPOCHS          = 10
BATCH_SIZE      = 4
MAX_INPUT_LEN   = 256
MAX_TARGET_LEN  = 32     # label 很短，不需要 128
EVAL_SPLIT      = 0.1
MIN_FEEDBACK_ROWS = 5    # 至少要有這麼多筆反饋資料才納入訓練

# ── 從 MySQL 匯出使用者反饋資料 ───────────────────────────────────────────────
def load_feedback_from_db() -> list[dict]:
    """
    從 training_data 表取出情緒修正紀錄，
    轉換為 Stage 2 訓練格式。

    input_text 格式：
      emotion inference: {context} Rationale: [User feedback correction.]
    target_text：
      {correct_label}
    """
    try:
        import pymysql
        conn = pymysql.connect(
            host="127.0.0.1", user="root", password="",
            database="mcot_chat_db", charset="utf8mb4"
        )
        with conn.cursor() as cur:
            cur.execute(
                "SELECT context, correct_label FROM training_data "
                "WHERE context != '' AND correct_label != '' "
                "ORDER BY correction_time DESC"
            )
            rows = cur.fetchall()
        conn.close()
    except Exception as e:
        print(f"  ⚠️ 無法連接 MySQL，跳過反饋資料：{e}")
        return []

    records = []
    for context, correct_label in rows:
        # context 已包含 "Context: ... Current Stream: ... User Prior: ..."
        input_text = (
            f"emotion inference: {context} "
            f"Rationale: [User feedback correction.]"
        )
        records.append({
            "input_text":  input_text,
            "target_text": correct_label,
        })

    print(f"  ✅ 從 MySQL 匯出 {len(records)} 筆反饋訓練資料")
    return records


# ── 載入模型 ──────────────────────────────────────────────────────────────────
print(f"📂 從 checkpoint 載入 Stage 2 模型：{CHECKPOINT_PATH}")
tokenizer = T5Tokenizer.from_pretrained(CHECKPOINT_PATH)
model     = T5ForConditionalGeneration.from_pretrained(CHECKPOINT_PATH)
print("✅ 模型載入完成")

# ── 載入原始合成資料 ──────────────────────────────────────────────────────────
print("📊 載入訓練資料...")
# 只載入實際存在的檔案
existing_files = [f for f in DATA_FILES if Path(f).exists()]
missing_files  = [f for f in DATA_FILES if not Path(f).exists()]
for f in missing_files:
    print(f"   ⚠️  跳過（不存在）：{Path(f).name}")

synth_dataset = load_dataset("json", data_files=existing_files, split="train")
print(f"   合計：{len(synth_dataset)} 筆（來自 {len(existing_files)} 個檔案）")

# ── 載入 MySQL 反饋資料 ───────────────────────────────────────────────────────
print("📊 從 MySQL 載入使用者反饋資料...")
feedback_records = load_feedback_from_db()

if len(feedback_records) >= MIN_FEEDBACK_ROWS:
    feedback_dataset = Dataset.from_list(feedback_records)
    # 反饋資料較少，重複 3 次以平衡比例（避免被合成資料淹沒）
    feedback_dataset = concatenate_datasets([feedback_dataset] * 3)
    combined = concatenate_datasets([synth_dataset, feedback_dataset])
    print(f"   合成 + 反饋合併：{len(combined)} 筆")
else:
    combined = synth_dataset
    print(f"   反饋資料不足 {MIN_FEEDBACK_ROWS} 筆，僅使用合成資料")

# ── 分割 train / eval ────────────────────────────────────────────────────────
split = combined.train_test_split(test_size=EVAL_SPLIT, seed=42)
dataset = DatasetDict({"train": split["train"], "eval": split["test"]})
print(f"   Train: {len(dataset['train'])}  Eval: {len(dataset['eval'])}")

# ── Tokenize ──────────────────────────────────────────────────────────────────
def preprocess(examples):
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=MAX_INPUT_LEN,
        truncation=True,
        padding="max_length",
    )
    labels = tokenizer(
        text_target=examples["target_text"],
        max_length=MAX_TARGET_LEN,
        truncation=True,
        padding="max_length",
    )
    labels["input_ids"] = [
        [(tok if tok != tokenizer.pad_token_id else -100) for tok in label]
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


print("⚙️  Tokenizing...")
tokenized = dataset.map(preprocess, batched=True, remove_columns=["input_text", "target_text"])

# ── 情感標籤集合 ──────────────────────────────────────────────────────────────
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

def compute_metrics(eval_preds):
    """
    將模型生成的 token ids 解碼回文字，與真實標籤比對，
    計算 Accuracy、Macro F1、Per-class F1。
    """
    predictions, label_ids = eval_preds

    # label_ids 中 -100 是 padding，換回 pad_token_id 才能 decode
    label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)

    decoded_preds  = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(label_ids,   skip_special_tokens=True)

    # 清理空白，統一小寫
    decoded_preds  = [p.strip().lower() for p in decoded_preds]
    decoded_labels = [l.strip().lower() for l in decoded_labels]

    # 過濾掉不在已知標籤集合的預測（模型偶爾會輸出亂碼）
    valid_labels = set(EMOTION_LABELS)
    filtered = [(p, l) for p, l in zip(decoded_preds, decoded_labels)
                if l in valid_labels]

    if not filtered:
        return {"accuracy": 0.0, "f1_macro": 0.0}

    preds_f, labels_f = zip(*filtered)
    preds_f  = [p if p in valid_labels else "neutral" for p in preds_f]

    acc      = accuracy_score(labels_f, preds_f)
    f1_macro = f1_score(labels_f, preds_f, average="macro", zero_division=0)
    f1_weighted = f1_score(labels_f, preds_f, average="weighted", zero_division=0)

    # 完整的 per-class 報告印到 console（不影響 return 值）
    print("\n" + "="*55)
    print("  Classification Report (Eval Set)")
    print("="*55)
    print(classification_report(labels_f, preds_f,
                                labels=EMOTION_LABELS,
                                zero_division=0))

    return {
        "accuracy":     round(acc,         4),
        "f1_macro":     round(f1_macro,    4),
        "f1_weighted":  round(f1_weighted, 4),
    }

# ── 訓練參數 ──────────────────────────────────────────────────────────────────
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    bf16=True,
    logging_steps=10,
    weight_decay=0.01,
    predict_with_generate=True,       # 必須開啟才能 decode 預測結果
    generation_max_length=MAX_TARGET_LEN,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["eval"],
    processing_class=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
    compute_metrics=compute_metrics,
)

# ── 開始訓練 ──────────────────────────────────────────────────────────────────
print(f"\n🚀 開始 Stage 2 繼續訓練（合成資料 + 使用者反饋）")
print(f"   輸出目錄：{OUTPUT_DIR}\n")
trainer.train()

print(f"\n✅ Stage 2 訓練完成！新模型已儲存至：{OUTPUT_DIR}")
print("   若要替換正式使用的模型，將此資料夾內容複製到 models/mcot_answer_model/")

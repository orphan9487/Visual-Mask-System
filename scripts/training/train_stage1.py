"""
train_stage1.py — Stage 1 繼續訓練腳本
SemanticIntentDecoder（社會語境推理 → rationale）

改良點（相較於原始 train_decoder.py）：
  1. 從現有 checkpoint 繼續訓練，不從 flan-t5-base 從頭開始
  2. 學習率降為 1e-5（避免破壞已學知識）
  3. 訓練輪數縮短（繼續訓練不需 30 輪）
  4. 加入 eval split（10%）觀察 loss 曲線
  5. 資料路徑集中在頂部方便修改

使用方式：
  conda activate mask_env
  python scripts/training/train_stage1.py
"""

from pathlib import Path
from datasets import load_dataset, DatasetDict
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

# ── 路徑設定 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parents[2]
BACKUP_DIR      = Path(r"D:\Independent_study2_Backup")

# 從這個 checkpoint 繼續訓練（已微調好的 Stage 1 模型）
CHECKPOINT_PATH = str(PROJECT_ROOT / "models" / "mcot_rationale_model")

# 訓練資料（Stage 1 合成資料集）
DATA_FILES = [
    str(BACKUP_DIR / "train_stage1_synthetic.jsonl"),
    # 如有其他 Stage 1 資料可在此增加：
    # str(BACKUP_DIR / "train_augmented.jsonl"),
]

# 輸出新 checkpoint 的位置
OUTPUT_DIR = str(PROJECT_ROOT / "models" / "mcot_rationale_model_v2")

# ── 訓練超參數 ────────────────────────────────────────────────────────────────
LEARNING_RATE   = 1e-5    # 繼續訓練用低學習率，避免破壞已學知識
EPOCHS          = 10      # 繼續訓練不需 30 輪
BATCH_SIZE      = 4
MAX_INPUT_LEN   = 256
MAX_TARGET_LEN  = 128
EVAL_SPLIT      = 0.1     # 10% 作為驗證集

# ── 載入模型（從現有 checkpoint）────────────────────────────────────────────
print(f"📂 從 checkpoint 載入 Stage 1 模型：{CHECKPOINT_PATH}")
tokenizer = T5Tokenizer.from_pretrained(CHECKPOINT_PATH)
model     = T5ForConditionalGeneration.from_pretrained(CHECKPOINT_PATH)
print("✅ 模型載入完成")

# ── 載入並合併訓練資料 ────────────────────────────────────────────────────────
print(f"📊 載入訓練資料...")
raw_dataset = load_dataset("json", data_files=DATA_FILES, split="train")
print(f"   共 {len(raw_dataset)} 筆")

# 分割 train / eval
split = raw_dataset.train_test_split(test_size=EVAL_SPLIT, seed=42)
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
    # padding token 不計入 loss
    labels["input_ids"] = [
        [(tok if tok != tokenizer.pad_token_id else -100) for tok in label]
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


print("⚙️  Tokenizing...")
tokenized = dataset.map(preprocess, batched=True, remove_columns=["input_text", "target_text"])

# ── 訓練參數 ──────────────────────────────────────────────────────────────────
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    eval_strategy="epoch",          # 每個 epoch 評估一次
    save_strategy="epoch",
    save_total_limit=2,             # 只保留最近 2 個 checkpoint
    load_best_model_at_end=True,    # 訓練結束時載入最佳模型
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=True,
    logging_steps=10,
    weight_decay=0.01,
    predict_with_generate=False,    # 只看 loss，不跑 generate（加速）
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["eval"],
    processing_class=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
)

# ── 開始訓練 ──────────────────────────────────────────────────────────────────
print(f"\n🚀 開始 Stage 1 繼續訓練（從 checkpoint-390 之後）")
print(f"   輸出目錄：{OUTPUT_DIR}\n")
trainer.train()

print(f"\n✅ Stage 1 訓練完成！新模型已儲存至：{OUTPUT_DIR}")
print("   若要替換正式使用的模型，將此資料夾內容複製到 models/mcot_rationale_model/")

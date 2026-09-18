# -*- coding: utf-8 -*-
"""
ERC LoRA 微調（PEFT + transformers Trainer，不用 TRL，過程透明可讀）。

要點：
  - 底座預設 Qwen2.5-1.5B-Instruct，bf16。
  - LoRA：rank/alpha 可調，只掛注意力層 q/k/v/o_proj。
  - **Completion-only masking**：只對 assistant 的標籤 token 算 loss，
    prompt 部分的 label 設 -100 不算 loss —— 這樣模型是「學著在該 prompt 下輸出正確標籤」，
    而不是去背整段 prompt。
  - 樣本用 tokenizer.apply_chat_template 組裝，與推論時 backbone.chat 的組法一致。

用法：
    .venv\\Scripts\\python -m src.training.train_erc_lora \\
        --data data/erc_sft/meld_train.jsonl --model qwen2.5-1.5b \\
        --rank 16 --alpha 32 --lr 2e-4 --epochs 1 --out models/qwen_erc_lora
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from src.reasoning.backbone import PRESET_MODELS

IGNORE = -100


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_tokenize_fn(tokenizer, max_len):
    """
    把 {system,user,assistant} 組成對話並 tokenize，
    只讓 assistant 部分參與 loss（completion-only）。
    """
    def _tok(sample):
        # prompt 段（到 assistant 開頭為止）
        prompt_msgs = [
            {"role": "system", "content": sample["system"]},
            {"role": "user", "content": sample["user"]},
        ]
        prompt_ids = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=True, add_generation_prompt=True
        )
        # 完整段（prompt + assistant 標籤 + eos）
        answer = sample["assistant"] + tokenizer.eos_token
        answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + answer_ids
        labels = [IGNORE] * len(prompt_ids) + answer_ids[:]   # 只算 answer 的 loss
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]
        return {"input_ids": input_ids, "labels": labels,
                "attention_mask": [1] * len(input_ids)}
    return _tok


class Collator:
    """動態 padding：input 用 pad_token，labels 用 -100，避免 pad 參與 loss。"""
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        ids, lbl, am = [], [], []
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * pad)
            lbl.append(b["labels"] + [IGNORE] * pad)
            am.append(b["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(lbl),
            "attention_mask": torch.tensor(am),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/erc_sft/meld_train.jsonl")
    ap.add_argument("--model", default="qwen2.5-1.5b")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--out", default="models/qwen_erc_lora")
    args = ap.parse_args()

    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset

    model_id = PRESET_MODELS.get(args.model.lower(), args.model)
    print(f"[train] 底座：{model_id}  rank={args.rank} alpha={args.alpha} lr={args.lr}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    raw = load_jsonl(args.data)
    ds = Dataset.from_list(raw).map(
        build_tokenize_fn(tokenizer, args.max_len),
        remove_columns=["system", "user", "assistant"],
    )
    print(f"[train] 訓練樣本：{len(ds)}")

    targs = TrainingArguments(
        output_dir=args.out + "_ckpt",
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to=[],
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=Collator(tokenizer))
    trainer.train()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"[train] 完成，adapter 存於：{args.out}")


if __name__ == "__main__":
    main()

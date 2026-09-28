# -*- coding: utf-8 -*-
"""
Breeze2 (Llama-Breeze2-3B-Instruct / InternVL VLM) 上的 ERC LoRA 微調。

為什麼需要獨立的訓練腳本（不能用 train_erc_lora.py）：
  - Breeze2 是 InternVL 複合模型（vision + language），用 `AutoModel`+trust_remote_code
    載入，不是 `AutoModelForCausalLM`。
  - 它的複合 `forward()` 硬性要求 pixel_values / image_flags（見 remote code
    modeling_internvl_chat.py:105），純文字無法直接呼叫 → 我們改成直接對
    `model.language_model`（內部是 LlamaForCausalLM）算 loss。
  - 它的對話模板不是 tokenizer.apply_chat_template，而是 `mtkresearch` 的
    `MRPromptV3`（Llama3.2 格式）。推論時 backbone._chat_internvl 把
    system+user 併成一段丟進 user turn，讓 MRPromptV3 補上預設 MediaTek 前言。
    **訓練必須用同一條路徑組 prompt，否則訓練≠推論。**

LoRA 掛法：
  - 對「整個複合模型」get_peft_model，但 target_modules 只指名 q/k/v/o_proj。
    InternViT 視覺塔用的是融合的 `qkv` 與 `proj`（非 *_proj），故不會被掛到。
  - 因此存出的 adapter 鍵是 `language_model.model.layers.*.self_attn.{q,k,v,o}_proj`，
    推論端 `PeftModel.from_pretrained(full_breeze2, adapter)` 路徑正好對齊。

Completion-only masking：
  - 只對 assistant 標籤 token 算 loss，prompt 段 label=-100。
  - 邊界用「prompt 與 full 的共同 token 前綴長度」界定，避開 BPE 跨邊界合併。

用法：
    .venv\\Scripts\\python -m src.training.train_erc_breeze2_lora \\
        --data data/erc_sft/cped_train.jsonl --rank 16 --alpha 32 \\
        --lr 2e-4 --epochs 2 --out models/breeze2_erc_cped_lora
"""

import argparse
import json
import math
import sys
import time
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
BREEZE2_ID = PRESET_MODELS["breeze2-3b"]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _swap_to_sdpa(language_model):
    """把 LlamaForCausalLM 各層的 eager attention 換成 SDPA（同參數、只換 forward）。"""
    try:
        from transformers.models.llama.modeling_llama import LlamaSdpaAttention
    except ImportError:
        print("[train] 找不到 LlamaSdpaAttention，維持 eager 注意力。")
        return
    language_model.config._attn_implementation = "sdpa"
    n = 0
    for layer in language_model.model.layers:
        attn = layer.self_attn
        if attn.__class__ is not LlamaSdpaAttention:
            attn.__class__ = LlamaSdpaAttention
            n += 1
    print(f"[train] 已將 {n} 層注意力切換為 SDPA（省記憶體、比 eager 快很多）。")


def build_tokenize_fn(tokenizer, mr_prompt, max_len):
    """
    用 MRPromptV3 組 prompt，複製推論 (_chat_internvl) 的組法：
    system 與 user 併成一段當作 user turn 內容，MRPromptV3 自動補預設系統前言。
    只讓 assistant 段參與 loss（completion-only）。
    """
    def _tok(sample):
        merged_user = f"{sample['system']}\n\n{sample['user']}"
        prompt_conv = [{"role": "user", "content": merged_user}]
        full_conv = prompt_conv + [{"role": "assistant", "content": sample["assistant"]}]

        # get_prompt(add_bos_token=False)：BOS 交給 tokenizer(add_special_tokens=True) 補，
        # 與 model.chat() 內部 tokenizer(query, return_tensors='pt') 一致。
        prompt_str = mr_prompt.get_prompt(prompt_conv)
        full_str = mr_prompt.get_prompt(full_conv)

        prompt_ids = tokenizer(prompt_str, add_special_tokens=True)["input_ids"]
        full_ids = tokenizer(full_str, add_special_tokens=True)["input_ids"]

        # 共同前綴長度 = prompt 段實際佔的 token 數（穩健對抗 BPE 邊界合併）
        n = 0
        limit = min(len(prompt_ids), len(full_ids))
        while n < limit and prompt_ids[n] == full_ids[n]:
            n += 1

        labels = [IGNORE] * n + full_ids[n:]
        full_ids = full_ids[:max_len]
        labels = labels[:max_len]
        return {"input_ids": full_ids, "labels": labels,
                "attention_mask": [1] * len(full_ids)}
    return _tok


class Collator:
    """動態 padding：input 用 pad_token，labels 用 -100。"""
    def __init__(self, pad_id):
        self.pad_id = pad_id

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
    ap.add_argument("--data", default="data/erc_sft/cped_train.jsonl")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--limit", type=int, default=None,
                    help="只取前 N 筆樣本（煙霧測試用）")
    ap.add_argument("--save_every", type=int, default=0,
                    help="每 N 個優化步存一次 adapter 到 <out>_ckpt（0=不中途存）")
    ap.add_argument("--quant", default=None, choices=[None, "4bit", "8bit"],
                    help="底座量化：4bit(QLoRA，省 VRAM、可開大 batch) / 8bit / None(bf16)")
    ap.add_argument("--no_grad_ckpt", action="store_true",
                    help="關閉 gradient checkpointing（較快但吃更多 VRAM）")
    ap.add_argument("--free_vision", action="store_true", default=True,
                    help="訓練前把視覺塔移到 CPU 釋放 VRAM（純文字訓練用不到；4bit 下不適用）")
    ap.add_argument("--out", default="models/breeze2_erc_cped_lora")
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
    from transformers import get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from mtkresearch.llm.prompt import MRPromptV3

    dtype = torch.bfloat16
    print(f"[train] 底座：{BREEZE2_ID}  rank={args.rank} alpha={args.alpha} "
          f"lr={args.lr} quant={args.quant}")
    print("[train] 載入 Breeze2（InternVL 複合模型，trust_remote_code）…")

    tokenizer = AutoTokenizer.from_pretrained(BREEZE2_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.quant in ("4bit", "8bit"):
        # QLoRA：4-bit 底座大幅省 VRAM（3B bf16 ~6GB → ~2GB），
        # 消除逼近 OOM 的記憶體搬移、可開大 batch。Breeze2 是 remote-code 複合模型，
        # 量化載入時 Accelerate 的 dispatch 需 force_hooks（沿用 backbone 的作法）。
        if args.quant == "4bit":
            bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            )
        else:
            bnb = BitsAndBytesConfig(load_in_8bit=True)

        import transformers.modeling_utils as _mu
        _orig = _mu.dispatch_model

        def _dispatch_force_hooks(m, *a, **k):
            k["force_hooks"] = True
            return _orig(m, *a, **k)

        _mu.dispatch_model = _dispatch_force_hooks
        try:
            model = AutoModel.from_pretrained(
                BREEZE2_ID, trust_remote_code=True,
                quantization_config=bnb, device_map={"": 0},
            )
        finally:
            _mu.dispatch_model = _orig

        model.language_model.config.use_cache = False
        # 複合模型本身不支援 gradient_checkpointing_enable（只有內層 language_model 支援），
        # 故這裡關掉 kbit prep 的 GC，改由下方統一在 language_model 上手動開。
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    else:
        model = AutoModel.from_pretrained(
            BREEZE2_ID, trust_remote_code=True, torch_dtype=dtype,
        )
        # 純文字訓練只走 language_model；視覺塔/投影層可移到 CPU 省 VRAM。
        if args.free_vision:
            try:
                model.vision_model.to("cpu")
                model.mlp1.to("cpu")
                print("[train] 視覺塔 + mlp1 已移到 CPU（釋放 VRAM）。")
            except Exception as e:  # noqa: BLE001
                print(f"[train] 視覺塔移到 CPU 失敗（忽略）：{e}")
        model.language_model.to("cuda")
        model.language_model.config.use_cache = False

    # ---- 強制 SDPA 注意力 ----
    # Breeze2 remote code 在無 FlashAttention 時硬寫死 'eager'（最慢、backward 會保留
    # 完整 [batch,heads,seq,seq]）。PyTorch 2.x 內建的 SDPA 又快又省記憶體且不需 flash-attn。
    # LlamaSdpaAttention 與 LlamaAttention 參數相同、只換 forward，故可直接換 __class__。
    _swap_to_sdpa(model.language_model)

    lora = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # 確認 LoRA 只掛在 language_model（不應碰到 vision_model）
    lora_on_vision = [n for n, _ in model.named_parameters()
                      if "lora_" in n and "vision_model" in n]
    assert not lora_on_vision, f"LoRA 誤掛到視覺塔：{lora_on_vision[:3]}"

    if not args.no_grad_ckpt:
        # 在內層 language_model 上開 grad ckpt（複合模型本身不支援）。
        # 無 FlashAttention 時，eager 注意力會為 backward 保留 [batch,heads,seq,seq]，
        # 不開 grad ckpt 會累積 28 層而 OOM，故此為必要。
        model.language_model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        print("[train] gradient checkpointing 已開啟。")

    # ---- 資料 ----
    raw = load_jsonl(args.data)
    if args.limit:
        raw = raw[:args.limit]
    tok_fn = build_tokenize_fn(tokenizer, MRPromptV3(), args.max_len)
    dataset = [tok_fn(s) for s in raw]
    print(f"[train] 訓練樣本：{len(dataset)}")

    collate = Collator(tokenizer.pad_token_id)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch, shuffle=True, collate_fn=collate,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr)

    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = int(steps_per_epoch * args.epochs)
    sched = get_cosine_schedule_with_warmup(
        optim, int(total_steps * args.warmup_ratio), total_steps,
    )
    print(f"[train] 優化步數：{total_steps}（每 epoch {steps_per_epoch} 步，"
          f"batch={args.batch} × grad_accum={args.grad_accum}）")

    # ---- 手寫訓練迴圈（透明可讀）----
    model.train()
    lm = model.base_model.model.language_model  # 直接對語言模型算 loss（略過複合 forward）
    global_step = 0
    running = 0.0
    t0 = time.time()
    n_epochs = int(math.ceil(args.epochs))
    for epoch in range(n_epochs):
        optim.zero_grad(set_to_none=True)
        for it, batch in enumerate(loader):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            out = lm(input_ids=batch["input_ids"],
                     attention_mask=batch["attention_mask"],
                     labels=batch["labels"])
            loss = out.loss / args.grad_accum
            loss.backward()
            running += out.loss.item()

            if (it + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 10 == 0:
                    avg = running / (10 * args.grad_accum)
                    running = 0.0
                    elapsed = time.time() - t0
                    print(f"[train] step {global_step}/{total_steps} "
                          f"loss={avg:.4f} lr={sched.get_last_lr()[0]:.2e} "
                          f"({elapsed:.0f}s)", flush=True)
                if args.save_every and global_step % args.save_every == 0:
                    ckpt = args.out + "_ckpt"
                    Path(ckpt).mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(ckpt)
                    print(f"[train] 中途存檔 → {ckpt}（step {global_step}）", flush=True)
                if global_step >= total_steps:
                    break
        if global_step >= total_steps:
            break

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"[train] 完成，adapter 存於：{args.out}")


if __name__ == "__main__":
    main()

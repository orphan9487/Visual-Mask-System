# -*- coding: utf-8 -*-
"""
7 個情緒標籤各自的 log p(標籤 | prompt)（Breeze2 / MRPromptV3 格式，與訓練完全一致）。
供 (1) 評估時的校正分析（src/evaluation/score_labels.py）與 (2) 部署時的先驗校正判讀共用。
"""

import torch

from .backbone import Backbone
from .labels import CANONICAL_EMOTIONS


class LabelScorer:
    """對給定 prompt 一次批次計算 7 個標籤的 log p。"""

    def __init__(self, bk: Backbone):
        from mtkresearch.llm.prompt import MRPromptV3
        self.tok = bk.tokenizer
        self.mr = MRPromptV3()
        model = bk.model
        base = model.base_model.model if hasattr(model, "base_model") else model
        self.lm = base.language_model          # 套了 LoRA 的內層 LlamaForCausalLM
        # Keep inference independent of training-script dependencies.
        try:
            from transformers.models.llama.modeling_llama import LlamaSdpaAttention
        except ImportError:
            LlamaSdpaAttention = None
        if LlamaSdpaAttention is not None:
            self.lm.config._attn_implementation = 'sdpa'
            for layer in self.lm.model.layers:
                layer.self_attn.__class__ = LlamaSdpaAttention
        self.device = next(self.lm.parameters()).device
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id

    @torch.inference_mode()
    def score(self, system: str, user: str) -> list[float]:
        merged = f"{system}\n\n{user}"
        prompt_ids = self.tok(self.mr.get_prompt([{"role": "user", "content": merged}]),
                              add_special_tokens=True)["input_ids"]
        seqs, starts = [], []
        for lab in CANONICAL_EMOTIONS:
            full = self.tok(self.mr.get_prompt([{"role": "user", "content": merged},
                                                {"role": "assistant", "content": lab}]),
                            add_special_tokens=True)["input_ids"]
            n = 0                                       # 共同前綴（與訓練的 masking 同法）
            while n < min(len(prompt_ids), len(full)) and prompt_ids[n] == full[n]:
                n += 1
            seqs.append(full)
            starts.append(n)
        maxlen = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), maxlen), self.pad_id, dtype=torch.long)
        att = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            att[i, :len(s)] = 1
        ids, att = ids.to(self.device), att.to(self.device)
        # 只對「標籤 token」那幾個位置算輸出層：若對全部位置算 7×seq×128k 詞表的 float32
        # logits 會吃掉數 GB，在 Windows 上溢出到系統記憶體、慢數十倍。
        hidden = self.lm.model(input_ids=ids, attention_mask=att).last_hidden_state
        out = []
        for i, s in enumerate(seqs):
            pos = torch.arange(starts[i], len(s), device=self.device)
            logits = self.lm.lm_head(hidden[i, pos - 1]).float()      # [k, vocab]，k≈2–3
            logp = torch.log_softmax(logits, dim=-1)
            out.append(float(logp.gather(1, ids[i, pos].unsqueeze(1)).sum()))
        return out

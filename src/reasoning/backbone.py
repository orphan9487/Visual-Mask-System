# -*- coding: utf-8 -*-
"""
可設定的語言模型後端 (Backbone)。

統一封裝不同底座的載入與對話式生成，讓推理層與評估程式碼與「用哪個模型」解耦：
  - 標準 CausalLM：Qwen2.5、TinyLlama、Breeze-7B（走 AutoModelForCausalLM）
  - Breeze2（InternVL 架構 VLM）：trust_remote_code，純文字路徑

在 12GB VRAM（RTX 4070，且桌面已佔用部分）限制下，支援 4-bit 量化載入，
使 3B/7B 模型也能塞入。此設計直接對應計畫書「資源受限下的可行性」論述。
"""

from dataclasses import dataclass, field
from typing import Optional

import torch


# 預設候選底座（計畫書 A 節：多語言、繁中友善、支援 PEFT）
PRESET_MODELS = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b":   "Qwen/Qwen2.5-3B-Instruct",
    "breeze-7b":    "MediaTek-Research/Breeze-7B-Instruct-v1_0",   # 純文字，標準架構
    "breeze2-3b":   "MediaTek-Research/Llama-Breeze2-3B-Instruct",  # InternVL VLM，需 remote code
    "tinyllama":    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}


@dataclass
class BackboneConfig:
    model: str = "qwen2.5-1.5b"     # preset 名或 HF repo id 或本地路徑
    quantization: Optional[str] = None   # None / "4bit" / "8bit"
    dtype: str = "float16"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens: int = 320
    temperature: float = 0.3         # 情緒判讀求穩，用低溫
    do_sample: bool = False          # 評估求可重現，預設 greedy
    trust_remote_code: bool = False  # Breeze2/InternVL 會自動開啟
    lora_path: Optional[str] = None  # 若提供，載入基座後套用此 LoRA adapter


class Backbone:
    def __init__(self, cfg: BackboneConfig):
        self.cfg = cfg
        self.model_id = PRESET_MODELS.get(cfg.model.lower(), cfg.model)
        self.is_internvl = "breeze2" in cfg.model.lower() or "internvl" in self.model_id.lower()
        self._load()
        if cfg.lora_path:
            self._apply_lora(cfg.lora_path)

    def _apply_lora(self, lora_path: str):
        from peft import PeftModel

        print(f"[backbone] 套用 LoRA adapter：{lora_path}")
        self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model.eval()

    # ------------------------------------------------------------------ #
    def _quant_config(self):
        if self.cfg.quantization not in ("4bit", "8bit"):
            return None
        from transformers import BitsAndBytesConfig
        if self.cfg.quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        return BitsAndBytesConfig(load_in_8bit=True)

    def _load(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

        trust = self.cfg.trust_remote_code or self.is_internvl
        dtype = getattr(torch, self.cfg.dtype)
        quant = self._quant_config()

        print(f"[backbone] 載入 {self.model_id} "
              f"(quant={self.cfg.quantization}, dtype={self.cfg.dtype}, device={self.cfg.device})")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=trust)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        common = dict(trust_remote_code=trust)
        if quant is not None:
            common["quantization_config"] = quant
            common["device_map"] = "auto"
        else:
            common["torch_dtype"] = dtype

        loader = AutoModel if self.is_internvl else AutoModelForCausalLM
        if quant is not None and self.is_internvl:
            # Breeze2 is a remote-code composite model. Transformers 4.44.2
            # correctly loads its config, but does not expose the outer model's
            # 4-bit marker early enough for Accelerate. On a single GPU,
            # Accelerate then calls model.to(), which quantized models reject.
            # Force dispatch hooks only for this load and restore the library
            # function immediately afterwards.
            import transformers.modeling_utils as modeling_utils

            original_dispatch = modeling_utils.dispatch_model

            def dispatch_quantized_with_hooks(model, *args, **kwargs):
                kwargs["force_hooks"] = True
                return original_dispatch(model, *args, **kwargs)

            modeling_utils.dispatch_model = dispatch_quantized_with_hooks
            try:
                self.model = loader.from_pretrained(self.model_id, **common)
            finally:
                modeling_utils.dispatch_model = original_dispatch
        else:
            self.model = loader.from_pretrained(self.model_id, **common)

        if quant is None and self.cfg.device == "cuda":
            self.model = self.model.to("cuda")
        self.model.eval()

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def chat(self, system: str, user: str, max_new_tokens: Optional[int] = None) -> str:
        """對話式單輪生成，回傳「新產生」的文字（不含 prompt）。"""
        max_new = max_new_tokens or self.cfg.max_new_tokens

        # InternVL/Breeze2 走自己的 chat 介面（純文字，pixel_values=None）
        if self.is_internvl:
            return self._chat_internvl(system, user, max_new)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = (f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
                      if system else f"{user}\n")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=self.cfg.do_sample,
            temperature=self.cfg.temperature if self.cfg.do_sample else None,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _chat_internvl(self, system: str, user: str, max_new: int) -> str:
        """Breeze2 (InternVL) 純文字對話。介面依模型 remote code 而定，做 best-effort。"""
        gen_cfg = dict(max_new_tokens=max_new, do_sample=self.cfg.do_sample)
        prompt = f"{system}\n\n{user}" if system else user
        # InternVL chat 常見簽名：model.chat(tokenizer, pixel_values, question, generation_config, history)
        try:
            resp = self.model.chat(
                self.tokenizer, None, prompt, gen_cfg, history=None, return_history=False
            )
            if isinstance(resp, tuple):
                resp = resp[0]
            return str(resp).strip()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"Breeze2/InternVL 純文字 chat 介面呼叫失敗：{e}。"
                f"此模型 remote code 可能與目前 transformers 版本不相容，"
                f"建議改用 --model qwen2.5-3b 或 breeze-7b。"
            ) from e

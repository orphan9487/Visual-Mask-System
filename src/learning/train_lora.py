import torch
from peft import LoraConfig, get_peft_model
from diffusers import StableDiffusionPipeline

# 1. 載入底層凍結模型 W0
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
unet = pipe.unet

# 2. 定義 LoRA 配置 (A, B 矩陣設計)
config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    target_modules=["to_q", "to_v"], # 針對 Attention 層進行微調
    lora_dropout=0.05,
    bias="none"
)

# 3. 注入 LoRA 層並凍結 W0
lora_model = get_peft_model(unet, config)
lora_model.print_trainable_parameters()

# 4. 開始訓練 (此處簡化為邏輯流程)
# 訓練結束後，僅需儲存數十 MB 的 A, B 權重
lora_model.save_pretrained("./data/identity_db/user_henry_lora")
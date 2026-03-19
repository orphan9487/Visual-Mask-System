import torch
from diffusers import StableDiffusionPipeline

model_path = "runwayml/stable-diffusion-v1-5"
lora_path = "./models/ethan_mask_lora/pytorch_lora_weights_fixed.safetensors"

print("--- [Digital Mask Verification] ---")

# 1. 載入基礎模型 (使用 FP16 節省顯存)
pipe = StableDiffusionPipeline.from_pretrained(
    model_path, 
    torch_dtype=torch.float16
).to("cuda")

# 2. 強制關閉那個會把圖塗黑的安全檢查器
pipe.safety_checker = None 

# 3. 加載你剛剛練成的 LoRA
print(f"Loading LoRA weights from: {lora_path}")
pipe.load_lora_weights(lora_path)

# 4. 測試生成
prompt = "a photo of ethan_mask, crying, highly detailed, sadly, frustrated"
print(f"Generating image with prompt: {prompt}")

image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]

# 5. 儲存結果
output_file = "ethan_mask_test_result.png"
image.save(output_file)
print(f"Success! Result saved as {output_file}")
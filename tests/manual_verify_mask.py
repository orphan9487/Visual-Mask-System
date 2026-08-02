"""Manually render one LoRA verification image; requires the GPU stack."""

from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- 設定區 ---
model_id = "runwayml/stable-diffusion-v1-5"
# 確保指向的是『_fixed』的版本，否則會噴 Target modules 錯誤
lora_path = PROJECT_ROOT / "models" / "yourname_mask_lora" / "pytorch_lora_weights_fixed.safetensors"
# 使用你在 scripts/training/preprocess_data.py 設定的新觸發詞
trigger_word = "yourname_mask"
output_dir = PROJECT_ROOT / "output" / "current" / "verification"
output_name = "v2_mask_test_result.png"

def verify():
    print(f"🚀 正在啟動驗證程序，準備載入面具: {trigger_word}")

    # 1. 載入 Stable Diffusion 管線 (使用 FP16 節省顯存)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None, # 直接關閉，避免 NSFW 誤判導致黑圖
        requires_safety_checker=False
    ).to("cuda")

    # 2. 加載你剛剛煉成的 LoRA 權重
    if lora_path.exists():
        print(f"📦 偵測到權重檔案，正在注入低秩矩陣 (LoRA)...")
        pipe.load_lora_weights(str(lora_path))
    else:
        print(f"❌ 錯誤：找不到檔案 {lora_path}，請先執行修復腳本！")
        return

    # 3. 設定 Prompt (加入一些高品質標籤提高效果)
    prompt = f"a detailed photo of a cute cat, black cat, styled with {trigger_word}, 8k, masterpiece"
    negative_prompt = "human, person, human face, blurry, bad anatomy"

    print(f"🎨 正在生成影像，請稍候...")

    # 4. 執行推論
    # 使用 30 步 (Steps) 兼顧速度與品質，guidance_scale 7.5 是標準值
    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]

    # 5. 儲存結果
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    image.save(output_path)
    print(f"✅ 驗證完成！結果已儲存至: {output_path}")

if __name__ == "__main__":
    verify()

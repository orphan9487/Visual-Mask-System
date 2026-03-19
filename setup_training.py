import os
import requests

def setup_lora_env():
    print("--- [Visual Mask] Training Environment Setup ---")
    
    # 1. 下載官方訓練腳本
    script_url = "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/lora/train_text_to_image_lora.py"
    script_name = "train_text_to_image_lora.py"
    
    if not os.path.exists(script_name):
        print(f"Downloading {script_name}...")
        response = requests.get(script_url)
        with open(script_name, "wb") as f:
            f.write(response.content)
        print("Download successful.")
    else:
        print(f"{script_name} already exists.")

    # 2. 建立輸出模型資料夾
    output_dir = "./models/ethan_mask_lora"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 3. 檢查硬體加速庫
    try:
        import xformers
        import bitsandbytes
        print("Check: xformers and bitsandbytes are installed.")
    except ImportError:
        print("Warning: Missing optimization libraries. Please run:")
        print("pip install xformers bitsandbytes accelerate")

    print("\nNext Step: Run 'accelerate config' in your terminal, then use the training command.")

if __name__ == "__main__":
    setup_lora_env()
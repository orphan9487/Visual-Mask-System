import torch
from diffusers import StableDiffusionPipeline

def verify_system_readiness():
    # 1. 強化硬體檢查
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Checking Hardware... Using Device: {device}")
    
    if device == "cpu":
        print("!!! 警告: 系統未偵測到 CUDA，請確認 GPU 驅動與 PyTorch 版本 !!!")
    
    # 2. 引入基礎模型 (W0) 
    print("Loading Base Model (W0)...")
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    # 3. 引入穩定版測試 LoRA 
    print("Injecting Test LoRA (A, B Matrices)...")
    try:
        # 使用極其穩定的 LCM-LoRA 作為功能測試
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        print("LoRA Injection Successful!")
    except Exception as e:
        print(f"LoRA Injection Failed: {e}")
        return

    # 4. 執行端到端推論
    print("Generating Test Visual Feedback...")
    prompt = "a professional digital avatar, visual mask style, masterpiece"
    # CPU 模式下請將 steps 調低 (如 4-8 steps)，否則會跑很久
    steps = 4 if device == "cpu" else 20 
    image = pipe(prompt, num_inference_steps=steps).images[0]

    # 5. 儲存結果
    image.save("system_test_output.png")
    print(f"Success! Test image saved. Device used: {device}")

if __name__ == "__main__":
    verify_system_readiness()
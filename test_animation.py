import torch
import os
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif

def test_animate_diff_local():
    # 1. 硬體與路徑檢查
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Initializing Visual Mask Animation Engine ---")
    print(f"Device: {device}")

    model_path = "./models/motion_adapter/"
    if not os.path.exists(model_path):
        print(f"Error: 找不到路徑 {model_path}，請確保模型檔案已放入該資料夾。")
        return

    # 2. 載入本地運動適配器 (Motion Module)
    # 使用 local_files_only=True 繞過 Hugging Face 權限檢查
    print("Step 1: Loading Local Motion Adapter (Motion Priors)...")
    adapter = MotionAdapter.from_pretrained(
        model_path, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        local_files_only=True
    )

    # 3. 建立整合管線 (引入 SD v1.5 作為骨幹)
    print("Step 2: Building Pipeline with Stable Diffusion v1.5...")
    pipe = AnimateDiffPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        motion_adapter=adapter,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    # 4. 針對 Windows 11 / 中階顯卡進行顯存優化 (防止 OOM)
    if device == "cuda":
        print("Applying VRAM Optimizations...")
        pipe.enable_model_cpu_offload() # 視顯存大小開啟，如果你的顯存大於 12GB 可註解掉此行
        pipe.enable_vae_slicing()

    # 5. 設定調度器 (確保動作平滑)
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, 
        clip_sample=False, 
        timestep_spacing="linspace",
        steps_offset=1
    )

    # 6. 使用推理層建議的「關鍵字指令」進行生成
    # 這裡模擬「諷刺/無奈」的表情指令
    prompt = "1girl, rolling eyes, slight smirk, facepalm, visual mask style, high quality, masterpiece"
    negative_prompt = "bad quality, distorted, static, blurry, nsfw"

    print(f"Step 3: Generating Dynamic Mask (16 Frames)... Prompt: {prompt}")
    
    with torch.autocast("cuda"):
        output = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_frames=16,          # 生成 16 幀 (約 2 秒短片)
            guidance_scale=7.5,
            num_inference_steps=25, # 步數越多細節越好，但速度較慢
            generator=torch.manual_seed(42),
        )

    # 7. 匯出 GIF 結果
    print("Step 4: Exporting Result...")
    frames = output.frames[0]
    output_filename = "mask_animation_test.gif"
    export_to_gif(frames, output_filename)
    
    print("-" * 40)
    print(f"SUCCESS! Dynamic mask saved as '{output_filename}'.")
    print(f"Check your folder: {os.path.abspath(output_filename)}")

if __name__ == "__main__":
    test_animate_diff_local()
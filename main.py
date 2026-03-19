import torch
import os
from transformers import pipeline
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif

class VisualMaskSystem:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- [System Initializing] Using Device: {self.device} ---")
        
        # 1. 初始化推理層 (1.7B LLM)
        self.reasoning_model = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        self.reasoner = pipeline("text-generation", model=self.reasoning_model, device=0 if self.device=="cuda" else -1)

        # 2. 初始化生成層 (AnimateDiff)
        print("Loading Generation Components...")
        adapter_path = "./models/motion_adapter/"
        adapter = MotionAdapter.from_pretrained(adapter_path, torch_dtype=torch.float16, local_files_only=True)
        
        self.generator_pipe = AnimateDiffPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            motion_adapter=adapter,
            torch_dtype=torch.float16
        ).to(self.device)

        # 3. 載入 LoRA 面具權重 (Identity-DB)
        self.generator_pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        
        # 顯存優化
        self.generator_pipe.enable_model_cpu_offload()
        self.generator_pipe.enable_vae_slicing()

    def run_pipeline(self, user_input, context="General conversation"):
        print(f"\n[Step 1: Reasoning] Decoding Intent for: '{user_input}'")
        
        # M-CoT Prompt 設計
        prompt = f"<|im_start|>system\nTranslate social intent to: Rationale, Emotion, and Keywords Script.\n<|im_end|>\n" \
                 f"<|im_start|>user\nContext: {context}\nInput: {user_input}\n<|im_end|>\n<|im_start|>assistant\n"
        
        reasoning_out = self.reasoner(prompt, max_new_tokens=150)[0]['generated_text'].split("<|im_start|>assistant")[-1]
        print(f"--- Reasoning Logic ---\n{reasoning_out.strip()}")

        # 提取 Script 關鍵字 (簡單的字串處理)
        # 假設模型產出格式為 Script: [Keywords]
        try:
            visual_script = reasoning_out.split("Script:")[-1].strip()
        except:
            visual_script = "1girl, neutral expression, high quality"

        print(f"\n[Step 2: Generation] Animating Mask with Script: {visual_script}")
        
        # 執行生成 (16 幀動態)
        output = self.generator_pipe(
            prompt=f"{visual_script}, visual mask style, masterpiece",
            negative_prompt="bad anatomy, distorted, blurry, nsfw",
            num_frames=16,
            guidance_scale=7.5,
            num_inference_steps=20,
            generator=torch.manual_seed(42)
        )

        # 儲存結果
        output_path = "final_mask_output.gif"
        export_to_gif(output.frames[0], output_path)
        print(f"\n[Success] Visual Mask generated at: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    system = VisualMaskSystem()
    user_text = input("請輸入一段想轉換為面具的文字: ")
    system.run_pipeline(user_text)
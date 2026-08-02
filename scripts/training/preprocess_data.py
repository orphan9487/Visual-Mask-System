"""Prepare and caption identity images for LoRA training."""

import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

class DataPreprocessor:
    def __init__(self, trigger_word="ethan_mask"):
        self.trigger_word = trigger_word
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 引入 BLIP 模型進行自動打標
        print(f"Loading BLIP Captioner on {self.device}...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

    def process_images(self, input_dir, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"Processing images from: {input_dir}")
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(input_dir, filename)
                raw_image = Image.open(img_path).convert('RGB')

                # 1. 影像處理：正方形裁切並縮放至 512x512
                w, h = raw_image.size
                min_dim = min(w, h)
                left = (w - min_dim) / 2
                top = (h - min_dim) / 2
                raw_image = raw_image.crop((left, top, left + min_dim, top + min_dim))
                raw_image = raw_image.resize((512, 512), Image.LANCZOS)

                # 儲存處理後的影像
                base_name = os.path.splitext(filename)[0]
                save_path = os.path.join(output_dir, f"{base_name}.png")
                raw_image.save(save_path)

                # 2. 自動產出標籤
                inputs = self.processor(raw_image, return_tensors="pt").to(self.device, torch.float16 if self.device == "cuda" else torch.float32)
                out = self.model.generate(**inputs)
                caption = self.processor.decode(out[0], skip_special_tokens=True)

                # 加入觸發詞與視覺面具風格標籤
                final_caption = f"a photo of {self.trigger_word}, {caption}, visual mask style, high quality"
                
                with open(os.path.join(output_dir, f"{base_name}.txt"), "w") as f:
                    f.write(final_caption)
                
                print(f"Processed: {filename} -> Caption: {final_caption}")

if __name__ == "__main__":
    preprocessor = DataPreprocessor(trigger_word="person5805")
    preprocessor.process_images(
        r"C:/Ethan/Edu_proj/DataSets/CelebA/train_data/train_data_5805/10_id_5805",
        r"C:/Ethan/Edu_proj/DataSets/CelebA/train_data/train_data_5805_512/10_id_5805",
    )

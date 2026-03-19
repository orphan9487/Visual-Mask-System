import torch
from peft import PeftModel, PeftConfig
from diffusers import StableDiffusionPipeline

class IdentityManager:
    def __init__(self, base_model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model = StableDiffusionPipeline.from_pretrained(
            base_model_path, torch_dtype=torch.float16
        ).to(self.device)
        self.current_identity = None

    def load_mask(self, adapter_path, adapter_name):
        if self.current_identity is None:
            self.model = PeftModel.from_pretrained(
                self.base_model.unet, 
                adapter_path, 
                adapter_name=adapter_name
            )
            self.current_identity = adapter_name
        else:
            self.model.load_adapter(adapter_path, adapter_name=adapter_name)
            self.model.set_adapter(adapter_name)
            self.current_identity = adapter_name

    def generate_with_mask(self, prompt):
        image = self.base_model(prompt).images[0]
        return image

if __name__ == "__main__":
    manager = IdentityManager("runwayml/stable-diffusion-v1-5")
    
    manager.load_mask("./data/identity_db/user_a_lora", "persona_a")
    print(f"Current Identity: {manager.current_identity}")
    
    manager.load_mask("./data/identity_db/user_b_lora", "persona_b")
    print(f"Switched to: {manager.current_identity}")
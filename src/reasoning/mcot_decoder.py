import torch
from transformers import pipeline

class SemanticIntentDecoder:
    def __init__(self, model_id="meta-llama/Llama-3.2-1B-Instruct"):
        self.device = 0 if torch.cuda.is_available() else -1
        self.generator = pipeline("text-generation", model=model_id, device=self.device)

    def decode_intent(self, synthesized_context):
        # 建立 M-CoT 提示詞，強迫模型產出 Rationale
        prompt = f"""
        Analyze the following social context and user intent. 
        Context: {synthesized_context}
        
        Step 1: Determine if the tone is sincere or sarcastic (Rationale).
        Step 2: Output the final emotion tag and scene description.
        
        Result:
        """
        
        output = self.generator(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
        return output[0]['generated_text']

if __name__ == "__main__":
    decoder = SemanticIntentDecoder()
    test_context = "User history: Often jokes with friends. Current text: 'You are such a genius!'"
    result = decoder.decode_intent(test_context)
    print(result)
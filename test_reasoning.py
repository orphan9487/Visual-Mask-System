import torch
from transformers import pipeline

def test_reasoning_layer():
    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading Reasoning Model (1B Scale)... Device: {'cuda' if device == 0 else 'cpu'}")
    
    model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    
    generator = pipeline(
        "text-generation", 
        model=model_id, 
        device=device, 
        torch_dtype=torch.float16 if device == 0 else torch.float32
    )

    context = "User History: Often uses sarcasm with friends. Current Text: 'Oh great, I forgot my keys again. You are such a genius!'"
    
    prompt = f"""
    <|im_start|>system
    You are the 'Reasoning & Intelligence Layer' of the Visual Mask system.
    Task: Use Multi-modal Chain-of-Thought (M-CoT) to decode intent.
    <|im_end|>
    <|im_start|>user
    Context: {context}
    
    Step 1: Rationale (Analyze sincerity vs. sarcasm).
    Step 2: Emotion Tag.
    Step 3: Scene Description Script for AnimateDiff.
    <|im_end|>
    <|im_start|>assistant
    """

    print("Decoding Intent...")
    output = generator(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
    print("\n--- Reasoning Result ---")
    print(output[0]['generated_text'].split("<|im_start|>assistant")[-1].strip())

if __name__ == "__main__":
    test_reasoning_layer()
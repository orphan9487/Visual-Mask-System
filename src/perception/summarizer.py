import torch
from transformers import pipeline

class TextSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.device = 0 if torch.cuda.is_available() else -1
        self.summarizer = pipeline("summarization", model=model_name, device=self.device)

    def extract_salient_tokens(self, text, max_length=50, min_length=10):
        summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']

if __name__ == "__main__":
    summarizer = TextSummarizer()
    sample_text = "I forgot the keys again. You are unbelievable!"
    tokens = summarizer.extract_salient_tokens(sample_text)
    print(f"Original: {sample_text}")
    print(f"Salient Tokens: {tokens}")
    

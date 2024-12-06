import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from config import MODEL_PATH, DEVICE

def load_model():
    """
    Load the model from the saved path.
    """
    tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
    model = BartForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
    return model, tokenizer

def generate_summary(model, tokenizer, text):
    """
    Generate a summary for the given text using BART.
    """
    inputs = tokenizer(
        text, return_tensors="pt", max_length=1024, truncation=True
    ).to(DEVICE)
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=128,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

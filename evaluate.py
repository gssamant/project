# evaluate.py

from transformers import BartForConditionalGeneration, BartTokenizer
from datasets import load_dataset
import torch
from tqdm import tqdm
from config import MODEL_NAME, DEVICE, MAX_INPUT_LENGTH, MAX_SUMMARY_LENGTH

def load_trained_model(model_path):
    model = BartForConditionalGeneration.from_pretrained(model_path).to(DEVICE)
    tokenizer = BartTokenizer.from_pretrained(model_path)
    return model, tokenizer

def evaluate(model, tokenizer, dataset):
    model.eval()

    for i, example in tqdm(enumerate(dataset), desc="Evaluating", total=len(dataset)):
        input_text = example['content']
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT_LENGTH).to(DEVICE)
        
        summary_ids = model.generate(
            inputs['input_ids'], 
            max_length=MAX_SUMMARY_LENGTH, 
            num_beams=4, 
            length_penalty=2.0, 
            early_stopping=True
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        print(f"Input: {input_text}\nGenerated Summary: {summary}\n")

def main():
    model_path = './saved_model_20241206-135116'  # Adjust with your saved model path
    model, tokenizer = load_trained_model(model_path)
    dataset = load_dataset("reddit", split="train[:5%]").select(range(30))
    evaluate(model, tokenizer, dataset)

if __name__ == "__main__":
    main()

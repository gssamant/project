import torch
from torch.utils.data import DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration
from datasets import load_dataset
import os
import time
from tqdm import tqdm
from config import MODEL_PATH, DEVICE, MAX_INPUT_LENGTH, MAX_SUMMARY_LENGTH

# Set the dataset cache path
os.environ["HF_DATASETS_CACHE"] = r"D:\Study Material\Learning\project\datasets"

# Load dataset
def load_data():
    dataset = load_dataset("reddit", split="train", trust_remote_code=True)
    dataset = dataset.select(range(30)) 
    return dataset

# Tokenizer and model initialization
def load_model():
    tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
    model = BartForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
    return model, tokenizer

# Tokenization and padding with parallelization
def preprocess_data(dataset, tokenizer):
    def tokenize_function(examples):
        inputs = tokenizer(
            examples['content'], truncation=True, padding='max_length', max_length=MAX_INPUT_LENGTH
        )
        targets = tokenizer(
            examples['summary'], truncation=True, padding='max_length', max_length=MAX_SUMMARY_LENGTH
        )
        inputs['labels'] = targets['input_ids']
        return inputs

    num_workers = os.cpu_count()
    dataset = dataset.map(tokenize_function, batched=True, num_proc=num_workers)
    dataset.save_to_disk('D:/Study Material/Learning/project/tokenized_data')
    return dataset

# Fine-tuning loop 
def train(model, tokenizer, dataset):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    checkpoint_path = "checkpoint.pt"
    start_epoch = 0

    # Load checkpoint if it exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {start_epoch}...")

    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Training loop
    for epoch in range(start_epoch, 5): 
        model.train()
        total_loss = 0

        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
            inputs = torch.stack([torch.tensor(x) for x in batch['input_ids']]).to(DEVICE)
            labels = torch.stack([torch.tensor(x) for x in batch['labels']]).to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids=inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_dataloader)}")

        # Save checkpoint after each epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved for epoch {epoch+1}.")

    # Save the final model
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model.save_pretrained(f"./saved_model_{timestamp}")
    tokenizer.save_pretrained(f"./saved_model_{timestamp}")
    print(f"Final model saved as saved_model_{timestamp}")

def main():
    dataset = load_data()
    model, tokenizer = load_model()
    dataset = preprocess_data(dataset, tokenizer)
    train(model, tokenizer, dataset)

if __name__ == "__main__":
    main()

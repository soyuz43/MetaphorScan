import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Define model paths
MODEL_DIR = "data/models/transformers/distilbert-base-uncased"
MODEL_NAME = "distilbert-base-uncased"

def download_distilbert():
    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Download and save model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Save to local directory
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Downloaded and saved {MODEL_NAME} to {MODEL_DIR}")

if __name__ == "__main__":
    download_distilbert()

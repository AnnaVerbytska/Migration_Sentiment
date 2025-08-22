# aspect_based_sentiment_analysis.py

# Import libraries
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from tqdm import tqdm
import time
import sys
import os

# Setting secret credentials (optional, for consistency)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from dotenv import load_dotenv
load_dotenv()

# Optional custom preprocessing
# sys.path.append('../src')
# from preprocess import clean_text

# Using the yangheng model with the fix
model_name = "yangheng/deberta-v3-base-absa-v1.1"
# The fix is to add use_fast=False to prevent the tokenizer conversion error
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# Use Apple M1 acceleration or CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Label order to match the yangheng model's configuration 
labels = ['negative', 'neutral', 'positive']

# Accepts texts and keywords (aspects)
def predict_sentiment_batched(texts, keywords, batch_size=32):
    """
    Predicts sentiment towards a specific keyword (aspect) in each text.
    """
    sentiments = []
    confidences = []
    n = len(texts)

    for i in tqdm(range(0, n, batch_size), desc="Predicting Aspect-Based Sentiment"):
        batch_texts = texts[i:i+batch_size]
        batch_keywords = keywords[i:i+batch_size]

        # Prepare inputs in the format required by the ABSA model: [CLS] text [SEP] aspect [SEP]
        inputs = tokenizer(
            batch_texts,
            batch_keywords,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            confs, preds = torch.max(probs, dim=1)

            for pred, conf in zip(preds, confs):
                sentiments.append(labels[pred.item()])
                confidences.append(round(conf.item(), 4))

    return sentiments, confidences


# --- Main Execution Block ---
if __name__ == '__main__':
    # --- IMPROVED FILE HANDLING ---
    # Define file paths
    input_path_initial = '../data/reddit_predicted_sentiment.csv'
    output_path_final = '../data/reddit_predicted_sentiment.csv'

    # Load the existing predictions if the file exists, otherwise load the initial data
    if os.path.exists(output_path_final):
        print(f"Loading existing predictions from {output_path_final}...")
        df = pd.read_csv(output_path_final)
    else:
        print(f"Loading initial data from {input_path_initial}...")
        df = pd.read_csv(input_path_initial)
    
    # Optional: Apply text cleaning
    # df['text'] = df['text'].apply(clean_text)

    # Ensure keyword column exists
    if 'keyword' not in df.columns:
        raise ValueError("Input CSV must contain a 'keyword' column.")

    # --- CHECK TO PREVENT RE-RUNNING ---
    # Changed column name to be model-specific
    if 'yangheng_sentiment' in df.columns:
        print("\nYangheng sentiment analysis has already been run. Skipping.")
    else:
        # Pass both text and keyword columns to the function
        print("\nStarting deBERTa sentiment analysis...")
        start = time.time()
        # Use clearer, model-specific column names
        df['deberta_sentiment'], df['deberta_confidence'] = predict_sentiment_batched(
            df['text'].tolist(),
            df['keyword'].tolist()
        )
        print(f"Runtime for {len(df)} posts: {time.time() - start:.2f} seconds")

        # Save predictions back to the same file
        df.to_csv(output_path_final, index=False)
        print(f"\n✅ Predictions updated and saved to {output_path_final}")

    print("\n--- Final DataFrame Head ---")
    print(df.head())

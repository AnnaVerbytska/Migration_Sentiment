import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from datasets import load_dataset
import time

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from dotenv import load_dotenv
load_dotenv()

# Optional custom preprocessing
sys.path.append('../src')
from preprocess import clean_text

# Load CardiffNLP model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# Use Apple M1 acceleration or CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Label order used by the model
labels = ['negative', 'neutral', 'positive']
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {v: k for k, v in label2id.items()}

# Sentiment prediction function
def predict_sentiment_batched(texts, batch_size=32):
    sentiments = []
    confidences = []
    n = len(texts)

    for i in tqdm(range(0, n, batch_size), desc="Predicting Sentiment in Batches"):
        batch_texts = texts[i:i+batch_size]
        clean_batch = [text if isinstance(text, str) and text.strip() != "" else "" for text in batch_texts]

        inputs = tokenizer(clean_batch, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            confs, preds = torch.max(probs, dim=1)

            for pred, conf, text in zip(preds, confs, clean_batch):
                if text == "":
                    sentiments.append(None)
                    confidences.append(None)
                else:
                    sentiments.append(labels[pred.item()])
                    confidences.append(round(conf.item(), 4))

    return sentiments, confidences


# ----------- PART 1: Predict on Your Custom Unlabeled Dataset ----------
df = pd.read_csv('../data/reddit_labels.csv')
df['text'] = df['text'].apply(clean_text)

start = time.time()
df['cardiffnlp_sentiment'], df['cardiffnlp_confidence'] = predict_sentiment_batched(df['text'].tolist())
print(f"Runtime for 100 posts: {time.time() - start:.2f} seconds")

# Save predictions
df.to_csv('../data/reddit_predicted_sentiment.csv', index=False)
print("\n✅ Predictions on custom dataset saved to reddit_predicted_sentiment.csv")
print(df[['text', 'cardiffnlp_sentiment', 'cardiffnlp_confidence']].head())




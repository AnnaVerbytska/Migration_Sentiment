# irr_analysis_and_visualization.py

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from scipy.stats import pearsonr

def interpret_kappa(kappa_score):
    """Provides a standard interpretation for a Cohen's Kappa score."""
    if kappa_score < 0.0:
        return "Poor agreement (less than chance)"
    elif kappa_score < 0.20:
        return "Slight agreement"
    elif kappa_score < 0.40:
        return "Fair agreement"
    elif kappa_score < 0.60:
        return "Moderate agreement"
    elif kappa_score < 0.80:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"

def interpret_pearson(r_value):
    """Provides a standard interpretation for a Pearson correlation coefficient."""
    abs_r = abs(r_value)
    if abs_r < 0.1:
        return "Negligible correlation"
    elif abs_r < 0.3:
        return "Weak correlation"
    elif abs_r < 0.5:
        return "Moderate correlation"
    else:
        return "Strong correlation"

def run_analysis_and_visualization(filepath):
    """
    Loads a prediction file, calculates IRR, and creates visualizations for model agreement.
    """
    # --- 1. Load Data and Validate ---
    if not os.path.exists(filepath):
        print(f"Error: The file was not found at '{filepath}'")
        return

    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows from {filepath}")

    # Define the columns needed for the analysis
    sentiment_cols = ['gemini_sentiment', 'deberta_sentiment'] 
    confidence_cols = ['gemini_confidence', 'deberta_confidence']
    required_cols = sentiment_cols + confidence_cols

    if not all(col in df.columns for col in required_cols):
        print(f"Error: The CSV must contain the following columns: {required_cols}")
        return

    # --- FIX: Use a mapping for robust standardization ---
    print("\nStandardizing sentiment labels...")
    sentiment_map = {
        'Positive': 'positive', 'positive': 'positive',
        'Neutral': 'neutral', 'neutral': 'neutral',
        'Negative': 'negative', 'negative': 'negative'
        # Add any other variations you find in your data here
    }
    for col in sentiment_cols:
        # The .map() function will apply the standardization.
        # It's more robust than just converting to lowercase.
        df[col] = df[col].map(sentiment_map)
    # --- End of fix ---

    # --- 2. Analyze and Visualize Categorical Sentiment Agreement ---
    print("\n--- Agreement on Sentiment Labels (Categorical) ---")
    
    df_sent = df.dropna(subset=sentiment_cols)
    
    # Calculate and print Cohen's Kappa
    kappa = cohen_kappa_score(df_sent[sentiment_cols[0]], df_sent[sentiment_cols[1]])
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"Interpretation: {interpret_kappa(kappa)}")
    
    # Generate and display the Confusion Matrix
    present_labels = sorted(list(set(df_sent[sentiment_cols[0]].unique()) | set(df_sent[sentiment_cols[1]].unique())))
    cm = confusion_matrix(df_sent[sentiment_cols[0]], df_sent[sentiment_cols[1]], labels=present_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=present_labels, yticklabels=present_labels)
    plt.title('Confusion Matrix: Model Agreement on Sentiment Labels', fontsize=16)
    plt.ylabel(f'{sentiment_cols[0]} (Model 1)', fontsize=12)
    plt.xlabel(f'{sentiment_cols[1]} (Model 2)', fontsize=12)
    plt.show()

    # --- 3. Analyze and Visualize Continuous Confidence Score Correlation ---
    print("\n--- Agreement on Confidence Scores (Continuous) ---")
    
    df_conf = df.dropna(subset=confidence_cols)

    # Calculate and print Pearson Correlation
    r_value, p_value = pearsonr(df_conf[confidence_cols[0]], df_conf[confidence_cols[1]])
    print(f"Pearson Correlation Coefficient (r): {r_value:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Interpretation: {interpret_pearson(r_value)}")
    if p_value < 0.05:
        print("The correlation is statistically significant (p < 0.05).")
    else:
        print("The correlation is not statistically significant (p >= 0.05).")

    # Generate and display the Scatter Plot
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df_conf, x=confidence_cols[0], y=confidence_cols[1], alpha=0.6)
    plt.title('Correlation of Model Confidence Scores', fontsize=16)
    plt.xlabel(f'{confidence_cols[0]} Confidence', fontsize=12)
    plt.ylabel(f'{confidence_cols[1]} Confidence', fontsize=12)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--') # Add a line for perfect correlation
    plt.grid(True)
    plt.show()


# --- Main Execution Block ---
if __name__ == '__main__':
    # Path to the file containing predictions from all your models
    predictions_file = '../data/reddit_predicted_sentiment.csv'
    
    run_analysis_and_visualization(predictions_file)

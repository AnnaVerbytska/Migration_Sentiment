# Import libraries
import pandas as pd
from typing import Dict
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
# Setting secret credentials
from dotenv import load_dotenv #pip install python-dotenv
load_dotenv()
sys.path.append('../src')
# Import feature engineering functions
from preprocess import clean_text

# load the dataset
df = pd.read_csv('../data/reddit_raw.csv')
# Preprocess the text data
df['text'] = df['text'].apply(clean_text)

from google import genai
from google.genai import types # pip install google-genai==1.7.0

from IPython.display import HTML, Markdown, display

from google.api_core import retry # pip install google-api-core
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})
genai.models.Models.generate_content = retry.Retry(
    predicate=is_retriable)(genai.models.Models.generate_content)

# Set up your API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini client
client = genai.Client(api_key=GOOGLE_API_KEY)


def analyze_sentiment_gemini(text: str, max_output_tokens=50) -> Dict[str, float]:
    """
    Performs sentiment analysis on a single text using Gemini 2.0 Flash.

    Args:
        text (str): The input text to analyze.
        max_output_tokens (int): Max tokens for Gemini output.

    Returns:
        Dict[str, Any]: Dictionary with keys 'sentiment' (str) and 'confidence' (float).
    """
    try:
        prompt = (
            "Classify the sentiment of the following text as Positive, Negative, or Neutral. "
            "Respond ONLY with the sentiment label and confidence score as a decimal between 0 and 1, "
            "in JSON format like this: {\"sentiment\": \"Positive\", \"confidence\": 0.87}\n\n"
            f"Text:\n{text}"
        )
        
        config = types.GenerateContentConfig(temperature=0.0, max_output_tokens=max_output_tokens)

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=config
        )

        # Expecting JSON output, so parse it
        import json
        result = json.loads(response.text.strip())

        # Validate and normalize keys (optional)
        sentiment = result.get('sentiment', 'Neutral').capitalize()
        confidence = float(result.get('confidence', 0.0))

        return {
            'sentiment': sentiment,
            'confidence': confidence
        }

    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return {
            'sentiment': 'Neutral',
            'confidence': 0.0
        }


def analyze_sentiments_in_df(df: pd.DataFrame, text_col='text') -> pd.DataFrame:
    """
    Applies Gemini sentiment analysis to all texts in a DataFrame column and adds results.

    Args:
        df (pd.DataFrame): Input DataFrame.
        text_col (str): Name of the column with text to analyze.

    Returns:
        pd.DataFrame: Original DataFrame with two new columns:
            - 'gemini_sentiment' (str)
            - 'gemini_confidence' (float)
    """
    def extract_sentiment(text):
        result = analyze_sentiment_gemini(text)
        return pd.Series([result['sentiment'], result['confidence']])
    
    df[['gemini_sentiment', 'gemini_confidence']] = df[text_col].apply(extract_sentiment, result_type='expand')
    return df

# Perform sentiment analysis on the dataset
df = analyze_sentiments_in_df(df, text_col='text')
print(df[['text', 'gemini_sentiment', 'gemini_confidence']].head())

# Save the results to a new CSV file
df.to_csv('../data/reddit_sentiment_predicted.csv', index=False)
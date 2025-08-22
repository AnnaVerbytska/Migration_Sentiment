# generate_gemini_sentiment.py

# Import libraries
import pandas as pd
from typing import Dict
import sys
import os
from tqdm import tqdm  # For progress bars
import json # Import json at the top

# Setting secret credentials
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from dotenv import load_dotenv  # pip install python-dotenv
load_dotenv()

# Optional custom preprocessing
# sys.path.append('../src')
# from preprocess import clean_text

# Import Google GenAI libraries
from google import genai
from google.genai import types
from google.api_core import retry

# --- Global Setup ---
# This section is for configurations that are set once.

# Configure retry logic for API calls to handle temporary errors
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})
genai.models.Models.generate_content = retry.Retry(
    predicate=is_retriable)(genai.models.Models.generate_content)

def setup_gemini_client():
    """
    Sets up and initializes the Gemini client.
    Returns the client object or raises an error if the API key is missing.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Error: GOOGLE_API_KEY not found in your .env file or environment variables.")
    return genai.Client(api_key=api_key)

def analyze_sentiment_gemini(client, text: str, aspect: str, max_output_tokens=50) -> Dict[str, float]:
    """
    Performs Aspect-Based Sentiment Analysis on a single text using Gemini.

    Args:
        client: The initialized Gemini client.
        text (str): The input text to analyze.
        aspect (str): The specific keyword or topic to find the sentiment for.
        max_output_tokens (int): Max tokens for Gemini output.

    Returns:
        Dict[str, Any]: Dictionary with keys 'sentiment' (str) and 'confidence' (float).
    """
    # Handle empty or invalid text input gracefully
    if not isinstance(text, str) or not text.strip() or not isinstance(aspect, str) or not aspect.strip():
        return {'sentiment': None, 'confidence': None}

    try:
        # --- MODIFIED PROMPT FOR ASPECT-BASED SENTIMENT ---
        prompt = (
            f"What is the sentiment towards '{aspect}' in the following text? "
            "Classify it as Positive, Negative, or Neutral. "
            "Respond ONLY with the sentiment label and confidence score as a decimal between 0 and 1, "
            "in JSON format like this: {\"sentiment\": \"Positive\", \"confidence\": 0.87}\n\n"
            f"Text:\n{text}"
        )
        
        config = types.GenerateContentConfig(temperature=0.0, max_output_tokens=max_output_tokens)

        response = client.models.generate_content(
            model="gemini-2.0-flash", # Using a specific, recent model
            contents=prompt,
            config=config
        )

        # --- MODIFIED SECTION TO FIX JSON ERROR ---
        response_text = response.text.strip()
        
        # More robustly clean the response to remove Markdown code fences
        if response_text.startswith("```") and response_text.endswith("```"):
            response_text = response_text.strip("```").strip()
            # Also remove the language identifier (like 'json') if it exists
            if response_text.startswith("json"):
                 response_text = response_text.strip("json").strip()
        
        # Attempt to parse the JSON, with specific error handling
        try:
            result = json.loads(response_text)
            sentiment = result.get('sentiment', 'Neutral').capitalize()
            confidence = float(result.get('confidence', 0.0))
            return {
                'sentiment': sentiment,
                'confidence': confidence
            }
        except json.JSONDecodeError:
            print(f"\nWarning: Failed to decode JSON. Model returned non-JSON response: '{response_text}'")
            return {
                'sentiment': 'JSON Error',
                'confidence': 0.0
            }
        # --- END OF MODIFIED SECTION ---

    except Exception as e:
        print(f"\nError processing text: '{text[:50]}...'. Error: {e}")
        return {
            'sentiment': 'API Error',
            'confidence': 0.0
        }

def analyze_sentiments_in_df(client, df: pd.DataFrame, text_col='text', keyword_col='keyword') -> pd.DataFrame:
    """
    Applies Gemini ABSA to a DataFrame, using a different keyword for each row.

    Args:
        client: The initialized Gemini client.
        df (pd.DataFrame): Input DataFrame with text and keyword columns.
        text_col (str): Name of the column with text to analyze.
        keyword_col (str): Name of the column with the keyword for each text.

    Returns:
        pd.DataFrame: Original DataFrame with two new columns:
            - 'gemini_sentiment' (str)
            - 'gemini_confidence' (float)
    """
    # Set up tqdm to work with pandas.apply on rows
    tqdm.pandas(desc="Analyzing Sentiment per Keyword")

    # Define a function to apply to each row of the DataFrame
    def extract_sentiment_for_row(row):
        text = row[text_col]
        aspect = row[keyword_col]
        return analyze_sentiment_gemini(client, text, aspect)

    # Apply the function row-wise (axis=1)
    results = df.progress_apply(extract_sentiment_for_row, axis=1)
    
    # Unpack the results into two new columns
    df['gemini_sentiment'] = [res['sentiment'] for res in results]
    df['gemini_confidence'] = [res['confidence'] for res in results]
    
    return df

# --- Main Execution Block ---
# This block runs only when the script is executed directly.
if __name__ == '__main__':
    try:
        # 1. Initialize Gemini client
        gemini_client = setup_gemini_client()
        print("Gemini client initialized successfully.")

        # 2. Load the dataset
        input_path = '../data/reddit_cleaned.csv'
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} rows from {input_path}")
        
        # Ensure the 'keyword' column exists
        if 'keyword' not in df.columns:
            raise ValueError("Error: The input CSV must contain a 'keyword' column.")

        # 3. Perform sentiment analysis on the dataset
        df_analyzed = analyze_sentiments_in_df(gemini_client, df, text_col='text', keyword_col='keyword')
        
        print("\n--- Analysis Complete ---")
        print(df_analyzed[['text', 'keyword', 'gemini_sentiment', 'gemini_confidence']].head())

        # 4. Save the full, updated DataFrame to a new CSV file
        output_path = '../data/reddit_predicted_sentiment.csv'
        df_analyzed.to_csv(output_path, index=False)
        print(f"\n✅ Results for all {len(df_analyzed)} rows saved to {output_path}")

    except Exception as e:
        print(f"\nAn error occurred during the main process: {e}")

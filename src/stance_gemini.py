# advanced_stance_extraction_full.py

import os
import pandas as pd
from google import genai
from google.genai import types
from google.api_core import retry
import sys
from dotenv import load_dotenv
import json
from tqdm import tqdm

# Load environment variables from a .env file
load_dotenv()

def setup_gemini_client():
    """Sets up and configures the Gemini client with an API key and retry logic."""
    is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})
    genai.models.Models.generate_content = retry.Retry(predicate=is_retriable)(genai.models.Models.generate_content)
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Error: GOOGLE_API_KEY not found.")
    return genai.Client(api_key=api_key)

def extract_stance_pairs(client, text: str) -> list:
    """
    Uses Gemini to perform target-based stance detection with an intensity score.

    Args:
        client: The initialized Gemini client.
        text (str): The input text to analyze.

    Returns:
        A list of dictionaries, e.g., 
        [{"target": "refugees", "stance": "Supportive", "intensity": 4}],
        or an empty list if an error occurs.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    prompt = f"""
    You are an expert research assistant. Your task is to perform a detailed stance analysis on the following Reddit post.

    First, identify all the key targets or entities being discussed. These could be people (refugees, politicians), concepts (immigration policy), or groups (host countries, media).

    Second, for EACH target you identify, determine the author's stance towards it. The stance must be one of:
    - Supportive
    - Critical
    - Neutral

    Third, rate the INTENSITY of that stance on a scale from 1 (very mild) to 5 (very strong). A 'Neutral' stance must have an intensity of 0.

    Respond ONLY with a JSON object containing a list of your findings. Each item in the list should be an object with a "target", a "stance", and an "intensity". If no clear stance is taken, do not include the target.

    Example Response Format:
    {{
      "analysis": [
        {{"target": "refugees", "stance": "Supportive", "intensity": 4}},
        {{"target": "government policy", "stance": "Critical", "intensity": 5}},
        {{"target": "the media", "stance": "Neutral", "intensity": 0}}
      ]
    }}

    Reddit Post:
    ---
    {text}
    ---
    """
    
    try:
        config = types.GenerateContentConfig(temperature=0.0, max_output_tokens=500)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=config
        )
        
        response_text = response.text.strip()
        
        if response_text.startswith("```") and response_text.endswith("```"):
            response_text = response_text.strip("```").strip("json").strip()

        data = json.loads(response_text)
        
        return data.get("analysis", [])

    except Exception as e:
        print(f"\nError processing text: '{text[:50]}...'. Error: {e}")
        return []


if __name__ == '__main__':
    try:
        gemini_client = setup_gemini_client()
        print("Gemini client initialized successfully.")

        input_path = '../data/reddit_labels.csv'
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} rows from {input_path}")

        all_results = []

        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Analyzing Posts"):
            post_text = row['text']
            stance_pairs = extract_stance_pairs(gemini_client, post_text)
            
            # --- THIS IS THE UPDATED SECTION ---
            # It now copies all original columns and adds the new analysis columns.
            for pair in stance_pairs:
                # 1. Copy all columns from the original row into a dictionary
                new_row_data = row.to_dict()
    
                # 2. Add the new target, stance, and intensity information to it
                new_row_data['target'] = pair.get('target')
                new_row_data['stance'] = pair.get('stance')
                new_row_data['confidence_intensity'] = pair.get('intensity')
    
                # 3. Append the complete dictionary to our master list
                all_results.append(new_row_data)
            # --- END OF UPDATED SECTION ---

        final_df = pd.DataFrame(all_results)

        print("\n--- Analysis Complete ---")
        print("Final DataFrame Head:")
        print(final_df.head())

        output_path = '../data/reddit_stance_analysis_full.csv'
        final_df.to_csv(output_path, index=False)
        print(f"\n✅ Full analysis saved to {output_path}")

    except Exception as e:
        print(f"\nAn error occurred during the main process: {e}")
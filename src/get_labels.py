# generate_gemini_labels.py
import os
import pandas as pd
from google import genai
from google.genai import types
from google.api_core import retry
# --- Added for .env support ---
import sys
from dotenv import load_dotenv
load_dotenv()
# --- End of addition ---

def setup_gemini_client():
    """
    Sets up and configures the Gemini client with an API key and retry logic.
    Returns the configured client object or None if the API key is not found.
    """
    # Set up retry logic for API calls
    is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})
    genai.models.Models.generate_content = retry.Retry(predicate=is_retriable)(genai.models.Models.generate_content)
    
    # Attempt to get the API key from environment variables (now loaded by dotenv)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in your .env file or environment variables.")
        print("Please set your API key to run this script.")
        return None
        
    # Initialize and return the Gemini client
    return genai.Client(api_key=api_key)

def generate_labels_with_gemini(client, posts_list: list) -> list:
    """
    Uses Gemini to analyze a list of texts and suggest classification labels.

    Parameters:
    -----------
    client: genai.Client
        The initialized Gemini client.
    posts_list: list
        A list of strings, where each string is a Reddit post.

    Returns:
    --------
    list: A list of suggested string labels, or an empty list if an error occurs.
    """
    if not client:
        return []

    # Combine all posts into a single block of text for the prompt
    all_posts_text = "\n\n---\n\n".join(posts_list)

    prompt = f"""
    Your task is to identify universal, generalized themes from a list of Reddit posts about migration and refugees.

    The goal is to create a set of universal labels that can be used to compare discussions about refugees from different conflicts (e.g., Ukraine, Gaza). 
    Therefore, the labels MUST be general and not mention specific countries, nationalities, or locations.

    Based on the posts provided below, please do the following:

    1. Read all the posts to understand the different viewpoints.
    2. Identify 3-4 common, distinct, and GENERALIZED themes. Focus on the *type* of argument (e.g., humanitarian, security, political), not the specific conflict.
    3. Suggest a concise, descriptive label for each theme. For example, good labels would be 'Humanitarian support for refugees' or 'Concerns about security and economic impact'.
    4. Include an "Unrelated or off-topic" label as a catch-all category.
    5. Present the final list of suggested labels as a simple Python list of strings, like this: ['Label 1', 'Label 2', 'Label 3']
    Here are the Reddit posts:
    ---
    {all_posts_text}
    ---
    """

    print("Sending request to Gemini to generate labels...")
    try:
        config = types.GenerateContentConfig(temperature=0.1, max_output_tokens=200)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=config
        )
        response_text = response.text.strip()
        print(f"Gemini's raw response: {response_text}")

        if response_text.startswith("```python"):
            response_text = response_text.strip("```python").strip()
        elif response_text.startswith("```"):
            response_text = response_text.strip("```").strip()

        import ast
        suggested_labels = ast.literal_eval(response_text)
        
        if isinstance(suggested_labels, list):
            return suggested_labels
        else:
            print("Error: Gemini did not return a valid list of labels.")
            return []

    except Exception as e:
        print(f"An error occurred while generating labels: {e}")
        return []

def classify_post_with_gemini(client, post: str, labels: list) -> str:
    """
    Classifies a single post using a provided list of labels.
    """
    prompt = f"""
    Classify the following Reddit post into ONE of the following categories:
    {', '.join(labels)}

    Respond ONLY with the single most appropriate category name.

    Reddit Post:
    ---
    {post}
    ---
    """
    try:
        config = types.GenerateContentConfig(temperature=0.0, max_output_tokens=50)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=config
        )
        classification = response.text.strip()
        return classification if classification in labels else "Classification Error"
    except Exception as e:
        print(f"An error occurred during classification: {e}")
        return "Classification Error"


if __name__ == '__main__':
    # --- Main Execution Block ---
    
    gemini_client = setup_gemini_client()

    if gemini_client:
        input_csv_path = '../data/reddit_cleaned.csv'
        df = None # Initialize df to None
        
        try:
            if os.path.exists(input_csv_path):
                df = pd.read_csv(input_csv_path)
                if 'text' not in df.columns or df.empty:
                    print(f"Error: CSV file '{input_csv_path}' is empty or does not have a 'text' column.")
                    df = None # Invalidate df
            else:
                print(f"Error: Input file not found at '{input_csv_path}'")
        except Exception as e:
            print(f"An error occurred while reading the CSV file: {e}")

        if df is not None:
            # <<< CHANGED SECTION START >>>
            # Use the entire DataFrame to generate labels
            all_posts = df['text'].dropna().tolist()
            print(f"Successfully loaded {len(df)} posts. Using all {len(all_posts)} posts to generate labels.")

            custom_labels = generate_labels_with_gemini(gemini_client, all_posts)

            if custom_labels:
                print("\n--- Suggested Labels for Zero-Shot Classification ---")
                print(custom_labels)

                print(f"\nClassifying all {len(df.dropna(subset=['text']))} posts with the new labels...")
                classifications = []
                # Classify every post in the original dataframe
                for post in df['text'].dropna():
                    classification = classify_post_with_gemini(gemini_client, post, custom_labels)
                    classifications.append(classification)
                    print(f".", end="", flush=True) # Progress indicator
                
                print("\nClassification complete.")

                # Create a copy of the original DF to avoid SettingWithCopyWarning
                df_labeled = df.dropna(subset=['text']).copy()
                # Add the new column with the name 'label'
                df_labeled['label'] = classifications
                
                # Save the entire DataFrame with the new label column
                output_path = '../data/reddit_labels.csv'
                try:
                    df_labeled.to_csv(output_path, index=False)
                    print(f"\nSuccessfully saved the entire DataFrame with labels to {output_path}")
                    print("\n--- Final DataFrame Head ---")
                    print(df_labeled.head())
                except Exception as e:
                    print(f"\nCould not save final DataFrame to file: {e}")
            else:
                print("\nCould not generate labels. Halting process.")
        

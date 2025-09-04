# File: dashboard.py

import dash
from dash import dcc, html
import pandas as pd
import os
import io
import base64
import matplotlib.pyplot as plt

# --- Use a relative import for local modules ---
from .visualisations import (plot_stance_and_intensity_summary)
# --- 1. App Setup ---

# --- Create a robust, absolute path to the data file ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'reddit_stance_analysis_full_with_groups.csv')
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: The data file was not found at the expected path: {DATA_PATH}")
    df = pd.DataFrame()

# Initialise Dash app
app = dash.Dash(__name__)
# Expose the server variable for Gunicorn
server = app.server

# --- 2. App Layout ---
app.layout = html.Div([
    html.H1("Migration as a Strategic Narrative in War", style={'textAlign': 'center', 'fontFamily': 'Arial'}),

    # Stance & Intensity Across Target Groups and Subreddits
    html.H3("How do the stances and intensity levels for the target groups compare between the two subreddits?", style={'textAlign': 'center', 'fontFamily': 'Arial'}),
    dcc.Graph(figure=plot_stance_and_intensity_summary(df, category_col='target_label', subreddit_col='subreddit'))

    ])

# --- 3. Run the App ---
if __name__ == '__main__':
    app.run(debug=True)


# File: dashboard.py

import dash
from dash import dcc, html
import pandas as pd
import sys
import os

# --- Use a relative import for local modules ---
from .visualisations import plot_top_targets_by_subreddit, plot_score_distribution, plot_intensity_correlation

# --- 1. App Setup ---

# --- Create a robust, absolute path to the data file ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'reddit_stance_analysis_full.csv')
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
    
    # Target Distribution Graph
    html.H3("Which migration-related targets are most discussed?", style={'textAlign': 'center', 'fontFamily': 'Arial'}),
    dcc.Graph(figure=plot_top_targets_by_subreddit(df, top_n=25)),

    # Engagement Score Distribution by Stance and Subreddit
    html.H3("Engagement Score Distribution by Stance and Subreddit", style={'textAlign': 'center', 'fontFamily': 'Arial'}),
    dcc.Graph(figure=plot_score_distribution(df))

])

# --- 3. Run the App ---
if __name__ == '__main__':
    app.run(debug=True)

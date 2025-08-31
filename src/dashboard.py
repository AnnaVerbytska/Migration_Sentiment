# File: dashboard.py

import dash
from dash import dcc, html
import pandas as pd
import os
import io
import base64
import matplotlib.pyplot as plt

# --- Use a relative import for local modules ---
from .visualisations import (
    plot_top_targets_by_subreddit,
    plot_stance_heatmap_by_subreddit,
    plot_target_group_proportions,
    plot_stance_and_intensity_summary,
    plot_most_polarized_targets
    
)
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

    # Subheading for Target and Stance Analysis
    html.H2("Stance and Target Analysis", style={'textAlign': 'center', 'fontFamily': 'Arial', 'marginTop': '50px', 'marginBottom': '20px'}),
    
    # Target Distribution Graph
    html.H3("Which migration-related targets are most discussed in the contexts of Israel-Gaza and Russia-Ukraine wars?", style={'textAlign': 'center', 'fontFamily': 'Arial'}),
    dcc.Graph(figure=plot_top_targets_by_subreddit(df, subreddit_col='subreddit', target_col='target')),

    # Polarization Index by Subreddit
    html.H3("To what extent is discourse polarized in each subreddit?", style={'textAlign': 'center', 'fontFamily': 'Arial'}),
    dcc.Graph(figure=plot_stance_heatmap_by_subreddit(df, subreddit_col='subreddit', stance_col='stance', intensity_col='confidence_intensity')),

     # Proportion of Target Group Mentions
    html.H3("What is the proportional breakdown of mentions for each migration-related target group?", style={'textAlign': 'center', 'fontFamily': 'Arial'}),
    dcc.Graph(figure=plot_target_group_proportions(df, target_label_col='target_label')),

    # Stance & Intensity Across Target Groups and Subreddits
    html.H3("How do the stances and intensity levels for the target groups compare between the two subreddits?", style={'textAlign': 'center', 'fontFamily': 'Arial'}),
    dcc.Graph(figure=plot_stance_and_intensity_summary(df, category_col='target_label', subreddit_col='subreddit')),

    # Most Polarized Targets Across Subreddits
    html.H3("Which targets are the most polarized, and what is the proportion of supportive versus critical stances for each across subreddits?", style={'textAlign': 'center', 'fontFamily': 'Arial'}),
    dcc.Graph(figure=plot_most_polarized_targets(df, subreddit_col='subreddit', target_col='target', stance_col='stance'))
    

    ])

# --- 3. Run the App ---
if __name__ == '__main__':
    app.run(debug=True)


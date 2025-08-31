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
    plot_stance_wordclouds,
    plot_stance_heatmap_by_subreddit,
    plot_polarization_by_post_label
)
# --- 1. App Setup ---

# --- Helper Function to Convert Matplotlib Plot to Image URI ---
def fig_to_uri(fig):
    """Convert Matplotlib figure to PNG image URI"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close(fig)  # Crucial for memory management
    return f"data:image/png;base64,{encoded}"

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

    # Distribution of Targets by Stance
    html.H3("How do the targets differ between supportive and critical stances?", style={'textAlign': 'center', 'fontFamily': 'Arial'}),
    html.Img(src=fig_to_uri(plot_stance_wordclouds(df)), style={'display': 'block', 'margin': 'auto', 'width': '80%'}),

    # Polarization Index by Subreddit
    html.H3("To what extent is discourse polarized in each subreddit?", style={'textAlign': 'center', 'fontFamily': 'Arial'}),
    dcc.Graph(figure=plot_stance_heatmap_by_subreddit(df, subreddit_col='subreddit', stance_col='stance', intensity_col='confidence_intensity')),

    # Polarization Index by Post Label
    html.H3("To what extent is discourse polarized in each general post category?", style={'textAlign': 'center', 'fontFamily': 'Arial'}),
    dcc.Graph(figure=plot_polarization_by_post_label(df, label_col='target_label', stance_col='stance', intensity_col='confidence_intensity'))
    ])

# --- 3. Run the App ---
if __name__ == '__main__':
    app.run(debug=True)


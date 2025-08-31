# File: dashboard.py

import dash
from dash import dcc, html
import pandas as pd
import sys
import os

# --- Use a relative import for local modules ---
from .visualisations import (
    plot_top_targets_by_subreddit,
    plot_stance_wordclouds,
    plot_stance_heatmap_by_subreddit,
    plot_polarization_by_post_label,
    plot_weekly_post_counts,
    plot_stance_trend,
    plot_weekly_avg_stance_intensity,
    plot_engagement_distribution,
    plot_intensity_correlation,
    plot_target_group_proportions,
    plot_stance_and_intensity_summary,
    plot_most_polarized_targets,
    plot_polarization_heatmap,
    plot_intensity_vs_engagement
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

    # Distribution of Targets by Stance
    html.H3("How do the targets differ between supportive and critical stances?", style={'textAlign': 'center', 'fontFamily': 'Arial'}),
    dcc.Graph(figure=plot_stance_wordclouds(df)),

    # Polarization Index by Subreddit
    html.H3("To what extent is discourse polarized in each subreddit?", style={'textAlign': 'center', 'fontFamily': 'Arial'}),
    dcc.Graph(figure=plot_stance_heatmap_by_subreddit(df, subreddit_col='subreddit', stance_col='stance', intensity_col='confidence_intensity')),

    # Polarization Index by Post Label
    html.H3("To what extent is discourse polarized in each general post category?", style={'textAlign': 'center', 'fontFamily': 'Arial'}),
    dcc.Graph(figure=plot_polarization_by_post_label(df, label_col='target_label', stance_col='stance', intensity_col='confidence_intensity')),

    # Proportion of Target Group Mentions
    html.H3("What is the proportional breakdown of mentions for each migration-related target group?", style={'textAlign': 'center', 'fontFamily': 'Arial'}),
    dcc.Graph(figure=plot_target_group_proportions(df, target_label_col='target_label')),

    # Stance & Intensity Across Target Groups and Subreddits
    html.H3("How do the stances and intensity levels for the target groups compare between the two subreddits?", style={'textAlign': 'center', 'fontFamily': 'Arial'}),
    dcc.Graph(figure=plot_stance_and_intensity_summary(df, category_col='target_label', subreddit_col='subreddit')),

    # Most Polarized Targets Across Subreddits
    html.H3("Which targets are the most polarized, and what is the proportion of supportive versus critical stances for each across subreddits?", style={'textAlign': 'center', 'fontFamily': 'Arial'}),
    dcc.Graph(figure=plot_most_polarized_targets(df, subreddit_col='subreddit', target_col='target', stance_col='stance')),
    
    # Stance Intensity Distribution by Target Group
    html.H3("How does the proportion of supportive versus critical stances vary in each target group?", style={'textAlign': 'center', 'fontFamily': 'Arial'}),
    dcc.Graph(figure=plot_polarization_heatmap(df, category_col='target_label')),

    # Subheading for Temporal and Engagement Trends
    html.H2("Temporal and Engagement Trends", style={'textAlign': 'center', 'fontFamily': 'Arial', 'marginTop': '50px', 'marginBottom': '20px'}),

    # Weekly Post Counts by Stance
    html.H3("How have the post counts for supportive versus critical stances evolved in each subreddit over time?", style={'textAlign': 'center', 'fontFamily': 'Arial'}),
    dcc.Graph(figure=plot_weekly_post_counts(df, date_col='date', subreddit_col='subreddit', stance_col='stance')),

    # Trend of Supportive vs Critical Stance Proportion Over Time
    html.H3("How has the proportion of supportive vs critical posts changed over time in each subreddit?", style={'textAlign': 'center', 'fontFamily': 'Arial'}),
    dcc.Graph(figure=plot_stance_trend(df, subreddit_col="subreddit", stance_col="stance", target_stances=["Supportive", "Critical"])),

    # Weekly Average Stance Intensity by Subreddit
    html.H3("How has the average intensity of supportive versus critical stances varied in each subreddit on a weekly basis?", style={'textAlign': 'center', 'fontFamily': 'Arial'}),
    dcc.Graph(figure=plot_weekly_avg_stance_intensity(df, subreddit_col="subreddit", stance_col="stance", intensity_col="confidence_intensity")),
    
    # Engagement Score Distribution by Stance and Subreddit
    html.H3("How does the distribution of engagement scores for supportive posts compare to that of critical posts within each subreddit?", style={'textAlign': 'center', 'fontFamily': 'Arial'}),
    dcc.Graph(figure=plot_engagement_distribution(df, stance_col='stance', subreddit_col='subreddit', score_col='score', intensity_col='confidence_intensity')),

    # Correlation of Intensity and Engagement by Subreddit
    html.H3("What is the correlation between stance intensity and user engagement for both supportive and critical posts in each subreddit?", style={'textAlign': 'center', 'fontFamily': 'Arial'}),
    dcc.Graph(figure=plot_intensity_correlation(df, stance_col='stance', subreddit_col='subreddit', score_col='score', intensity_col='confidence_intensity')),

    # Intensity for Top Target Groups and Their Engagement on Reddit
    html.H3("Intensity for Top Target Groups and Their Engagement on Reddit", style={'textAlign': 'center', 'fontFamily': 'Arial'}),
    dcc.Graph(figure=plot_intensity_vs_engagement(df))
    
    ])

# --- 3. Run the App ---
if __name__ == '__main__':
    app.run(debug=True)


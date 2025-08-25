# In visualisations.py

import plotly.express as px
import pandas as pd

def plot_top_targets_by_subreddit(df, subreddit_col='subreddit', target_col='target', top_n=25, title="Top Migration Targets by Subreddit"):
    """
    Plots the top N migration-related targets for each subreddit using Plotly facets.
    """
    # --- 1. Calculate top N targets within each subreddit ---
    df_counts = df.groupby([subreddit_col, target_col]).size().reset_index(name='Count')
    top_targets_per_sub = df_counts.groupby(subreddit_col, group_keys=False).apply(lambda x: x.nlargest(top_n, 'Count'))

    # --- 2. Create the faceted bar chart ---
    fig = px.bar(
        top_targets_per_sub,
        x=target_col,
        y='Count',
        text='Count',
        title=title,
        labels={target_col: 'Target', 'Count': 'Count'},
        color='Count',
        color_continuous_scale='Teal',
        facet_col=subreddit_col,
        facet_col_wrap=1
    )

    # --- 3. Clean up the layout ---
    fig.update_traces(textposition='outside')
    fig.update_xaxes(matches=None) # Give each plot its own x-axis
    fig.update_yaxes(matches=None) # Give each plot its own y-axis
    fig.update_layout(
        xaxis_tickangle=-45,
        template='plotly_white',
        height=450 * len(df[subreddit_col].unique()),
        margin=dict(l=40, r=40, t=80, b=150)
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    # --- CRITICAL FIX: REMOVED fig.show() ---
    # The function now only returns the figure for Dash to display.
    return fig
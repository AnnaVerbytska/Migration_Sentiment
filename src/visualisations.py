# In visualisations.py

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
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

def plot_score_distribution(df):
    """
    Creates a Plotly violin plot of score distribution by stance and subreddit.
    """
    # --- Data preparation ---
    df_analysis = df[df['stance'].isin(['Supportive', 'Critical'])].copy()
    df_analysis['log_score'] = np.log(df_analysis['score'] + 1)

    # --- Create the visualization ---
    fig = px.violin(
        df_analysis,
        x="stance",
        y="log_score",
        color="subreddit",
        box=True,
        points="all", # Show all data points
        hover_data=df_analysis.columns, # Show all columns on hover
        color_discrete_map={
            "ukraine": "#0057b7",
            "IsraelPalestine": "#009639"
        },
        title="Engagement Score Distribution by Stance and Subreddit"
    )
    fig.update_layout(
        xaxis_title="Stance",
        yaxis_title="Log of Reddit Score (Higher = more engagement)",
        legend_title="Subreddit",
        template="plotly_white"
    )
    return fig

def plot_intensity_correlation(df):
    """
    Creates a Plotly scatter plot with a trendline for intensity vs. score.
    """
    # --- Data preparation ---
    df_analysis = df[df['stance'].isin(['Supportive', 'Critical'])].copy()
    df_analysis['log_score'] = np.log(df_analysis['score'] + 1)

    # --- Create the visualization ---
    fig = px.scatter(
        df_analysis,
        x="confidence_intensity",
        y="log_score",
        color="stance",
        facet_col="subreddit",
        opacity=0.5,
        trendline="ols",  # Ordinary Least Squares trendline
        color_discrete_map={
            "Supportive": "#2ca02c",
            "Critical": "#d62728"
        },
        hover_data=df_analysis.columns, # Show all columns on hover
        title="Correlation of Stance Intensity and Engagement by Subreddit"
    )
    fig.update_layout(
        xaxis_title="Stance Intensity (1-5)",
        yaxis_title="Log of Reddit Score",
        legend_title="Stance",
        template="plotly_white"
    )
    # Clean up the subplot titles
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig




# ----------------Visualisations for target_label analysis-----------------

def plot_polarization_heatmap(df, category_col, top_n=15, title="Stance Distribution by Category"):
    """
    Creates a 100% stacked bar chart (as a heatmap) to show stance proportions for the whole dataset.
    """
    df_plot = df[df['stance'].isin(['Supportive', 'Critical'])].copy()
    top_categories = df_plot[category_col].value_counts().nlargest(top_n).index
    df_plot = df_plot[df_plot[category_col].isin(top_categories)]

    crosstab = pd.crosstab(df_plot[category_col], df_plot['stance'], normalize='index')
    
    for stance in ['Supportive', 'Critical']:
        if stance not in crosstab.columns:
            crosstab[stance] = 0
            
    crosstab = crosstab[['Critical', 'Supportive']]
    
    fig = px.imshow(
        crosstab,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale='RdBu_r',
        color_continuous_midpoint=0.5,
        labels=dict(x="Stance", y=category_col, color="Proportion"),
        title=title
    )
    fig.update_layout(template='plotly_white')
    return fig

def plot_stance_and_intensity_summary(df, category_col, subreddit_col='subreddit', top_n=10, title="Stance & Intensity Towards Target Groups by Subreddit"):
    """
    OPTIMIZED FUNCTION: Creates a faceted diverging bar chart.
    - Bar position (left/right) shows stance.
    - Bar length shows count.
    - Bar color shows average intensity.
    """
    df_plot = df[df['stance'].isin(['Supportive', 'Critical'])].copy()

    # --- Calculate top N categories PER subreddit based on count ---
    top_categories_per_sub = df_plot.groupby(subreddit_col)[category_col].value_counts().groupby(level=0).nlargest(top_n).reset_index(level=0, drop=True)
    df_plot = df_plot[df_plot.set_index([subreddit_col, category_col]).index.isin(top_categories_per_sub.index)]

    if df_plot.empty:
        print(f"Warning: Not enough data to generate plot for top {top_n} categories.")
        return go.Figure().update_layout(title_text="Not enough data for summary plot")

    # --- Group and aggregate both count and mean intensity ---
    summary_df = df_plot.groupby([subreddit_col, category_col, 'stance']).agg(
        count=('stance', 'size'),
        avg_intensity=('confidence_intensity', 'mean')
    ).reset_index()

    # --- Create a new column for plotting, making 'Critical' counts negative ---
    summary_df['plot_count'] = summary_df.apply(
        lambda row: -row['count'] if row['stance'] == 'Critical' else row['count'],
        axis=1
    )

    fig = px.bar(
        summary_df,
        x='plot_count',
        y=category_col,
        color='avg_intensity',
        facet_col=subreddit_col,
        facet_col_spacing=0.1, # <-- FIX: Adds space between the subplots
        orientation='h',
        title=title,
        labels={
            'plot_count': 'Mentions (Critical < 0 | Supportive > 0)',
            category_col: 'Target Category',
            'avg_intensity': 'Avg. Intensity'
        },
        color_continuous_scale='RdBu_r',
        color_continuous_midpoint=3.0,
        range_color=[1, 5],
        hover_data={'count': True, 'stance': True} 
    )

    fig.update_layout(
        template='plotly_white',
        yaxis={'categoryorder':'total ascending'},
        barmode='relative',
        height=600,
        margin=dict(b=120),
        xaxis_title_standoff=25
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_xaxes(matches=None)
    fig.update_yaxes(matches=None, tickfont=dict(size=9))

    return fig

def plot_intensity_vs_engagement(df, top_n=5, min_mentions=10):
    """
    Creates scatter plots to show the correlation between intensity and score for top target labels.
    """
    df_plot = df[df['stance'].isin(['Supportive', 'Critical'])].copy()
    df_plot['log_score'] = np.log(df_plot['score'] + 1)
    
    valid_labels = df_plot['target_label'].value_counts()
    valid_labels = valid_labels[valid_labels >= min_mentions].nlargest(top_n).index
    df_plot = df_plot[df_plot['target_label'].isin(valid_labels)]

    if df_plot.empty:
        print(f"Warning: No target labels met the minimum mention threshold of {min_mentions}.")
        return go.Figure().update_layout(title_text=f"Not enough data for Intensity vs. Engagement (min {min_mentions} mentions required)")

    fig = px.scatter(
        df_plot,
        x="confidence_intensity",
        y="log_score",
        color="stance",
        facet_col="target_label",
        facet_col_wrap=3,
        trendline="ols",
        title=f"Intensity vs. Engagement for Top {len(valid_labels)} Target Labels",
        labels={
            "confidence_intensity": "Stance Intensity (1-5)",
            "log_score": "Log of Reddit Score"
        },
        color_discrete_map={'Supportive': '#2ca02c', 'Critical': '#d62728'}
    )
    fig.update_layout(template='plotly_white')
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig

def plot_top_targets_within_group(df, target_group, top_n=10, title_prefix="Stance on Top Targets within"):
    """
    Visualizes the stance counts for the top N raw targets within a specific target group.
    """
    # Filter the DataFrame for the selected target group
    df_group = df[df['target_label'] == target_group]
    
    if df_group.empty:
        print(f"Warning: No data found for the target group: {target_group}")
        return go.Figure().update_layout(title_text=f"No data for '{target_group}'")

    # Find the top N raw targets within this group
    top_targets_in_group = df_group['target'].value_counts().nlargest(top_n).index
    df_plot = df_group[df_group['target'].isin(top_targets_in_group)]

    # Count stances for each target
    stance_counts = df_plot.groupby(['target', 'stance']).size().reset_index(name='count')

    # Create the bar chart
    fig = px.bar(
        stance_counts,
        x='count',
        y='target',
        color='stance',
        orientation='h',
        title=f'{title_prefix} "{target_group}"',
        labels={'count': 'Number of Mentions', 'target': 'Specific Target'},
        color_discrete_map={'Supportive': '#2ca02c', 'Critical': '#d62728', 'Neutral': '#cccccc'},
        category_orders={'target': top_targets_in_group} # Order bars by frequency
    )
    
    fig.update_layout(
        template='plotly_white',
        yaxis={'categoryorder': 'total ascending'},
        height=200 + (top_n * 40) # Dynamically adjust height
    )
    
    return fig

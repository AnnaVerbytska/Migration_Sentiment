# In visualisations.py

# Import necessary libraries
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import seaborn as sns
from wordcloud import WordCloud
from matplotlib import pyplot as plt
# ----------------Visualisaions for subreddit analysis-----------------t
def plot_top_targets_by_subreddit(df_stance, subreddit_col, target_col, top_n=30, title='Top Migration Targets by Subreddit'):
    """
    Plots the top N migration-related targets for each subreddit using Plotly facets.
    """
    # Calculate counts for each target within each subreddit 
    df_counts = df_stance.groupby([subreddit_col, target_col]).size().reset_index(name='Count')

    # Get the top N targets for each subreddit
    # Sort the data by count, then group by subreddit and take the top N from each group.
    top_targets_per_sub = df_counts.sort_values('Count', ascending=False).groupby(subreddit_col).head(top_n)

    # Create the faceted bar chart
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
        facet_col_wrap=1,
        facet_row_spacing=0.1
    )

    # Clean up the layout
    fig.update_traces(textposition='outside')
    fig.update_xaxes(matches=None, showticklabels=True)
    fig.update_yaxes(matches=None)
    fig.update_layout(
        xaxis_tickangle=-45,
        template='plotly_white',
        height=450 * len(df_stance[subreddit_col].unique()),
        margin=dict(l=40, r=40, t=80, b=150)
    )
    # This loop cleans up the subplot titles (e.g., "subreddit=ukraine" becomes "ukraine")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    return fig

def plot_stance_wordclouds(df_stance, stance_col='stance', target_col='target'):
    """
    Generates and displays side-by-side word clouds for 'Supportive' and 'Critical' stances.

    Args:
        df_stance (pd.DataFrame): DataFrame containing the stance and target data.
        stance_col (str): The name of the column with stance labels (e.g., 'Supportive', 'Critical').
        target_col (str): The name of the column with the target text.
    """
    # --- 1. Prepare texts for each stance ---
    # Filter the DataFrame for each stance and join the target words into a single string.
    try:
        supportive_targets = " ".join(df_stance[df_stance[stance_col] == "Supportive"][target_col].astype(str))
        critical_targets = " ".join(df_stance[df_stance[stance_col] == "Critical"][target_col].astype(str))
    except KeyError as e:
        print(f"Error: Make sure the DataFrame has the columns '{stance_col}' and '{target_col}'. Original error: {e}")
        return

    # --- 2. Generate word cloud objects ---
    wc_supportive = WordCloud(width=800, height=500, background_color="white", collocations=False).generate(supportive_targets)
    wc_critical = WordCloud(width=800, height=500, background_color="white", collocations=False).generate(critical_targets)

    # --- 3. Plot the word clouds side-by-side ---
    # Create a figure with two subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    # Add the general title to the entire figure.
    plt.suptitle("Distribution of Targets by Stance", fontsize=24, y=0.88)
    # Plot Supportive Word Cloud on the first subplot
    axes[0].imshow(wc_supportive, interpolation="bilinear")
    axes[0].set_title("Supportive Stance Targets", fontsize=20)
    axes[0].axis("off")

    # Plot Critical Word Cloud on the second subplot
    axes[1].imshow(wc_critical, interpolation="bilinear")
    axes[1].set_title("Critical Stance Targets", fontsize=20)
    axes[1].axis("off")

    # Adjust layout to prevent titles from overlapping and display the plot
    plt.tight_layout(pad=3.0)
    plt.close(fig)   # <- prevents double display in Jupyter
    return fig

def plot_polarization_by_post_label(df_stance, label_col='label', stance_col='stance', intensity_col='confidence_intensity'):
    """
    Calculates a polarization index for different labels in a DataFrame and
    visualizes the results as a sorted horizontal bar chart using Plotly.

    Args:
        df_stance (pd.DataFrame): The input DataFrame.
        label_col (str): The column name for the categories to group by (e.g., topics).
        stance_col (str): The column name for the stance information.
        intensity_col (str): The column name for the confidence/intensity score.
    """
    
    # --- 1. Define a nested helper function to calculate the index ---
    def polarization_index(group):
        """Calculates the Gini-Simpson Index for a group."""
        weighted_counts = group.groupby(stance_col)[intensity_col].sum()
        total = weighted_counts.sum()
        if total == 0:
            return 0
        p = weighted_counts / total
        return 1 - (p**2).sum()

    # --- 2. Calculate the polarization index for each label ---
    polarization_labels = df_stance.groupby(label_col).apply(
        polarization_index, include_groups=False
    ).reset_index(name='polarization_index')

    # --- 3. Sort the data for better readability ---
    polarization_labels_sorted = polarization_labels.sort_values('polarization_index', ascending=False)

    # --- 4. Create an interactive horizontal bar plot with Plotly ---
    fig = px.bar(
        data_frame=polarization_labels_sorted,
        y=label_col,
        x='polarization_index',
        color='polarization_index', # Color bars based on the index value
        # (FIX) Use a standard, built-in reversed colorscale string ('RdBu_r') 
        # to ensure compatibility across different Plotly versions.
        color_continuous_scale='RdBu_r',
        title=f"Polarization Index by {label_col.replace('_', ' ').title()}",
        labels={
            'polarization_index': 'Polarization Index',
            label_col: label_col.replace('_', ' ').title()
        },
        text='polarization_index' # Add data labels to the bars
    )

    # --- 5. Clean up the layout ---
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(
        template='plotly_white',
        yaxis={'categoryorder':'total ascending'} # Ensure the y-axis sorting is correct
    )
    fig.show()

def plot_stance_heatmap_by_subreddit(df_stance, subreddit_col='subreddit', stance_col='stance', intensity_col='confidence_intensity'):
    """
    Creates an interactive heatmap showing the normalized stance distribution
    by subreddit, weighted by confidence intensity.

    Args:
        df_stance (pd.DataFrame): The input DataFrame.
        subreddit_col (str): The column name for the subreddit.
        stance_col (str): The column name for the stance information.
        intensity_col (str): The column name for the confidence/intensity score.
    """
    
    # --- 1. Calculate the weighted stance distribution ---
    stance_dist = df_stance.groupby([subreddit_col, stance_col])[intensity_col].sum().reset_index()
    
    # --- 2. Pivot the data to create a matrix suitable for a heatmap ---
    stance_pivot = stance_dist.pivot(index=subreddit_col, columns=stance_col, values=intensity_col).fillna(0)
    
    # --- 3. Normalize the data so each row sums to 1 (100%) ---
    stance_pivot_normalized = stance_pivot.div(stance_pivot.sum(axis=1), axis=0)

    # --- 4. Create an interactive heatmap with Plotly ---
    fig = px.imshow(
        stance_pivot_normalized,
        text_auto=True,  # Automatically display the values on the heatmap
        aspect="auto",   # Adjust aspect ratio to fit the plot area
        color_continuous_scale='RdBu', # Use the same Red-Blue color scale
        labels=dict(x="Stance", y="Subreddit", color="Proportion"),
        title="Stance Distribution by Subreddit"
    )

    # --- 5. Customize the appearance ---
    fig.update_traces(texttemplate="%{z:.2f}") # Format the annotations to 2 decimal places
    fig.update_xaxes(side="top") # Move x-axis labels to the top for a classic heatmap look
    
    fig.show()

def plot_stance_over_time(df_stance, date_col='date', subreddit_col='subreddit', stance_col='stance', intensity_col='confidence_intensity'):
    """
    Generates three interactive time-series plots to analyze stance trends using Plotly.
    It displays plots for post counts, stance proportion, and stance intensity over time.

    Args:
        df_stance (pd.DataFrame): The input DataFrame.
        date_col (str): The column name for the date information.
        subreddit_col (str): The column name for the subreddit.
        stance_col (str): The column name for the stance information.
        intensity_col (str): The column name for the confidence/intensity score.
    """
    # --- 1. Prepare Data ---
    # Ensure date column is in datetime format for time-series analysis
    df_stance[date_col] = pd.to_datetime(df_stance[date_col])
    
    # Filter out 'Neutral' stance for cleaner trend lines
    df_analysis = df_stance[df_stance[stance_col].isin(['Supportive', 'Critical'])].copy()

    # Resample data by week for smoother plotting
    df_analysis['week'] = df_analysis[date_col].dt.to_period('W').apply(lambda r: r.start_time)

    # --- 2. Plot 1: Post Count Trends Side-by-Side ---
    posts_per_week = df_analysis.groupby(['week', subreddit_col, stance_col]).size().reset_index(name='count')
    
    print("Displaying Plot 1: Weekly Post Counts by Stance")
    fig1 = px.line(
        posts_per_week,
        x='week',
        y='count',
        color=stance_col,
        facet_col=subreddit_col, # Creates side-by-side plots for each subreddit
        markers=True,
        color_discrete_map={'Supportive': '#2ca02c', 'Critical': '#d62728'},
        labels={"week": "Week", "count": "Number of Posts"},
        title='Weekly Post Counts by Stance'
    )
    fig1.update_layout(template="plotly_white")
    fig1.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1])) # Cleans subplot titles
    fig1.show()

    # --- 3. Plot 2: Direct Comparison of Stance Proportion ---
    weekly_props = df_analysis.groupby(['week', subreddit_col])[stance_col].value_counts(normalize=True).unstack().fillna(0)
    weekly_props = weekly_props.reset_index()

    print("\nDisplaying Plot 2: Trend of Supportive Stance Proportion")
    fig2 = px.line(
        weekly_props,
        x='week',
        y='Supportive',
        color=subreddit_col, # Shows each subreddit as a different colored line
        markers=True,
        labels={"week": "Week", "Supportive": 'Proportion of Posts that are "Supportive"'},
        title='Trend of Supportive Stance Proportion Over Time'
    )
    fig2.add_hline(y=0.5, line_dash="dot", annotation_text="50% Mark", annotation_position="bottom right")
    fig2.update_yaxes(range=[0, 1]) # Lock y-axis between 0 and 1
    fig2.update_layout(template="plotly_white")
    fig2.show()
    
    # --- 4. Plot 3: Stance Intensity Trends Side-by-Side ---
    avg_intensity_week = df_analysis.groupby(['week', subreddit_col, stance_col])[intensity_col].mean().reset_index()

    print("\nDisplaying Plot 3: Weekly Average Stance Intensity")
    fig3 = px.line(
        avg_intensity_week,
        x='week',
        y=intensity_col,
        color=stance_col,
        facet_col=subreddit_col, # Creates side-by-side plots
        markers=True,
        color_discrete_map={'Supportive': '#2ca02c', 'Critical': '#d62728'},
        labels={"week": "Week", intensity_col: "Average Intensity (1-5)"},
        title='Weekly Average Stance Intensity'
    )
    fig3.update_layout(template="plotly_white")
    fig3.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1])) # Cleans subplot titles
    fig3.show()

def plot_engagement_visuals(df_stance, stance_col='stance', subreddit_col='subreddit', score_col='score', intensity_col='confidence_intensity'):
    """
    Generates two interactive plots to analyze the relationship between stance,
    engagement (Reddit score), and intensity.

    Args:
        df_stance (pd.DataFrame): The input DataFrame.
        stance_col (str): The column name for stance information.
        subreddit_col (str): The column name for the subreddit.
        score_col (str): The column name for the Reddit post score.
        intensity_col (str): The column name for the confidence/intensity score.
    """
    # --- 1. Data preparation ---
    df_analysis = df_stance[df_stance[stance_col].isin(['Supportive', 'Critical'])].copy()
    # Use np.log1p for a more stable log transformation (handles zeros)
    df_analysis['log_score'] = np.log1p(df_analysis[score_col])

    # --- 2. Visualization 1: Score Distribution by Stance ---
    print("Displaying Plot 1: Engagement Score Distribution")
    fig1 = px.violin(
        df_analysis,
        x=stance_col,
        y="log_score",
        color=subreddit_col,
        box=True,
        points="all",
        hover_data=df_analysis.columns,
        color_discrete_map={
            "ukraine": "#0057b7",
            "IsraelPalestine": "#009639"
        },
        title="Engagement Score Distribution by Stance and Subreddit"
    )
    fig1.update_layout(
        xaxis_title="Stance",
        yaxis_title="Log of Reddit Score (Higher = more engagement)",
        legend_title="Subreddit",
        template="plotly_white"
    )
    fig1.show()


    # --- 3. Visualization 2: Correlation of Intensity and Score ---
    print("\nDisplaying Plot 2: Correlation of Stance Intensity and Engagement")
    fig2 = px.scatter(
        df_analysis,
        x=intensity_col,
        y="log_score",
        color=stance_col,
        facet_col=subreddit_col,
        opacity=0.5,
        trendline="ols",  # requires statsmodels to be installed
        color_discrete_map={
            "Supportive": "#2ca02c",
            "Critical": "#d62728"
        },
        hover_data=df_analysis.columns,
        title="Correlation of Stance Intensity and Engagement by Subreddit"
    )
    fig2.update_layout(
        xaxis_title="Stance Intensity (1-5)",
        yaxis_title="Log of Reddit Score",
        legend_title="Stance",
        template="plotly_white"
    )
    fig2.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1])) # Clean subplot titles
    fig2.show()


# ----------------Visualisations for target_label analysis-----------------

def plot_target_group_proportions(df_stance, target_label_col='target_label'):
    """
    Calculates and visualizes the proportion of mentions for different target groups
    in a DataFrame using a horizontal bar chart.

    Args:
        df_stance (pd.DataFrame): The input DataFrame.
        target_label_col (str): The column name containing the target group labels.
    """
    # --- 1. Calculate value counts directly from the DataFrame ---
    try:
        df_counts = df_stance[target_label_col].value_counts().reset_index()
    except KeyError:
        print(f"Error: The DataFrame does not have a column named '{target_label_col}'.")
        return

    # --- 2. Rename the columns for clarity ---
    df_counts.columns = [target_label_col, 'count']

    # --- 3. Calculate the percentage ---
    total_count = df_counts['count'].sum()
    if total_count == 0:
        print("Warning: The total count is zero. Cannot calculate percentages.")
        df_counts['Percentage'] = 0
    else:
        df_counts['Percentage'] = (df_counts['count'] / total_count) * 100

    # --- 4. Create the horizontal bar chart ---
    fig = px.bar(
        df_counts,
        x='count',
        y=target_label_col,
        orientation='h',
        title='Proportion of Target Group Mentions',
        text=df_counts['Percentage'].apply(lambda x: f'{x:.1f}%'),
        labels={'count': 'Number of Mentions', target_label_col: 'Target Group'}
    )

    # --- 5. Refine the layout for clarity ---
    fig.update_layout(
        template='plotly_white',
        yaxis={'categoryorder': 'total ascending'}  # Order from smallest to largest
    )
    fig.update_traces(
        textposition='outside',
        marker_color='#4682B4'  # A nice steel blue color
    )

    fig.show()

def plot_stance_and_intensity_summary(df_stance, category_col, subreddit_col='subreddit', top_n=10, title="Stance & Intensity Towards Target Groups by Subreddit"):
    """
    Creates a side-by-side faceted diverging bar chart with unified target categories
    across subreddits (so y-axis categories align).
    """
    df_plot = df_stance[df_stance['stance'].isin(['Supportive', 'Critical'])].copy()

    top_categories = df_plot[category_col].value_counts().nlargest(top_n).index
    df_plot = df_plot[df_plot[category_col].isin(top_categories)]

    if df_plot.empty:
        print(f"Warning: Not enough data to generate plot for top {top_n} categories.")
        return go.Figure().update_layout(title_text="Not enough data for summary plot")

    # --- Aggregation ---
    summary_agg = (
        df_plot
        .groupby([subreddit_col, category_col, 'stance'])
        .agg(
            count=('stance', 'size'),
            avg_intensity=('confidence_intensity', 'mean')
        )
    )

    # --- Create a complete data grid ---
    subreddits = df_plot[subreddit_col].unique()
    stances = df_plot['stance'].unique()
    full_index = pd.MultiIndex.from_product(
        [subreddits, top_categories, stances],
        names=[subreddit_col, category_col, 'stance']
    )
    summary_df = summary_agg.reindex(full_index, fill_value=0).reset_index()

    # --- Plotting ---
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
        orientation='h',
        title=title,
        labels={
            'plot_count': 'Mentions (Critical < 0 | Supportive > 0)',
            category_col: 'Target Category',
            'avg_intensity': 'Avg. Intensity'
        },
        color_continuous_scale='RdBu_r',
        color_continuous_midpoint=3.0,
        range_color=[0.9, 5.1],
        hover_data={'count': True, 'stance': True}
    )

    fig.update_layout(
        template='plotly_white',
        barmode='relative',
        height=600,
        width=1200,
        autosize=False,
        margin=dict(b=150),  # Increased bottom margin for the new shared title
        coloraxis_colorbar=dict(
            title_font=dict(size=10),
            tickfont=dict(size=8)
        ),
        title_x=0.5 # This centers the main title
    )
    # Add the shared x-axis title using add_annotation so it doesn't overwrite existing titles
    fig.add_annotation(
        text="Mentions (Critical < 0 | Supportive > 0)",
        showarrow=False,
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.25,  # Positioned below the plot area
        xanchor="center",
        yanchor="top",
        font=dict(size=12)
    )

    # Remove the individual, overlapping x-axis titles from each facet
    fig.update_xaxes(title_text="")
    
    fig.update_yaxes(matches='y', categoryorder='total ascending', tickfont=dict(size=9))
    
    # This loop cleans the facet titles (e.g., "subreddit=ukraine" to "ukraine")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    return fig

def plot_most_polarized_targets(df_stance, subreddit_col='subreddit', target_col='target', stance_col='stance', top_n=10):
    """
    Identifies and plots the Top N most polarized targets (closest to a 50/50 split
    between Supportive and Critical stances).
    """
    df_filtered = df_stance[df_stance[stance_col].isin(['Supportive', 'Critical'])]
    
    for subreddit in df_filtered[subreddit_col].unique():
        df_subreddit = df_filtered[df_filtered[subreddit_col] == subreddit]
        
        # Calculate stance proportions for each target
        stance_props = df_subreddit.groupby(target_col)[stance_col].value_counts(normalize=True).unstack().fillna(0)

        # A simple polarization score: 0.5 - |Supportive % - 0.5|
        # Score is highest (0.5) when Supportive is exactly 50%
        stance_props['polarization'] = 0.5 - abs(stance_props['Supportive'] - 0.5)
        
        # Filter for targets with a minimum number of mentions to avoid noise
        min_mentions = 5
        mention_counts = df_subreddit[target_col].value_counts()
        valid_targets = mention_counts[mention_counts >= min_mentions].index
        
        if valid_targets.empty:
            print(f"--> Skipping r/{subreddit}: No targets found with at least {min_mentions} mentions.")
            continue
        
        most_polarized = stance_props.loc[valid_targets].nlargest(top_n, 'polarization')
        
        actual_top_n = len(most_polarized)

        fig = px.bar(
            most_polarized,
            y=most_polarized.index,
            x=['Supportive', 'Critical'],
            title=f"Top {actual_top_n} (up to {top_n}) Most Polarized Targets in r/{subreddit} (Min. {min_mentions} Mentions)",
            labels={'y': 'Target', 'value': 'Proportion of Stance'},
            barmode='stack',
            orientation='h',
            color_discrete_map={'Supportive': '#2ca02c', 'Critical': '#d62728'}
        )
        fig.update_layout(template='plotly_white', xaxis_ticksuffix='%', height=400 + (actual_top_n * 20))
        fig.update_xaxes(range=[0, 1])
        fig.show()

def plot_polarization_heatmap(df_stance, category_col, top_n=15, title="Stance Distribution by Category"):
    """
    Creates a 100% stacked bar chart (as a heatmap) to show stance proportions for the whole dataset.
    """
    df_plot = df_stance[df_stance['stance'].isin(['Supportive', 'Critical'])].copy()
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

def plot_intensity_vs_engagement(df_stance, top_n=5, min_mentions=10):
    """
    Creates scatter plots to show the correlation between intensity and score for top target labels.
    """
    df_plot = df_stance[df_stance['stance'].isin(['Supportive', 'Critical'])].copy()
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
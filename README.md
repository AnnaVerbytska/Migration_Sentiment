# Migration_Sentiment
It is a small-scale open source proof-of-concept (PoC) project on polarisation of posts about migration on Reddit in the context of the wars in Ukraine and Gaza. The core technique is target-based stance detection with Gemini API.


Table of Contents

## 1. Background

## 2. Research Question

## 3. Methodology

## 4. Visualizations

## 5. Getting Started

## 6. License

# 1. **Background**

This proof-of-concept (PoC) project analyzes online polarization by comparing discussions on the r/ukraine and r/IsraelPalestine subreddits from May–July 2025. Standard sentiment analysis proved ineffective because the discussions rarely focused on ‘migration’ directly. Instead, they centered on specific targets or aspects (e.g., political actors, groups, events).

# 2. **Research Question**

How do Reddit communities frame migration-related conflicts, and what drives user engagement when direct entity-based sentiment is not the main indicator of stance?

# 3. **Methodology**

Data Collection: We collected 100 posts (50 from each subreddit) for May–July 2025 containing migration-related keywords.

LLM-powered Stance Analysis: We used Gemini 2.0 Flash to perform a novel target-based stance analysis on each post, extracting:

* The specific target of the stance (e.g., ‘Hamas’, ‘Ukrainian refugees’, ‘Pseudo-pacifism’, etc.).

* The stance towards that target (Supportive, Critical, Neutral).

* The intensity of the stance on a 1-5 scale.

# 4. Visualizations

This project includes a series of visualizations to explore the data, including:

* Distribution of Targets by Stance: Word clouds showing the most frequently discussed targets for both supportive and critical stances.

* Polarization Index: Heatmaps illustrating the degree of polarization in each subreddit and general post category.

* Temporal Trends: A graph showing the evolution of stances across subreddits over time.

 * Engagement Analysis: Plots that correlate discussion intensity and stance with user engagement (e.g., upvotes, comments).

# 5. Getting Started
Prerequisites

Python 3.12

Required libraries: `pandas`, `wordcloud`, `matplotlib`, `dash`, `dash-core-components`, `dash-html-components`

Installation

To install all the required packages, you can use the pip command.

`pip install -r requirements.txt`

Usage

Once all prerequisites are installed, you can run the Dash application.

`python dashboard.py`

The application will launch in your web browser, allowing you to interact with the visualizations.

# 6. **License**

This project is licensed under the MIT License - see the LICENSE.md file for details.
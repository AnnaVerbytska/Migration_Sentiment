# 🌍 Stance Detection of Migration on Social Media with Gemini  

**Target-based stance detection for understanding migration polarisation on Reddit**  

This open-source **proof-of-concept (PoC)** explores how online communities frame migration during the wars in **Ukraine** and **Gaza**. Instead of generic sentiment analysis, it zooms in on **targets** (actors, groups, events) and how people take a stance toward them — supportive, critical, or neutral (aka ABSA).  

Built with **LLM-powered stance analysis (Gemini API)** and interactive **Dash visualizations**, the project is a step toward understanding how polarising narratives around migration emerge and spread online.  

---

## 📖 Table of Contents  

1. [Background](#1-background)  
2. [Research Question](#2-research-question)  
3. [Methodology](#3-methodology)  
4. [Visualisations](#4-visualisations)  
5. [Getting Started](#5-getting-started)  
6. [License](#6-license)  

---

## 1. Background  

This project compares discussions on **r/ukraine** and **r/IsraelPalestine** (May–July 2025).  

Traditional sentiment analysis failed here: discussions rarely used the word *migration* directly. Instead, debates revolved around **who** migration is about (e.g., *Ukrainian refugees, Hamas, EU states*) and **how** they were framed.  

---

## 2. Research Question  

👉 *How do Reddit communities frame migration-related conflicts, and what drives user engagement when direct entity-based sentiment is not the main indicator of stance?*  

---

## 3. Methodology  

- **Data Collection**  
  - 100 posts (50 per subreddit)  
  - Keywords related to migration  
  - Period: May–July 2025  

- **LLM-powered Stance Analysis** (Gemini 2.0 Flash)  
  Extracted per post:  
  - 🎯 **Target** (*Hamas, Ukrainian refugees, pseudo-pacifism, etc.*)  
  - 📌 **Stance** (Supportive, Critical, Neutral)  
  - 🔥 **Intensity** (1–5 scale)  

---

## 4. Visualisations  

This PoC comes with an interactive **Dashboard**:  

- ☁️ **Word Clouds** — Most discussed targets by stance (supportive vs critical)  
- 🔥 **Polarisation Index** — Heatmaps of stance intensity across subreddits  
- ⏳ **Temporal Trends** — Evolution of stances over time  
- 📈 **Engagement Analysis** — Linking stance & intensity to user engagement (upvotes, comments)  

---

## 5. Getting Started  

### Prerequisites  
- ![Python](https://img.shields.io/badge/Python-3.12.9-blue?logo=python&logoColor=white)  
- Libraries: `pandas`, `wordcloud`, `matplotlib`, `dash`, `dash-core-components`, `dash-html-components`  

### Installation  
```bash
pip install -r requirements.txt

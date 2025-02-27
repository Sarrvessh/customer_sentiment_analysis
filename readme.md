# Review Analysis System

## Overview

The **Review Analysis System** is an AI-powered tool that processes user reviews, detects emotions, extracts topics, computes sentiment scores (AdoreScore), and visualizes insights through an interactive dashboard.

## Features

- **Multilingual Support**: Translates non-English reviews into English.
- **Emotion Detection**: Identifies emotions (e.g., happy, sad, angry) in reviews.
- **Topic Extraction**: Extracts key themes and subtopics from the text.
- **AdoreScore Computation**: Assigns a sentiment score to each review.
- **Trend Analysis**: Tracks sentiment changes over time.
- **Dashboard Visualization**: Displays insights using interactive graphs.
- **Bulk Review Processing**: Supports large datasets with JSON export.
- **Real-Time Filtering**: Dynamic updates based on emotion or topic selection.

## Tech Stack

- **Programming Language**: Python
- **Frameworks & Libraries**:
  - `Dash`, `dash-bootstrap-components` (Web UI)
  - `Pandas` (Data Handling)
  - `nltk`, `TextBlob`, `spaCy` (NLP Processing)
  - `BERTopic`, `KeyBERT` (Topic Modeling)
  - `Transformers` (Sentiment Analysis)
  - `Scikit-learn` (ML Utilities)

## Installation

1. **Install dependencies:**
   pip install -r requirements.txt

2. **Run the application:**
   streamlit run dash.py

## How It Works

1. Upload a CSV file with reviews or enter a review manually.
2. The system translates, processes, and analyzes the text.
3. Insights are displayed on an interactive dashboard.
4. The analyzed data can be downloaded as a JSON file.

## Challenges Faced

- Translation inconsistencies → Fixed with better error handling.
- Performance issues with large datasets → Improved using multi-threading.
- Visualization layout problems → Optimized UI components.

'''
The adore_score.py containsthe code using VADER and TextBlob for sentiment analysis.
Valence Aware Dictionary and sEntiment Reasoner (VADER) is used for real-time sentiment scoring.
It keeps track of the adore scores for trend analysis.
It also incorporates topic relevance in the scoring calculation.
The adore score is normalized to a 0-100 scale.
The code also computes the trend change based on the overall sentiment score.
The adore scores are saved to a file for future trend analysis.
The code also loads previous adore scores from the file for trend analysis.
The code is used to compute the adore score using VADER & TextBlob.
It incorporates topic relevance in the scoring calculation.
'''

import json
import os
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

#the adore_trends.json file is used to store the adore scores for trend analysis
TREND_FILE = "adore_trends.json"

def load_past_scores():

    """
    loads previous adore scores from file for trend analysis.
    """
    if os.path.exists(TREND_FILE):
        with open(TREND_FILE, "r") as f:
            return json.load(f)
    return []

def save_past_scores(scores):
    """
    saves adore scores to file for future trend analysis.
    """
    with open(TREND_FILE, "w") as f:
        json.dump(scores, f, indent=2)

def compute_adore_score(text, topics):
    """
    computes adore score using VADER & TextBlob.
    incorporates topic relevance in the scoring calculation.
    """
    
    overall_sentiment = sia.polarity_scores(text)["compound"]

    adore_score = {}
    for topic in topics["main"]:
        subtopic_text = " ".join(topics["subtopics"].get(topic, []))
        topic_sentiment = sia.polarity_scores(subtopic_text)["compound"] if subtopic_text.strip() else overall_sentiment # to avoid empty subtopics affecting sentiment calculation
        relevance = topics["relevance"].get(topic, 1)  # Default relevance is 1
        weighted_score = topic_sentiment * relevance
        adore_score[topic] = round((weighted_score + 1) * 50) # Normalize scores to a 0-100 scale

    past_scores = load_past_scores() #loading the past scores

    if past_scores:
        previous_overall = past_scores[-1]["overall"]
        trend = "Increasing" if overall_sentiment > previous_overall else "Decreasing" if overall_sentiment < previous_overall else "Stable"
    else:
        trend = "No Data"

    #append the current score to the past ones
    past_scores.append({
        "text": text,
        "overall": round((overall_sentiment + 1) * 50),
        "breakdown": adore_score
    })
    save_past_scores(past_scores[-10:])  #storing only last 10 to maintain storage

    return {
        "overall": round((overall_sentiment + 1) * 50),
        "breakdown": adore_score,
        "trend": trend
    }
'''
The emotions.py script contains the detect_emotions function,
which uses a pre-trained distilBERT model to detect primary and secondary emotions from text.
The function returns a dictionary containing the primary and secondary emotions, their activation levels, and intensity scores.
'''

from transformers import pipeline
import numpy as np

# used a lightweight version of BERT called distilBERT for faster processing
# The emotion classifier model is trained on the GoEmotions dataset, which contains 58 emotion categories.
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

def detect_emotions(text):
    """
    Detects primary and secondary emotions from text.
    Returns a dictionary with emotion labels, activation levels, and intensity scores.
    """
    results = emotion_classifier(text, top_k=3)  # Detect top 3 emotions
    
    # determine activation levels based on intensity
    def get_activation(intensity):
        if intensity >= 0.5:
            return "High"
        elif intensity >= 0.3:
            return "Medium"
        else:
            return "Low"
    
    # return primary, secondary and tertiary emotions with activation levels
    return {
        "primary": {
            "emotion": results[0]['label'],
            "activation": get_activation(results[0]['score']),
            "intensity": round(results[0]['score'], 2)
        },
        "secondary": {
            "emotion": results[1]['label'],
            "activation": get_activation(results[1]['score']),
            "intensity": round(results[1]['score'], 2)
        },
        "tertiary": {
            "emotion": results[2]['label'],
            "activation": get_activation(results[2]['score']),
            "intensity": round(results[2]['score'], 2)
        }
    }

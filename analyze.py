'''
The analyze.py script is the main script which integrates all the other three scripts i.e, emotions.py, topics.py, and adore_score.py.
It takes an input text, translates it to English, performs emotion analysis, topic extraction, and computes the Adore Score.
The results are then appended to a JSON file named results.json.
'''
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from emotions import detect_emotions
from topics import extract_topics
from adore_score import compute_adore_score

# to support multilanguage support we are using the small100 model from huggingface transformers library 
# which is a multilingual model that supports 100 languages
model_name = "alirezamsh/small100"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def translate_to_english(text):
    """
    translates the input text to English using the small100 model.
    called in the analyze_text() function.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

def analyze_text(text):
    """
    translates text (if necessary), performs emotion analysis, topic extraction, and computes adore score.
    Appends results to a JSON file.
    """
    try:
        # translate the text to English
        translated_text = translate_to_english(text)
        
        # Perform all 3 analysis
        emotions = detect_emotions(translated_text)
        topics = extract_topics(translated_text)
        adore_score = compute_adore_score(translated_text, topics)

        result = {
            "original_text": text,
            "translated_text": translated_text,
            "emotions": emotions,
            "topics": topics,
            "adorescore": adore_score
        }

        # Load existing data if available
        try:
            with open("results.json", "r") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []
        except (FileNotFoundError, json.JSONDecodeError):
            data = []
        data.append(result)
        with open("results.json", "w") as f:
            json.dump(data, f, indent=2)

        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    text = input("Enter the text to analyze: ")
    output = analyze_text(text)
    print(output)

'''
The topics.py script contains functions to extract refined topics and subtopics from a given text using KeyBERT and spaCy NER.
It also calculates topic relevance scores based on the frequency of each topic in the text.
'''
import spacy
from keybert import KeyBERT

#spacy provides advanced capabilities to conduct natural language processing (NLP) on large volumes of text at high speed
nlp = spacy.load("en_core_web_sm")
kw_model = KeyBERT()

def clean_topics(topics):
    """
    Removes irrelevant or low-quality topics.
    """
    filtered_topics = [t for t in topics if len(t.split()) > 1]  # to remove single words
    return list(set(filtered_topics))  # to remove duplicates

def extract_topics(text):
    """
    Extracts refined topics and subtopics using KeyBERT and spaCy NER.
    Also calculates topic relevance scores.
    """
    key_topics = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=5)  #keyword extraction using KeyBERT
    key_topics = [t[0] for t in key_topics]
    doc = nlp(text)  #ner
    named_entities = list(set(ent.text for ent in doc.ents))
    main_topics = clean_topics(key_topics + named_entities)  #merging both key_topics and named_entities
    main_topics = [t for t in main_topics if t.lower() not in ["general", "one", "thing"]] # removal of any genral topics
    subtopics = {topic: [kw for kw in key_topics if kw in topic] for topic in main_topics if any(kw in topic for kw in key_topics)} #subtopics structure
    relevance = {topic: text.lower().count(topic.lower()) for topic in main_topics}  #relevance score calculation

    return {
        "main": main_topics,
        "subtopics": subtopics,
        "relevance": relevance
    }
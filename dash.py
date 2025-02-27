'''
The dash.py script is the main script that contains the Streamlit application code.
It loads the JSON data from the results.json file, provides filters for emotions, and displays the analysis results in different tabs.
The tabs include Trends, Emotions, Topics, Custom Analysis, and Upload Dataset.
The script also allows users to upload a CSV file containing reviews for bulk analysis.
The custom analysis tab allows users to enter a review text for analysis.
The script uses the emotions.py, topics.py, adore_score.py, and analyze.py scripts 
for emotion detection, topic extraction, AdoreScore computation, and translation, respectively.
'''
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px 
import json
from emotions import detect_emotions
from topics import extract_topics
from adore_score import compute_adore_score
from analyze import translate_to_english


def load_results():
    try:
        with open("results.json", "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

data = load_results()
df = pd.DataFrame(data)
if "emotions" not in df.columns or "topics" not in df.columns:
    st.error("Invalid JSON format: Missing required keys.")
    st.stop()
st.sidebar.title("Filters")
all_emotions = set()  #dynamic extraction of emotions
for entry in data:
    if "emotions" in entry:
        all_emotions.update(entry["emotions"].keys())

emotion_type_filter = st.sidebar.selectbox("Select Emotion Type", list(all_emotions))

# filter working to get unique emotions
available_emotions = list(set(
    entry["emotions"].get(emotion_type_filter, {}).get("emotion")
    for entry in data if "emotions" in entry and emotion_type_filter in entry["emotions"]
))
emotion_filter = st.sidebar.selectbox("Select Specific Emotion", available_emotions)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Trends", "Emotions", "Topics", "Custom Analysis", "Upload Dataset"])  #tabs for different analysis



#the tab1 is for trends where adore score is plotted against review index
#and here we can see the adore score trend over time

with tab1:
    st.title("AdoreScore Trend Comparison")
    adore_scores = [entry.get("adorescore", {}).get("overall") for entry in data if "adorescore" in entry]
    adore_trend_df = pd.DataFrame({"Review Index": range(1, len(adore_scores) + 1), "AdoreScore": adore_scores})
    if not adore_trend_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=adore_trend_df["Review Index"], y=adore_trend_df["AdoreScore"], mode='lines+markers', fill='tozeroy'))
        fig.update_layout(title="AdoreScore Trend Over Time", xaxis_title="Review Index", yaxis_title="AdoreScore")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No valid AdoreScore data available.")



#the tab2 is for emotions where we can see the emotions of the reviews
#and the intensity of the emotions

with tab2:
    st.title(f"🎭 {emotion_type_filter.capitalize()} Emotion Analysis")
    emotion_data = [
        {
            "Review ID": idx + 1,  
            "Emotion": entry["emotions"].get(emotion_type_filter, {}).get("emotion"),
            "Intensity": entry["emotions"].get(emotion_type_filter, {}).get("intensity", 0),
            "Activation": entry["emotions"].get(emotion_type_filter, {}).get("activation", "N/A"),
        }
        for idx, entry in enumerate(data)
        if "emotions" in entry and emotion_type_filter in entry["emotions"] and entry["emotions"].get(emotion_type_filter, {}).get("emotion") == emotion_filter
    ]
    emotion_df = pd.DataFrame(emotion_data)
    if not emotion_df.empty:
        st.dataframe(emotion_df, height=300)
    else:
        st.warning(f"No valid data available for {emotion_type_filter.capitalize()} emotions matching '{emotion_filter}'.")

    #spider plot or radar plot for emotions
    all_emotion_data = [
        {
            "Emotion": entry["emotions"].get(emotion_type_filter, {}).get("emotion"),
            "Intensity": entry["emotions"].get(emotion_type_filter, {}).get("intensity", 0)
        }
        for entry in data if "emotions" in entry and emotion_type_filter in entry["emotions"]
    ]
    emotion_chart_df = pd.DataFrame(all_emotion_data)
    if not emotion_chart_df.empty:
        categories = list(emotion_chart_df["Emotion"])
        values = list(emotion_chart_df["Intensity"])
        values.append(values[0])  # Close the shape
        categories.append(categories[0])
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            mode='lines',
            line=dict(color='black', width=2),
            fillcolor='gray',
            opacity=0.6
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=False,
            title=f"{emotion_type_filter.capitalize()} Emotions"
        )
        st.plotly_chart(fig, use_container_width=True)



#the tab3 is for topics where we can see the topics of the reviews
#and the relevance of the topics based on the reviews

with tab3:
    st.title("Topic Analysis")
    topic_data = []
    for entry in data:
        if "topics" in entry and "main" in entry["topics"]:
            for topic in entry["topics"]["main"]:
                relevance = entry["topics"]["relevance"].get(topic, 0)  # get relevance score
                topic_data.append({"Topic": topic, "Relevance": relevance})
    topic_df = pd.DataFrame(topic_data)

    if topic_df.empty:
        st.warning("No topics detected in the dataset.")
    else:
        #get top 5 topics based on relevance
        top_topics = topic_df.sort_values(by="Relevance", ascending=False).head(5)
        
        #bar plot
        fig = px.bar(
            top_topics, x="Topic", y="Relevance",
            title="🔝 Top 5 Topics by Relevance",
            color="Relevance", color_continuous_scale="blues"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("📜 Topic Breakdown")
        st.dataframe(topic_df)




#the tab4 is for custom analysis where we can enter a review for analysis
#and see the emotions, topics, and adore score of the review
#it returns the output in json format

with tab4:
    st.title("🔍 Enter Text for Custom Analysis")
    user_input = st.text_area("Enter a review to analyze:", placeholder="Type your review here...")
    analyze_button = st.button("Analyze Review")
    
    if analyze_button and user_input.strip():
        translated_text = translate_to_english(user_input)
        emotions = detect_emotions(translated_text)
        topics = extract_topics(translated_text)
        adorescore = compute_adore_score(translated_text, topics)
        custom_analysis_result = {
            "original_text": user_input,
            "translated_text": translated_text, 
            "emotions": emotions,
            "topics": topics,
            "adorescore": adorescore
        }
        with open("custom_analysis.json", "w") as json_file:
            json.dump(custom_analysis_result, json_file, indent=2)
        data.append(custom_analysis_result)
        with open("results.json", "w") as json_file:
            json.dump(data, json_file, indent=2)
        st.subheader("Custom Analysis Result")
        st.json(custom_analysis_result)
        st.subheader("Adorescore")
        if isinstance(adorescore, dict) and "overall" in adorescore:
            st.metric("Adorescore", f"{adorescore['overall']}")
        else:
            st.warning("No valid Adorescore data available.")



#the tab5 is for bulk analysis where we can upload a CSV file containing reviews
#and see the emotions, topics, and adore score of the reviews
#it returns the output in json format

with tab5:
    st.title("Upload Dataset for Bulk Analysis")
    uploaded_file = st.file_uploader("Upload a CSV file containing reviews", type=["csv"])
    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        if "review" in df_uploaded.columns:
            st.success("Dataset uploaded successfully! Analyzing...")
            analyzed_data = []
            for review in df_uploaded["review"]:
                translated_text = translate_to_english(review)
                emotions = detect_emotions(translated_text)
                topics = extract_topics(translated_text)
                adorescore = compute_adore_score(translated_text, topics)
                result_entry = {
                    "original_text": review,
                    "translated_text": translated_text,
                    "emotions": emotions,
                    "topics": topics,
                    "adorescore": adorescore
                }
                analyzed_data.append(result_entry)
                data.append(result_entry)
            with open("results.json", "w") as json_file:
                json.dump(data, json_file, indent=2)
            
            bulk_analysis_json = json.dumps(analyzed_data, indent=2)
            st.download_button(
                label="Download Analysis JSON",
                data=bulk_analysis_json,
                file_name="bulk_analysis.json",
                mime="application/json"
            )
        else:
            st.error("Uploaded CSV must contain a 'review' column.")

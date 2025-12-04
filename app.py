# app.py

import streamlit as st
import pandas as pd
from joblib import load
from src.detector import ChampionDetector   # needed so joblib can load the class


# -------------------------
# Load trained model once
# -------------------------
@st.cache_resource
def load_model():
    return load("models/champion_detector.pkl")


# Load train data once (some features depend on it)
@st.cache_data
def load_train_df():
    train = pd.read_csv("data/train.csv")
    train["answer"] = train["answer"].fillna("")
    if "topic" in train.columns:
        train["topic"] = train["topic"].fillna("")
    return train


detector = load_model()
train_df = load_train_df()


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="AI Detection", layout="centered")

st.title("AI Cheating Detection System")
st.write("This model predicts whether a student's answer is AI-generated .")

st.markdown("### ✍️ Enter a student's answer")

topic = st.text_input("Topic (optional)")
answer = st.text_area("Student answer", height=220)


# -------------------------
# Prediction
# -------------------------
if st.button("Check for AI-generated content"):
    if not answer.strip():
        st.warning("Please provide a student answer.")
    else:
        input_df = pd.DataFrame({
            "id": ["input_1"],
            "topic": [topic],
            "answer": [answer],
        })

        # NOTE: predict(train_df, test_df) → use positional arguments
        preds = detector.predict(train_df, input_df)
        prob = float(preds[0])

        st.markdown("### 📊 Result")
        st.write(f"**Cheating probability:** `{prob:.3f}`")
        st.progress(prob)

        if prob > 0.7:
            st.error("⚠️ Likely AI-generated")
        elif prob > 0.4:
            st.warning("⚠️ Possibly AI-assisted")
        else:
            st.success("✅ Likely human-written")

import streamlit as st
import pandas as pd
from joblib import load
from pathlib import Path

# Needed so joblib can deserialize the custom class inside the pickle
from src.detector import ChampionDetector  # noqa: F401


# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent


# -------------------------
# Load trained model once
# -------------------------
@st.cache_resource
def load_model():
    """Load the champion detector once per process."""
    model_path = BASE_DIR / "models" / "champion_detector.pkl"
    return load(model_path)


# -------------------------
# Load train data once
# -------------------------
@st.cache_data
def load_train_df():
    """Load train dataframe; some features depend on it."""
    csv_path = BASE_DIR / "data" / "train.csv"
    train = pd.read_csv(csv_path)

    # Basic cleaning
    train["answer"] = train["answer"].fillna("")
    if "topic" in train.columns:
        train["topic"] = train["topic"].fillna("")

    return train


# Instantiate heavy objects once (thanks to caching)
detector = load_model()
train_df = load_train_df()


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="AI Detection", layout="centered")

st.title("AI Cheating Detection System")
st.write("This model predicts whether a student's answer is AI-generated.")

st.markdown("### ✍️ Enter a student's answer")

topic = st.text_input("Topic (optional)")
answer = st.text_area("Student answer", height=220)


# -------------------------
# Prediction logic
# -------------------------
if st.button("Check for AI-generated content"):
    if not answer.strip():
        st.warning("Please provide a student answer.")
    else:
        # Build single-row test dataframe
        input_df = pd.DataFrame(
            {
                "id": ["input_1"],
                "topic": [topic],
                "answer": [answer],
            }
        )

        with st.spinner("Analyzing answer..."):
            # NOTE: detector.predict(train_df, test_df) → positional args
            preds = detector.predict(train_df, input_df)
            prob = float(preds[0])

        st.markdown("### 📊 Result")
        st.write(f"**Cheating probability:** `{prob:.3f}`")
        st.progress(prob)

        if prob > 0.7:
            st.error("⚠️ Likely AI-generated")
        elif prob > 0.05:
            st.warning("⚠️ Possibly AI-assisted")
        else:
            st.success("✅ Likely human-written")

import re
import numpy as np
import streamlit as st
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# --- Must match notebook exactly, or unpickling fails ---
def clean_text(text):
    # PASTE your exact clean_text() function from the notebook here
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    return text

class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return [clean_text(t) for t in X]

class TextStatsExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        feats = []
        for text in X:
            num_words   = len(text.split())
            num_sent    = len(re.split(r'[.!?]+', text.strip())) if text.strip() else 0
            num_chars   = len(text)
            num_special = len(re.findall(r'[^a-zA-Z0-9\s]', text))
            num_digits  = len(re.findall(r'\d', text))
            feats.append([num_words, num_sent, num_chars, num_special, num_digits])
        return np.array(feats)
# ----------------------------------------------------------

@st.cache_resource
def load_pipeline():
    return joblib.load('final_pipeline.pkl')

pipeline = load_pipeline()

st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon="🐦")
st.title("🐦 Twitter Sentiment Analysis")
st.write("Enter a tweet to predict its sentiment.")

user_input = st.text_area("Tweet text", height=100, placeholder="Type or paste a tweet...")

if st.button("Predict Sentiment", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        prediction = pipeline.predict([user_input])[0]
        st.subheader("Prediction")
        st.success(f"**{prediction}**")

        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba([user_input])[0]
            classes = pipeline.classes_
            st.write("Confidence:")
            for cls, p in sorted(zip(classes, proba), key=lambda x: -x[1]):
                st.write(f"- {cls}: {p:.2%}")

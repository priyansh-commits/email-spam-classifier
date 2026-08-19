import streamlit as st
import joblib
import nltk

# NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# Load model
model = joblib.load("spam_classifier_pipeline.pkl")

st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📱"
)

st.title("📱 SMS Spam Classifier")
st.write("Enter an SMS to check whether it is Spam or Ham.")

message = st.text_area(
    "Enter your message:",
    placeholder="Congratulations! You have won a free prize..."
)

if st.button("Predict"):

    if not message.strip():
        st.warning("Please enter a message.")

    else:
        prediction = model.predict([message])[0]
        probability = model.predict_proba([message])[0][1]

        if prediction == 1:
            st.error("🚨 SPAM")
            st.write(f"Spam probability: {probability * 100:.2f}%")
        else:
            st.success("✅ HAM")
            st.write(f"Spam probability: {probability * 100:.2f}%")

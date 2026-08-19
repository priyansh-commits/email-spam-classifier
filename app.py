import streamlit as st
import joblib

# Load trained pipeline
model = joblib.load("spam_classifier_pipeline.pkl")

# Page configuration
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📱",
    layout="centered"
)

# Title
st.title("📱 SMS Spam Classifier")
st.write("Enter an SMS message to check whether it is **Spam** or **Ham**.")

# Text input
message = st.text_area(
    "Enter your message:",
    placeholder="Congratulations! You have won a free prize..."
)

# Prediction
if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:
        prediction = model.predict([message])[0]

        probability = model.predict_proba([message])[0][1]

        if prediction == 1:
            st.error("🚨 SPAM")
            st.write(f"Spam probability: **{probability * 100:.2f}%**")

        else:
            st.success("✅ HAM")
            st.write(f"Spam probability: **{probability * 100:.2f}%**")

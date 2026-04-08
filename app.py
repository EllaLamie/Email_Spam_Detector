import streamlit as st
import joblib
from preprocessing import clean_text

# Load model and vectorizer
model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Page settings
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)

# Title
st.title("📧 Email Spam Detector")
st.write("Paste an email below and I'll predict whether it's spam or not.")

# Input box
email_text = st.text_area(
    "Enter email text:",
    height=200,
    placeholder="Example: Congratulations! You've won a free iPhone..."
)

# Predict button
if st.button("Check Email"):
    if email_text.strip() == "":
        st.warning("Please enter some email text first.")
    else:
        # Clean text
        cleaned_text = clean_text(email_text)

        # Convert to TF-IDF
        features = vectorizer.transform([cleaned_text])

        # Predict
        prediction = model.predict(features)[0]

        # Show result
        if prediction == "spam":
            st.error("🚨 This email looks like SPAM.")
        else:
            st.success("This email looks safe (not spam).")
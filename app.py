import joblib
import streamlit as st
from preprocessing import clean_text

model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
st.title("Email Spam Detector")
st.write("Paste your email below to check if it is spam")
user_input = st.text_area("Enter your email:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter your email")
    else:
        cleaned_text = clean_text(user_input)
        text_features = vectorizer.transform([cleaned_text])
        prediction = model.predict(text_features)[0]

        if prediction == "spam":
            st.error("Email is predicted to not be spam")
        else:
            st.success("Email is predicted to be spam")

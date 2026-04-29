import streamlit as st
import pickle
import re

# Load model (tumhare file name ke hisaab se)
with open("logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

# Prediction function
def predict_news(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)

    if prediction[0] == 0:
        return "Fake News ❌"
    else:
        return "Real News ✅"

# UI
st.title("📰 Fake News Detection")

input_text = st.text_area("Enter News Text")

if st.button("Predict"):
    result = predict_news(input_text)
    st.subheader(result)

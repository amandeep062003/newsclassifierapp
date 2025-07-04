import streamlit as st
import pickle
import re
import string
from nltk.corpus import stopwords
import nltk

# Ensure stopwords are downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
with open("model/news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Clean user input (improved)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # remove URLs
    text = re.sub(r'\@w+|\#', '', text)  # remove mentions/hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # keep only letters
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuations
    text = " ".join([word for word in text.split() if word not in stop_words])  # remove stopwords
    return text

# Page settings
st.set_page_config(page_title="News Classifier", layout="centered")
st.title("📰 News Category Classifier")
st.write("Enter a news **headline or article** and get its predicted category!")

# Input from user
user_input = st.text_area("✍️ Enter your news text:")

if st.button("🔍 Predict Category"):
    cleaned = clean_text(user_input)

    if not cleaned or len(cleaned.split()) < 2:
        st.warning("⚠️ Please enter a valid news sentence or headline (e.g., 'Tech Stocks Plunge').")

    else:
        vectorized = vectorizer.transform([cleaned])
        proba = model.predict_proba(vectorized)[0]
        confidence = max(proba)
        prediction = model.classes_[proba.argmax()]

        category_map = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Sci/Tech"
        }

        if confidence < 0.6:
            st.warning("🤔 Not confident about this prediction. Try rephrasing the sentence.")
        else:
            st.success(f"📌 Predicted Category: **{category_map[int(prediction)]}** ({confidence*100:.2f}% confidence)")

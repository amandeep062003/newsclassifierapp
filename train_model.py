import pandas as pd
import re
import pickle
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download("stopwords")

# Load dataset
df = pd.read_csv("train.csv", header=None, names=["class", "title", "description"])
df["text"] = df["title"] + " " + df["description"]

# Stopwords once
stop_words = set(stopwords.words("english"))

# Clean function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # remove URLs
    text = re.sub(r'\@w+|\#', '', text)  # remove mentions/hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # keep only letters
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuations
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Apply cleaning
df["clean_text"] = df["text"].apply(clean_text)

# Vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean_text"])
y = df["class"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = MultinomialNB()
model.fit(X_train, y_train)
TfidfVectorizer(ngram_range=(1, 2))  # bigrams help learn phrases like 'shubman gill', 'first innings'


# Save
import os
os.makedirs("model", exist_ok=True)

with open("model/news_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model trained and saved.")

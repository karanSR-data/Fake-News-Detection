# cleaning.py
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()

    # Remove numbers, punctuation etc.
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove stopwords
    text = " ".join([word for word in text.split() if word not in stop_words])

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("data/combined.csv")
    df["text"] = df["text"].apply(preprocess_text)
    df.to_csv("data/combined_cleaned.csv", index=False)
    print(df.head())

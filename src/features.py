# features.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from cleaning import preprocess_text

def load_cleaned_data(path="data/combined.csv"):
    df = pd.read_csv(path)

    # clean using SAME preprocessing as prediction
    df["text"] = df["text"].astype(str).apply(preprocess_text)

    # remove empty rows
    df = df[df["text"].str.strip() != ""]
    df = df.reset_index(drop=True)

    return df


def extract_features(df):
    vectorizer = TfidfVectorizer(
        max_features=5000,
    )

    X = vectorizer.fit_transform(df['text'])
    y = df['label']

    return X, y, vectorizer


if __name__ == "__main__":
    df = load_cleaned_data()
    print(df.head())

    X, y, vectorizer = extract_features(df)
    print("Shape:", X.shape)
    print("Labels:", y.value_counts())

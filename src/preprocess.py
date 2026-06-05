# preprocess.py
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from cleaning import preprocess_text
from load_data import load_data

if __name__ == "__main__":
    df = load_data("data/Fake.csv", "data/True.csv")
    df['text_clean'] = df['text'].apply(preprocess_text)
    
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['text_clean'])
    y = df['label']

    # Save vectorizer and features for future use
    joblib.dump(vectorizer, '../data/tfidf_vectorizer.pkl')
    joblib.dump((X, y), '../data/features_labels.pkl')
    
    print(X.shape, y.shape)

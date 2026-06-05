# src/predict_pipeline.py
import joblib
import os

BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
VECT_PATH = os.path.join(BASE, "tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join(BASE, "best_model_logistic.pkl")

if not os.path.exists(VECT_PATH) or not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model or vectorizer file missing")

vectorizer = joblib.load(VECT_PATH)
model = joblib.load(MODEL_PATH)

def predict_news(text):
    # NOTE: model was trained on raw text → so don’t clean here
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    return "Fake News ❌" if pred == 1 else "Real News ✅"

if __name__ == "__main__":
    s = input("Enter text:\n")
    print(predict_news(s))


# “Bhai, real world app mein prediction chahiye.”
# → Proper preprocessing
# → Stable results
# → Deployment
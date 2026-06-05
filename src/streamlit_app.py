import streamlit as st
import joblib
from cleaning import preprocess_text   #  use SAME cleaning function

# ------------------
# Load model + vectorizer
# ------------------
vectorizer = joblib.load("data/tfidf_vectorizer.pkl")
model = joblib.load("data/best_model_logistic.pkl")   # whichever got saved


# ------------------
# Prediction
# ------------------
def predict_news(text):
    cleaned = preprocess_text(text)      # exact same cleaning
    vectorized = vectorizer.transform([cleaned])
    pred = model.predict(vectorized)[0]
    return "Fake News ❌" if pred == 1 else "Real News ✅"


# ------------------
# Streamlit UI
# ------------------
st.title("📰 Fake News Detection App")
st.write("Enter any news headline or article below:")

user_input = st.text_area("News text here:", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        result = predict_news(user_input)
        st.subheader("Prediction:")
        st.success(result)

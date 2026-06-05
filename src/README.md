# Fake News Detection using Machine Learning

## Description
This project detects fake news articles using machine learning and natural language
processing techniques. The model classifies news as Fake or Real based on textual data.

## Dataset
- Source: Kaggle
- Fake news: ~25,000 rows
- True news: ~21,000 rows
- Columns: title, text, subject, date
- A label column was added after merging datasets

## Tech Stack
- Python
- NumPy
- Pandas
- Scikit-learn
- NLTK
- Streamlit

## Machine Learning
- Feature Extraction: TF-IDF Vectorizer
- Algorithms:
  - Logistic Regression
  - Multinomial Naive Bayes
- Final Model: Logistic Regression

## Project Structure
- Modular pipeline with separate files for:
  - Data loading
  - Cleaning & preprocessing
  - Feature extraction
  - Model training
  - Prediction
  - Streamlit app

## How to Run
1. Install dependencies
2. Run the Streamlit app:
   streamlit run src/streamlit_app.py

## Author
Karan Singh Rajput| 📧 Email ID -> karansinghrajput2022@vitbhopal.ac.in 
🔗 GitHub:[https://github.com/karanSR-data]

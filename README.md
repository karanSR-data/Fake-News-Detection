# Fake News Detection

A machine learning project that classifies news articles as **Fake** or **True** using Natural Language Processing (NLP).

The goal of this project was to understand the complete machine learning workflow, starting from data preprocessing and text vectorization to model training and deployment through a Streamlit web application.

## Project Overview

News articles are converted into numerical features using **TF-IDF Vectorization** and then classified using a **Logistic Regression** model.

The application allows users to enter any news headline or article and instantly receive a prediction indicating whether the content is likely to be fake or genuine.

## Features

* Text preprocessing and cleaning
* TF-IDF feature extraction
* Logistic Regression classifier
* Interactive Streamlit web interface
* Real-time prediction on custom news articles
* Saved model and vectorizer for deployment

## Project Structure

```text
Fake-News-Detection/
│
├── src/
│   └── streamlit_app.py
│
├── data/
│   ├── best_model_logistic.pkl
│   └── tfidf_vectorizer.pkl
│
├── requirement.txt
└── README.md
```

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* NLTK
* Streamlit

## Model Pipeline

1. Collect and clean news article data.
2. Convert text into numerical features using TF-IDF.
3. Train a Logistic Regression classifier.
4. Save the trained model and vectorizer.
5. Deploy the prediction system using Streamlit.

## Running the Project

Clone the repository:

```bash
git clone https://github.com/karanSR-data/Fake-News-Detection.git
```

Install dependencies:

```bash
pip install -r requirement.txt
```

Run the Streamlit application:

```bash
streamlit run src/streamlit_app.py
```

## Sample Prediction

Input:

```text
Scientists discover a new method to improve solar panel efficiency.
```

Output:

```text
Prediction: True News
```

## What I Learned

Through this project, I gained practical experience in:

* Text preprocessing for NLP tasks
* Feature engineering using TF-IDF
* Training and evaluating classification models
* Saving and loading machine learning models
* Building simple ML web applications with Streamlit

## Future Improvements

* Experiment with advanced models such as XGBoost and BERT
* Add confidence scores for predictions
* Improve text preprocessing pipeline
* Deploy the application on Hugging Face Spaces or Streamlit Cloud

## Author

Karan Singh Rajput
Integrated M.Tech (Computational Data Science)
VIT Bhopal University

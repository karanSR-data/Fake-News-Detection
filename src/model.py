# src/train_models.py
"""
Train baseline models for Fake News Detection:
 - LogisticRegression (fast, interpretable)
 - MultinomialNB (classic for text)
Saves vectorizer and best model to disk.
"""

from features import load_cleaned_data, extract_features
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import numpy as np

def main():
    # 1) Load cleaned data and extract TF-IDF features
    df = load_cleaned_data("data/combined_cleaned.csv")   # path used in your features.py
    X, y, vectorizer = extract_features(df)              # X is sparse matrix, y is series

    print("Dataset size:", X.shape, "Labels:", np.bincount(y))
    # 2) Train-test split (use stratify to keep class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3) Baseline model 1: Logistic Regression
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=5000, solver='saga')  # saga works well with sparse TF-IDF
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    print("Logistic Regression accuracy:", accuracy_score(y_test, y_pred_lr))
    print(classification_report(y_test, y_pred_lr))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_lr))

    # 4) Baseline model 2: Multinomial Naive Bayes
    print("\nTraining MultinomialNB...")
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    print("MultinomialNB accuracy:", accuracy_score(y_test, y_pred_nb))
    print(classification_report(y_test, y_pred_nb))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_nb))

    # 5) Save vectorizer and the better model
    # Decide which model is better by accuracy (we can change metric later)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    acc_nb = accuracy_score(y_test, y_pred_nb)

    if acc_lr >= acc_nb:
        best_model = lr
        best_name = "logistic"
        best_acc = acc_lr
    else:
        best_model = nb
        best_name = "naive_bayes"
        best_acc = acc_nb

    joblib.dump(vectorizer, "data/tfidf_vectorizer.pkl")
    joblib.dump(best_model, f"data/best_model_{best_name}.pkl")
    print(f"\nSaved vectorizer and best model ({best_name}) with accuracy {best_acc:.4f}")

if __name__ == "__main__":
    main()

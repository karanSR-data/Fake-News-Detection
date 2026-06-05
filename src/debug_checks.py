# debug_checks.py  -- run in project root
import joblib, pandas as pd, numpy as np
from cleaning import preprocess_text   # your cleaning function
from features import load_cleaned_data  # loads combined_cleaned.csv

# 1) load artifacts
vectorizer = joblib.load("data/tfidf_vectorizer.pkl")
model = joblib.load("data/best_model_logistic.pkl")  # ensure this is the file you expect

print("Vectorizer type:", type(vectorizer))
print("Model type:", type(model))

# 2) load data used for training (or the combined cleaned file)
df = load_cleaned_data("data/combined_cleaned.csv")
print("Loaded data shape:", df.shape)
print("Label distribution:\n", df['label'].value_counts())

# 3) Recreate the exact cleaned text we will feed to vectorizer
df['text_for_pred'] = df['text'].astype(str).apply(preprocess_text)
# quick check: any empty after cleaning?
empty_count = (df['text_for_pred'].str.strip() == "").sum()
print("Empty rows after cleaning:", empty_count)

# 4) Transform with saved vectorizer (do NOT fit again)
X_all = vectorizer.transform(df['text_for_pred'])

# 5) Compute model predictions and accuracy on full dataset
y_true = df['label'].values
y_pred = model.predict(X_all)
acc = (y_pred == y_true).mean()
print("Model accuracy on combined_cleaned.csv (using saved artifacts):", acc)

# 6) Show confusion and some mismatch examples
from sklearn.metrics import confusion_matrix, classification_report
print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
print("Classification report:\n", classification_report(y_true, y_pred, digits=4))

# 7) Print first 10 mismatches for human inspection
mismatch_idx = np.where(y_pred != y_true)[0]
print("Total mismatches:", len(mismatch_idx))
for i in mismatch_idx[:10]:
    print("----")
    print("Index:", i)
    print("True label:", y_true[i], " Pred:", y_pred[i])
    print("Original text (first 300 chars):")
    print(df.loc[i, 'text'][:300])
    print("Cleaned text (first 200 chars):")
    print(df.loc[i, 'text_for_pred'][:200])

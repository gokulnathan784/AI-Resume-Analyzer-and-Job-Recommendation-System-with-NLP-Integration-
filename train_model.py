# train_model.py (with Kaggle dataset)
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Make sure models folder exists
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("data/resumes.csv")

# Kaggle dataset usually has columns: "Category" and "Resume"
df['clean'] = df['Resume'].astype(str).str.lower()
df['label'] = df['Category']

# Check label counts
label_counts = df['label'].value_counts()
print("Label counts:\n", label_counts.to_string())

# Check if enough samples exist for split
min_count = label_counts.min() if not label_counts.empty else 0
if min_count < 2 or len(df) < 8:
    print("⚠️ Not enough samples per class for a reliable test split.")
    print("Training on ALL data and skipping evaluation (please add more samples).")

    vec = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
    X = vec.fit_transform(df['clean'])

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X, df['label'])

    joblib.dump(vec, "models/tfidf_vectorizer.joblib")
    joblib.dump(clf, "models/resume_classifier.joblib")
    print("✅ Model trained on all data and saved to models/")
else:
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean'], df['label'], test_size=0.25, random_state=42, stratify=df['label']
    )

    vec = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, y_train)

    pred = clf.predict(Xte)

    print("\n📊 Classification Report:")
    print(classification_report(y_test, pred, zero_division=0))

    joblib.dump(vec, "models/tfidf_vectorizer.joblib")
    joblib.dump(clf, "models/resume_classifier.joblib")
    print("✅ Model trained, evaluated, and saved to models/")

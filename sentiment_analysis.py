"""
Amazon Review Sentiment Analysis
=================================
A complete ML pipeline to classify customer reviews as Positive or Negative.

Steps:
1. Generate/Load dataset
2. Preprocess text (lowercase, remove punctuation, stopwords, tokenize)
3. Vectorize using TF-IDF
4. Train Logistic Regression, Naive Bayes, Random Forest
5. Evaluate all models
6. Predict on new reviews
"""

import pandas as pd
import numpy as np
import re
import string
import warnings
import joblib
import os
warnings.filterwarnings("ignore")

# ── NLP & ML ────────────────────────────────────────────────────────────────
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)

# Download NLTK data quietly
for pkg in ["punkt", "stopwords", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

# ════════════════════════════════════════════════════════════════════════════
# STEP 1 ─ Dataset
# ════════════════════════════════════════════════════════════════════════════

def create_sample_dataset():
    """
    Creates a realistic synthetic Amazon review dataset.
    In a real project you would load a CSV file like:
        df = pd.read_csv("amazon_reviews.csv")
    """
    positive_reviews = [
        "This product is absolutely amazing! Best purchase I've made all year.",
        "Incredible quality, arrived on time, and works perfectly. Highly recommend!",
        "Love this item! It exceeded all my expectations. Five stars.",
        "Great value for money. The build quality is superb and durable.",
        "Fantastic product! Customer service was also very helpful.",
        "Works exactly as described. Very happy with my purchase.",
        "Excellent quality product. Would definitely buy again.",
        "Outstanding product. Delivered quickly and packaged well.",
        "This is a must-have. Solid construction and easy to use.",
        "Perfect! Does exactly what it's supposed to do. Very satisfied.",
        "Super fast shipping and the product looks even better in person.",
        "Wonderful item, my family loves it. Great gift idea too!",
        "Really impressed by the quality. Way better than I expected.",
        "Bought this as a replacement and it's even better than the original.",
        "Absolutely love it! Will be recommending to all my friends.",
        "Good product, good price, fast delivery. What more can you ask for?",
        "Very pleased with this purchase. Sturdy and well made.",
        "Amazing value. Works great and the instructions were easy to follow.",
        "Top-notch quality. Couldn't be happier with this buy.",
        "Brilliant product. Exactly what I needed and at a great price.",
    ]
    negative_reviews = [
        "Terrible product. Broke after just two days of use.",
        "Complete waste of money. Does not work as advertised at all.",
        "Very disappointed. The quality is extremely poor and cheap.",
        "Arrived damaged and customer service was completely unhelpful.",
        "Would not recommend. Stopped working within a week.",
        "Absolute garbage. Nothing like the pictures shown online.",
        "Really bad experience. The item is flimsy and cheaply made.",
        "Horrible product. It's already falling apart after one use.",
        "Defective item received. Had to return it immediately.",
        "Worst purchase ever. Doesn't do what the description says.",
        "Very poor quality. The materials feel cheap and flimsy.",
        "Total disappointment. Instructions are wrong and product fails.",
        "Not worth a single penny. Already broken and useless.",
        "Faulty product, returned it. Seller ignored my refund request.",
        "Ugly and poorly designed. Nothing matches the listing photos.",
        "Cheap knockoff. Smells bad and works only sometimes.",
        "Huge let-down. The product quality is far below expectations.",
        "Don't buy this! It overheated and stopped working after one hour.",
        "Extremely poor. The product arrived incomplete with missing parts.",
        "Useless junk. Completely broken out of the box.",
    ]

    reviews  = positive_reviews + negative_reviews
    sentiments = ["Positive"] * len(positive_reviews) + ["Negative"] * len(negative_reviews)

    df = pd.DataFrame({"Review": reviews, "Sentiment": sentiments})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)   # shuffle
    return df


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 ─ Text Preprocessing
# ════════════════════════════════════════════════════════════════════════════

STOP_WORDS = set(stopwords.words("english"))

def preprocess_text(text: str) -> str:
    """
    Clean and normalise a review string.

    1. Lowercase everything
    2. Remove punctuation and special characters
    3. Tokenize into words
    4. Remove stopwords (common words like 'the', 'is', 'and')
    5. Rejoin into a cleaned string
    """
    # 1. Lowercase
    text = text.lower()

    # 2. Remove punctuation & digits
    text = re.sub(r"[^a-z\s]", "", text)

    # 3. Tokenize
    tokens = word_tokenize(text)

    # 4. Remove stopwords
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

    # 5. Rejoin
    return " ".join(tokens)


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 ─ Vectorization (TF-IDF)
# ════════════════════════════════════════════════════════════════════════════

def build_vectorizer(X_train):
    """
    TF-IDF: Term Frequency–Inverse Document Frequency.
    Converts text into numbers, giving higher weight to rare but
    important words and lower weight to very common ones.
    """
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    return vectorizer, X_train_vec


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 ─ Train Models
# ════════════════════════════════════════════════════════════════════════════

def train_models(X_train_vec, y_train):
    """Train three classifiers and return them in a dict."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Naive Bayes":         MultinomialNB(),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    }
    trained = {}
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        trained[name] = model
        print(f"  ✓ Trained: {name}")
    return trained


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 & 6 ─ Evaluate + Confusion Matrix
# ════════════════════════════════════════════════════════════════════════════

def evaluate_models(models, vectorizer, X_test, y_test):
    """Print a full evaluation report for every model."""
    results = {}
    X_test_vec = vectorizer.transform(X_test)

    print("\n" + "═"*65)
    print("  MODEL EVALUATION REPORT")
    print("═"*65)

    for name, model in models.items():
        y_pred = model.predict(X_test_vec)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label="Positive", zero_division=0)
        rec  = recall_score(y_test, y_pred, pos_label="Positive", zero_division=0)
        f1   = f1_score(y_test, y_pred, pos_label="Positive", zero_division=0)
        cm   = confusion_matrix(y_test, y_pred, labels=["Positive", "Negative"])

        results[name] = {"accuracy": acc, "precision": prec,
                         "recall": rec, "f1": f1, "cm": cm}

        print(f"\n┌─ {name} {'─'*(50-len(name))}┐")
        print(f"│  Accuracy  : {acc:.4f}    Precision : {prec:.4f}")
        print(f"│  Recall    : {rec:.4f}    F1-Score  : {f1:.4f}")
        print(f"│  Confusion Matrix (rows=Actual, cols=Predicted):")
        print(f"│              Pos    Neg")
        print(f"│  Actual Pos  {cm[0][0]:>4}   {cm[0][1]:>4}")
        print(f"│  Actual Neg  {cm[1][0]:>4}   {cm[1][1]:>4}")
        print(f"└{'─'*54}┘")

    return results


# ════════════════════════════════════════════════════════════════════════════
# STEP 7 ─ Predict New Review
# ════════════════════════════════════════════════════════════════════════════

def predict_review(review: str, model, vectorizer) -> str:
    """
    Given a raw customer review string, preprocess it, vectorize it,
    and return 'Positive' or 'Negative'.
    """
    cleaned  = preprocess_text(review)
    vec      = vectorizer.transform([cleaned])
    label    = model.predict(vec)[0]
    proba    = model.predict_proba(vec)[0]
    classes  = model.classes_.tolist()
    conf     = proba[classes.index(label)]
    return label, conf


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═"*65)
    print("  AMAZON REVIEW SENTIMENT ANALYSIS")
    print("═"*65)

    # ── Step 1: Load data ────────────────────────────────────────────────
    print("\n[1] Loading dataset …")
    df = create_sample_dataset()
    print(f"    Total reviews  : {len(df)}")
    print(f"    Positive       : {(df.Sentiment=='Positive').sum()}")
    print(f"    Negative       : {(df.Sentiment=='Negative').sum()}")

    # ── Step 2: Preprocess ───────────────────────────────────────────────
    print("\n[2] Preprocessing text …")
    df["Cleaned"] = df["Review"].apply(preprocess_text)
    print(f"    Sample original : {df['Review'].iloc[0]}")
    print(f"    Sample cleaned  : {df['Cleaned'].iloc[0]}")

    # ── Step 3: Train/test split + TF-IDF ───────────────────────────────
    print("\n[3] Vectorizing with TF-IDF …")
    X = df["Cleaned"]
    y = df["Sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    vectorizer, X_train_vec = build_vectorizer(X_train)
    print(f"    Vocab size : {len(vectorizer.vocabulary_):,}")
    print(f"    Train rows : {X_train_vec.shape[0]}   Test rows : {len(X_test)}")

    # ── Step 4: Train ────────────────────────────────────────────────────
    print("\n[4] Training models …")
    models = train_models(X_train_vec, y_train)

    # ── Step 5 & 6: Evaluate ─────────────────────────────────────────────
    results = evaluate_models(models, vectorizer, X_test, y_test)

    # ── Step 7: Save best model ──────────────────────────────────────────
    best_name = max(results, key=lambda n: results[n]["f1"])
    best_model = models[best_name]
    print(f"\n[7] Best model by F1: {best_name}")

    os.makedirs("model_artifacts", exist_ok=True)
    joblib.dump(best_model, "model_artifacts/best_model.pkl")
    joblib.dump(vectorizer,  "model_artifacts/vectorizer.pkl")
    print("    Saved → model_artifacts/best_model.pkl & vectorizer.pkl")

    # ── Example predictions ──────────────────────────────────────────────
    print("\n" + "═"*65)
    print("  EXAMPLE PREDICTIONS  (using best model)")
    print("═"*65)
    examples = [
        "This product is absolutely fantastic! Works perfectly.",
        "Terrible quality, broke after two days. Total waste.",
        "Decent product but the packaging was damaged on arrival.",
        "Amazing! Exceeded all expectations. Will buy again.",
        "Horrible experience. The item stopped working immediately.",
    ]
    for rev in examples:
        label, conf = predict_review(rev, best_model, vectorizer)
        emoji = "✅" if label == "Positive" else "❌"
        print(f"\n  Review : {rev[:60]}")
        print(f"  Result : {emoji} {label}  (confidence {conf:.1%})")

    print("\n" + "═"*65)
    print("  Pipeline complete! Run app.py for the Streamlit UI.")
    print("═"*65 + "\n")


if __name__ == "__main__":
    main()

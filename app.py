"""
Streamlit UI for Amazon Review Sentiment Analysis
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

import matplotlib.pyplot as plt
import seaborn as sns

for pkg in ["punkt", "stopwords", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))

# ── helpers ──────────────────────────────────────────────────────────────────

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


def create_sample_dataset():
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
    reviews = positive_reviews + negative_reviews
    sentiments = ["Positive"] * 20 + ["Negative"] * 20
    df = pd.DataFrame({"Review": reviews, "Sentiment": sentiments})
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


@st.cache_resource
def train_pipeline():
    df = create_sample_dataset()
    df["Cleaned"] = df["Review"].apply(preprocess_text)
    X, y = df["Cleaned"], df["Sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Naive Bayes":         MultinomialNB(),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    }
    results = {}
    for name, m in models.items():
        m.fit(X_train_vec, y_train)
        y_pred = m.predict(X_test_vec)
        results[name] = {
            "model":     m,
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, pos_label="Positive", zero_division=0),
            "recall":    recall_score(y_test, y_pred, pos_label="Positive", zero_division=0),
            "f1":        f1_score(y_test, y_pred, pos_label="Positive", zero_division=0),
            "cm":        confusion_matrix(y_test, y_pred, labels=["Positive", "Negative"]),
        }
    return vectorizer, results, df


# ── Streamlit page ────────────────────────────────────────────────────────────

st.set_page_config(page_title="Amazon Sentiment Analyser", page_icon="🛒", layout="wide")

st.title("🛒 Amazon Review Sentiment Analysis")
st.markdown("Classify customer reviews as **Positive** or **Negative** using ML.")

with st.spinner("Training models …"):
    vectorizer, results, df = train_pipeline()

tabs = st.tabs(["🔍 Predict", "📊 Model Performance", "🗂️ Dataset"])

# ── TAB 1: Predict ────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Try a Customer Review")

    model_name = st.selectbox("Choose a model", list(results.keys()))
    user_review = st.text_area("Paste a customer review here", height=130,
                               placeholder="e.g. This product is absolutely amazing …")

    if st.button("Analyse Sentiment 🚀", use_container_width=True):
        if user_review.strip():
            model = results[model_name]["model"]
            cleaned = preprocess_text(user_review)
            vec     = vectorizer.transform([cleaned])
            label   = model.predict(vec)[0]
            proba   = model.predict_proba(vec)[0]
            classes = model.classes_.tolist()
            conf    = proba[classes.index(label)]

            if label == "Positive":
                st.success(f"✅ **Positive**  —  Confidence: {conf:.1%}")
            else:
                st.error(f"❌ **Negative**  —  Confidence: {conf:.1%}")

            with st.expander("Cleaned text (after preprocessing)"):
                st.code(cleaned)
        else:
            st.warning("Please enter a review first.")

    st.divider()
    st.subheader("Quick Examples")
    examples = {
        "⭐⭐⭐⭐⭐ Excellent": "This product is absolutely fantastic! Works perfectly, great value.",
        "⭐ Very bad": "Terrible quality, broke after two days. Complete waste of money.",
        "🤔 Mixed": "Decent product but packaging was damaged on arrival. Works okay.",
    }
    for label_ex, text in examples.items():
        col1, col2 = st.columns([3, 1])
        col1.write(f"**{label_ex}**: _{text}_")
        if col2.button("Try →", key=label_ex):
            model = results[model_name]["model"]
            cleaned = preprocess_text(text)
            vec     = vectorizer.transform([cleaned])
            pred    = model.predict(vec)[0]
            emoji   = "✅" if pred == "Positive" else "❌"
            col2.write(f"{emoji} {pred}")


# ── TAB 2: Model Performance ──────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Model Evaluation Metrics")

    metric_df = pd.DataFrame({
        name: {k: round(v, 4) for k, v in r.items() if k in ("accuracy","precision","recall","f1")}
        for name, r in results.items()
    }).T.rename(columns={"f1": "F1-Score"})
    st.dataframe(metric_df.style.highlight_max(axis=0, color="#d4edda"), use_container_width=True)

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 3.5))
    x = np.arange(len(results))
    width = 0.2
    metrics = ["accuracy", "precision", "recall", "f1"]
    colors  = ["#4e79a7","#f28e2b","#59a14f","#e15759"]
    for i, (m, c) in enumerate(zip(metrics, colors)):
        vals = [results[n][m] for n in results]
        ax.bar(x + i*width, vals, width, label=m.capitalize(), color=c)
    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels(list(results.keys()), fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8)
    ax.set_title("Metrics by Model", fontsize=11)
    st.pyplot(fig)

    # Confusion matrices
    st.subheader("Confusion Matrices")
    cols = st.columns(3)
    for col, (name, r) in zip(cols, results.items()):
        fig2, ax2 = plt.subplots(figsize=(3.5, 3))
        sns.heatmap(r["cm"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Pos","Neg"], yticklabels=["Pos","Neg"], ax=ax2)
        ax2.set_title(name, fontsize=9)
        ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
        col.pyplot(fig2)


# ── TAB 3: Dataset ────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Sample Dataset")
    st.dataframe(df[["Review","Sentiment"]], use_container_width=True)
    pos = (df.Sentiment=="Positive").sum()
    neg = (df.Sentiment=="Negative").sum()
    fig3, ax3 = plt.subplots(figsize=(4, 4))
    ax3.pie([pos, neg], labels=["Positive","Negative"], autopct="%1.0f%%",
            colors=["#59a14f","#e15759"], startangle=90)
    ax3.set_title("Sentiment Distribution")
    st.pyplot(fig3)

import pandas as pd
import json
import numpy as np
import nltk
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from collections import Counter

# -------------------------
# NLTK Setup
# -------------------------
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

# -------------------------
# Load Questions Mapping
# -------------------------
with open(r"C:\Users\Administrator\OneDrive\Attachments\Desktop\cap\code\questions1.json", "r", encoding="utf-8") as f:
    QUESTIONS = json.load(f)

CSV_PATH = r"C:\Users\Administrator\OneDrive\Attachments\Desktop\cap\Response.csv"

# -------------------------
# Load Deep Learning Model + Tokenizer
# -------------------------
MODEL_PATH = 'suicide_risk_model.h5'
TOKENIZER_PATH = 'tokenizer.pickle'

# ✅ Rename to dl_model (not ml_model)
dl_model = load_model(MODEL_PATH, compile=False)

with open(TOKENIZER_PATH, "rb") as handle:
    tokenizer = pickle.load(handle)

MAX_SEQUENCE_LENGTH = 100
EXPECTED_YESNO = 10
EXPECTED_RATING = 10

# -------------------------
# Rule-Based Risk Calculation
# -------------------------
def calculate_rule_based_risk(yesno_arr, rating_arr, text_arr):
    # Flatten
    yesno_arr = np.array(yesno_arr).flatten()
    rating_arr = np.array(rating_arr).flatten()
    text_list = list(text_arr[0])

    # Keep only actual answers for display
    actual_yesno = yesno_arr[:len(QUESTIONS['yesno'])]
    actual_rating = rating_arr[:len(QUESTIONS['rating'])]

    # Yes/No score (scaled 0-5)
    yesno_score = actual_yesno.sum() / max(len(actual_yesno), 1) * 5

    # Rating score (scaled 0-3)
    rating_score = actual_rating.mean() / 5 * 3

    # NLP sentiment score (scaled 0-2)
    sentiment_scores = []
    sentiment_details = []
    for t in text_list:
        if t.strip():
            sentiment = sia.polarity_scores(t)['compound']
            scaled_sentiment = (1 - sentiment) / 2
            sentiment_scores.append(scaled_sentiment)
            sentiment_details.append((t, sentiment, scaled_sentiment))
        else:
            sentiment_scores.append(0)
            sentiment_details.append((t, 0, 0))
    nlp_score = np.mean(sentiment_scores) * 2

    overall = yesno_score + rating_score + nlp_score  # max possible 5+3+2 = 10

    return overall, yesno_score, rating_score, nlp_score, actual_yesno, actual_rating, sentiment_details

def interpret_score(score):
    if score <= 2:
        return "Very Low Risk"
    elif score <= 4:
        return "Low Risk"
    elif score <= 6:
        return "Moderate Risk"
    else:
        return "High Risk"

# -------------------------
# DL Model Risk Prediction
# -------------------------
def calculate_dl_risk(texts, yesno_arr, rating_arr):
    # Pad to expected length for DL model
    yesno_arr = yesno_arr + [0] * (EXPECTED_YESNO - len(yesno_arr))
    rating_arr = rating_arr + [0] * (EXPECTED_RATING - len(rating_arr))

    seqs = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=MAX_SEQUENCE_LENGTH)

    pred = dl_model.predict([padded, np.array([yesno_arr]), np.array([rating_arr])], verbose=0)
    score = float(pred.flatten()[0])

    if score <= 2:
        return "Very Low Risk", score
    elif score <= 4:
        return "Low Risk", score
    elif score <= 6:
        return "Moderate Risk", score
    else:
        return "High Risk", score

# -------------------------
# Voting System
# -------------------------
def voting_system(rule_risk, dl_risk):
    levels = ["Very Low Risk", "Low Risk", "Moderate Risk", "High Risk"]
    if rule_risk == "High Risk" or dl_risk == "High Risk":
        return "High Risk"
    if rule_risk == dl_risk:
        return rule_risk
    return max(rule_risk, dl_risk, key=lambda x: levels.index(x))

# -------------------------
# Process each row
# -------------------------
def process_row(row):
    yesno_ans = [1 if str(row.get(q, "No")).strip().lower() == "yes" else 0 for q in QUESTIONS['yesno']]
    rating_ans = [int(row.get(q, 0)) if str(row.get(q, 0)).isdigit() else 0 for q in QUESTIONS['rating']]
    text_ans = [str(row.get(q, "")) for q in QUESTIONS['text']]
    return yesno_ans, rating_ans, text_ans

# -------------------------
# Runner with Detailed Output
# -------------------------
def run_csv():
    risk_summary = []

    df = pd.read_csv(CSV_PATH)

    for idx, row in df.iterrows():
        name = row.get("Name", "Unknown")
        email = row.get("Username", "Unknown")
        yesno_ans, rating_ans, text_ans = process_row(row)

        print("\n" + "=" * 80)
        print(f"Processing {name} ({email})")
        print("-" * 80)

        # Rule-Based
        overall_rule, yes_score, rating_score, nlp_score, actual_yesno, actual_rating, sentiment_details = calculate_rule_based_risk(
            [yesno_ans], [rating_ans], [text_ans])
        rule_risk = interpret_score(overall_rule)

        print("Yes/No Answers:", actual_yesno)
        print(f"Yes/No Score (0-5): {yes_score:.2f}")

        print("Rating Answers:", actual_rating)
        print(f"Rating Score (0-3): {rating_score:.2f}")

        print("Text Sentiments (compound, scaled 0-1):")
        for t, c, s in sentiment_details:
            print(f"  Text: '{t}' | Compound: {c:.2f} | Scaled: {s:.2f}")
        print(f"NLP Score (0-2): {nlp_score:.2f}")

        print(f"Overall Rule-Based Score: {overall_rule:.2f}/10 → Risk: {rule_risk}")

        # DL Model
        dl_risk, dl_score = calculate_dl_risk([" ".join(text_ans)], yesno_ans, rating_ans)
        print(f"DL Model Score: {dl_score:.2f}/10 → Risk: {dl_risk}")

        # Final Voting
        final_risk = voting_system(rule_risk, dl_risk)
        print(f"Final Risk (Voting System): {final_risk}")

        risk_summary.append((name, final_risk))

    plot_risk_summary(risk_summary)

# -------------------------
# Graph Summary
# -------------------------
def plot_risk_summary(risk_summary):
    if not risk_summary:
        print("No data to plot.")
        return

    counts = Counter([level for _, level in risk_summary])
    categories = ["Very Low Risk", "Low Risk", "Moderate Risk", "High Risk"]
    values = [counts.get(cat, 0) for cat in categories]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(categories, values, color=['green', 'lightgreen', 'orange', 'red'])
    plt.title("Final Risk Level Distribution (Voting System)")
    plt.ylabel("Number of Participants")
    plt.xlabel("Risk Levels")

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, str(value),
                 ha='center', va='bottom', fontsize=10)

    plt.ylim(0, max(values) + 2)
    plt.show()

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    run_csv()

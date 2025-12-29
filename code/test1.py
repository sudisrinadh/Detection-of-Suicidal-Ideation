import pandas as pd
import json
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from collections import Counter

# Download required NLTK data if not present
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
# Risk Calculation Logic
# -------------------------
def calculate_overall_risk(yesno_arr, rating_arr, text_arr):
    yesno_arr = np.array(yesno_arr).flatten()
    rating_arr = np.array(rating_arr).flatten()
    text_list = list(text_arr[0])  # assuming single row

    # Yes/No score
    yesno_score = yesno_arr.sum() / max(len(yesno_arr), 1) * 5  # scale to 5

    # Rating score
    rating_score = rating_arr.mean() / 5 * 3  # scale to 3

    # NLP sentiment score
    sentiment_scores = []
    for t in text_list:
        if t.strip():
            sentiment = sia.polarity_scores(t)['compound']  # -1 to 1
            sentiment_scores.append((1 - sentiment) / 2)  # scale to 0-1
        else:
            sentiment_scores.append(0)
    nlp_score = np.mean(sentiment_scores) * 2  # scale to 0-2

    overall = yesno_score + rating_score + nlp_score
    return overall, yesno_score, nlp_score

def interpret_score(score):
    if score <= 2:
        return "Very Low Risk", "Generally good mental health"
    elif score <= 4:
        return "Low Risk", "Minor concerns, maintain self-care"
    elif score <= 6:
        return "Moderate Risk", "Consider talking to someone you trust"
    else:
        return "High Risk", "Seek professional support immediately"

def provide_resources(risk_level):
    print("\n" + "="*60)
    print("RESOURCES AND RECOMMENDATIONS")
    print("="*60)
    
    if risk_level in ["Low Risk", "Very Low Risk"]:
        print("Maintain healthy habits, social connections, and mindfulness apps.")
    elif risk_level == "Moderate Risk":
        print("Talk to a counselor, trusted friend, or use Crisis Text Line: Text HOME to 741741")
    else:
        print("Seek immediate help: National Suicide Prevention Lifeline: 1-800-273-8255")

# -------------------------
# Process each row
# -------------------------
def process_row(row):
    yesno_ans = [1 if str(row.get(q, "No")).strip().lower() == "yes" else 0 for q in QUESTIONS['yesno']]
    rating_ans = [int(row.get(q,0)) if str(row.get(q,0)).isdigit() else 0 for q in QUESTIONS['rating']]
    text_ans = [str(row.get(q,"")) for q in QUESTIONS['text']]
    return yesno_ans, rating_ans, text_ans

# -------------------------
# CSV Runner
# -------------------------
def run_csv():
    risk_summary = []

    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"CSV file not found at {CSV_PATH}")
        return

    for idx, row in df.iterrows():
        name = row.get("Name", "Unknown")
        email = row.get("Username", "Unknown")
        yesno_ans, rating_ans, text_ans = process_row(row)

        print("\n" + "="*60)
        print(f"Processing {name} ({email})")
        print("-"*60)
        # Display question + answer mapping
        print("\nYes/No Questions and Answers:")
        for q, a in zip(QUESTIONS['yesno'], yesno_ans):
            print(f"Q: {q} --> A: {a}")
        print("\nRating Questions and Answers:")
        for q, a in zip(QUESTIONS['rating'], rating_ans):
            print(f"Q: {q} --> A: {a}")
        print("\nText Questions and Answers:")
        for q, a in zip(QUESTIONS['text'], text_ans):
            print(f"Q: {q} --> A: {a}")

        # Convert to arrays safely
        y_arr = np.array([yesno_ans])
        r_arr = np.array([rating_ans])
        t_arr = np.array([text_ans])

        # Calculate overall risk
        overall, mscore, nscore = calculate_overall_risk(y_arr, r_arr, t_arr)
        risk_level, interpretation = interpret_score(overall)

        print(f"\nOverall Risk Score: {overall:.2f}/10")
        print(f"Model Score: {mscore:.2f}, NLP Score: {nscore:.2f}")
        print(f"Risk Level: {risk_level}")
        print(f"Interpretation: {interpretation}")

        # Provide resources
        provide_resources(risk_level)
        print("="*60 + "\n")

        # Store for summary plot
        risk_summary.append((name, risk_level))

    # Plot graphical summary
    plot_risk_summary(risk_summary)

# -------------------------
# Graphical Summary
# -------------------------
def plot_risk_summary(risk_summary):
    if not risk_summary:
        print("No risk data to plot.")
        return

    counts = Counter([level for _, level in risk_summary])
    categories = ["Very Low Risk", "Low Risk", "Moderate Risk", "High Risk"]
    values = [counts.get(cat, 0) for cat in categories]

    plt.figure(figsize=(8,5))
    bars = plt.bar(categories, values, color=['green', 'lightgreen', 'orange', 'red'])
    plt.title("Risk Level Distribution of Participants")
    plt.ylabel("Number of Participants")
    plt.xlabel("Risk Levels")

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, str(value),
                 ha='center', va='bottom', fontsize=10)

    plt.ylim(0, max(values)+2)
    plt.show()

# -------------------------
# Main
# -------------------------
def main():
    run_csv()

if __name__ == "__main__":
    main()

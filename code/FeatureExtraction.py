# ✅ Install required libraries if in Colab
!pip install scikit-learn matplotlib seaborn pandas reportlab PyPDF2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, chi2
from sklearn.metrics.pairwise import cosine_similarity

from matplotlib.backends.backend_pdf import PdfPages
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from PyPDF2 import PdfMerger

import networkx as nx
from collections import Counter

# -------------------------
# Survey Questions
# -------------------------
questions = [
    "Have you ever had thoughts about ending your life?",
    "Do you feel like you're a burden to others?",
    "Have you ever seriously considered suicide in the past year?",
    "Do you ever think people would be better off without you?",
    "Have you been isolating yourself from friends or family lately?",
    "Do you feel comfortable talking to someone about your emotional struggles?",
    "Have you ever spoken to a counselor or therapist?",
    "Have you stopped doing things you once enjoyed?",
    "Have you recently had trouble focusing or staying motivated?",
    "Do you experience trouble sleeping or eating due to emotional stress?",
    "Rate how often you feel hopeless.",
    "Rate how often you feel emotionally exhausted.",
    "Rate your recent sleep quality.",
    "Rate your ability to enjoy daily activities.",
    "Rate your current level of stress.",
    "Rate your motivation to get out of bed in the morning.",
    "Rate how lonely you've felt in the past two weeks.",
    "Rate how often you feel overwhelmed by your responsibilities.",
    "Rate how safe you feel expressing your emotions to others.",
    "Rate how much support you feel from your friends/family.",
    "What do you usually think about when you feel sad or overwhelmed?",
    "How have your emotions been over the past two weeks? Please describe.",
    "Do you often feel empty, hopeless, or disconnected from others? Explain if you’d like.",
    "If your emotions were a weather report today, what would it be and why?",
    "Finish this sentence: 'Lately, I’ve been feeling...'",
    "What do you usually do when you’re feeling emotionally low?",
    "What kind of support would make you feel safer or more supported emotionally?",
    "What keeps you going on difficult days? What gives you hope?"
]

def main():
    # -------------------------
    # Open PDF for plots
    # -------------------------
    pdf_plots = PdfPages("all_plots.pdf")

    # -------------------------
    # 1. TF-IDF Scoring (Ranking for ALL 27 questions)
    # -------------------------
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(questions)
    scores = np.sum(tfidf_matrix.toarray(), axis=1)

    ranking = pd.DataFrame({
        "Q_No": [f"Q{i+1}" for i in range(len(questions))],
        "Question": questions,
        "TF-IDF Score": scores
    }).sort_values(by="TF-IDF Score", ascending=False)

    # Save full ranking to CSV
    ranking.to_csv("tfidf_ranking_all.csv", index=False)

    # ✅ Plot ALL 27 Questions
    plt.figure(figsize=(12,10))
    sns.barplot(
        y=ranking["Q_No"],
        x=ranking["TF-IDF Score"],
        palette="Blues_r"
    )
    plt.title("TF-IDF Ranking of All 27 Questions")
    plt.xlabel("TF-IDF Score")
    plt.ylabel("Question Number")
    plt.tight_layout()
    pdf_plots.savefig(); plt.close()

    # -------------------------
    # 2. TF-IDF + KMeans Clustering (still use top 10 for clarity)
    # -------------------------
    top_indices = ranking.index[:10]
    selected_tfidf = tfidf_matrix[top_indices]
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels = kmeans.fit_predict(selected_tfidf)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(selected_tfidf.toarray())

    plt.figure(figsize=(8,6))
    plt.scatter(reduced[:,0], reduced[:,1], c=labels, cmap="tab10", s=70)
    for i, idx in enumerate(top_indices):
        plt.text(reduced[i,0]+0.02, reduced[i,1]+0.02, f"Q{idx+1}", fontsize=9)
    plt.title("KMeans Clustering of Top 10 Questions (PCA Reduced)")
    pdf_plots.savefig(); plt.close()

    # -------------------------
    # 3. Correlation Heatmap
    # -------------------------
    vec = CountVectorizer().fit_transform(questions)
    similarity = cosine_similarity(vec)

    plt.figure(figsize=(10,8))
    sns.heatmap(similarity, cmap="viridis")
    plt.title("Question Similarity Heatmap")
    pdf_plots.savefig(); plt.close()

    # -------------------------
    # 4. Variance Threshold
    # -------------------------
    responses_df = pd.DataFrame(np.random.randint(0, 5, size=(100, len(questions))),
                                columns=[f"Q{i+1}" for i in range(len(questions))])
    selector = VarianceThreshold(threshold=0.1)
    selector.fit(responses_df)
    selected_var = responses_df.columns[selector.get_support()].tolist()[:10]

    variances = responses_df.var()
    top_var = variances.sort_values(ascending=False).head(10)
    plt.figure(figsize=(10,6))
    sns.barplot(x=top_var.values, y=top_var.index, palette="coolwarm")
    plt.title("Top 10 Questions by Variance in Responses")
    pdf_plots.savefig(); plt.close()

    # -------------------------
    # 5. Chi-Square
    # -------------------------
    y = np.random.randint(0, 2, size=responses_df.shape[0])  # fake labels
    chi_scores, _ = chi2(responses_df, y)
    top_indices_chi = np.argsort(chi_scores)[::-1][:10]
    selected_chi2 = [responses_df.columns[i] for i in top_indices_chi]

    chi_df = pd.DataFrame({
        "Question": responses_df.columns,
        "Chi2 Score": chi_scores
    }).sort_values(by="Chi2 Score", ascending=False).head(10)

    plt.figure(figsize=(10,6))
    sns.barplot(x="Chi2 Score", y="Question", data=chi_df, palette="magma")
    plt.title("Top 10 Questions by Chi-Square Score")
    pdf_plots.savefig(); plt.close()

    # -------------------------
    # 6. Combined Frequency
    # -------------------------
    all_methods = {
        "TF-IDF Score": ranking["Q_No"].tolist(),
        "Variance Threshold": selected_var,
        "Chi-Square": selected_chi2
    }

    combined = []
    for qs in all_methods.values():
        combined.extend(qs)
    freq = Counter(combined)
    freq_df = pd.DataFrame(freq.items(), columns=["Question", "Count"]).sort_values(by="Count", ascending=False)

    plt.figure(figsize=(12,8))
    sns.barplot(x="Count", y="Question", data=freq_df, palette="viridis")
    plt.title("Frequency of Selection Across Methods")
    pdf_plots.savefig(); plt.close()

    # 7. Bubble Chart
    bubble_df = pd.DataFrame(freq.items(), columns=["Question", "Count"])
    bubble_df["Index"] = bubble_df.index
    bubble_df["Score"] = bubble_df["Count"] * 200
    plt.figure(figsize=(12,8))
    plt.scatter(bubble_df["Index"], bubble_df["Count"],
                s=bubble_df["Score"], alpha=0.6, c=bubble_df["Count"], cmap="plasma")
    for i, q in enumerate(bubble_df["Question"]):
        plt.text(bubble_df["Index"][i], bubble_df["Count"][i]+0.1, q, fontsize=8, ha="center")
    plt.title("Bubble Chart: Frequency of Questions Across Methods")
    pdf_plots.savefig(); plt.close()

    # ✅ Close plots
    pdf_plots.close()

    # -------------------------
    # Save Text Report (full TF-IDF ranking table)
    # -------------------------
    doc = SimpleDocTemplate("all_texts.pdf")
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph("=== TF-IDF Ranking of All 27 Questions ===", styles["Heading1"]))
    elements.append(Spacer(1,12))
    elements.append(Paragraph(ranking.to_html(index=False), styles["Normal"]))
    elements.append(PageBreak())
    doc.build(elements)

    # -------------------------
    # Merge PDFs
    # -------------------------
    merger = PdfMerger()
    merger.append("all_texts.pdf")
    merger.append("all_plots.pdf")
    merger.write("final_report.pdf")
    merger.close()
     # Final Top 10 Selection
    # -------------------------
    final_top10 = freq_df.head(10)["Question"].tolist()
    print("\n🎯 Final Top 10 Selected Questions:")
    for q in final_top10:
        print(" -", q)

    # Save to CSV and JSON
    pd.DataFrame({"Final_Top10": final_top10}).to_csv("final_top10_questions.csv", index=False)

    import json
    with open("final_top10_questions.json", "w") as f:
        json.dump(final_top10, f, indent=4)

    print("\n✅ Final top 10 questions saved to:")
    print(" - final_top10_questions.csv")
    print(" - final_top10_questions.json")
    print(" - full report in final_report.pdf")


if __name__ == "__main__":
    main()

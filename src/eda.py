"""Exploratory Data Analysis (EDA) module for Stanford IMDB Movie Reviews.

Generates key statistical summaries and high-resolution figures in reports/figures/:
- eda_class_balance.png
- eda_length_distribution.png
- eda_top_words.png
- eda_token_length_cdf.png
"""

import re
import collections
from pathlib import Path
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.config import FIGURES_DIR, SEED
from src.data import load_raw_data
from src.preprocess import clean_text
from src.utils import setup_logger, set_seed

logger = setup_logger("eda")

# Set aesthetic styling for charts
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#CBD5E1"
plt.rcParams["axes.linewidth"] = 0.8

# Standard English stopwords list
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
    "will", "just", "don", "should", "now", "movie", "film", "one", "like", "br"
}


def compute_eda_statistics(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate key EDA metrics on movie reviews."""
    train_df = train_df.copy()
    train_df["word_count"] = train_df["text"].apply(lambda t: len(t.split()))
    
    pos_lengths = train_df[train_df["label"] == 1]["word_count"]
    neg_lengths = train_df[train_df["label"] == 0]["word_count"]
    
    over_256 = (train_df["word_count"] > 256).mean() * 100
    over_512 = (train_df["word_count"] > 512).mean() * 100

    # Build word frequencies after cleaning and stopword removal
    def get_top_words(subset_df: pd.DataFrame, n: int = 20) -> List[Tuple[str, int]]:
        words = []
        for text in subset_df["text"]:
            cleaned = clean_text(text, lower=True)
            tokens = re.findall(r"\b[a-z]{3,}\b", cleaned)
            words.extend([w for w in tokens if w not in STOPWORDS])
        return collections.Counter(words).most_common(n)

    top_pos = get_top_words(train_df[train_df["label"] == 1], n=20)
    top_neg = get_top_words(train_df[train_df["label"] == 0], n=20)

    # Calculate total vocabulary size across raw train split
    all_tokens = set()
    for text in train_df["text"]:
        all_tokens.update(clean_text(text, lower=True).split())
    vocab_size = len(all_tokens)

    stats = {
        "train_reviews_count": len(train_df),
        "test_reviews_count": len(test_df),
        "train_pos_share": float((train_df["label"] == 1).mean()),
        "train_neg_share": float((train_df["label"] == 0).mean()),
        "vocab_size": vocab_size,
        "mean_length_pos": float(pos_lengths.mean()),
        "median_length_pos": float(pos_lengths.median()),
        "mean_length_neg": float(neg_lengths.mean()),
        "median_length_neg": float(neg_lengths.median()),
        "pct_over_256_tokens": float(over_256),
        "pct_over_512_tokens": float(over_512),
        "top_pos_words": top_pos,
        "top_neg_words": top_neg,
    }
    return stats


def generate_eda_figures(train_df: pd.DataFrame, stats: Dict[str, Any]) -> None:
    """Generate and save publication-quality visual figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    train_df = train_df.copy()
    train_df["word_count"] = train_df["text"].apply(lambda t: len(t.split()))
    train_df["sentiment_label"] = train_df["label"].map({1: "Positive Movie Review", 0: "Negative Movie Review"})

    # 1. Class Balance Bar Chart
    plt.figure(figsize=(7, 4.5), dpi=300)
    ax = sns.countplot(
        data=train_df,
        x="sentiment_label",
        palette=["#3BA99C", "#E06D53"],
        hue="sentiment_label",
        legend=False,
    )
    plt.title("Class Balance in Stanford IMDB Training Dataset", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Review Sentiment Class", fontsize=11, labelpad=8)
    plt.ylabel("Number of Movie Reviews", fontsize=11, labelpad=8)
    for p in ax.patches:
        height = int(p.get_height())
        ax.annotate(f"{height:,} ({height/len(train_df)*100:.1f}%)",
                    (p.get_x() + p.get_width() / 2., height / 2),
                    ha="center", va="center", fontsize=11, color="white", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_class_balance.png")
    plt.close()
    logger.info("Saved eda_class_balance.png")

    # 2. Review Length Distribution by Class
    plt.figure(figsize=(9, 5), dpi=300)
    sns.histplot(
        data=train_df,
        x="word_count",
        hue="sentiment_label",
        element="step",
        kde=True,
        bins=60,
        palette=["#3BA99C", "#E06D53"],
        alpha=0.4,
    )
    plt.axvline(256, color="#1F2937", linestyle="--", linewidth=1.5, label="Max Seq Len = 256")
    plt.axvline(512, color="#6B7280", linestyle=":", linewidth=1.5, label="Token Cap = 512")
    plt.title("Movie Review Length Distribution (Word Count) by Sentiment", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Review Length (Words)", fontsize=11, labelpad=8)
    plt.ylabel("Review Frequency", fontsize=11, labelpad=8)
    plt.xlim(0, 1000)
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_length_distribution.png")
    plt.close()
    logger.info("Saved eda_length_distribution.png")

    # 3. Cumulative Distribution Function (CDF) for Token Length Justification
    plt.figure(figsize=(8, 4.8), dpi=300)
    sorted_lengths = np.sort(train_df["word_count"].values)
    cdf = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
    plt.plot(sorted_lengths, cdf * 100, color="#2563EB", linewidth=2.2, label="Cumulative Review Share (%)")
    
    p256 = (train_df["word_count"] <= 256).mean() * 100
    p512 = (train_df["word_count"] <= 512).mean() * 100
    
    plt.axvline(256, color="#DC2626", linestyle="--", linewidth=1.2)
    plt.axhline(p256, color="#DC2626", linestyle="--", linewidth=1.2)
    plt.plot([256], [p256], marker="o", markersize=7, color="#DC2626")
    plt.text(270, p256 - 7, f"256 tokens: {p256:.1f}% coverage", color="#DC2626", fontweight="bold", fontsize=10)

    plt.axvline(512, color="#D97706", linestyle=":", linewidth=1.2)
    plt.axhline(p512, color="#D97706", linestyle=":", linewidth=1.2)
    plt.plot([512], [p512], marker="s", markersize=7, color="#D97706")
    plt.text(525, p512 - 7, f"512 tokens: {p512:.1f}% coverage", color="#D97706", fontweight="bold", fontsize=10)

    plt.title("Cumulative Distribution Function (CDF) of Movie Review Word Length", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Review Length in Words", fontsize=11, labelpad=8)
    plt.ylabel("Reviews Covered (%)", fontsize=11, labelpad=8)
    plt.xlim(0, 1000)
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_token_length_cdf.png")
    plt.close()
    logger.info("Saved eda_token_length_cdf.png")

    # 4. Top Distinctive Frequent Words per Class
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), dpi=300)
    
    pos_words, pos_counts = zip(*stats["top_pos_words"][:15])
    neg_words, neg_counts = zip(*stats["top_neg_words"][:15])

    sns.barplot(x=list(pos_counts), y=list(pos_words), ax=ax1, color="#3BA99C")
    ax1.set_title("Top Words in Positive Movie Reviews", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Occurrences", fontsize=10)
    
    sns.barplot(x=list(neg_counts), y=list(neg_words), ax=ax2, color="#E06D53")
    ax2.set_title("Top Words in Negative Movie Reviews", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Occurrences", fontsize=10)

    plt.suptitle("Most Frequent Informative Words by Review Sentiment", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_top_words.png")
    plt.close()
    logger.info("Saved eda_top_words.png")


def run_eda() -> Dict[str, Any]:
    """Execute EDA workflow."""
    set_seed(SEED)
    train_df, test_df = load_raw_data()
    stats = compute_eda_statistics(train_df, test_df)
    generate_eda_figures(train_df, stats)
    logger.info("=== EDA Key Findings ===")
    logger.info(f"Total training reviews: {stats['train_reviews_count']} (Pos: {stats['train_pos_share']:.1%}, Neg: {stats['train_neg_share']:.1%})")
    logger.info(f"Vocabulary Size: {stats['vocab_size']:,} words")
    logger.info(f"Mean Length: Positive={stats['mean_length_pos']:.1f} words, Negative={stats['mean_length_neg']:.1f} words")
    logger.info(f"Reviews > 256 tokens: {stats['pct_over_256_tokens']:.1f}%")
    logger.info(f"Reviews > 512 tokens: {stats['pct_over_512_tokens']:.1f}%")
    return stats


if __name__ == "__main__":
    run_eda()

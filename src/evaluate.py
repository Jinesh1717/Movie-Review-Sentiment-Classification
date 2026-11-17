"""Evaluation and Error Analysis module for DL-06 Movie Review Sentiment Classification.

Runs the test set exactly once at the end.
Generates:
- reports/metrics.csv (Accuracy, Precision, Recall, F1, ROC-AUC, Inference Time per 1k reviews)
- reports/error_analysis.csv (Misclassified reviews with confidence and error patterns)
- reports/figures/:
    1. training_curves.png
    2. confusion_matrices_all.png
    3. confusion_matrix_top3.png
    4. roc_curves_all_models.png
    5. model_comparison_bar.png
    6. confidence_histogram_best_model.png
"""

import time
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import (
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    DEVICE,
    SEED,
    MAX_SEQ_LEN,
    DL_BATCH_SIZE,
    TRANSFORMER_BATCH_SIZE,
)
from src.data import load_raw_data
from src.preprocess import (
    clean_text,
    Vocabulary,
    IMDBSeqDataset,
    TransformerIMDBDataset,
)
from src.models import (
    BiLSTMClassifier,
    TextCNNClassifier,
)
from src.utils import setup_logger, set_seed

logger = setup_logger("evaluate")

# Aesthetics
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"


# ============================================================================
# Inference Helpers for Evaluation
# ============================================================================

def evaluate_classical(
    model_path: Path,
    texts: List[str],
    y_true: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Evaluate scikit-learn pipeline."""
    model = joblib.load(model_path)
    cleaned_texts = [clean_text(t, lower=True) for t in texts]
    
    t0 = time.time()
    probs = model.predict_proba(cleaned_texts)[:, 1]
    elapsed = time.time() - t0
    preds = (probs >= 0.5).astype(int)
    return preds, probs, elapsed


def evaluate_pytorch_seq(
    model_class,
    checkpoint_path: Path,
    texts: List[str],
    vocab: Vocabulary,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Evaluate BiLSTM or CNN PyTorch model."""
    model = model_class(vocab_size=len(vocab)).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    dataset = IMDBSeqDataset(texts, labels=None, vocab=vocab, max_len=MAX_SEQ_LEN)
    loader = DataLoader(dataset, batch_size=DL_BATCH_SIZE * 2, shuffle=False)

    all_probs = []
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            logits = model(input_ids)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
    elapsed = time.time() - t0
    
    all_probs = np.array(all_probs)
    preds = (all_probs >= 0.5).astype(int)
    return preds, all_probs, elapsed


def evaluate_distilbert(
    model_dir: Path,
    texts: List[str],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Evaluate DistilBERT transformer model."""
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(DEVICE)
    model.eval()

    dataset = TransformerIMDBDataset(texts, labels=None, tokenizer=tokenizer, max_len=MAX_SEQ_LEN)
    loader = DataLoader(dataset, batch_size=TRANSFORMER_BATCH_SIZE * 2, shuffle=False)

    all_probs = []
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
    elapsed = time.time() - t0

    all_probs = np.array(all_probs)
    preds = (all_probs >= 0.5).astype(int)
    return preds, all_probs, elapsed


# ============================================================================
# Main Evaluation Workflow
# ============================================================================

def run_evaluation() -> pd.DataFrame:
    """Run rigorous test set evaluation across all saved models."""
    set_seed(SEED)
    _, test_df = load_raw_data()
    
    # If on CPU, we test on a stratified test sample (e.g. 5,000) or full 25,000
    # To keep execution prompt and responsive on CPU, test on 5,000 stratified samples
    if DEVICE == "cpu" and len(test_df) > 1000:
        logger.warning("Running evaluation on CPU: using stratified test set of 1,000 movie reviews.")
        eval_df = test_df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(n=500, random_state=SEED)
        ).reset_index(drop=True)
    else:
        eval_df = test_df

    texts = eval_df["text"].tolist()
    y_true = eval_df["label"].values
    n_reviews = len(texts)
    logger.info(f"Evaluating {n_reviews:,} test movie reviews (50% positive, 50% negative)...")

    vocab_path = MODELS_DIR / "vocab.joblib"
    vocab = Vocabulary.load(vocab_path) if vocab_path.exists() else Vocabulary()

    model_registry = {
        "TF-IDF + Logistic Regression": ("classical", MODELS_DIR / "tfidf_logreg.joblib"),
        "TF-IDF + Linear SVM": ("classical", MODELS_DIR / "tfidf_svm.joblib"),
        "BiLSTM": ("bilstm", MODELS_DIR / "bilstm_best.pt"),
        "1D Text CNN": ("cnn", MODELS_DIR / "cnn_best.pt"),
        "DistilBERT": ("transformer", MODELS_DIR / "distilbert"),
    }

    metrics_rows = []
    predictions_dict = {}
    probabilities_dict = {}

    for name, (m_type, path) in model_registry.items():
        if not path.exists() or (path.is_dir() and not (path / "config.json").exists()):
            logger.warning(f"Skipping {name}: Checkpoint not found at {path}")
            continue

        logger.info(f"Evaluating {name}...")
        if m_type == "classical":
            preds, probs, elapsed = evaluate_classical(path, texts, y_true)
        elif m_type == "bilstm":
            preds, probs, elapsed = evaluate_pytorch_seq(BiLSTMClassifier, path, texts, vocab)
        elif m_type == "cnn":
            preds, probs, elapsed = evaluate_pytorch_seq(TextCNNClassifier, path, texts, vocab)
        elif m_type == "transformer":
            preds, probs, elapsed = evaluate_distilbert(path, texts)
        else:
            continue

        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        auc = roc_auc_score(y_true, probs)
        
        # Inference time per 1,000 reviews in seconds
        time_per_1k = (elapsed / n_reviews) * 1000

        predictions_dict[name] = preds
        probabilities_dict[name] = probs

        metrics_rows.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4),
            "Inference Time (s / 1k reviews)": round(time_per_1k, 3),
        })

    metrics_df = pd.DataFrame(metrics_rows).sort_values(by="F1-Score", ascending=False).reset_index(drop=True)
    metrics_csv_path = REPORTS_DIR / "metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    logger.info(f"\nSaved metrics to {metrics_csv_path}:\n{metrics_df.to_string(index=False)}")

    # ========================================================================
    # Plot Generation
    # ========================================================================
    generate_evaluation_plots(metrics_df, y_true, predictions_dict, probabilities_dict)

    # ========================================================================
    # Error Analysis on Best Performing Model
    # ========================================================================
    best_model_name = metrics_df.iloc[0]["Model"]
    logger.info(f"Best performing model on test set: {best_model_name}")
    run_error_analysis(best_model_name, eval_df, predictions_dict[best_model_name], probabilities_dict[best_model_name])

    return metrics_df


def generate_evaluation_plots(
    metrics_df: pd.DataFrame,
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    probabilities: Dict[str, np.ndarray],
) -> None:
    """Generate all required evaluation charts and figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Training & Validation Curves for Neural Models
    histories_path = MODELS_DIR / "training_histories.json"
    if histories_path.exists():
        with open(histories_path, "r", encoding="utf-8") as f:
            histories = json.load(f)

        neural_models = [m for m in ["BiLSTM", "1D Text CNN", "DistilBERT"] if m in histories]
        if neural_models:
            fig, axes = plt.subplots(len(neural_models), 2, figsize=(12, 3.5 * len(neural_models)), dpi=300)
            if len(neural_models) == 1:
                axes = np.expand_dims(axes, 0)
                
            for idx, m_name in enumerate(neural_models):
                hist = histories[m_name]
                epochs = range(1, len(hist["train_loss"]) + 1)
                
                # Loss curve
                ax_loss = axes[idx, 0]
                ax_loss.plot(epochs, hist["train_loss"], "o-", color="#DC2626", label="Train Loss")
                ax_loss.plot(epochs, hist["val_loss"], "s--", color="#2563EB", label="Val Loss")
                ax_loss.set_title(f"{m_name} - Loss Trajectory", fontsize=11, fontweight="bold")
                ax_loss.set_xlabel("Epoch", fontsize=10)
                ax_loss.set_ylabel("Loss", fontsize=10)
                ax_loss.legend()

                # Accuracy curve
                ax_acc = axes[idx, 1]
                ax_acc.plot(epochs, hist["train_acc"], "o-", color="#059669", label="Train Acc")
                ax_acc.plot(epochs, hist["val_acc"], "s--", color="#7C3AED", label="Val Acc")
                ax_acc.set_title(f"{m_name} - Accuracy Trajectory", fontsize=11, fontweight="bold")
                ax_acc.set_xlabel("Epoch", fontsize=10)
                ax_acc.set_ylabel("Accuracy", fontsize=10)
                ax_acc.legend()

            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "training_curves.png")
            plt.close()
            logger.info("Saved training_curves.png")

    # 2. Confusion Matrices for All Models
    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(4.2 * n_models, 3.8), dpi=300)
    if n_models == 1:
        axes = [axes]

    for ax, (m_name, preds) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_true, preds)
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        labels = [f"{v}\n({p:.1%})" for v, p in zip(cm.flatten(), cm_norm.flatten())]
        labels = np.asarray(labels).reshape(2, 2)

        sns.heatmap(
            cm,
            annot=labels,
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Neg", "Pos"],
            yticklabels=["Neg", "Pos"],
        )
        ax.set_title(f"{m_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)

    plt.suptitle("Confusion Matrices (Counts and Normalized) Across Evaluated Models", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrices_all.png")
    plt.close()
    logger.info("Saved confusion_matrices_all.png")

    # 3. Side-by-side Confusion Matrix for Top 3 Models
    top3_models = metrics_df["Model"].head(3).tolist()
    fig, axes = plt.subplots(1, len(top3_models), figsize=(4.5 * len(top3_models), 4), dpi=300)
    if len(top3_models) == 1:
        axes = [axes]
        
    for ax, m_name in zip(axes, top3_models):
        preds = predictions[m_name]
        cm = confusion_matrix(y_true, preds)
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        labels = [f"{v:,}\n({p:.1%})" for v, p in zip(cm.flatten(), cm_norm.flatten())]
        labels = np.asarray(labels).reshape(2, 2)

        sns.heatmap(
            cm,
            annot=labels,
            fmt="",
            cmap="GnBu",
            cbar=False,
            ax=ax,
            xticklabels=["Negative Review", "Positive Review"],
            yticklabels=["Negative Review", "Positive Review"],
        )
        ax.set_title(f"{m_name}", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Predicted Class", fontsize=10)
        ax.set_ylabel("True Class", fontsize=10)

    plt.suptitle("Top-3 Models Side-by-Side Confusion Matrix Comparison", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix_top3.png")
    plt.close()
    logger.info("Saved confusion_matrix_top3.png")

    # 4. Overlaid ROC Curves for All Models
    plt.figure(figsize=(8, 6), dpi=300)
    palette = sns.color_palette("tab10", n_colors=len(probabilities))

    for (m_name, probs), color in zip(probabilities.items(), palette):
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        plt.plot(fpr, tpr, color=color, linewidth=2.0, label=f"{m_name} (AUC = {auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Chance (AUC = 0.5000)")
    plt.title("Receiver Operating Characteristic (ROC) Curves on IMDB Test Set", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, labelpad=8)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, labelpad=8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right", frameon=True, facecolor="white", fontsize=9.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curves_all_models.png")
    plt.close()
    logger.info("Saved roc_curves_all_models.png")

    # 5. Model Comparison Bar Chart (Accuracy & F1 vs Inference Time)
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
    x = np.arange(len(metrics_df))
    width = 0.28

    rects1 = ax1.bar(x - width/2, metrics_df["Accuracy"], width, label="Accuracy", color="#3BA99C")
    rects2 = ax1.bar(x + width/2, metrics_df["F1-Score"], width, label="F1-Score", color="#2563EB")

    ax1.set_ylabel("Score (0 to 1)", fontsize=11, labelpad=8)
    ax1.set_ylim(0.70, 1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_df["Model"], rotation=15, ha="right", fontsize=10)
    ax1.legend(loc="upper left")

    # Secondary axis for inference latency
    ax2 = ax1.twinx()
    ax2.plot(x, metrics_df["Inference Time (s / 1k reviews)"], color="#DC2626", marker="o", linewidth=2.2, label="Latency (s / 1k reviews)")
    ax2.set_ylabel("Inference Time (seconds per 1,000 reviews)", color="#DC2626", fontsize=11, labelpad=8)
    ax2.tick_params(axis="y", labelcolor="#DC2626")
    ax2.grid(False)
    ax2.legend(loc="upper right")

    plt.title("Model Performance vs Inference Speed Trade-off", fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison_bar.png")
    plt.close()
    logger.info("Saved model_comparison_bar.png")

    # 6. Confidence Histogram for Best Model (Correct vs Incorrect)
    best_model_name = metrics_df.iloc[0]["Model"]
    best_probs = probabilities[best_model_name]
    best_preds = predictions[best_model_name]
    
    # Prediction confidence is max(p, 1-p)
    confidence = np.maximum(best_probs, 1.0 - best_probs)
    correct_mask = (best_preds == y_true)

    plt.figure(figsize=(8, 5), dpi=300)
    plt.hist(
        confidence[correct_mask],
        bins=30,
        alpha=0.65,
        color="#059669",
        label=f"Correct Predictions (n={correct_mask.sum():,})",
        density=True,
    )
    plt.hist(
        confidence[~correct_mask],
        bins=30,
        alpha=0.75,
        color="#DC2626",
        label=f"Misclassifications (n={(~correct_mask).sum():,})",
        density=True,
    )
    plt.title(f"Confidence Distribution ({best_model_name}) - Correct vs Incorrect", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Model Confidence (Posterior Probability of Predicted Class)", fontsize=11, labelpad=8)
    plt.ylabel("Density", fontsize=11, labelpad=8)
    plt.xlim(0.5, 1.0)
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confidence_histogram_best_model.png")
    plt.close()
    logger.info("Saved confidence_histogram_best_model.png")


# ============================================================================
# Step 5: Error Analysis
# ============================================================================

def classify_error_pattern(text: str, true_label: int, predicted_label: int) -> str:
    """Heuristic rule-based categorization of error failure modes."""
    lower = text.lower()
    word_count = len(text.split())

    # Sarcasm / irony patterns
    sarcasm_keywords = ["yeah right", "surely", "supposedly", "as if", "give me a break", "masterpiece of garbage", "laughable"]
    if any(k in lower for k in sarcasm_keywords):
        return "Sarcasm / Irony"

    # Mixed sentiment
    contrast_markers = ["great acting but", "good acting but", "terrible script", "worst movie ever made although", "despite the great", "loved the actors but hated"]
    if any(k in lower for k in contrast_markers) or ("but" in lower and ("good" in lower or "great" in lower) and ("bad" in lower or "boring" in lower)):
        return "Mixed Sentiment (Conflict)"

    # Negation
    negation_patterns = ["not good", "not great", "hardly", "never recommend", "scarcely", "not worth", "not a single"]
    if any(k in lower for k in negation_patterns):
        return "Complex Negation"

    # Extreme length
    if word_count > 450:
        return "Very Long Review (Context Dilution)"

    # Plot summary heavy
    plot_terms = ["character", "plot", "story", "director", "scene", "played by", "begins when"]
    if sum(1 for term in plot_terms if term in lower) >= 4:
        return "Plot Summary Heavy (Objective Tone)"

    return "Subtle / Boundary Ambiguity"


def run_error_analysis(
    model_name: str,
    df: pd.DataFrame,
    preds: np.ndarray,
    probs: np.ndarray,
) -> pd.DataFrame:
    """Extract and analyze misclassified movie reviews."""
    y_true = df["label"].values
    misclassified_idx = np.where(preds != y_true)[0]

    logger.info(f"Analyzing {len(misclassified_idx)} misclassified reviews out of {len(df)} ({len(misclassified_idx)/len(df)*100:.2f}% error rate)...")

    errors_list = []
    for idx in misclassified_idx:
        text = df.iloc[idx]["text"]
        true_lbl = int(y_true[idx])
        pred_lbl = int(preds[idx])
        pos_prob = float(probs[idx])
        conf = pos_prob if pred_lbl == 1 else (1.0 - pos_prob)
        
        pattern = classify_error_pattern(text, true_lbl, pred_lbl)
        
        errors_list.append({
            "review_id": idx,
            "text": text,
            "true_label": true_lbl,
            "true_sentiment": "Positive" if true_lbl == 1 else "Negative",
            "predicted_label": pred_lbl,
            "predicted_sentiment": "Positive" if pred_lbl == 1 else "Negative",
            "positive_probability": round(pos_prob, 4),
            "confidence": round(conf, 4),
            "word_count": len(text.split()),
            "error_pattern": pattern,
        })

    error_df = pd.DataFrame(errors_list)
    error_csv_path = REPORTS_DIR / "error_analysis.csv"
    error_df.to_csv(error_csv_path, index=False)
    logger.info(f"Saved error analysis report to {error_csv_path}")

    # Summary of patterns
    pattern_counts = error_df["error_pattern"].value_counts()
    logger.info(f"Error Pattern Breakdown:\n{pattern_counts.to_string()}")

    return error_df


if __name__ == "__main__":
    run_evaluation()

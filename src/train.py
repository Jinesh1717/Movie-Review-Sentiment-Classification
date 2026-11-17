"""Training orchestration module for DL-06 Movie Review Sentiment Classification.

Trains, tunes, and saves checkpoints for:
1. Baseline: TF-IDF + Logistic Regression
2. Classical alternative: TF-IDF + Calibrated Linear SVM
3. Deep learning: Bidirectional LSTM
4. Deep learning: 1D Text CNN
5. Transformer: DistilBERT Sequence Classifier

Rules:
- Validation split is taken strictly from TRAIN set only.
- Test set is untouched during training.
- Early stopping on validation loss.
- Parameter count and training duration logged for every model.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.config import (
    MODELS_DIR,
    SEED,
    DEVICE,
    DL_BATCH_SIZE,
    DL_LEARNING_RATE,
    DL_WEIGHT_DECAY,
    DL_EPOCHS,
    DL_PATIENCE,
    TRANSFORMER_BATCH_SIZE,
    TRANSFORMER_EPOCHS,
    TRANSFORMER_LR,
    TRANSFORMER_WARMUP_RATIO,
    TRANSFORMER_WEIGHT_DECAY,
    CPU_TRANSFORMER_SUBSET_SIZE,
)
from src.data import get_dataset_splits
from src.preprocess import (
    clean_text,
    Vocabulary,
    IMDBSeqDataset,
    TransformerIMDBDataset,
)
from src.models import (
    build_tfidf_logreg,
    build_tfidf_svm,
    BiLSTMClassifier,
    TextCNNClassifier,
    build_distilbert_model,
    count_parameters,
)
from src.utils import setup_logger, set_seed

logger = setup_logger("train")


# ============================================================================
# Classical Models Training
# ============================================================================

def train_classical_model(
    model_name: str,
    pipeline_builder,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    save_filename: str,
) -> Dict[str, Any]:
    """Train and evaluate a classical scikit-learn pipeline on movie reviews."""
    logger.info(f"--- Training {model_name} ---")
    t0 = time.time()
    
    # Preprocess text (clean HTML and lowercase)
    X_train = [clean_text(t, lower=True) for t in train_df["text"]]
    y_train = train_df["label"].values
    
    X_val = [clean_text(t, lower=True) for t in val_df["text"]]
    y_val = val_df["label"].values
    
    pipeline = pipeline_builder()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0
    
    val_acc = float(pipeline.score(X_val, y_val))
    save_path = MODELS_DIR / save_filename
    joblib.dump(pipeline, save_path)
    logger.info(f"{model_name} trained in {train_time:.2f}s | Val Acc: {val_acc:.4f} | Saved to {save_path}")
    
    # Estimate parameter count: features * classes
    vectorizer = pipeline.named_steps["tfidf"]
    num_features = len(vectorizer.vocabulary_) if hasattr(vectorizer, "vocabulary_") else 0
    param_count = num_features  # Linear weights
    
    return {
        "model_name": model_name,
        "train_time_sec": round(train_time, 2),
        "val_accuracy": round(val_acc, 4),
        "parameters": param_count,
        "checkpoint_path": str(save_path),
    }


# ============================================================================
# Deep Learning PyTorch Models Training (BiLSTM & CNN)
# ============================================================================

def train_pytorch_seq_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    save_filename: str,
    epochs: int = DL_EPOCHS,
    lr: float = DL_LEARNING_RATE,
    weight_decay: float = DL_WEIGHT_DECAY,
    patience: int = DL_PATIENCE,
) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
    """Train BiLSTM or Text CNN with Early Stopping on validation loss."""
    logger.info(f"--- Training {model_name} on {DEVICE} ---")
    model = model.to(DEVICE)
    param_count = count_parameters(model)
    logger.info(f"{model_name} trainable parameters: {param_count:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    save_path = MODELS_DIR / save_filename
    
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for b_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += len(labels)

            if (b_idx + 1) % 50 == 0 or (b_idx + 1) == len(train_loader):
                logger.info(f"[{model_name}] Epoch {epoch}/{epochs} | Batch {b_idx+1}/{len(train_loader)} | Running Loss: {total_loss/total:.4f}")

        train_loss = total_loss / total
        train_acc = correct / total

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                logits = model(input_ids)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                preds = (torch.sigmoid(logits) >= 0.5).long()
                val_correct += (preds == labels.long()).sum().item()
                val_total += len(labels)

        val_loss /= val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_acc"].append(round(val_acc, 4))

        logger.info(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            logger.info(f"--> Saved best checkpoint: {save_path} (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break

    train_time = time.time() - t0
    summary = {
        "model_name": model_name,
        "train_time_sec": round(train_time, 2),
        "val_accuracy": round(best_val_acc, 4),
        "val_loss": round(best_val_loss, 4),
        "parameters": param_count,
        "checkpoint_path": str(save_path),
    }
    return summary, history


# ============================================================================
# Transformer Model Training (DistilBERT)
# ============================================================================

def train_distilbert_classifier(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
    """Fine-tune DistilBERT sequence classifier for movie review sentiment."""
    logger.info("--- Fine-Tuning DistilBERT ---")
    model, tokenizer = build_distilbert_model()
    model = model.to(DEVICE)
    param_count = count_parameters(model)
    logger.info(f"DistilBERT parameters: {param_count:,}")

    # Subsample for CPU execution if running without GPU to prevent indefinite hangs
    if DEVICE == "cpu" and len(train_df) > CPU_TRANSFORMER_SUBSET_SIZE:
        logger.warning(
            f"Running on CPU without hardware accelerator. Subsampling training data to {CPU_TRANSFORMER_SUBSET_SIZE} "
            f"reviews for CPU feasibility as specified in project requirements."
        )
        train_subset = train_df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(n=CPU_TRANSFORMER_SUBSET_SIZE // 2, random_state=SEED)
        ).reset_index(drop=True)
        val_subset = val_df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(n=min(500, len(val_df) // 2), random_state=SEED)
        ).reset_index(drop=True)
        epochs = 2
    else:
        train_subset = train_df
        val_subset = val_df
        epochs = TRANSFORMER_EPOCHS

    # Prepare datasets & loaders
    train_dataset = TransformerIMDBDataset(
        texts=train_subset["text"].tolist(),
        labels=train_subset["label"].tolist(),
        tokenizer=tokenizer,
    )
    val_dataset = TransformerIMDBDataset(
        texts=val_subset["text"].tolist(),
        labels=val_subset["label"].tolist(),
        tokenizer=tokenizer,
    )

    train_loader = DataLoader(train_dataset, batch_size=TRANSFORMER_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TRANSFORMER_BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRANSFORMER_LR,
        weight_decay=TRANSFORMER_WEIGHT_DECAY,
    )
    
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * TRANSFORMER_WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_val_acc = 0.0
    save_dir = MODELS_DIR / "distilbert"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for b_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

            if (b_idx + 1) % 40 == 0 or (b_idx + 1) == len(train_loader):
                logger.info(f"[DistilBERT] Epoch {epoch}/{epochs} | Step {b_idx+1}/{len(train_loader)} | Running Loss: {total_loss/total:.4f}")

        train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item() * len(labels)
                preds = torch.argmax(outputs.logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        val_loss /= val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_acc"].append(round(val_acc, 4))

        logger.info(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            logger.info(f"--> Saved best DistilBERT model to {save_dir}")

    train_time = time.time() - t0
    summary = {
        "model_name": "DistilBERT",
        "train_time_sec": round(train_time, 2),
        "val_accuracy": round(best_val_acc, 4),
        "val_loss": round(best_val_loss, 4),
        "parameters": param_count,
        "checkpoint_path": str(save_dir),
        "cpu_subsampled": (DEVICE == "cpu" and len(train_df) > CPU_TRANSFORMER_SUBSET_SIZE),
        "train_samples_used": len(train_subset),
    }
    return summary, history


# ============================================================================
# Main Training Orchestration
# ============================================================================

def train_all_models() -> Dict[str, Any]:
    """Train all 5 models and persist models, histories, and metadata."""
    set_seed(SEED)
    train_df, val_df, _ = get_dataset_splits()
    
    logger.info(f"Training split size: {len(train_df):,} | Validation split size: {len(val_df):,}")
    summaries: Dict[str, Any] = {}
    histories: Dict[str, Any] = {}

    # 1. Baseline: TF-IDF + Logistic Regression
    summaries["TF-IDF + Logistic Regression"] = train_classical_model(
        model_name="TF-IDF + Logistic Regression",
        pipeline_builder=build_tfidf_logreg,
        train_df=train_df,
        val_df=val_df,
        save_filename="tfidf_logreg.joblib",
    )

    # 2. Classical Alternative: TF-IDF + Linear SVM
    summaries["TF-IDF + Linear SVM"] = train_classical_model(
        model_name="TF-IDF + Linear SVM",
        pipeline_builder=build_tfidf_svm,
        train_df=train_df,
        val_df=val_df,
        save_filename="tfidf_svm.joblib",
    )

    # Build and save Vocabulary for Deep Learning Sequence Models (BiLSTM & CNN)
    vocab = Vocabulary()
    cleaned_train_texts = [clean_text(t, lower=True) for t in train_df["text"]]
    vocab.build_vocab(cleaned_train_texts)
    vocab.save(MODELS_DIR / "vocab.joblib")

    # DataLoader for BiLSTM & CNN
    train_seq_ds = IMDBSeqDataset(train_df["text"].tolist(), train_df["label"].tolist(), vocab)
    val_seq_ds = IMDBSeqDataset(val_df["text"].tolist(), val_df["label"].tolist(), vocab)
    train_seq_loader = DataLoader(train_seq_ds, batch_size=DL_BATCH_SIZE, shuffle=True)
    val_seq_loader = DataLoader(val_seq_ds, batch_size=DL_BATCH_SIZE, shuffle=False)

    # 3. Deep Learning: BiLSTM
    bilstm_model = BiLSTMClassifier(vocab_size=len(vocab))
    bilstm_summary, bilstm_hist = train_pytorch_seq_model(
        model_name="BiLSTM",
        model=bilstm_model,
        train_loader=train_seq_loader,
        val_loader=val_seq_loader,
        save_filename="bilstm_best.pt",
        epochs=DL_EPOCHS,
    )
    summaries["BiLSTM"] = bilstm_summary
    histories["BiLSTM"] = bilstm_hist

    # 4. Deep Learning: 1D Text CNN
    cnn_model = TextCNNClassifier(vocab_size=len(vocab))
    cnn_summary, cnn_hist = train_pytorch_seq_model(
        model_name="1D Text CNN",
        model=cnn_model,
        train_loader=train_seq_loader,
        val_loader=val_seq_loader,
        save_filename="cnn_best.pt",
        epochs=DL_EPOCHS,
    )
    summaries["1D Text CNN"] = cnn_summary
    histories["1D Text CNN"] = cnn_hist

    # 5. Transformer: DistilBERT
    distilbert_summary, distilbert_hist = train_distilbert_classifier(
        train_df=train_df,
        val_df=val_df,
    )
    summaries["DistilBERT"] = distilbert_summary
    histories["DistilBERT"] = distilbert_hist

    # Save summary and neural training histories
    with open(MODELS_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    with open(MODELS_DIR / "training_histories.json", "w", encoding="utf-8") as f:
        json.dump(histories, f, indent=2)

    logger.info("=== All models successfully trained and artifacts saved to models/ ===")
    return summaries


if __name__ == "__main__":
    train_all_models()

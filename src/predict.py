"""Inference API and CLI for DL-06 Movie Review Sentiment Classification.

Provides:
1. predict(texts) -> list of dicts {label, positive_probability, confidence, model_used}
   Uses DistilBERT by default, with automatic fallback to TF-IDF if checkpoint is absent.
2. CLI:
   python -m src.predict --text "Great movie!"
   python -m src.predict --input reviews.csv --output scored.csv
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import numpy as np
import pandas as pd
import joblib
import torch

from src.config import (
    MODELS_DIR,
    DEVICE,
    MAX_SEQ_LEN,
    DL_BATCH_SIZE,
    TRANSFORMER_BATCH_SIZE,
)
from src.preprocess import clean_text, Vocabulary, IMDBSeqDataset
from src.models import BiLSTMClassifier, TextCNNClassifier
from src.utils import setup_logger

logger = setup_logger("predict")

# Global cached model instances for fast repeated inference
_CACHED_MODELS: Dict[str, Any] = {}


def load_inference_model(preferred_model: str = "auto") -> Tuple[str, Any, Any]:
    """Load and cache the best available model for inference.
    
    Order of preference for 'auto':
    1. DistilBERT (Fine-tuned transformer)
    2. TF-IDF + Logistic Regression
    3. TF-IDF + Linear SVM
    4. BiLSTM
    5. 1D Text CNN
    """
    global _CACHED_MODELS

    distilbert_path = MODELS_DIR / "distilbert"
    tfidf_logreg_path = MODELS_DIR / "tfidf_logreg.joblib"
    tfidf_svm_path = MODELS_DIR / "tfidf_svm.joblib"
    bilstm_path = MODELS_DIR / "bilstm_best.pt"
    cnn_path = MODELS_DIR / "cnn_best.pt"
    vocab_path = MODELS_DIR / "vocab.joblib"

    target = preferred_model.lower()

    # DistilBERT
    has_valid_distilbert = distilbert_path.exists() and (distilbert_path / "config.json").exists()
    if (target in ["auto", "distilbert"]) and has_valid_distilbert:
        if "distilbert" not in _CACHED_MODELS:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            logger.info(f"Loading DistilBERT from {distilbert_path} on {DEVICE}...")
            tokenizer = AutoTokenizer.from_pretrained(str(distilbert_path))
            model = AutoModelForSequenceClassification.from_pretrained(str(distilbert_path)).to(DEVICE)
            model.eval()
            _CACHED_MODELS["distilbert"] = (model, tokenizer)
        m, tok = _CACHED_MODELS["distilbert"]
        return "DistilBERT", m, tok

    # TF-IDF + Logistic Regression
    if (target in ["auto", "tfidf", "tfidf_logreg", "logreg"]) and tfidf_logreg_path.exists():
        if "tfidf_logreg" not in _CACHED_MODELS:
            logger.info(f"Loading TF-IDF Logistic Regression from {tfidf_logreg_path}...")
            _CACHED_MODELS["tfidf_logreg"] = joblib.load(tfidf_logreg_path)
        return "TF-IDF + Logistic Regression", _CACHED_MODELS["tfidf_logreg"], None

    # TF-IDF + SVM
    if (target in ["auto", "tfidf_svm", "svm"]) and tfidf_svm_path.exists():
        if "tfidf_svm" not in _CACHED_MODELS:
            logger.info(f"Loading TF-IDF Linear SVM from {tfidf_svm_path}...")
            _CACHED_MODELS["tfidf_svm"] = joblib.load(tfidf_svm_path)
        return "TF-IDF + Linear SVM", _CACHED_MODELS["tfidf_svm"], None

    # BiLSTM
    if (target in ["auto", "bilstm"]) and bilstm_path.exists() and vocab_path.exists():
        if "bilstm" not in _CACHED_MODELS:
            vocab = Vocabulary.load(vocab_path)
            model = BiLSTMClassifier(vocab_size=len(vocab)).to(DEVICE)
            model.load_state_dict(torch.load(bilstm_path, map_location=DEVICE))
            model.eval()
            _CACHED_MODELS["bilstm"] = (model, vocab)
        m, v = _CACHED_MODELS["bilstm"]
        return "BiLSTM", m, v

    # Text CNN
    if (target in ["auto", "cnn", "1d_cnn"]) and cnn_path.exists() and vocab_path.exists():
        if "cnn" not in _CACHED_MODELS:
            vocab = Vocabulary.load(vocab_path)
            model = TextCNNClassifier(vocab_size=len(vocab)).to(DEVICE)
            model.load_state_dict(torch.load(cnn_path, map_location=DEVICE))
            model.eval()
            _CACHED_MODELS["cnn"] = (model, vocab)
        m, v = _CACHED_MODELS["cnn"]
        return "1D Text CNN", m, v

    raise FileNotFoundError(
        "No trained movie sentiment model found in models/! "
        "Please run 'python -m src.train' or 'make train' to train models first."
    )


def predict(
    texts: Union[str, List[str]],
    model_name: str = "auto",
) -> List[Dict[str, Any]]:
    """Predict sentiment for one or more movie reviews.
    
    Parameters
    ----------
    texts : Union[str, List[str]]
        A single movie review string or a list of movie review strings.
    model_name : str
        Model choice ('auto', 'distilbert', 'tfidf_logreg', 'tfidf_svm', 'bilstm', 'cnn').
        
    Returns
    -------
    List[Dict[str, Any]]
        List of dicts containing:
        - text: original text
        - label: 'Positive' or 'Negative'
        - label_id: 1 or 0
        - positive_probability: float [0, 1]
        - confidence: float [0.5, 1.0]
        - model_used: name of the model executed
    """
    if isinstance(texts, str):
        texts = [texts]

    model_type, model, auxiliary = load_inference_model(preferred_model=model_name)
    results: List[Dict[str, Any]] = []

    # Filter and validate texts
    valid_indices: List[int] = []
    cleaned_texts: List[str] = []

    for i, raw_text in enumerate(texts):
        if not isinstance(raw_text, str) or not raw_text.strip():
            # Handle empty review gracefully
            results.append({
                "text": raw_text if isinstance(raw_text, str) else "",
                "label": "Neutral / Inconclusive",
                "label_id": -1,
                "positive_probability": 0.5,
                "confidence": 0.5,
                "model_used": model_type,
                "warning": "Empty or whitespace-only movie review provided.",
            })
            continue

        # Check for very long review and notify
        words = raw_text.split()
        if len(words) > 512:
            logger.warning(
                f"Review at index {i} has {len(words)} words, exceeding standard 512 token boundary. "
                "Truncation will be applied."
            )

        valid_indices.append(i)
        cleaned_texts.append(raw_text)

    if not valid_indices:
        return results

    # Run inference for valid reviews
    probs = np.zeros(len(cleaned_texts), dtype=float)

    if model_type == "DistilBERT":
        tokenizer = auxiliary
        encodings = tokenizer(
            [clean_text(t, lower=False) for t in cleaned_texts],
            truncation=True,
            padding=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        )
        with torch.no_grad():
            input_ids = encodings["input_ids"].to(DEVICE)
            attention_mask = encodings["attention_mask"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            p = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            probs = p

    elif "TF-IDF" in model_type:
        processed = [clean_text(t, lower=True) for t in cleaned_texts]
        probs = model.predict_proba(processed)[:, 1]

    elif model_type in ["BiLSTM", "1D Text CNN"]:
        vocab = auxiliary
        dataset = IMDBSeqDataset(cleaned_texts, labels=None, vocab=vocab, max_len=MAX_SEQ_LEN)
        loader = torch.utils.data.DataLoader(dataset, batch_size=DL_BATCH_SIZE, shuffle=False)
        all_p = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(DEVICE)
                logits = model(input_ids)
                p = torch.sigmoid(logits).cpu().numpy()
                all_p.extend(p)
        probs = np.array(all_p)

    # Build response objects
    output_dict: Dict[int, Dict[str, Any]] = {}
    for idx, raw_idx in enumerate(valid_indices):
        p_pos = float(probs[idx])
        lbl_id = 1 if p_pos >= 0.5 else 0
        lbl_str = "Positive" if lbl_id == 1 else "Negative"
        conf = p_pos if lbl_id == 1 else (1.0 - p_pos)

        item = {
            "text": cleaned_texts[idx],
            "label": lbl_str,
            "label_id": lbl_id,
            "positive_probability": round(p_pos, 4),
            "confidence": round(conf, 4),
            "model_used": model_type,
        }
        output_dict[raw_idx] = item

    # Merge in original order
    final_results = []
    valid_ptr = 0
    empty_ptr = 0
    for i in range(len(texts)):
        if i in output_dict:
            final_results.append(output_dict[i])
        else:
            final_results.append(results[empty_ptr])
            empty_ptr += 1

    return final_results


# ============================================================================
# CLI Interface
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict sentiment (Positive/Negative) for movie reviews using DL-06 trained models."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Single movie review string to score.")
    group.add_argument("--input", type=str, help="Input CSV file path containing movie reviews.")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path for batch predictions.")
    parser.add_argument("--model", type=str, default="auto", help="Model to use: auto, distilbert, tfidf_logreg, tfidf_svm, bilstm, cnn.")
    parser.add_argument("--column", type=str, default="review", help="Column name containing review text (for CSV input).")

    args = parser.parse_args()

    if args.text:
        pred = predict(args.text, model_name=args.model)[0]
        print("\n" + "=" * 50)
        print(f"Movie Review: {pred['text']}")
        print(f"Sentiment:    {pred['label']}")
        print(f"Confidence:   {pred['confidence']:.2%}")
        print(f"Positive Pr:  {pred['positive_probability']:.4f}")
        print(f"Model Used:   {pred['model_used']}")
        print("=" * 50 + "\n")

    elif args.input:
        in_path = Path(args.input)
        if not in_path.exists():
            logger.error(f"Input file not found: {in_path}")
            sys.exit(1)

        df = pd.read_csv(in_path)
        col = args.column if args.column in df.columns else ("text" if "text" in df.columns else df.columns[0])
        logger.info(f"Scoring {len(df)} movie reviews from column '{col}'...")

        preds = predict(df[col].tolist(), model_name=args.model)
        df["sentiment"] = [p["label"] for p in preds]
        df["positive_probability"] = [p["positive_probability"] for p in preds]
        df["confidence"] = [p["confidence"] for p in preds]
        df["model_used"] = [p["model_used"] for p in preds]

        out_path = Path(args.output) if args.output else in_path.parent / f"scored_{in_path.name}"
        df.to_csv(out_path, index=False)
        logger.info(f"Scored {len(df)} reviews successfully. Saved output to {out_path}")


if __name__ == "__main__":
    main()

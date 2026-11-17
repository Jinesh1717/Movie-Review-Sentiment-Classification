"""Model definitions for DL-06 Movie Review Sentiment Classification.

Includes:
1. Baseline: TF-IDF + Logistic Regression
2. Classical alternative: TF-IDF + Linear SVM (Calibrated for probabilities)
3. Deep learning: Bidirectional LSTM with Embedding, Spatial Dropout, & Dense Head
4. Deep learning: 1D Text CNN with Multi-Scale Convolutions & Max Pooling
5. Transformer: DistilBERT Sequence Classifier
"""

from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import (
    VOCAB_SIZE,
    EMBEDDING_DIM,
    LSTM_HIDDEN_DIM,
    LSTM_NUM_LAYERS,
    LSTM_DROPOUT,
    CNN_NUM_FILTERS,
    CNN_FILTER_SIZES,
    CNN_DROPOUT,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    LOGREG_C,
    SVM_C,
    TRANSFORMER_MODEL_NAME,
    SEED,
)


# ============================================================================
# Classical Models (scikit-learn pipelines)
# ============================================================================

def build_tfidf_logreg() -> Pipeline:
    """Build TF-IDF (unigrams + bigrams) + Logistic Regression pipeline."""
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=TFIDF_NGRAM_RANGE,
                max_features=TFIDF_MAX_FEATURES,
                sublinear_tf=True,
            ),
        ),
        (
            "clf",
            LogisticRegression(
                C=LOGREG_C,
                max_iter=1000,
                random_state=SEED,
                solver="liblinear",
            ),
        ),
    ])


def build_tfidf_svm() -> Pipeline:
    """Build TF-IDF + Calibrated Linear SVM pipeline.
    
    Calibrated with Platt scaling to provide reliable posterior probabilities and ROC-AUC.
    """
    base_svm = LinearSVC(C=SVM_C, random_state=SEED, max_iter=2000, dual="auto")
    calibrated_svm = CalibratedClassifierCV(estimator=base_svm, cv=3)
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=TFIDF_NGRAM_RANGE,
                max_features=TFIDF_MAX_FEATURES,
                sublinear_tf=True,
            ),
        ),
        ("clf", calibrated_svm),
    ])


# ============================================================================
# Deep Learning PyTorch Models
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM sentiment classifier for movie reviews."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embedding_dim: int = EMBEDDING_DIM,
        hidden_dim: int = LSTM_HIDDEN_DIM,
        num_layers: int = LSTM_NUM_LAYERS,
        dropout: float = LSTM_DROPOUT,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(input_ids))  # [batch_size, seq_len, emb_dim]
        lstm_out, (hn, cn) = self.lstm(embedded)
        
        # Concatenate final forward and backward hidden states
        # hn shape: [num_layers * 2, batch_size, hidden_dim]
        last_hidden = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)  # [batch_size, hidden_dim * 2]
        
        dense = F.relu(self.fc1(self.dropout(last_hidden)))
        logits = self.fc2(self.dropout(dense))  # [batch_size, 1]
        return logits.squeeze(1)


class TextCNNClassifier(nn.Module):
    """1D Multi-Kernel Convolutional Neural Network for text classification."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embedding_dim: int = EMBEDDING_DIM,
        num_filters: int = CNN_NUM_FILTERS,
        filter_sizes: Optional[List[int]] = None,
        dropout: float = CNN_DROPOUT,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        if filter_sizes is None:
            filter_sizes = CNN_FILTER_SIZES
            
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k,
            )
            for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(num_filters * len(filter_sizes), 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [batch_size, seq_len]
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, emb_dim]
        embedded = embedded.permute(0, 2, 1)  # [batch_size, emb_dim, seq_len] for Conv1d
        
        pooled_outputs = []
        for conv in self.convs:
            c = F.relu(conv(embedded))  # [batch_size, num_filters, L_out]
            p = F.adaptive_max_pool1d(c, 1).squeeze(2)  # [batch_size, num_filters]
            pooled_outputs.append(p)
            
        cat = torch.cat(pooled_outputs, dim=1)  # [batch_size, num_filters * len(filter_sizes)]
        dense = F.relu(self.fc1(self.dropout(cat)))
        logits = self.fc2(self.dropout(dense))  # [batch_size, 1]
        return logits.squeeze(1)


# ============================================================================
# Transformer Model (DistilBERT)
# ============================================================================

def build_distilbert_model(model_name: str = TRANSFORMER_MODEL_NAME) -> Tuple[Any, Any]:
    """Load pretrained DistilBERT sequence classification model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )
    return model, tokenizer

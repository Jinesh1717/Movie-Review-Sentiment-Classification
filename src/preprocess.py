"""Text cleaning, tokenization, vocabulary building, and PyTorch dataset definitions.

Rule: Keep preprocessing identical between training and inference.
Rule: Tokenizer/vocab for LSTM and CNN must be built from the training split only.
"""

import re
import html
import collections
from typing import List, Dict, Optional, Tuple, Union
import joblib
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import (
    MAX_SEQ_LEN,
    VOCAB_SIZE,
    PAD_TOKEN,
    OOV_TOKEN,
    MODELS_DIR,
)
from src.utils import setup_logger

logger = setup_logger("preprocess")

# Compiled regex patterns for efficient cleaning
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")
TOKEN_PATTERN = re.compile(r"\b\w+\b|[^\w\s]")


def clean_text(text: str, lower: bool = True) -> str:
    """Clean a raw movie review string.
    
    1. Unescape HTML entities (e.g., &quot; -> ", &amp; -> &)
    2. Strip HTML tags (e.g., <br />, <p>, etc.)
    3. Normalize whitespace (replaces tabs, newlines, multiple spaces with a single space)
    4. Optionally lowercase (True for classical/LSTM/CNN; False for transformer tokenizer)
    
    Used consistently across training, evaluation, and production inference.
    """
    if not isinstance(text, str):
        return ""
    
    # Unescape HTML entities
    cleaned = html.unescape(text)
    # Strip HTML tags
    cleaned = HTML_TAG_PATTERN.sub(" ", cleaned)
    # Normalize multiple whitespace characters
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    
    if lower:
        cleaned = cleaned.lower()
        
    return cleaned


class Vocabulary:
    """Vocabulary mapping words to integer IDs for deep learning models (LSTM & CNN).
    
    Built exclusively from the training split to prevent data leakage.
    Index 0: <pad>
    Index 1: <unk>
    """

    def __init__(
        self,
        max_size: int = VOCAB_SIZE,
        pad_token: str = PAD_TOKEN,
        unk_token: str = OOV_TOKEN,
    ) -> None:
        self.max_size = max_size
        self.pad_token = pad_token
        self.unk_token = unk_token
        
        self.token2idx: Dict[str, int] = {self.pad_token: 0, self.unk_token: 1}
        self.idx2token: Dict[int, str] = {0: self.pad_token, 1: self.unk_token}
        self.pad_idx: int = 0
        self.unk_idx: int = 1

    def build_vocab(self, texts: List[str]) -> "Vocabulary":
        """Build vocabulary from a collection of cleaned training movie reviews."""
        counter: collections.Counter = collections.Counter()
        for text in texts:
            tokens = TOKEN_PATTERN.findall(text)
            counter.update(tokens)

        # Select most common tokens up to max_size - 2
        most_common = counter.most_common(self.max_size - 2)
        for idx, (token, _) in enumerate(most_common, start=2):
            self.token2idx[token] = idx
            self.idx2token[idx] = token

        logger.info(f"Vocabulary successfully built with {len(self.token2idx)} tokens.")
        return self

    def __len__(self) -> int:
        return len(self.token2idx)

    def encode(self, text: str, max_len: int = MAX_SEQ_LEN) -> List[int]:
        """Convert cleaned text to a padded or truncated list of token IDs."""
        tokens = TOKEN_PATTERN.findall(text)
        token_ids = [self.token2idx.get(t, self.unk_idx) for t in tokens]
        
        if len(token_ids) > max_len:
            return token_ids[:max_len]
        return token_ids + [self.pad_idx] * (max_len - len(token_ids))

    def decode(self, ids: List[int]) -> List[str]:
        """Convert a list of token IDs back into token strings."""
        return [self.idx2token.get(idx, self.unk_token) for idx in ids if idx != self.pad_idx]

    def save(self, file_path: Optional[str] = None) -> str:
        """Persist vocabulary object with joblib."""
        target_path = file_path or (MODELS_DIR / "vocab.joblib")
        joblib.dump(self, target_path)
        logger.info(f"Saved vocabulary to {target_path}")
        return str(target_path)

    @classmethod
    def load(cls, file_path: Optional[str] = None) -> "Vocabulary":
        """Load vocabulary object from disk."""
        target_path = file_path or (MODELS_DIR / "vocab.joblib")
        vocab = joblib.load(target_path)
        return vocab


class IMDBSeqDataset(Dataset):
    """PyTorch Dataset for sequence models (BiLSTM, 1D CNN)."""

    def __init__(self, texts: List[str], labels: Optional[List[int]], vocab: Vocabulary, max_len: int = MAX_SEQ_LEN) -> None:
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        cleaned = clean_text(text, lower=True)
        token_ids = self.vocab.encode(cleaned, max_len=self.max_len)
        
        item = {
            "input_ids": torch.tensor(token_ids, dtype=torch.long)
        }
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


class TransformerIMDBDataset(Dataset):
    """PyTorch Dataset for Hugging Face transformer models (DistilBERT)."""

    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]],
        tokenizer,
        max_len: int = MAX_SEQ_LEN
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Clean HTML artifacts but keep casing for uncased transformer tokenizer
        cleaned = clean_text(self.texts[idx], lower=False)
        encoding = self.tokenizer(
            cleaned,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

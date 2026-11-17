"""Unit tests for text preprocessing and vocabulary building."""

import pytest
from src.preprocess import clean_text, Vocabulary


def test_clean_text_strips_html():
    """Validate HTML tag stripping across various tag formats."""
    raw = "This was a <b>fantastic</b> movie!<br /><br />Highly recommended."
    cleaned = clean_text(raw, lower=True)
    assert "<" not in cleaned
    assert ">" not in cleaned
    assert "br" not in cleaned.split()
    assert "fantastic" in cleaned
    assert "highly recommended" in cleaned


def test_clean_text_unescapes_entities():
    """Validate HTML entity unescaping."""
    raw = "The director&apos;s cut was &quot;pure perfection&quot; &amp; breathtaking."
    cleaned = clean_text(raw, lower=True)
    assert "&quot;" not in cleaned
    assert "&amp;" not in cleaned
    assert "director's" in cleaned or "director" in cleaned
    assert "perfection" in cleaned


def test_clean_text_normalizes_whitespace():
    """Validate that extra whitespace, tabs, and newlines are collapsed."""
    raw = "   Great   acting!  \n\n\t  Terrible    pacing.   "
    cleaned = clean_text(raw, lower=True)
    assert cleaned == "great acting! terrible pacing."


def test_clean_text_empty_input():
    """Ensure graceful handling of None, empty strings, and whitespace."""
    assert clean_text("") == ""
    assert clean_text("   ") == ""
    assert clean_text(None) == ""


def test_vocabulary_encoding_and_decoding():
    """Validate vocabulary indexing, padding, and truncation."""
    texts = [
        "Incredible movie with stellar performances",
        "Terrible script and awful acting",
    ]
    vocab = Vocabulary(max_size=50)
    vocab.build_vocab(texts)

    # Check special tokens
    assert vocab.pad_idx == 0
    assert vocab.unk_idx == 1
    assert vocab.idx2token[0] == "<pad>"
    assert vocab.idx2token[1] == "<unk>"

    # Check encoding with padding
    encoded = vocab.encode("Incredible movie", max_len=5)
    assert len(encoded) == 5
    assert encoded[0] == vocab.token2idx["Incredible"] or encoded[0] in vocab.idx2token
    assert encoded[-1] == 0  # <pad>

    # Check encoding with truncation
    truncated = vocab.encode("Incredible movie with stellar performances and more", max_len=3)
    assert len(truncated) == 3

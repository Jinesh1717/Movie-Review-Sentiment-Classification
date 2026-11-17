"""Unit tests for inference API (predict function)."""

import pytest
from src.predict import predict


def test_predict_empty_text():
    """Verify predict handles empty text gracefully without throwing exceptions."""
    results = predict("")
    assert len(results) == 1
    res = results[0]
    assert "label" in res
    assert "positive_probability" in res
    assert "confidence" in res
    assert 0.0 <= res["positive_probability"] <= 1.0
    assert 0.0 <= res["confidence"] <= 1.0


def test_predict_valid_inputs_and_keys():
    """Verify predict returns required keys and valid probabilities on movie reviews."""
    # Test positive movie review
    pos_res = predict("An absolute masterpiece with brilliant acting and stunning cinematography!")[0]
    assert "label" in pos_res
    assert "positive_probability" in pos_res
    assert "confidence" in pos_res
    assert "model_used" in pos_res
    assert pos_res["label"] in ["Positive", "Negative"]
    assert 0.0 <= pos_res["positive_probability"] <= 1.0
    assert 0.5 <= pos_res["confidence"] <= 1.0

    # Test negative movie review
    neg_res = predict("A horrible and boring movie with terrible acting and a messy plot.")[0]
    assert 0.0 <= neg_res["positive_probability"] <= 1.0
    assert 0.5 <= neg_res["confidence"] <= 1.0


def test_predict_batch():
    """Verify batch inference handles lists of movie reviews properly."""
    reviews = [
        "One of the greatest films ever made.",
        "Completely unwatchable and dull.",
        "   ",  # whitespace edge case
    ]
    results = predict(reviews)
    assert len(results) == 3
    for r in results:
        assert "label" in r
        assert 0.0 <= r["positive_probability"] <= 1.0
        assert 0.0 <= r["confidence"] <= 1.0

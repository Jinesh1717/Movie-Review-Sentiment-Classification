"""Unit tests for data loading, binary labels, balance, and split isolation."""

import pytest
import pandas as pd
from src.data import load_raw_data, create_splits, validate_data


@pytest.fixture(scope="module")
def raw_data():
    """Load cached raw data for testing."""
    return load_raw_data()


def test_data_loading_returns_balanced_binary_labels(raw_data):
    """Validate that loaded dataset has binary labels and is 50/50 balanced."""
    train_df, test_df = raw_data
    
    assert "text" in train_df.columns
    assert "label" in train_df.columns
    assert "text" in test_df.columns
    assert "label" in test_df.columns

    # Binary labels check
    assert set(train_df["label"].unique()) == {0, 1}
    assert set(test_df["label"].unique()) == {0, 1}

    # Balance check: 50% positive (1) and 50% negative (0)
    train_pos_ratio = train_df["label"].mean()
    test_pos_ratio = test_df["label"].mean()
    assert abs(train_pos_ratio - 0.5) < 0.01, f"Train imbalance: {train_pos_ratio}"
    assert abs(test_pos_ratio - 0.5) < 0.01, f"Test imbalance: {test_pos_ratio}"


def test_stratified_validation_split(raw_data):
    """Verify validation split is created from train only with stratification."""
    train_df, _ = raw_data
    val_ratio = 0.15
    train_split, val_split = create_splits(train_df, val_ratio=val_ratio, seed=42)

    assert len(train_split) + len(val_split) == len(train_df)
    assert abs(val_split["label"].mean() - 0.5) < 0.02
    assert abs(train_split["label"].mean() - 0.5) < 0.02


def test_validation_integrity_function(raw_data):
    """Verify validate_data executes without assertion errors."""
    train_df, test_df = raw_data
    train_split, val_split = create_splits(train_df, val_ratio=0.15, seed=42)
    report = validate_data(train_split, val_split, test_df)

    assert report["train_size"] > 0
    assert report["val_size"] > 0
    assert report["test_size"] > 0
    assert "train_test_overlap_count" in report

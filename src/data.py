"""Data downloading, extraction, loading, validation, and splitting module.

Domain: Movie Reviews (Stanford IMDB Large Movie Review Dataset).
50,000 labeled movie reviews (25,000 train / 25,000 test, balanced positive/negative).
Unlabeled reviews in unsup folder are strictly excluded from supervised training/eval.
"""

import os
import sys
import tarfile
import urllib.request
from typing import Tuple, Dict, Any, List
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config import (
    RAW_DATA_DIR,
    ACL_IMDB_DIR,
    IMDB_TAR_URL,
    IMDB_TAR_PATH,
    HF_DATASET_NAME,
    HF_DATASET_FALLBACK,
    VAL_SPLIT_RATIO,
    SEED,
)
from src.utils import setup_logger, set_seed

logger = setup_logger("data")


class DownloadProgressBar(tqdm):
    """Progress bar hook for urlretrieve."""
    def update_to(self, b: int = 1, bsize: int = 1, tsize: int = None) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_and_extract_imdb() -> None:
    """Download and extract the Stanford IMDB movie review dataset.
    
    Falls back to Hugging Face 'stanfordnlp/imdb' or 'imdb' if direct URL fails.
    """
    if (ACL_IMDB_DIR / "train" / "pos").exists() and (ACL_IMDB_DIR / "test" / "pos").exists():
        logger.info(f"Raw IMDB movie review dataset already present at: {ACL_IMDB_DIR}")
        return

    logger.info("Raw dataset not found locally. Initiating download...")
    download_success = False

    # Attempt 1: Direct download from Stanford AI lab with timeout
    try:
        logger.info(f"Attempting download from {IMDB_TAR_URL} (timeout 10s)...")
        # Use urllib request with timeout
        req = urllib.request.Request(
            IMDB_TAR_URL, 
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            total_size = response.length or None
            with open(IMDB_TAR_PATH, "wb") as out_file:
                with tqdm(total=total_size, unit="B", unit_scale=True, desc="IMDB Dataset") as pbar:
                    while True:
                        chunk = response.read(65536)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        pbar.update(len(chunk))
        logger.info(f"Extracting {IMDB_TAR_PATH} to {RAW_DATA_DIR}...")
        with tarfile.open(IMDB_TAR_PATH, "r:gz") as tar:
            tar.extractall(path=RAW_DATA_DIR)
        download_success = True
        logger.info("Direct extraction complete.")
    except Exception as e:
        logger.warning(f"Direct download/extraction failed ({e}). Proceeding to Hugging Face fallback...")

    # Attempt 2: Fallback to Hugging Face datasets
    if not download_success:
        try:
            from datasets import load_dataset
            logger.info(f"Loading '{HF_DATASET_NAME}' from Hugging Face...")
            try:
                hf_ds = load_dataset(HF_DATASET_NAME)
            except Exception:
                logger.info(f"Falling back to '{HF_DATASET_FALLBACK}'...")
                hf_ds = load_dataset(HF_DATASET_FALLBACK)

            # Reconstruct folder structure so everything is standardized
            for split in ["train", "test"]:
                for label_str in ["pos", "neg"]:
                    (ACL_IMDB_DIR / split / label_str).mkdir(parents=True, exist_ok=True)
            
            for split in ["train", "test"]:
                ds_split = hf_ds[split]
                for idx, row in enumerate(ds_split):
                    text = row["text"]
                    label_str = "pos" if row["label"] == 1 else "neg"
                    file_path = ACL_IMDB_DIR / split / label_str / f"{idx}_{row['label']}.txt"
                    file_path.write_text(text, encoding="utf-8")
            download_success = True
            logger.info("Successfully populated dataset from Hugging Face cache.")
        except Exception as hf_err:
            logger.error(f"Hugging Face fallback also failed: {hf_err}")

    # Manual instructions if both failed
    if not download_success:
        logger.critical(
            "\n" + "=" * 70 + "\n"
            "DATASET DOWNLOAD FAILED.\n"
            "Please manually download 'aclImdb_v1.tar.gz' from:\n"
            "  https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz\n"
            f"Extract it so that the following directory exists:\n"
            f"  {ACL_IMDB_DIR}\n"
            "Containing 'train/pos', 'train/neg', 'test/pos', 'test/neg'.\n"
            + "=" * 70
        )
        sys.exit(1)


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test movie reviews into DataFrames.
    
    Labels: 1 for positive movie review, 0 for negative movie review.
    Unsupervised reviews ('unsup') are explicitly ignored.
    Uses cached parquet for near-instant subsequent loads.
    """
    download_and_extract_imdb()

    cache_train = RAW_DATA_DIR / "imdb_train.parquet"
    cache_test = RAW_DATA_DIR / "imdb_test.parquet"

    if cache_train.exists() and cache_test.exists():
        logger.info(f"Loading cached movie reviews from {cache_train} and {cache_test}...")
        train_df = pd.read_parquet(cache_train)
        test_df = pd.read_parquet(cache_test)
        logger.info(f"Loaded {len(train_df)} training reviews and {len(test_df)} test reviews from cache.")
        return train_df, test_df

    def _read_dir(split: str) -> pd.DataFrame:
        texts: List[str] = []
        labels: List[int] = []
        for label_name, label_val in [("pos", 1), ("neg", 0)]:
            folder = ACL_IMDB_DIR / split / label_name
            if not folder.exists():
                raise FileNotFoundError(f"Missing expected directory: {folder}")
            filenames = os.listdir(folder)
            for fname in filenames:
                if fname.endswith(".txt"):
                    filepath = os.path.join(str(folder), fname)
                    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                        texts.append(f.read())
                    labels.append(label_val)
        return pd.DataFrame({"text": texts, "label": labels})

    logger.info("Reading raw movie review text files from disk...")
    train_df = _read_dir("train")
    test_df = _read_dir("test")

    # Cache as parquet for ultra-fast reloading
    try:
        train_df.to_parquet(cache_train, index=False)
        test_df.to_parquet(cache_test, index=False)
        logger.info("Cached raw movie reviews to parquet.")
    except Exception as e:
        logger.warning(f"Could not write parquet cache: {e}")

    logger.info(f"Loaded {len(train_df)} training movie reviews and {len(test_df)} test movie reviews.")
    return train_df, test_df


def create_splits(
    train_df: pd.DataFrame, val_ratio: float = VAL_SPLIT_RATIO, seed: int = SEED
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a stratified validation split from the TRAIN set only.
    
    The test set remains completely untouched until final evaluation.
    """
    train_split, val_split = train_test_split(
        train_df,
        test_size=val_ratio,
        random_state=seed,
        stratify=train_df["label"],
        shuffle=True,
    )
    return train_split.reset_index(drop=True), val_split.reset_index(drop=True)


def validate_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Dict[str, Any]:
    """Validate data integrity:
    
    1. No missing / null values.
    2. Labels are strictly binary (0 or 1).
    3. Check for train/test text overlap and report duplicates.
    4. Balanced positive/negative distribution.
    """
    # Check nulls
    assert train_df["text"].isnull().sum() == 0, "Null texts found in training set!"
    assert val_df["text"].isnull().sum() == 0, "Null texts found in validation set!"
    assert test_df["text"].isnull().sum() == 0, "Null texts found in test set!"

    # Check labels
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        unique_labels = set(df["label"].unique())
        assert unique_labels.issubset({0, 1}), f"Non-binary labels found in {name}: {unique_labels}"

    # Check text overlap between train+val and test
    train_texts = set(train_df["text"].str.strip().tolist() + val_df["text"].str.strip().tolist())
    test_texts = set(test_df["text"].str.strip().tolist())
    overlap = train_texts.intersection(test_texts)
    
    report = {
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "train_pos_ratio": float(train_df["label"].mean()),
        "val_pos_ratio": float(val_df["label"].mean()),
        "test_pos_ratio": float(test_df["label"].mean()),
        "train_test_overlap_count": len(overlap),
    }

    logger.info(
        f"Validation Report: Train={report['train_size']} (pos={report['train_pos_ratio']:.3f}), "
        f"Val={report['val_size']} (pos={report['val_pos_ratio']:.3f}), "
        f"Test={report['test_size']} (pos={report['test_pos_ratio']:.3f})"
    )
    if len(overlap) > 0:
        logger.warning(f"Identified {len(overlap)} exact duplicate texts between train and test sets.")
    else:
        logger.info("Zero duplicate texts found between train and test sets.")

    return report


def get_dataset_splits(
    val_ratio: float = VAL_SPLIT_RATIO, seed: int = SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Orchestrate data download, loading, validation, and stratified splitting."""
    set_seed(seed)
    train_raw, test_raw = load_raw_data()
    train_df, val_df = create_splits(train_raw, val_ratio=val_ratio, seed=seed)
    validate_data(train_df, val_df, test_raw)
    return train_df, val_df, test_raw


if __name__ == "__main__":
    logger.info("Running standalone data pipeline...")
    tr, val, te = get_dataset_splits()
    logger.info("Data pipeline finished successfully.")

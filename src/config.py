"""Configuration settings, paths, seeds, and hyperparameters for DL-06 Movie Review Sentiment Classification."""

import os
from pathlib import Path
import torch

# Ensure Hugging Face and PyTorch never write cache to C: drive
os.environ["HF_HOME"] = "D:\\cache\\huggingface"
os.environ["TORCH_HOME"] = "D:\\cache\\torch"

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
ACL_IMDB_DIR = RAW_DATA_DIR / "aclImdb"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure runtime directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset URLs & names
IMDB_TAR_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
IMDB_TAR_PATH = RAW_DATA_DIR / "aclImdb_v1.tar.gz"
HF_DATASET_NAME = "stanfordnlp/imdb"
HF_DATASET_FALLBACK = "imdb"

# Global reproducibility seed
SEED = 42

# Device selection (Graceful CPU fallback if CUDA unavailable)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data splits & preprocessing parameters
VAL_SPLIT_RATIO = 0.15
MAX_SEQ_LEN = 256
VOCAB_SIZE = 25000
OOV_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"

# Classical Models Hyperparameters
TFIDF_MAX_FEATURES = 25000
TFIDF_NGRAM_RANGE = (1, 2)
LOGREG_C = 1.0
SVM_C = 1.0

# Deep Learning (LSTM & CNN) Hyperparameters
EMBEDDING_DIM = 128
LSTM_HIDDEN_DIM = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
CNN_NUM_FILTERS = 128
CNN_FILTER_SIZES = [3, 4, 5]
CNN_DROPOUT = 0.4
DL_BATCH_SIZE = 128
DL_LEARNING_RATE = 1e-3
DL_WEIGHT_DECAY = 1e-5
DL_EPOCHS = 3
DL_PATIENCE = 2

# Transformer Hyperparameters (DistilBERT)
TRANSFORMER_MODEL_NAME = "distilbert-base-uncased"
TRANSFORMER_MAX_LEN = 256
TRANSFORMER_BATCH_SIZE = 16
TRANSFORMER_EPOCHS = 2
TRANSFORMER_LR = 2e-5
TRANSFORMER_WARMUP_RATIO = 0.1
TRANSFORMER_WEIGHT_DECAY = 0.01

# Subsample for CPU execution if GPU is not available to maintain interactive feasibility
# If GPU is available, use all reviews; on CPU, subsample 3,000 for fast training
CPU_TRANSFORMER_SUBSET_SIZE = 3000

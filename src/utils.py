"""Shared constants and helpers for the AI Fingerprint Tracker pipeline."""
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "cifake"
EMB_DIR = PROJECT_ROOT / "embeddings"
OUT_DIR = PROJECT_ROOT / "outputs"

QUICK_MODE = False
MAX_TRAIN_SAMPLES = 10_000
CLUSTER_TRAIN_SAMPLES = 10_000
RANDOM_SEED = 42


def get_stratified_subset_indices(
    labels: np.ndarray, max_samples: int, seed: int = RANDOM_SEED
) -> np.ndarray:
    """Return indices with balanced REAL (0) / FAKE (1) sampling."""
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    per_class = max_samples // len(np.unique(labels))

    indices = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        n_take = min(per_class, len(cls_idx))
        chosen = rng.choice(cls_idx, size=n_take, replace=False)
        indices.append(chosen)

    result = np.concatenate(indices)
    rng.shuffle(result)
    return result.astype(np.int64)

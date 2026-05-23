from pathlib import Path

import pytest

from src.preprocess import CIFAKEDataset, CLIP_TRANSFORM, validate_dataset
from src.utils import DATA_ROOT


def test_dataset_loads():
    ds = CIFAKEDataset(DATA_ROOT, split="train", transform=CLIP_TRANSFORM)
    assert len(ds) > 0, "Dataset is empty"


def test_class_balance():
    ds = CIFAKEDataset(DATA_ROOT, split="train", transform=CLIP_TRANSFORM)
    import numpy as np

    n = min(2000, len(ds))
    idx = np.random.RandomState(42).choice(len(ds), size=n, replace=False)
    labels = [ds[int(i)][1] for i in idx]
    real = labels.count(0)
    fake = labels.count(1)
    assert fake > 0, "Expected both classes in random sample"
    assert 0.4 <= real / len(labels) <= 0.6, f"Class imbalance: {real}/{fake}"


def test_image_shape_and_range():
    ds = CIFAKEDataset(DATA_ROOT, split="train", transform=CLIP_TRANSFORM)
    img, label, path = ds[0]
    assert img.shape == (3, 224, 224), f"Wrong shape: {img.shape}"
    assert img.dtype.is_floating_point, "Image should be float tensor"
    assert isinstance(label, int) and label in [0, 1]
    assert Path(path).exists()


def test_validate_dataset_runs():
    stats = validate_dataset(DATA_ROOT)
    assert "train" in stats and "test" in stats
    for split in stats:
        assert "REAL" in stats[split] and "FAKE" in stats[split]
        assert stats[split]["REAL"] > 0 and stats[split]["FAKE"] > 0
    assert stats["train"]["REAL"] >= 40000
    assert stats["test"]["REAL"] >= 5000


def test_normalization_values():
    ds = CIFAKEDataset(DATA_ROOT, split="train", transform=CLIP_TRANSFORM)
    img, _, _ = ds[0]
    assert -3.0 < img.mean().item() < 3.0, "Unusual normalization range"

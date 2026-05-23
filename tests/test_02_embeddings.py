import numpy as np

from src.utils import EMB_DIR, MAX_TRAIN_SAMPLES, QUICK_MODE


def test_embedding_files_exist():
    for split in ["train", "test"]:
        for suffix in ["embeddings", "labels", "paths"]:
            p = EMB_DIR / f"{split}_{suffix}.npy"
            assert p.exists(), f"Missing: {p}"


def test_embedding_shape():
    embs = np.load(EMB_DIR / "train_embeddings.npy")
    labels = np.load(EMB_DIR / "train_labels.npy")
    paths = np.load(EMB_DIR / "train_paths.npy")
    assert embs.ndim == 2 and embs.shape[1] == 512, f"Bad shape: {embs.shape}"
    assert len(embs) == len(labels) == len(paths)
    if QUICK_MODE:
        assert len(embs) == MAX_TRAIN_SAMPLES
    else:
        assert len(embs) == 100_000


def test_embedding_normalized():
    embs = np.load(EMB_DIR / "train_embeddings.npy")
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), f"Not L2 normalized: {norms[:5]}"


def test_label_values():
    labels = np.load(EMB_DIR / "train_labels.npy")
    assert set(labels.tolist()).issubset({0, 1}), "Labels must be 0 or 1"
    real_frac = (labels == 0).mean()
    assert 0.4 <= real_frac <= 0.6, f"Unexpected class balance: {real_frac:.2f}"


def test_real_fake_separation():
    embs = np.load(EMB_DIR / "train_embeddings.npy")
    labels = np.load(EMB_DIR / "train_labels.npy")
    n = min(2000, len(embs))
    idx = np.random.RandomState(42).choice(len(embs), size=n, replace=False)
    embs, labels = embs[idx], labels[idx]
    real_mean = embs[labels == 0].mean(axis=0)
    fake_mean = embs[labels == 1].mean(axis=0)
    real_mean /= np.linalg.norm(real_mean)
    fake_mean /= np.linalg.norm(fake_mean)
    sim = float(np.dot(real_mean, fake_mean))
    print(f"REAL/FAKE mean cosine sim: {sim:.4f}")
    assert sim < 0.999, "Embeddings are suspiciously identical"

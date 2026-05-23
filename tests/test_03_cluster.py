import json
from pathlib import Path

import numpy as np

from src.utils import EMB_DIR, OUT_DIR


def test_output_files_exist():
    for fname in [
        "umap_2d.npy",
        "umap_50d.npy",
        "pca_50d.npy",
        "cluster_assignments_kmeans.npy",
        "cluster_evaluation.json",
    ]:
        assert (OUT_DIR / fname).exists(), f"Missing {fname}"


def test_umap_2d_shape():
    umap_2d = np.load(OUT_DIR / "umap_2d.npy")
    assert umap_2d.ndim == 2 and umap_2d.shape[1] == 2


def test_cluster_labels_valid():
    labels = np.load(OUT_DIR / "cluster_assignments_kmeans.npy")
    assert labels.ndim == 1
    unique = set(labels.tolist())
    assert len(unique) >= 6, f"Too few clusters: {len(unique)}"
    assert len(unique) <= 20, f"Too many clusters: {len(unique)}"


def test_cluster_evaluation_loaded():
    with open(OUT_DIR / "cluster_evaluation.json", encoding="utf-8") as f:
        eval_data = json.load(f)
    assert "kmeans_k" in eval_data
    assert "cluster_stats" in eval_data
    stats = eval_data["cluster_stats"]
    fake_dominant = [
        v
        for k, v in stats.items()
        if isinstance(v, dict) and v.get("dominant") == "FAKE"
    ]
    assert len(fake_dominant) > 0, "No FAKE-dominant clusters found"


def test_silhouette_reasonable():
    with open(OUT_DIR / "cluster_evaluation.json", encoding="utf-8") as f:
        eval_data = json.load(f)
    best_k = eval_data["kmeans_k"]
    score = eval_data["kmeans_scores"][str(best_k)]
    assert score > 0.05, f"Silhouette score too low: {score}"

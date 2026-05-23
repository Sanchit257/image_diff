import json
from pathlib import Path

import numpy as np

from src.utils import EMB_DIR, OUT_DIR


def test_pipeline_output_completeness():
    required = [
        EMB_DIR / "train_embeddings.npy",
        EMB_DIR / "train_labels.npy",
        OUT_DIR / "umap_2d.npy",
        OUT_DIR / "cluster_assignments_kmeans.npy",
        OUT_DIR / "fingerprint_report.json",
        OUT_DIR / "visualization.html",
    ]
    for path in required:
        assert path.exists(), f"MISSING: {path}"


def test_pipeline_consistency():
    embs = np.load(EMB_DIR / "train_embeddings.npy")
    labels = np.load(OUT_DIR / "cluster_assignments_kmeans.npy")
    umap = np.load(OUT_DIR / "umap_2d.npy")
    assert len(labels) <= len(embs)
    assert len(labels) == len(umap)


def test_fingerprint_report_meaningful():
    with open(OUT_DIR / "fingerprint_report.json", encoding="utf-8") as f:
        report = json.load(f)
    high_purity = [
        v
        for v in report.values()
        if isinstance(v, dict) and max(v.get("fake_pct", 0), v.get("real_pct", 0)) > 0.7
    ]
    assert len(high_purity) >= 2, f"Only {len(high_purity)} high-purity clusters"

"""PCA, UMAP, and clustering on CLIP embeddings."""
import json
from pathlib import Path

import hdbscan
import numpy as np
import umap
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src.utils import EMB_DIR, OUT_DIR

EMB_PATH = EMB_DIR / "train_embeddings.npy"
LABEL_PATH = EMB_DIR / "train_labels.npy"


def load_embeddings():
    embs = np.load(EMB_PATH).astype(np.float32)
    labels = np.load(LABEL_PATH).astype(np.int32)
    return embs, labels


def run_pca(embs: np.ndarray, n_components: int = 50):
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embs)
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA {n_components}D: {explained:.1%} variance explained")
    np.save(OUT_DIR / "pca_50d.npy", reduced)
    return reduced, pca


def run_umap_2d(pca_embs: np.ndarray):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        verbose=True,
    )
    umap_2d = reducer.fit_transform(pca_embs)
    np.save(OUT_DIR / "umap_2d.npy", umap_2d)
    print(f"UMAP 2D: {umap_2d.shape}")
    return umap_2d, reducer


def run_umap_50d(pca_embs: np.ndarray):
    reducer = umap.UMAP(
        n_components=50,
        n_neighbors=30,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        verbose=True,
    )
    umap_50d = reducer.fit_transform(pca_embs)
    np.save(OUT_DIR / "umap_50d.npy", umap_50d)
    return umap_50d, reducer


def run_kmeans(umap_50d: np.ndarray, k_range=range(6, 20)):
    best_k, best_score, best_labels = None, -1, None
    scores = {}
    for k in k_range:
        km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=10000, n_init=5)
        cluster_labels = km.fit_predict(umap_50d)
        score = silhouette_score(
            umap_50d, cluster_labels, sample_size=min(5000, len(umap_50d)), random_state=42
        )
        scores[str(k)] = float(score)
        print(f"  k={k}: silhouette={score:.4f}")
        if score > best_score:
            best_score, best_k, best_labels = score, k, cluster_labels
    print(f"Best k={best_k} (silhouette={best_score:.4f})")
    return best_labels, best_k, scores


def run_hdbscan(umap_50d: np.ndarray):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=200,
        min_samples=10,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(umap_50d)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = (labels == -1).mean()
    print(f"HDBSCAN: {n_clusters} clusters, {noise_pct:.1%} noise")
    return labels, clusterer


def evaluate_clusters(cluster_labels, true_labels):
    results = {}
    for c in sorted(set(cluster_labels)):
        if c == -1:
            continue
        mask = cluster_labels == c
        real_pct = (true_labels[mask] == 0).mean()
        fake_pct = (true_labels[mask] == 1).mean()
        dominant = "REAL" if real_pct > 0.5 else "FAKE"
        purity = max(real_pct, fake_pct)
        results[int(c)] = {
            "size": int(mask.sum()),
            "real_pct": float(real_pct),
            "fake_pct": float(fake_pct),
            "dominant": dominant,
            "purity": float(purity),
        }
    ari = adjusted_rand_score(true_labels, cluster_labels)
    results["ARI"] = float(ari)
    print(f"ARI (cluster vs REAL/FAKE): {ari:.4f}")
    return results


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading embeddings...")
    embs, labels = load_embeddings()
    print(f"  Shape: {embs.shape}")

    print("\n── PCA ──")
    pca_embs, _ = run_pca(embs, n_components=50)

    print("\n── UMAP 2D ──")
    run_umap_2d(pca_embs)

    print("\n── UMAP 50D ──")
    umap_50d, _ = run_umap_50d(pca_embs)

    print("\n── KMeans sweep ──")
    km_labels, best_k, km_scores = run_kmeans(umap_50d)
    np.save(OUT_DIR / "cluster_assignments_kmeans.npy", km_labels)

    print("\n── HDBSCAN ──")
    hdb_labels, _ = run_hdbscan(umap_50d)
    np.save(OUT_DIR / "cluster_assignments_hdbscan.npy", hdb_labels)

    print("\n── Evaluation ──")
    km_eval = evaluate_clusters(km_labels, labels)
    with open(OUT_DIR / "cluster_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(
            {"kmeans_k": best_k, "kmeans_scores": km_scores, "cluster_stats": km_eval},
            f,
            indent=2,
        )

    print("Done. Results saved to outputs/")

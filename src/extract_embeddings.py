"""CLIP ViT-B/32 embedding extraction."""
import os
from pathlib import Path

import numpy as np
import open_clip
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.preprocess import CIFAKEDataset, CLIP_TRANSFORM
from src.utils import (
    DATA_ROOT,
    EMB_DIR,
    MAX_TRAIN_SAMPLES,
    QUICK_MODE,
    get_stratified_subset_indices,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_clip_model():
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(DEVICE).eval()
    print(f"CLIP ViT-B/32 loaded on {DEVICE}")
    return model


@torch.no_grad()
def extract_embeddings(model, split: str = "train", subset_indices=None):
    ds = CIFAKEDataset(DATA_ROOT, split=split, transform=CLIP_TRANSFORM)
    if subset_indices is not None:
        ds.samples = [ds.samples[i] for i in subset_indices]

    loader = DataLoader(
        ds,
        batch_size=512,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    all_embs, all_labels, all_paths = [], [], []
    for imgs, labels, paths in tqdm(loader, desc=f"Extracting [{split}]"):
        imgs = imgs.to(DEVICE)
        embs = model.encode_image(imgs)
        embs = embs / embs.norm(dim=-1, keepdim=True)
        all_embs.append(embs.cpu().numpy())
        all_labels.extend(labels.numpy().tolist())
        all_paths.extend(paths)

    embeddings = np.vstack(all_embs).astype(np.float32)
    labels = np.array(all_labels, dtype=np.int32)
    paths = np.array(all_paths)
    return embeddings, labels, paths


def save_embeddings(split: str = "train", subset_indices=None):
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    model = load_clip_model()
    embs, labels, paths = extract_embeddings(model, split, subset_indices=subset_indices)
    np.save(EMB_DIR / f"{split}_embeddings.npy", embs)
    np.save(EMB_DIR / f"{split}_labels.npy", labels)
    np.save(EMB_DIR / f"{split}_paths.npy", paths)
    print(f"Saved {split}: embeddings={embs.shape}, labels={labels.shape}")
    norms = np.linalg.norm(embs, axis=1)
    print(f"  Embedding L2 norms: mean={norms.mean():.4f}, std={norms.std():.6f}")
    return embs, labels, paths


if __name__ == "__main__":
    os.makedirs(EMB_DIR, exist_ok=True)
    model = load_clip_model()

    if QUICK_MODE:
        full_ds = CIFAKEDataset(DATA_ROOT, split="train", transform=None)
        all_labels = np.array([s[1] for s in full_ds.samples], dtype=np.int32)
        subset_idx = get_stratified_subset_indices(all_labels, MAX_TRAIN_SAMPLES)
        np.save(EMB_DIR / "train_subset_idx.npy", subset_idx)
        print(f"Quick mode: {len(subset_idx)} train samples")
        embs, labels, paths = extract_embeddings(model, "train", subset_indices=subset_idx)
        np.save(EMB_DIR / "train_embeddings.npy", embs)
        np.save(EMB_DIR / "train_labels.npy", labels)
        np.save(EMB_DIR / "train_paths.npy", paths)
        print(f"Saved train: embeddings={embs.shape}")
    else:
        save_embeddings("train")

    save_embeddings("test")
    print("Extraction complete.")

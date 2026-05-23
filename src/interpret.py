"""Fingerprint interpretation: avg images, FFT, CLIP descriptions."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from src.utils import DATA_ROOT, EMB_DIR, OUT_DIR

CANDIDATE_TEXTS = [
    "a real photograph",
    "an AI-generated image",
    "a synthetic image",
    "sharp edges and artifacts",
    "smooth gradients",
    "photorealistic texture",
    "blurry or soft focus",
    "high frequency noise",
    "uniform color blocks",
    "checkerboard pattern",
    "natural scene",
    "computer generated graphics",
    "JPEG compression artifacts",
    "overly saturated colors",
    "washed out colors",
    "geometric patterns",
    "organic textures",
    "repetitive patterns",
]


def load_cluster_assignments():
    labels = np.load(OUT_DIR / "cluster_assignments_kmeans.npy")
    paths = np.load(EMB_DIR / "train_paths.npy")
    true_labels = np.load(EMB_DIR / "train_labels.npy")
    min_len = min(len(labels), len(paths), len(true_labels))
    return labels[:min_len], paths[:min_len], true_labels[:min_len]


def compute_average_image(paths, max_images=500):
    avg = np.zeros((32, 32, 3), dtype=np.float64)
    count = 0
    for path in paths[:max_images]:
        try:
            img = np.array(Image.open(path).convert("RGB").resize((32, 32))) / 255.0
            avg += img
            count += 1
        except OSError:
            pass
    return (avg / max(count, 1)).clip(0, 1)


def compute_fft_profile(paths, max_images=500):
    fft_sum = np.zeros((32, 32), dtype=np.float64)
    count = 0
    for path in paths[:max_images]:
        try:
            img = np.array(Image.open(path).convert("L").resize((32, 32))) / 255.0
            fft = np.fft.fft2(img)
            fft_shifted = np.fft.fftshift(fft)
            magnitude = np.log1p(np.abs(fft_shifted))
            fft_sum += magnitude
            count += 1
        except OSError:
            pass
    return fft_sum / max(count, 1)


def compute_noise_residual(paths, max_images=200):
    noise_sum = np.zeros((32, 32, 3), dtype=np.float64)
    count = 0
    for path in paths[:max_images]:
        try:
            img = np.array(Image.open(path).convert("RGB").resize((32, 32))) / 255.0
            blurred = ndimage.gaussian_filter(img, sigma=1.5)
            noise = np.abs(img - blurred)
            noise_sum += noise
            count += 1
        except OSError:
            pass
    return noise_sum / max(count, 1)


def get_clip_cluster_description(paths, model, tokenizer, max_images=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    tokens = tokenizer(CANDIDATE_TEXTS).to(device)
    with torch.no_grad():
        text_feats = model.encode_text(tokens)
        text_feats /= text_feats.norm(dim=-1, keepdim=True)

    from torchvision import transforms as T

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ])

    img_feats = []
    for path in paths[:max_images]:
        try:
            img = Image.open(path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode_image(tensor)
                feat /= feat.norm(dim=-1, keepdim=True)
            img_feats.append(feat.cpu())
        except OSError:
            pass

    if not img_feats:
        return []

    cluster_feat = torch.stack(img_feats).mean(0)
    cluster_feat /= cluster_feat.norm()

    sims = (cluster_feat @ text_feats.cpu().T).squeeze()
    top_idx = sims.argsort(descending=True)[:5]
    return [(CANDIDATE_TEXTS[i], float(sims[i])) for i in top_idx]


def generate_fingerprint_report(cluster_assignments, paths, true_labels, model=None, tokenizer=None):
    report_dir = OUT_DIR / "cluster_avg_images"
    fft_dir = OUT_DIR / "fft_profiles"
    report_dir.mkdir(parents=True, exist_ok=True)
    fft_dir.mkdir(parents=True, exist_ok=True)

    fingerprint_data = {}

    for c in tqdm(sorted(set(cluster_assignments)), desc="Interpreting clusters"):
        if c == -1:
            continue
        mask = cluster_assignments == c
        cluster_paths = paths[mask]
        cluster_true = true_labels[mask]

        avg_img = compute_average_image(cluster_paths)
        fft_map = compute_fft_profile(cluster_paths)

        avg_pil = Image.fromarray((avg_img * 255).astype(np.uint8)).resize(
            (128, 128), Image.NEAREST
        )
        avg_pil.save(report_dir / f"cluster_{c:02d}_avg.png")

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(fft_map, cmap="inferno")
        ax.set_title(f"Cluster {c} FFT Profile")
        ax.axis("off")
        fig.savefig(fft_dir / f"cluster_{c:02d}_fft.png", dpi=80, bbox_inches="tight")
        plt.close(fig)

        clip_desc = []
        if model is not None:
            clip_desc = get_clip_cluster_description(cluster_paths, model, tokenizer)

        fingerprint_data[int(c)] = {
            "size": int(mask.sum()),
            "real_pct": float((cluster_true == 0).mean()),
            "fake_pct": float((cluster_true == 1).mean()),
            "dominant": "REAL" if (cluster_true == 0).mean() > 0.5 else "FAKE",
            "clip_descriptions": clip_desc,
            "avg_image_path": str(report_dir / f"cluster_{c:02d}_avg.png"),
            "fft_path": str(fft_dir / f"cluster_{c:02d}_fft.png"),
        }

    with open(OUT_DIR / "fingerprint_report.json", "w", encoding="utf-8") as f:
        json.dump(fingerprint_data, f, indent=2)
    print("Fingerprint report saved.")
    return fingerprint_data


if __name__ == "__main__":
    cluster_labels, paths, true_labels = load_cluster_assignments()

    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    report = generate_fingerprint_report(cluster_labels, paths, true_labels, model, tokenizer)
    print(f"Generated fingerprints for {len(report)} clusters")

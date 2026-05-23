"""Data preprocessing and validation for CIFAKE."""
import json
import os
from pathlib import Path

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.utils import DATA_ROOT, OUT_DIR

IMAGE_SIZE = 224
BATCH_SIZE = 256
NUM_WORKERS = 0

CLIP_TRANSFORM = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ),
])


class CIFAKEDataset(Dataset):
    """CIFAKE dataset: root/{split}/{REAL|FAKE}/*.jpg"""

    def __init__(self, root: Path, split: str = "train", transform=None):
        self.transform = transform
        self.samples = []
        for label_name, label_idx in [("REAL", 0), ("FAKE", 1)]:
            folder = root / split / label_name
            if not folder.exists():
                continue
            for img_path in sorted(folder.glob("*.jpg")):
                self.samples.append((str(img_path), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, path


def validate_dataset(root: Path):
    """Check structure and balance, save stats."""
    stats = {}
    for split in ["train", "test"]:
        stats[split] = {}
        for cls in ["REAL", "FAKE"]:
            folder = root / split / cls
            if not folder.exists():
                raise FileNotFoundError(f"Missing: {folder}")
            count = len(list(folder.glob("*.jpg")))
            stats[split][cls] = count
            print(f"  {split}/{cls}: {count} images")

    for split in ["train", "test"]:
        r, f = stats[split]["REAL"], stats[split]["FAKE"]
        ratio = r / f if f > 0 else 0
        print(f"  {split} balance (REAL/FAKE): {ratio:.3f}")
        if not (0.8 <= ratio <= 1.2):
            print(f"  WARNING: Imbalanced {split} split")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "dataset_stats.json", "w", encoding="utf-8") as fp:
        json.dump(stats, fp, indent=2)
    return stats


def build_dataloader(split: str = "train") -> DataLoader:
    ds = CIFAKEDataset(DATA_ROOT, split=split, transform=CLIP_TRANSFORM)
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Validating dataset...")
    stats = validate_dataset(DATA_ROOT)
    print("\nDataset valid. Stats saved to outputs/dataset_stats.json")

    loader = build_dataloader("train")
    imgs, labels, paths = next(iter(loader))
    print(f"\nSpot-check batch: imgs={imgs.shape}, labels={labels.shape}")
    print(f"Sample path: {paths[0]}")
    print(f"Pixel range after normalization: [{imgs.min():.2f}, {imgs.max():.2f}]")

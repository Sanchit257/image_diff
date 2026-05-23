import json
from pathlib import Path

from PIL import Image

from src.utils import OUT_DIR


def test_fingerprint_report_exists():
    assert (OUT_DIR / "fingerprint_report.json").exists()


def test_fingerprint_report_content():
    with open(OUT_DIR / "fingerprint_report.json", encoding="utf-8") as f:
        report = json.load(f)
    assert len(report) >= 4, "Too few clusters in report"
    for _, data in report.items():
        assert "size" in data and "real_pct" in data and "fake_pct" in data
        assert "dominant" in data and data["dominant"] in ["REAL", "FAKE"]


def test_average_images_exist():
    avg_dir = OUT_DIR / "cluster_avg_images"
    assert avg_dir.exists()
    avg_files = list(avg_dir.glob("cluster_*_avg.png"))
    assert len(avg_files) >= 4, f"Only {len(avg_files)} avg images"


def test_avg_image_size():
    avg_files = list((OUT_DIR / "cluster_avg_images").glob("*.png"))
    img = Image.open(avg_files[0])
    assert img.size == (128, 128), f"Wrong size: {img.size}"


def test_fft_profiles_exist():
    fft_dir = OUT_DIR / "fft_profiles"
    assert fft_dir.exists()
    fft_files = list(fft_dir.glob("cluster_*_fft.png"))
    assert len(fft_files) >= 4


def test_fake_clusters_present():
    with open(OUT_DIR / "fingerprint_report.json", encoding="utf-8") as f:
        report = json.load(f)
    fake_clusters = [v for v in report.values() if v["dominant"] == "FAKE"]
    assert len(fake_clusters) >= 1, "No FAKE-dominant clusters"

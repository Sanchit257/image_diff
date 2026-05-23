from src.utils import OUT_DIR


def test_html_visualization_exists():
    assert (OUT_DIR / "visualization.html").exists()


def test_html_not_empty():
    content = (OUT_DIR / "visualization.html").read_text(encoding="utf-8")
    assert len(content) > 50000, "HTML seems too small"
    assert "plotly" in content.lower(), "Missing Plotly.js"
    assert "UMAP" in content or "Fingerprint" in content, "Missing expected labels"


def test_html_has_multiple_traces():
    content = (OUT_DIR / "visualization.html").read_text(encoding="utf-8")
    count = content.count("Cluster")
    assert count >= 6, f"Only {count} cluster references found"

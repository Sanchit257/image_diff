"""Interactive Plotly UMAP visualization."""
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from src.utils import EMB_DIR, OUT_DIR

CLUSTER_COLORS = [
    "#FF6B6B",
    "#4ECDC4",
    "#45B7D1",
    "#96CEB4",
    "#FFEAA7",
    "#DDA0DD",
    "#98D8C8",
    "#F7B731",
    "#A29BFE",
    "#FD79A8",
    "#00B894",
    "#E17055",
    "#74B9FF",
    "#55EFC4",
    "#FDCB6E",
]


def build_plotly_scatter(
    umap_2d: np.ndarray,
    cluster_labels: np.ndarray,
    true_labels: np.ndarray,
    paths: np.ndarray,
    fingerprint_report: dict,
    max_points: int = 10000,
    output_path: str = "outputs/visualization.html",
):
    n = len(umap_2d)
    if n > max_points:
        idx = np.random.RandomState(42).choice(n, max_points, replace=False)
    else:
        idx = np.arange(n)

    x = umap_2d[idx, 0]
    y = umap_2d[idx, 1]
    c_labels = cluster_labels[idx]
    t_labels = true_labels[idx]
    p = paths[idx]

    x_pad = max((x.max() - x.min()) * 0.08, 0.5)
    y_pad = max((y.max() - y.min()) * 0.08, 0.5)
    x_range = [float(x.min() - x_pad), float(x.max() + x_pad)]
    y_range = [float(y.min() - y_pad), float(y.max() + y_pad)]

    fig = go.Figure()
    for cluster_id in sorted(set(c_labels)):
        if cluster_id == -1:
            continue
        mask = c_labels == cluster_id
        info = fingerprint_report.get(str(cluster_id), fingerprint_report.get(int(cluster_id), {}))
        dominant = info.get("dominant", "?")
        fake_pct = info.get("fake_pct", 0)
        clip_desc = info.get("clip_descriptions", [])
        clip_top = clip_desc[0][0] if clip_desc else "No description"

        label_name = f"C{cluster_id}: {dominant} ({fake_pct:.0%} AI)"
        color = CLUSTER_COLORS[int(cluster_id) % len(CLUSTER_COLORS)]

        hover_texts = []
        for ti, pi in zip(t_labels[mask], p[mask]):
            hover_texts.append(
                f"<b>Cluster {cluster_id}</b><br>"
                f"True label: {'REAL' if ti == 0 else 'AI-GENERATED'}<br>"
                f"CLIP says: {clip_top}<br>"
                f"File: {Path(pi).name}"
            )

        fig.add_trace(
            go.Scattergl(
                x=x[mask],
                y=y[mask],
                mode="markers",
                name=label_name,
                marker=dict(
                    color=color,
                    size=4,
                    opacity=0.65,
                    symbol="circle-open" if dominant == "REAL" else "circle",
                ),
                hovertext=hover_texts,
                hoverinfo="text",
            )
        )

    n_fake_clusters = sum(
        1
        for v in fingerprint_report.values()
        if isinstance(v, dict) and v.get("dominant") == "FAKE"
    )
    fig.update_layout(
        title=dict(
            text="AI Fingerprint Map — CIFAKE Visual Clusters",
            font=dict(size=20, color="#e0e0ff"),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title="UMAP-1",
            showgrid=False,
            zeroline=False,
            range=x_range,
            constrain="domain",
        ),
        yaxis=dict(
            title="UMAP-2",
            showgrid=False,
            zeroline=False,
            range=y_range,
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
        ),
        plot_bgcolor="#0d0d1a",
        paper_bgcolor="#0d0d1a",
        font=dict(color="#e0e0ff"),
        legend=dict(
            title=dict(
                text="Clusters<br><span style='font-size:10px'>○ REAL-dominant · ● AI-dominant</span>",
            ),
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(0,0,0,0.6)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1,
            font=dict(size=11),
        ),
        autosize=True,
        width=None,
        height=700,
        margin=dict(l=60, r=200, t=80, b=60),
    )
    fig.add_annotation(
        text=f"Discovered {n_fake_clusters} distinct AI fingerprint clusters",
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.02,
        showarrow=False,
        font=dict(size=13, color="#a0a0ff"),
        bgcolor="rgba(0,0,0,0.4)",
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(out),
        include_plotlyjs="cdn",
        full_html=True,
        config={"responsive": True, "displayModeBar": True, "scrollZoom": True},
    )
    print(f"Saved interactive visualization: {out}")
    return fig


def build_umap_figure(max_points: int = 15000):
    """Load pipeline outputs and return the UMAP scatter figure."""
    umap_2d = np.load(OUT_DIR / "umap_2d.npy")
    cluster_labels = np.load(OUT_DIR / "cluster_assignments_kmeans.npy")
    true_labels = np.load(EMB_DIR / "train_labels.npy")
    paths = np.load(EMB_DIR / "train_paths.npy")

    with open(OUT_DIR / "fingerprint_report.json", encoding="utf-8") as f:
        fingerprint_report = json.load(f)

    min_len = min(len(umap_2d), len(cluster_labels), len(true_labels), len(paths))
    return build_plotly_scatter(
        umap_2d[:min_len],
        cluster_labels[:min_len],
        true_labels[:min_len],
        paths[:min_len],
        fingerprint_report,
        max_points=max_points,
        output_path=str(OUT_DIR / "visualization.html"),
    )


if __name__ == "__main__":
    build_umap_figure(max_points=15000)

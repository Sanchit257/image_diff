"""Streamlit demo for AI Fingerprint Tracker."""
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image

from src.utils import EMB_DIR, OUT_DIR
from src.visualize import build_umap_figure

st.set_page_config(
    page_title="AI Fingerprint Tracker",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    .main { background-color: #0a0a1a; color: #e0e0ff; }
    .stApp { background-color: #0a0a1a; }
    h1, h2, h3 { color: #7b9fff; }
    .fingerprint-tag {
        display: inline-block;
        background: #2d2060;
        border: 1px solid #7b9fff;
        border-radius: 20px;
        padding: 4px 14px;
        margin: 4px;
        font-size: 0.85em;
        color: #b0c4ff;
    }
    .verdict-real { color: #4ade80; font-size: 1.5em; font-weight: bold; }
    .verdict-fake { color: #f87171; font-size: 1.5em; font-weight: bold; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    return model, tokenizer


@st.cache_data
def get_umap_landscape_figure():
    return build_umap_figure(max_points=15000)


@st.cache_data
def load_cluster_data():
    embs = np.load(EMB_DIR / "train_embeddings.npy")
    cluster_labels = np.load(OUT_DIR / "cluster_assignments_kmeans.npy")
    umap_2d = np.load(OUT_DIR / "umap_2d.npy")
    paths = np.load(EMB_DIR / "train_paths.npy")
    true_labels = np.load(EMB_DIR / "train_labels.npy")

    with open(OUT_DIR / "fingerprint_report.json", encoding="utf-8") as f:
        fingerprint_report = json.load(f)

    min_len = min(len(embs), len(cluster_labels), len(umap_2d), len(paths), len(true_labels))
    return {
        "embs": embs[:min_len],
        "cluster_labels": cluster_labels[:min_len],
        "umap_2d": umap_2d[:min_len],
        "paths": paths[:min_len],
        "true_labels": true_labels[:min_len],
        "report": fingerprint_report,
    }


@st.cache_data
def compute_cluster_centroids(embs, cluster_labels):
    centroids = {}
    for c in set(cluster_labels):
        if c == -1:
            continue
        mask = cluster_labels == c
        centroids[int(c)] = embs[mask].mean(axis=0)
    return centroids


def get_clip_embedding(model, image: Image.Image) -> np.ndarray:
    from torchvision import transforms as T

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ])
    device = next(model.parameters()).device
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().squeeze()


def assign_cluster(emb: np.ndarray, centroids: dict) -> tuple[int, float]:
    best_c, best_sim = -1, -1.0
    for c, centroid in centroids.items():
        sim = float(np.dot(emb, centroid / np.linalg.norm(centroid)))
        if sim > best_sim:
            best_sim, best_c = sim, c
    return best_c, best_sim


def compute_fft(image: Image.Image) -> np.ndarray:
    gray = np.array(image.convert("L").resize((32, 32))) / 255.0
    fft = np.fft.fftshift(np.fft.fft2(gray))
    return np.log1p(np.abs(fft))


st.markdown("# 🔍 AI Fingerprint Tracker")
st.markdown(
    "*Upload an image to discover its visual fingerprint and see how it compares "
    "to known AI generation patterns.*"
)
st.divider()

with st.spinner("Loading models and cluster data..."):
    model, _tokenizer = load_models()
    data = load_cluster_data()
    centroids = compute_cluster_centroids(data["embs"], data["cluster_labels"])

left_col, right_col = st.columns([1, 2])

image = None
with left_col:
    st.subheader("Upload an Image")
    uploaded = st.file_uploader(
        "JPEG, PNG, or WEBP",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    use_example = st.checkbox("Or use a random example from dataset")
    if use_example and not uploaded:
        sample_path = random.choice(data["paths"].tolist())
        image = Image.open(sample_path).convert("RGB")
        paths_list = data["paths"].tolist()
        true_label = data["true_labels"][paths_list.index(sample_path)]
        st.image(
            image,
            caption=f"Random example ({'REAL' if true_label == 0 else 'AI-GENERATED'})",
            use_container_width=True,
        )
    elif uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

    if image is not None and st.button("🔍 Analyze Fingerprint", type="primary", use_container_width=True):
        with st.spinner("Extracting visual fingerprint..."):
            emb = get_clip_embedding(model, image)
            cluster_id, confidence = assign_cluster(emb, centroids)
            cluster_info = data["report"].get(str(cluster_id), data["report"].get(int(cluster_id), {}))
            st.session_state["result"] = {
                "emb": emb,
                "cluster_id": cluster_id,
                "confidence": confidence,
                "info": cluster_info,
                "image": image,
            }

with right_col:
    if "result" in st.session_state:
        res = st.session_state["result"]
        cluster_id = res["cluster_id"]
        info = res["info"]
        dominant = info.get("dominant", "UNKNOWN")
        fake_pct = info.get("fake_pct", 0.5)

        if dominant == "FAKE":
            st.markdown(
                '<div class="verdict-fake">⚠️ AI-GENERATED FINGERPRINT DETECTED</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="verdict-real">✅ NATURAL IMAGE FINGERPRINT</div>',
                unsafe_allow_html=True,
            )

        m1, m2, m3 = st.columns(3)
        m1.metric("Cluster", f"#{cluster_id}")
        m2.metric("AI Probability", f"{fake_pct:.0%}")
        m3.metric("Confidence", f"{res['confidence']:.3f}")

        clip_desc = info.get("clip_descriptions", [])
        if clip_desc:
            st.subheader("Visual Fingerprint Tags")
            tags_html = " ".join(
                f'<span class="fingerprint-tag">{desc} ({score:.2f})</span>'
                for desc, score in clip_desc[:5]
            )
            st.markdown(tags_html, unsafe_allow_html=True)

        st.divider()

        tab1, tab2, tab3 = st.tabs(["📊 UMAP Position", "🌀 Frequency Analysis", "🖼 Similar Images"])

        with tab1:
            umap_2d = data["umap_2d"]
            cluster_labels = data["cluster_labels"]

            fig = go.Figure()
            colors = px.colors.qualitative.Set3
            sample_idx = np.random.choice(len(umap_2d), min(5000, len(umap_2d)), replace=False)

            for c in sorted(set(cluster_labels)):
                if c == -1:
                    continue
                mask = cluster_labels[sample_idx] == c
                c_info = data["report"].get(str(c), data["report"].get(int(c), {}))
                c_dominant = c_info.get("dominant", "?")
                fig.add_trace(
                    go.Scattergl(
                        x=umap_2d[sample_idx][mask, 0],
                        y=umap_2d[sample_idx][mask, 1],
                        mode="markers",
                        marker=dict(color=colors[int(c) % len(colors)], size=3, opacity=0.4),
                        name=f"C{c} ({c_dominant})",
                    )
                )

            centroid_2d = umap_2d[cluster_labels == cluster_id].mean(axis=0)
            fig.add_trace(
                go.Scatter(
                    x=[centroid_2d[0]],
                    y=[centroid_2d[1]],
                    mode="markers",
                    marker=dict(
                        color="white",
                        size=18,
                        symbol="star",
                        line=dict(color="yellow", width=2),
                    ),
                    name="Your Image",
                )
            )

            fig.update_layout(
                title="Your image in the fingerprint landscape",
                plot_bgcolor="#0d0d1a",
                paper_bgcolor="#0d0d1a",
                font=dict(color="#e0e0ff"),
                height=400,
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False),
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fft_user = compute_fft(res["image"])
            fft_cluster_path = OUT_DIR / f"fft_profiles/cluster_{cluster_id:02d}_fft.png"

            col_a, col_b = st.columns(2)
            with col_a:
                fig2, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(fft_user, cmap="inferno")
                ax.set_title("Your Image — FFT", color="white")
                ax.axis("off")
                fig2.patch.set_facecolor("#0a0a1a")
                st.pyplot(fig2)
                plt.close(fig2)
            with col_b:
                if fft_cluster_path.exists():
                    st.image(fft_cluster_path, caption=f"Cluster {cluster_id} avg FFT")
                    if dominant == "FAKE":
                        st.caption("⚠️ Grid/ring patterns in the FFT = AI generation artifact")

        with tab3:
            cluster_paths = data["paths"][data["cluster_labels"] == cluster_id]
            sample_paths = np.random.choice(
                cluster_paths, min(6, len(cluster_paths)), replace=False
            )
            cols = st.columns(3)
            for i, p in enumerate(sample_paths):
                try:
                    img = Image.open(p).convert("RGB").resize((64, 64), Image.NEAREST)
                    cols[i % 3].image(img, caption=Path(p).parent.name, use_container_width=True)
                except OSError:
                    pass
    else:
        st.subheader("The AI Fingerprint Landscape")
        st.markdown(
            """
            This map shows thousands of images plotted by their visual fingerprints.
            **Clusters** represent distinct patterns — some are natural photography,
            others reveal specific AI generation artifacts like spectral ringing,
            checkerboard upsampling, or unnaturally smooth gradients.
            """
        )
        if (OUT_DIR / "umap_2d.npy").exists():
            st.plotly_chart(
                get_umap_landscape_figure(),
                use_container_width=True,
                config={"displayModeBar": True, "scrollZoom": True, "responsive": True},
            )
        else:
            st.info("Run `python src/visualize.py` to generate the interactive UMAP")

with st.sidebar:
    st.subheader("Cluster Explorer")
    report = data["report"]
    for cid, info in sorted(report.items(), key=lambda x: int(x[0])):
        if not isinstance(info, dict):
            continue
        dominant = info.get("dominant", "?")
        icon = "🤖" if dominant == "FAKE" else "📷"
        size = info.get("size", 0)
        purity = max(info.get("fake_pct", 0), info.get("real_pct", 0))
        st.markdown(f"{icon} **C{cid}** — {dominant} ({size:,} imgs, {purity:.0%} pure)")

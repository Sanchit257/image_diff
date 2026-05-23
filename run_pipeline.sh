#!/bin/bash
# Run full pipeline end to end
set -e

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)"
PYTHON="${PYTHON:-.venv/bin/python}"

echo "=== Step 1: Validate dataset ==="
$PYTHON src/preprocess.py

echo "=== Step 2: Extract embeddings (train + test) ==="
$PYTHON src/extract_embeddings.py

echo "=== Step 3: Cluster ==="
$PYTHON src/cluster.py

echo "=== Step 4: Interpret fingerprints ==="
$PYTHON src/interpret.py

echo "=== Step 5: Build visualization ==="
$PYTHON src/visualize.py

echo "=== Run all unit tests ==="
$PYTHON -m pytest tests/ -v --tb=short

echo "=== Pipeline complete ==="
echo "Launch demo: streamlit run app.py"

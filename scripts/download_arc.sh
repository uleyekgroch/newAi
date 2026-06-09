#!/bin/bash
# Download ARC-AGI public dataset for benchmark testing
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$REPO_DIR/data"

echo "Downloading ARC-AGI dataset..."

if [ -d "$DATA_DIR/arc_training" ] && [ "$(ls -1 "$DATA_DIR/arc_training"/*.json 2>/dev/null | wc -l)" -ge 400 ]; then
    echo "ARC training data already exists ($(ls -1 "$DATA_DIR/arc_training"/*.json | wc -l) files)"
else
    TMP_DIR=$(mktemp -d)
    git clone --depth 1 https://github.com/fchollet/ARC-AGI.git "$TMP_DIR"
    mkdir -p "$DATA_DIR/arc_training" "$DATA_DIR/arc_evaluation"
    cp "$TMP_DIR/data/training/"*.json "$DATA_DIR/arc_training/"
    cp "$TMP_DIR/data/evaluation/"*.json "$DATA_DIR/arc_evaluation/"
    rm -rf "$TMP_DIR"
    echo "Downloaded $(ls -1 "$DATA_DIR/arc_training"/*.json | wc -l) training tasks"
    echo "Downloaded $(ls -1 "$DATA_DIR/arc_evaluation"/*.json | wc -l) evaluation tasks"
fi

echo "Done. Data is in $DATA_DIR/"

#!/usr/bin/env bash
# scripts/snapshot_and_upload.sh
# Create Qdrant snapshots for all arXiv collections and upload to HF
set -euo pipefail

QDRANT_URL="http://localhost:6333"
QDRANT_API_KEY="simple-api-key"
SNAPSHOT_DIR="$HOME/RAG/snapshots"
HF_REPO="MARKYMARK55/spark-scholar-arxiv-snapshots"
HF_CLI="$HOME/venv/bin/huggingface-cli"

mkdir -p "$SNAPSHOT_DIR"

COLLECTIONS=(
    "arXiv"
    "arxiv-astro"
    "arxiv-condmat"
    "arxiv-cs-cv"
    "arxiv-cs-ml-ai"
    "arxiv-cs-nlp-ir"
    "arxiv-cs-systems-theory"
    "arxiv-hep"
    "arxiv-math-applied"
    "arxiv-math-phys"
    "arxiv-math-pure"
    "arxiv-misc"
    "arxiv-nucl-nlin-physother"
    "arxiv-qbio-qfin-econ"
    "arxiv-quantph-grqc"
    "arxiv-stat-eess"
)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Creating snapshots for ${#COLLECTIONS[@]} collections"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for col in "${COLLECTIONS[@]}"; do
    echo ""
    echo "→ [$col] Getting point count..."
    count=$(curl -s -H "api-key: $QDRANT_API_KEY" "$QDRANT_URL/collections/$col" \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['result']['points_count'])")
    echo "  Points: $count"

    echo "  Creating snapshot..."
    snapshot_name=$(curl -s -X POST -H "api-key: $QDRANT_API_KEY" \
        "$QDRANT_URL/collections/$col/snapshots" \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['result']['name'])")
    echo "  Snapshot: $snapshot_name"

    echo "  Downloading..."
    curl -s -H "api-key: $QDRANT_API_KEY" \
        "$QDRANT_URL/collections/$col/snapshots/$snapshot_name" \
        -o "$SNAPSHOT_DIR/${col}.snapshot"

    size=$(du -h "$SNAPSHOT_DIR/${col}.snapshot" | cut -f1)
    echo "  Downloaded: $size"

    # Generate checksum
    sha256sum "$SNAPSHOT_DIR/${col}.snapshot" > "$SNAPSHOT_DIR/${col}.snapshot.sha256"
    echo "  Checksum: $(cat "$SNAPSHOT_DIR/${col}.snapshot.sha256" | cut -d' ' -f1)"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  All snapshots created. Summary:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ls -lhS "$SNAPSHOT_DIR"/*.snapshot
echo ""
total=$(du -sh "$SNAPSHOT_DIR" | cut -f1)
echo "Total: $total"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Uploading to HuggingFace: $HF_REPO"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for col in "${COLLECTIONS[@]}"; do
    echo "→ Uploading ${col}.snapshot..."
    $HF_CLI upload "$HF_REPO" \
        "$SNAPSHOT_DIR/${col}.snapshot" \
        "snapshots/${col}.snapshot" \
        --repo-type dataset
    echo "  ✓ Done"
done

# Upload checksums
echo "→ Uploading checksums..."
cat "$SNAPSHOT_DIR"/*.sha256 > "$SNAPSHOT_DIR/checksums.sha256"
$HF_CLI upload "$HF_REPO" \
    "$SNAPSHOT_DIR/checksums.sha256" \
    "checksums.sha256" \
    --repo-type dataset
echo "  ✓ Done"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Upload complete!"
echo "  https://huggingface.co/datasets/$HF_REPO"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

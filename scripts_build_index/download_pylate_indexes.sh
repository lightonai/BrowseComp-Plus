#!/bin/bash
# Download PyLate indexes from HuggingFace
# Usage: ./download_pylate_indexes.sh [index_name]
# Without arguments, lists available indexes.
# Example: ./download_pylate_indexes.sh LateOn-distilled

REPO="lightonai/browsecomp-plus-indexes"

INDEXES=(
  "GTE-ModernColBERT-v1"
  "Reason-ModernColBERT"
)

if [ -z "$1" ]; then
  echo "Usage: $0 <index_name|all>"
  echo ""
  echo "Available indexes:"
  for idx in "${INDEXES[@]}"; do
    echo "  $idx"
  done
  echo ""
  echo "  all  (download all indexes)"
  exit 1
fi

if [ "$1" = "all" ]; then
  for idx in "${INDEXES[@]}"; do
    echo "Downloading $idx..."
    huggingface-cli download "$REPO" --repo-type=dataset --include="pylate/$idx/*" --local-dir ./indexes
  done
else
  echo "Downloading $1..."
  huggingface-cli download "$REPO" --repo-type=dataset --include="pylate/$1/*" --local-dir ./indexes
fi

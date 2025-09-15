#!/usr/bin/env bash
# Pull only the model files stored in Git LFS so users don't need to download the whole repo's LFS history.
# Usage:
#   bash scripts/download_models.sh                    # pulls defaults
#   bash scripts/download_models.sh "pattern1,pattern2"  # custom LFS include patterns

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

INC_PATTERNS=${1:-"runs/retina_rfs_balanced1/*,runs/detr_debug500_rfsAlbu/*"}

if ! command -v git >/dev/null 2>&1; then
  echo "[error] git not found in PATH" >&2
  exit 1
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[error] This script must be run inside a git repository" >&2
  exit 1
fi

# Ensure git-lfs is available
if ! git lfs version >/dev/null 2>&1; then
  echo "[error] git-lfs is not installed. Install it then run: git lfs install" >&2
  echo "  Ubuntu/Debian: sudo apt-get install -y git-lfs" >&2
  echo "  macOS (brew):  brew install git-lfs" >&2
  exit 1
fi

echo "[info] Enabling Git LFS smudge/clean filters"
git lfs install --force

echo "[info] Pulling LFS objects for patterns: $INC_PATTERNS"
git lfs pull --include "$INC_PATTERNS" --exclude "" || {
  echo "[warn] git lfs pull failed. Do you have access to the remote and the correct branch checked out?" >&2
  exit 1
}

echo "[ok] LFS models pulled. Listing sizes for convenience:"
echo "--- runs/retina_rfs_balanced1 ---"; ls -lh runs/retina_rfs_balanced1 2>/dev/null || true
echo "--- runs/detr_debug500_rfsAlbu ---"; ls -lh runs/detr_debug500_rfsAlbu 2>/dev/null || true


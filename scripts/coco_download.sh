#!/usr/bin/env bash
# COCO 2017 downloader: annotations + images (train/val/test)
# Usage:
#   ./coco_download.sh /absolute/or/relative/path/to/coco
#   ./coco_download.sh /path/to/coco --images train val
#   ./coco_download.sh /path/to/coco --all
#   ./coco_download.sh /path/to/coco --ann-only
#
# Safe defaults:
#   - Downloads annotations only (to avoid huge accidental downloads)
# Notes:
#   - Resumable downloads with curl (-C -)
#   - Skips unzip if target folder already exists
#   - Works whether you run it from project root or scripts/ (use a correct DEST)

set -euo pipefail

DEST="${1:-/data/coco}"
shift || true

# --- Resolve DEST to absolute path and ensure it exists ---
mkdir -p "$DEST"
ABS_DEST="$(cd "$DEST" && pwd -P)"

# ---------------- Defaults (safe) ----------------
DO_ANN=1
REQ_IMAGES=()  # none by default
# -------------------------------------------------

# --------------- Parse flags --------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      REQ_IMAGES=("train" "val" "test")
      shift
      ;;
    --ann-only)
      DO_ANN=1
      REQ_IMAGES=()
      shift
      ;;
    --images)
      REQ_IMAGES=()
      shift
      while [[ $# -gt 0 ]]; do
        case "$1" in
          train|val|test) REQ_IMAGES+=("$1"); shift;;
          *) break;;
        esac
      done
      ;;
    *)
      # ignore unknown args to be flexible
      shift
      ;;
  esac
done
# ------------------------------------------------

# Work inside the destination directory from here on
cd "$ABS_DEST"

download_zip() {
  local url="$1" zip="$2" outdir="$3"
  if [[ -d "$outdir" ]]; then
    echo "[skip] $outdir already exists."
    return 0
  fi
  echo "[info] Downloading: $url"
  curl -L -C - -o "$zip" "$url"
  echo "[info] Unzipping: $zip"
  unzip -q "$zip"
  if [[ ! -d "$outdir" ]]; then
    echo "[warn] Expected directory '$outdir' after unzip. Contents are:"
    ls -la
  fi
}

post_check_counts() {
  echo "---------- Post-download file counts ----------"
  [[ -d "train2017"     ]] && echo "train2017  jpg: $(find "train2017" -maxdepth 1 -type f -name '*.jpg' | wc -l)"
  [[ -d "val2017"       ]] && echo "val2017    jpg: $(find "val2017"   -maxdepth 1 -type f -name '*.jpg' | wc -l)"
  [[ -d "test2017"      ]] && echo "test2017   jpg: $(find "test2017"  -maxdepth 1 -type f -name '*.jpg' | wc -l)"
  [[ -d "annotations"   ]] && echo "annotations json: $(find "annotations" -type f -name '*.json' | wc -l)"
  echo "-----------------------------------------------"
}

# ---------------- Annotations -------------------
if [[ "$DO_ANN" -eq 1 ]]; then
  ANN_ZIP="annotations_trainval2017.zip"
  ANN_URL="http://images.cocodataset.org/annotations/${ANN_ZIP}"

  if [[ ! -d "annotations" ]]; then
    echo "[info] Downloading annotations..."
    curl -L -C - -o "$ANN_ZIP" "$ANN_URL"
    # Unzip into current directory so result is $ABS_DEST/annotations/*.json
    unzip -q "$ANN_ZIP" -d "."
  else
    echo "[skip] annotations already present."
  fi
fi
# ------------------------------------------------

# ------------------ Images ----------------------
for split in "${REQ_IMAGES[@]}"; do
  case "$split" in
    train)
      download_zip \
        "http://images.cocodataset.org/zips/train2017.zip" \
        "train2017.zip" \
        "train2017"
      ;;
    val)
      download_zip \
        "http://images.cocodataset.org/zips/val2017.zip" \
        "val2017.zip" \
        "val2017"
      ;;
    test)
      download_zip \
        "http://images.cocodataset.org/zips/test2017.zip" \
        "test2017.zip" \
        "test2017"
      ;;
    *)
      echo "[warn] Unknown split '$split' (valid: train, val, test). Skipping."
      ;;
  esac
done
# ------------------------------------------------

post_check_counts

echo "[ok] COCO assets ready under: $ABS_DEST"
echo "[note] Typical paths:"
echo "  Images:       $ABS_DEST/train2017, $ABS_DEST/val2017, $ABS_DEST/test2017"
echo "  Annotations:  $ABS_DEST/annotations/*.json"

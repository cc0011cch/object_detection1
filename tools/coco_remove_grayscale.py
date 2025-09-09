#!/usr/bin/env python3
"""
Remove grayscale images from a COCO dataset and update annotations.

A grayscale image is detected if:
- Its PIL mode is inherently grayscale (e.g., 'L', 'LA', 'I;16', '1'), OR
- After resizing to a small RGB thumbnail, all channels are equal for every pixel.

Usage:
  python coco_remove_grayscale.py \
      --images-dir /path/to/train2017 \
      --ann-in /path/to/annotations/instances_train2017_subset.json \
      --ann-out /path/to/annotations/instances_train2017_subset_nogray.json \
      --list-out /path/to/removed_grayscale_image_ids.txt
"""
import argparse
import json
import os
from typing import Dict, Set
from PIL import Image
import numpy as np

GRAYSCALE_MODES = {"1", "L", "LA", "I", "I;16", "F"}

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def is_grayscale_image(img_path: str, thumb_size: int = 128) -> bool:
    """
    Fast, reasonably robust grayscale check:
    - If PIL mode implies grayscale → True
    - Else convert to RGB, downscale, and test if R==G==B for all pixels
    """
    try:
        with Image.open(img_path) as im:
            if im.mode in GRAYSCALE_MODES:
                return True
            # Convert to RGB to normalize modes like 'P', 'CMYK', etc.
            im = im.convert("RGB")
            # Downsize for speed; this is sufficient to detect grayscale
            im = im.resize((thumb_size, thumb_size))
            arr = np.asarray(im, dtype=np.uint8)  # (H, W, 3)
            # Check if all channels equal for all pixels
            r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
            return np.all((r == g) & (g == b))
    except Exception as e:
        # If unreadable, treat as grayscale to be safe (and to avoid training crashes)
        print(f"[warn] Failed to open {img_path}: {e}. Marking as grayscale/removing.")
        return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True, help="Folder with COCO images (e.g., train2017)")
    ap.add_argument("--ann-in", required=True, help="Path to COCO annotations JSON (subset)")
    ap.add_argument("--ann-out", required=True, help="Output JSON path (grayscale removed)")
    ap.add_argument("--list-out", default=None, help="Optional path to save list of removed image IDs")
    ap.add_argument("--thumb-size", type=int, default=128, help="Resize for grayscale check (default: 128)")
    args = ap.parse_args()

    data = load_json(args.ann_in)

    # Build image_id -> file_name map
    id2filename: Dict[int, str] = {im["id"]: im["file_name"] for im in data["images"]}

    # Detect grayscale images
    to_remove: Set[int] = set()
    for img in data["images"]:
        img_id = img["id"]
        fname = img["file_name"]
        path = os.path.join(args.images_dir, fname)
        if not os.path.exists(path):
            print(f"[warn] Missing file on disk: {path}. Removing from dataset.")
            to_remove.add(img_id)
            continue
        if is_grayscale_image(path, thumb_size=args.thumb_size):
            to_remove.add(img_id)

    if args.list_out:
        with open(args.list_out, "w", encoding="utf-8") as f:
            for iid in sorted(to_remove):
                f.write(f"{iid}\n")
        print(f"Saved removed image IDs to {args.list_out}")

    # Filter images and annotations
    kept_images = [im for im in data["images"] if im["id"] not in to_remove]
    kept_img_ids = {im["id"] for im in kept_images}
    kept_annotations = [a for a in data["annotations"] if a["image_id"] in kept_img_ids]

    # (Optional but sensible) remove images that end up with 0 annotations
    img_has_ann = {a["image_id"] for a in kept_annotations}
    final_images = [im for im in kept_images if im["id"] in img_has_ann]
    final_img_ids = {im["id"] for im in final_images}
    final_annotations = [a for a in kept_annotations if a["image_id"] in final_img_ids]

    out = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "images": final_images,
        "annotations": final_annotations,
        "categories": data["categories"],  # unchanged
    }

    save_json(out, args.ann_out)
    print(f"Removed grayscale images: {len(to_remove)}")
    print(f"Saved updated annotations to {args.ann_out}")
    print(f"Final Images: {len(final_images)} | Annotations: {len(final_annotations)} | Categories: {len(data['categories'])}")

if __name__ == "__main__":
    main()

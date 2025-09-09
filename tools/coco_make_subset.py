#!/usr/bin/env python3
"""
Create a smaller COCO subset:
- Keep only specified categories (by name).
- Drop images with no remaining annotations.
- (Optional) remap category_ids to contiguous IDs starting from 1.

Usage:
  python coco_make_subset.py \
      --ann-in /path/to/annotations/instances_train2017.json \
      --cat-names person car bus \
      --ann-out /path/to/annotations/instances_train2017_subset.json \
      --remap-category-ids
"""
import argparse
import json
from typing import List, Dict, Set

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def build_category_maps(categories: List[Dict]) -> Dict[str, Dict]:
    """Map category name -> category dict."""
    return {c["name"]: c for c in categories}

def filter_categories(categories: List[Dict], keep_names: Set[str]) -> List[Dict]:
    return [c for c in categories if c["name"] in keep_names]

def remap_category_ids(categories: List[Dict]) -> Dict[int, int]:
    """Return mapping old_id -> new_id (1..N), preserving category order."""
    sorted_cats = sorted(categories, key=lambda c: c["id"])
    return {c["id"]: i+1 for i, c in enumerate(sorted_cats)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-in", required=True, help="Path to COCO instances_*.json")
    ap.add_argument("--ann-out", required=True, help="Output JSON path")
    ap.add_argument("--cat-names", nargs="+", required=True,
                    help="Category names to keep, e.g., person car bus")
    ap.add_argument("--remap-category-ids", action="store_true",
                    help="Remap kept categories to contiguous IDs (1..N)")
    args = ap.parse_args()

    data = load_json(args.ann_in)

    # 1) Pick categories by name
    name2cat = build_category_maps(data["categories"])
    missing = [n for n in args.cat_names if n not in name2cat]
    if missing:
        raise ValueError(f"Category names not found in file: {missing}")

    kept_cats = filter_categories(data["categories"], set(args.cat_names))
    kept_cat_ids = {c["id"] for c in kept_cats}

    # 2) Filter annotations to the kept categories
    anns = [a for a in data["annotations"] if a["category_id"] in kept_cat_ids]
    kept_img_ids = {a["image_id"] for a in anns}

    # 3) Keep only images that still have at least one annotation
    imgs = [im for im in data["images"] if im["id"] in kept_img_ids]

    # 4) Optionally remap category IDs to contiguous integers starting at 1
    if args.remap_category_ids:
        id_map = remap_category_ids(kept_cats)
        # Update category IDs in categories and annotations
        new_categories = []
        for c in kept_cats:
            c2 = dict(c)
            c2["id"] = id_map[c["id"]]
            new_categories.append(c2)
        for a in anns:
            a["category_id"] = id_map[a["category_id"]]
        kept_cats = new_categories

    # 5) Construct new JSON
    out = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "images": imgs,
        "annotations": anns,
        "categories": kept_cats,
    }

    save_json(out, args.ann_out)
    print(f"Saved subset to {args.ann_out}")
    print(f"Images: {len(imgs)} | Annotations: {len(anns)} | Categories: {len(kept_cats)}")

    # --- NEW: Print category summary ---
    print("\n[Category summary]")
    print(f"Total categories: {len(kept_cats)}")
    for c in kept_cats:
        print(f"{c['id']}: {c['name']}")

if __name__ == "__main__":
    main()

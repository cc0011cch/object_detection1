#!/usr/bin/env python3
"""
Split a COCO subset into train/eval with per-class 90%/10% by *annotation counts*,
ensuring images are non-overlapping between splits. Also duplicate the provided
val subset into a test subset JSON.

Inputs:
  --train-json  ./data/instances_train2017_subset_nogray.json
  --val-json    ./data/instances_val2017_subset_nogray.json

Outputs:
  --out-train   ./data/instances_train2017_split_train.json
  --out-eval    ./data/instances_train2017_split_eval.json
  --out-test    ./data/instances_test2017_subset_nogray.json   (copy of val)

Strategy:
  - Compute total annotations per category from the input train JSON.
  - Target train annotations per category = ceil(0.9 * total).
  - Build per-image *contribution vectors* (how many anns of each class the image contains).
  - Greedy covering:
      While any class is below its target, pick the *eval* image that maximizes
      total "benefit" (how many needed anns it adds) with minimal overflow.
      Move it to the train split and update counts.
  - Remaining images stay in eval split.

Notes:
  - Reproducible via --seed (default 1337).
  - Images are never duplicated between train/eval.
  - Categories and licenses/info are carried over unchanged.
"""
import argparse
import json
import math
import os
import random
from collections import defaultdict, Counter
from copy import deepcopy
from typing import Dict, List, Tuple, Set


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_indices(data: dict):
    images = data["images"]
    anns = data["annotations"]
    cats = data["categories"]

    imgid_to_img = {im["id"]: im for im in images}
    imgid_to_anns: Dict[int, List[dict]] = defaultdict(list)
    for a in anns:
        imgid_to_anns[a["image_id"]].append(a)

    cat_ids = [c["id"] for c in cats]
    catid_to_name = {c["id"]: c["name"] for c in cats}
    return images, anns, cats, imgid_to_img, imgid_to_anns, cat_ids, catid_to_name


def count_total_annotations_per_cat(anns: List[dict]) -> Counter:
    cnt = Counter()
    for a in anns:
        cnt[a["category_id"]] += 1
    return cnt


def per_image_contrib(imgid_to_anns: Dict[int, List[dict]], cat_ids: List[int]) -> Dict[int, Counter]:
    contrib = {}
    for img_id, alist in imgid_to_anns.items():
        c = Counter()
        for a in alist:
            c[a["category_id"]] += 1
        contrib[img_id] = c
    return contrib


def compute_benefit_and_overflow(
    img_counter: Counter, current: Counter, target: Dict[int, int], cat_ids: List[int]
) -> Tuple[int, int]:
    """Return (benefit, overflow) for adding this image to train."""
    benefit = 0
    overflow = 0
    for cid in cat_ids:
        need = max(0, target[cid] - current[cid])
        add = img_counter.get(cid, 0)
        take = min(add, need)
        benefit += take
        # overflow if we add more than needed
        extra = max(0, (current[cid] + add) - target[cid])
        overflow += extra
    return benefit, overflow


def greedy_split(
    images: List[dict],
    anns: List[dict],
    cats: List[dict],
    imgid_to_anns: Dict[int, List[dict]],
    cat_ids: List[int],
    train_target_ratio: float = 0.9,
    seed: int = 1337,
) -> Tuple[Set[int], Set[int], Counter, Counter, Dict[int, int]]:
    """
    Returns:
      train_img_ids, eval_img_ids, final_train_counts, target_counts, missed_by_cat
    """
    rng = random.Random(seed)

    # Total and target per category
    total_by_cat = count_total_annotations_per_cat(anns)
    target_by_cat = {cid: int(math.ceil(total_by_cat[cid] * train_target_ratio)) for cid in cat_ids}

    # Start with all images in eval; train empty
    all_img_ids = [im["id"] for im in images]
    rng.shuffle(all_img_ids)

    train_img_ids: Set[int] = set()
    eval_img_ids: Set[int] = set(all_img_ids)

    # Precompute per-image contributions
    contrib = per_image_contrib(imgid_to_anns, cat_ids)

    # Current train counts
    current = Counter()

    # While any class is below target, move best eval image to train
    def any_below_target() -> bool:
        return any(current[cid] < target_by_cat[cid] for cid in cat_ids)

    # Build a quick list of candidate eval images that have at least one annotation
    def has_any_ann(img_id: int) -> bool:
        return any(cid in contrib[img_id] for cid in cat_ids)

    # Greedy covering
    iterations = 0
    max_iters = len(all_img_ids) * 3  # safety
    while any_below_target() and iterations < max_iters:
        iterations += 1

        # Score each eval image by (benefit, overflow) given current counts
        best_img = None
        best_score = (-1, float("inf"))  # higher benefit, lower overflow is better
        for img_id in list(eval_img_ids):
            if not has_any_ann(img_id):
                continue
            b, o = compute_benefit_and_overflow(contrib[img_id], current, target_by_cat, cat_ids)
            if b <= 0 and o > 0:
                continue  # doesn’t help and overflows; skip
            # tie-break with ratio benefit/(1+overflow), then absolute benefit, then smaller overflow
            ratio = b / (1 + o)
            cur_score = (ratio, b, -o, rng.random())  # add randomness to break ties
            if best_img is None or cur_score > best_score:
                best_img = img_id
                best_score = cur_score

        # If no candidate improves, allow minimal overflow to progress
        if best_img is None:
            # pick an eval image that contributes most to the most-missing class
            # find most missing class
            missing_cid = max(cat_ids, key=lambda cid: target_by_cat[cid] - current[cid])
            candidates = [(img_id, contrib[img_id].get(missing_cid, 0)) for img_id in eval_img_ids]
            candidates = [x for x in candidates if x[1] > 0]
            if not candidates:
                break
            candidates.sort(key=lambda x: (x[1], rng.random()), reverse=True)
            best_img = candidates[0][0]

        # Move to train
        eval_img_ids.remove(best_img)
        train_img_ids.add(best_img)
        current.update(contrib[best_img])

    # Report unmet targets (if any)
    missed_by_cat = {cid: max(0, target_by_cat[cid] - current[cid]) for cid in cat_ids}
    return train_img_ids, eval_img_ids, current, Counter(target_by_cat), missed_by_cat


def make_split_json(data: dict, keep_img_ids: Set[int]) -> dict:
    keep_img_ids = set(keep_img_ids)
    images = [im for im in data["images"] if im["id"] in keep_img_ids]
    img_id_set = {im["id"] for im in images}
    anns = [a for a in data["annotations"] if a["image_id"] in img_id_set]
    out = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "images": images,
        "annotations": anns,
        "categories": data["categories"],
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-json", required=True, help="instances_train2017_subset_nogray.json")
    ap.add_argument("--val-json", required=True, help="instances_val2017_subset_nogray.json")
    ap.add_argument("--out-train", default="./data/instances_train2017_split_train.json")
    ap.add_argument("--out-eval",  default="./data/instances_train2017_split_eval.json")
    ap.add_argument("--out-test",  default="./data/instances_test2017_subset_nogray.json")
    ap.add_argument("--ratio", type=float, default=0.9, help="Train ratio by annotation counts per class (default 0.9)")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    # 1) Copy val -> test
    val_data = load_json(args.val_json)
    save_json(val_data, args.out_test)
    print(f"[ok] Wrote test JSON (copied from val): {args.out_test} "
          f"(images={len(val_data['images'])}, anns={len(val_data['annotations'])})")

    # 2) Split train into train/eval by annotation counts per class, non-overlapping images
    tr_data = load_json(args.train_json)
    images, anns, cats, imgid_to_img, imgid_to_anns, cat_ids, catid_to_name = build_indices(tr_data)

    train_img_ids, eval_img_ids, final_train_counts, target_counts, missed = greedy_split(
        images, anns, cats, imgid_to_anns, cat_ids, train_target_ratio=args.ratio, seed=args.seed
    )

    # Persist split JSONs
    split_train = make_split_json(tr_data, train_img_ids)
    split_eval  = make_split_json(tr_data, eval_img_ids)
    save_json(split_train, args.out_train)
    save_json(split_eval,  args.out_eval)

    # Stats
    total_by_cat = count_total_annotations_per_cat(anns)
    eval_counts = count_total_annotations_per_cat(split_eval["annotations"])

    print("\n[summary] Category-wise annotation counts:")
    header = f"{'cat_id':>6}  {'name':<20}  {'total':>6}  {'target_train(>=)':>14}  {'actual_train':>12}  {'eval':>6}  {'missed':>6}"
    print(header)
    print("-" * len(header))
    for cid in sorted(cat_ids):
        name = catid_to_name[cid]
        tot = total_by_cat[cid]
        target = target_counts[cid]
        actual_train = final_train_counts[cid]
        ev = eval_counts[cid]
        miss = missed[cid]
        print(f"{cid:6d}  {name:<20}  {tot:6d}  {target:14d}  {actual_train:12d}  {ev:6d}  {miss:6d}")

    print(f"\n[images] train={len(split_train['images'])}  eval={len(split_eval['images'])}")
    print(f"[anns]   train={len(split_train['annotations'])}  eval={len(split_eval['annotations'])}")
    print(f"[files]  train_json={args.out_train}\n        eval_json={args.out_eval}\n        test_json={args.out_test}")


if __name__ == "__main__":
    main()

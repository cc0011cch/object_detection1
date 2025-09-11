import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import List
import logging

import numpy as np


def _rfs_cache_path(train_ann_path: str, threshold: float, alpha: float) -> Path:
    h = hashlib.md5(f"{train_ann_path}|{threshold}|{alpha}".encode("utf-8")).hexdigest()[:10]
    return Path(train_ann_path).with_suffix(f".rfs_t{threshold}_a{alpha}_{h}.npy")


def compute_repeat_factors_fast(train_ann_path: str,
                                img_ids_dataset_order: List[int],
                                threshold: float,
                                alpha: float = 0.5) -> List[float]:
    """Fast RFS using a single JSON pass; caches to .npy next to the ann file."""
    logger = logging.getLogger("train")
    cache = _rfs_cache_path(train_ann_path, threshold, alpha)
    if cache.exists():
        rf = np.load(cache)
        if len(rf) == len(img_ids_dataset_order):
            logger.info(f"[rfs] loaded cached repeat-factors: {cache.name}")
            return rf.tolist()
        else:
            logger.info("[rfs] cache length mismatch; recomputing...")

    with open(train_ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    N_images = len(coco["images"])

    imgs_with_cat = defaultdict(set)    # cat_id -> set(image_id)
    cats_in_img   = defaultdict(set)    # image_id -> set(cat_id)
    for a in coco["annotations"]:
        img_id = a["image_id"]
        cat_id = a["category_id"]
        imgs_with_cat[cat_id].add(img_id)
        cats_in_img[img_id].add(cat_id)

    f_c = {c: (len(imgs_with_cat[c]) / max(1, N_images)) for c in imgs_with_cat.keys()}

    r_c = {}
    for c, f in f_c.items():
        if f >= threshold:
            r_c[c] = 1.0
        else:
            r_c[c] = (threshold / max(f, 1e-12)) ** alpha

    img_id_to_r = {}
    for im in coco["images"]:
        img_id = im["id"]
        img_cats = cats_in_img.get(img_id, set())
        if not img_cats:
            img_id_to_r[img_id] = 1.0
        else:
            img_id_to_r[img_id] = max(r_c.get(c, 1.0) for c in img_cats)

    rf = np.array([img_id_to_r.get(int(img_id), 1.0) for img_id in img_ids_dataset_order], dtype=np.float32)

    try:
        np.save(cache, rf)
        logger.info(f"[rfs] cached repeat-factors -> {cache.name} (len={len(rf)})")
    except Exception as e:
        logger.info(f"[rfs] cache save failed ({e}); continuing without cache.")
    return rf.tolist()


def build_imgid_list_for_dataset(ds) -> List[int]:
    """Try to obtain per-index image_id for the dataset fast."""
    if hasattr(ds, "images"):
        try:
            return [int(im["id"]) for im in ds.images]
        except Exception:
            pass

    for attr in ("img_ids", "image_ids", "ids"):
        if hasattr(ds, attr):
            li = list(getattr(ds, attr))
            return [int(x) for x in li]

    img_ids = []
    for i in range(len(ds)):
        _, t = ds[i]
        img_ids.append(int(t["image_id"]))
    return img_ids


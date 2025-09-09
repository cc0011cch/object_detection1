#!/usr/bin/env python3
"""
Create a small debug COCO train set from a full train split.
- Select ~30 images per class.
- Non-overlapping images in output JSON.
- Prints per-class annotation counts in the debug set.
"""

import argparse, json, os, random
from collections import defaultdict, Counter

def load_json(p): return json.load(open(p,"r",encoding="utf-8"))
def save_json(o,p):
    os.makedirs(os.path.dirname(p),exist_ok=True)
    json.dump(o,open(p,"w",encoding="utf-8"),ensure_ascii=False,indent=2)

def make_debug(train_json, out_json, samples_per_class=30, seed=42):
    random.seed(seed)
    data = load_json(train_json)
    anns = data["annotations"]
    imgs = {im["id"]:im for im in data["images"]}
    cats = data["categories"]

    # Build indexes
    anns_by_cat = defaultdict(list)
    anns_by_img = defaultdict(list)
    for a in anns:
        anns_by_cat[a["category_id"]].append(a)
        anns_by_img[a["image_id"]].append(a)

    selected_imgs = set()
    for c in cats:
        cid = c["id"]
        # all images containing this category
        img_ids = list({a["image_id"] for a in anns_by_cat[cid]})
        random.shuffle(img_ids)
        chosen = 0
        for img_id in img_ids:
            selected_imgs.add(img_id)
            chosen += 1
            if chosen >= samples_per_class:
                break

    # Build debug JSON
    images = [imgs[i] for i in selected_imgs]
    annos = [a for a in anns if a["image_id"] in selected_imgs]
    out = {"info":data.get("info",{}),
           "licenses":data.get("licenses",[]),
           "images":images,"annotations":annos,"categories":cats}
    save_json(out,out_json)

    # Summary
    counts = Counter([a["category_id"] for a in annos])
    print(f"[ok] Debug set saved: {out_json}")
    print(f"Images: {len(images)}  Annotations: {len(annos)}")
    print(f"{'cat_id':>6} {'name':<20} {'anns':>6}")
    print("-"*35)
    for c in cats:
        cid=c["id"]
        print(f"{cid:6d} {c['name']:<20} {counts[cid]:6d}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--train-json",required=True,help="Full train JSON (e.g., instances_train2017_split_train.json)")
    ap.add_argument("--out-json",default="./data/instances_train2017_debug.json")
    ap.add_argument("--samples",type=int,default=30,help="Images per class (default 30)")
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()
    make_debug(args.train_json,args.out_json,args.samples,args.seed)

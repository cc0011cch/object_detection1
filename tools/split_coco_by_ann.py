#!/usr/bin/env python3
"""
Fast COCO train/eval split by annotation counts.
- Ensures images are non-overlapping.
- For each class, ~10% of its annotations are assigned to eval (via images).
- Remaining images -> train.
- Also copies val JSON -> test JSON.
- Prints per-class summary of annotation counts in train/eval/total.
"""

import argparse, json, os, random
from collections import defaultdict, Counter

def load_json(p): return json.load(open(p,"r",encoding="utf-8"))
def save_json(o,p):
    os.makedirs(os.path.dirname(p),exist_ok=True)
    json.dump(o,open(p,"w",encoding="utf-8"),ensure_ascii=False,indent=2)

def split(train_json, val_json, out_train, out_eval, out_test, ratio=0.9, seed=1337):
    random.seed(seed)

    # Copy val -> test
    val = load_json(val_json)
    save_json(val,out_test)
    print(f"[ok] test json written: {out_test} "
          f"(images={len(val['images'])}, anns={len(val['annotations'])})")

    data = load_json(train_json)
    anns = data["annotations"]
    imgs = {im["id"]:im for im in data["images"]}
    cats = data["categories"]

    # Build ann index
    anns_by_cat = defaultdict(list)
    anns_by_img = defaultdict(list)
    for a in anns:
        anns_by_cat[a["category_id"]].append(a)
        anns_by_img[a["image_id"]].append(a)

    total_per_cat = {c["id"]:len(anns_by_cat[c["id"]]) for c in cats}
    eval_target_per_cat = {cid:int(total_per_cat[cid]*(1-ratio)+0.5) for cid in total_per_cat}

    eval_imgs = set()
    counts_eval = Counter()

    # For each class, assign images to eval until ~10% annotations are covered
    for cid, annlist in anns_by_cat.items():
        target = eval_target_per_cat[cid]
        if target == 0: 
            continue
        # Shuffle images containing this class
        img_ids = list({a["image_id"] for a in annlist})
        random.shuffle(img_ids)
        acc = 0
        for img_id in img_ids:
            if img_id in eval_imgs:
                continue
            n_in_img = sum(1 for a in anns_by_img[img_id] if a["category_id"]==cid)
            if acc >= target:
                break
            eval_imgs.add(img_id)
            acc += n_in_img
        counts_eval[cid] = acc

    train_imgs = set(imgs.keys()) - eval_imgs

    # Build output JSONs
    def make_json(img_ids):
        img_ids=set(img_ids)
        images=[imgs[i] for i in img_ids]
        annos=[a for a in anns if a["image_id"] in img_ids]
        return {"info":data.get("info",{}),
                "licenses":data.get("licenses",[]),
                "images":images,"annotations":annos,"categories":cats}

    train_json_out = make_json(train_imgs)
    eval_json_out = make_json(eval_imgs)
    save_json(train_json_out, out_train)
    save_json(eval_json_out, out_eval)

    # ---- Summary ----
    train_counts = Counter([a["category_id"] for a in train_json_out["annotations"]])
    eval_counts  = Counter([a["category_id"] for a in eval_json_out["annotations"]])
    print(f"[ok] train={len(train_imgs)} imgs, eval={len(eval_imgs)} imgs")
    print(f"{'cat_id':>6} {'name':<20} {'total':>6} {'train':>8} {'eval':>8} {'eval_target':>12}")
    print("-"*65)
    for c in cats:
        cid=c["id"]
        name=c["name"]
        tot=total_per_cat[cid]
        trn=train_counts[cid]
        evl=eval_counts[cid]
        tgt=eval_target_per_cat[cid]
        print(f"{cid:6d} {name:<20} {tot:6d} {trn:8d} {evl:8d} {tgt:12d}")

    print(f"[files]\n train={out_train}\n eval ={out_eval}\n test ={out_test}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--train-json",required=True)
    ap.add_argument("--val-json",required=True)
    ap.add_argument("--out-train",default="./data/train_split.json")
    ap.add_argument("--out-eval",default="./data/eval_split.json")
    ap.add_argument("--out-test",default="./data/test.json")
    ap.add_argument("--ratio",type=float,default=0.9,help="Train ratio (default 0.9 -> eval=0.1)")
    ap.add_argument("--seed",type=int,default=1337)
    args=ap.parse_args()
    split(args.train_json,args.val_json,args.out_train,args.out_eval,args.out_test,args.ratio,args.seed)

#!/usr/bin/env python3
"""
Evaluate best model on a COCO-style test subset and plot per-class PR curves.

Requirements:
  pip install pycocotools matplotlib

Usage example:
  python evaluate_test.py \
    --model retinanet \
    --ckpt runs/retina_rfs001/best.pth \
    --test-ann ./data/coco/annotations_used/instances_test2017_subset_nogray.json \
    --test-images ./data/coco/val2017 \
    --batch-size 8 --num-workers 4 \
    --resize-short 512 \
    --pr-plot runs/retina_rfs001/pr_curves_iou50.jpg
"""
import argparse
import io
import os
from pathlib import Path
from typing import Dict, List, Any

import torch
from torch.utils.data import DataLoader

# --- external deps ---
import matplotlib
matplotlib.use("Agg")  # for headless environments
import matplotlib.pyplot as plt
import numpy as np

# --- your project modules ---
from train import build_model_and_helpers, collate_fn  # re-use your helpers
from dataset import CocoDetDataset                    # your dataset.py

# COCO eval
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


@torch.no_grad()
def run_inference(
    model_name: str,
    model,
    dl,
    ds,
    device: torch.device,
    predict_batch_fn,  # may be None for DETR (we’ll build inside)
) -> List[Dict[str, Any]]:
    """
    Produce COCO-format detections for the entire dataloader.
    """
    model.eval()
    detections = []
    cat_ids_sorted = sorted([c["id"] for c in ds.categories])

    if model_name == "detr" and predict_batch_fn is None:
        # Build a minimal DETR predict closure (same as train.py logic)
        from transformers import DetrImageProcessor
        proc = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

        def predict_batch_fn(images: List[torch.Tensor], img_ids: List[int]):
            np_imgs = [(img.permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()) for img in images]
            enc = proc(images=np_imgs, return_tensors="pt")
            enc["pixel_values"] = enc["pixel_values"].to(device)
            if "pixel_mask" in enc:
                enc["pixel_mask"] = enc["pixel_mask"].to(device)
            outputs = model(**enc)
            sizes = [(ds.imgid_to_img[int(i)]["height"], ds.imgid_to_img[int(i)]["width"]) for i in img_ids]
            processed = proc.post_process_object_detection(outputs, target_sizes=torch.tensor(sizes, device=device))
            results = []
            for img_id, p in zip(img_ids, processed):
                boxes = p["boxes"].detach().cpu().numpy()
                scores = p["scores"].detach().cpu().numpy()
                labels = p["labels"].detach().cpu().numpy()
                for (x1, y1, x2, y2), s, l in zip(boxes, scores, labels):
                    cat_id = cat_ids_sorted[int(l)]
                    results.append({
                        "image_id": int(img_id),
                        "category_id": int(cat_id),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(s),
                    })
            return results

    # Iterate loader
    for images, targets in dl:
        img_ids = [int(t["image_id"].item()) for t in targets]
        batch_dets = predict_batch_fn(images, img_ids)
        detections.extend(batch_dets)

    return detections


def summarize_and_per_class_ap(coco_gt: COCO, detections: List[Dict[str, Any]], iou_type="bbox"):
    """
    Run COCOeval + compute per-class AP (macro) and per-class AP50.
    Returns dict with overall stats and per-class table, and a COCOeval object.
    """
    coco_dt = coco_gt.loadRes(detections) if len(detections) > 0 else None
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()

    # Capture standard table
    buf = io.StringIO()
    coco_eval.summarize()  # prints to stdout
    table_text = buf.getvalue() if buf.getvalue() else ""  # sometimes summarize prints directly

    # Overall stats
    stats = coco_eval.stats  # [AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl]
    overall = {
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "APS": float(stats[3]),
        "APM": float(stats[4]),
        "APL": float(stats[5]),
        "AR1": float(stats[6]),
        "AR10": float(stats[7]),
        "AR100": float(stats[8]),
        "ARS": float(stats[9]),
        "ARM": float(stats[10]),
        "ARL": float(stats[11]),
    }

    # ----- Per-class AP (macro fairness for long-tail) -----
    # coco_eval.eval['precision'] has shape [T, R, K, A, M]
    # where T=IoU thresholds, R=recall thresholds, K=categories, A=areas, M=maxDets
    precisions = coco_eval.eval["precision"]  # (T, R, K, A, M)
    if precisions is None:
        raise RuntimeError("COCOeval precision array is None — did evaluation run?")
    T, R, K, A, M = precisions.shape

    # Index choices: area='all' => a=0; maxDets=100 => m=2 in standard COCO setup
    a = 0
    m = 2 if M >= 3 else M - 1

    # List in category id order used by COCOeval (params.catIds)
    cat_ids_eval_order = coco_eval.params.catIds
    id_to_name = {c["id"]: c["name"] for c in coco_gt.loadCats(coco_gt.getCatIds())}

    # Macro AP across IoU thresholds & recall for each class
    per_class_ap = {}
    for k_idx, cat_id in enumerate(cat_ids_eval_order):
        pr = precisions[:, :, k_idx, a, m]  # (T, R)
        pr = pr[pr > -1]                    # valid entries only
        ap_k = float(np.mean(pr)) if pr.size > 0 else float("nan")
        per_class_ap[cat_id] = ap_k

    macro_mAP = float(np.nanmean(list(per_class_ap.values()))) if per_class_ap else float("nan")

    # ----- Per-class AP50 (IoU==0.50) -----
    iou_thrs = coco_eval.params.iouThrs  # (T,)
    # find closest index to 0.50
    iou_target = 0.50
    t_50 = int(np.argmin(np.abs(iou_thrs - iou_target)))

    per_class_ap50 = {}
    for k_idx, cat_id in enumerate(cat_ids_eval_order):
        pr = precisions[t_50, :, k_idx, a, m]  # (R,)
        pr = pr[pr > -1]
        ap50_k = float(np.mean(pr)) if pr.size > 0 else float("nan")
        per_class_ap50[cat_id] = ap50_k

    return {
        "overall": overall,
        "macro_mAP": macro_mAP,
        "per_class_ap": {id_to_name[cid]: per_class_ap[cid] for cid in per_class_ap},
        "per_class_ap50": {id_to_name[cid]: per_class_ap50[cid] for cid in per_class_ap50},
        "cat_eval_order": [id_to_name[cid] for cid in cat_ids_eval_order],
        "table_text": table_text,
        "coco_eval": coco_eval,
    }


def plot_pr_curves_iou50(coco_eval: COCOeval, out_path: str):
    """
    Plot per-class PR curves at IoU=0.50, area=all, maxDets=100.
    """
    precisions = coco_eval.eval["precision"]  # T x R x K x A x M
    if precisions is None:
        raise RuntimeError("COCOeval precision array is None — cannot plot PR curves.")

    iou_thrs = coco_eval.params.iouThrs  # (T,)
    rec_thrs = coco_eval.params.recThrs  # (R,)
    cat_ids = coco_eval.params.catIds
    cat_names = {c["id"]: c["name"] for c in coco_eval.cocoGt.loadCats(cat_ids)}

    # indices for area='all' (0) and maxDets=100 (2)
    a = 0
    M = precisions.shape[-1]
    m = 2 if M >= 3 else M - 1

    # select IoU = 0.50
    t_50 = int(np.argmin(np.abs(iou_thrs - 0.50)))

    plt.figure(figsize=(8, 6))
    for k_idx, cat_id in enumerate(cat_ids):
        pr = precisions[t_50, :, k_idx, a, m]  # (R,)
        valid = pr > -1
        if not np.any(valid):
            continue
        plt.plot(rec_thrs[valid], pr[valid], label=f"{cat_names[cat_id]}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Per-class PR Curves @ IoU=0.50 (area=all, maxDets=100)")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="lower left")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["retinanet", "detr"], required=True)
    ap.add_argument("--ckpt", required=True, help="Path to best.pth")
    ap.add_argument("--test-ann", required=True, help="COCO JSON for test subset")
    ap.add_argument("--test-images", required=True, help="Folder with images (val2017 for your subset)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--resize-short", type=int, default=640)
    ap.add_argument("--pr-plot", default="runs/pr_curves_iou50.jpg")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset (no aug)
    ds = CocoDetDataset(
        images_dir=args.test_images,
        ann_json=args.test_ann,
        model_family=args.model,
        augment=False,
        use_albu=False,
    )
    if hasattr(ds, "set_target_size"):
        try:
            ds.set_target_size(args.resize_short)
        except Exception:
            pass

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        prefetch_factor=(2 if args.num_workers > 0 else None),
        persistent_workers=(args.num_workers > 0),
    )

    # Build model with correct num_classes / label maps
    cat_ids_sorted = sorted([c["id"] for c in ds.categories])
    id2label_0based = getattr(ds, "id2label_0based", {})
    label2id_0based = getattr(ds, "label2id_0based", {})
    num_classes = ds.num_classes

    model, _, _, predict_batch_fn = build_model_and_helpers(
        model_name=args.model,
        num_classes=num_classes,
        id2label_0based=id2label_0based,
        label2id_0based=label2id_0based,
        cat_ids_sorted=cat_ids_sorted,
        device=device,
    )

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Inference → COCO eval
    detections = run_inference(
        model_name=args.model,
        model=model,
        dl=dl,
        ds=ds,
        device=device,
        predict_batch_fn=predict_batch_fn,
    )

    coco_gt = COCO(args.test_ann)
    results = summarize_and_per_class_ap(coco_gt, detections, iou_type="bbox")

    # Print a compact summary (overall + fair macro + per-class)
    print("\n==== Overall COCO (bbox) ====")
    for k, v in results["overall"].items():
        print(f"{k:>6}: {v:.4f}")
    print(f"\nMacro-mAP (per-class mean AP@[.5:.95]): {results['macro_mAP']:.4f}")

    print("\nPer-class AP@[.5:.95]:")
    for name in results["cat_eval_order"]:
        ap = results["per_class_ap"].get(name, float("nan"))
        print(f"  {name:>10}: {ap:.4f}")

    print("\nPer-class AP50 (IoU=0.50):")
    for name in results["cat_eval_order"]:
        ap50 = results["per_class_ap50"].get(name, float("nan"))
        print(f"  {name:>10}: {ap50:.4f}")

    # Plot PR curves @ IoU=0.50
    plot_pr_curves_iou50(results["coco_eval"], args.pr_plot)
    print(f"\n[OK] Saved PR curves (IoU=0.50) to: {args.pr_plot}")


if __name__ == "__main__":
    main()

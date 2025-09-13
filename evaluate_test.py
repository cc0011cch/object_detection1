#!/usr/bin/env python3
"""
Evaluate best model on a COCO-style test subset and plot per-class PR curves.

Supports:
  - PyTorch (default)
  - ONNXRuntime (with ONNXRetinaNetWrapper)

Usage example (PyTorch):
  python evaluate_test.py \
    --backend torch --model retinanet \
    --ckpt runs/retina_rfs001/best.pth \
    --test-ann ./data/coco/annotations_used/instances_test2017_subset_nogray.json \
    --test-images ./data/coco/val2017 \
    --batch-size 8 --num-workers 4 \
    --resize-short 512 \
    --pr-plot runs/retina_rfs001/pr_curves_iou50.jpg

Usage example (ONNX):
  python evaluate_test.py \
    --backend onnx --onnx retinanet_head.onnx \
    --test-ann ./data/coco/annotations_used/instances_test2017_subset_nogray.json \
    --test-images ./data/coco/val2017 \
    --batch-size 8 --num-workers 4 \
    --resize-short 512 \
    --pr-plot runs/retina_rfs001/pr_curves_iou50.jpg
"""
import argparse
import io
from pathlib import Path
from typing import Dict, List, Any

import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# project modules
from train import build_model_and_helpers, collate_fn
from dataset import CocoDetDataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ONNX wrapper
from onnx_retinanet_wrapper import ONNXRetinaNetWrapper
from onnx_detr_wrapper import ONNXDetrWrapper


@torch.no_grad()
def run_inference_torch(model_name, model, dl, ds, device, predict_batch_fn):
    model.eval()
    detections = []
    cat_ids_sorted = sorted([c["id"] for c in ds.categories])

    if model_name == "detr" and predict_batch_fn is None:
        from transformers import DetrImageProcessor
        proc = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

        def predict_batch_fn(images, img_ids):
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

    for images, targets in dl:
        img_ids = [int(t["image_id"].item()) for t in targets]
        batch_dets = predict_batch_fn(images, img_ids)
        detections.extend(batch_dets)

    return detections


def _xywh_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]


def _xyxy_to_xywh(b):
    x1, y1, x2, y2 = b
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def _nms_merge_per_image(dets: List[Dict[str, Any]], ow: int, oh: int, iou_thresh: float = 0.6):
    try:
        from torchvision.ops import nms
    except Exception:
        nms = None

    merged = []
    # Group by category
    by_cat: Dict[int, List[Dict[str, Any]]] = {}
    for d in dets:
        by_cat.setdefault(int(d["category_id"]), []).append(d)

    for cat_id, items in by_cat.items():
        if not items:
            continue
        boxes_xyxy = torch.tensor([_xywh_to_xyxy(it["bbox"]) for it in items], dtype=torch.float32)
        scores = torch.tensor([float(it.get("score", 0.0)) for it in items], dtype=torch.float32)
        if nms is not None and boxes_xyxy.numel() and scores.numel():
            keep = nms(boxes_xyxy, scores, float(iou_thresh))
            keep_idx = keep.cpu().tolist()
        else:
            # Fallback: keep all
            keep_idx = list(range(len(items)))
        for i in keep_idx:
            b = boxes_xyxy[i].tolist()
            merged.append({
                "image_id": int(items[i]["image_id"]),
                "category_id": int(cat_id),
                "bbox": _xyxy_to_xywh(b),
                "score": float(items[i].get("score", 0.0)),
            })
    return merged


def summarize_and_per_class_ap(coco_gt: COCO, detections: List[Dict[str, Any]], iou_type="bbox"):
    coco_dt = coco_gt.loadRes(detections) if len(detections) > 0 else None
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    overall = {
        "AP": float(stats[0]), "AP50": float(stats[1]), "AP75": float(stats[2]),
        "APS": float(stats[3]), "APM": float(stats[4]), "APL": float(stats[5]),
        "AR1": float(stats[6]), "AR10": float(stats[7]), "AR100": float(stats[8]),
        "ARS": float(stats[9]), "ARM": float(stats[10]), "ARL": float(stats[11]),
    }

    precisions = coco_eval.eval["precision"]
    if precisions is None:
        raise RuntimeError("COCOeval precision array is None — did evaluation run?")
    a, m = 0, 2 if precisions.shape[-1] >= 3 else precisions.shape[-1] - 1
    cat_ids_eval_order = coco_eval.params.catIds
    id_to_name = {c["id"]: c["name"] for c in coco_gt.loadCats(coco_gt.getCatIds())}

    per_class_ap = {}
    for k_idx, cat_id in enumerate(cat_ids_eval_order):
        pr = precisions[:, :, k_idx, a, m]
        pr = pr[pr > -1]
        ap_k = float(np.mean(pr)) if pr.size > 0 else float("nan")
        per_class_ap[cat_id] = ap_k
    macro_mAP = float(np.nanmean(list(per_class_ap.values()))) if per_class_ap else float("nan")

    iou_thrs = coco_eval.params.iouThrs
    t_50 = int(np.argmin(np.abs(iou_thrs - 0.50)))
    per_class_ap50 = {}
    for k_idx, cat_id in enumerate(cat_ids_eval_order):
        pr = precisions[t_50, :, k_idx, a, m]
        pr = pr[pr > -1]
        ap50_k = float(np.mean(pr)) if pr.size > 0 else float("nan")
        per_class_ap50[cat_id] = ap50_k

    return {
        "overall": overall, "macro_mAP": macro_mAP,
        "per_class_ap": {id_to_name[cid]: per_class_ap[cid] for cid in per_class_ap},
        "per_class_ap50": {id_to_name[cid]: per_class_ap50[cid] for cid in per_class_ap50},
        "cat_eval_order": [id_to_name[cid] for cid in cat_ids_eval_order],
        "coco_eval": coco_eval,
    }


def plot_pr_curves_iou50(coco_eval: COCOeval, out_path: str):
    precisions = coco_eval.eval["precision"]
    if precisions is None:
        raise RuntimeError("COCOeval precision array is None — cannot plot PR curves.")

    iou_thrs, rec_thrs = coco_eval.params.iouThrs, coco_eval.params.recThrs
    cat_ids = coco_eval.params.catIds
    cat_names = {c["id"]: c["name"] for c in coco_eval.cocoGt.loadCats(cat_ids)}
    a, m = 0, 2 if precisions.shape[-1] >= 3 else precisions.shape[-1] - 1
    t_50 = int(np.argmin(np.abs(iou_thrs - 0.50)))

    plt.figure(figsize=(8, 6))
    for k_idx, cat_id in enumerate(cat_ids):
        pr = precisions[t_50, :, k_idx, a, m]
        valid = pr > -1
        if not np.any(valid):
            continue
        plt.plot(rec_thrs[valid], pr[valid], label=f"{cat_names[cat_id]}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Per-class PR Curves @ IoU=0.50")
    plt.xlim([0, 1]); plt.ylim([0, 1])
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="lower left")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["torch", "onnx"], default="torch")
    ap.add_argument("--model", choices=["retinanet", "detr"], help="For torch backend only")
    ap.add_argument("--ckpt", help="Path to best.pth (torch backend)")
    ap.add_argument("--onnx", help="Path to .onnx file (onnx backend)")
    ap.add_argument("--test-ann", required=True)
    ap.add_argument("--test-images", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--resize-short", type=int, default=640)
    ap.add_argument("--pr-plot", default="runs/pr_curves_iou50.jpg")
    # TTA controls (evaluate-only)
    ap.add_argument("--tta-hflip", action="store_true", help="Enable horizontal flip TTA and NMS merge")
    ap.add_argument("--tta-nms-iou", type=float, default=0.6, help="NMS IoU for merging TTA outputs")
    ap.add_argument(
        "--tta-multiscales",
        default="",
        help="Comma-separated list of square sizes (e.g., '512,640,800') to TTA and merge",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = CocoDetDataset(
        images_dir=args.test_images,
        ann_json=args.test_ann,
        augment=False, use_albu=False,
    )
    if hasattr(ds, "set_target_size"):
        try: ds.set_target_size(args.resize_short)
        except Exception: pass

    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=False, prefetch_factor=(2 if args.num_workers > 0 else None),
        persistent_workers=(args.num_workers > 0),
    )

    cat_ids_sorted = sorted([c["id"] for c in ds.categories])
    # Parse TTA scales
    tta_scales: List[int] = []
    if args.tta_multiscales:
        try:
            tta_scales = [int(s.strip()) for s in args.tta_multiscales.split(",") if s.strip()]
        except Exception:
            tta_scales = []

    # Backend switch
    if args.backend == "torch":
        # Prepare original size map for proper box scaling
        orig_size_map = {int(im["id"]): (int(im["height"]), int(im["width"])) for im in ds.images}

        model, _, _, predict_batch_fn = build_model_and_helpers(
            model_name=args.model, num_classes=ds.num_classes,
            id2label_0based=getattr(ds, "id2label_0based", {}),
            label2id_0based=getattr(ds, "label2id_0based", {}),
            cat_ids_sorted=cat_ids_sorted, device=device,
            orig_size_map=orig_size_map,
            detr_short=800, detr_max=800,
        )
        ckpt = torch.load(args.ckpt, map_location=device)
        try:
            model.load_state_dict(ckpt["model"], assign=True)
        except TypeError:
            model.load_state_dict(ckpt["model"])
        model.eval()
        if not args.tta_hflip and not tta_scales:
            detections = run_inference_torch(args.model, model, dl, ds, device, predict_batch_fn)
        else:
            # Torch backend with optional hflip + multiscale
            def infer_one_dl(local_dl):
                out = []
                ow_map = {int(im["id"]): int(im["width"]) for im in ds.images}
                for images, targets in local_dl:
                    img_ids = [int(t["image_id"].item()) for t in targets]
                    dets_base = predict_batch_fn(images, img_ids)
                    dets_all = dets_base
                    if args.tta_hflip:
                        images_flip = [torch.flip(img, dims=[2]) for img in images]
                        dets_flip = predict_batch_fn(images_flip, img_ids)
                        dets_flip_u = []
                        for d in dets_flip:
                            ow = ow_map[int(d["image_id"])]
                            x, y, w, h = d["bbox"]
                            x_new = max(0.0, float(ow) - float(x) - float(w))
                            d2 = dict(d); d2["bbox"] = [x_new, y, w, h]
                            dets_flip_u.append(d2)
                        dets_all = dets_base + dets_flip_u
                    by_img: Dict[int, List[Dict[str, Any]]] = {}
                    for d in dets_all:
                        by_img.setdefault(int(d["image_id"]), []).append(d)
                    for iid, lst in by_img.items():
                        ow = ow_map[int(iid)]
                        oh = int(ds.imgid_to_img[int(iid)]["height"])
                        merged = _nms_merge_per_image(lst, ow=ow, oh=oh, iou_thresh=float(args.tta_nms_iou))
                        out.extend(merged)
                return out

            # Base scale (current dl)
            merged_all = infer_one_dl(dl)
            # Extra multiscale passes
            if tta_scales:
                from dataset import CocoDetDataset
                # Build per-scale DLs by cloning dataset with target_size
                for sz in tta_scales:
                    ds_s = CocoDetDataset(ds.images_dir, ds.ann_json, augment=False, use_albu=False)
                    try: ds_s.set_target_size(int(sz))
                    except Exception: pass
                    dl_s = DataLoader(
                        ds_s, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_fn,
                        pin_memory=False, prefetch_factor=(2 if args.num_workers > 0 else None),
                        persistent_workers=(args.num_workers > 0),
                    )
                    merged_all.extend(infer_one_dl(dl_s))
            # Final merge across scales per image
            by_img_all: Dict[int, List[Dict[str, Any]]] = {}
            for d in merged_all:
                by_img_all.setdefault(int(d["image_id"]), []).append(d)
            detections = []
            for iid, lst in by_img_all.items():
                ow = int(ds.imgid_to_img[int(iid)]["width"]) ; oh = int(ds.imgid_to_img[int(iid)]["height"]) 
                detections.extend(_nms_merge_per_image(lst, ow=ow, oh=oh, iou_thresh=float(args.tta_nms_iou)))

    else:  # ONNX backend
        detections = []
        if args.model == "retinanet":
            onnx_model = ONNXRetinaNetWrapper(
                onnx_path=args.onnx,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                class_ids=cat_ids_sorted,
                score_thresh=0.3, iou_thresh=0.5,
            )
            for images, targets in dl:
                img_ids = [int(t["image_id"].item()) for t in targets]
                dets = onnx_model.predict(images, img_ids)
                # Scale boxes back to original size
                sizes_in = {int(iid): tuple(images[idx].shape[-2:]) for idx, iid in enumerate(img_ids)}
                scaled = []
                for det in dets:
                    iid = int(det["image_id"])
                    oh = int(ds.imgid_to_img[iid]["height"]) ; ow = int(ds.imgid_to_img[iid]["width"]) 
                    ih, iw = sizes_in.get(iid, (oh, ow))
                    s = min(ih / max(1, oh), iw / max(1, ow))
                    new_h = int(round(oh * s)); new_w = int(round(ow * s))
                    x, y, w, h = det["bbox"]
                    x1, y1, x2, y2 = x, y, x + w, y + h
                    x1 = max(0.0, min(x1, new_w - 1)); y1 = max(0.0, min(y1, new_h - 1))
                    x2 = max(0.0, min(x2, new_w - 1)); y2 = max(0.0, min(y2, new_h - 1))
                    if s > 0:
                        x1 /= s; y1 /= s; x2 /= s; y2 /= s
                    det = dict(det)
                    det["bbox"] = [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))]
                    scaled.append(det)
                detections.extend(scaled)
        else:  # DETR ONNX
            # Map image_id -> (H,W) originals for postprocess
            orig_size_map = {int(im["id"]): (int(im["height"]), int(im["width"])) for im in ds.images}
            onnx_model = ONNXDetrWrapper(
                onnx_path=args.onnx,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                class_ids=cat_ids_sorted,
                score_thresh=0.05, detr_short=800, detr_max=800,
                orig_size_map=orig_size_map,
            )
            if not args.tta_hflip and not tta_scales:
                for images, targets in dl:
                    img_ids = [int(t["image_id"].item()) for t in targets]
                    dets = onnx_model.predict(images, img_ids)
                    detections.extend(dets)
            else:
                detections = []
                ow_map = {int(im["id"]): int(im["width"]) for im in ds.images}
                def collect_for_wrapper(wrapper):
                    out = []
                    for images, targets in dl:
                        img_ids = [int(t["image_id"].item()) for t in targets]
                        dets_base = wrapper.predict(images, img_ids)
                        dets_all = dets_base
                        if args.tta_hflip:
                            images_flip = [torch.flip(img, dims=[2]) for img in images]
                            dets_flip = wrapper.predict(images_flip, img_ids)
                            dets_flip_u = []
                            for d in dets_flip:
                                ow = ow_map[int(d["image_id"])]
                                x, y, w, h = d["bbox"]
                                x_new = max(0.0, float(ow) - float(x) - float(w))
                                d2 = dict(d); d2["bbox"] = [x_new, y, w, h]
                                dets_flip_u.append(d2)
                            dets_all = dets_base + dets_flip_u
                        by_img: Dict[int, List[Dict[str, Any]]] = {}
                        for d in dets_all:
                            by_img.setdefault(int(d["image_id"]), []).append(d)
                        for iid, lst in by_img.items():
                            ow = ow_map[int(iid)]
                            oh = int(ds.imgid_to_img[int(iid)]["height"])
                            out.extend(_nms_merge_per_image(lst, ow=ow, oh=oh, iou_thresh=float(args.tta_nms_iou)))
                    return out

                # Base size
                merged_all = collect_for_wrapper(onnx_model)
                # Additional multiscale sizes
                if tta_scales:
                    for sz in tta_scales:
                        onnx_model_s = ONNXDetrWrapper(
                            onnx_path=args.onnx,
                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                            class_ids=cat_ids_sorted,
                            score_thresh=0.05, detr_short=int(sz), detr_max=int(sz),
                            orig_size_map=orig_size_map,
                        )
                        merged_all.extend(collect_for_wrapper(onnx_model_s))
                # Final merge across scales
                by_img_all: Dict[int, List[Dict[str, Any]]] = {}
                for d in merged_all:
                    by_img_all.setdefault(int(d["image_id"]), []).append(d)
                for iid, lst in by_img_all.items():
                    ow = int(ds.imgid_to_img[int(iid)]["width"]) ; oh = int(ds.imgid_to_img[int(iid)]["height"]) 
                    detections.extend(_nms_merge_per_image(lst, ow=ow, oh=oh, iou_thresh=float(args.tta_nms_iou)))

    coco_gt = COCO(args.test_ann)
    results = summarize_and_per_class_ap(coco_gt, detections)

    print("\n==== Overall COCO (bbox) ====")
    for k, v in results["overall"].items():
        print(f"{k:>6}: {v:.4f}")
    print(f"\nMacro-mAP (per-class mean AP@[.5:.95]): {results['macro_mAP']:.4f}")

    print("\nPer-class AP@[.5:.95]:")
    for name in results["cat_eval_order"]:
        print(f"  {name:>10}: {results['per_class_ap'].get(name, float('nan')):.4f}")

    print("\nPer-class AP50 (IoU=0.50):")
    for name in results["cat_eval_order"]:
        print(f"  {name:>10}: {results['per_class_ap50'].get(name, float('nan')):.4f}")

    plot_pr_curves_iou50(results["coco_eval"], args.pr_plot)
    print(f"\n[OK] Saved PR curves (IoU=0.50) to: {args.pr_plot}")


if __name__ == "__main__":
    main()

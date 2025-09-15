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
from typing import Dict, List, Any, Optional, Callable

import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

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


def _resize_images_to_square(images: List[torch.Tensor], target: int) -> List[torch.Tensor]:
    """Resize each CHW tensor so that max(H,W)=target, then pad to target x target.
    Keeps value range and dtype; uses bilinear for resize and zero-pad bottom/right.
    """
    if target is None or int(target) <= 0:
        return images
    out: List[torch.Tensor] = []
    for img in images:
        c, h, w = img.shape
        if max(h, w) == target:
            # still ensure square pad if needed
            new_h, new_w = h, w
            resized = img
        else:
            s = float(target) / float(max(h, w))
            new_h = int(round(h * s))
            new_w = int(round(w * s))
            resized = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
        pad_bottom = int(target - new_h)
        pad_right = int(target - new_w)
        if pad_bottom > 0 or pad_right > 0:
            padded = F.pad(resized, (0, pad_right, 0, pad_bottom), mode="constant", value=0.0)
        else:
            padded = resized
        out.append(padded)
    return out


def _predict_with_hflip_merge(
    predict_batch_fn,
    images: List[torch.Tensor],
    img_ids: List[int],
    ow_map: Dict[int, int],
    oh_map: Dict[int, int],
    iou_thresh: float,
    do_hflip: bool,
) -> List[Dict[str, Any]]:
    """Run base predict (and optional hflip) and merge per-image with NMS in original coords.
    Assumes predict_batch_fn returns boxes already mapped to original coordinates.
    """
    dets_base = predict_batch_fn(images, img_ids)
    dets_all = dets_base
    if do_hflip:
        images_flip = [torch.flip(img, dims=[2]) for img in images]
        dets_flip = predict_batch_fn(images_flip, img_ids)
        dets_flip_u = []
        for d in dets_flip:
            iid = int(d["image_id"])
            ow = int(ow_map[iid])
            x, y, w, h = d["bbox"]
            x_new = max(0.0, float(ow) - float(x) - float(w))
            d2 = dict(d); d2["bbox"] = [x_new, y, w, h]
            dets_flip_u.append(d2)
        dets_all = dets_base + dets_flip_u

    # Per-image merge
    by_img: Dict[int, List[Dict[str, Any]]] = {}
    for d in dets_all:
        by_img.setdefault(int(d["image_id"]), []).append(d)
    merged = []
    for iid, lst in by_img.items():
        merged.extend(_nms_merge_per_image(lst, ow=int(ow_map[iid]), oh=int(oh_map[iid]), iou_thresh=float(iou_thresh)))
    return merged


def _tta_over_loader(
    dl,
    predict_batch_fn,
    ds,
    base_size: int,
    tta_scales: List[int],
    do_hflip: bool,
    iou_thresh: float,
    resize_images_fn=_resize_images_to_square,
    scale_predict_fns: Optional[Dict[int, Callable[[List[torch.Tensor], List[int]], List[Dict[str, Any]]]]] = None,
) -> List[Dict[str, Any]]:
    """Generic TTA collector: base pass + optional multiscales, then final per-image merge.
    predict_batch_fn must output boxes in original coordinates.
    """
    ow_map = {int(im["id"]): int(im["width"]) for im in ds.images}
    oh_map = {int(im["id"]): int(im["height"]) for im in ds.images}

    merged_all: List[Dict[str, Any]] = []

    # Base pass (use images as-is)
    for images, targets in dl:
        img_ids = [int(t["image_id"].item()) for t in targets]
        merged_batch = _predict_with_hflip_merge(
            predict_batch_fn, images, img_ids, ow_map, oh_map, iou_thresh, do_hflip
        )
        merged_all.extend(merged_batch)

    # Multiscale passes (resize in-memory; do not recreate dataset)
    for sz in (tta_scales or []):
        sz = int(sz)
        for images, targets in dl:
            img_ids = [int(t["image_id"].item()) for t in targets]
            # If a scale-specific predictor is provided (e.g., DETR), use it and avoid pre-resize
            if scale_predict_fns is not None and sz in scale_predict_fns:
                pfn = scale_predict_fns[sz]
                images_s = images
            else:
                pfn = predict_batch_fn
                images_s = resize_images_fn(images, sz)
            merged_batch = _predict_with_hflip_merge(pfn, images_s, img_ids, ow_map, oh_map, iou_thresh, do_hflip)
            merged_all.extend(merged_batch)

    # Final merge across all passes per image
    detections: List[Dict[str, Any]] = []
    by_img_all: Dict[int, List[Dict[str, Any]]] = {}
    for d in merged_all:
        by_img_all.setdefault(int(d["image_id"]), []).append(d)
    for iid, lst in by_img_all.items():
        detections.extend(_nms_merge_per_image(lst, ow=int(ow_map[iid]), oh=int(oh_map[iid]), iou_thresh=float(iou_thresh)))
    return detections


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

    # Per-class AP by area (small/medium/large)
    area_idx = {"S": 1, "M": 2, "L": 3}
    per_class_aps, per_class_apm, per_class_apl = {}, {}, {}
    for k_idx, cat_id in enumerate(cat_ids_eval_order):
        # small
        pr_s = precisions[:, :, k_idx, area_idx["S"], m]
        pr_s = pr_s[pr_s > -1]
        per_class_aps[cat_id] = float(np.mean(pr_s)) if pr_s.size > 0 else float("nan")
        # medium
        pr_m = precisions[:, :, k_idx, area_idx["M"], m]
        pr_m = pr_m[pr_m > -1]
        per_class_apm[cat_id] = float(np.mean(pr_m)) if pr_m.size > 0 else float("nan")
        # large
        pr_l = precisions[:, :, k_idx, area_idx["L"], m]
        pr_l = pr_l[pr_l > -1]
        per_class_apl[cat_id] = float(np.mean(pr_l)) if pr_l.size > 0 else float("nan")

    # Per-class AR by area (small/medium/large)
    recalls = coco_eval.eval.get("recall", None)
    per_class_ars, per_class_arm, per_class_arl = {}, {}, {}
    if recalls is not None:
        # recall shape: [T, K, A, M]; use maxDets index m (same as above)
        for k_idx, cat_id in enumerate(cat_ids_eval_order):
            rc_s = recalls[:, k_idx, area_idx["S"], m]
            rc_s = rc_s[rc_s > -1]
            per_class_ars[cat_id] = float(np.mean(rc_s)) if rc_s.size > 0 else float("nan")
            rc_m = recalls[:, k_idx, area_idx["M"], m]
            rc_m = rc_m[rc_m > -1]
            per_class_arm[cat_id] = float(np.mean(rc_m)) if rc_m.size > 0 else float("nan")
            rc_l = recalls[:, k_idx, area_idx["L"], m]
            rc_l = rc_l[rc_l > -1]
            per_class_arl[cat_id] = float(np.mean(rc_l)) if rc_l.size > 0 else float("nan")

    return {
        "overall": overall, "macro_mAP": macro_mAP,
        "per_class_ap": {id_to_name[cid]: per_class_ap[cid] for cid in per_class_ap},
        "per_class_ap50": {id_to_name[cid]: per_class_ap50[cid] for cid in per_class_ap50},
        "per_class_aps": {id_to_name[cid]: per_class_aps[cid] for cid in per_class_aps},
        "per_class_apm": {id_to_name[cid]: per_class_apm[cid] for cid in per_class_apm},
        "per_class_apl": {id_to_name[cid]: per_class_apl[cid] for cid in per_class_apl},
        "per_class_ars": {id_to_name[cid]: per_class_ars[cid] for cid in per_class_ars},
        "per_class_arm": {id_to_name[cid]: per_class_arm[cid] for cid in per_class_arm},
        "per_class_arl": {id_to_name[cid]: per_class_arl[cid] for cid in per_class_arl},
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


def plot_pr_curves_per_class_iou50(coco_eval: COCOeval, out_dir: str):
    """Save a separate PR curve (IoU=0.50) image for each class.

    Filenames: pr_<class_name>_iou50.jpg with spaces replaced by underscores.
    """
    precisions = coco_eval.eval["precision"]
    if precisions is None:
        raise RuntimeError("COCOeval precision array is None — cannot plot PR curves.")

    iou_thrs, rec_thrs = coco_eval.params.iouThrs, coco_eval.params.recThrs
    cat_ids = coco_eval.params.catIds
    cat_names = {c["id"]: c["name"] for c in coco_eval.cocoGt.loadCats(cat_ids)}
    a, m = 0, 2 if precisions.shape[-1] >= 3 else precisions.shape[-1] - 1
    t_50 = int(np.argmin(np.abs(iou_thrs - 0.50)))

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    for k_idx, cat_id in enumerate(cat_ids):
        pr = precisions[t_50, :, k_idx, a, m]
        valid = pr > -1
        if not np.any(valid):
            continue
        plt.figure(figsize=(5, 4))
        plt.plot(rec_thrs[valid], pr[valid], label=f"{cat_names[cat_id]}")
        plt.title(f"PR: {cat_names[cat_id]} (IoU=0.50)")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.xlim([0, 1]); plt.ylim([0, 1])
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(loc="lower left", fontsize=8)
        safe = str(cat_names[cat_id]).replace(" ", "_")
        plt.tight_layout(); plt.savefig(str(out_dir_p / f"pr_{safe}_iou50.jpg"), dpi=150); plt.close()


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
    ap.add_argument("--pr-class-dir", default="", help="If set, save one PR curve per class (IoU=0.50) into this folder")
    ap.add_argument("--csv-out", default="", help="Optional path to save metrics as CSV")
    # RetinaNet ONNX preprocessing to mimic torchvision GeneralizedRCNNTransform
    ap.add_argument("--retina-short", type=int, default=800, help="RetinaNet shortest edge (ONNX path)")
    ap.add_argument("--retina-max", type=int, default=1333, help="RetinaNet longest edge cap (ONNX path)")
    ap.add_argument("--retina-onnx-score", type=float, default=0.05, help="RetinaNet ONNX score threshold (match Torch default ~0.05)")
    ap.add_argument("--retina-onnx-nms", type=float, default=0.5, help="RetinaNet ONNX NMS IoU (match Torch default 0.5)")
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
            detr_short=800, detr_max=1333,
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
            # Use generic TTA collector with in-memory resize
            detections = _tta_over_loader(
                dl=dl,
                predict_batch_fn=predict_batch_fn,
                ds=ds,
                base_size=int(args.resize_short),
                tta_scales=tta_scales,
                do_hflip=bool(args.tta_hflip),
                iou_thresh=float(args.tta_nms_iou),
            )

    else:  # ONNX backend
        detections = []
        if args.model == "retinanet":
            # Build a per-batch predictor that returns boxes in original coordinates
            onnx_model = ONNXRetinaNetWrapper(
                onnx_path=args.onnx,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                class_ids=cat_ids_sorted,
                score_thresh=float(args.retina_onnx_score), iou_thresh=float(args.retina_onnx_nms),
            )

            def predict_retina_onnx_torchlike(images: List[torch.Tensor], img_ids: List[int]):
                # Resize each image like torchvision's transform: shortest->retina_short, cap longest->retina_max
                resized: List[torch.Tensor] = []
                scales: Dict[int, float] = {}
                for t, iid in zip(images, img_ids):
                    _, h, w = t.shape
                    s1 = float(args.retina_short) / max(1.0, float(min(h, w)))
                    s2 = float(args.retina_max) / max(1.0, float(max(h, w)))
                    s = min(s1, s2)
                    new_h = int(round(h * s)); new_w = int(round(w * s))
                    if new_h <= 0: new_h = 1
                    if new_w <= 0: new_w = 1
                    if new_h == h and new_w == w:
                        t_resized = t
                    else:
                        t_resized = torch.nn.functional.interpolate(
                            t.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
                        ).squeeze(0)
                    # Pad to stride 32 like the RCNN transform
                    pad_h = (32 - (t_resized.shape[-2] % 32)) % 32
                    pad_w = (32 - (t_resized.shape[-1] % 32)) % 32
                    if pad_h or pad_w:
                        t_resized = torch.nn.functional.pad(t_resized, (0, pad_w, 0, pad_h), value=0.0)
                    resized.append(t_resized)
                    scales[int(iid)] = s

                # Stack to batch with extra pad if necessary
                max_h = max(int(t.shape[-2]) for t in resized)
                max_w = max(int(t.shape[-1]) for t in resized)
                batch = []
                for t in resized:
                    _, h, w = t.shape
                    if h == max_h and w == max_w:
                        batch.append(t)
                    else:
                        pad = torch.nn.functional.pad(t, (0, max_w - w, 0, max_h - h), value=0.0)
                        batch.append(pad)
                images_b = torch.stack(batch, dim=0)

                dets = onnx_model.predict(images_b, img_ids)
                out = []
                for det in dets:
                    iid = int(det["image_id"])
                    oh = int(ds.imgid_to_img[iid]["height"]); ow = int(ds.imgid_to_img[iid]["width"]) 
                    s = float(scales.get(iid, 1.0))
                    new_h = int(round(oh * s)); new_w = int(round(ow * s))
                    x, y, w, h = det["bbox"]
                    x1, y1, x2, y2 = x, y, x + w, y + h
                    x1 = max(0.0, min(x1, new_w - 1)); y1 = max(0.0, min(y1, new_h - 1))
                    x2 = max(0.0, min(x2, new_w - 1)); y2 = max(0.0, min(y2, new_h - 1))
                    if s > 0:
                        x1 /= s; y1 /= s; x2 /= s; y2 /= s
                    d2 = dict(det)
                    d2["bbox"] = [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))]
                    out.append(d2)
                # Cap detections per image to align with Torch default (100)
                by_img: Dict[int, List[Dict[str, Any]]] = {}
                for d in out:
                    by_img.setdefault(int(d["image_id"]), []).append(d)
                capped: List[Dict[str, Any]] = []
                for iid, lst in by_img.items():
                    lst_sorted = sorted(lst, key=lambda x: float(x.get("score", 0.0)), reverse=True)
                    capped.extend(lst_sorted[:100])
                return capped

            if not args.tta_hflip and not tta_scales:
                for images, targets in dl:
                    img_ids = [int(t["image_id"].item()) for t in targets]
                    detections.extend(predict_retina_onnx_torchlike(images, img_ids))
            else:
                detections = _tta_over_loader(
                    dl=dl,
                    predict_batch_fn=predict_retina_onnx_torchlike,
                    ds=ds,
                    base_size=int(args.resize_short),
                    tta_scales=tta_scales,
                    do_hflip=bool(args.tta_hflip),
                    iou_thresh=float(args.tta_nms_iou),
                )
        else:  # DETR ONNX
            # Map image_id -> (H,W) originals for postprocess
            orig_size_map = {int(im["id"]): (int(im["height"]), int(im["width"])) for im in ds.images}
            base_wrapper = ONNXDetrWrapper(
                onnx_path=args.onnx,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                class_ids=cat_ids_sorted,
                score_thresh=0.05, detr_short=800, detr_max=1333,
                orig_size_map=orig_size_map,
            )

            def predict_detr(images: List[torch.Tensor], img_ids: List[int]):
                return base_wrapper.predict(images, img_ids)

            if not args.tta_hflip and not tta_scales:
                for images, targets in dl:
                    img_ids = [int(t["image_id"].item()) for t in targets]
                    detections.extend(predict_detr(images, img_ids))
            else:
                # Build per-scale predictors that change shortest/longest edge inside the wrapper
                scale_predict: Dict[int, Callable[[List[torch.Tensor], List[int]], List[Dict[str, Any]]]] = {}
                for sz in (tta_scales or []):
                    wrapper_s = ONNXDetrWrapper(
                        onnx_path=args.onnx,
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                        class_ids=cat_ids_sorted,
                        score_thresh=0.05, detr_short=int(sz), detr_max=int(sz),
                        orig_size_map=orig_size_map,
                    )

                    def make_fn(w):
                        return lambda images, img_ids, _w=w: _w.predict(images, img_ids)
                    scale_predict[int(sz)] = make_fn(wrapper_s)

                detections = _tta_over_loader(
                    dl=dl,
                    predict_batch_fn=predict_detr,
                    ds=ds,
                    base_size=int(args.resize_short),
                    tta_scales=tta_scales,
                    do_hflip=bool(args.tta_hflip),
                    iou_thresh=float(args.tta_nms_iou),
                    # For DETR, avoid pre-resizing; use per-scale predictor
                    resize_images_fn=lambda imgs, s: imgs,
                    scale_predict_fns=scale_predict if tta_scales else None,
                )

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
    if args.pr_class_dir:
        plot_pr_curves_per_class_iou50(results["coco_eval"], args.pr_class_dir)
        print(f"[OK] Saved per-class PR curves (IoU=0.50) to: {args.pr_class_dir}")

    # Optional: write concise CSV with overall, macro, per-class AP and AP50
    try:
        import csv
        from pathlib import Path as _P
        csv_path = getattr(args, "csv_out", None)
    except Exception:
        csv_path = None
    if csv_path:
        p = _P(csv_path); p.parent.mkdir(parents=True, exist_ok=True)
        cols = [
            "section", "name", "AP", "AP50", "AP75", "APS", "APM", "APL",
            "AR1", "AR10", "AR100", "ARS", "ARM", "ARL", "macro_mAP",
        ]
        def _fmt4(x):
            try:
                return f"{float(x):.4f}"
            except Exception:
                return x
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            # Row 1: COCO overall
            row_overall = {"section": "COCO", "name": "overall"}
            row_overall.update({k: _fmt4(v) for k, v in results["overall"].items()})
            w.writerow(row_overall)
            # Row 2: Macro-mAP
            w.writerow({"section": "Macro", "name": "macro_mAP", "macro_mAP": _fmt4(results["macro_mAP"])})

            # Rows: unified per-class line with AP/AP50 + APS/APM/APL + ARS/ARM/ARL
            for cls_name in results.get("cat_eval_order", []):
                w.writerow({
                    "section": "PerClass",
                    "name": cls_name,
                    "AP": _fmt4(results.get("per_class_ap", {}).get(cls_name, "")),
                    "AP50": _fmt4(results.get("per_class_ap50", {}).get(cls_name, "")),
                    "APS": _fmt4(results.get("per_class_aps", {}).get(cls_name, "")),
                    "APM": _fmt4(results.get("per_class_apm", {}).get(cls_name, "")),
                    "APL": _fmt4(results.get("per_class_apl", {}).get(cls_name, "")),
                    "ARS": _fmt4(results.get("per_class_ars", {}).get(cls_name, "")),
                    "ARM": _fmt4(results.get("per_class_arm", {}).get(cls_name, "")),
                    "ARL": _fmt4(results.get("per_class_arl", {}).get(cls_name, "")),
                })
        print(f"[OK] Wrote metrics CSV to: {p}")


if __name__ == "__main__":
    main()

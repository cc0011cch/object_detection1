#!/usr/bin/env python3
"""
Visualize predictions from PyTorch and/or ONNX models (RetinaNet/DETR) vs. COCO ground truth.

Features:
- Samples N images from a COCO annotations file + images folder.
- Runs optional PyTorch model (ckpt) and/or ONNX model (onnx) for RetinaNet or DETR.
- Overlays bounding boxes with labels:
    - Ground truth: green
    - Torch: blue
    - ONNX: red
- Saves composite JPGs to an output directory.

Examples:

  # RetinaNet: compare Torch vs ONNX
  python tools/visualize_compare_predictions.py \
    --images ./data/coco/train2017 \
    --ann ./data/coco/annotations_used/instances_train2017_debug500.json \
    --model retinanet \
    --torch-ckpt runs/retina_rfs001/best.pth \
    --onnx runs/retina_rfs001/retinanet_head.onnx \
    --num 12 --out-dir runs/viz_retina

  # DETR: ONNX only
  python tools/visualize_compare_predictions.py \
    --images ./data/coco/train2017 \
    --ann ./data/coco/annotations_used/instances_train2017_debug500.json \
    --model detr \
    --onnx runs/detr_debug500_rfsAlbu/detr_best.onnx \
    --num 12 --out-dir runs/viz_detr_onnx
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import cv2

# Ensure project root is on sys.path when running from tools/
import sys
from pathlib import Path as _PathForSys
_THIS = _PathForSys(__file__).resolve()
_ROOT = _THIS.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset import CocoDetDataset
from train import build_model_and_helpers
from onnx_retinanet_wrapper import ONNXRetinaNetWrapper
from onnx_detr_wrapper import ONNXDetrWrapper


def draw_boxes(im_rgb: np.ndarray,
               dets: List[Dict[str, Any]],
               catid_to_name: Dict[int, str],
               color: Tuple[int, int, int],
               label_prefix: str,
               thickness: int = 2) -> np.ndarray:
    """Draw list of COCO-style detections (xywh in original coords).
    color in RGB; im_rgb must be RGB uint8.
    """
    out = im_rgb.copy()
    for d in dets:
        x, y, w, h = d["bbox"]
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + w)), int(round(y + h))
        cv2.rectangle(out, (x1, y1), (x2, y2), color[::-1], thickness)  # cv2 expects BGR
        name = catid_to_name.get(int(d["category_id"]), str(d.get("category_id")))
        score = d.get("score", None)
        text = f"{label_prefix}:{name}" if score is None else f"{label_prefix}:{name} {score:.2f}"
        cv2.putText(out, text, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[::-1], 1, cv2.LINE_AA)
    return out


def collect_gt_for_image(ds: CocoDetDataset, img_id: int) -> List[Dict[str, Any]]:
    dets = []
    info = ds.imgid_to_img[int(img_id)]
    w, h = int(info["width"]), int(info["height"])  # not needed; GT already in original
    for a in ds.anns_by_img.get(int(img_id), []):
        if a.get("iscrowd", 0) == 1:
            continue
        x, y, bw, bh = a["bbox"]
        if bw <= 0 or bh <= 0:
            continue
        dets.append({
            "image_id": int(img_id),
            "category_id": int(a["category_id"]),  # COCO category id
            "bbox": [float(x), float(y), float(bw), float(bh)],
        })
    return dets


def build_torch_predictor(model_name: str,
                          ds: CocoDetDataset,
                          ckpt_path: str,
                          device: torch.device):
    cat_ids_sorted = [c["id"] for c in ds.categories]
    id2label = getattr(ds, "id2label_0based", {i: c["name"] for i, c in enumerate(ds.categories)})
    label2id = getattr(ds, "label2id_0based", {c["name"]: i for i, c in enumerate(ds.categories)})

    # Original size map for proper scaling inside predictors
    orig_size_map = {int(im["id"]): (int(im["height"]), int(im["width"])) for im in ds.images}

    model, _, _, predict_batch = build_model_and_helpers(
        model_name=model_name,
        num_classes=ds.num_classes,
        id2label_0based=id2label,
        label2id_0based=label2id,
        cat_ids_sorted=cat_ids_sorted,
        device=device,
        orig_size_map=orig_size_map,
        detr_short=800,
        detr_max=1333,
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(ckpt["model"], assign=True)
    except TypeError:
        model.load_state_dict(ckpt["model"])
    model.eval()
    return predict_batch


def build_onnx_predictor(model_name: str,
                         ds: CocoDetDataset,
                         onnx_path: str,
                         retina_short: int = 800,
                         retina_max: int = 1333):
    cat_ids_sorted = [c["id"] for c in ds.categories]

    if model_name == "retinanet":
        base = ONNXRetinaNetWrapper(
            onnx_path=onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            class_ids=cat_ids_sorted,
            score_thresh=0.3,
            iou_thresh=0.5,
        )

        def predict_batch(images: List[torch.Tensor], img_ids: List[int]):
            # Resize each image like torchvision GeneralizedRCNNTransform
            resized: List[torch.Tensor] = []
            scales: Dict[int, float] = {}
            for t, iid in zip(images, img_ids):
                _, h, w = t.shape
                s1 = float(retina_short) / max(1.0, float(min(h, w)))
                s2 = float(retina_max) / max(1.0, float(max(h, w)))
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
                resized.append(t_resized)
                scales[int(iid)] = s

            # Pad each image so H,W are divisible by 32 (match Torch transform),
            # then pad to a common batch size (bottom/right).
            stride = 32
            resized2: List[torch.Tensor] = []
            for t in resized:
                _, h, w = t.shape
                pad_h = (stride - (h % stride)) % stride
                pad_w = (stride - (w % stride)) % stride
                if pad_h or pad_w:
                    t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
                resized2.append(t)

            max_h = max(int(t.shape[-2]) for t in resized2)
            max_w = max(int(t.shape[-1]) for t in resized2)
            batch = []
            for t in resized2:
                _, h, w = t.shape
                if h == max_h and w == max_w:
                    batch.append(t)
                else:
                    pad = torch.nn.functional.pad(t, (0, max_w - w, 0, max_h - h), mode="constant", value=0.0)
                    batch.append(pad)
            images_b = torch.stack(batch, dim=0)

            dets = base.predict(images_b, img_ids)
            out = []
            for d in dets:
                iid = int(d["image_id"])
                oh = int(ds.imgid_to_img[iid]["height"]); ow = int(ds.imgid_to_img[iid]["width"]) 
                s = float(scales.get(iid, 1.0))
                new_h = int(round(oh * s)); new_w = int(round(ow * s))
                x, y, w, h = d["bbox"]
                x1, y1, x2, y2 = x, y, x + w, y + h
                x1 = max(0.0, min(x1, new_w - 1)); y1 = max(0.0, min(y1, new_h - 1))
                x2 = max(0.0, min(x2, new_w - 1)); y2 = max(0.0, min(y2, new_h - 1))
                if s > 0:
                    x1 /= s; y1 /= s; x2 /= s; y2 /= s
                d2 = dict(d)
                d2["bbox"] = [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))]
                out.append(d2)
            return out

        return predict_batch

    else:  # DETR
        orig_size_map = {int(im["id"]): (int(im["height"]), int(im["width"])) for im in ds.images}
        base = ONNXDetrWrapper(
            onnx_path=onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            class_ids=cat_ids_sorted,
            score_thresh=0.05,
            detr_short=800,
            detr_max=1333,
            orig_size_map=orig_size_map,
        )

        def predict_batch(images: List[torch.Tensor], img_ids: List[int]):
            return base.predict(images, img_ids)
        # attach wrapper for debug access
        setattr(predict_batch, "_wrapper", base)
        return predict_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--model", choices=["retinanet", "detr"], required=True)
    ap.add_argument("--torch-ckpt", default=None, help="Path to PyTorch checkpoint (.pth)")
    ap.add_argument("--onnx", default=None, help="Path to ONNX model (.onnx)")
    ap.add_argument("--num", type=int, default=12)
    ap.add_argument("--out-dir", default="runs/viz")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--retina-short", type=int, default=800, help="RetinaNet ONNX shortest edge resize")
    ap.add_argument("--retina-max", type=int, default=1333, help="RetinaNet ONNX longest edge cap")
    # Visualization filtering controls
    ap.add_argument("--torch-thresh", type=float, default=0.5, help="Score threshold for Torch dets (after postprocess)")
    ap.add_argument("--onnx-thresh", type=float, default=None, help="Score threshold for ONNX dets (overrides wrapper default)")
    ap.add_argument("--topk", type=int, default=100, help="Keep at most K highest-score dets per image per source")
    ap.add_argument("--debug-overlay", action="store_true", help="Overlay debug info (sizes/scales) for DETR ONNX")
    args = ap.parse_args()

    assert args.torch_ckpt or args.onnx, "Provide at least one of --torch-ckpt or --onnx"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CocoDetDataset(images_dir=args.images, ann_json=args.ann, augment=False, use_albu=False)

    random.seed(args.seed)
    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    idxs = idxs[: max(1, int(args.num))]

    # Category id->name (COCO ids)
    catid_to_name = {int(c["id"]): c["name"] for c in ds.categories}

    # Predictors
    predict_torch = None
    predict_onnx = None
    if args.torch_ckpt:
        predict_torch = build_torch_predictor(args.model, ds, args.torch_ckpt, device)
    if args.onnx:
        predict_onnx = build_onnx_predictor(
            args.model, ds, args.onnx,
            retina_short=int(args.retina_short), retina_max=int(args.retina_max)
        )

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for i in idxs:
        img_t, tgt = ds[i]
        img_id = int(tgt["image_id"].item())
        im_meta = ds.imgid_to_img[img_id]
        img_path = os.path.join(ds.images_dir, im_meta["file_name"])
        # Load original RGB for drawing
        img_rgb = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if img_rgb is None:
            print(f"[warn] failed to read image: {img_path}")
            continue

        overlays = img_rgb.copy()

        # Ground truth (green)
        gt_dets = collect_gt_for_image(ds, img_id)
        overlays = draw_boxes(overlays, gt_dets, catid_to_name, color=(0, 255, 0), label_prefix="GT", thickness=2)

        # Prepare 1-image batch for predictors
        images_batch = [img_t]  # CHW float [0,1]
        img_ids = [img_id]

        # Torch predictions (blue)
        if predict_torch is not None:
            try:
                torch_dets_all = predict_torch(images_batch, img_ids)
                torch_dets = [d for d in torch_dets_all if int(d.get("image_id", -1)) == img_id]
                # Filter by score and keep top-K
                if args.torch_thresh is not None:
                    torch_dets = [d for d in torch_dets if float(d.get("score", 0.0)) >= float(args.torch_thresh)]
                if args.topk and len(torch_dets) > int(args.topk):
                    torch_dets = sorted(torch_dets, key=lambda x: float(x.get("score", 0.0)), reverse=True)[: int(args.topk)]
                overlays = draw_boxes(overlays, torch_dets, catid_to_name, color=(0, 128, 255), label_prefix="Torch", thickness=2)
            except Exception as e:
                print(f"[warn] Torch predict failed for {img_path}: {e}")

        # ONNX predictions (red)
        if predict_onnx is not None:
            try:
                onnx_dets_all = predict_onnx(images_batch, img_ids)
                onnx_dets = [d for d in onnx_dets_all if int(d.get("image_id", -1)) == img_id]
                # Optional threshold override and top-K
                if args.onnx_thresh is not None:
                    onnx_dets = [d for d in onnx_dets if float(d.get("score", 0.0)) >= float(args.onnx_thresh)]
                if args.topk and len(onnx_dets) > int(args.topk):
                    onnx_dets = sorted(onnx_dets, key=lambda x: float(x.get("score", 0.0)), reverse=True)[: int(args.topk)]
                overlays = draw_boxes(overlays, onnx_dets, catid_to_name, color=(255, 64, 64), label_prefix="ONNX", thickness=2)

                # Optional debug overlay (DETR only): draw resized (unpadded) region mapped back to original
                if args.debug_overlay and args.model == "detr" and hasattr(predict_onnx, "_wrapper"):
                    dbg = getattr(predict_onnx, "_wrapper").last_debug or {}
                    if int(img_id) in dbg:
                        info = dbg[int(img_id)]
                        oh, ow = info.get("orig", (overlays.shape[0], overlays.shape[1]))
                        h_in, w_in = info.get("input", (oh, ow))
                        s = info.get("scale", min(h_in / max(1, oh), w_in / max(1, ow)))
                        new_h = int(round(h_in / max(1e-6, s)))
                        new_w = int(round(w_in / max(1e-6, s)))
                        # Clip to original bounds
                        new_h = min(new_h, overlays.shape[0]); new_w = min(new_w, overlays.shape[1])
                        cv2.rectangle(overlays, (0, 0), (new_w - 1, new_h - 1), (255, 255, 0), 2)
                        txt = f"orig={oh}x{ow} input={h_in}x{w_in} s={s:.4f}"
                        cv2.putText(overlays, txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
            except Exception as e:
                print(f"[warn] ONNX predict failed for {img_path}: {e}")

        # Save
        ofn = f"{Path(im_meta['file_name']).stem}_viz.jpg"
        out_path = out_dir / ofn
        cv2.imwrite(str(out_path), cv2.cvtColor(overlays, cv2.COLOR_RGB2BGR))
        print(f"[ok] saved {out_path}")


if __name__ == "__main__":
    main()

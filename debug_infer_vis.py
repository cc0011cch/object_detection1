#!/usr/bin/env python3
"""
Debug visualizer for RetinaNet:
- Saves JPGs with GT (green) + predictions (red).
- By default draws on the RESIZED/PADDED image (matches model space).
- Optional: --on-original to project boxes back to ORIGINAL image space.

Usage:
  python debug_infer_save_gt_fixed.py \
    --ckpt runs/retina_rfs001/best.pth \
    --ann ./data/coco/annotations_used/instances_train2017_debug500.json \
    --images ./data/coco/train2017 \
    --save-dir debug_detections_gt_fixed \
    --num 5 \
    --score 0.3 \
    --target-size 512
    # add --on-original to draw on original images
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from train import build_model_and_helpers
from dataset import CocoDetDataset


def tensor_image_to_uint8(img_tensor):
    """
    img_tensor: torch.Tensor [3,H,W], values in [0,1]
    return: HxW x 3 uint8 numpy for matshow
    """
    img = img_tensor.detach().cpu().clamp(0,1).permute(1,2,0).numpy()  # H,W,3
    img = (img * 255.0 + 0.5).astype(np.uint8)
    return img


def invert_letterbox_boxes(boxes_xyxy, orig_w, orig_h, target_size):
    """
    Invert typical letterbox: scale short side to target_size, then center-pad to (target_size, target_size).
    boxes_xyxy are in resized/padded coords (target_size x target_size).
    Returns boxes in ORIGINAL image coords.
    """
    # compute scale and padding used
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w, new_h = orig_w * scale, orig_h * scale
    pad_x = (target_size - new_w) * 0.5
    pad_y = (target_size - new_h) * 0.5

    # unpad, then unscale
    x1 = (boxes_xyxy[:,0] - pad_x) / scale
    y1 = (boxes_xyxy[:,1] - pad_y) / scale
    x2 = (boxes_xyxy[:,2] - pad_x) / scale
    y2 = (boxes_xyxy[:,3] - pad_y) / scale

    boxes = np.stack([x1, y1, x2, y2], axis=1)
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--save-dir", default="debug_detections_gt_fixed")
    ap.add_argument("--num", type=int, default=5)
    ap.add_argument("--score", type=float, default=0.3)
    ap.add_argument("--target-size", type=int, default=512)
    ap.add_argument("--on-original", action="store_true",
                    help="Project boxes back to original image coords and draw on original image.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir); 
    save_dir.mkdir(parents=True, exist_ok=True)
    # Dataset (no aug)
    ds = CocoDetDataset(
        images_dir=args.images,
        ann_json=args.ann,
        augment=False,
        use_albu=False,
    )
    # IMPORTANT: match your validation/inference target size
    try:
        ds.set_target_size(args.target_size)
    except Exception:
        pass

    # Build model
    cat_ids_sorted = sorted([c["id"] for c in ds.categories])
    id2label_0based = getattr(ds, "id2label_0based", {})
    label2id_0based = getattr(ds, "label2id_0based", {})
    model, _, _, predict_batch_fn = build_model_and_helpers(
        model_name="retinanet",
        num_classes=ds.num_classes,
        id2label_0based=id2label_0based,
        label2id_0based=label2id_0based,
        cat_ids_sorted=cat_ids_sorted,
        device=device,
    )
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # choose random samples
    idxs = random.sample(range(len(ds)), min(args.num, len(ds)))

    for idx in idxs:
        img, target = ds[idx]  # img: [3,H',W'] in resized space
        img_id = int(target["image_id"].item())

        # run model
        with torch.no_grad():
            preds = predict_batch_fn([img.to(device)], [img_id])

        # choose drawing canvas + coordinate space
        if args.on_original:
            # original image + invert letterbox for both GT and predictions
            img_info = ds.imgid_to_img[img_id]
            orig_path = img_info["file_name"] if os.path.isabs(img_info["file_name"]) else os.path.join(args.images, img_info["file_name"])
            orig = Image.open(orig_path).convert("RGB")
            canvas = np.array(orig)
            Hc, Wc = canvas.shape[:2]
            # boxes come as [x1,y1,x2,y2] for GT; predictions are [x,y,w,h] -> convert to xyxy first
            gt_xyxy = target["boxes"].cpu().numpy() if "boxes" in target else target["bbox"].cpu().numpy()
            gt_xyxy = gt_xyxy.copy()
            # invert to original
            gt_xyxy_orig = invert_letterbox_boxes(gt_xyxy, Wc, Hc, args.target_size)

            # predictions
            dets_xyxy, det_scores, det_names = [], [], []
            for d in preds:
                if d["image_id"] != img_id or d["score"] < args.score:
                    continue
                x, y, w, h = d["bbox"]
                xyxy_resized = np.array([x, y, x+w, y+h], dtype=np.float32)[None, :]
                xyxy_orig = invert_letterbox_boxes(xyxy_resized, Wc, Hc, args.target_size)[0]
                dets_xyxy.append(xyxy_orig)
                det_scores.append(d["score"])
                name = [c["name"] for c in ds.categories if c["id"] == d["category_id"]][0]
                det_names.append(name)

            # draw
            fig = plt.figure(figsize=(8,8)); plt.imshow(canvas); ax = plt.gca()
            # GT (green)
            for b in gt_xyxy_orig:
                x1,y1,x2,y2 = b
                rect = plt.Rectangle((x1,y1), x2-x1, y2-y1, fill=False, color="lime", linewidth=2)
                ax.add_patch(rect)
            # PRED (red)
            for b, s, n in zip(dets_xyxy, det_scores, det_names):
                x1,y1,x2,y2 = b
                rect = plt.Rectangle((x1,y1), x2-x1, y2-y1, fill=False, color="red", linewidth=2)
                ax.add_patch(rect)
                ax.text(x1, y1, f"{n} {s:.2f}", color="yellow", fontsize=10)

            title = f"img {img_id} (original) GT=green, Pred=red"
        else:
            # resized canvas (model space) — simplest, always correct
            canvas = tensor_image_to_uint8(img)
            Hc, Wc = canvas.shape[:2]
            gt_xyxy = target["boxes"].cpu().numpy() if "boxes" in target else target["bbox"].cpu().numpy()
            fig = plt.figure(figsize=(8,8)); plt.imshow(canvas); ax = plt.gca()
            # GT
            for b in gt_xyxy:
                x1,y1,x2,y2 = b
                rect = plt.Rectangle((x1,y1), x2-x1, y2-y1, fill=False, color="lime", linewidth=2)
                ax.add_patch(rect)
            # PRED
            for d in preds:
                if d["image_id"] != img_id or d["score"] < args.score:
                    continue
                x, y, w, h = d["bbox"]
                rect = plt.Rectangle((x, y), w, h, fill=False, color="red", linewidth=2)
                ax.add_patch(rect)
                name = [c["name"] for c in ds.categories if c["id"] == d["category_id"]][0]
                ax.text(x, y, f"{name} {d['score']:.2f}", color="yellow", fontsize=10)

            title = f"img {img_id} (resized) GT=green, Pred=red"

        plt.title(title); plt.axis("off")
        out_path = save_dir / f"img_{img_id}{'_orig' if args.on_original else ''}.jpg"
        plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
        print(f"[OK] Saved {out_path}")


if __name__ == "__main__":
    main()

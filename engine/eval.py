from typing import Dict, List, Any, Optional
import logging
import numpy as np
import torch

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    COCO_EVAL_AVAILABLE = True
except Exception:
    COCO_EVAL_AVAILABLE = False


@torch.no_grad()
def coco_map_with_macro(model_name: str,
                        model,
                        dl_val,
                        ds_val,
                        device,
                        predict_batch_fn,
                        val_ann_path: str,
                        max_batches: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Compute COCO metrics + macro mAP and AP50. Returns dict or None if COCO tools missing."""
    logger = logging.getLogger("train")
    if not COCO_EVAL_AVAILABLE:
        logger.info("[warn] pycocotools not available; skipping mAP.")
        return None

    detections = []
    cat_ids_sorted = sorted([c["id"] for c in ds_val.categories])

    proc = None
    if model_name == "detr":
        from transformers import DetrImageProcessor
        proc = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    for bidx, (images, targets) in enumerate(dl_val):
        if max_batches is not None and bidx >= max_batches:
            break
        img_ids = [int(t["image_id"].item()) for t in targets]

        if model_name == "retinanet":
            # Use predict_batch_fn that already returns bboxes in original image coordinates
            batch_dets = predict_batch_fn(images, img_ids)
        else:
            model.eval()
            np_imgs = [(img.permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()) for img in images]
            enc = proc(images=np_imgs, return_tensors="pt")
            enc["pixel_values"] = enc["pixel_values"].to(device)
            if "pixel_mask" in enc:
                enc["pixel_mask"] = enc["pixel_mask"].to(device)
            outputs = model(**enc)
            sizes = [(ds_val.imgid_to_img[i]["height"], ds_val.imgid_to_img[i]["width"]) for i in img_ids]
            sizes = torch.tensor(sizes, device=device)
            processed = proc.post_process_object_detection(outputs, target_sizes=sizes)
            batch_dets = []
            for img_id, p in zip(img_ids, processed):
                boxes = p["boxes"].detach().cpu().numpy()
                scores = p["scores"].detach().cpu().numpy()
                labels = p["labels"].detach().cpu().numpy()
                for (x1, y1, x2, y2), s, l in zip(boxes, scores, labels):
                    cat_id = cat_ids_sorted[int(l)]
                    batch_dets.append({
                        "image_id": int(img_id),
                        "category_id": int(cat_id),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(s),
                    })

        detections.extend(batch_dets)

    coco_gt = COCO(val_ann_path)
    if len(detections) == 0:
        logger.info("[warn] No detections produced; skipping mAP this epoch.")
        return None

    coco_dt = coco_gt.loadRes(detections)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()

    # summarize() populates stats; capture its printed table for logs
    try:
        from contextlib import redirect_stdout
        import io as _io
        buf = _io.StringIO()
        with redirect_stdout(buf):
            coco_eval.summarize()
        logger.info("\n[COCOeval summary]\n" + buf.getvalue().strip())
    except Exception:
        pass

    ap = ap50 = ap75 = aps = apm = apl = float("nan")
    if getattr(coco_eval, "stats", None) is not None and len(coco_eval.stats) >= 6:
        ap, ap50, ap75, aps, apm, apl = [float(c) for c in coco_eval.stats[:6]]

    precisions = coco_eval.eval.get("precision", None)
    if precisions is None or precisions.size == 0:
        logger.info("[warn] COCO precision tensor is empty; returning NaNs for Macro metrics.")
        return {
            "AP": ap, "AP50": ap50, "AP75": ap75,
            "APS": aps, "APM": apm, "APL": apl,
            "macro_mAP": float("nan"), "macro_AP50": float("nan"),
            "per_class_ap": {}, "per_class_ap50": {},
        }

    a = 0
    m = 2 if precisions.shape[-1] >= 3 else precisions.shape[-1] - 1
    iou_thrs = coco_eval.params.iouThrs
    t50 = int(np.argmin(np.abs(iou_thrs - 0.50)))

    id_to_name = {c["id"]: c["name"] for c in coco_gt.loadCats(coco_gt.getCatIds())}
    per_class_ap, per_class_ap50 = {}, {}

    for k_idx, cat_id in enumerate(coco_eval.params.catIds):
        pr = precisions[:, :, k_idx, a, m]
        pr = pr[pr > -1]
        per_class_ap[cat_id] = float(np.mean(pr)) if pr.size > 0 else float("nan")

        pr50 = precisions[t50, :, k_idx, a, m]
        pr50 = pr50[pr50 > -1]
        per_class_ap50[cat_id] = float(np.mean(pr50)) if pr50.size > 0 else float("nan")

    macro_map  = float(np.nanmean(list(per_class_ap.values()))) if per_class_ap else float("nan")
    macro_ap50 = float(np.nanmean(list(per_class_ap50.values()))) if per_class_ap50 else float("nan")

    return {
        "AP": ap, "AP50": ap50, "AP75": ap75,
        "APS": aps, "APM": apm, "APL": apl,
        "macro_mAP": macro_map, "macro_AP50": macro_ap50,
        "per_class_ap":   {id_to_name[c]: per_class_ap[c]   for c in per_class_ap},
        "per_class_ap50": {id_to_name[c]: per_class_ap50[c] for c in per_class_ap50},
    }

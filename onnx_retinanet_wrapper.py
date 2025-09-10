#!/usr/bin/env python3
"""
Wrapper around ONNX Runtime RetinaNet export for evaluation.

- Loads the ONNX model (CPU or GPU provider).
- Accepts a batch of preprocessed images [B,3,H,W].
- Returns COCO-format detections: list[dict(image_id, category_id, bbox, score)].
"""

import numpy as np
import onnxruntime as ort
import torch


class ONNXRetinaNetWrapper:
    def __init__(self, onnx_path: str, providers=None, class_ids=None,
                 score_thresh: float = 0.3, iou_thresh: float = 0.5):
        # Prefer CUDA if present; else CPU
        providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = ort.get_available_providers()
        chosen = []
        for p in providers:
            if p in available:
                chosen.append(p)
        if not chosen:
            chosen = ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, providers=chosen)

        self.class_ids = class_ids or list(range(1, 2))  # default: 1 class with id=1
        self.num_classes = len(self.class_ids)
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _decode_deltas_to_boxes(self, anchors_xyxy, deltas):
        # anchors/deltas: [N,4]
        ax1, ay1, ax2, ay2 = anchors_xyxy.T
        aw = ax2 - ax1
        ah = ay2 - ay1
        ax = ax1 + 0.5 * aw
        ay = ay1 + 0.5 * ah

        dx, dy, dw, dh = deltas.T
        px = dx * aw + ax
        py = dy * ah + ay
        pw = np.exp(dw) * aw
        ph = np.exp(dh) * ah

        x1 = px - 0.5 * pw
        y1 = py - 0.5 * ph
        x2 = px + 0.5 * pw
        y2 = py + 0.5 * ph
        return np.stack([x1, y1, x2, y2], axis=1)

    def _nms(self, boxes, scores):
        if boxes.size == 0:
            return np.empty((0,), dtype=np.int64)
        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            denom = areas[i] + areas[order[1:]] - inter + 1e-6
            iou = inter / denom
            inds = np.where(iou <= self.iou_thresh)[0]
            order = order[inds + 1]
        return np.array(keep, dtype=np.int64)

    def predict(self, images, img_ids):
        """
        Args:
            images: torch.Tensor [B,3,H,W], already normalized to ImageNet mean/std.
            img_ids: list of image IDs (ints) matching batch size.
        Returns:
            list of dicts in COCO detection format.
        """
        if isinstance(images, torch.Tensor):
            imgs_np = images.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            imgs_np = np.asarray(images, dtype=np.float32)

        outputs = self.sess.run(
            ["cls_logits", "bbox_deltas", "anchors_xyxy"],
            {"images": imgs_np},
        )
        cls_logits, bbox_deltas, anchors = outputs

        # Shapes
        if cls_logits.ndim != 3:
            raise RuntimeError(f"cls_logits must be [B,N,C], got {cls_logits.shape}")
        if bbox_deltas.ndim != 3:
            raise RuntimeError(f"bbox_deltas must be [B,N,4], got {bbox_deltas.shape}")
        if anchors.ndim != 3:
            # Some exports yield anchors [N,4]; add batch dim
            if anchors.ndim == 2 and anchors.shape[-1] == 4:
                anchors = anchors[None, ...]
            else:
                raise RuntimeError(f"anchors must be [B,N,4] or [N,4], got {anchors.shape}")

        B_logits, N_logits, C = cls_logits.shape
        B_deltas, N_deltas, four = bbox_deltas.shape
        B_anch, N_anch, four_a = anchors.shape

        if C != self.num_classes:
            raise RuntimeError(f"Class mismatch: logits C={C} vs expected {self.num_classes}")

        if four != 4 or four_a != 4:
            raise RuntimeError("bbox_deltas and anchors must have 4 in last dim")

        # If anchors are returned for a single image only, tile across batch
        if B_anch == 1 and B_logits > 1:
            anchors = np.repeat(anchors, B_logits, axis=0)
            B_anch = B_logits
        # If deltas come as single batch (unlikely), tile them too for safety
        if B_deltas == 1 and B_logits > 1:
            bbox_deltas = np.repeat(bbox_deltas, B_logits, axis=0)
            B_deltas = B_logits

        # Final sanity
        if not (B_logits == B_deltas == B_anch):
            raise RuntimeError(f"Batch mismatch among outputs: "
                               f"logits B={B_logits}, deltas B={B_deltas}, anchors B={B_anch}")
        if not (N_logits == N_deltas == N_anch):
            raise RuntimeError(f"N mismatch among outputs: "
                               f"logits N={N_logits}, deltas N={N_deltas}, anchors N={N_anch}")

        detections = []
        for b in range(B_logits):
            probs = self._sigmoid(cls_logits[b])  # [N,C]
            deltas = bbox_deltas[b]
            anch   = anchors[b]

            for ci, cat_id in enumerate(self.class_ids):
                scores = probs[:, ci]
                keep = scores > self.score_thresh
                if not np.any(keep):
                    continue
                boxes = self._decode_deltas_to_boxes(anch[keep], deltas[keep])
                kept_idx = self._nms(boxes, scores[keep])
                if kept_idx.size == 0:
                    continue
                sel_boxes = boxes[kept_idx]
                sel_scores = scores[keep][kept_idx]
                for (x1, y1, x2, y2), s in zip(sel_boxes, sel_scores):
                    detections.append({
                        "image_id": int(img_ids[b]),
                        "category_id": int(cat_id),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(s),
                    })
        return detections

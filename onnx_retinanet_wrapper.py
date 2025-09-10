#!/usr/bin/env python3
"""
Wrapper around ONNX Runtime RetinaNet export for evaluation.

- Loads the ONNX model (CPU or GPU provider).
- Accepts a batch of preprocessed images [B,3,H,W].
- Returns COCO-format detections: list[dict(image_id, category_id, bbox, score)].

Usage:
    from onnx_retinanet_wrapper import ONNXRetinaNetWrapper

    model = ONNXRetinaNetWrapper(
        onnx_path="retinanet_head.onnx",
        providers=["CUDAExecutionProvider","CPUExecutionProvider"],
        class_ids=[1,2,3],  # COCO IDs for your subset
    )

    dets = model.predict(images, img_ids)
"""

import numpy as np
import onnxruntime as ort
import torch


class ONNXRetinaNetWrapper:
    def __init__(self, onnx_path: str, providers=None, class_ids=None,
                 score_thresh: float = 0.3, iou_thresh: float = 0.5):
        self.sess = ort.InferenceSession(onnx_path, providers=providers or ["CPUExecutionProvider"])
        self.class_ids = class_ids or list(range(1, 2))  # default: 1 class with id=1
        self.num_classes = len(self.class_ids)
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _decode_deltas_to_boxes(self, anchors_xyxy, deltas):
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
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
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
            imgs_np = images.detach().cpu().numpy()
        else:
            imgs_np = np.asarray(images, dtype=np.float32)

        cls_logits, bbox_deltas, anchors = self.sess.run(
            ["cls_logits", "bbox_deltas", "anchors_xyxy"],
            {"images": imgs_np},
        )

        detections = []
        B, N, C = cls_logits.shape
        for b in range(B):
            probs = self._sigmoid(cls_logits[b])  # [N,C]
            deltas = bbox_deltas[b]
            anch = anchors[b]

            for ci, cat_id in enumerate(self.class_ids):
                scores = probs[:, ci]
                keep = scores > self.score_thresh
                if not np.any(keep):
                    continue
                boxes = self._decode_deltas_to_boxes(anch[keep], deltas[keep])
                kept_idx = self._nms(boxes, scores[keep])
                for j in kept_idx:
                    x1, y1, x2, y2 = boxes[j]
                    w = x2 - x1
                    h = y2 - y1
                    detections.append({
                        "image_id": int(img_ids[b]),
                        "category_id": int(cat_id),
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "score": float(scores[keep][j]),
                    })
        return detections

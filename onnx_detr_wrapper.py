#!/usr/bin/env python3
"""
ONNX wrapper for DETR (facebook/detr-resnet-50) exported via export_detr_onnx.py.

Usage in evaluation: feed a batch of torch images [B,3,H,W] in [0,1], use the
Hugging Face DetrImageProcessor to encode pixel_values with a fixed size policy
(shortest_edge/longest_edge). Run ONNX, then post-process with the processor to
get per-image boxes in original coordinates.
"""

from typing import List, Dict, Tuple, Optional

import numpy as np
import onnxruntime as ort
import torch


class ONNXDetrWrapper:
    def __init__(
        self,
        onnx_path: str,
        providers=None,
        class_ids: Optional[List[int]] = None,
        score_thresh: float = 0.05,
        detr_short: int = 800,
        detr_max: int = 800,
        orig_size_map: Optional[Dict[int, Tuple[int, int]]] = None,
    ):
        from transformers import DetrImageProcessor

        requested = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = ort.get_available_providers()
        chosen = [p for p in requested if p in available] or ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, providers=chosen)

        print("[onnxruntime] requested providers:", requested)
        print("[onnxruntime] available providers:", available)
        print("[onnxruntime] using providers:", self.sess.get_providers())

        self.class_ids = class_ids or []  # kept only for parity with retina path
        self.score_thresh = float(score_thresh)
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.size_kwargs = {"shortest_edge": int(detr_short), "longest_edge": int(detr_max)}
        self.orig_size_map = orig_size_map or {}

        # Resolve input/output names (robust to renames)
        self.input_names = {i.name for i in self.sess.get_inputs()}
        self.has_mask = ("pixel_mask" in self.input_names)
        self.out_names = [o.name for o in self.sess.get_outputs()]

    def predict(self, images: List[torch.Tensor], img_ids: List[int]):
        # Convert to numpy images for the processor (uint8 HWC)
        np_imgs = [
            (img.permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()) for img in images
        ]
        enc = self.processor(images=np_imgs, return_tensors="pt", size=self.size_kwargs)
        pixel_values = enc["pixel_values"].numpy().astype(np.float32, copy=False)
        ort_inputs = {"pixel_values": pixel_values}
        if self.has_mask and "pixel_mask" in enc:
            ort_inputs["pixel_mask"] = enc["pixel_mask"].numpy().astype(np.int64, copy=False)

        logits, pred_boxes = self.sess.run(["logits", "pred_boxes"], ort_inputs)

        # Build a minimal outputs-like dict for the processor
        outputs_like = {
            "logits": torch.from_numpy(logits),
            "pred_boxes": torch.from_numpy(pred_boxes),
        }

        # Target sizes: original (H,W) for each image
        sizes = []
        for iid in img_ids:
            oh, ow = self.orig_size_map.get(int(iid), (None, None))
            if oh is None or ow is None:
                # fallback to current tensor size
                t = images[0]
                sizes.append((int(t.shape[-2]), int(t.shape[-1])))
            else:
                sizes.append((int(oh), int(ow)))
        sizes_t = torch.tensor(sizes, dtype=torch.long)

        processed = self.processor.post_process_object_detection(
            outputs_like, target_sizes=sizes_t, threshold=self.score_thresh
        )

        # Convert to COCO detections (xywh in original coords)
        dets = []
        # Map DETR label indices (0..num_labels-1) to dataset category IDs if provided
        # Here we assume class_ids are sorted ds categories; if empty, use raw labels
        for img_id, p in zip(img_ids, processed):
            boxes = p["boxes"].detach().cpu().numpy()
            scores = p["scores"].detach().cpu().numpy()
            labels = p["labels"].detach().cpu().numpy()
            for (x1, y1, x2, y2), s, l in zip(boxes, scores, labels):
                if self.class_ids:
                    cat_id = int(self.class_ids[int(l)])
                else:
                    cat_id = int(l)
                dets.append({
                    "image_id": int(img_id),
                    "category_id": int(cat_id),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(s),
                })
        return dets


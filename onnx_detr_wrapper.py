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
import cv2


class _MinimalDetrProcessor:
    """Lightweight DETR pre/post without transformers dependency.

    - preprocess(images, size): resizes each HWC uint8 image to obey
      shortest_edge/longest_edge, normalizes with ImageNet mean/std,
      pads to the max H,W in batch, returns NCHW float32 and mask.
    - post_process_object_detection(outputs, target_sizes, threshold):
      converts normalized cxcywh boxes to xyxy absolute coords in the
      provided original sizes, filters by score threshold, returns a
      list of dicts with 'boxes','scores','labels' (torch tensors).
    """

    def __init__(self,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1)

    def _resize_keep_ar(self, img: np.ndarray, shortest_edge: int, longest_edge: int) -> np.ndarray:
        h, w = img.shape[:2]
        if shortest_edge is None or shortest_edge <= 0:
            shortest_edge = min(h, w)
        if longest_edge is None or longest_edge <= 0:
            longest_edge = max(h, w)
        scale = min(float(shortest_edge) / max(1.0, float(min(h, w))),
                    float(longest_edge) / max(1.0, float(max(h, w))))
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        if new_h == h and new_w == w:
            return img
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def preprocess(self, images: List[np.ndarray], size: Dict[str, int]):
        se = int(size.get("shortest_edge", 800))
        le = int(size.get("longest_edge", 800))
        proc: List[np.ndarray] = []
        sizes: List[Tuple[int, int]] = []
        for im in images:
            im = self._resize_keep_ar(im, se, le)
            h, w = im.shape[:2]
            sizes.append((h, w))
            # HWC uint8 -> CHW float32 in [0,1]
            chw = im.transpose(2, 0, 1).astype(np.float32) / 255.0
            # normalize
            chw = (chw - self.mean) / self.std
            proc.append(chw)
        max_h = max(s[0] for s in sizes)
        max_w = max(s[1] for s in sizes)
        B = len(proc)
        pixel_values = np.zeros((B, 3, max_h, max_w), dtype=np.float32)
        pixel_mask = np.zeros((B, max_h, max_w), dtype=np.int64)
        for i, (chw, (h, w)) in enumerate(zip(proc, sizes)):
            pixel_values[i, :, :h, :w] = chw
            pixel_mask[i, :h, :w] = 1
        return {"pixel_values": pixel_values, "pixel_mask": pixel_mask}

    @staticmethod
    def _box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
        cx, cy, w, h = x.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def post_process_object_detection(self, outputs_like: Dict[str, torch.Tensor],
                                      target_sizes: torch.Tensor,
                                      threshold: float = 0.05,
                                      input_sizes: Optional[List[Tuple[int, int]]] = None):
        logits: torch.Tensor = outputs_like["logits"]  # [B,Q,C]
        boxes: torch.Tensor = outputs_like["pred_boxes"]  # [B,Q,4] normalized cxcywh
        prob = logits.softmax(-1)
        prob = prob[..., :-1]  # drop no-object class
        scores, labels = prob.max(-1)
        boxes_xyxy = self._box_cxcywh_to_xyxy(boxes)
        results = []
        for i, (b, s, l) in enumerate(zip(boxes_xyxy, scores, labels)):
            # If provided, use the actual unpadded input size (h_in,w_in) to undo normalization
            if input_sizes is not None and i < len(input_sizes) and input_sizes[i] is not None:
                h_in, w_in = int(input_sizes[i][0]), int(input_sizes[i][1])
                in_scale = torch.tensor([w_in, h_in, w_in, h_in], dtype=b.dtype)
                b = b * in_scale
                # Then scale from input size to target (original) size
                h_t, w_t = int(target_sizes[i, 0].item()), int(target_sizes[i, 1].item())
                ratio = torch.tensor([w_t / max(1.0, float(w_in)),
                                      h_t / max(1.0, float(h_in)),
                                      w_t / max(1.0, float(w_in)),
                                      h_t / max(1.0, float(h_in))], dtype=b.dtype)
                b_abs = b * ratio
            else:
                h_t, w_t = int(target_sizes[i, 0].item()), int(target_sizes[i, 1].item())
                scale = torch.tensor([w_t, h_t, w_t, h_t], dtype=b.dtype)
                b_abs = b * scale
            keep = s > float(threshold)
            results.append({
                "boxes": b_abs[keep],
                "scores": s[keep],
                "labels": l[keep],
            })
        return results



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
        requested = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = ort.get_available_providers()
        chosen = [p for p in requested if p in available] or ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, providers=chosen)

        print("[onnxruntime] requested providers:", requested)
        print("[onnxruntime] available providers:", available)
        print("[onnxruntime] using providers:", self.sess.get_providers())

        self.class_ids = class_ids or []  # kept only for parity with retina path
        self.score_thresh = float(score_thresh)
        # Minimal processor that avoids from_pretrained dependency and network calls
        self.processor = _MinimalDetrProcessor()
        self.size_kwargs = {"shortest_edge": int(detr_short), "longest_edge": int(detr_max)}
        self.orig_size_map = orig_size_map or {}

        # Resolve input/output names (robust to renames)
        self.input_names = {i.name for i in self.sess.get_inputs()}
        self.has_mask = ("pixel_mask" in self.input_names)
        self.out_names = [o.name for o in self.sess.get_outputs()]
        # Debug info from last predict call
        self.last_debug: Dict[int, Dict[str, Tuple[int, int]]] = {}

    def predict(self, images: List[torch.Tensor], img_ids: List[int]):
        # Convert to numpy images for the processor (uint8 HWC)
        np_imgs = [(img.permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()) for img in images]
        enc = self.processor.preprocess(np_imgs, size=self.size_kwargs)
        pixel_values = enc["pixel_values"].astype(np.float32, copy=False)
        ort_inputs = {"pixel_values": pixel_values}
        if self.has_mask and ("pixel_mask" in enc):
            ort_inputs["pixel_mask"] = enc["pixel_mask"].astype(np.int64, copy=False)

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

        # Infer actual unpadded input sizes from the pixel_mask if available
        input_sizes: Optional[List[Tuple[int, int]]] = None
        if "pixel_mask" in enc:
            input_sizes = []
            pm = enc["pixel_mask"].astype(np.int64, copy=False)
            for i in range(pm.shape[0]):
                mask = pm[i] > 0
                h_in = int(mask.any(axis=1).sum())
                w_in = int(mask.any(axis=0).sum())
                input_sizes.append((h_in, w_in))

        processed = self.processor.post_process_object_detection(outputs_like, target_sizes=sizes_t, threshold=self.score_thresh, input_sizes=input_sizes)

        # Save debug information per image id
        try:
            self.last_debug = {}
            Hpad = int(pixel_values.shape[2]); Wpad = int(pixel_values.shape[3])
            for idx, iid in enumerate(img_ids):
                oh, ow = sizes[idx]
                if input_sizes is not None and idx < len(input_sizes):
                    h_in, w_in = input_sizes[idx]
                else:
                    h_in, w_in = oh, ow
                # approximate actual scale used (kept AR)
                s = min(float(h_in) / max(1.0, float(oh)), float(w_in) / max(1.0, float(ow)))
                self.last_debug[int(iid)] = {
                    "orig": (int(oh), int(ow)),
                    "input": (int(h_in), int(w_in)),
                    "padded": (int(Hpad), int(Wpad)),
                    "scale": s,
                }
        except Exception:
            pass

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

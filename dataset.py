# dataset.py
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# Optional: Albumentations
try:
    import albumentations as A
    ALBU_OK = True
except Exception:
    ALBU_OK = False

import torchvision.transforms as T
import torchvision.transforms.functional as TF
import cv2


def _load_json(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _pil_to_numpy(img: Image.Image) -> np.ndarray:
    # RGB PIL -> HWC uint8
    return np.array(img)


def _resize_and_pad_np(img_np: np.ndarray, boxes_xyxy: np.ndarray, target: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize so that max(H,W) == target, then pad to target x target (RGB).
    Scale boxes accordingly. boxes_xyxy: (N,4) in xyxy.
    """
    h, w = img_np.shape[:2]
    if target is None or target <= 0:
        return img_np, boxes_xyxy

    scale = float(target) / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # Resize
    resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to target x target
    pad_top = 0
    pad_left = 0
    pad_bottom = target - new_h
    pad_right = target - new_w
    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Scale boxes
    if boxes_xyxy.size > 0:
        boxes = boxes_xyxy.copy().astype(np.float32)
        boxes *= scale  # xyxy scale
    else:
        boxes = boxes_xyxy
    return padded, boxes


class CocoDetDataset(Dataset):
    """
    COCO-style detection dataset with:
      - Contiguous label mapping 0..K-1 for ALL models (RetinaNet, DETR, etc.)
      - Robust bbox cleanup & clipping
      - Optional Albumentations or Torchvision pipeline
      - Optional target-size resize+pad for CPU speed
      - debug_sample() to sanity-check labels/boxes
    """

    def __init__(
        self,
        images_dir: str,
        ann_json: str,
        model_family: str = "retinanet",   # kept only for logging; labels are always 0..K-1
        augment: bool = False,
        use_albu: bool = False,
        albu_strength: str = "light",
        drop_empty: bool = False,          # drop images that end with 0 valid boxes after cleanup
    ):
        super().__init__()
        self.images_dir = images_dir
        self.ann_json = ann_json
        self.model_family = model_family.lower()

        data = _load_json(ann_json)

        # --- Categories and contiguous mapping (0..K-1) ---
        self.categories: List[Dict] = sorted(data["categories"], key=lambda c: c["id"])
        self.cat_ids_sorted: List[int] = [c["id"] for c in self.categories]      # original kept ids
        self.id2label_0based: Dict[int, str] = {i: c["name"] for i, c in enumerate(self.categories)}
        self.label2id_0based: Dict[str, int] = {c["name"]: i for i, c in enumerate(self.categories)}
        self.catid_to_contig: Dict[int, int] = {cid: i for i, cid in enumerate(self.cat_ids_sorted)}
        self.num_classes = len(self.categories)

        # --- Images and annotations index ---
        self.images: List[Dict] = data["images"]
        self.imgid_to_img: Dict[int, Dict] = {im["id"]: im for im in self.images}

        self.anns_by_img: Dict[int, List[Dict]] = defaultdict(list)
        for a in data["annotations"]:
            self.anns_by_img[a["image_id"]].append(a)

        # Optionally drop images with no valid boxes (after cleanup rules)
        if drop_empty:
            keep = []
            for im in self.images:
                img_id = im["id"]
                valid_count = self._count_valid_anns_for_image(im, self.anns_by_img.get(img_id, []))
                if valid_count > 0:
                    keep.append(im)
            self.images = keep

        # --- Aug & size control ---
        self.augment = augment
        self.use_albu = bool(use_albu and ALBU_OK)
        self.albu_strength = albu_strength
        self._target_size = None  # set via set_target_size()

        if self.use_albu:
            self._albu = self._build_albu(albu_strength, augment)
        else:
            # IMPORTANT: no geometric ops here (to keep boxes correct); we handle resize manually.
            self._tfm = T.Compose([T.ToTensor()])  # C,H,W float [0..1]

    # -------- External hook for train.py -------- #
    def set_target_size(self, size: int):
        self._target_size = int(size)

    # ---------- Aug pipelines ---------- #
    def _build_albu(self, strength: str, augment: bool):
        # Build ops list (color/light geo if augment)
        if not augment:
            ops = []
        else:
            if strength == "medium":
                ops = [
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                    A.HueSaturationValue(p=0.3),
                    A.MotionBlur(p=0.2),
                    A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.1, rotate_limit=5,
                                       border_mode=cv2.BORDER_CONSTANT, p=0.4),
                ]
            else:
                ops = [
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.3),
                    A.HueSaturationValue(p=0.2),
                ]

        # Always finish with size + pad if target_size is set
        if self._target_size and self._target_size > 0:
            ops += [
                A.LongestMaxSize(max_size=self._target_size, p=1.0),
                A.PadIfNeeded(min_height=self._target_size, min_width=self._target_size,
                              border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), p=1.0),
            ]
        # Convert to float [0,1]
        ops += [A.ToFloat(max_value=255.0)]

        return A.Compose(
            ops,
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_id"], min_visibility=0.1),
        )

    # ---------- Core helpers ---------- #
    def _count_valid_anns_for_image(self, im_meta: Dict, ann_list: List[Dict]) -> int:
        w, h = im_meta["width"], im_meta["height"]
        cnt = 0
        for a in ann_list:
            if a.get("iscrowd", 0) == 1:
                continue
            x, y, bw, bh = a["bbox"]
            if bw <= 0 or bh <= 0:
                continue
            x1, y1, x2, y2 = x, y, x + bw, y + bh
            x1 = max(0.0, min(x1, w - 1)); y1 = max(0.0, min(y1, h - 1))
            x2 = max(0.0, min(x2, w - 1)); y2 = max(0.0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            if a["category_id"] not in self.catid_to_contig:
                continue
            cnt += 1
        return cnt

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        im_meta = self.images[idx]
        img_id = im_meta["id"]
        w, h = im_meta["width"], im_meta["height"]
        path = os.path.join(self.images_dir, im_meta["file_name"])
        img = Image.open(path).convert("RGB")
        img_np = _pil_to_numpy(img)  # HWC uint8

        # ----- Clean boxes + map labels to 0..K-1 -----
        boxes_xyxy: List[List[float]] = []
        labels_list: List[int] = []

        for ann in self.anns_by_img.get(img_id, []):
            if ann.get("iscrowd", 0) == 1:
                continue

            cid = ann["category_id"]
            if cid not in self.catid_to_contig:
                continue
            contig = self.catid_to_contig[cid]  # 0..K-1

            x, y, bw, bh = ann["bbox"]
            if bw <= 0 or bh <= 0:
                continue
            x1, y1, x2, y2 = x, y, x + bw, y + bh
            # clip to image bounds
            x1 = max(0.0, min(x1, w - 1)); y1 = max(0.0, min(y1, h - 1))
            x2 = max(0.0, min(x2, w - 1)); y2 = max(0.0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            label_val = contig
            boxes_xyxy.append([x1, y1, x2, y2])
            labels_list.append(int(label_val))

        boxes_np = np.array(boxes_xyxy, dtype=np.float32) if boxes_xyxy else np.zeros((0, 4), dtype=np.float32)

        # --- Apply size transform early if NOT using Albumentations ---
        if not self.use_albu and self._target_size and self._target_size > 0:
            img_np, boxes_np = _resize_and_pad_np(img_np, boxes_np, self._target_size)

        # --- Albumentations path ---
        if self.use_albu:
            if len(boxes_xyxy) == 0:
                # Albumentations can crash with empty bboxes; do a no-op on image only
                aug = self._albu(image=img_np)
                img_aug = aug["image"]
                img_t = torch.from_numpy(img_aug.transpose(2, 0, 1)).float() if isinstance(img_aug, np.ndarray) \
                        else torch.as_tensor(img_aug)
                boxes_t = torch.zeros((0, 4), dtype=torch.float32)
                labels_t = torch.zeros((0,), dtype=torch.int64)
            else:
                aug = self._albu(image=img_np, bboxes=boxes_np.tolist(), category_id=labels_list)
                img_aug = aug["image"]; bbs = aug.get("bboxes", []); labs = aug.get("category_id", [])
                img_t = torch.from_numpy(img_aug.transpose(2, 0, 1)).float() if isinstance(img_aug, np.ndarray) \
                        else torch.as_tensor(img_aug)
                boxes_t = torch.as_tensor(bbs, dtype=torch.float32) if bbs else torch.zeros((0, 4), dtype=torch.float32)
                labels_t = torch.as_tensor(labs, dtype=torch.int64) if labs else torch.zeros((0,), dtype=torch.int64)

        # --- Torchvision path (simple: ToTensor only; size already handled) ---
        else:
            img_t = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
            boxes_t = torch.as_tensor(boxes_np, dtype=torch.float32)
            labels_t = torch.as_tensor(labels_list, dtype=torch.int64) if len(labels_list) else torch.zeros((0,), dtype=torch.int64)

        # Final safety clamp (should be no-op if mapping is correct)
        if labels_t.numel():
            labels_t = labels_t.clamp(0, self.num_classes - 1)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor(img_id, dtype=torch.int64),
        }
        return img_t, target

    # ---------- Debug helper ---------- #
    def debug_sample(self, n: int = 100):
        """
        Check label ranges (0..K-1), dtype=int64, box positive areas, and print a small histogram.
        """
        import random as _r
        n = min(n, len(self))
        idxs = _r.sample(range(len(self)), n) if n > 0 else []
        counts = Counter()
        empty_images = 0

        for i in idxs:
            _, t = self[i]
            boxes, labels = t["boxes"], t["labels"]
            if labels.dtype != torch.int64:
                print(f"[warn] labels dtype {labels.dtype} != int64 at index {i}")
            if boxes.numel() == 0:
                empty_images += 1
                continue
            if (labels < 0).any() or (labels >= self.num_classes).any():
                print(f"[warn] label outside [0,{self.num_classes-1}] at {i}: {labels.tolist()}")
            wh = (boxes[:, 2:] - boxes[:, :2])
            if (wh[:, 0] <= 0).any() or (wh[:, 1] <= 0).any():
                print(f"[warn] non-positive area bbox at index {i}: {boxes.tolist()}")
            counts.update(labels.tolist())

        print(f"[debug] Checked {n} samples. Empty-target images: {empty_images}/{n}")
        print(f"[debug] Label histogram (sampled): {dict(counts)}")

    # ---------- Utility: human-readable category summary ---------- #
    def category_summary(self) -> str:
        lines = [f"Total categories: {self.num_classes}"]
        for i, c in enumerate(self.categories):
            lines.append(f"{i:2d} -> COCO_ID {c['id']:>3}  name: {c['name']}")
        return "\n".join(lines)


# ------------- Visualization CLI ------------- #
def _draw_boxes(img: np.ndarray, boxes: np.ndarray, labels: np.ndarray,
                label_map: Dict[int, str], out_path: Path):
    im = img.copy()
    if im.dtype != np.uint8:
        im = (np.clip(im, 0, 1) * 255).astype(np.uint8)
    for b, lab in zip(boxes, labels):
        x1, y1, x2, y2 = [int(round(v)) for v in b.tolist()]
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
        name = label_map.get(int(lab), "?")  # labels already 0..K-1
        cv2.putText(im, name, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="COCO dataset visualizer/debugger")
    ap.add_argument("--images", required=True, help="Path to images dir (e.g., ./data/coco/train2017)")
    ap.add_argument("--ann", required=True, help="Path to COCO annotations JSON")
    ap.add_argument("--model-family", choices=["retinanet", "detr"], default="retinanet")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--use-albu", action="store_true")
    ap.add_argument("--albu-strength", choices=["light", "medium"], default="light")
    ap.add_argument("--target-size", type=int, default=640, help="Resize+pad to this size (square)")
    ap.add_argument("--out-dir", default="./viz_out", help="Folder to save visualized images")
    ap.add_argument("--num", type=int, default=20, help="How many samples to save")
    ap.add_argument("--debug-n", type=int, default=100, help="How many samples to sanity-check")
    args = ap.parse_args()

    ds = CocoDetDataset(
        images_dir=args.images,
        ann_json=args.ann,
        model_family=args.model_family,
        augment=args.augment,
        use_albu=args.use_albu,
        albu_strength=args.albu_strength,
    )
    ds.set_target_size(args.target_size)

    print("[categories]\n" + ds.category_summary())

    # Run the debug sampler
    ds.debug_sample(args.debug_n)

    # Save N visualizations
    N = min(args.num, len(ds))
    id2name = {i: c["name"] for i, c in enumerate(ds.categories)}
    for i in range(N):
        img_t, tgt = ds[i]
        img_np = (img_t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        boxes = tgt["boxes"].cpu().numpy()
        labels = tgt["labels"].cpu().numpy()  # already 0..K-1
        out_path = Path(args.out_dir) / f"sample_{i:04d}.jpg"
        _draw_boxes(img_np, boxes, labels, id2name, out_path)

    print(f"[done] Saved {N} images with boxes to {args.out_dir}")

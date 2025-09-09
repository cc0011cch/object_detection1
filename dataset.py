#!/usr/bin/env python3
# dataset.py
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    A = None  # Albumentations optional

from torchvision.transforms import Compose as C
from torchvision import transforms as T


def _build_img_index(ann: Dict) -> Tuple[List[Dict], Dict[int, Dict], Dict[int, List[Dict]], List[Dict]]:
    images = ann["images"]
    imgid_to_img = {im["id"]: im for im in images}
    imgid_to_anns: Dict[int, List[Dict]] = {}
    for a in ann["annotations"]:
        imgid_to_anns.setdefault(a["image_id"], []).append(a)
    categories = ann["categories"]
    return images, imgid_to_img, imgid_to_anns, categories


class CocoDetDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        ann_json: str,
        model_family: str = "retinanet",  # 'retinanet' | 'detr'
        augment: bool = True,
        use_albu: bool = True,
        albu_strength: str = "light",  # 'light' | 'medium'
    ):
        super().__init__()
        assert model_family in ("retinanet", "detr")
        self.images_dir = images_dir
        with open(ann_json, "r", encoding="utf-8") as f:
            self.ann = json.load(f)

        self.images, self.imgid_to_img, self.imgid_to_anns, self.categories = _build_img_index(self.ann)

        # Build mappings
        cats_sorted = sorted(self.categories, key=lambda c: c["id"])
        # 0-based contiguous mapping for DETR display
        self.idx0_to_name = {i: c["name"] for i, c in enumerate(cats_sorted)}
        # For training targets
        self.catid_to_idx_for_detr = {c["id"]: i for i, c in enumerate(cats_sorted)}      # 0..K-1
        self.catid_to_idx_for_retina = {c["id"]: i + 1 for i, c in enumerate(cats_sorted)} # 1..K

        self.model_family = model_family
        self.augment = augment
        self.use_albu = use_albu and (A is not None)

        # Albumentations pipeline
        self.albu_train = None
        if self.use_albu:
            if albu_strength == "medium":
                aug_list = [
                    A.OneOf([
                        A.RandomBrightnessContrast(p=1.0),
                        A.HueSaturationValue(p=1.0),
                        A.RGBShift(p=1.0),
                    ], p=0.8),
                    A.GaussNoise(p=0.15),
                    A.MotionBlur(blur_limit=3, p=0.1),
                    A.RandomResizedCrop(height=640, width=640, scale=(0.7, 1.0), ratio=(0.9, 1.1), p=0.6),
                    A.HorizontalFlip(p=0.5),
                ]
            else:
                aug_list = [
                    A.RandomBrightnessContrast(p=0.2),
                    A.HueSaturationValue(p=0.2),
                    A.HorizontalFlip(p=0.5),
                    A.LongestMaxSize(max_size=800, p=1.0),
                    A.PadIfNeeded(min_height=800, min_width=800, border_mode=0, p=1.0),
                ]

            self.albu_train = A.Compose(
                aug_list + [ToTensorV2()],
                bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"], min_visibility=0.1),
            )

        # torchvision transforms
        self.torch_train = C([T.PILToTensor()])
        self.torch_eval = C([T.PILToTensor()])

    def __len__(self):
        return len(self.images)

    def _load_image(self, file_name: str) -> Image.Image:
        path = os.path.join(self.images_dir, file_name)
        with Image.open(path) as im:
            im = im.convert("RGB")
            return im.copy()

    def _anns_to_boxes_labels(self, anns: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        boxes, labels_retina, labels_detr, areas = [], [], [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            x2, y2 = x + w, y + h
            boxes.append([x, y, x2, y2])
            cid = a["category_id"]
            labels_retina.append(self.catid_to_idx_for_retina[cid])  # 1..K
            labels_detr.append(self.catid_to_idx_for_detr[cid])      # 0..K-1
            areas.append(w * h)
        if not boxes:
            return np.zeros((0,4),np.float32), np.zeros((0,),np.int64), np.zeros((0,),np.int64), np.zeros((0,),np.float32)
        return (
            np.asarray(boxes, dtype=np.float32),
            np.asarray(labels_retina, dtype=np.int64),
            np.asarray(labels_detr, dtype=np.int64),
            np.asarray(areas, dtype=np.float32),
        )

    def __getitem__(self, idx: int):
        img_info = self.images[idx]
        img_id = img_info["id"]
        file_name = img_info["file_name"]
        im = self._load_image(file_name)
        width, height = im.size
        anns = self.imgid_to_anns.get(img_id, [])
        boxes, labels_retina, labels_detr, areas = self._anns_to_boxes_labels(anns)

        # Albumentations branch
        if self.augment and self.use_albu and self.albu_train is not None and boxes.shape[0] > 0:
            np_img = np.array(im)
            transformed = self.albu_train(image=np_img, bboxes=boxes.tolist(), class_labels=labels_detr.tolist())
            tensor_img = transformed["image"]
            t_boxes = np.array(transformed["bboxes"], dtype=np.float32)
            if self.model_family == "detr":
                t_labels = torch.as_tensor(transformed["class_labels"], dtype=torch.int64)   # 0-based
            else:
                t_labels = torch.as_tensor([l + 1 for l in transformed["class_labels"]], dtype=torch.int64)  # 1-based
            target = {
                "boxes": torch.as_tensor(t_boxes, dtype=torch.float32),
                "labels": t_labels,
                "image_id": torch.as_tensor([img_id]),
            }
            return tensor_img, target

        # torchvision fallback
        tensor_img = (self.torch_train(im) if self.augment else self.torch_eval(im)).float() / 255.0
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels_detr if self.model_family=="detr" else labels_retina, dtype=torch.int64),
            "image_id": torch.as_tensor([img_id]),
        }
        return tensor_img, target

    def category_summary(self) -> str:
        lines = [f"Total categories: {len(self.categories)}"]
        for i, name in self.idx0_to_name.items():
            lines.append(f"{i}: {name}")
        return "\n".join(lines)

    @property
    def num_classes(self) -> int:
        return len(self.categories)


def draw_and_save(img_tensor: torch.Tensor, target: Dict, idx0_to_name: Dict[int, str], model_family: str, out_path: str):
    """Draw bboxes + labels and save to disk."""
    img = img_tensor.permute(1,2,0).cpu().numpy()
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img)
    boxes = target["boxes"].cpu().numpy()
    labels = target["labels"].cpu().numpy()

    for box, lbl in zip(boxes, labels):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="r", facecolor="none")
        ax.add_patch(rect)
        # Map label to name
        if model_family == "retinanet":
            # labels are 1..K -> convert to 0..K-1 for name lookup
            idx0 = int(lbl) - 1
        else:
            idx0 = int(lbl)
        name = idx0_to_name.get(idx0, str(lbl))
        ax.text(x1, max(y1 - 3, 0), name, fontsize=10, color="yellow",
                bbox=dict(facecolor="black", alpha=0.5, pad=1.0))

    ax.axis("off")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-json", required=True, help="Path to COCO subset annotation json")
    ap.add_argument("--images-dir", required=True, help="Path to COCO images folder")
    ap.add_argument("--model", choices=["retinanet","detr"], default="retinanet")
    ap.add_argument("--albu", action="store_true", help="Use Albumentations")
    ap.add_argument("--augment", action="store_true", help="Apply augmentation (train-like)")
    ap.add_argument("--num-samples", type=int, default=5, help="How many images to save")
    ap.add_argument("--save-dir", type=str, default="./debug_samples", help="Folder to save visualizations")
    args = ap.parse_args()

    ds = CocoDetDataset(
        images_dir=args.images_dir,
        ann_json=args.ann_json,
        model_family=args.model,
        augment=args.augment,
        use_albu=args.albu
    )

    print("Category summary:\n", ds.category_summary())

    n = min(args.num_samples, len(ds))
    for i in range(n):
        img, target = ds[i]
        out_path = os.path.join(args.save_dir, f"sample_{i:03d}.jpg")
        draw_and_save(img, target, ds.idx0_to_name, args.model, out_path)
    print(f"[ok] Saved {n} samples to: {args.save_dir}")

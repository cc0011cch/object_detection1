#!/usr/bin/env python3
"""
Quick debug: run best.pth RetinaNet on a few train images and plot detections.
"""
import os
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
import random
from pathlib import Path

from train import build_model_and_helpers  # reuse your train.py builder
from dataset import CocoDetDataset         # your dataset loader

# --- config ---
CKPT = "runs/retina_rfs001/best.pth"
ANN_JSON = "./data/coco/annotations_used/instances_train2017_debug500.json"
IMG_DIR = "./data/coco/train2017"
SAVE_DIR = Path("debug_detections")
NUM_SAMPLES = 5
SCORE_THRESH = 0.3
RESIZE_SHORT = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
# --- dataset (no augmentation) ---
ds = CocoDetDataset(
    images_dir=IMG_DIR,
    ann_json=ANN_JSON,
    augment=False,
    use_albu=False,
)
try:
    ds.set_target_size(RESIZE_SHORT)
except Exception:
    pass

# --- build model exactly as in training ---
cat_ids_sorted = sorted([c["id"] for c in ds.categories])
id2label_0based = getattr(ds, "id2label_0based", {})
label2id_0based = getattr(ds, "label2id_0based", {})

model, _, _, predict_batch_fn = build_model_and_helpers(
    model_name="retinanet",
    num_classes=ds.num_classes,
    id2label_0based=id2label_0based,
    label2id_0based=label2id_0based,
    cat_ids_sorted=cat_ids_sorted,
    device=DEVICE,
)
ckpt = torch.load(CKPT, map_location=DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()

# --- pick random images from dataset ---
idxs = random.sample(range(len(ds)), NUM_SAMPLES)

for idx in idxs:
    img, target = ds[idx]  # img is tensor [3,H,W]
    img_id = target["image_id"].item()
    img_pil = Image.open(os.path.join(IMG_DIR, ds.imgid_to_img[img_id]["file_name"])).convert("RGB")

    # batchify
    with torch.no_grad():
        detections = predict_batch_fn([img.to(DEVICE)], [img_id])

    # show detections
    plt.figure(figsize=(8,8))
    plt.imshow(img_pil)
    ax = plt.gca()

    # --- draw ground truth (green) ---
    gt_boxes = target["boxes"].cpu().numpy() if "boxes" in target else target["bbox"].cpu().numpy()
    gt_labels = target["labels"].cpu().numpy() if "labels" in target else []
    for i, box in enumerate(gt_boxes):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        name = ""
        if len(gt_labels) > i:
            cid = int(gt_labels[i].item())
            name = ds.id2label_0based.get(cid, str(cid))
        rect = plt.Rectangle((x1, y1), w, h, fill=False, color="lime", linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, y1, f"GT {name}", color="lime", fontsize=10)

    for det in detections:
        if det["image_id"] != img_id:
            continue
        if det["score"] < SCORE_THRESH:
            continue
        x, y, w, h = det["bbox"]
        cat = det["category_id"]
        name = [c["name"] for c in ds.categories if c["id"] == cat][0]
        rect = plt.Rectangle((x, y), w, h, fill=False, color="red", linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, f"{name} {det['score']:.2f}", color="yellow", fontsize=10)

    plt.title(f"Image {img_id}")
    plt.axis("off")

    out_path = SAVE_DIR / f"img_{img_id}.jpg"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved detections to {out_path}")

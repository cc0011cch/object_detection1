#!/usr/bin/env python3
# train.py — CPU-friendly, single-script trainer for RetinaNet (torchvision) & DETR (HF)
import argparse
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_

from dataset import CocoDetDataset  # your dataset.py

# ---- logging (added) ----
import logging, sys, io
from contextlib import redirect_stdout

def setup_logger(log_file: str = "", to_console: bool = True, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    if to_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger

# placeholder; will be set in main()
LOG = print
# -------------------------

# Hugging Face (for DETR)
try:
    from transformers import DetrForObjectDetection, DetrImageProcessor
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# COCO eval (optional) and for RFS stats
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    COCO_EVAL_AVAILABLE = True
except Exception:
    COCO_EVAL_AVAILABLE = False

torch.backends.cudnn.benchmark = True


# -------------------- Utils -------------------- #
def set_seed(seed: int = 1337):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def collate_fn(batch):
    imgs, targets = list(zip(*batch))
    return list(imgs), list(targets)


def make_scheduler(optimizer, num_warmup_steps: int, num_training_steps: int):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        progress = float(current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine to 0
    return LambdaLR(optimizer, lr_lambda)


def evaluate_loss(val_loader, val_forward_fn, device, max_batches=None):
    model_loss = 0.0
    n = 0
    for bidx, (images, targets) in enumerate(val_loader):
        if max_batches is not None and bidx >= max_batches:
            break
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
        loss = val_forward_fn(images, targets)
        model_loss += float(loss.detach().cpu().item())
        n += 1
    return model_loss / max(1, n)


def save_checkpoint(state: Dict[str, Any], out_dir: Path, is_best: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, out_dir / "last.pth")
    if is_best:
        torch.save(state, out_dir / "best.pth")
    LOG(f"[checkpoint] Saved last.pth (best={is_best}) in {out_dir}")


def load_checkpoint_if_any(model, optimizer, scheduler, ckpt_path: Path, device):
    if ckpt_path is None or not ckpt_path.exists():
        return 0, float("inf"), None, None
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt and ckpt["scheduler"] is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt.get("epoch", 0)
    best_val = ckpt.get("best_val", float("inf"))
    best_epoch = ckpt.get("best_epoch", None)
    LOG(f"[resume] Loaded checkpoint from {ckpt_path} at epoch {start_epoch} (best_val={best_val:.4f}, best_epoch={best_epoch})")
    return start_epoch, best_val, best_epoch, ckpt.get("extra", None)


# -------------------- Param groups -------------------- #
def split_param_groups_backbone(model, head_lr, backbone_lr, weight_decay):
    bb_decay, bb_nodecay, hd_decay, hd_nodecay = [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_backbone = ("backbone" in n)
        no_decay = (p.ndim <= 1) or n.endswith(".bias")
        if is_backbone:
            (bb_nodecay if no_decay else bb_decay).append(p)
        else:
            (hd_nodecay if no_decay else hd_decay).append(p)
    groups = []
    if bb_decay:
        groups.append({"params": bb_decay, "lr": backbone_lr, "weight_decay": weight_decay, "name": "backbone_decay"})
    if bb_nodecay:
        groups.append({"params": bb_nodecay, "lr": backbone_lr, "weight_decay": 0.0, "name": "backbone_nodecay"})
    if hd_decay:
        groups.append({"params": hd_decay, "lr": head_lr, "weight_decay": weight_decay, "name": "head_decay"})
    if hd_nodecay:
        groups.append({"params": hd_nodecay, "lr": head_lr, "weight_decay": 0.0, "name": "head_nodecay"})
    idx_bb = [i for i, g in enumerate(groups) if g["name"].startswith("backbone")]
    return groups, idx_bb


def detr_param_groups(model, head_lr, backbone_lr, weight_decay):
    no_decay_keywords = ("bias", "LayerNorm.weight", "layer_norm", "norm.weight")
    pos_embed_keywords = ("position_embeddings", "row_embed", "col_embed", "query_position_embeddings")

    def needs_no_decay(n):
        return any(k in n for k in no_decay_keywords) or any(k in n for k in pos_embed_keywords)

    groups = {"bb_decay": [], "bb_nodecay": [], "hd_decay": [], "hd_nodecay": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_backbone = ("backbone" in n)
        nodecay = needs_no_decay(n)
        if is_backbone:
            (groups["bb_nodecay"] if nodecay else groups["bb_decay"]).append(p)
        else:
            (groups["hd_nodecay"] if nodecay else groups["hd_decay"]).append(p)

    param_groups, idx_bb = [], []
    if groups["bb_decay"]:
        idx_bb.append(len(param_groups))
        param_groups.append({"params": groups["bb_decay"], "lr": backbone_lr, "weight_decay": weight_decay, "name": "backbone_decay"})
    if groups["bb_nodecay"]:
        idx_bb.append(len(param_groups))
        param_groups.append({"params": groups["bb_nodecay"], "lr": backbone_lr, "weight_decay": 0.0, "name": "backbone_nodecay"})
    if groups["hd_decay"]:
        param_groups.append({"params": groups["hd_decay"], "lr": head_lr, "weight_decay": weight_decay, "name": "head_decay"})
    if groups["hd_nodecay"]:
        param_groups.append({"params": groups["hd_nodecay"], "lr": head_lr, "weight_decay": 0.0, "name": "head_nodecay"})
    return param_groups, idx_bb


def find_backbone_module(model):
    if hasattr(model, "backbone"):
        return model.backbone
    if hasattr(model, "model") and hasattr(model.model, "backbone"):
        return model.model.backbone
    return None


# -------------------- Repeat Factor Sampler -------------------- #
class RepeatFactorSampler(Sampler[int]):
    """
    LVIS-style Repeat Factor Sampler.
    Given per-index repeat factors r_i, yields each index i ~ ceil(r_i) times in expectation.
    """
    def __init__(self, indices: List[int], repeat_factors: List[float], shuffle: bool = True):
        self.indices = list(indices)
        self.r = list(repeat_factors)
        assert len(self.indices) == len(self.r)
        self.shuffle = shuffle

    def __iter__(self):
        out = []
        for idx, ri in zip(self.indices, self.r):
            m = int(math.floor(ri))
            frac = ri - m
            reps = m + (1 if random.random() < frac else 0)
            if reps > 0:
                out.extend([idx] * reps)
        if self.shuffle:
            random.shuffle(out)
        return iter(out)

    def __len__(self):
        # Upper bound-ish length; not used critically by DataLoader beyond prefetching.
        return int(sum(max(1, int(math.floor(ri)) + (1 if (ri - math.floor(ri)) > 0 else 0)) for ri in self.r))


# ---------- FAST RFS (replaces slow per-image COCO calls) ----------
import json, hashlib, numpy as np
from collections import defaultdict

def _rfs_cache_path(train_ann_path: str, threshold: float, alpha: float) -> Path:
    h = hashlib.md5(f"{train_ann_path}|{threshold}|{alpha}".encode("utf-8")).hexdigest()[:10]
    return Path(train_ann_path).with_suffix(f".rfs_t{threshold}_a{alpha}_{h}.npy")

def compute_repeat_factors_fast(train_ann_path: str,
                                img_ids_dataset_order: List[int],
                                threshold: float,
                                alpha: float = 0.5) -> List[float]:
    """
    Fast RFS:
      - single pass over JSON annotations (no per-image pycocotools calls)
      - caches to .npy next to the ann file
    """
    cache = _rfs_cache_path(train_ann_path, threshold, alpha)
    if cache.exists():
        rf = np.load(cache)
        if len(rf) == len(img_ids_dataset_order):
            LOG(f"[rfs] loaded cached repeat-factors: {cache.name}")
            return rf.tolist()
        else:
            LOG(f"[rfs] cache length mismatch; recomputing...")

    with open(train_ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    N_images = len(coco["images"])

    # Build: images containing each category, and categories present in each image
    imgs_with_cat = defaultdict(set)    # cat_id -> set(image_id)
    cats_in_img   = defaultdict(set)    # image_id -> set(cat_id)
    for a in coco["annotations"]:
        img_id = a["image_id"]
        cat_id = a["category_id"]
        imgs_with_cat[cat_id].add(img_id)
        cats_in_img[img_id].add(cat_id)

    # f_c = (#images containing c) / N_images
    f_c = {c: (len(imgs_with_cat[c]) / max(1, N_images)) for c in imgs_with_cat.keys()}

    # r_c = max(1, (t / f_c)**alpha)
    r_c = {}
    for c, f in f_c.items():
        if f >= threshold:
            r_c[c] = 1.0
        else:
            r_c[c] = (threshold / max(f, 1e-12)) ** alpha

    # Image-level r_i = max_{c in image i} r_c, or 1.0 if no cats
    img_id_to_r = {}
    for img_id in (im["id"] for im in coco["images"]):
        img_cats = cats_in_img.get(img_id, set())
        if not img_cats:
            img_id_to_r[img_id] = 1.0
        else:
            img_id_to_r[img_id] = max(r_c.get(c, 1.0) for c in img_cats)

    # Align with the dataset's sample order
    rf = np.array([img_id_to_r.get(int(img_id), 1.0) for img_id in img_ids_dataset_order], dtype=np.float32)

    # Persist cache
    try:
        np.save(cache, rf)
        LOG(f"[rfs] cached repeat-factors -> {cache.name} (len={len(rf)})")
    except Exception as e:
        LOG(f"[rfs] cache save failed ({e}); continuing without cache.")
    return rf.tolist()
# -------------------------------------------------------------------


def build_imgid_list_for_dataset(ds) -> List[int]:
    """
    Try to obtain per-index image_id for the dataset fast.
    Prefer dataset.images if present; otherwise fall back to other attrs; finally scan.
    """
    # Fast path: your CocoDetDataset has 'images'
    if hasattr(ds, "images"):
        try:
            return [int(im["id"]) for im in ds.images]
        except Exception:
            pass

    for attr in ("img_ids", "image_ids", "ids"):
        if hasattr(ds, attr):
            li = list(getattr(ds, attr))
            return [int(x) for x in li]

    # Fallback (slow): scan the dataset once
    img_ids = []
    for i in range(len(ds)):
        _, t = ds[i]
        img_ids.append(int(t["image_id"]))
    return img_ids


# -------------------- Models & helpers -------------------- #
def build_model_and_helpers(
    model_name: str,
    num_classes: int,
    id2label_0based: Dict[int, str],
    label2id_0based: Dict[str, int],
    cat_ids_sorted: List[int],
    device: torch.device,
):
    model_name = model_name.lower()

    if model_name == "retinanet":
        from torchvision.models.detection import retinanet_resnet50_fpn_v2
        from torchvision.models import ResNet50_Weights

        model = retinanet_resnet50_fpn_v2(
            weights=None,
            weights_backbone=ResNet50_Weights.IMAGENET1K_V2,
            num_classes=num_classes,
        ).to(device)

        def _train_forward(images, targets):
            model.train()
            losses: Dict[str, torch.Tensor] = model(images, targets)
            loss = sum(losses.values())
            return losses, loss

        @torch.no_grad()
        def _val_forward(images, targets):
            model.train()
            losses: Dict[str, torch.Tensor] = model(images, targets)
            return sum(losses.values())

        @torch.no_grad()
        def _predict_batch(images: List[torch.Tensor], img_ids: List[int]):
            model.eval()
            preds = model([img.to(device) for img in images])
            results = []
            for img_id, p in zip(img_ids, preds):
                boxes = p["boxes"].detach().cpu().numpy()
                scores = p["scores"].detach().cpu().numpy()
                labels = p["labels"].detach().cpu().numpy()
                for (x1, y1, x2, y2), s, l in zip(boxes, scores, labels):
                    cat_id = cat_ids_sorted[int(l)]
                    results.append({
                        "image_id": int(img_id),
                        "category_id": int(cat_id),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(s),
                    })
            return results

        return model, _train_forward, _val_forward, _predict_batch

    elif model_name == "detr":
        assert HF_AVAILABLE, "Install transformers to use DETR."
        model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            id2label=id2label_0based,
            label2id=label2id_0based,
        ).to(device)

        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

        def _encode_batch(images: List[torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
            np_imgs = [(img.permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()) for img in images]

            annotations = []
            for t in targets:
                boxes_xyxy = t["boxes"].cpu().numpy()
                labels_0based = t["labels"].cpu().numpy().tolist()

                coco_objs = []
                for (x1, y1, x2, y2), lbl in zip(boxes_xyxy, labels_0based):
                    w = float(x2 - x1); h = float(y2 - y1)
                    coco_objs.append({
                        "bbox": [float(x1), float(y1), w, h],
                        "category_id": int(lbl),
                        "area": float(w * h),
                        "iscrowd": 0,
                    })

                annotations.append({
                    "image_id": int(t["image_id"].item()),
                    "annotations": coco_objs,
                })

            enc = processor(images=np_imgs, annotations=annotations, return_tensors="pt")
            enc["pixel_values"] = enc["pixel_values"].to(device)
            if "pixel_mask" in enc:
                enc["pixel_mask"] = enc["pixel_mask"].to(device)
            if "labels" in enc:
                enc["labels"] = [
                    {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in lab.items()}
                    for lab in enc["labels"]
                ]
            return enc

        def _train_forward(images, targets):
            model.train()
            out = model(**_encode_batch(images, targets))
            loss_dict = dict(getattr(out, "loss_dict", {}))
            loss = out.loss
            if not loss_dict:
                loss_dict = {"loss": loss}
            return loss_dict, loss

        @torch.no_grad()
        def _val_forward(images, targets):
            model.train()
            out = model(**_encode_batch(images, targets))
            return out.loss

        return model, _train_forward, _val_forward, None

    else:
        raise ValueError(f"Unknown model_name: {model_name}. Use 'retinanet' or 'detr'.")


# -------------------- mAP (COCO) -------------------- #
@torch.no_grad()
def coco_map(model_name, model, dl_val, ds_val, device, predict_batch_fn, val_ann_path, print_limit=None):
    if not COCO_EVAL_AVAILABLE:
        LOG("[warn] pycocotools not available; skipping mAP.")
        return None

    detections = []
    seen = 0
    cat_ids_sorted = sorted([c["id"] for c in ds_val.categories])

    if model_name == "detr":
        proc = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    for _, (images, targets) in enumerate(dl_val):
        img_ids = [int(t["image_id"].item()) for t in targets]

        if model_name == "retinanet":
            batch_dets = predict_batch_fn(images, img_ids)
        else:
            model.eval()
            np_imgs = [(img.permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()) for img in images]
            enc = proc(images=np_imgs, return_tensors="pt")
            enc["pixel_values"] = enc["pixel_values"].to(device)
            if "pixel_mask" in enc:
                enc["pixel_mask"] = enc["pixel_mask"].to(device)
            outputs = model(**enc)

            sizes = [(ds_val.imgid_to_img[int(i)]["height"], ds_val.imgid_to_img[int(i)]["width"]) for i in img_ids]
            processed = proc.post_process_object_detection(outputs, target_sizes=torch.tensor(sizes, device=device))

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
        seen += len(img_ids)
        if print_limit is not None and seen >= print_limit:
            break

    coco_gt = COCO(val_ann_path)
    if len(detections) == 0:
        LOG("[warn] No detections; mAP skipped.")
        return None
    coco_dt = coco_gt.loadRes(detections)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()

    # capture the printed table
    buf = io.StringIO()
    with redirect_stdout(buf):
        coco_eval.summarize()
    LOG("\n[COCOeval summary]\n" + buf.getvalue().strip())

    ap, ap50, ap75, aps, apm, apl = [float(coco_eval.stats[i]) for i in range(6)]
    LOG(f"[mAP] AP@[.5:.95]={ap:.4f} | AP50={ap50:.4f} | AP75={ap75:.4f} | APs={aps:.4f} APm={apm:.4f} APl={apl:.4f}")
    return {"AP": ap, "AP50": ap50, "AP75": ap75, "APS": aps, "APM": apm, "APL": apl}


# -------------------- Main -------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["retinanet", "detr"], required=True)
    ap.add_argument("--train-ann", required=True)
    ap.add_argument("--val-ann", required=True)
    ap.add_argument("--train-images", default="./data/coco/train2017")
    ap.add_argument("--val-images", default="./data/coco/val2017")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2)

    # Fine-tuning knobs
    ap.add_argument("--head-lr", type=float, default=1e-3, help="LR for heads (RetinaNet) / non-backbone (DETR)")
    ap.add_argument("--backbone-lr", type=float, default=1e-4, help="LR for backbone")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--freeze-backbone-epochs", type=int, default=0, help="Set backbone LR=0 for first N epochs")
    ap.add_argument("--freeze-bn-when-frozen", action="store_true",
                    help="When freezing backbone, also set its BatchNorm layers to eval().")

    # Loader / CPU performance
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--num-threads", type=int, default=4, help="Torch intra/inter-op threads")
    ap.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch_factor (>=2)")
    ap.add_argument("--persistent-workers", action="store_true", help="Keep worker processes alive")
    ap.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation steps")
    ap.add_argument("--resize-short", type=int, default=640, help="Resize short side if dataset supports")

    # Logging/checkpointing/eval
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--out", default="./runs/exp1")
    ap.add_argument("--early-stop-patience", type=int, default=5)
    ap.add_argument("--resume", default="")
    ap.add_argument("--albu", action="store_true")
    ap.add_argument("--albu-strength", choices=["light", "medium"], default="light")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-train-batches", type=int, default=None)
    ap.add_argument("--max-val-batches", type=int, default=None)
    ap.add_argument("--print-freq", type=int, default=10)

    # ---------- RFS ----------
    ap.add_argument("--rfs", type=float, default=0.0,
                    help="Repeat-Factor Sampling threshold t (e.g., 0.001). Set 0 to disable.")
    ap.add_argument("--rfsAlpha", type=float, default=0.5,
                    help="Exponent alpha for RFS (LVIS uses 0.5 for sqrt).")
    # -------------------------

    # ---- logging flags (added) ----
    ap.add_argument("--log-file", default="", help="Path to save logs (in addition to console).")
    ap.add_argument("--log-console", action="store_true", help="Also print logs to stdout (default off).")
    # -------------------------------

    args = ap.parse_args()

    # init logger
    global LOG
    logger = setup_logger(args.log_file, to_console=args.log_console)
    LOG = logger.info

    # Threads control for CPU speed
    torch.set_num_threads(max(1, args.num_threads))
    torch.set_num_interop_threads(max(1, args.num_threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(args.num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.num_threads))

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG(f"[info] device={device} | torch_threads={args.num_threads}")

    # Datasets
    ds_train = CocoDetDataset(
        images_dir=args.train_images,
        ann_json=args.train_ann,
        model_family=args.model,
        augment=True,
        use_albu=args.albu,
        albu_strength=args.albu_strength,
    )
    ds_val = CocoDetDataset(
        images_dir=args.val_images,
        ann_json=args.val_ann,
        model_family=args.model,
        augment=False,
        use_albu=False,
    )

    # (Optional) if your dataset supports a dynamic resize hook, use it
    for ds in (ds_train, ds_val):
        if hasattr(ds, "set_target_size"):
            try:
                ds.set_target_size(args.resize_short)
                LOG(f"[dataset] set_target_size({args.resize_short}) applied.")
            except Exception:
                pass

    cat_ids_sorted = sorted([c["id"] for c in ds_train.categories])
    LOG("[train categories]\n" + ds_train.category_summary())

    # --- Inspect image-level class frequencies WITHOUT touching ds_train internals ---
    if COCO_EVAL_AVAILABLE:
        LOG("[image frequency] f(c) = fraction of images containing class c (via COCO ann)")
        coco_tr = COCO(args.train_ann)
        N_images = len(coco_tr.imgs)
        for label_idx, coco_cat_id in enumerate(cat_ids_sorted):
            img_ids_for_c = set(coco_tr.getImgIds(catIds=[coco_cat_id]))
            cnt = len(img_ids_for_c)
            f = cnt / max(1, N_images)
            name = getattr(ds_train, "id2label_0based", {}).get(label_idx, str(label_idx))
            LOG(f"  class {label_idx:>3} ({name:>10}): f(c)={f:.6f}  ({cnt}/{N_images} images)")
    else:
        LOG("[warn] pycocotools not available; skip image-frequency printout.")
    # -------------------------------------------------------------------------

    # --------- Build (optional) Repeat-Factor Sampler ---------
    train_sampler = None
    if args.rfs and args.rfs > 0.0:
        if not COCO_EVAL_AVAILABLE:
            LOG("[warn] --rfs specified but pycocotools not available. RFS disabled.")
        else:
            LOG(f"[rfs] enabled with threshold t={args.rfs}, alpha={args.rfsAlpha}")
            # per-index image_id for the dataset (fast)
            ds_img_ids = build_imgid_list_for_dataset(ds_train)
            rf = compute_repeat_factors_fast(args.train_ann, ds_img_ids, threshold=args.rfs, alpha=args.rfsAlpha)
            train_indices = list(range(len(ds_train)))
            train_sampler = RepeatFactorSampler(train_indices, rf, shuffle=True)
            LOG(f"[rfs] constructed sampler with base N={len(train_indices)}; "
                f"approx effective length ≈ {len(train_sampler)}")
    # ----------------------------------------------------------

    # DataLoaders (faster on CPU)
    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,  # CPU training
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
        persistent_workers=(args.persistent_workers and args.num_workers > 0),
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
        persistent_workers=(args.persistent_workers and args.num_workers > 0),
    )

    steps_per_epoch = len(dl_train)
    if args.max_train_batches is not None:
        steps_per_epoch = min(steps_per_epoch, args.max_train_batches)

    # Model & helpers
    model, train_forward, val_forward, predict_batch = build_model_and_helpers(
        model_name=args.model,
        num_classes=ds_train.num_classes,
        id2label_0based=getattr(ds_train, "id2label_0based", {}),
        label2id_0based=getattr(ds_train, "label2id_0based", {}),
        cat_ids_sorted=cat_ids_sorted,
        device=device,
    )

    # Param groups
    if args.model == "retinanet":
        param_groups, idx_bb_groups = split_param_groups_backbone(
            model, head_lr=args.head_lr, backbone_lr=args.backbone_lr, weight_decay=args.weight_decay
        )
    else:
        param_groups, idx_bb_groups = detr_param_groups(
            model, head_lr=args.head_lr, backbone_lr=args.backbone_lr, weight_decay=args.weight_decay
        )

    # Initial freeze (LR=0) for backbone groups
    if args.freeze_backbone_epochs > 0:
        for gi in idx_bb_groups:
            param_groups[gi]["lr"] = 0.0

    optimizer = AdamW(param_groups)
    steps_per_epoch = max(1, steps_per_epoch)
    total_steps = steps_per_epoch * args.epochs
    scheduler = make_scheduler(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    # AMP
    use_amp = (device.type == "cuda")
    bf16_supported = bool(use_amp and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
    amp_dtype = torch.bfloat16 if bf16_supported else torch.float16

    try:
        scaler = torch.amp.GradScaler(device_type="cuda", enabled=use_amp and (amp_dtype is torch.float16))
        has_new_scaler = True
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and (amp_dtype is torch.float16))
        has_new_scaler = False

    try:
        from torch.amp import autocast as autocast_new
        has_new_autocast = True
    except Exception:
        has_new_autocast = False

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir))
    LOG(f"[info] Checkpoints & logs will be saved to: {out_dir}")

    # DETR prediction closure (only for final mAP)
    detr_proc = None
    if args.model == "detr":
        detr_proc = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

        @torch.no_grad()
        def predict_batch(images: List[torch.Tensor], img_ids: List[int]):
            model.eval()
            np_imgs = [(img.permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()) for img in images]
            enc = detr_proc(images=np_imgs, return_tensors="pt")
            enc["pixel_values"] = enc["pixel_values"].to(device)
            if "pixel_mask" in enc:
                enc["pixel_mask"] = enc["pixel_mask"].to(device)
            outputs = model(**enc)
            sizes = [(ds_val.imgid_to_img[int(i)]["height"], ds_val.imgid_to_img[int(i)]["width"]) for i in img_ids]
            processed = detr_proc.post_process_object_detection(outputs, target_sizes=torch.tensor(sizes, device=device))
            results = []
            for img_id, p in zip(img_ids, processed):
                boxes = p["boxes"].detach().cpu().numpy()
                scores = p["scores"].detach().cpu().numpy()
                labels = p["labels"].detach().cpu().numpy()
                for (x1, y1, x2, y2), s, l in zip(boxes, scores, labels):
                    cat_id = cat_ids_sorted[int(l)]
                    results.append({"image_id": int(img_id),
                                    "category_id": int(cat_id),
                                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                    "score": float(s)})
            return results

    # Backbone module for BN control
    backbone_module = find_backbone_module(model)

    # Resume
    start_epoch, best_val, best_epoch, _ = 0, float("inf"), None, None
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.is_file():
            start_epoch, best_val, best_epoch, _ = load_checkpoint_if_any(model, optimizer, scheduler, ckpt_path, device)

    patience = args.early_stop_patience
    epochs_no_improve = 0
    global_step = 0
    accum = max(1, args.accum_steps)

    for epoch in range(start_epoch, args.epochs):
        # Freeze/unfreeze backbone schedule
        if args.freeze_backbone_epochs > 0:
            if epoch < args.freeze_backbone_epochs:
                if args.freeze_bn_when_frozen and backbone_module is not None:
                    for m in backbone_module.modules():
                        if isinstance(m, torch.nn.BatchNorm2d):
                            m.eval()
            elif epoch == args.freeze_backbone_epochs:
                for gi in idx_bb_groups:
                    param_groups[gi]["lr"] = args.backbone_lr
                LOG(f"[unfreeze] Epoch {epoch}: backbone LR -> {args.backbone_lr}")
                if args.freeze_bn_when_frozen and backbone_module is not None:
                    for m in backbone_module.modules():
                        if isinstance(m, torch.nn.BatchNorm2d):
                            m.train()

        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)

        # ---- grad clipping stats for this epoch ----
        clip_total_steps = 0
        clip_triggered_steps = 0
        grad_norms_epoch: List[float] = []

        for bidx, (images, targets) in enumerate(dl_train):
            if args.max_train_batches is not None and bidx >= args.max_train_batches:
                break

            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

            # forward + loss
            if use_amp:
                if has_new_autocast:
                    cm = autocast_new('cuda', dtype=amp_dtype)
                else:
                    cm = torch.cuda.amp.autocast(dtype=amp_dtype)
                with cm:
                    loss_dict, loss = train_forward(images, targets)
            else:
                loss_dict, loss = train_forward(images, targets)

            # grad accumulation
            loss = loss / accum
            if use_amp and (amp_dtype is torch.float16):
                scaler.scale(loss).backward()
            else:
                loss.backward()

            step_now = ((bidx + 1) % accum == 0)
            if step_now:
                # ---- gradient clipping (pre-step) ----
                if use_amp and (amp_dtype is torch.float16):
                    scaler.unscale_(optimizer)
                grad_norm = clip_grad_norm_(model.parameters(), max_norm=0.1)
                clip_total_steps += 1
                grad_norms_epoch.append(float(grad_norm))
                if float(grad_norm) > 0.1:
                    clip_triggered_steps += 1

                # Optimizer step
                if use_amp and (amp_dtype is torch.float16):
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            # ----- logging -----
            if isinstance(loss_dict, dict):
                preferred = ["classification", "bbox_regression", "giou", "loss_ce", "loss_bbox", "loss_giou"]
                ordered_keys = [k for k in preferred if k in loss_dict] + [k for k in loss_dict if k not in preferred]
                parts = [f"{(k)}:{(loss_dict[k].item() if torch.is_tensor(loss_dict[k]) else float(loss_dict[k])):.4f}" for k in ordered_keys]
                comp = " ".join(parts)
            else:
                comp = f"loss:{float(loss_dict):.4f}"

            disp_loss = float((loss * accum).detach().cpu().item())
            running += disp_loss
            writer.add_scalar("train/loss_step", disp_loss, global_step)

            if bidx % args.print_freq == 0:
                cur_epoch = epoch + 1
                cur_batch = bidx + 1
                LOG(f"Epoch: {cur_epoch}/{args.epochs}, "
                    f"Batches: {cur_batch}/{steps_per_epoch}, "
                    f"{comp} total:{disp_loss:.4f}")

            global_step += 1

        # flush leftover grads if dataloader length not divisible by accum
        if (bidx + 1) % accum != 0:
            if use_amp and (amp_dtype is torch.float16):
                scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm_(model.parameters(), max_norm=0.1)
            clip_total_steps += 1
            grad_norms_epoch.append(float(grad_norm))
            if float(grad_norm) > 0.1:
                clip_triggered_steps += 1

            if use_amp and (amp_dtype is torch.float16):
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        train_epoch_loss = running / max(1, (bidx + 1))
        writer.add_scalar("train/loss_epoch", train_epoch_loss, epoch)

        # ---- log clipping stats for this epoch ----
        if clip_total_steps > 0:
            clip_pct = 100.0 * clip_triggered_steps / clip_total_steps
            LOG(f"[clip] {clip_triggered_steps}/{clip_total_steps} steps clipped ({clip_pct:.1f}%)")
            writer.add_scalar("train/clip_pct", clip_pct, epoch)
            writer.add_scalar("train/clip_count", clip_triggered_steps, epoch)
            try:
                if len(grad_norms_epoch) > 0:
                    writer.add_histogram("train/grad_norm", torch.tensor(grad_norms_epoch), epoch)
            except Exception:
                pass

        # Val loss (fp32) — keep your previous customization here if needed
#        val_loss = 21.0
        val_loss = evaluate_loss(dl_val, lambda im, tg: val_forward(im, tg), device, max_batches=args.max_val_batches)
        writer.add_scalar("val/loss_epoch", val_loss, epoch)
        LOG(f"[Eval] Epoch {epoch+1} finished. train_loss={train_epoch_loss:.4f}, val_loss={val_loss:.4f}")

        # Save / early stop
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "best_val": best_val,
                "best_epoch": best_epoch,
                "args": vars(args),
            },
            out_dir=out_dir,
            is_best=is_best,
        )

        if patience is not None and epochs_no_improve >= patience:
            LOG(f"[early stop] No improvement for {patience} epochs. Best val loss={best_val:.4f} (epoch {best_epoch})")
            break

    # -------- Final mAP: best checkpoint only --------
    if COCO_EVAL_AVAILABLE:
        best_ckpt = out_dir / "best.pth"
        if best_ckpt.exists():
            LOG(f"[mAP] Loading best checkpoint from {best_ckpt} for final COCO eval...")
            ckpt = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(ckpt["model"])
            be = ckpt.get("best_epoch", best_epoch)
            bv = ckpt.get("best_val", best_val)
            LOG(f"[best] epoch={be}, val_loss={bv:.4f}")

            if args.model == "detr":
                proc = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

                @torch.no_grad()
                def predict_batch(images: List[torch.Tensor], img_ids: List[int]):
                    model.eval()
                    np_imgs = [(img.permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()) for img in images]
                    enc = proc(images=np_imgs, return_tensors="pt")
                    enc["pixel_values"] = enc["pixel_values"].to(device)
                    if "pixel_mask" in enc:
                        enc["pixel_mask"] = enc["pixel_mask"].to(device)
                    outputs = model(**enc)
                    sizes = [(ds_val.imgid_to_img[int(i)]["height"], ds_val.imgid_to_img[int(i)]["width"]) for i in img_ids]
                    processed = proc.post_process_object_detection(outputs, target_sizes=torch.tensor(sizes, device=device))
                    cat_ids_sorted_local = sorted([c["id"] for c in ds_val.categories])
                    results = []
                    for img_id, p in zip(img_ids, processed):
                        boxes = p["boxes"].detach().cpu().numpy()
                        scores = p["scores"].detach().cpu().numpy()
                        labels = p["labels"].detach().cpu().numpy()
                        for (x1, y1, x2, y2), s, l in zip(boxes, scores, labels):
                            cat_id = cat_ids_sorted_local[int(l)]
                            results.append({
                                "image_id": int(img_id),
                                "category_id": int(cat_id),
                                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                "score": float(s),
                            })
                    return results

            coco_map(
                model_name=args.model,
                model=model,
                dl_val=dl_val,
                ds_val=ds_val,
                device=device,
                predict_batch_fn=predict_batch,
                val_ann_path=args.val_ann,
            )
        else:
            LOG("[warn] best.pth not found; skipping final mAP.")
    else:
        LOG("[warn] pycocotools not found; skipping final mAP.")

    writer.close()
    LOG(f"[done] Best model: epoch {best_epoch} with val_loss={best_val:.4f}. Checkpoints at {args.out}")


if __name__ == "__main__":
    main()

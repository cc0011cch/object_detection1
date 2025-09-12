#!/usr/bin/env python3
"""
Modular training entrypoint for RetinaNet (torchvision) and DETR (HF).

This script now delegates most logic to engine/ and models/ modules and keeps
the CLI thin. Public functions collate_fn and build_model_and_helpers are
re-exported for compatibility with other scripts (evaluate_test.py, etc.).
"""
import argparse
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from dataset import CocoDetDataset

# Re-exported utilities for compatibility
from engine.data_utils import set_seed, collate_fn  # re-exported
from models.factory import build_model_and_helpers  # re-exported

from engine.logging_utils import setup_logger
from engine.sampler import RepeatFactorSampler
from engine.rfs import compute_repeat_factors_fast, build_imgid_list_for_dataset
from engine.eval import COCO_EVAL_AVAILABLE
from engine.trainer import Trainer
from models.factory import split_param_groups_backbone, detr_param_groups


torch.backends.cudnn.benchmark = True

# Explicit re-exports for backward compatibility
__all__ = [
    "build_model_and_helpers",
    "collate_fn",
    "set_seed",
]


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
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--backbone-lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--freeze-backbone-epochs", type=int, default=1)
    ap.add_argument("--freeze-bn-when-frozen", action="store_true",
                    help="Keep BatchNorm in backbone at eval() while frozen")
    ap.add_argument("--grad-clip", type=float, default=None)

    # Loader / CPU performance
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--num-threads", type=int, default=4, help="Torch intra/inter-op threads")
    ap.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch_factor (>=2)")
    ap.add_argument("--persistent-workers", action="store_true", help="Keep worker processes alive")
    ap.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation steps")
    ap.add_argument("--resize-short", type=int, default=640, help="Resize short side if dataset supports")
    ap.add_argument("--amp", choices=["auto", "fp16", "bf16", "off"], default="auto",
                    help="Mixed precision mode: auto (prefer bf16 on supported GPUs), fp16, bf16, or off")
    ap.add_argument("--empty-cache-every", type=int, default=0,
                    help="If >0, call torch.cuda.empty_cache() every N steps to smooth nvidia-smi memory")

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

    # RFS (Repeat-Factor Sampling)
    ap.add_argument("--rfs", type=float, default=0.0,
                    help="Repeat-Factor Sampling threshold t (e.g., 0.001). Set 0 to disable.")
    ap.add_argument("--rfsAlpha", type=float, default=0.5,
                    help="Exponent alpha for RFS (LVIS uses 0.5 for sqrt).")

    # Metric evaluation controls
    ap.add_argument("--eval-map-every", type=int, default=0,
                    help="If >0, compute COCO mAP/Macro-mAP every N epochs (costly). 0=only at the very end.")
    ap.add_argument("--eval-map-max-batches", type=int, default=None,
                    help="Cap #val batches used for mAP each epoch (speed/debug). None=all.")
    ap.add_argument("--early-metric", choices=["val_loss", "coco_ap", "macro_map", "macro_ap50"],
                    default="val_loss",
                    help="Metric used to select best checkpoint & early stop.")

    # Logging flags
    ap.add_argument("--log-file", default="", help="Path to save logs (in addition to console).")
    ap.add_argument("--log-console", action="store_true", help="Also print logs to stdout (default off).")

    args = ap.parse_args()

    # init logger and threads
    logger = setup_logger(args.log_file, to_console=args.log_console)
    torch.set_num_threads(max(1, args.num_threads))
    torch.set_num_interop_threads(max(1, args.num_threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(args.num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.num_threads))

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[info] device={device} | torch_threads={args.num_threads}")

    # Datasets
    ds_train = CocoDetDataset(
        images_dir=args.train_images,
        ann_json=args.train_ann,
        augment=True,
        use_albu=args.albu,
        albu_strength=args.albu_strength,
    )
    ds_val = CocoDetDataset(
        images_dir=args.val_images,
        ann_json=args.val_ann,
        augment=False,
        use_albu=False,
    )

    for ds in (ds_train, ds_val):
        if hasattr(ds, "set_target_size"):
            try:
                ds.set_target_size(args.resize_short)
                logger.info(f"[dataset] set_target_size({args.resize_short}) applied.")
            except Exception:
                pass

    cat_ids_sorted = sorted([c["id"] for c in ds_train.categories])
    logger.info("[train categories]\n" + ds_train.category_summary())

    # Optional: class frequency printout (requires pycocotools)
    if COCO_EVAL_AVAILABLE:
        from pycocotools.coco import COCO
        logger.info("[image frequency] f(c) = fraction of images containing class c (via COCO ann)")
        coco_tr = COCO(args.train_ann)
        N_images = len(coco_tr.imgs)
        for label_idx, coco_cat_id in enumerate(cat_ids_sorted):
            img_ids_for_c = set(coco_tr.getImgIds(catIds=[coco_cat_id]))
            cnt = len(img_ids_for_c)
            f = cnt / max(1, N_images)
            name = getattr(ds_train, "id2label_0based", {}).get(label_idx, str(label_idx))
            logger.info(f"  class {label_idx:>3} ({name:>10}): f(c)={f:.6f}  ({cnt}/{N_images} images)")
    else:
        logger.info("[warn] pycocotools not available; skip image-frequency printout.")

    # Optional RFS sampler
    train_sampler = None
    if args.rfs and args.rfs > 0.0:
        if not COCO_EVAL_AVAILABLE:
            logger.info("[warn] --rfs specified but pycocotools not available. RFS disabled.")
        else:
            logger.info(f"[rfs] enabled with threshold t={args.rfs}, alpha={args.rfsAlpha}")
            ds_img_ids = build_imgid_list_for_dataset(ds_train)
            rf = compute_repeat_factors_fast(args.train_ann, ds_img_ids, threshold=args.rfs, alpha=args.rfsAlpha)
            train_indices = list(range(len(ds_train)))
            train_sampler = RepeatFactorSampler(train_indices, rf, shuffle=True)
            logger.info(
                f"[rfs] constructed sampler with base N={len(train_indices)}; approx effective length ≈ {len(train_sampler)}"
            )

    # DataLoaders
    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
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

    # Model & helpers
    # Build a map image_id -> (height, width) for original sizes
    orig_size_map = {int(im["id"]): (int(im["height"]), int(im["width"])) for im in ds_val.images}

    model, train_forward, val_forward, predict_batch = build_model_and_helpers(
        model_name=args.model,
        num_classes=ds_train.num_classes,
        id2label_0based=getattr(ds_train, "id2label_0based", {}),
        label2id_0based=getattr(ds_train, "label2id_0based", {}),
        cat_ids_sorted=cat_ids_sorted,
        device=device,
        orig_size_map=orig_size_map,
    )

    # Param groups and trainer
    if args.model == "retinanet":
        param_groups, idx_bb_groups = split_param_groups_backbone(
            model, head_lr=args.head_lr, backbone_lr=args.backbone_lr, weight_decay=args.weight_decay
        )
    else:
        param_groups, idx_bb_groups = detr_param_groups(
            model, head_lr=args.head_lr, backbone_lr=args.backbone_lr, weight_decay=args.weight_decay
        )

    trainer = Trainer(
        args=args,
        model_name=args.model,
        model=model,
        train_forward=train_forward,
        val_forward=val_forward,
        predict_batch=predict_batch,
        dl_train=dl_train,
        dl_val=dl_val,
        ds_train=ds_train,
        ds_val=ds_val,
        device=device,
        param_groups=param_groups,
        idx_bb_groups=idx_bb_groups,
    )

    logger.info(f"[info] Checkpoints & logs will be saved to: {Path(args.out)}")
    trainer.fit()


if __name__ == "__main__":
    main()

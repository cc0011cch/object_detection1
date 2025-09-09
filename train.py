#!/usr/bin/env python3
# train.py
import argparse
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import CocoDetDataset

# Optional HF imports for DETR
try:
    from transformers import DetrForObjectDetection, DetrImageProcessor
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# COCO eval
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# -------------------- Utilities -------------------- #
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
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def evaluate_loss(val_loader, val_forward_fn, device, scaler=None, amp=False, max_batches=None):
    model_loss = 0.0
    n = 0
    for bidx, (images, targets) in enumerate(val_loader):
        if max_batches is not None and bidx >= max_batches:
            break
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
        if amp and scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                loss = val_forward_fn(images, targets)
        else:
            loss = val_forward_fn(images, targets)
        model_loss += float(loss.detach().cpu().item())
        n += 1
    return model_loss / max(1, n)


def save_checkpoint(state: Dict[str, Any], out_dir: Path, is_best: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, out_dir / "last.pth")
    if is_best:
        torch.save(state, out_dir / "best.pth")
    print(f"[checkpoint] Saved last.pth (best={is_best}) in {out_dir}")


def load_checkpoint_if_any(model, optimizer, scheduler, ckpt_path: Path, device):
    if ckpt_path is None or not ckpt_path.exists():
        return 0, float("inf"), None
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt and ckpt["scheduler"] is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt.get("epoch", 0)
    best_val = ckpt.get("best_val", float("inf"))
    print(f"[resume] Loaded checkpoint from {ckpt_path} at epoch {start_epoch} (best_val={best_val:.4f})")
    return start_epoch, best_val, ckpt.get("extra", None)


# -------------------- Param groups (discriminative LR) -------------------- #
def _split_decay(named_params):
    decay, no_decay = [], []
    for n, p in named_params:
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay


def retinanet_param_groups(model, head_lr, backbone_lr, weight_decay, freeze_bn_when_frozen):
    """
    Returns param_groups + index set of backbone groups to toggle LR later.
    """
    backbone = model.backbone     # CNN body + FPN
    head = model.head             # cls & bbox subnets

    # Optionally put BN of backbone into eval mode while LR==0 (done in main each epoch)
    def set_bn_eval(m):
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()

    head_decay, head_no_decay = _split_decay(list(head.named_parameters()))
    body_decay, body_no_decay = _split_decay([(f"backbone.{n}", p) for n, p in backbone.named_parameters()])

    groups = []
    idx_backbone_groups = []

    if body_decay:
        idx_backbone_groups.append(len(groups))
        groups.append({"params": body_decay, "lr": backbone_lr, "weight_decay": weight_decay, "name": "backbone_decay"})
    if body_no_decay:
        idx_backbone_groups.append(len(groups))
        groups.append({"params": body_no_decay, "lr": backbone_lr, "weight_decay": 0.0, "name": "backbone_nodecay"})
    if head_decay:
        groups.append({"params": head_decay, "lr": head_lr, "weight_decay": weight_decay, "name": "head_decay"})
    if head_no_decay:
        groups.append({"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0, "name": "head_nodecay"})

    return groups, idx_backbone_groups, backbone, set_bn_eval if freeze_bn_when_frozen else None


def detr_param_groups(model, head_lr, backbone_lr, weight_decay):
    """
    HF DETR: smaller LR for backbone, no-decay for bias/Norm/positional embeddings.
    Returns param_groups + index set of backbone groups.
    """
    no_decay_keywords = ("bias", "LayerNorm.weight", "layer_norm", "norm.weight")
    pos_embed_keywords = ("position_embeddings", "row_embed", "col_embed", "query_position_embeddings")

    def needs_no_decay(n):
        return any(k in n for k in no_decay_keywords) or any(k in n for k in pos_embed_keywords)

    groups = {"bb_decay": [], "bb_nodecay": [], "head_decay": [], "head_nodecay": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_backbone = ("backbone" in n)
        nodecay = needs_no_decay(n)
        if is_backbone:
            (groups["bb_nodecay"] if nodecay else groups["bb_decay"]).append(p)
        else:
            (groups["head_nodecay"] if nodecay else groups["head_decay"]).append(p)

    param_groups = []
    idx_backbone_groups = []
    if groups["bb_decay"]:
        idx_backbone_groups.append(len(param_groups))
        param_groups.append({"params": groups["bb_decay"], "lr": backbone_lr, "weight_decay": weight_decay, "name": "backbone_decay"})
    if groups["bb_nodecay"]:
        idx_backbone_groups.append(len(param_groups))
        param_groups.append({"params": groups["bb_nodecay"], "lr": backbone_lr, "weight_decay": 0.0, "name": "backbone_nodecay"})
    if groups["head_decay"]:
        param_groups.append({"params": groups["head_decay"], "lr": head_lr, "weight_decay": weight_decay, "name": "head_decay"})
    if groups["head_nodecay"]:
        param_groups.append({"params": groups["head_nodecay"], "lr": head_lr, "weight_decay": 0.0, "name": "head_nodecay"})

    return param_groups, idx_backbone_groups


# -------------------- Models & predictors -------------------- #
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
        from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
        weights = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
        model = retinanet_resnet50_fpn_v2(weights=weights, num_classes=num_classes).to(device)

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
                labels = p["labels"].detach().cpu().numpy()  # 1..K
                for (x1, y1, x2, y2), s, l in zip(boxes, scores, labels):
                    cat_id = cat_ids_sorted[int(l) - 1]
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
                boxes_xyxy = t["boxes"].cpu().numpy().tolist()
                labels = t["labels"].cpu().numpy().tolist()  # 0-based
                anns = [{"bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                         "category_id": int(lbl)} for (x1, y1, x2, y2), lbl in zip(boxes_xyxy, labels)]
                annotations.append({"image_id": int(t["image_id"].item()), "annotations": anns})
            enc = processor(images=np_imgs, annotations=annotations, return_tensors="pt")
            for k in enc:
                enc[k] = enc[k].to(device)
            return enc

        def _train_forward(images, targets):
            model.train()
            out = model(**_encode_batch(images, targets))
            return {"loss": out.loss}, out.loss

        @torch.no_grad()
        def _val_forward(images, targets):
            model.train()
            out = model(**_encode_batch(images, targets))
            return out.loss

        # predict closure added in main (needs ds_val image sizes)
        return model, _train_forward, _val_forward, None

    else:
        raise ValueError(f"Unknown model_name: {model_name}. Use 'retinanet' or 'detr'.")


# -------------------- mAP evaluation -------------------- #
@torch.no_grad()
def coco_map(model_name, model, dl_val, ds_val, device, predict_batch_fn, val_ann_path, print_limit=None):
    detections = []
    seen = 0
    cat_ids_sorted = sorted([c["id"] for c in ds_val.categories])

    if model_name == "detr":
        proc = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    for bidx, (images, targets) in enumerate(dl_val):
        img_ids = [int(t["image_id"].item()) for t in targets]

        if model_name == "retinanet":
            batch_dets = predict_batch_fn(images, img_ids)
        else:
            model.eval()
            np_imgs = [(img.permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()) for img in images]
            enc = proc(images=np_imgs, return_tensors="pt")
            for k in enc:
                enc[k] = enc[k].to(device)
            outputs = model(**enc)

            sizes = []
            for img_id in img_ids:
                meta = ds_val.imgid_to_img[int(img_id)]
                sizes.append((meta["height"], meta["width"]))
            processed = proc.post_process_object_detection(outputs, target_sizes=torch.tensor(sizes, device=device))

            batch_dets = []
            for img_id, p in zip(img_ids, processed):
                boxes = p["boxes"].detach().cpu().numpy()
                scores = p["scores"].detach().cpu().numpy()
                labels = p["labels"].detach().cpu().numpy()  # 0..K-1
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
        print("[warn] No detections; mAP skipped.")
        return None
    coco_dt = coco_gt.loadRes(detections)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    ap, ap50, ap75, aps, apm, apl = [float(coco_eval.stats[i]) for i in range(6)]
    print(f"[mAP] AP@[.5:.95]={ap:.4f} | AP50={ap50:.4f} | AP75={ap75:.4f} | APs={aps:.4f} APm={apm:.4f} APl={apl:.4f}")
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

    # New fine-tuning knobs
    ap.add_argument("--head-lr", type=float, default=1e-3, help="LR for heads (RetinaNet) / non-backbone (DETR)")
    ap.add_argument("--backbone-lr", type=float, default=1e-4, help="LR for backbone")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--freeze-backbone-epochs", type=int, default=0, help="Set backbone LR=0 for first N epochs")
    ap.add_argument("--freeze-bn-when-frozen", action="store_true", help="RetinaNet only: put BN in eval while frozen")

    # Other training knobs
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default="./runs/exp1")
    ap.add_argument("--early-stop-patience", type=int, default=5)
    ap.add_argument("--resume", default="")
    ap.add_argument("--albu", action="store_true")
    ap.add_argument("--albu-strength", choices=["light", "medium"], default="light")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-train-batches", type=int, default=None)
    ap.add_argument("--max-val-batches", type=int, default=None)
    ap.add_argument("--print-freq", type=int, default=10)
    ap.add_argument("--eval-map-every", type=int, default=1)
    ap.add_argument("--map-skip-batches", type=int, default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")

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
    cat_ids_sorted = sorted([c["id"] for c in ds_train.categories])
    print("[train categories]\n" + ds_train.category_summary())

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                          collate_fn=collate_fn, pin_memory=(device.type == "cuda"))
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        collate_fn=collate_fn, pin_memory=(device.type == "cuda"))

    # Model + helpers
    model, train_forward, val_forward, predict_batch = build_model_and_helpers(
        model_name=args.model,
        num_classes=ds_train.num_classes,
        id2label_0based=getattr(ds_train, "id2label_0based", {}),
        label2id_0based=getattr(ds_train, "label2id_0based", {}),
        cat_ids_sorted=cat_ids_sorted,
        device=device,
    )

    # Build param groups with discriminative LR
    if args.model == "retinanet":
        param_groups, idx_bb_groups, backbone_module, set_bn_eval_fn = retinanet_param_groups(
            model, head_lr=args.head_lr, backbone_lr=args.backbone_lr,
            weight_decay=args.weight_decay, freeze_bn_when_frozen=args.freeze_bn_when_frozen
        )
    else:
        param_groups, idx_bb_groups = detr_param_groups(
            model, head_lr=args.head_lr, backbone_lr=args.backbone_lr, weight_decay=args.weight_decay
        )
        backbone_module, set_bn_eval_fn = None, None  # not used

    # If freezing for N epochs => set backbone LR=0 at start
    if args.freeze-backbone-epochs if False else None:
        pass  # (placeholder to avoid linter)

    if args.freeze_backbone_epochs > 0:
        for gi in idx_bb_groups:
            param_groups[gi]["lr"] = 0.0

    optimizer = AdamW(param_groups)
    total_steps = (len(dl_train) if args.max_train_batches is None else min(len(dl_train), args.max_train_batches)) * args.epochs
    scheduler = make_scheduler(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir))
    print(f"[info] Checkpoints & logs will be saved to: {out_dir}")

    # DETR predict closure (needs original sizes)
    if args.model == "detr":
        proc = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        @torch.no_grad()
        def predict_batch(images: List[torch.Tensor], img_ids: List[int]):
            model.eval()
            np_imgs = [(img.permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()) for img in images]
            enc = proc(images=np_imgs, return_tensors="pt")
            for k in enc: enc[k] = enc[k].to(device)
            outputs = model(**enc)
            sizes = [(ds_val.imgid_to_img[int(i)]["height"], ds_val.imgid_to_img[int(i)]["width"]) for i in img_ids]
            processed = proc.post_process_object_detection(outputs, target_sizes=torch.tensor(sizes, device=device))
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

    # Resume
    start_epoch, best_val = 0, float("inf")
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.is_file():
            start_epoch, best_val, _ = load_checkpoint_if_any(model, optimizer, scheduler, ckpt_path, device)

    patience = args.early_stop_patience
    epochs_no_improve = 0
    global_step = 0

    for epoch in range(start_epoch, args.epochs):
        # Toggle backbone freeze/unfreeze by LR
        if args.freeze_backbone_epochs > 0:
            if epoch < args.freeze_backbone_epochs:
                # keep LR=0 for backbone
                if backbone_module is not None and set_bn_eval_fn is not None:
                    backbone_module.apply(set_bn_eval_fn)  # Retinanet BN eval
            elif epoch == args.freeze_backbone_epochs:
                # unfreeze: set LR to user-provided backbone LR
                for gi in idx_bb_groups:
                    param_groups[gi]["lr"] = args.backbone_lr
                # also switch BN back to train automatically (we don't force train() here; the main loop sets .train())
                print(f"[unfreeze] Epoch {epoch}: backbone LR set to {args.backbone_lr}")

        model.train()
        running = 0.0
        for bidx, (images, targets) in enumerate(dl_train):
            if args.max_train_batches is not None and bidx >= args.max_train_batches:
                break
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    loss_dict, loss = train_forward(images, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_dict, loss = train_forward(images, targets)
                loss.backward()
                optimizer.step()

            scheduler.step()

            loss_val = float(loss.detach().cpu().item())
            running += loss_val
            writer.add_scalar("train/loss_step", loss_val, global_step)

            if bidx % args.print_freq == 0:
                if isinstance(loss_dict, dict):
                    parts = [f"{k}:{(v.item() if torch.is_tensor(v) else float(v)):.4f}" for k, v in loss_dict.items()]
                    comp = " ".join(parts)
                else:
                    comp = f"loss:{float(loss_dict):.4f}"
                print(f"Epoch {epoch}, Iter {bidx}, {comp} total:{loss_val:.4f}")

            global_step += 1

        train_epoch_loss = running / max(1, (bidx + 1))
        writer.add_scalar("train/loss_epoch", train_epoch_loss, epoch)

        # Val loss
        val_loss = evaluate_loss(dl_val, lambda im, tg: val_forward(im, tg), device, scaler=None, amp=False,
                                 max_batches=args.max_val_batches)
        writer.add_scalar("val/loss_epoch", val_loss, epoch)
        print(f"[Eval] Epoch {epoch} finished. train_loss={train_epoch_loss:.4f}, val_loss={val_loss:.4f}")

        # mAP (every N epochs)
        if args.eval_map_every and ((epoch + 1) % args.eval_map_every == 0):
            print("[mAP] Running COCO evaluation...")
            coco_map(
                model_name=args.model,
                model=model,
                dl_val=dl_val,
                ds_val=ds_val,
                device=device,
                predict_batch_fn=predict_batch,
                val_ann_path=args.val_ann,
                print_limit=None if args.map_skip_batches is None else (args.map_skip_batches * args.batch_size),
            )

        # Save / early stop
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
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
                "args": vars(args),
            },
            out_dir=Path(args.out),
            is_best=is_best,
        )

        if patience is not None and epochs_no_improve >= patience:
            print(f"[early stop] No improvement for {patience} epochs. Best val loss={best_val:.4f}")
            break

    writer.close()
    print(f"[done] Best val loss={best_val:.4f}. Checkpoints at {args.out}")


if __name__ == "__main__":
    main()

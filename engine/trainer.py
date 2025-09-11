from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_

from .scheduler import make_scheduler
from .checkpointing import save_checkpoint, load_checkpoint_if_any
from .data_utils import collate_fn
from .eval import coco_map_with_macro, COCO_EVAL_AVAILABLE


class Trainer:
    def __init__(
        self,
        args,
        model_name: str,
        model,
        train_forward,
        val_forward,
        predict_batch,
        dl_train,
        dl_val,
        ds_train,
        ds_val,
        device: torch.device,
        param_groups: List[Dict[str, Any]],
        idx_bb_groups: List[int],
    ):
        self.args = args
        self.model_name = model_name
        self.model = model
        self.train_forward = train_forward
        self.val_forward = val_forward
        self.predict_batch = predict_batch
        self.dl_train = dl_train
        self.dl_val = dl_val
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.device = device
        self.param_groups = param_groups
        self.idx_bb_groups = idx_bb_groups
        self.logger = logging.getLogger("train")

        # Apply initial LR 0 to backbone groups if requested
        if args.freeze_backbone_epochs > 0:
            for gi in idx_bb_groups:
                self.param_groups[gi]["lr"] = 0.0

        self.optimizer = AdamW(self.param_groups)
        steps_per_epoch = len(self.dl_train)
        if args.max_train_batches is not None:
            steps_per_epoch = min(steps_per_epoch, args.max_train_batches)
        steps_per_epoch = max(1, steps_per_epoch)
        self.total_steps = steps_per_epoch * args.epochs
        self.scheduler = make_scheduler(self.optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=self.total_steps)

        self.use_amp = (device.type == "cuda")
        bf16_supported = bool(self.use_amp and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
        self.amp_dtype = torch.bfloat16 if bf16_supported else torch.float16

        try:
            self.scaler = torch.amp.GradScaler(device_type="cuda", enabled=self.use_amp and (self.amp_dtype is torch.float16))
        except Exception:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp and (self.amp_dtype is torch.float16))

        self.out_dir = Path(args.out); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.out_dir))

        self.global_step = 0
        self.best_val = float("inf")
        self.best_epoch: Optional[int] = None

    def _val_loss(self, max_batches: Optional[int] = None) -> float:
        model_loss = 0.0
        n = 0
        for bidx, (images, targets) in enumerate(self.dl_val):
            if max_batches is not None and bidx >= max_batches:
                break
            images = [img.to(self.device) for img in images]
            targets = [{k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
            loss = self.val_forward(images, targets)
            model_loss += float(loss.detach().cpu().item())
            n += 1
        return model_loss / max(1, n)

    def resume_if_needed(self) -> Tuple[int, float, Optional[int]]:
        start_epoch = 0
        if getattr(self.args, "resume", None):
            ckpt_path = Path(self.args.resume)
            if ckpt_path.is_file():
                start_epoch, self.best_val, self.best_epoch, _ = load_checkpoint_if_any(
                    self.model, self.optimizer, self.scheduler, ckpt_path, self.device
                )
            else:
                raise FileNotFoundError(f"--resume file not found: {ckpt_path.resolve()}")
        return start_epoch, self.best_val, self.best_epoch

    def fit(self):
        args = self.args
        model = self.model
        device = self.device
        backbone_module = None
        # locate backbone for BN freezing
        if hasattr(model, "backbone"):
            backbone_module = model.backbone
        elif hasattr(model, "model") and hasattr(model.model, "backbone"):
            backbone_module = model.model.backbone

        start_epoch, _, _ = self.resume_if_needed()

        epochs_no_improve = 0
        accum = max(1, args.accum_steps)

        for epoch in range(start_epoch, args.epochs):
            # Freeze BN during backbone-frozen period
            if args.freeze_backbone_epochs > 0:
                if epoch < args.freeze_backbone_epochs:
                    if args.freeze_bn_when_frozen and backbone_module is not None:
                        for m in backbone_module.modules():
                            if isinstance(m, torch.nn.BatchNorm2d):
                                m.eval()
                elif epoch == args.freeze_backbone_epochs:
                    for gi in self.idx_bb_groups:
                        self.param_groups[gi]["lr"] = args.backbone_lr
                    self.logger.info(f"[unfreeze] Epoch {epoch}: backbone LR -> {args.backbone_lr}")

            # Train one epoch
            model.train()
            running = 0.0
            for bidx, (images, targets) in enumerate(self.dl_train):
                if args.max_train_batches is not None and bidx >= args.max_train_batches:
                    break

                images = [img.to(device) for img in images]
                targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]

                if self.use_amp and (self.amp_dtype is torch.float16):
                    with torch.cuda.amp.autocast(enabled=True):
                        loss_dict, loss = self.train_forward(images, targets)
                else:
                    loss_dict, loss = self.train_forward(images, targets)

                loss = loss / accum
                if self.scaler is not None and self.use_amp and (self.amp_dtype is torch.float16):
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (bidx + 1) % accum == 0:
                    if args.grad_clip is not None and args.grad_clip > 0:
                        if self.scaler is not None and self.use_amp and (self.amp_dtype is torch.float16):
                            self.scaler.unscale_(self.optimizer)
                        clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

                    if self.scaler is not None and self.use_amp and (self.amp_dtype is torch.float16):
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()

                running += float(loss.detach().cpu().item())
                self.global_step += 1

                if (bidx + 1) % args.print_freq == 0:
                    self.logger.info(f"[train] epoch {epoch+1} step {bidx+1} loss={running/args.print_freq:.4f}")
                    self.writer.add_scalar("train/loss", running / args.print_freq, self.global_step)
                    running = 0.0

            # Validation loss
            val_loss = self._val_loss(max_batches=args.max_val_batches)
            self.writer.add_scalar("val/loss", val_loss, epoch)
            self.logger.info(f"[val] epoch {epoch+1} loss={val_loss:.4f}")

            # Optional mAP/macro
            metrics = None
            if args.eval_map_every and ((epoch + 1) % args.eval_map_every == 0):
                metrics = coco_map_with_macro(
                    model_name=self.model_name,
                    model=self.model,
                    dl_val=self.dl_val,
                    ds_val=self.ds_val,
                    device=self.device,
                    predict_batch_fn=self.predict_batch,
                    val_ann_path=args.val_ann,
                    max_batches=args.eval_map_max_batches,
                )
                if metrics is not None:
                    self.logger.info(
                        f"[COCO@epoch] AP={metrics['AP']:.4f} AP50={metrics['AP50']:.4f} "
                        f"Macro-mAP={metrics['macro_mAP']:.4f} Macro-AP50={metrics['macro_AP50']:.4f}"
                    )
                    self.writer.add_scalar("val/coco_AP", metrics["AP"], epoch)
                    self.writer.add_scalar("val/coco_AP50", metrics["AP50"], epoch)
                    self.writer.add_scalar("val/macro_mAP", metrics["macro_mAP"], epoch)
                    self.writer.add_scalar("val/macro_AP50", metrics["macro_AP50"], epoch)

            # Select best metric
            metric_now = val_loss
            metric_name = "val_loss"
            if metrics is not None:
                if args.early_metric == "coco_ap":
                    metric_now = -metrics["AP"];        metric_name = "COCO_AP"
                elif args.early_metric == "macro_map":
                    metric_now = -metrics["macro_mAP"]; metric_name = "Macro_mAP"
                elif args.early_metric == "macro_ap50":
                    metric_now = -metrics["macro_AP50"]; metric_name = "Macro_AP50"

            is_best = metric_now < self.best_val
            if is_best:
                self.best_val = metric_now
                self.best_epoch = epoch + 1
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                    "best_val": self.best_val,
                    "best_epoch": self.best_epoch,
                    "best_metric_name": metric_name,
                    "args": vars(args),
                },
                out_dir=self.out_dir,
                is_best=is_best,
            )

            if args.early_stop_patience is not None and epochs_no_improve >= args.early_stop_patience:
                best_readable = self.best_val if metric_name == "val_loss" else -self.best_val
                self.logger.info(
                    f"[early stop] No improvement for {args.early_stop_patience} epochs. "
                    f"Best {metric_name}={best_readable:.4f} (epoch {self.best_epoch})"
                )
                break

        # Final mAP for best checkpoint if available
        if COCO_EVAL_AVAILABLE:
            best_ckpt = self.out_dir / "best.pth"
            if best_ckpt.exists():
                self.logger.info(f"[mAP] Loading best checkpoint from {best_ckpt} for final COCO eval...")
                ckpt = torch.load(best_ckpt, map_location=self.device)
                model.load_state_dict(ckpt["model"])
                be = ckpt.get("best_epoch", self.best_epoch)
                bv = ckpt.get("best_val", self.best_val)
                self.logger.info(f"[best] epoch={be}, val_loss={bv:.4f}")

                # For DETR, build a predict closure with processor
                predict_fn = self.predict_batch
                if self.model_name == "detr":
                    from transformers import DetrImageProcessor
                    proc = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

                    @torch.no_grad()
                    def predict_fn(images: List[torch.Tensor], img_ids: List[int]):
                        model.eval()
                        np_imgs = [(img.permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()) for img in images]
                        enc = proc(images=np_imgs, return_tensors="pt")
                        enc["pixel_values"] = enc["pixel_values"].to(device)
                        if "pixel_mask" in enc:
                            enc["pixel_mask"] = enc["pixel_mask"].to(device)
                        outputs = model(**enc)
                        sizes = [(self.ds_val.imgid_to_img[int(i)]["height"], self.ds_val.imgid_to_img[int(i)]["width"]) for i in img_ids]
                        processed = proc.post_process_object_detection(outputs, target_sizes=torch.tensor(sizes, device=device))
                        cat_ids_sorted_local = sorted([c["id"] for c in self.ds_val.categories])
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

                coco_map_with_macro(
                    model_name=self.model_name,
                    model=model,
                    dl_val=self.dl_val,
                    ds_val=self.ds_val,
                    device=self.device,
                    predict_batch_fn=predict_fn,
                    val_ann_path=self.args.val_ann,
                )
            else:
                self.logger.info("[warn] best.pth not found; skipping final mAP.")

        self.writer.close()
        self.logger.info(
            f"[done] Best model: epoch {self.best_epoch} with val_loss={self.best_val:.4f}. Checkpoints at {self.args.out}"
        )


from typing import Any, Dict, List, Tuple, Optional
import warnings
import torch

try:
    from transformers import DetrForObjectDetection, DetrImageProcessor, DetrConfig
    from transformers.utils import logging as hf_logging
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False


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


def build_model_and_helpers(
    model_name: str,
    num_classes: int,
    id2label_0based: Dict[int, str],
    label2id_0based: Dict[str, int],
    cat_ids_sorted: List[int],
    device: torch.device,
    orig_size_map: Optional[Dict[int, Tuple[int, int]]] = None,
    detr_short: Optional[int] = None,
    detr_max: Optional[int] = None,
    detr_grad_ckpt: bool = False,
    detr_dropout: Optional[float] = None,
    detr_attn_dropout: Optional[float] = None,
) -> Tuple[Any, Any, Any, Any]:
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
            # Build input tensor sizes map: image_id -> (H,W)
            sizes_in = {int(iid): tuple(images[idx].shape[-2:]) for idx, iid in enumerate(img_ids)}
            for img_id, p in zip(img_ids, preds):
                boxes = p["boxes"].detach().cpu().numpy()
                scores = p["scores"].detach().cpu().numpy()
                labels = p["labels"].detach().cpu().numpy()

                # Optional: rescale back to original image size
                ih, iw = sizes_in.get(int(img_id), (None, None))
                oh, ow = None, None
                if orig_size_map is not None:
                    ohow = orig_size_map.get(int(img_id))
                    if ohow is not None:
                        oh, ow = int(ohow[0]), int(ohow[1])

                if ih is not None and oh is not None and ow is not None and oh > 0 and ow > 0:
                    s = min(ih / oh, iw / ow)
                    new_h = int(round(oh * s)); new_w = int(round(ow * s))
                    # Clip to resized region then divide by scale
                    if boxes.size > 0:
                        boxes[:, 0] = boxes[:, 0].clip(0, new_w - 1)
                        boxes[:, 1] = boxes[:, 1].clip(0, new_h - 1)
                        boxes[:, 2] = boxes[:, 2].clip(0, new_w - 1)
                        boxes[:, 3] = boxes[:, 3].clip(0, new_h - 1)
                        boxes = boxes / max(1e-6, s)

                for (x1, y1, x2, y2), s_, l in zip(boxes, scores, labels):
                    cat_id = cat_ids_sorted[int(l)]
                    results.append({
                        "image_id": int(img_id),
                        "category_id": int(cat_id),
                        "bbox": [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))],
                        "score": float(s_),
                    })
            return results

        return model, _train_forward, _val_forward, _predict_batch

    elif model_name == "detr":
        assert HF_AVAILABLE, "Install transformers to use DETR."
        # Quiet transformers info logs and suppress noisy meta-parameter UserWarnings during init
        try:
            hf_logging.set_verbosity_error()
        except Exception:
            pass
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*copying from a non-meta parameter in the checkpoint to a meta parameter.*",
                category=UserWarning,
            )
            # Prepare config to override dropout/labels if requested
            cfg = DetrConfig.from_pretrained("facebook/detr-resnet-50")
            cfg.num_labels = num_classes
            cfg.id2label = id2label_0based
            cfg.label2id = label2id_0based
            if detr_dropout is not None:
                try:
                    cfg.dropout = float(detr_dropout)
                except Exception:
                    pass
            if detr_attn_dropout is not None:
                try:
                    cfg.attention_dropout = float(detr_attn_dropout)
                except Exception:
                    pass
            model = DetrForObjectDetection.from_pretrained(
                "facebook/detr-resnet-50",
                config=cfg,
                ignore_mismatched_sizes=True,
                low_cpu_mem_usage=False,
            ).to(device)
        # Optional: gradient checkpointing
        if detr_grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                pass

        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        size_kwargs = None
        if detr_short is not None and detr_short > 0:
            lm = detr_max if (detr_max is not None and detr_max > 0) else detr_short
            size_kwargs = {"shortest_edge": int(detr_short), "longest_edge": int(lm)}

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

            if size_kwargs is not None:
                enc = processor(images=np_imgs, annotations=annotations, return_tensors="pt", size=size_kwargs)
            else:
                enc = processor(images=np_imgs, annotations=annotations, return_tensors="pt")
            enc["pixel_values"] = enc["pixel_values"].to(model.device)
            if "pixel_mask" in enc:
                enc["pixel_mask"] = enc["pixel_mask"].to(model.device)
            if "labels" in enc:
                enc["labels"] = [
                    {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in lab.items()}
                    for lab in enc["labels"]
                ]
            return enc

        def _train_forward(images, targets):
            model.train()
            inputs = _encode_batch(images, targets)
            outputs = model(**inputs)
            loss = outputs.loss
            # Surface DETR component losses if available
            loss_dict = {"loss": loss}
            try:
                if hasattr(outputs, "loss_dict") and isinstance(outputs.loss_dict, dict):
                    for k, v in outputs.loss_dict.items():
                        if torch.is_tensor(v):
                            loss_dict[k] = v
            except Exception:
                pass
            return loss_dict, loss

        @torch.no_grad()
        def _val_forward(images, targets):
            model.train()
            inputs = _encode_batch(images, targets)
            outputs = model(**inputs)
            return outputs.loss

        @torch.no_grad()
        def _predict_batch(images: List[torch.Tensor], img_ids: List[int]):
            model.eval()
            np_imgs = [(img.permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()) for img in images]
            if size_kwargs is not None:
                enc = processor(images=np_imgs, return_tensors="pt", size=size_kwargs)
            else:
                enc = processor(images=np_imgs, return_tensors="pt")
            enc["pixel_values"] = enc["pixel_values"].to(model.device)
            if "pixel_mask" in enc:
                enc["pixel_mask"] = enc["pixel_mask"].to(model.device)
            outputs = model(**enc)
            # Use original sizes if provided; else fallback to current tensor sizes
            if orig_size_map is not None:
                sizes = [(int(orig_size_map[int(i)][0]), int(orig_size_map[int(i)][1])) for i in img_ids]
            else:
                sizes = [tuple(images[idx].shape[-2:]) for idx, _ in enumerate(img_ids)]
            sizes_t = torch.tensor(sizes, device=model.device)
            processed = processor.post_process_object_detection(outputs, target_sizes=sizes_t, threshold=0.05)
            results = []
            for img_id, p in zip(img_ids, processed):
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

    else:
        raise ValueError(f"Unknown model: {model_name}")

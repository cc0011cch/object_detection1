#!/usr/bin/env python3
"""
Export a trained torchvision RetinaNet to ONNX (pre-NMS).
Outputs:
  - cls_logits:   [B, N, C]
  - bbox_deltas:  [B, N, 4]
  - anchors_xyxy: [B, N, 4]

Examples:
  # Legacy exporter
  python export_retinanet_onnx.py \
    --ckpt runs/retina_rfs001/best.pth \
    --out runs/retina_rfs001/retinanet_head.onnx \
    --num-classes 3 \
    --img-size 512 \
    --opset 17 \
    --device cuda

  # New exporter
  python export_retinanet_onnx.py \
    --ckpt runs/retina_rfs001/best.pth \
    --out runs/retina_rfs001/retinanet_head.onnx \
    --num-classes 3 \
    --img-size 512 \
    --opset 18 \
    --device cuda \
    --dynamo
"""
import argparse
import warnings
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.onnx
import torchvision
from torchvision.models.detection.image_list import ImageList

# Rebuild your model exactly as in training
from train import build_model_and_helpers


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def _as_list(x):
    return x if isinstance(x, (list, tuple)) else [x]


class RetinaNetHeadExport(nn.Module):
    """
    Wrap RetinaNet to export pre-NMS head outputs + anchors.
    Handles multiple torchvision head/anchor API variants and shapes.
    """
    def __init__(self, model: torchvision.models.detection.RetinaNet, num_classes: int):
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head
        self.anchor_gen = model.anchor_generator
        self.num_classes = num_classes
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std",  torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),  persistent=False)

    def _extract_head(self, head_out):
        """
        Support both:
          - (cls_list, bbox_list)
          - {'cls_logits': [...], 'bbox_regression': [...]} + common aliases
        """
        if isinstance(head_out, (list, tuple)) and len(head_out) == 2:
            cls_logits_list, bbox_deltas_list = head_out
            return _as_list(cls_logits_list), _as_list(bbox_deltas_list)

        if isinstance(head_out, dict):
            if "cls_logits" in head_out:       cls_logits_list = head_out["cls_logits"]
            elif "classification" in head_out: cls_logits_list = head_out["classification"]
            elif "logits" in head_out:         cls_logits_list = head_out["logits"]
            else:
                raise RuntimeError(f"Cannot find classification logits in head keys: {list(head_out.keys())}")

            if "bbox_regression" in head_out:  bbox_deltas_list = head_out["bbox_regression"]
            elif "regression" in head_out:     bbox_deltas_list = head_out["regression"]
            elif "bbox_deltas" in head_out:    bbox_deltas_list = head_out["bbox_deltas"]
            else:
                raise RuntimeError(f"Cannot find bbox regression in head keys: {list(head_out.keys())}")

            return _as_list(cls_logits_list), _as_list(bbox_deltas_list)

        raise RuntimeError(f"Unsupported head output type: {type(head_out)}")

    def _flatten_levels_mixed(self, xs: List[torch.Tensor], kind: str) -> torch.Tensor:
        """
        Accept per-level tensors that may be:
          - 4D [B, A*X, H, W]  (classic TV)
          - 3D [B, N, X]       (already flat)
          - 2D [N, X]          (already flat, add batch dim)
        Returns [B, N_total, X]
        """
        outs: List[torch.Tensor] = []
        for x in xs:
            if x.dim() == 4:
                B, AX, H, W = x.shape
                if kind == "cls":
                    if AX % self.num_classes != 0:
                        raise RuntimeError(
                            f"Classification channels ({AX}) not divisible by num_classes ({self.num_classes})"
                        )
                    A = AX // self.num_classes
                    X = self.num_classes
                else:  # bbox
                    if AX % 4 != 0:
                        raise RuntimeError(f"BBox channels ({AX}) not divisible by 4")
                    A = AX // 4
                    X = 4
                x = x.view(B, A, X, H, W).permute(0, 3, 4, 1, 2).contiguous()  # [B,H,W,A,X]
                x = x.view(B, H * W * A, X)  # [B,N_level,X]
                outs.append(x)

            elif x.dim() == 3:
                outs.append(x)  # [B,N,X]

            elif x.dim() == 2:
                outs.append(x.unsqueeze(0))  # [1,N,X]

            else:
                raise RuntimeError(f"Unsupported tensor rank for head output: {tuple(x.shape)}")

        # Concat over N; ensure same B across levels
        B0 = outs[0].shape[0]
        for t in outs:
            if int(t.shape[0]) != int(B0):
                raise RuntimeError(f"Mismatched batch dims across levels: {[tuple(o.shape) for o in outs]}")
        return torch.cat(outs, dim=1)  # [B, sumN, X]

    def _build_anchors(self, images: torch.Tensor, feats: List[torch.Tensor]) -> torch.Tensor:
        """
        Handle both AnchorGenerator APIs:
          - forward(feature_maps)
          - forward(ImageList, feature_maps)
        Returns [B, N, 4] (XYXY)
        """
        B, _, H, W = images.shape
        # Try old API first
        try:
            anchors_per_img = self.anchor_gen(feats)  # old API
        except TypeError:
            # Newer API: expects ImageList with .tensors and .image_sizes
            # Use the normalized tensor x (same spatial size as images here) and per-image sizes
            image_sizes = [(int(H), int(W)) for _ in range(B)]
            # We pass the preprocessed images (normalized) as ImageList.tensors
            # Anchor generator only needs spatial sizes; values don't matter.
            image_list = ImageList(images, image_sizes)
            anchors_per_img = self.anchor_gen(image_list, feats)

        anchors_xyxy = []
        for ai in anchors_per_img:
            if isinstance(ai, list):   # per-level list
                ai = torch.cat(ai, dim=0)
            anchors_xyxy.append(ai)
        return torch.stack(anchors_xyxy, dim=0)  # [B, N, 4]

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Normalize like torchvision GeneralizedRCNNTransform
        x = (images - self.mean) / self.std

        # Backbone + FPN
        features = self.backbone(x)
        feats = list(features.values()) if isinstance(features, dict) else [features]

        # Head (robust to API variant and shapes)
        head_out = self.head(feats)
        cls_list, bbox_list = self._extract_head(head_out)

        # Flatten/concat per-level outputs, regardless of shape flavor
        cls_logits  = self._flatten_levels_mixed(cls_list,  kind="cls")   # [B,N,C]
        bbox_deltas = self._flatten_levels_mixed(bbox_list, kind="bbox")  # [B,N,4]

        # Anchors (pass an ImageList when required)
        anchors_xyxy = self._build_anchors(x, feats)                       # [B,N,4]

        return cls_logits, bbox_deltas, anchors_xyxy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to best.pth")
    ap.add_argument("--out",  required=True, help="Output ONNX path")
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--img-size", type=int, default=512, help="H=W; must be divisible by 32")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--device", choices=["cpu", "cuda"],
                    default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--dynamo", action="store_true", help="Use new torch.export-based ONNX exporter")
    args = ap.parse_args()

    device = torch.device(args.device)

    # Rebuild like training, then load weights
    model, _, _, _ = build_model_and_helpers(
        model_name="retinanet",
        num_classes=args.num_classes,
        id2label_0based={i: str(i) for i in range(args.num_classes)},
        label2id_0based={str(i): i for i in range(args.num_classes)},
        cat_ids_sorted=list(range(1, args.num_classes + 1)),
        device=device,
    )
    model.eval()

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    export_mod = RetinaNetHeadExport(model, num_classes=args.num_classes).to(device).eval()

    H = W = int(args.img_size)
    if (H % 32 != 0) or (W % 32 != 0):
        raise RuntimeError("img-size must be divisible by 32")
    dummy = torch.randn(1, 3, H, W, device=device, dtype=torch.float32)

    dynamic_axes = {
        "images":       {0: "batch", 2: "height", 3: "width"},
        "cls_logits":   {0: "batch", 1: "num_anchors"},
        "bbox_deltas":  {0: "batch", 1: "num_anchors"},
        "anchors_xyxy": {0: "batch", 1: "num_anchors"},
    }

    # Export (with graceful fallback if dynamo exporter fails)
    def _export_with(dynamo_flag: bool):
        torch.onnx.export(
            export_mod,
            dummy,
            args.out,
            input_names=["images"],
            output_names=["cls_logits", "bbox_deltas", "anchors_xyxy"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes if not dynamo_flag else dynamic_axes,  # keep for BC; ORT ignores names
            dynamo=dynamo_flag,
            training=torch.onnx.TrainingMode.EVAL,
        )

    tried_fallback = False
    if args.dynamo:
        try:
            # PyTorch warns that dynamic_axes is discouraged with dynamo; keep as-is and rely on shapes.
            _export_with(True)
        except Exception as e:
            warnings.warn(
                "Dynamo export failed; falling back to legacy exporter. "
                "This is common with torchvision detection utilities like AnchorGenerator.\n"
                f"Reason: {type(e).__name__}: {e}"
            )
            tried_fallback = True
            _export_with(False)
    else:
        _export_with(False)

    if tried_fallback:
        print("[OK] Exported with legacy exporter after dynamo fallback.")
    print(f"[OK] Exported ONNX to: {args.out}")
    print("    Inputs : images [B,3,H,W] (float32, ImageNet-normalized)")
    print("    Outputs: cls_logits [B,N,C], bbox_deltas [B,N,4], anchors_xyxy [B,N,4]")


if __name__ == "__main__":
    main()

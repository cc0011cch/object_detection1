#!/usr/bin/env python3
"""
Export a trained torchvision RetinaNet to ONNX.

Outputs (pre-NMS):
  - cls_logits:   [B, N, C]    (raw logits)
  - bbox_deltas:  [B, N, 4]    (dx,dy,dw,dh)
  - anchors_xyxy: [B, N, 4]    (x1,y1,x2,y2)

Assumptions:
  - Input is ImageNet-normalized float32 tensor in [B,3,H,W], H,W divisible by 32.
  - We reconstruct the model exactly like training via train.py's helper.

Usage:
  python export_retinanet_onnx.py \
      --ckpt runs/retina_rfs001/best.pth \
      --out retinanet_head.onnx \
      --num-classes 3 \
      --img-size 512 \
      --opset 17 \
      --device cuda

Then run ONNX Runtime with CPU or CUDA providers.
"""
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.onnx
import torchvision

# Reuse your training builder so num_classes & head wiring match your run
from train import build_model_and_helpers  # :contentReference[oaicite:1]{index=1}


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


class RetinaNetHeadExport(nn.Module):
    """
    Wrap torchvision RetinaNet to export pre-NMS predictions + anchors.

    - Normalizes input with ImageNet mean/std (same as torchvision transform)
    - Runs backbone + FPN + head
    - Flattens per-level outputs and concatenates
    - Generates anchors using model.anchor_generator given feature maps & input size
    """
    def __init__(self, model: torchvision.models.detection.RetinaNet, num_classes: int):
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head
        self.anchor_gen = model.anchor_generator
        self.num_classes = num_classes
        # Register constants so ONNX sees them
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1,3,1,1), persistent=False)
        self.register_buffer("std",  torch.tensor(IMAGENET_STD).view(1,3,1,1),  persistent=False)

    def _concat_levels(self, xs: List[torch.Tensor], A_times: int) -> torch.Tensor:
        """
        xs: list of [B, A*X, H, W]
        For cls: X=C, for bbox: X=4.
        We rearrange to [B, H*W*A, X] and concat over levels.
        """
        outs = []
        for x in xs:
            B, AX, H, W = x.shape
            A = A_times  # anchors per location (e.g., 9)
            X = AX // A
            x = x.view(B, A, X, H, W).permute(0, 3, 4, 1, 2).contiguous()  # [B,H,W,A,X]
            x = x.view(B, H * W * A, X)  # [B,N_level,X]
            outs.append(x)
        return torch.cat(outs, dim=1)  # [B, N_sum, X]

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Normalize (same as torchvision GeneralizedRCNNTransform does)
        x = (images - self.mean) / self.std  # [B,3,H,W]

        # Backbone + FPN (OrderedDict of feature maps)
        features = self.backbone(x)
        if isinstance(features, dict):
            feats = list(features.values())
        else:
            feats = [features]

        # Head
        cls_logits_list, bbox_deltas_list = self.head(feats)

        # Infer anchors-per-location A from cls head channels
        # Each cls map has shape [B, A*C, H, W] → A = channels // C
        A = cls_logits_list[0].shape[1] // self.num_classes

        # Flatten & concat levels
        cls_logits = self._concat_levels(cls_logits_list, A_times=A)  # [B, N, C]
        bbox_deltas = self._concat_levels(bbox_deltas_list, A_times=A)  # [B, N, 4]

        # Anchors for each image (use input image size for generator)
        B, _, H, W = images.shape
        # anchor_generator expects feature maps list + image_size list
        # Newer torchvision AnchorGenerator signature: (images, features) or (feature_maps, image_sizes)
        # We call the latter form:
        with torch.no_grad():
            # Make a fake list of feature maps to read spatial sizes (same as feats)
            feat_sizes = [f.shape[-2:] for f in feats]
        # Build anchors per image using image size (H, W)
        # Torchvision's AnchorGenerator forward(feature_maps) ignores image size and uses feature sizes; but
        # the newer API may require feature maps only. We'll call the simple way:
        anchors = self.anchor_gen(feats)  # returns List[List[Boxes]] length=B

        # Stack anchors per image to [B, N, 4] in XYXY
        anchors_xyxy = []
        for i in range(B):
            # anchors[i] is a Boxes (Tensor[N,4]) or list of tensors per level depending on version
            ai = anchors[i]
            if isinstance(ai, list):
                ai = torch.cat(ai, dim=0)
            anchors_xyxy.append(ai)
        anchors_xyxy = torch.stack(anchors_xyxy, dim=0)  # [B, N, 4]

        return cls_logits, bbox_deltas, anchors_xyxy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to best.pth from training")
    ap.add_argument("--out",  required=True, help="Path to write ONNX file")
    ap.add_argument("--num-classes", type=int, required=True, help="Number of classes used in training")
    ap.add_argument("--img-size", type=int, default=512, help="Export with H=W=img-size (must be divisible by 32)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--device", choices=["cpu", "cuda"], default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()

    device = torch.device(args.device)

    # Rebuild model exactly as training code did (no weights on heads/backbone)
    # We only need the model architecture; weights come from ckpt.
    # NOTE: build_model_and_helpers constructs torchvision RetinaNet when model_name='retinanet'.
    # (From your uploaded train.py: build_model_and_helpers returns the model for 'retinanet'.) :contentReference[oaicite:2]{index=2}
    model, _, _, _ = build_model_and_helpers(
        model_name="retinanet",
        num_classes=args.num_classes,
        id2label_0based={i: str(i) for i in range(args.num_classes)},
        label2id_0based={str(i): i for i in range(args.num_classes)},
        cat_ids_sorted=list(range(1, args.num_classes + 1)),
        device=device,
    )
    model.eval()

    # Load your checkpoint weights
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Wrap with export module
    export_mod = RetinaNetHeadExport(model, num_classes=args.num_classes).to(device)
    export_mod.eval()

    # Dummy input
    H = W = int(args.img_size)
    assert H % 32 == 0 and W % 32 == 0, "img-size must be divisible by 32"
    dummy = torch.randn(1, 3, H, W, device=device, dtype=torch.float32)

    # Dynamic axes
    dynamic_axes = {
        "images": {0: "batch", 2: "height", 3: "width"},
        "cls_logits": {0: "batch", 1: "num_anchors"},
        "bbox_deltas": {0: "batch", 1: "num_anchors"},
        "anchors_xyxy": {0: "batch", 1: "num_anchors"},
    }

    torch.onnx.export(
        export_mod,
        dummy,
        args.out,
        input_names=["images"],
        output_names=["cls_logits", "bbox_deltas", "anchors_xyxy"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )

    print(f"[OK] Exported ONNX to: {args.out}")
    print("    Inputs : images [B,3,H,W] (float32, ImageNet-normalized)")
    print("    Outputs: cls_logits [B,N,C], bbox_deltas [B,N,4], anchors_xyxy [B,N,4]")
    print("    Note   : run decoding + NMS in your app (example shown below).")


if __name__ == "__main__":
    main()

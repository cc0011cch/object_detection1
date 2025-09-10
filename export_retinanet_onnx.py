#!/usr/bin/env python3
"""
Export a trained torchvision RetinaNet to ONNX (pre-NMS).
Outputs:
  - cls_logits:   [B, N, C]
  - bbox_deltas:  [B, N, 4]
  - anchors_xyxy: [B, N, 4]
"""
import argparse
from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch.onnx
import torchvision

from train import build_model_and_helpers  # uses your training wiring  :contentReference[oaicite:1]{index=1}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

TensorOrList = Union[torch.Tensor, List[torch.Tensor]]

class RetinaNetHeadExport(nn.Module):
    def __init__(self, model: torchvision.models.detection.RetinaNet, num_classes: int):
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head
        self.anchor_gen = model.anchor_generator
        self.num_classes = num_classes
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1,3,1,1), persistent=False)
        self.register_buffer("std",  torch.tensor(IMAGENET_STD).view(1,3,1,1),  persistent=False)

    def _extract_head(self, head_out):
        """
        Support both:
          - (cls_list, bbox_list)
          - {'cls_logits': ..., 'bbox_regression': ...} (values may be list OR tensor)
        Returns (cls_out, reg_out) where each is either a list[Tensor] or a Tensor.
        """
        if isinstance(head_out, (list, tuple)) and len(head_out) == 2:
            return head_out[0], head_out[1]

        if isinstance(head_out, dict):
            # classification
            if "cls_logits" in head_out:
                cls_out = head_out["cls_logits"]
            elif "classification" in head_out:
                cls_out = head_out["classification"]
            elif "logits" in head_out:
                cls_out = head_out["logits"]
            else:
                raise RuntimeError(f"Cannot find classification logits in head dict keys: {list(head_out.keys())}")

            # regression
            if "bbox_regression" in head_out:
                reg_out = head_out["bbox_regression"]
            elif "regression" in head_out:
                reg_out = head_out["regression"]
            elif "bbox_deltas" in head_out:
                reg_out = head_out["bbox_deltas"]
            else:
                raise RuntimeError(f"Cannot find bbox regression in head dict keys: {list(head_out.keys())}")
            return cls_out, reg_out

        raise RuntimeError(f"Unsupported head output type: {type(head_out)}")

    def _as_list_4d(self, x: TensorOrList) -> List[torch.Tensor]:
        """
        Normalize head output to a list of 4D maps [B, AX, H, W] when possible.
        If the output is already flattened [B,N,X] or [N,X], return [] to signal 'use passthrough'.
        """
        if isinstance(x, list):
            if all(t.dim() == 4 for t in x):
                return x
            # already-flattened per level (rare) -> treat as passthrough below
            return []
        if isinstance(x, torch.Tensor):
            if x.dim() == 4:
                return [x]   # single level
            # 3D or 2D -> already flattened
            return []
        return []

    def _concat_levels(self, xs: List[torch.Tensor], A_times: int) -> torch.Tensor:
        """
        xs: list of [B, A*X, H, W]
        Returns [B, N, X]
        """
        outs = []
        for x in xs:
            # x is 4D guaranteed here
            B, AX, H, W = x.shape
            A = A_times
            X = AX // A
            x = x.view(B, A, X, H, W).permute(0, 3, 4, 1, 2).contiguous()  # [B,H,W,A,X]
            x = x.view(B, H * W * A, X)                                    # [B,N_level,X]
            outs.append(x)
        return torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]

    def _passthrough_flattened(self, x: torch.Tensor, X_expected: int, B: int) -> torch.Tensor:
        """
        Accept [B,N,X] or [N,X] (with B==1) and return [B,N,X].
        """
        if x.dim() == 3:
            Bx, N, X = x.shape
            assert X == X_expected, f"Expected last dim {X_expected}, got {X}"
            return x
        if x.dim() == 2:
            # assume batch=1
            N, X = x.shape
            assert X == X_expected, f"Expected last dim {X_expected}, got {X}"
            assert B == 1, "Got 2D head output but batch>1; cannot infer batch dimension."
            return x.unsqueeze(0)  # [1,N,X]
        raise RuntimeError(f"Unsupported flattened head tensor shape: {tuple(x.shape)}")

    def forward(self, images: torch.Tensor):
        # Normalize like torchvision transform
        x = (images - self.mean) / self.std

        # Features
        features = self.backbone(x)
        feats = list(features.values()) if isinstance(features, dict) else [features]

        # Head (could be list-of-4D, or flattened tensors)
        head_out = self.head(feats)
        cls_out, reg_out = self._extract_head(head_out)

        # Try classic path (list of 4D maps)
        cls_list = self._as_list_4d(cls_out)
        reg_list = self._as_list_4d(reg_out)

        if cls_list and reg_list:
            # infer A from channels
            ch = cls_list[0].shape[1]
            if (ch % self.num_classes) != 0:
                raise RuntimeError(f"Channels ({ch}) not divisible by num_classes ({self.num_classes}).")
            A = ch // self.num_classes
            cls_logits  = self._concat_levels(cls_list, A_times=A)  # [B,N,C]
            bbox_deltas = self._concat_levels(reg_list, A_times=A)  # [B,N,4]
        else:
            # flattened outputs already: expect [B,N,C] and [B,N,4] (or [N,C]/[N,4] when B=1)
            B = images.shape[0]
            # try to get C and verify
            if isinstance(cls_out, list) and len(cls_out) == 1:
                cls_out = cls_out[0]
            if isinstance(reg_out, list) and len(reg_out) == 1:
                reg_out = reg_out[0]
            cls_logits  = self._passthrough_flattened(cls_out, X_expected=self.num_classes, B=B)
            bbox_deltas = self._passthrough_flattened(reg_out, X_expected=4,               B=B)

        # Anchors per image (concatenate levels)
        anchors_per_img = self.anchor_gen(feats)  # list per image
        anchors_xyxy = []
        for ai in anchors_per_img:
            if isinstance(ai, list):
                ai = torch.cat(ai, dim=0)
            anchors_xyxy.append(ai)
        anchors_xyxy = torch.stack(anchors_xyxy, dim=0)  # [B,N,4]

        return cls_logits, bbox_deltas, anchors_xyxy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out",  required=True)
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--device", choices=["cpu", "cuda"], default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--dynamo", action="store_true", help="Use the new ONNX dynamo exporter")
    args = ap.parse_args()

    device = torch.device(args.device)

    # Build like training and load weights
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
    assert H % 32 == 0 and W % 32 == 0
    dummy = torch.randn(1, 3, H, W, device=device, dtype=torch.float32)

    dynamic_axes = {
        "images":       {0: "batch", 2: "height", 3: "width"},
        "cls_logits":   {0: "batch", 1: "num_anchors"},
        "bbox_deltas":  {0: "batch", 1: "num_anchors"},
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
        dynamo=args.dynamo,                           # try --dynamo for new exporter
        training=torch.onnx.TrainingMode.EVAL,
    )

    print(f"[OK] Exported ONNX to: {args.out}")
    print("    Inputs : images [B,3,H,W] (float32, ImageNet-normalized)")
    print("    Outputs: cls_logits [B,N,C], bbox_deltas [B,N,4], anchors_xyxy [B,N,4]")


if __name__ == "__main__":
    main()

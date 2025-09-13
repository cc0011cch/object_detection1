#!/usr/bin/env python3
"""
Export a Hugging Face DETR model (facebook/detr-resnet-50) to ONNX for inference.

The exported graph takes inputs:
  - pixel_values: float32 [B,3,H,W]
  - pixel_mask:   int64  [B,H,W] (optional but included in the export)

And produces outputs:
  - logits:      float32 [B, num_queries, num_labels]
  - pred_boxes:  float32 [B, num_queries, 4]   (normalized cxcywh in [0,1])

Post-processing (rescaling to original sizes, thresholding, NMS-free decoding)
is performed on the Python side using transformers' DetrImageProcessor in the
ONNX wrapper (see onnx_detr_wrapper.py).
"""
import argparse
from pathlib import Path

import torch

from models.factory import HF_AVAILABLE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="detr_resnet50.onnx", help="Output ONNX path")
    ap.add_argument("--num-labels", type=int, default=None, help="Override number of labels (default from HF config)")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--width", type=int, default=800)
    args = ap.parse_args()

    assert HF_AVAILABLE, "Transformers is required to export DETR to ONNX."
    from transformers import DetrForObjectDetection, DetrConfig

    cfg = DetrConfig.from_pretrained("facebook/detr-resnet-50")
    if args.num_labels is not None:
        cfg.num_labels = int(args.num_labels)

    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        config=cfg,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=False,
    )
    model.eval()

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values, pixel_mask=None):
            out = self.m(pixel_values=pixel_values, pixel_mask=pixel_mask)
            # Return only the tensors required for post-processing
            return out.logits, out.pred_boxes

    wrap = Wrapper(model)

    B, C, H, W = 1, 3, args.height, args.width
    example_pixel_values = torch.randn(B, C, H, W, dtype=torch.float32)
    example_pixel_mask = torch.ones(B, H, W, dtype=torch.int64)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = {
        "pixel_values": {0: "batch", 2: "height", 3: "width"},
        "pixel_mask":   {0: "batch", 1: "height", 2: "width"},
        "logits":       {0: "batch"},
        "pred_boxes":   {0: "batch"},
    }

    torch.onnx.export(
        wrap,
        (example_pixel_values, example_pixel_mask),
        f=str(out_path),
        input_names=["pixel_values", "pixel_mask"],
        output_names=["logits", "pred_boxes"],
        dynamic_axes=dynamic_axes,
        opset_version=int(args.opset),
        do_constant_folding=True,
    )
    print(f"[OK] Exported DETR ONNX to: {out_path}")


if __name__ == "__main__":
    main()


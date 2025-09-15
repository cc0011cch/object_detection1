Docker setup for evaluation and tools

Two images are provided:

1) CPU-only: `docker/Dockerfile.cpu` (simplest, no GPU)
2) GPU (CUDA 12.1): `docker/Dockerfile.gpu` (requires NVIDIA Container Toolkit)

Build

- CPU:
  - docker build -f docker/Dockerfile.cpu -t objdet:cpu .

- GPU:
  - docker build -f docker/Dockerfile.gpu -t objdet:gpu .

Run (mount your repo inside the container)

- CPU:
  - docker run --rm -it --shm-size=2g \
      -v "$(pwd)":/workspace -w /workspace objdet:cpu bash

- GPU (make sure `nvidia-smi` works on host and toolkit is installed):
  - docker run --rm -it --gpus all --shm-size=2g \
      -v "$(pwd)":/workspace -w /workspace objdet:gpu bash

Quick sanity checks inside the container

- Python + libs:
  - python -c "import torch, torchvision; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
  - python -c "import onnxruntime as ort; print('providers', ort.get_available_providers())"

Evaluate examples (adjust paths if different)

- Torch RetinaNet (keep dataset resize at 0):
  - python evaluate_test.py \
    --backend torch --model retinanet \
    --ckpt /workspace/runs/retina_rfs_balanced1/last.pth \
    --test-ann /workspace/data/coco/annotations_used/instances_train2017_valdebug50.json \
    --test-images /workspace/data/coco/train2017 \
    --batch-size 8 --num-workers 4 --resize-short 0 \
    --csv-out /workspace/runs/retina/Eva_pylast.csv \
    --pr-plot /workspace/runs/retina/Eva_pylast_pr_all_iou50.jpg

- ONNX RetinaNet (CPU or GPU):
  - python evaluate_test.py \
    --backend onnx --model retinanet \
    --onnx /workspace/runs/retina_rfs_balanced1/retinanet_last.onnx \
    --test-ann /workspace/data/coco/annotations_used/instances_train2017_valdebug50.json \
    --test-images /workspace/data/coco/train2017 \
    --batch-size 8 --num-workers 4 --resize-short 0 \
    --retina-onnx-score 0.05 --retina-onnx-nms 0.5 \
    --csv-out /workspace/runs/retina/Eva_onnxlast.csv \
    --pr-plot /workspace/runs/retina/Eva_onnxlast_pr_all_iou50.jpg

- Torch DETR:
  - python evaluate_test.py \
    --backend torch --model detr \
    --ckpt /workspace/runs/detr_debug500_rfsAlbu/last.pth \
    --test-ann /workspace/data/coco/annotations_used/instances_train2017_valdebug50.json \
    --test-images /workspace/data/coco/train2017 \
    --batch-size 8 --num-workers 4 --resize-short 0 \
    --csv-out /workspace/runs/detr/Eva_pylast.csv \
    --pr-plot /workspace/runs/detr/Eva_pylast_pr_all_iou50.jpg

- ONNX DETR:
  - python evaluate_test.py \
    --backend onnx --model detr \
    --onnx /workspace/runs/detr_debug500_rfsAlbu/detr_best.onnx \
    --test-ann /workspace/data/coco/annotations_used/instances_train2017_valdebug50.json \
    --test-images /workspace/data/coco/train2017 \
    --batch-size 8 --num-workers 4 --resize-short 0 \
    --csv-out /workspace/runs/detr/Eva_onnx.csv \
    --pr-plot /workspace/runs/detr/Eva_onnx_pr_all_iou50.jpg

Notes

- Map your local repository into `/workspace` so paths match inside the container.
- If you keep model paths absolute from your host (e.g., /home/ubuntu/object_detection1/...), convert them to `/workspace/...` inside the container.
- For GPU use, ensure NVIDIA drivers and `nvidia-container-toolkit` are installed on the host.
- If you see "Unexpected bus error encountered in worker" from PyTorch DataLoader, increase shared memory (already set via `--shm-size=2g` above or Compose's `shm_size`), or reduce DataLoader workers: add `--num-workers 0` to the evaluate command as a quick fallback.

---

## Dataset Prep (Optional quick start)

After starting the container and changing to `/workspace`, you can download and prepare COCO subsets like this. Adjust paths if you already have the dataset.

### 1) Download COCO 2017

```
bash scripts/coco_download.sh data/coco/ --images train val
```

Expected file counts:

```
---------- Post-download file counts ----------
train2017  jpg: 118287
val2017    jpg: 5000
annotations json: 6
-----------------------------------------------
```

### 2) Prepare Training Dataset

#### (a) Subset categories (person, car, bus)

```
python tools/coco_make_subset.py \
  --ann-in ./data/coco/annotations/instances_train2017.json \
  --cat-names person car bus \
  --ann-out ./data/coco/annotations_used/instances_train2017_subset.json \
  --remap-category-ids
```

Output:

```
Saved subset to ./data/coco/annotations_used/instances_train2017_subset.json
Images: 68339 | Annotations: 312401 | Categories: 3
1: person
2: car
3: bus
```

#### (b) Remove grayscale images

```
python tools/coco_remove_grayscale.py \
  --images-dir ./data/coco/train2017 \
  --ann-in ./data/coco/annotations_used/instances_train2017_subset.json \
  --ann-out ./data/coco/annotations_used/instances_train2017_subset_nogray.json \
  --list-out ./data/coco/removed_train_grayscale_ids.txt
```

Output:

```
Removed grayscale images: 2389
Final Images: 65950 | Annotations: 301246 | Categories: 3
```

### 3) Prepare Validation Dataset

#### (a) Subset categories

```
python tools/coco_make_subset.py \
  --ann-in ./data/coco/annotations/instances_val2017.json \
  --cat-names person car bus \
  --ann-out ./data/coco/annotations_used/instances_val2017_subset.json \
  --remap-category-ids
```

#### (b) Remove grayscale images

```
python tools/coco_remove_grayscale.py \
  --images-dir ./data/coco/val2017 \
  --ann-in ./data/coco/annotations_used/instances_val2017_subset.json \
  --ann-out ./data/coco/annotations_used/instances_val2017_subset_nogray.json \
  --list-out ./data/coco/removed_val_grayscale_ids.txt
```

Output:

```
Removed grayscale images: 98
Final Images: 2797 | Annotations: 12785 | Categories: 3
```

### 4) Split Train/Eval/Test Sets (use COCO val-based subset as test)

```
python tools/split_coco_by_ann.py \
  --train-json ./data/coco/annotations_used/instances_train2017_subset_nogray.json \
  --val-json   ./data/coco/annotations_used/instances_val2017_subset_nogray.json \
  --out-train  ./data/coco/annotations_used/instances_train2017_split_train.json \
  --out-eval   ./data/coco/annotations_used/instances_train2017_split_eval.json \
  --out-test   ./data/coco/annotations_used/instances_test2017_subset_nogray.json \
  --ratio 0.9 \
  --seed 1337
```

Output:

```
[ok] train=58154 imgs, eval=7796 imgs
```

### 5) Create Debug Subsets (for quick sanity checks)

```
# Tiny debug set (30 samples)
python tools/make_debug_subset.py \
  --train-json ./data/coco/annotations_used/instances_train2017_split_train.json \
  --out-json ./data/coco/annotations_used/instances_train2017_debug.json \
  --samples 30

# Small train set (500 samples)
python tools/make_debug_subset.py \
  --train-json ./data/coco/annotations_used/instances_train2017_split_train.json \
  --out-json ./data/coco/annotations_used/instances_train2017_debug500.json \
  --samples 500

# Validation debug set (50 samples)
python tools/make_debug_subset.py \
  --train-json ./data/coco/annotations_used/instances_train2017_split_eval.json \
  --out-json ./data/coco/annotations_used/instances_train2017_valdebug50.json \
  --samples 50
```

---

## Visualization: Torch vs ONNX vs GT

Use `tools/visualize_compare_predictions.py` to draw ground truth (green), Torch predictions (blue), and ONNX predictions (red) on the same image set. Results are saved as JPGs.

### RetinaNet (Torch + ONNX)

```
python tools/visualize_compare_predictions.py \
  --images ./data/coco/train2017 \
  --ann ./data/coco/annotations_used/instances_train2017_debug500.json \
  --model retinanet \
  --torch-ckpt runs/retina_rfs_balanced1/last.pth \
  --onnx runs/retina_rfs_balanced1/retinanet_last.onnx \
  --num 12 \
  --out-dir runs/viz_retina \
  --torch-thresh 0.5 --onnx-thresh 0.5 --topk 100
```

Notes:
- Keep dataset `--resize-short` at 0 for evaluation; the visualizer mirrors the model’s internal transforms.
- You can tweak ONNX RetinaNet preprocessing with `--retina-short` and `--retina-max` if desired (defaults 800/1333).

### DETR (Torch + ONNX)

```
python tools/visualize_compare_predictions.py \
  --images ./data/coco/train2017 \
  --ann ./data/coco/annotations_used/instances_train2017_debug500.json \
  --model detr \
  --torch-ckpt runs/detr_debug500_rfsAlbu/last.pth \
  --onnx runs/detr_debug500_rfsAlbu/detr_best.onnx \
  --num 12 \
  --out-dir runs/viz_detr \
  --torch-thresh 0.5 --onnx-thresh 0.5 --topk 50 \
  --debug-overlay
```

Notes:
- `--debug-overlay` draws a yellow band with sizes and a rectangle showing the effective resized (unpadded) input region mapped back to original coordinates. This helps diagnose any scaling/padding mismatch.
- Torch and ONNX paths both use the standard DETR policy (shortest=800, longest≤1333) inside the visualizer.

---

## Models via Git LFS (optional)

If the repo stores model checkpoints via Git LFS, you can pull only the model folders without downloading all LFS objects:

```
# one-time on your machine
git lfs install

# inside the repo (host or container at /workspace)
bash scripts/download_models.sh

# or specify exact include patterns
bash scripts/download_models.sh "runs/retina_rfs_balanced1/*,runs/detr_debug500_rfsAlbu/*"
```

The script verifies git-lfs is installed, enables LFS smudge/clean filters, pulls the specified patterns, and lists the resulting files.

# Object Detection Experiments

This repository provides scripts and tools for preparing COCO 2017 subsets (person, car, bus), applying preprocessing/augmentation, and training object detection models (RetinaNet and DETR).

---

## 1. Download COCO 2017 Dataset

bash scripts/coco_download.sh data/coco/ --images train val  

Expected file counts:

---------- Post-download file counts ----------
train2017  jpg: 118287  
val2017    jpg: 5000  
annotations json: 6  
-----------------------------------------------

---

## 2. Prepare Training Dataset

### (a) Subset categories (person, car, bus)

python tools/coco_make_subset.py \
  --ann-in ./data/coco/annotations/instances_train2017.json \
  --cat-names person car bus \
  --ann-out ./data/coco/annotations_used/instances_train2017_subset.json \
  --remap-category-ids  

Output:  
Saved subset to ./data/coco/annotations_used/instances_train2017_subset.json  
Images: 68339 | Annotations: 312401 | Categories: 3  
1: person  
2: car  
3: bus  

### (b) Remove grayscale images

python tools/coco_remove_grayscale.py \
  --images-dir ./data/coco/train2017 \
  --ann-in ./data/coco/annotations_used/instances_train2017_subset.json \
  --ann-out ./data/coco/annotations_used/instances_train2017_subset_nogray.json \
  --list-out ./data/coco/removed_train_grayscale_ids.txt  

Output:  
Removed grayscale images: 2389  
Final Images: 65950 | Annotations: 301246 | Categories: 3  

---

## 3. Prepare Validation Dataset

### (a) Subset categories

python tools/coco_make_subset.py \
  --ann-in ./data/coco/annotations/instances_val2017.json \
  --cat-names person car bus \
  --ann-out ./data/coco/annotations_used/instances_val2017_subset.json \
  --remap-category-ids  

### (b) Remove grayscale images

python tools/coco_remove_grayscale.py \
  --images-dir ./data/coco/val2017 \
  --ann-in ./data/coco/annotations_used/instances_val2017_subset.json \
  --ann-out ./data/coco/annotations_used/instances_val2017_subset_nogray.json \
  --list-out ./data/coco/removed_val_grayscale_ids.txt  

Output:  
Removed grayscale images: 98  
Final Images: 2797 | Annotations: 12785 | Categories: 3  

---

## 4. Split Train/Eval/Test Sets

python tools/split_coco_by_ann.py \
  --train-json ./data/coco/annotations_used/instances_train2017_subset_nogray.json \
  --val-json   ./data/coco/annotations_used/instances_val2017_subset_nogray.json \
  --out-train  ./data/coco/annotations_used/instances_train2017_split_train.json \
  --out-eval   ./data/coco/annotations_used/instances_train2017_split_eval.json \
  --out-test   ./data/coco/annotations_used/instances_test2017_subset_nogray.json \
  --ratio 0.9 \
  --seed 1337  

Output:  
[ok] train=58154 imgs, eval=7796 imgs  

---

## 5. Create Debug Subsets

# Tiny debug set (30 samples)
python tools/make_debug_subset.py \
  --train-json ./data/coco/annotations_used/instances_train2017_split_train.json \
  --out-json ./data/coco/annotations_used/instances_train2017_debug.json \
  --samples 30  

# Small debug set (500 samples)
python tools/make_debug_subset.py \
  --train-json ./data/coco/annotations_used/instances_train2017_split_train.json \
  --out-json ./data/coco/annotations_used/instances_train2017_debug500.json \
  --samples 500  

Validation debug set:  
python tools/make_debug_subset.py \
  --train-json ./data/coco/annotations_used/instances_train2017_split_eval.json \
  --out-json ./data/coco/annotations_used/instances_train2017_valdebug50.json \
  --samples 50  

---

## 6. Visualize Samples with Augmentation

### Training samples (Albumentations ON)

python dataset.py \
  --ann ./data/coco/annotations_used/instances_train2017_debug500.json \
  --images ./data/coco/train2017 \
  --augment \
  --use-albu \
  --albu-strength medium \
  --target-size 512 \
  --num 8 \
  --out-dir ./debug_samples/train500_aug  

### Validation samples (clean)

python dataset.py \
  --ann ./data/coco/annotations_used/instances_test2017_subset_nogray.json \
  --images ./data/coco/val2017 \
  --num 5 \
  --out-dir ./debug_samples/val_clean  

---

## 7. Train Models

### (a) Warm-up run (build RFS cache quickly)

python train.py \
  --model retinanet \
  --train-ann ./data/coco/annotations_used/instances_train2017_debug500.json \
  --val-ann   ./data/coco/annotations_used/instances_train2017_valdebug50.json \
  --train-images ./data/coco/train2017 \
  --val-images ./data/coco/train2017 \
  --epochs 1 --batch-size 4 --resize-short 800 \
  --num-workers 4 --rfs 0.001 --rfsAlpha 0.5 \
  --max-train-batches 10 --max-val-batches 5 \
  --eval-map-every 3 --eval-map-max-batches 50 --early-metric macro_map \
  --out runs/sanity_rfs_cache  

### (b) RetinaNet (full run)

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python train.py --model retinanet --train-ann ./data/coco/annotations_used/instances_train2017_debug500.json --val-ann   ./data/coco/annotations_used/instances_train2017_valdebug50.json --train-images ./data/coco/train2017 --val-images   ./data/coco/train2017 --epochs 16 --batch-size 6 --accum-steps 4 --num-workers 2 --prefetch-factor 2 --resize-short 800 --albu --albu-strength medium --head-lr 5e-4 --backbone-lr 5e-5 --weight-decay 1e-4 --freeze-backbone-epochs 1 --freeze-bn-when-frozen --warmup-steps 300 --rfs 0.001 --rfsAlpha 0.75 --eval-map-every 1 --eval-map-max-batches 50 --early-metric macro_map --print-freq 20 --eval-map-every 3 --eval-map-max-batches 50 --early-metric macro_map --log-file runs/retina/train2.log --log-console --out runs/retina_rfs002

# Reason of failure to converging:
Short answer: 100% clipping every update means your clip threshold (2.0) is too tight. It’s not “broken,” but you’re truncating the update direction/size on every step, which can slow or distort learning.

What this tells us

clip_grad_norm_: The value reported is the pre‑clip total norm. 185/185 clipped means every update had norm > 2.0.
With focal loss + accumulation, norms often exceed 2.0. Clipping that aggressively can mask LR tuning and reduce effective step size.
Recommended adjustments

Raise the threshold: try --grad-clip 5.0 first; if still ~100%, try 10.0. Or disable clipping entirely (omit --grad-clip) to see if loss/AP improve.
Reduce accumulation: --accum-steps 1 (accumulation sums gradients and increases the norm; reducing it often lowers clipping frequency).
Lower head LR slightly if you keep tight clipping: --head-lr 5e-4.
Keep bf16 AMP: it stabilizes numerics without scaling; clipping is mostly for outlier spikes.


### (b1a) RetinaNet (full run tuned)
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python train.py --model retinanet --train-ann ./data/coco/annotations_used/instances_train2017_debug500.json --val-ann ./data/coco/annotations_used/instances_train2017_valdebug50.json --train-images ./data/coco/train2017 --val-images ./data/coco/train2017 --epochs 28 --batch-size 6 --accum-steps 1 --num-workers 2 --prefetch-factor 2 --resize-short 800 --albu --albu-strength light --head-lr 1e-3 --backbone-lr 1e-4 --weight-decay 5e-5 --freeze-backbone-epochs 1 --freeze-bn-when-frozen --warmup-steps 80 --rfs 0.0 --eval-map-every 3 --eval-map-max-batches 100 --early-metric coco_ap --grad-clip 10.0 --print-freq 20 --amp bf16 --empty-cache-every 50 --log-file runs/retina/train_tuned.log --log-console --out runs/retina_rfs002_tuned

### (b1b) RetinaNet (full balance run tuned)
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python train.py --model retinanet --train-ann ./data/coco/annotations_used/instances_train2017_debug500.json --val-ann ./data/coco/annotations_used/instances_train2017_valdebug50.json --train-images ./data/coco/train2017 --val-images ./data/coco/train2017 --epochs 50 --batch-size 6 --accum-steps 1 --num-workers 2 --prefetch-factor 2 --resize-short 800 --albu --albu-strength light --head-lr 8e-4 --backbone-lr 1e-4 --weight-decay 5e-5 --freeze-backbone-epochs 1 --freeze-bn-when-frozen --warmup-steps 80 --rfs 0.60 --rfsAlpha 0.75  --eval-map-every 1 --eval-map-max-batches 150 --early-metric macro_map --early-stop-patience 8 --auto-clip --auto-clip-mult 1.15 --auto-clip-min 8 --auto-clip-max 24 --auto-clip-start-epoch 1 --amp bf16 --empty-cache-every 50 --print-freq 20 --log-file runs/retina/train_tuned_rfs1.log --log-console --out runs/retina_rfs_balanced1

nohup env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python train.py --model retinanet --train-ann ./data/coco/annotations_used/instances_train2017_debug500.json --val-ann ./data/coco/annotations_used/instances_train2017_valdebug50.json --train-images ./data/coco/train2017 --val-images ./data/coco/train2017 --epochs 50 --batch-size 6 --accum-steps 1 --num-workers 2 --prefetch-factor 2 --resize-short 800 --albu --albu-strength light --head-lr 8e-4 --backbone-lr 1e-4 --weight-decay 5e-5 --freeze-backbone-epochs 1 --freeze-bn-when-frozen --warmup-steps 80 --rfs 0.60 --rfsAlpha 0.75 --eval-map-every 1 --eval-map-max-batches 150 --early-metric macro_map --early-stop-patience 8 --auto-clip --auto-clip-mult 1.15 --auto-clip-min 8 --auto-clip-max 24 --auto-clip-start-epoch 1 --amp bf16 --empty-cache-every 50 --print-freq 20 --log-file runs/retina/train_tuned_rfs1.log --log-console --out runs/retina_rfs_balanced1 
### (b2) RetinaNet (resume)
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python train.py --model retinanet --train-ann ./data/coco/annotations_used/instances_train2017_debug500.json --val-ann ./data/coco/annotations_used/instances_train2017_valdebug50.json --train-images ./data/coco/train2017 --val-images ./data/coco/train2017 --epochs 32 --batch-size 6 --accum-steps 1 --num-workers 2 --prefetch-factor 2 --resize-short 800 --albu --albu-strength light --head-lr 8e-4 --backbone-lr 1e-4 --weight-decay 5e-5 --freeze-backbone-epochs 1 --freeze-bn-when-frozen --warmup-steps 80 --rfs 0.60 --rfsAlpha 0.75 --max-train-batches 300 --eval-map-every 3 --eval-map-max-batches 150 --early-metric macro_map --early-stop-patience 8 --grad-clip 10.0 --amp bf16 --empty-cache-every 50 --print-freq 20 --log-file runs/retina/train_tuned_resume_rfs.log --log-console --resume runs/retina_rfs_balanced/last.pth  

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python train.py --model retinanet --train-ann ./data/coco/annotations_used/instances_train2017_debug500.json --val-ann ./data/coco/annotations_used/instances_train2017_valdebug50.json --train-images ./data/coco/train2017 --val-images ./data/coco/train2017 --epochs 60 --batch-size 6 --num-workers 2 --prefetch-factor 2 --resize-short 800 --albu --albu-strength light --head-lr 8e-4 --backbone-lr 1e-4 --weight-decay 5e-5 --freeze-backbone-epochs 1 --freeze-bn-when-frozen --warmup-steps 80 --rfs 0.60 --rfsAlpha 0.75 --eval-map-every 1 --eval-map-max-batches 150 --early-metric coco_ap --early-stop-patience 12 --grad-clip 30.0 --amp bf16 --empty-cache-every 50 --print-freq 20 --log-file runs/retina/train_tuned_resume_rfs_ap.log --log-console --out runs/exp1 --resume runs/retina_rfs_balanced/last.pth

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python train.py --model retinanet --train-ann ./data/coco/annotations_used/instances_train2017_debug500.json --val-ann ./data/coco/annotations_used/instances_train2017_valdebug50.json --train-images ./data/coco/train2017 --val-images ./data/coco/train2017 --epochs 60 --batch-size 6 --num-workers 2 --prefetch-factor 2 --resize-short 800 --albu --albu-strength light --head-lr 8e-4 --backbone-lr 1e-4 --weight-decay 5e-5 --freeze-backbone-epochs 1 --freeze-bn-when-frozen --warmup-steps 80 --rfs 0.60 --rfsAlpha 0.75 --eval-map-every 1 --eval-map-max-batches 150 --early-metric coco_ap --early-stop-patience 12 --grad-clip 12 --auto-clip --auto-clip-mult 1.15 --auto-clip-min 8 --auto-clip-max 24 --auto-clip-start-epoch 1 --amp bf16 --empty-cache-every 50 --print-freq 20 --log-file runs/retina/train_tuned_rfs_autoclip.log --log-console --out runs/exp1 --resume runs/retina_rfs_balanced/last.pth

### (c) Debugging run

python train.py \
  --model retinanet \
  --train-ann ./data/coco/annotations_used/instances_train2017_debug.json \
  --val-ann   ./data/coco/annotations_used/instances_train2017_split_eval.json \
  --train-images ./data/coco/train2017 \
  --val-images   ./data/coco/train2017 \
  --epochs 20 --batch-size 2 --accum-steps 8 \
  --num-workers 4 --prefetch-factor 4 --persistent-workers \
  --resize-short 640 \
  --albu \
  --head-lr 1e-3 --backbone-lr 1e-4 --weight-decay 1e-4 \
  --freeze-backbone-epochs 1 --freeze-bn-when-frozen \
  --warmup-steps 100 \
  --rfs 0.001 \
  --out runs/retinanet_exp1  

### (d) DETR
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python train.py --model detr --train-ann ./data/coco/annotations_used/instances_train2017_debug500.json --val-ann ./data/coco/annotations_used/instances_train2017_valdebug50.json --train-images ./data/coco/train2017 --val-images ./data/coco/train2017 --epochs 60 --batch-size 15 --num-workers 2 --prefetch-factor 2 --resize-short 0 --detr-short 800 --detr-max 800 --detr-grad-checkpoint --detr-dropout 0.1 --detr-attn-dropout 0.1 --head-lr 2e-4 --backbone-lr 2e-5 --weight-decay 5e-4 --warmup-steps 1000 --grad-clip 0.1 --amp auto --rfs 0.6 --rfsAlpha 0.75 --eval-map-every 5 --eval-map-max-batches 100 --early-metric macro_map --early-stop-patience 10 --print-freq 20 --log-file runs/detr_debug/train_debug500_rfs.log --log-console --out runs/detr_debug500_rfs

## debug run
 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python train.py --model detr --train-ann ./data/coco/annotations_used/instances_train2017_debug.json --val-ann ./data/coco/annotations_used/instances_train2017_valdebug50.json --train-images ./data/coco/train2017 --val-images ./data/coco/train2017 --epochs 50 --batch-size 10 --num-workers 2 --prefetch-factor 2 --resize-short 0 --detr-short 800 --detr-max 800 --detr-grad-checkpoint --detr-dropout 0.1 --detr-attn-dropout 0.1 --head-lr 2e-4 --backbone-lr 2e-5 --weight-decay 5e-4 --warmup-steps 1000 --grad-clip 0.1 --amp auto --eval-map-every 3 --eval-map-max-batches 30 --early-metric macro_map --early-stop-patience 8 --print-freq 10 --log-file runs/detr_debug/train_long.log --log-console --out runs/detr_debug

### (did) DETR (run )

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python train.py --model detr --train-ann ./data/coco/annotations_used/instances_train2017_debug500.json --val-ann ./data/coco/annotations_used/instances_train2017_valdebug50.json --train-images ./data/coco/train2017 --val-images ./data/coco/train2017 --epochs 150 --batch-size 12 --num-workers 2 --prefetch-factor 2 --resize-short 0 --detr-short 800 --detr-max 800 --detr-dropout 0.0 --detr-attn-dropout 0.0 --head-lr 1e-4 --backbone-lr 2e-5 --weight-decay 1e-4 --warmup-steps 1000 --grad-clip 512 --amp auto --rfs 0 --eval-map-every 5 --eval-map-max-batches 100 --early-metric macro_map --early-stop-patience 15 --print-freq 20 --log-file runs/detr_debug/train_debug500_fresh.log --log-console --out runs/detr_debug500_fresh

### (did) DETR (full balance run tuned)
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python train.py --model detr --train-ann ./data/coco/annotations_used/instances_train2017_debug500.json --val-ann ./data/coco/annotations_used/instances_train2017_valdebug50.json --train-images ./data/coco/train2017 --val-images ./data/coco/train2017 --epochs 60 --batch-size 15 --num-workers 2 --prefetch-factor 2 --resize-short 0 --detr-short 800 --detr-max 800 --detr-grad-checkpoint --detr-dropout 0.1 --detr-attn-dropout 0.1 --head-lr 2e-4 --backbone-lr 2e-5 --weight-decay 5e-4 --warmup-steps 1000 --grad-clip 0.1 --amp auto --rfs 0.6 --rfsAlpha 0.75 --eval-map-every 5 --eval-map-max-batches 100 --early-metric macro_map --early-stop-patience 10 --print-freq 20 --log-file runs/detr_debug/train_debug500_rfs.log --log-console --out runs/detr_debug500_rfs

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python train.py --model detr --train-ann ./data/coco/annotations_used/instances_train2017_debug500.json --val-ann ./data/coco/annotations_used/instances_train2017_valdebug50.json --train-images ./data/coco/train2017 --val-images ./data/coco/train2017 --epochs 80 --batch-size 10 --num-workers 2 --prefetch-factor 2 --resize-short 0 --detr-short 800 --detr-max 800 --detr-grad-checkpoint --detr-dropout 0.05 --detr-attn-dropout 0.05 --head-lr 2e-4 --backbone-lr 2e-5 --weight-decay 5e-4 --warmup-steps 1000 --grad-clip 0.2 --amp auto --rfs 0.6 --rfsAlpha 0.75 --eval-map-every 5 --eval-map-max-batches 100 --early-metric macro_map --early-stop-patience 10 --print-freq 20 --log-file runs/detr_debug/train_debug500_rfs_phase2.log --log-console --out runs/detr_debug500_rfs --resume runs/detr_debug500_rfs/last.pth
---
## 8. validation
# debug
python evaluate_test.py --backend torch --model retinanet --ckpt runs/retina_rfs_balanced/last.pth --test-ann ./data/coco/annotations_used/instances_train2017_valdebug50.json --test-
images ./data/coco/train2017 --batch-size 8 --num-workers 4 --resize-short 800 --pr-plot runs/retina_rfs_balanced/pr_curves_iou
50_torch.jpg

python evaluate_test.py \
  --backend torch \
  --model retinanet \
  --ckpt runs/retina_rfs001/best.pth \
  --test-ann ./data/coco/annotations_used/instances_test2017_subset_nogray.json \
  --test-images ./data/coco/val2017 \
  --batch-size 8 --num-workers 4 \
  --resize-short 512 \
  --pr-plot runs/retina_rfs001/pr_curves_iou50_torch.jpg
## 9. onnx
python export_retinanet_onnx.py \
  --ckpt runs/retina_rfs001/best.pth \
  --out runs/retina_rfs001/retinanet_head.onnx \
  --num-classes 3 \
  --img-size 512 \
  --opset 18 \
  --device cuda \
  --dynamo

## 10 onnx evaluation
python evaluate_test.py \
  --backend onnx \
  --onnx runs/retina_rfs001/retinanet_head.onnx \
  --test-ann ./data/coco/annotations_used/instances_train2017_valdebug50.json \
  --test-images ./data/coco/train2017 \
  --batch-size 8 --num-workers 4 \
  --resize-short 512 \
  --pr-plot runs/retina_rfs001/pr_curves_iou50Val_onnx.jpg

## 11. TensorBoard (with Remote Access)

On EC2 instance:  
tensorboard --logdir=data/model --port=8080  

Local terminal (port forwarding):  
ssh -i /path/to/your/AWS/key/file -NL 8080:localhost:8080 user@host  

Then open http://localhost:8080 in your browser.


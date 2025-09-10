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
  --ann-json ./data/coco/annotations_used/instances_train2017_debug.json \
  --images-dir ./data/coco/train2017 \
  --model retinanet \
  --albu \
  --augment \
  --num-samples 8 \
  --save-dir ./debug_samples/train_aug  

### Validation samples (clean)

python dataset.py \
  --ann-json ./data/coco/annotations_used/instances_test2017_subset_nogray.json \
  --images-dir ./data/coco/val2017 \
  --model detr \
  --num-samples 5 \
  --save-dir ./debug_samples/val_clean  

---

## 7. Train Models

### (a) Warm-up run (build RFS cache quickly)

python train.py \
  --model retinanet \
  --train-ann ./data/coco/annotations_used/instances_train2017_debug500.json \
  --val-ann   ./data/coco/annotations_used/instances_train2017_valdebug50.json \
  --train-images ./data/coco/train2017 \
  --val-images ./data/coco/train2017 \
  --epochs 1 --batch-size 2 --resize-short 512 \
  --num-workers 4 --rfs 0.001 --rfsAlpha 0.75 \
  --max-train-batches 10 --max-val-batches 5 \
  --out runs/sanity_rfs_cache  

### (b) RetinaNet (full run)

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python train.py \
  --model retinanet \
  --train-ann ./data/coco/annotations_used/instances_train2017_debug500.json \
  --val-ann   ./data/coco/annotations_used/instances_train2017_valdebug50.json \
  --train-images ./data/coco/train2017 \
  --val-images   ./data/coco/train2017 \
  --epochs 16 --batch-size 8 --accum-steps 2 \
  --num-workers 8 --prefetch-factor 4 --persistent-workers \
  --resize-short 512 \
  --albu --albu-strength medium \
  --head-lr 5e-4 --backbone-lr 5e-5 --weight-decay 1e-4 \
  --freeze-backbone-epochs 1 --freeze-bn-when-frozen \
  --warmup-steps 300 \
  --rfs 0.001 --rfsAlpha 0.75 \
  --print-freq 20 \
  --log-file runs/retina/train1.log \
  --log-console \
  --out runs/retina_rfs001  

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

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python train.py \
  --model detr \
  --train-ann ./data/coco/annotations_used/instances_train2017_debug.json \
  --val-ann   ./data/coco/annotations_used/instances_train2017_split_eval.json \
  --train-images ./data/coco/train2017 \
  --val-images   ./data/coco/train2017 \
  --epochs 60 --batch-size 4 --accum-steps 8 \
  --num-workers 4 --prefetch-factor 4 --persistent-workers \
  --resize-short 640 \
  --albu \
  --head-lr 1e-3 --backbone-lr 1e-4 --weight-decay 1e-4 \
  --freeze-backbone-epochs 0 --freeze-bn-when-frozen \
  --warmup-steps 500 \
  --out runs/detr_exp1  

---

## 8. TensorBoard (with Remote Access)

On EC2 instance:  
tensorboard --logdir=data/model --port=8080  

Local terminal (port forwarding):  
ssh -i /path/to/your/AWS/key/file -NL 8080:localhost:8080 user@host  

Then open http://localhost:8080 in your browser.

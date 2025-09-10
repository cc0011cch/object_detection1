# object_detection1
object_detection


bash scripts/coco_download.sh data/coco/ --images train val
---------- Post-download file counts ----------
train2017  jpg: 118287
val2017    jpg: 5000
annotations json: 6
-----------------------------------------------
#Train dataset
# 1) Make a small COCO subset with only person, car, bus
python tools/coco_make_subset.py \
  --ann-in ./data/coco/annotations/instances_train2017.json \
  --cat-names person car bus \
  --ann-out ./data/coco/annotations_used/instances_train2017_subset.json \
  --remap-category-ids

Saved subset to ./data/coco/annotations_used/instances_train2017_subset.json
Images: 68339 | Annotations: 312401 | Categories: 3

[Category summary]
Total categories: 3
1: person
2: car
3: bus
# 2) Remove grayscale images from that subset
python tools/coco_remove_grayscale.py \
  --images-dir ./data/coco/train2017 \
  --ann-in ./data/coco/annotations_used/instances_train2017_subset.json \
  --ann-out ./data/coco/annotations_used/instances_train2017_subset_nogray.json \
  --list-out ./data/coco/removed_train_grayscale_ids.txt
Saved removed image IDs to ./data/coco/removed_train_grayscale_ids.txt
Removed grayscale images: 2389
Saved updated annotations to ./data/coco/annotations_used/instances_train2017_subset_nogray.json
Final Images: 65950 | Annotations: 301246 | Categories: 3

#Val dataset
# 1) Make a small COCO subset with only person, car, bus
python tools/coco_make_subset.py \
  --ann-in ./data/coco/annotations/instances_val2017.json \
  --cat-names person car bus \
  --ann-out ./data/coco/annotations_used/instances_val2017_subset.json \
  --remap-category-ids
Saved subset to ./data/coco/annotations_used/instances_val2017_subset.json
Images: 2895 | Annotations: 13221 | Categories: 3

[Category summary]
Total categories: 3
1: person
2: car
3: bus

# 2) Remove grayscale images from that subset
python tools/coco_remove_grayscale.py \
  --images-dir ./data/coco/val2017 \
  --ann-in ./data/coco/annotations_used/instances_val2017_subset.json \
  --ann-out ./data/coco/annotations_used/instances_val2017_subset_nogray.json \
  --list-out ./data/coco/removed_val_grayscale_ids.txt
Saved removed image IDs to ./data/coco/removed_val_grayscale_ids.txt
Removed grayscale images: 98
Saved updated annotations to ./data/coco/annotations/instances_val2017_subset_nogray.json
Final Images: 2797 | Annotations: 12785 | Categories: 3

# 3)split the train set and rename val set to test set
python tools/split_coco_by_ann.py \
  --train-json ./data/coco/annotations_used/instances_train2017_subset_nogray.json \
  --val-json   ./data/coco/annotations_used/instances_val2017_subset_nogray.json \
  --out-train  ./data/coco/annotations_used/instances_train2017_split_train.json \
  --out-eval   ./data/coco/annotations_used/instances_train2017_split_eval.json \
  --out-test   ./data/coco/annotations_used/instances_test2017_subset_nogray.json \
  --ratio 0.9 \
  --seed 1337

  [ok] train=58154 imgs, eval=7796 imgs
cat_id name                  total    train     eval  eval_target
-----------------------------------------------------------------
     1 person               252943   221490    31453        25294
     2 car                   42363    34796     7567         4236
     3 bus                    5940     4638     1302          594

# 4) make train debug set
python tools/make_debug_subset.py \
  --train-json ./data/coco/annotations_used/instances_train2017_split_train.json \
  --out-json ./data/coco/annotations_used/instances_train2017_debug.json \
  --samples 30

# Save 8 augmented training-look samples (Albumentations on)
python dataset.py \
  --ann-json ./data/coco/annotations_used/instances_train2017_debug.json \
  --images-dir ./data/coco/train2017 \
  --model retinanet \
  --albu \
  --augment \
  --num-samples 8 \
  --save-dir ./debug_samples/train_aug

  # Save 5 validation-look samples (no augmentation)
python dataset.py \
  --ann-json ./data/coco/annotations_used/instances_test2017_subset_nogray.json \
  --images-dir ./data/coco/val2017 \
  --model detr \
  --num-samples 5 \
  --save-dir ./debug_samples/val_clean

# 6) train model
python train.py \
  --model retinanet \
  --train-ann ./data/coco/annotations_used/instances_train2017_debug.json \
  --val-ann   ./data/coco/annotations_used/instances_train2017_split_eval.json \
  --train-images ./data/coco/train2017 \
  --val-images   ./data/coco/train2017 \
  --epochs 10 \
  --batch-size 2 \
  --albu \
  --out runs/retinanet_exp1

  # Warm start: freeze backbone for 3 epochs, then unfreeze with small LR

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \

python train.py \
  --model retinanet \
  --train-ann ./data/coco/annotations_used/instances_train2017_split_train.json \
  --val-ann   ./data/coco/annotations_used/instances_train2017_split_eval.json \
  --train-images ./data/coco/train2017 --val-images   ./data/coco/train2017 \
  --epochs 16 --batch-size 8 --accum-steps 1 --num-workers 4 \
  --prefetch-factor 4 --persistent-workers --resize-short 512 \
  --albu --albu-strength medium --head-lr 1e-3 --backbone-lr 1e-4 \
  --weight-decay 1e-4 --freeze-backbone-epochs 1 --freeze-bn-when-frozen \
  --warmup-steps 1000 --rfs 0.001 --rfsAlpha 0.75 --print-freq 20 --out runs/retina_rfs001

  

#debug
python train.py \
  --model retinanet \
 --train-ann ./data/coco/annotations_used/instances_train2017_debug.json \
  --val-ann   ./data/coco/annotations_used/instances_train2017_split_eval.json \
   --train-images ./data/coco/train2017 \
  --val-images   ./data/coco/train2017 \
  --epochs 20 \
  --batch-size 2 \
  --num-workers 4 \
  --prefetch-factor 4 \
  --persistent-workers \
  --accum-steps 8 \
  --rfs 0.001
  --resize-short 640 \
  --albu \
  --head-lr 1e-3 \
  --backbone-lr 1e-4 \
  --weight-decay 1e-4 \
  --freeze-backbone-epochs 1 \
  --freeze-bn-when-frozen \
  --warmup-steps 100 \
  --out runs/retinanet_exp1

# DETR with smaller backbone LR, no freezing

  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \

  python train.py \
  --model detr \
  --train-ann ./data/coco/annotations_used/instances_train2017_debug.json \
  --val-ann   ./data/coco/annotations_used/instances_train2017_split_eval.json \
  --train-images ./data/coco/train2017 \
  --val-images   ./data/coco/train2017 \
  --epochs 60 \
  --batch-size 4 \
  --num-workers 4 \
  --prefetch-factor 4 \
  --persistent-workers \
  --accum-steps 8 \
  --resize-short 640 \
  --albu \
  --head-lr 1e-3 \
  --backbone-lr 1e-4 \
  --weight-decay 1e-4 \
  --freeze-backbone-epochs 0 \
  --freeze-bn-when-frozen \
  --warmup-steps 500 \
  --out runs/detr_exp1
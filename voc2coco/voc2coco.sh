#!/bin/bash
set -e 

echo "Converting train set ..."
sed -i 's/$/.xml/' data/VOC2012_train_val/ImageSets/Main/train.txt
python voc2coco/voc2coco.py \
    --ann_dir data/VOC2012_train_val/Annotations \
    --ann_ids data/VOC2012_train_val/ImageSets/Main/train.txt \
    --labels data/VOC2012_train_val/labels.txt \
    --output data/VOC2012_train_val/train.json \
    --extract_num_from_imgid

echo "Converting val set ..."
sed -i 's/$/.xml/' data/VOC2012_train_val/ImageSets/Main/val.txt
python voc2coco/voc2coco.py \
    --ann_dir data/VOC2012_train_val/Annotations \
    --ann_ids data/VOC2012_train_val/ImageSets/Main/val.txt \
    --labels data/VOC2012_train_val/labels.txt \
    --output data/VOC2012_train_val/val.json \
    --extract_num_from_imgid


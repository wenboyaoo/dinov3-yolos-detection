#!/bin/bash
set -e 

echo "Generating labels.txt ..."
grep -ERoh '<name>(.*)</name>' data/voc2012/Annotations | sort | uniq | sed 's/<name>//g' | sed 's/<\/name>//g' > data/voc2012/labels.txt

echo "Converting train set ..."
sed -i 's/$/.xml/' data/voc2012/ImageSets/Main/train.txt
python voc2coco/voc2coco.py \
    --ann_dir data/voc2012/Annotations \
    --ann_ids data/voc2012/ImageSets/Main/train.txt \
    --labels data/voc2012/labels.txt \
    --output data/voc2012/train.json \
    --extract_num_from_imgid

echo "Converting val set ..."
sed -i 's/$/.xml/' data/voc2012/ImageSets/Main/val.txt
python voc2coco/voc2coco.py \
    --ann_dir data/voc2012/Annotations \
    --ann_ids data/voc2012/ImageSets/Main/val.txt \
    --labels data/voc2012/labels.txt \
    --output data/voc2012/val.json \
    --extract_num_from_imgid


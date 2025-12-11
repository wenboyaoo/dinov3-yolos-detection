# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco
from .voc import build as build_voc

DATASET_CONFIGS = {
    'coco':{'num_classes':91},
    'coco_panoptic':{'num_classes':250}, # for panoptic, we just add a num_classes that is large enough to hold max_obj_id + 1, but the exact value doesn't really matter
    'voc':{'num_classes':24}
}

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'voc':
        return build_voc(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')

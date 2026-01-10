# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO evaluator that works in distributed mode.

Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import contextlib
import copy
import os
from typing import Optional

import numpy as np
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from util.misc import all_gather, is_main_process


class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types, collect_stats: bool = False):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt
        if 'info' not in self.coco_gt.dataset:
            self.coco_gt.dataset['info'] = {}
        if 'licenses' not in self.coco_gt.dataset:
            self.coco_gt.dataset['licenses'] = []

        self.iou_types = iou_types
        self.coco_eval = {iou_type: COCOeval(coco_gt, iouType=iou_type) for iou_type in iou_types}
        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

        # Stats/visualization collector (lazy loaded, optional).
        self.collect_stats = bool(collect_stats)
        self._stats = None
        if self.collect_stats:
            try:
                from .coco_stats import CocoEvaluatorStats

                self._stats = CocoEvaluatorStats(self.coco_gt, self.iou_types, collect_stats=True)
            except Exception:
                # If optional deps are missing, we still keep normal COCOeval working.
                self._stats = None
                self.collect_stats = False

    def maybe_collect_sample_visuals(self, samples, targets, results, attns=None):
        if self._stats is None:
            return
        return self._stats.maybe_collect_sample_visuals(samples=samples, targets=targets, results=results, attns=attns)

    def update_attentions(self, attns, samples):
        if self._stats is None:
            return
        return self._stats.update_attentions(attns=attns, samples=samples)

    def export_stats_to_csv(self, output_dir='./outputs'):
        if self._stats is None:
            return
        return self._stats.export_stats_to_csv(output_dir)

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        if self._stats is not None:
            try:
                self._stats._collect_det_token_box_stats(predictions, img_ids)
            except Exception:
                pass

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)

            # suppress pycocotools prints
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()

            coco_eval = self.coco_eval[iou_type]
            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            _, eval_imgs = evaluate(coco_eval)
            self.eval_imgs[iou_type].append(eval_imgs)

            if self._stats is not None and iou_type == 'bbox':
                try:
                    self._stats._collect_det_token_recall_matched_from_eval_imgs(coco_dt, coco_eval, eval_imgs)
                except Exception:
                    pass

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

        if self._stats is not None:
            try:
                self._stats.synchronize_between_processes()
            except Exception:
                pass

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()
            self.summarize_per_category(iou_type=iou_type, iou_thr=0.50, max_dets=100, area="all")

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                        "token_idx": k,
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            masks = prediction["masks"]
            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"].flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        'keypoints': keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results

    def summarize_per_category(self, iou_type="bbox", iou_thr=0.50, max_dets=100, area="all"):
        coco_eval = self.coco_eval[iou_type]
        if coco_eval.eval is None:
            raise RuntimeError("Please run accumulate() before summarize_per_category().")

        cat_ids = coco_eval.params.catIds
        cats = coco_eval.cocoGt.loadCats(cat_ids)
        cat_id_to_name = {c["id"]: c.get("name", str(c["id"])) for c in cats}

        prec = coco_eval.eval["precision"]
        rec = coco_eval.eval["recall"]

        ious = coco_eval.params.iouThrs
        t = int(np.where(np.isclose(ious, iou_thr))[0][0])

        area_lbls = coco_eval.params.areaRngLbl
        if area not in area_lbls:
            raise ValueError(f"area must be one of {area_lbls}, got {area}")
        a = area_lbls.index(area)

        m = coco_eval.params.maxDets.index(max_dets)

        ap_per_class = []
        for k, cid in enumerate(cat_ids):
            p = prec[t, :, k, a, m]
            p = p[p > -1]
            ap = float(p.mean()) if p.size else float("nan")
            ap_per_class.append((cid, cat_id_to_name[cid], ap))

        recall_per_class = []
        for k, cid in enumerate(cat_ids):
            r = rec[t, k, a, m]
            recall_per_class.append((cid, cat_id_to_name[cid], float(r)))

        ap_sorted = sorted(ap_per_class, key=lambda x: (np.nan_to_num(x[2], nan=-1.0)))
        print(f"\nPer-category AP@{iou_thr:.2f} ({iou_type}, area={area}, maxDets={max_dets}):")
        for cid, name, ap in ap_sorted:
            print(f"  {name:<12} (cid={cid:<3}) AP={ap:.3f}")

        r_sorted = sorted(recall_per_class, key=lambda x: x[2])
        print(f"\nPer-category Recall@{iou_thr:.2f} ({iou_type}, area={area}, maxDets={max_dets}):")
        for cid, name, r in r_sorted:
            print(f"  {name:<12} (cid={cid:<3}) R={r:.3f}")
        print()


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


def evaluate(self, *args, **kwargs):
    '''
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    '''
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = self.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = self.computeOks
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds}

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    # this is NOT in the pycocotools code, but could be done outside
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    return p.imgIds, evalImgs

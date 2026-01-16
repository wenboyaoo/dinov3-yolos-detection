# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

import util.misc as utils
from util import box_ops
from datasets.coco_eval import CocoEvaluator

# def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
#                     data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                     device: torch.device, epoch: int, max_norm: float = 0):
#     model.train()
#     criterion.train()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#     print_freq = 100

#     # AMP
#     scaler = torch.cuda.amp.GradScaler()

#     for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
#         samples = samples.to(device)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#         with torch.cuda.amp.autocast():

#             outputs = model(samples)            
#             loss_dict = criterion(outputs, targets)
#             weight_dict = criterion.weight_dict
#             losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

#             # reduce losses over all GPUs for logging purposes
#             loss_dict_reduced = utils.reduce_dict(loss_dict)
#             loss_dict_reduced_unscaled = {f'{k}_unscaled': v
#                                         for k, v in loss_dict_reduced.items()}
#             loss_dict_reduced_scaled = {k: v * weight_dict[k]
#                                         for k, v in loss_dict_reduced.items() if k in weight_dict}
#             losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

#             loss_value = losses_reduced_scaled.item()

#         if not math.isfinite(loss_value):
#             print("Loss is {}, stopping training".format(loss_value))
#             print(loss_dict_reduced)
#             sys.exit(1)

#         optimizer.zero_grad()
#         scaler.scale(losses).backward()

#         if max_norm > 0:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

#         scaler.step(optimizer)
#         scaler.update()

#         metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
#         metric_logger.update(class_error=loss_dict_reduced['class_error'])
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    tb_writer: Optional[SummaryWriter] = None, use_bf16: bool = False,
                    # >>> DEBUG COST RATIO LOGGING (TEMP; safe to delete) >>>
                    debug_cost_ratio: bool = False):
                    # <<< DEBUG COST RATIO LOGGING (TEMP; safe to delete) <<<
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    # count = 0

    num_batches = len(data_loader)
    for batch_idx, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        # count += 1
        # if count == 10: break

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Mixed precision training with bfloat16
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_bf16):
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)

        log_weight_dict = getattr(criterion, 'log_weight_dict', weight_dict)
        loss_dict_reduced_scaled_total = {k: v * weight_dict[k]
                          for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled_total.values())

        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                  for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * log_weight_dict[k]
                for k, v in loss_dict_reduced.items() if k in log_weight_dict}

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # >>> DEBUG COST RATIO LOGGING (TEMP; safe to delete) >>>
        # Print per-class ratio stats computed inside HungarianMatcher.forward().
        # We print after optimizer.step() to match "end of batch".
        if debug_cost_ratio and utils.is_main_process():
            matcher = getattr(criterion, 'matcher', None)
            stats = getattr(matcher, '_last_debug_cost_ratio_stats', None) if matcher is not None else None
            if isinstance(stats, dict) and isinstance(stats.get('per_class', None), dict):
                per_class = stats['per_class']
                ks = stats.get('ks', (1, 5, 10))
                if len(per_class) > 0:
                    parts = []
                    for cls_id in sorted(per_class.keys()):
                        entry = per_class[cls_id]
                        n = entry.get('n', 0)
                        k_str = ' '.join([f"k{k}={entry.get(k, float('nan')):.3f}" for k in ks])
                        parts.append(f"cls{cls_id}(n={n}): {k_str}")
                    msg = f"[DBG cost_ratio] epoch={epoch} batch={batch_idx+1}/{num_batches} | " + ' | '.join(parts)
                    print(msg)
        # <<< DEBUG COST RATIO LOGGING (TEMP; safe to delete) <<<

        if tb_writer is not None and utils.is_main_process():
            global_step = epoch * num_batches + batch_idx
            tb_writer.add_scalar('train/loss', loss_value, global_step)
            tb_writer.add_scalar('train/lr', optimizer.param_groups[0]["lr"], global_step)
            tb_writer.add_scalar('train/class_error', loss_dict_reduced['class_error'].item(), global_step)
            for k, v in loss_dict_reduced_scaled.items():
                tb_writer.add_scalar(f'train/{k}', v.item(), global_step)
            for k, v in loss_dict_reduced_unscaled.items():
                tb_writer.add_scalar(f'train/{k}', v.item(), global_step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    epoch_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if tb_writer is not None and utils.is_main_process():
        for k, v in epoch_stats.items():
            tb_writer.add_scalar(f'epoch_train/{k}', v, epoch)
    return epoch_stats


@torch.no_grad()
def coco_evaluate(model, criterion, postprocessors, data_loader, base_ds, device, epoch: Optional[int] = None, tb_writer: Optional[SummaryWriter] = None, output_dir: Optional[str] = None, use_bf16: bool = False, **kwargs):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]


    for samples, targets in metric_logger.log_every(data_loader, 256, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]


        amp_enabled = bool(use_bf16) and device.type == 'cuda'
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=amp_enabled):
            outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)

        log_weight_dict = getattr(criterion, 'log_weight_dict', weight_dict)
        loss_dict_reduced_scaled_total = {k: v * weight_dict[k]
                  for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                  for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * log_weight_dict[k]
                for k, v in loss_dict_reduced.items() if k in log_weight_dict}

        metric_logger.update(loss=sum(loss_dict_reduced_scaled_total.values()),
                     **loss_dict_reduced_scaled,
                     **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    
    panoptic_res = None
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    if tb_writer is not None and utils.is_main_process():
        log_step = epoch if epoch is not None else 0
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                tb_writer.add_scalar(f'eval/{k}', v, log_step)
        if 'coco_eval_bbox' in stats:
            bbox_stats = stats['coco_eval_bbox']
            bbox_names = ['AP','AP50','AP75','AP_small','AP_medium','AP_large','AR1','AR10','AR100','AR_small','AR_medium','AR_large']
            for name, val in zip(bbox_names, bbox_stats):
                tb_writer.add_scalar(f'eval/bbox_{name}', val, log_step)
        if 'coco_eval_masks' in stats:
            mask_stats = stats['coco_eval_masks']
            mask_names = ['AP','AP50','AP75','AP_small','AP_medium','AP_large','AR1','AR10','AR100','AR_small','AR_medium','AR_large']
            for name, val in zip(mask_names, mask_stats):
                tb_writer.add_scalar(f'eval/mask_{name}', val, log_step)

    return stats, coco_evaluator

EVALUATOR_CONFIG = {
    'coco': coco_evaluate
}

@torch.no_grad()
def evaluate(evaluator, **kwargs):
    if evaluator in EVALUATOR_CONFIG.keys():
        return EVALUATOR_CONFIG[evaluator](**kwargs)
    else:
        raise ValueError(f'Unknown evaluator: {evaluator}')
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO evaluator that works in distributed mode.

Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import os
import contextlib
import copy
import numpy as np
import torch
from collections import defaultdict
import csv
from typing import Optional
from pathlib import Path

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask_util

from util.misc import all_gather, is_main_process


class CocoEvaluatorStats(object):
    def __init__(self, coco_gt, iou_types, collect_stats=False):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt
        if 'info' not in self.coco_gt.dataset:
                self.coco_gt.dataset['info'] = {}
        if 'licenses' not in self.coco_gt.dataset:
                self.coco_gt.dataset['licenses'] = []
        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}
        
        self.collect_stats = collect_stats

        self._det_token_num = None
        self.det_token_box_stats = {
            'cx': None,
            'cy': None,
            'area': None,
            'score': None,
            'label': None,
            'x0': None,
            'y0': None,
            'x1': None,
            'y1': None,
        } if collect_stats else None

        self.det_token_box_stats_iou50_recall_matched_useCats1 = {
            'cx': None,
            'cy': None,
            'area': None,
            'score': None,
            'label': None,
            'x0': None,
            'y0': None,
            'x1': None,
            'y1': None,
        } if collect_stats else None

        self.det_token_attn_lastlayer_det_to_patch_meanheads = {
            'sum': None,
            'cnt': None,
        } if collect_stats else None

        # Per-layer det->patch attention (mean heads), aggregated over val set.
        # - sum: list[n_layers] of list[n_tokens] of (g,g) arrays
        # - cnt: (g,g) array shared across layers
        self.det_token_attn_layers_det_to_patch_meanheads = {
            'sum': None,
            'cnt': None,
        } if collect_stats else None

        self._attn_grid_size = 100

        # Visualization output layout
        self._viz_total_dirname = "total"
        # NOTE: per-image sampled visualization has been removed.

    @staticmethod
    def _infer_patch_size_from_hw_and_num_patches(img_h: int, img_w: int, num_patches: int) -> Optional[int]:
        if img_h <= 0 or img_w <= 0 or num_patches <= 0:
            return None
        common_divs = [d for d in range(1, min(img_h, img_w) + 1) if (img_h % d == 0 and img_w % d == 0)]
        preferred = [16, 14, 8, 32]
        patch_size_candidates = [d for d in preferred if d in common_divs] + [d for d in common_divs if d not in preferred]
        for ps in patch_size_candidates:
            hp = img_h // ps
            wp = img_w // ps
            if hp * wp == num_patches:
                return int(ps)
        return None

    def maybe_collect_sample_visuals(self, samples, targets, results, attns=None):
        # Per-image sampled visualization has been removed.
        return


    @staticmethod
    def _print_stats(tag: str, message: str):
        print(f"[Stats][{tag}] {message}")

    @classmethod
    def _print_saved(cls, kind: str, path):
        cls._print_stats(kind, str(path))

    @classmethod
    def _print_det_token_box_stats_brief(cls, label: str, stats_obj):
        if not isinstance(stats_obj, dict) or not isinstance(stats_obj.get('cx'), list):
            cls._print_stats("DetToken", f"{label}: no data")
            return

        n_tokens = len(stats_obj['cx'])
        token_counts = []
        for t in range(n_tokens):
            vals = stats_obj['cx'][t]
            token_counts.append(len(vals) if isinstance(vals, list) else 0)
        tokens_with_data = sum(1 for c in token_counts if c > 0)
        total_samples = int(sum(token_counts))

        def flatten(key: str):
            arr = stats_obj.get(key)
            if not isinstance(arr, list):
                return []
            out = []
            for v in arr:
                if isinstance(v, list) and v:
                    out.extend(v)
            return out

        cx_s = cls._summarize_1d(flatten('cx'))
        cy_s = cls._summarize_1d(flatten('cy'))
        a_s = cls._summarize_1d(flatten('area'))
        x0_s = cls._summarize_1d(flatten('x0'))
        y0_s = cls._summarize_1d(flatten('y0'))
        x1_s = cls._summarize_1d(flatten('x1'))
        y1_s = cls._summarize_1d(flatten('y1'))

        cls._print_stats(
            "DetToken",
            f"{label}: tokens={n_tokens}, tokens_with_data={tokens_with_data}, samples={total_samples}",
        )
        if total_samples <= 0:
            return
        cls._print_stats(
            "DetToken",
            "mean±std (normalized): "
            f"cx={cx_s['mean']:.6f}±{cx_s['std']:.6f}, "
            f"cy={cy_s['mean']:.6f}±{cy_s['std']:.6f}, "
            f"area={a_s['mean']:.6f}±{a_s['std']:.6f}",
        )
        cls._print_stats(
            "DetToken",
            "mean±std (normalized): "
            f"x0={x0_s['mean']:.6f}±{x0_s['std']:.6f}, "
            f"y0={y0_s['mean']:.6f}±{y0_s['std']:.6f}, "
            f"x1={x1_s['mean']:.6f}±{x1_s['std']:.6f}, "
            f"y1={y1_s['mean']:.6f}±{y1_s['std']:.6f}",
        )

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        if self.collect_stats:
            self._collect_det_token_box_stats(predictions, img_ids)

        bbox_coco_dt = None
        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)

            # suppress pycocotools prints
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()

            if iou_type == "bbox":
                bbox_coco_dt = coco_dt
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

            if self.collect_stats and iou_type == "bbox":
                self._collect_det_token_recall_matched_from_eval_imgs(coco_dt, coco_eval, eval_imgs)

    def _ensure_det_token_attn_buffers(self, n_tokens: int):
        if not self.collect_stats or self.det_token_attn_lastlayer_det_to_patch_meanheads is None:
            return
        if n_tokens <= 0:
            return
        if self.det_token_attn_lastlayer_det_to_patch_meanheads.get('sum') is None:
            self.det_token_attn_lastlayer_det_to_patch_meanheads['sum'] = [
                np.zeros((self._attn_grid_size, self._attn_grid_size), dtype=np.float64) for _ in range(int(n_tokens))
            ]
            self.det_token_attn_lastlayer_det_to_patch_meanheads['cnt'] = np.zeros(
                (self._attn_grid_size, self._attn_grid_size), dtype=np.float64
            )
            return

        cur = len(self.det_token_attn_lastlayer_det_to_patch_meanheads.get('sum', []))
        if int(n_tokens) > cur:
            grow = int(n_tokens) - cur
            self.det_token_attn_lastlayer_det_to_patch_meanheads['sum'].extend(
                [np.zeros((self._attn_grid_size, self._attn_grid_size), dtype=np.float64) for _ in range(grow)]
            )

    def _ensure_det_token_attn_layer_buffers(self, n_layers: int, n_tokens: int):
        if not self.collect_stats or self.det_token_attn_layers_det_to_patch_meanheads is None:
            return
        if n_layers <= 0 or n_tokens <= 0:
            return

        obj = self.det_token_attn_layers_det_to_patch_meanheads
        if obj.get('sum') is None:
            obj['sum'] = [
                [np.zeros((self._attn_grid_size, self._attn_grid_size), dtype=np.float64) for _ in range(int(n_tokens))]
                for _ in range(int(n_layers))
            ]
            obj['cnt'] = np.zeros((self._attn_grid_size, self._attn_grid_size), dtype=np.float64)
            return

        sums = obj.get('sum')
        if not isinstance(sums, list):
            return

        if int(n_layers) > len(sums):
            for _ in range(int(n_layers) - len(sums)):
                sums.append([
                    np.zeros((self._attn_grid_size, self._attn_grid_size), dtype=np.float64) for _ in range(int(n_tokens))
                ])

        for li in range(min(int(n_layers), len(sums))):
            if not isinstance(sums[li], list):
                sums[li] = []
            if int(n_tokens) > len(sums[li]):
                grow = int(n_tokens) - len(sums[li])
                sums[li].extend(
                    [np.zeros((self._attn_grid_size, self._attn_grid_size), dtype=np.float64) for _ in range(grow)]
                )

    def update_attentions(self, attns, samples):
        """Accumulate last-layer det->patch attentions into per-token 100x100 grids.

        Expected attention layout (dinov3):
        - attns: [n_layers, B, n_heads, seq, seq] (or a compatible variant)
        - token order: [cls][register(optional)][patch][det]

        We take the last layer, average over heads, then accumulate det(query)->patch(key)
        into a 100x100 grid using patch-center normalized coordinates.
        """
        if not self.collect_stats:
            return
        if self.det_token_attn_lastlayer_det_to_patch_meanheads is None:
            return

        if attns is None:
            return

        # Accept either:
        # - reduced format: [B, det, num_patches] (preferred)
        # - reduced per-layer format: [L, B, det, num_patches]
        # - legacy/full format: [L,B,H,S,S] or [B,H,S,S]

        reduced = None
        reduced_layers = None
        if torch.is_tensor(attns) and attns.ndim == 4:
            reduced_layers = attns
            # Use last layer for the existing aggregate path.
            try:
                reduced = attns[-1]
            except Exception:
                reduced = None
        elif torch.is_tensor(attns) and attns.ndim == 3:
            reduced = attns
        else:
            # Fallback to previous parsing if a full attention tensor is provided.
            last = attns[-1] if isinstance(attns, (list, tuple)) and len(attns) > 0 else attns
            if not torch.is_tensor(last):
                return
            if last.ndim == 5:
                last = last[-1]
            if last.ndim != 4:
                return
            B, Hh, S, S2 = last.shape
            if S != S2:
                return

            # Determine det token count.
            n_det = None
            if isinstance(self.det_token_box_stats, dict) and isinstance(self.det_token_box_stats.get('cx'), list):
                n_det = len(self.det_token_box_stats['cx'])
            if n_det is None or n_det <= 0:
                if self._det_token_num is not None and int(self._det_token_num) > 0:
                    n_det = int(self._det_token_num)
            if n_det is None or n_det <= 0:
                return

            # Infer register token count and patch_size by matching patch_count.
            # (Uses padded batch size; this matches the tokenization performed by the conv patch embed.)
            # Extract det->patch attentions and average heads.
            # Token order: [cls][register][patch][det]
            reg_candidates = [0, 1, 2, 4, 6, 8, 16]
            try:
                pixel_values = samples.tensors if hasattr(samples, 'tensors') else samples
                if not torch.is_tensor(pixel_values) or pixel_values.ndim != 4:
                    return
                _, _, img_h, img_w = pixel_values.shape
                img_h = int(img_h)
                img_w = int(img_w)
            except Exception:
                return

            common_divs = [d for d in range(1, min(img_h, img_w) + 1) if (img_h % d == 0 and img_w % d == 0)]
            preferred = [16, 14, 8, 32]
            patch_size_candidates = [d for d in preferred if d in common_divs] + [d for d in common_divs if d not in preferred]

            reg = None
            patch_size = None
            num_patches = None
            for r in reg_candidates:
                pcount = int(S) - 1 - int(r) - int(n_det)
                if pcount <= 0:
                    continue
                for ps in patch_size_candidates:
                    hp = img_h // ps
                    wp = img_w // ps
                    if hp * wp == pcount:
                        reg = int(r)
                        patch_size = int(ps)
                        num_patches = int(pcount)
                        break
                if patch_size is not None:
                    break
            if patch_size is None or num_patches is None:
                return

            patch_s = 1 + reg
            patch_e = patch_s + num_patches
            det_s = patch_e
            det_e = det_s + n_det
            if det_e > S:
                return

            det_to_patch = last[:, :, det_s:det_e, patch_s:patch_e].mean(dim=1)
            reduced = det_to_patch

        if reduced is None or (not torch.is_tensor(reduced)) or reduced.ndim != 3:
            return

        B = int(reduced.shape[0])
        n_det = int(reduced.shape[1])
        num_patches = int(reduced.shape[2])

        # Image sizes from samples (padded batch); we use the tensor size for normalized patch grid.
        try:
            pixel_values = samples.tensors if hasattr(samples, 'tensors') else samples
            if not torch.is_tensor(pixel_values) or pixel_values.ndim != 4:
                return
            _, _, img_h, img_w = pixel_values.shape
            img_h = int(img_h)
            img_w = int(img_w)
        except Exception:
            return

        common_divs = [d for d in range(1, min(img_h, img_w) + 1) if (img_h % d == 0 and img_w % d == 0)]
        # Prefer typical ViT patch sizes first, otherwise fall back to any divisor.
        preferred = [16, 14, 8, 32]
        patch_size_candidates = [d for d in preferred if d in common_divs] + [d for d in common_divs if d not in preferred]

        patch_size = None
        for ps in patch_size_candidates:
            hp = img_h // ps
            wp = img_w // ps
            if hp * wp == num_patches:
                patch_size = int(ps)
                break

        if patch_size is None:
            return

        hp = img_h // patch_size
        wp = img_w // patch_size
        if hp <= 0 or wp <= 0 or hp * wp != num_patches:
            return

        det_to_patch = reduced.detach().float().cpu().numpy()
        det_to_patch_layers = None
        if reduced_layers is not None:
            try:
                det_to_patch_layers = reduced_layers.detach().float().cpu().numpy()
            except Exception:
                det_to_patch_layers = None

        # Precompute patch-center bins in [0,1].
        # Flatten order matches Conv2d output flatten(2): row-major over (h,w).
        ys = (np.arange(hp, dtype=np.float64) + 0.5) / float(hp)
        xs = (np.arange(wp, dtype=np.float64) + 0.5) / float(wp)
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        px = xx.reshape(-1)
        py = yy.reshape(-1)
        g = int(self._attn_grid_size)
        bx = np.clip((px * g).astype(np.int64), 0, g - 1)
        by = np.clip((py * g).astype(np.int64), 0, g - 1)

        base_cnt = np.zeros((g, g), dtype=np.float64)
        np.add.at(base_cnt, (by, bx), 1.0)

        self._ensure_det_token_attn_buffers(n_det)
        sum_grids = self.det_token_attn_lastlayer_det_to_patch_meanheads.get('sum')
        cnt_grid = self.det_token_attn_lastlayer_det_to_patch_meanheads.get('cnt')
        if not (isinstance(sum_grids, list) and isinstance(cnt_grid, np.ndarray)):
            return

        layer_sum = None
        layer_cnt = None
        if det_to_patch_layers is not None and self.det_token_attn_layers_det_to_patch_meanheads is not None:
            try:
                n_layers = int(det_to_patch_layers.shape[0])
                self._ensure_det_token_attn_layer_buffers(n_layers, n_det)
                layer_sum = self.det_token_attn_layers_det_to_patch_meanheads.get('sum')
                layer_cnt = self.det_token_attn_layers_det_to_patch_meanheads.get('cnt')
            except Exception:
                layer_sum, layer_cnt = None, None

        for b in range(min(B, det_to_patch.shape[0])):
            # Accumulate counts once per image (shared across tokens).
            cnt_grid += base_cnt
            if isinstance(layer_cnt, np.ndarray) and layer_cnt.shape == cnt_grid.shape:
                layer_cnt += base_cnt

            # Accumulate sums per token.
            for t in range(min(n_det, det_to_patch.shape[1], len(sum_grids))):
                a = det_to_patch[b, t]
                if a.shape[0] != num_patches:
                    continue
                np.add.at(sum_grids[t], (by, bx), a.astype(np.float64, copy=False))

            # Accumulate per-layer sums per token.
            if det_to_patch_layers is not None and isinstance(layer_sum, list) and layer_sum:
                for li in range(min(int(det_to_patch_layers.shape[0]), len(layer_sum))):
                    per_tok = layer_sum[li]
                    if not isinstance(per_tok, list):
                        continue
                    if b >= int(det_to_patch_layers.shape[1]):
                        continue
                    for t in range(min(n_det, int(det_to_patch_layers.shape[2]), len(per_tok))):
                        a = det_to_patch_layers[li, b, t]
                        if a.shape[0] != num_patches:
                            continue
                        np.add.at(per_tok[t], (by, bx), a.astype(np.float64, copy=False))

    def synchronize_between_processes(self):
        # NOTE: This stats-only helper is used as a collector by datasets/coco_eval.py.
        # We intentionally do NOT synchronize COCOeval evalImgs here to avoid requiring
        # that this object runs the full evaluation loop.

        def _merge_det_token_stats(stats_obj):
            if not (self.collect_stats and isinstance(stats_obj, dict) and isinstance(stats_obj.get('cx'), list)):
                return stats_obj
            gathered = all_gather(stats_obj)
            max_n = 0
            for s in gathered:
                if not isinstance(s, dict):
                    continue
                cx = s.get('cx')
                if isinstance(cx, list):
                    max_n = max(max_n, len(cx))
            if max_n <= 0:
                return stats_obj

            merged = {
                'cx': [[] for _ in range(max_n)],
                'cy': [[] for _ in range(max_n)],
                'area': [[] for _ in range(max_n)],
                'score': [[] for _ in range(max_n)],
                'label': [[] for _ in range(max_n)],
                'x0': [[] for _ in range(max_n)],
                'y0': [[] for _ in range(max_n)],
                'x1': [[] for _ in range(max_n)],
                'y1': [[] for _ in range(max_n)],
            }
            for s in gathered:
                if not isinstance(s, dict):
                    continue
                for k in ('cx', 'cy', 'area', 'score', 'label', 'x0', 'y0', 'x1', 'y1'):
                    arr = s.get(k)
                    if not isinstance(arr, list):
                        continue
                    for t in range(min(len(arr), max_n)):
                        if isinstance(arr[t], list) and arr[t]:
                            merged[k][t].extend(arr[t])
            return merged

        def _merge_det_token_attn(attn_obj):
            if not (self.collect_stats and isinstance(attn_obj, dict)):
                return attn_obj
            if not (isinstance(attn_obj.get('sum'), list) and isinstance(attn_obj.get('cnt'), np.ndarray)):
                return attn_obj

            gathered = all_gather(attn_obj)
            max_n = 0
            for s in gathered:
                if not isinstance(s, dict):
                    continue
                sums = s.get('sum')
                if isinstance(sums, list):
                    max_n = max(max_n, len(sums))
            if max_n <= 0:
                return attn_obj

            merged_sum = [np.zeros((self._attn_grid_size, self._attn_grid_size), dtype=np.float64) for _ in range(max_n)]
            merged_cnt = np.zeros((self._attn_grid_size, self._attn_grid_size), dtype=np.float64)
            for s in gathered:
                if not isinstance(s, dict):
                    continue
                sums = s.get('sum')
                cnts = s.get('cnt')
                if not (isinstance(sums, list) and isinstance(cnts, np.ndarray)):
                    continue
                for t in range(min(max_n, len(sums))):
                    if isinstance(sums[t], np.ndarray) and sums[t].shape == merged_sum[t].shape:
                        merged_sum[t] += sums[t]
                if cnts.shape == merged_cnt.shape:
                    merged_cnt += cnts

            return {'sum': merged_sum, 'cnt': merged_cnt}

        if self.collect_stats and self.det_token_box_stats is not None:
            self.det_token_box_stats = _merge_det_token_stats(self.det_token_box_stats)
            if isinstance(self.det_token_box_stats.get('cx'), list):
                self._det_token_num = len(self.det_token_box_stats['cx'])
        if self.collect_stats and self.det_token_box_stats_iou50_recall_matched_useCats1 is not None:
            self.det_token_box_stats_iou50_recall_matched_useCats1 = _merge_det_token_stats(
                self.det_token_box_stats_iou50_recall_matched_useCats1
            )

        if self.collect_stats and self.det_token_attn_lastlayer_det_to_patch_meanheads is not None:
            self.det_token_attn_lastlayer_det_to_patch_meanheads = _merge_det_token_attn(
                self.det_token_attn_lastlayer_det_to_patch_meanheads
            )

        def _merge_det_token_attn_layers(attn_obj):
            if not (self.collect_stats and isinstance(attn_obj, dict)):
                return attn_obj
            if not (isinstance(attn_obj.get('sum'), list) and isinstance(attn_obj.get('cnt'), np.ndarray)):
                return attn_obj

            gathered = all_gather(attn_obj)
            max_layers = 0
            max_tokens = 0
            for s in gathered:
                if not isinstance(s, dict):
                    continue
                sums = s.get('sum')
                if not isinstance(sums, list):
                    continue
                max_layers = max(max_layers, len(sums))
                for layer_sum in sums:
                    if isinstance(layer_sum, list):
                        max_tokens = max(max_tokens, len(layer_sum))
            if max_layers <= 0 or max_tokens <= 0:
                return attn_obj

            merged_sum = [
                [np.zeros((self._attn_grid_size, self._attn_grid_size), dtype=np.float64) for _ in range(max_tokens)]
                for _ in range(max_layers)
            ]
            merged_cnt = np.zeros((self._attn_grid_size, self._attn_grid_size), dtype=np.float64)

            for s in gathered:
                if not isinstance(s, dict):
                    continue
                sums = s.get('sum')
                cnts = s.get('cnt')
                if not (isinstance(sums, list) and isinstance(cnts, np.ndarray)):
                    continue
                for li in range(min(max_layers, len(sums))):
                    layer_sum = sums[li]
                    if not isinstance(layer_sum, list):
                        continue
                    for t in range(min(max_tokens, len(layer_sum))):
                        arr = layer_sum[t]
                        if isinstance(arr, np.ndarray) and arr.shape == merged_sum[li][t].shape:
                            merged_sum[li][t] += arr
                if cnts.shape == merged_cnt.shape:
                    merged_cnt += cnts

            return {'sum': merged_sum, 'cnt': merged_cnt}

        if self.collect_stats and self.det_token_attn_layers_det_to_patch_meanheads is not None:
            self.det_token_attn_layers_det_to_patch_meanheads = _merge_det_token_attn_layers(
                self.det_token_attn_layers_det_to_patch_meanheads
            )

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()
            self.summarize_per_category(iou_type=iou_type, iou_thr=0.50, max_dets=100, area="all")

    def _export_det_token_hittable_region_stats_to_csv(self, output_path: Path):
        """Export per-token hittable region and coverage ratios (all normalized).

        Definitions (per det token t):
        - Hit boxes: COCOeval recall-matched detections (useCats=1, IoU=0.50).
        - Hittable region R_t: [min(x0), min(y0), max(x1), max(y1)] over hit boxes.
        - Hit area range: [min(area), max(area)] over hit boxes.
        - Coverage ratios are computed over *all* predicted boxes of that token:
          p_in_region: box fully inside R_t;
          p_in_area_range: box area in [min,max];
          p_both: both conditions.
        """
        if not self.collect_stats:
            return
        all_obj = self.det_token_box_stats
        hit_obj = self.det_token_box_stats_iou50_recall_matched_useCats1
        if not (isinstance(all_obj, dict) and isinstance(hit_obj, dict)):
            return
        if not (isinstance(all_obj.get('x0'), list) and isinstance(hit_obj.get('x0'), list)):
            return

        n_tokens = max(len(all_obj.get('x0', [])), len(hit_obj.get('x0', [])))
        if n_tokens <= 0:
            return

        out_path = output_path / 'det_token_hittable_region_iou0.50_recall_matched_useCats1_normalized.csv'
        with open(out_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'token_idx',
                'all_cnt',
                'hit_cnt',
                'hit_region_area',
                'hit_area_min',
                'hit_area_max',
                'hit_area_min_over_region',
                'hit_area_max_over_region',
                'p_in_region',
                'p_in_hit_area_range',
                'p_both',

                # Best bucket by hit-rate = hit_count / all_count.
                # - area bins: 0.01 (rounded to 2 decimals)
                # - (cx,cy) bins: 0.1 (rounded to 1 decimal)
                # "token数" here means the number of token instances across images.
                'best_area_bin', 'best_area_hit_rate', 'best_area_hit_cnt', 'best_area_all_cnt',
                'best_cx_bin', 'best_cy_bin', 'best_xy_hit_rate', 'best_xy_hit_cnt', 'best_xy_all_cnt',
            ])

            def _best_bucket_1d(all_vals, hit_vals):
                if not (isinstance(all_vals, (list, tuple, np.ndarray)) and len(all_vals) > 0):
                    return (float('nan'), float('nan'), 0, 0)
                a = np.asarray(all_vals, dtype=np.float64)
                a = a[np.isfinite(a)]
                if a.size == 0:
                    return (float('nan'), float('nan'), 0, 0)
                a_bin = np.round(a, 2)

                h = np.asarray(hit_vals, dtype=np.float64) if isinstance(hit_vals, (list, tuple, np.ndarray)) else np.asarray([], dtype=np.float64)
                h = h[np.isfinite(h)]
                h_bin = np.round(h, 2) if h.size > 0 else np.asarray([], dtype=np.float64)

                uniq_a, cnt_a = np.unique(a_bin, return_counts=True)
                total = {float(k): int(v) for k, v in zip(uniq_a, cnt_a)}
                if h_bin.size > 0:
                    uniq_h, cnt_h = np.unique(h_bin, return_counts=True)
                    hit = {float(k): int(v) for k, v in zip(uniq_h, cnt_h)}
                else:
                    hit = {}

                best_k = None
                best_rate = -1.0
                best_hit = -1
                best_all = -1
                for k, all_c in total.items():
                    hit_c = hit.get(k, 0)
                    rate = float(hit_c) / float(all_c) if all_c > 0 else 0.0
                    if (
                        (rate > best_rate) or
                        (rate == best_rate and hit_c > best_hit) or
                        (rate == best_rate and hit_c == best_hit and all_c > best_all) or
                        (rate == best_rate and hit_c == best_hit and all_c == best_all and (best_k is None or k < best_k))
                    ):
                        best_k = k
                        best_rate = rate
                        best_hit = hit_c
                        best_all = all_c

                if best_k is None:
                    return (float('nan'), float('nan'), 0, 0)
                return (float(best_k), float(best_rate), int(best_hit), int(best_all))

            def _best_bucket_xy(all_cx, all_cy, hit_cx, hit_cy):
                if not (
                    isinstance(all_cx, (list, tuple, np.ndarray)) and isinstance(all_cy, (list, tuple, np.ndarray))
                ):
                    return (float('nan'), float('nan'), float('nan'), 0, 0)

                n_all_xy = min(len(all_cx), len(all_cy))
                if n_all_xy <= 0:
                    return (float('nan'), float('nan'), float('nan'), 0, 0)

                acx = np.asarray(all_cx[:n_all_xy], dtype=np.float64)
                acy = np.asarray(all_cy[:n_all_xy], dtype=np.float64)
                mask_a = np.isfinite(acx) & np.isfinite(acy)
                acx = np.round(acx[mask_a], 1)
                acy = np.round(acy[mask_a], 1)
                if acx.size == 0:
                    return (float('nan'), float('nan'), float('nan'), 0, 0)

                hcx = np.asarray(hit_cx, dtype=np.float64) if isinstance(hit_cx, (list, tuple, np.ndarray)) else np.asarray([], dtype=np.float64)
                hcy = np.asarray(hit_cy, dtype=np.float64) if isinstance(hit_cy, (list, tuple, np.ndarray)) else np.asarray([], dtype=np.float64)
                n_hit_xy = min(hcx.size, hcy.size)
                hcx = hcx[:n_hit_xy]
                hcy = hcy[:n_hit_xy]
                mask_h = np.isfinite(hcx) & np.isfinite(hcy)
                hcx = np.round(hcx[mask_h], 1)
                hcy = np.round(hcy[mask_h], 1)

                total = defaultdict(int)
                for x, y in zip(acx.tolist(), acy.tolist()):
                    total[(float(x), float(y))] += 1
                hit = defaultdict(int)
                for x, y in zip(hcx.tolist(), hcy.tolist()):
                    hit[(float(x), float(y))] += 1

                best_key = None
                best_rate = -1.0
                best_hit = -1
                best_all = -1
                for key, all_c in total.items():
                    hit_c = int(hit.get(key, 0))
                    rate = float(hit_c) / float(all_c) if all_c > 0 else 0.0
                    if (
                        (rate > best_rate) or
                        (rate == best_rate and hit_c > best_hit) or
                        (rate == best_rate and hit_c == best_hit and all_c > best_all) or
                        (
                            rate == best_rate and hit_c == best_hit and all_c == best_all and
                            (best_key is None or key < best_key)
                        )
                    ):
                        best_key = key
                        best_rate = rate
                        best_hit = hit_c
                        best_all = int(all_c)

                if best_key is None:
                    return (float('nan'), float('nan'), float('nan'), 0, 0)
                return (float(best_key[0]), float(best_key[1]), float(best_rate), int(best_hit), int(best_all))

            for t in range(n_tokens):
                # All predicted boxes for this token
                ax0 = all_obj.get('x0', [[] for _ in range(n_tokens)])[t] if isinstance(all_obj.get('x0'), list) and t < len(all_obj['x0']) else []
                ay0 = all_obj.get('y0', [[] for _ in range(n_tokens)])[t] if isinstance(all_obj.get('y0'), list) and t < len(all_obj['y0']) else []
                ax1 = all_obj.get('x1', [[] for _ in range(n_tokens)])[t] if isinstance(all_obj.get('x1'), list) and t < len(all_obj['x1']) else []
                ay1 = all_obj.get('y1', [[] for _ in range(n_tokens)])[t] if isinstance(all_obj.get('y1'), list) and t < len(all_obj['y1']) else []
                aa = all_obj.get('area', [[] for _ in range(n_tokens)])[t] if isinstance(all_obj.get('area'), list) and t < len(all_obj['area']) else []
                acx = all_obj.get('cx', [[] for _ in range(n_tokens)])[t] if isinstance(all_obj.get('cx'), list) and t < len(all_obj['cx']) else []
                acy = all_obj.get('cy', [[] for _ in range(n_tokens)])[t] if isinstance(all_obj.get('cy'), list) and t < len(all_obj['cy']) else []
                n_all = 0
                if (
                    isinstance(ax0, list) and isinstance(ay0, list) and isinstance(ax1, list) and isinstance(ay1, list) and
                    isinstance(aa, list) and isinstance(acx, list) and isinstance(acy, list)
                ):
                    n_all = min(len(ax0), len(ay0), len(ax1), len(ay1), len(aa), len(acx), len(acy))

                # Hit boxes for this token
                hx0 = hit_obj.get('x0', [[] for _ in range(n_tokens)])[t] if isinstance(hit_obj.get('x0'), list) and t < len(hit_obj['x0']) else []
                hy0 = hit_obj.get('y0', [[] for _ in range(n_tokens)])[t] if isinstance(hit_obj.get('y0'), list) and t < len(hit_obj['y0']) else []
                hx1 = hit_obj.get('x1', [[] for _ in range(n_tokens)])[t] if isinstance(hit_obj.get('x1'), list) and t < len(hit_obj['x1']) else []
                hy1 = hit_obj.get('y1', [[] for _ in range(n_tokens)])[t] if isinstance(hit_obj.get('y1'), list) and t < len(hit_obj['y1']) else []
                ha = hit_obj.get('area', [[] for _ in range(n_tokens)])[t] if isinstance(hit_obj.get('area'), list) and t < len(hit_obj['area']) else []
                hcx = hit_obj.get('cx', [[] for _ in range(n_tokens)])[t] if isinstance(hit_obj.get('cx'), list) and t < len(hit_obj['cx']) else []
                hcy = hit_obj.get('cy', [[] for _ in range(n_tokens)])[t] if isinstance(hit_obj.get('cy'), list) and t < len(hit_obj['cy']) else []
                n_hit = 0
                if (
                    isinstance(hx0, list) and isinstance(hy0, list) and isinstance(hx1, list) and isinstance(hy1, list) and
                    isinstance(ha, list) and isinstance(hcx, list) and isinstance(hcy, list)
                ):
                    n_hit = min(len(hx0), len(hy0), len(hx1), len(hy1), len(ha), len(hcx), len(hcy))

                if n_hit <= 0:
                    writer.writerow([
                        t, int(n_all), 0,
                        'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan',
                        'nan', 'nan', 0, 0,
                        'nan', 'nan', 'nan', 0, 0,
                    ])
                    continue

                hx0a = np.asarray(hx0[:n_hit], dtype=np.float64)
                hy0a = np.asarray(hy0[:n_hit], dtype=np.float64)
                hx1a = np.asarray(hx1[:n_hit], dtype=np.float64)
                hy1a = np.asarray(hy1[:n_hit], dtype=np.float64)
                # Ensure ordering
                hx0a, hx1a = np.minimum(hx0a, hx1a), np.maximum(hx0a, hx1a)
                hy0a, hy1a = np.minimum(hy0a, hy1a), np.maximum(hy0a, hy1a)

                rx0 = float(np.min(hx0a))
                ry0 = float(np.min(hy0a))
                rx1 = float(np.max(hx1a))
                ry1 = float(np.max(hy1a))
                hit_region_area = max(0.0, rx1 - rx0) * max(0.0, ry1 - ry0)

                ha_arr = np.asarray(ha[:n_hit], dtype=np.float64)
                ha_arr = np.maximum(ha_arr, 0.0)
                hit_area_min = float(np.min(ha_arr))
                hit_area_max = float(np.max(ha_arr))

                if hit_region_area > 0.0:
                    r_min = hit_area_min / hit_region_area
                    r_max = hit_area_max / hit_region_area
                else:
                    r_min = float('nan')
                    r_max = float('nan')

                p_in_region = float('nan')
                p_in_area = float('nan')
                p_both = float('nan')
                if n_all > 0:
                    ax0a = np.asarray(ax0[:n_all], dtype=np.float64)
                    ay0a = np.asarray(ay0[:n_all], dtype=np.float64)
                    ax1a = np.asarray(ax1[:n_all], dtype=np.float64)
                    ay1a = np.asarray(ay1[:n_all], dtype=np.float64)
                    aa_arr = np.asarray(aa[:n_all], dtype=np.float64)
                    aa_arr = np.maximum(aa_arr, 0.0)

                    # Ensure ordering
                    ax0a, ax1a = np.minimum(ax0a, ax1a), np.maximum(ax0a, ax1a)
                    ay0a, ay1a = np.minimum(ay0a, ay1a), np.maximum(ay0a, ay1a)

                    in_region = (ax0a >= rx0) & (ay0a >= ry0) & (ax1a <= rx1) & (ay1a <= ry1)
                    in_area = (aa_arr >= hit_area_min) & (aa_arr <= hit_area_max)
                    both = in_region & in_area

                    p_in_region = float(in_region.mean())
                    p_in_area = float(in_area.mean())
                    p_both = float(both.mean())

                best_area_bin, best_area_rate, best_area_hit_cnt, best_area_all_cnt = _best_bucket_1d(
                    aa[:n_all] if n_all > 0 else [],
                    ha[:n_hit],
                )
                best_cx_bin, best_cy_bin, best_xy_rate, best_xy_hit_cnt, best_xy_all_cnt = _best_bucket_xy(
                    acx[:n_all] if n_all > 0 else [],
                    acy[:n_all] if n_all > 0 else [],
                    hcx[:n_hit],
                    hcy[:n_hit],
                )

                writer.writerow([
                    t,
                    int(n_all),
                    int(n_hit),
                    f"{hit_region_area:.6f}",
                    f"{hit_area_min:.6f}",
                    f"{hit_area_max:.6f}",
                    f"{r_min:.6f}" if np.isfinite(r_min) else 'nan',
                    f"{r_max:.6f}" if np.isfinite(r_max) else 'nan',
                    f"{p_in_region:.6f}" if np.isfinite(p_in_region) else 'nan',
                    f"{p_in_area:.6f}" if np.isfinite(p_in_area) else 'nan',
                    f"{p_both:.6f}" if np.isfinite(p_both) else 'nan',

                    f"{best_area_bin:.2f}" if np.isfinite(best_area_bin) else 'nan',
                    f"{best_area_rate:.6f}" if np.isfinite(best_area_rate) else 'nan',
                    int(best_area_hit_cnt),
                    int(best_area_all_cnt),
                    f"{best_cx_bin:.1f}" if np.isfinite(best_cx_bin) else 'nan',
                    f"{best_cy_bin:.1f}" if np.isfinite(best_cy_bin) else 'nan',
                    f"{best_xy_rate:.6f}" if np.isfinite(best_xy_rate) else 'nan',
                    int(best_xy_hit_cnt),
                    int(best_xy_all_cnt),
                ])

        self._print_stats("CSV", f"det-token hittable region stats (useCats=1 recall-matched @ IoU=0.50): {out_path}")

    def _export_det_token_heatmaps_to_png(self, output_path: Path):
        """Export visualization PNGs.

                Outputs (aggregate over the full val set):
                - Per token: center scatter plot + attention heatmap.
                    File: total/token_XXXX.png
                - Global: GT center scatter plot.
                    File: total/gt.png

        Notes:
        - Hit is COCOeval recall-matched (useCats=1, IoU=0.50).
        - All coordinates and areas are normalized to [0,1].
        """
        if not self.collect_stats:
            return
        all_obj = self.det_token_box_stats
        hit_obj = self.det_token_box_stats_iou50_recall_matched_useCats1
        if not (isinstance(all_obj, dict) and isinstance(hit_obj, dict)):
            return
        if not (isinstance(all_obj.get('cx'), list) and isinstance(all_obj.get('cy'), list)):
            return

        attn_obj = self.det_token_attn_lastlayer_det_to_patch_meanheads
        attn_sum = attn_obj.get('sum') if isinstance(attn_obj, dict) else None
        attn_cnt = attn_obj.get('cnt') if isinstance(attn_obj, dict) else None

        attn_layers_obj = self.det_token_attn_layers_det_to_patch_meanheads
        attn_layers_sum = attn_layers_obj.get('sum') if isinstance(attn_layers_obj, dict) else None
        attn_layers_cnt = attn_layers_obj.get('cnt') if isinstance(attn_layers_obj, dict) else None

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib import colors as mcolors
            from matplotlib.lines import Line2D
            import matplotlib.gridspec as gridspec
        except Exception as e:
            self._print_stats("WARN", f"matplotlib not available; skip PNG export: {e}")
            return

        n_tokens = len(all_obj.get('cx', [])) if isinstance(all_obj.get('cx'), list) else 0
        if n_tokens <= 0:
            return

        out_dir = output_path / self._viz_total_dirname
        out_dir.mkdir(parents=True, exist_ok=True)

        def _to_float_array(v):
            if not isinstance(v, (list, tuple, np.ndarray)):
                return np.asarray([], dtype=np.float64)
            a = np.asarray(v, dtype=np.float64)
            a = a[np.isfinite(a)]
            return a

        # Global GT scatter (one PNG)
        gt_cx = []
        gt_cy = []
        gt_area = []
        try:
            coco_imgs = getattr(self.coco_gt, 'imgs', {})
            coco_anns = getattr(self.coco_gt, 'anns', {})
            if isinstance(coco_imgs, dict) and isinstance(coco_anns, dict):
                for _, ann in coco_anns.items():
                    if not isinstance(ann, dict):
                        continue
                    img_id = ann.get('image_id', None)
                    if img_id is None:
                        continue
                    im = coco_imgs.get(int(img_id), None)
                    if not isinstance(im, dict):
                        continue
                    img_w = float(im.get('width', 0.0))
                    img_h = float(im.get('height', 0.0))
                    if img_w <= 0 or img_h <= 0:
                        continue
                    bbox = ann.get('bbox', None)
                    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                        continue
                    x0 = float(bbox[0])
                    y0 = float(bbox[1])
                    w = max(0.0, float(bbox[2]))
                    h = max(0.0, float(bbox[3]))
                    x1 = x0 + w
                    y1 = y0 + h
                    cx = ((x0 + x1) * 0.5) / img_w
                    cy = ((y0 + y1) * 0.5) / img_h
                    area = (w * h) / (img_w * img_h)
                    if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(area)):
                        continue
                    gt_cx.append(float(np.clip(cx, 0.0, 1.0)))
                    gt_cy.append(float(np.clip(cy, 0.0, 1.0)))
                    gt_area.append(float(np.clip(area, 0.0, 1.0)))
        except Exception:
            gt_cx, gt_cy, gt_area = [], [], []

        for t in range(n_tokens):
            acx_list = all_obj.get('cx', [])[t] if t < len(all_obj.get('cx', [])) else []
            acy_list = all_obj.get('cy', [])[t] if t < len(all_obj.get('cy', [])) else []
            aa_list = all_obj.get('area', [])[t] if t < len(all_obj.get('area', [])) else []
            asc_list = all_obj.get('score', [])[t] if isinstance(all_obj.get('score'), list) and t < len(all_obj.get('score', [])) else []

            hcx_list = hit_obj.get('cx', [])[t] if t < len(hit_obj.get('cx', [])) else []
            hcy_list = hit_obj.get('cy', [])[t] if t < len(hit_obj.get('cy', [])) else []
            ha_list = hit_obj.get('area', [])[t] if t < len(hit_obj.get('area', [])) else []

            acx = _to_float_array(acx_list)
            acy = _to_float_array(acy_list)
            aa = _to_float_array(aa_list)
            asc = _to_float_array(asc_list)

            hcx = _to_float_array(hcx_list)
            hcy = _to_float_array(hcy_list)
            ha = _to_float_array(ha_list)

            n_all = int(min(acx.size, acy.size, aa.size))
            n_hit = int(min(hcx.size, hcy.size, ha.size))
            if n_all <= 0:
                continue

            # Mean confidence score for this token across the full val set.
            mean_score = float('nan')
            if asc.size > 0:
                try:
                    mean_score = float(np.mean(np.clip(asc[:n_all], 0.0, 1.0)))
                except Exception:
                    mean_score = float('nan')

            token_hit_rate = (float(n_hit) / float(n_all)) if n_all > 0 else float('nan')

            sc_fig = plt.figure(figsize=(24, 10), dpi=200)
            gs = gridspec.GridSpec(nrows=2, ncols=12, figure=sc_fig, height_ratios=[4.0, 1.2])
            sc_ax = sc_fig.add_subplot(gs[0, 0:6])
            attn_ax = sc_fig.add_subplot(gs[0, 6:12])
            thumb_axes = [sc_fig.add_subplot(gs[1, i]) for i in range(12)]

            sc_ax.set_title(
                f"token={t} centers  hit_rate={token_hit_rate:.6f}  mean_score={mean_score:.4f}  (hit={n_hit}, all={n_all})",
                fontsize=12,
            )
            sc_ax.set_xlabel('cx (normalized)')
            sc_ax.set_ylabel('cy (normalized)')
            sc_ax.set_xlim(0.0, 1.0)
            sc_ax.set_ylim(0.0, 1.0)
            sc_ax.set_aspect('equal', adjustable='box')
            sc_ax.set_xticks([0.0, 0.5, 1.0])
            sc_ax.set_yticks([0.0, 0.5, 1.0])

            norm = mcolors.Normalize(vmin=0.0, vmax=1.0, clip=True)
            acx_p = np.clip(acx[:n_all], 0.0, 1.0)
            acy_p = np.clip(acy[:n_all], 0.0, 1.0)
            aa_p = np.clip(aa[:n_all], 0.0, 1.0)
            asc_p = np.clip(asc[:n_all], 0.0, 1.0) if asc.size >= n_all else None
            hcx_p = np.clip(hcx[:n_hit], 0.0, 1.0) if n_hit > 0 else np.asarray([], dtype=np.float64)
            hcy_p = np.clip(hcy[:n_hit], 0.0, 1.0) if n_hit > 0 else np.asarray([], dtype=np.float64)
            ha_p = np.clip(ha[:n_hit], 0.0, 1.0) if n_hit > 0 else np.asarray([], dtype=np.float64)

            # Map confidence score to alpha for dt centers.
            try:
                cmap = plt.cm.get_cmap('viridis')
                rgba = cmap(aa_p)
                if asc_p is not None:
                    a = 0.05 + 0.75 * asc_p
                    rgba[:, 3] = np.clip(a, 0.05, 0.85)
                else:
                    rgba[:, 3] = 0.35
                all_sc = sc_ax.scatter(
                    acx_p,
                    acy_p,
                    color=rgba,
                    marker='x',
                    s=10,
                    linewidths=0.6,
                )
            except Exception:
                all_sc = sc_ax.scatter(
                    acx_p,
                    acy_p,
                    c=aa_p,
                    cmap='viridis',
                    norm=norm,
                    marker='x',
                    s=10,
                    linewidths=0.6,
                    alpha=0.35,
                )
            if n_hit > 0:
                sc_ax.scatter(
                    hcx_p,
                    hcy_p,
                    c=ha_p,
                    cmap='viridis',
                    norm=norm,
                    marker='o',
                    s=18,
                    edgecolors='k',
                    linewidths=0.25,
                    alpha=0.85,
                )

            cbar = sc_fig.colorbar(all_sc, ax=sc_ax, fraction=0.046, pad=0.04)
            cbar.set_label('area (normalized)')

            legend_elems = [
                Line2D([0], [0], marker='x', color='0.3', linestyle='None', markersize=7, label='all'),
                Line2D([0], [0], marker='o', color='k', linestyle='None', markersize=7, label='hit'),
            ]
            sc_ax.legend(handles=legend_elems, loc='lower right', frameon=True)
            sc_ax.text(0.01, 0.99, f"all={n_all}  hit={n_hit}", transform=sc_ax.transAxes, va='top', ha='left', color='black', fontsize=10)

            # Right panel: attention heatmap (val-set averaged, 100x100)
            attn_ax.set_title('last-layer det→patch attention (mean heads)', fontsize=12)
            attn_ax.set_xlabel('x (normalized)')
            attn_ax.set_ylabel('y (normalized)')
            attn_ax.set_xlim(0.0, 1.0)
            attn_ax.set_ylim(0.0, 1.0)
            attn_ax.set_aspect('equal', adjustable='box')
            attn_ax.set_xticks([0.0, 0.5, 1.0])
            attn_ax.set_yticks([0.0, 0.5, 1.0])

            attn_img = None
            if (
                isinstance(attn_sum, list) and isinstance(attn_cnt, np.ndarray) and
                t < len(attn_sum) and isinstance(attn_sum[t], np.ndarray) and
                attn_sum[t].shape == attn_cnt.shape and attn_sum[t].ndim == 2 and attn_cnt.ndim == 2
            ):
                cnt = attn_cnt.astype(np.float64, copy=False)
                s = attn_sum[t].astype(np.float64, copy=False)
                with np.errstate(divide='ignore', invalid='ignore'):
                    avg = np.where(cnt > 0.0, s / cnt, 0.0)
                maxv = float(np.max(avg)) if avg.size > 0 else 0.0
                if np.isfinite(maxv) and maxv > 0.0:
                    avg = avg / maxv
                else:
                    avg = np.zeros_like(avg)
                attn_img = attn_ax.imshow(
                    avg,
                    origin='lower',
                    extent=(0.0, 1.0, 0.0, 1.0),
                    vmin=0.0,
                    vmax=1.0,
                    cmap='viridis',
                    interpolation='nearest',
                )
            else:
                attn_ax.text(0.5, 0.5, 'no attn data', ha='center', va='center', transform=attn_ax.transAxes)

            if attn_img is not None:
                cbar2 = sc_fig.colorbar(attn_img, ax=attn_ax, fraction=0.046, pad=0.04)
                cbar2.set_label('attn (normalized)')

            # Bottom row: 12-layer thumbnails (same as last-layer but per layer).
            try:
                layer_indices = list(range(12))
                if isinstance(attn_layers_sum, list) and len(attn_layers_sum) > 0:
                    L = int(len(attn_layers_sum))
                    start = max(0, L - 12)
                    layer_indices = list(range(start, start + 12))
                for i, ax in enumerate(thumb_axes):
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f"L{layer_indices[i]}", fontsize=7)
                    img_i = None
                    if (
                        isinstance(attn_layers_sum, list) and isinstance(attn_layers_cnt, np.ndarray) and
                        0 <= layer_indices[i] < len(attn_layers_sum)
                    ):
                        per_layer = attn_layers_sum[layer_indices[i]]
                        if (
                            isinstance(per_layer, list) and t < len(per_layer) and
                            isinstance(per_layer[t], np.ndarray) and per_layer[t].shape == attn_layers_cnt.shape
                        ):
                            cnt = attn_layers_cnt.astype(np.float64, copy=False)
                            s = per_layer[t].astype(np.float64, copy=False)
                            with np.errstate(divide='ignore', invalid='ignore'):
                                avg = np.where(cnt > 0.0, s / cnt, 0.0)
                            maxv = float(np.max(avg)) if avg.size > 0 else 0.0
                            if np.isfinite(maxv) and maxv > 0.0:
                                avg = avg / maxv
                            else:
                                avg = np.zeros_like(avg)
                            img_i = ax.imshow(
                                avg,
                                origin='lower',
                                vmin=0.0,
                                vmax=1.0,
                                cmap='viridis',
                                interpolation='nearest',
                            )
                    if img_i is None:
                        ax.text(0.5, 0.5, 'no', ha='center', va='center', transform=ax.transAxes, fontsize=8)
            except Exception:
                pass

            sc_out_path = out_dir / f"token_{t:04d}.png"
            sc_fig.tight_layout()
            sc_fig.savefig(sc_out_path)
            plt.close(sc_fig)

        if len(gt_cx) > 0:
            gt_fig, gt_ax = plt.subplots(1, 1, figsize=(9, 9), dpi=200)
            gt_ax.set_title(f"GT centers (n={len(gt_cx)})", fontsize=12)
            gt_ax.set_xlabel('cx (normalized)')
            gt_ax.set_ylabel('cy (normalized)')
            gt_ax.set_xlim(0.0, 1.0)
            gt_ax.set_ylim(0.0, 1.0)
            gt_ax.set_aspect('equal', adjustable='box')
            gt_ax.set_xticks([0.0, 0.5, 1.0])
            gt_ax.set_yticks([0.0, 0.5, 1.0])

            norm = mcolors.Normalize(vmin=0.0, vmax=1.0, clip=True)
            gt_sc = gt_ax.scatter(
                np.asarray(gt_cx, dtype=np.float64),
                np.asarray(gt_cy, dtype=np.float64),
                c=np.asarray(gt_area, dtype=np.float64),
                cmap='viridis',
                norm=norm,
                marker='o',
                s=10,
                linewidths=0.0,
                alpha=0.6,
            )
            cbar = gt_fig.colorbar(gt_sc, ax=gt_ax, fraction=0.046, pad=0.04)
            cbar.set_label('area (normalized)')
            gt_out_path = out_dir / 'gt.png'
            gt_fig.tight_layout()
            gt_fig.savefig(gt_out_path)
            plt.close(gt_fig)

        self._print_stats("PNG", f"center+attention PNGs exported to: {out_dir}")

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

            scores = prediction["scores"]
            labels = prediction["labels"]
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

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

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

    def _ensure_det_token_box_buffers(self, n_tokens: int):
        if not self.collect_stats or self.det_token_box_stats is None:
            return
        if n_tokens <= 0:
            return

        if self._det_token_num is None or self.det_token_box_stats.get('cx') is None:
            self._det_token_num = int(n_tokens)
            self.det_token_box_stats['cx'] = [[] for _ in range(self._det_token_num)]
            self.det_token_box_stats['cy'] = [[] for _ in range(self._det_token_num)]
            self.det_token_box_stats['area'] = [[] for _ in range(self._det_token_num)]
            self.det_token_box_stats['score'] = [[] for _ in range(self._det_token_num)]
            self.det_token_box_stats['label'] = [[] for _ in range(self._det_token_num)]
            self.det_token_box_stats['x0'] = [[] for _ in range(self._det_token_num)]
            self.det_token_box_stats['y0'] = [[] for _ in range(self._det_token_num)]
            self.det_token_box_stats['x1'] = [[] for _ in range(self._det_token_num)]
            self.det_token_box_stats['y1'] = [[] for _ in range(self._det_token_num)]
            return

        if int(n_tokens) > int(self._det_token_num):
            new_n = int(n_tokens)
            for k in ('cx', 'cy', 'area', 'score', 'label', 'x0', 'y0', 'x1', 'y1'):
                self.det_token_box_stats[k].extend([[] for _ in range(new_n - int(self._det_token_num))])
            self._det_token_num = new_n

    def _ensure_det_token_box_buffers_for(self, stats_obj, n_tokens: int):
        if not self.collect_stats or stats_obj is None:
            return
        if n_tokens <= 0:
            return
        if stats_obj.get('cx') is None:
            stats_obj['cx'] = [[] for _ in range(int(n_tokens))]
            stats_obj['cy'] = [[] for _ in range(int(n_tokens))]
            stats_obj['area'] = [[] for _ in range(int(n_tokens))]
            stats_obj['score'] = [[] for _ in range(int(n_tokens))]
            stats_obj['label'] = [[] for _ in range(int(n_tokens))]
            stats_obj['x0'] = [[] for _ in range(int(n_tokens))]
            stats_obj['y0'] = [[] for _ in range(int(n_tokens))]
            stats_obj['x1'] = [[] for _ in range(int(n_tokens))]
            stats_obj['y1'] = [[] for _ in range(int(n_tokens))]
            return

        cur = len(stats_obj['cx']) if isinstance(stats_obj.get('cx'), list) else 0
        if int(n_tokens) > cur:
            grow = int(n_tokens) - cur
            for k in ('cx', 'cy', 'area', 'score', 'label', 'x0', 'y0', 'x1', 'y1'):
                stats_obj[k].extend([[] for _ in range(grow)])

    def _collect_det_token_box_stats(self, predictions, img_ids):
        if not self.collect_stats or self.det_token_box_stats is None:
            return

        coco_imgs = self.coco_gt.loadImgs(img_ids)
        img_info = {img['id']: img for img in coco_imgs}

        for img_id, prediction in predictions.items():
            if len(prediction) == 0 or 'boxes' not in prediction:
                continue
            if img_id not in img_info:
                continue

            img_h = float(img_info[img_id]['height'])
            img_w = float(img_info[img_id]['width'])
            if img_h <= 0 or img_w <= 0:
                continue

            boxes = prediction['boxes']
            labels = prediction.get('labels', None)
            scores = prediction.get('scores', None)
            if boxes is None:
                continue
            if isinstance(boxes, np.ndarray):
                boxes_t = torch.from_numpy(boxes)
            else:
                boxes_t = boxes

            if not torch.is_tensor(boxes_t) or boxes_t.numel() == 0:
                continue
            if boxes_t.ndim != 2 or boxes_t.shape[-1] != 4:
                continue

            labels_t = None
            if labels is not None:
                if isinstance(labels, np.ndarray):
                    labels_t = torch.from_numpy(labels)
                else:
                    labels_t = labels
                if torch.is_tensor(labels_t):
                    labels_t = labels_t.detach().to('cpu')

            scores_t = None
            if scores is not None:
                if isinstance(scores, np.ndarray):
                    scores_t = torch.from_numpy(scores)
                else:
                    scores_t = scores
                if torch.is_tensor(scores_t):
                    scores_t = scores_t.detach().to('cpu', dtype=torch.float32)

            n_tokens = int(boxes_t.shape[0])
            self._ensure_det_token_box_buffers(n_tokens)
            if self._det_token_num is None or self._det_token_num <= 0:
                continue

            n_use = min(n_tokens, int(self._det_token_num))
            if torch.is_tensor(labels_t) and labels_t.numel() > 0:
                n_use = min(n_use, int(labels_t.numel()))
            if torch.is_tensor(scores_t) and scores_t.numel() > 0:
                n_use = min(n_use, int(scores_t.numel()))
            b = boxes_t[:n_use].detach().to('cpu', dtype=torch.float32)
            x0, y0, x1, y1 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
            bw = (x1 - x0).clamp(min=0.0)
            bh = (y1 - y0).clamp(min=0.0)

            cx = ((x0 + x1) * 0.5) / img_w
            cy = ((y0 + y1) * 0.5) / img_h
            area = (bw * bh) / (img_w * img_h)

            nx0 = x0 / img_w
            ny0 = y0 / img_h
            nx1 = x1 / img_w
            ny1 = y1 / img_h

            cx_list = cx.numpy().tolist()
            cy_list = cy.numpy().tolist()
            area_list = area.numpy().tolist()
            x0_list = nx0.numpy().tolist()
            y0_list = ny0.numpy().tolist()
            x1_list = nx1.numpy().tolist()
            y1_list = ny1.numpy().tolist()
            label_list = None
            if torch.is_tensor(labels_t) and labels_t.numel() >= n_use:
                try:
                    label_list = labels_t[:n_use].to(dtype=torch.int64).numpy().tolist()
                except Exception:
                    label_list = None

            score_list = None
            if torch.is_tensor(scores_t) and scores_t.numel() >= n_use:
                try:
                    score_list = scores_t[:n_use].numpy().tolist()
                except Exception:
                    score_list = None
            for t in range(n_use):
                self.det_token_box_stats['cx'][t].append(float(cx_list[t]))
                self.det_token_box_stats['cy'][t].append(float(cy_list[t]))
                self.det_token_box_stats['area'][t].append(float(area_list[t]))
                if score_list is not None:
                    self.det_token_box_stats['score'][t].append(float(score_list[t]))
                if label_list is not None:
                    self.det_token_box_stats['label'][t].append(int(label_list[t]))
                self.det_token_box_stats['x0'][t].append(float(x0_list[t]))
                self.det_token_box_stats['y0'][t].append(float(y0_list[t]))
                self.det_token_box_stats['x1'][t].append(float(x1_list[t]))
                self.det_token_box_stats['y1'][t].append(float(y1_list[t]))

    @staticmethod
    def _summarize_1d(values):
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return {
                'count': 0,
                'mean': float('nan'),
                'std': float('nan'),
                'min': float('nan'),
                'p25': float('nan'),
                'median': float('nan'),
                'p75': float('nan'),
                'max': float('nan'),
            }
        return {
            'count': int(arr.size),
            'mean': float(arr.mean()),
            'std': float(arr.std()),
            'min': float(arr.min()),
            'p25': float(np.percentile(arr, 25)),
            'median': float(np.percentile(arr, 50)),
            'p75': float(np.percentile(arr, 75)),
            'max': float(arr.max()),
        }

    def _collect_det_token_recall_matched_from_eval_imgs(self, coco_dt, coco_eval, eval_imgs):
        if not self.collect_stats:
            return
        if self.det_token_box_stats_iou50_recall_matched_useCats1 is None:
            return
        if coco_dt is None or coco_eval is None or eval_imgs is None:
            return

        p = coco_eval.params
        if int(getattr(p, 'useCats', 1)) != 1:
            return

        dt_id_to_token_idx = {}
        dt_id_to_bbox = {}
        dt_id_to_cat_id = {}
        dt_id_to_score = {}
        try:
            for ann_id, ann in getattr(coco_dt, 'anns', {}).items():
                if not isinstance(ann, dict):
                    continue
                if 'token_idx' in ann:
                    dt_id_to_token_idx[int(ann_id)] = int(ann['token_idx'])
                if 'bbox' in ann:
                    dt_id_to_bbox[int(ann_id)] = ann.get('bbox')
                if 'category_id' in ann:
                    dt_id_to_cat_id[int(ann_id)] = int(ann.get('category_id'))
                if 'score' in ann:
                    try:
                        dt_id_to_score[int(ann_id)] = float(ann.get('score'))
                    except Exception:
                        pass
        except Exception:
            return

        img_info = {}
        try:
            imgs = self.coco_gt.loadImgs(list(getattr(p, 'imgIds', [])))
            img_info = {int(im['id']): im for im in imgs if isinstance(im, dict) and 'id' in im}
        except Exception:
            img_info = {}

        target_area = "all"
        target_iou = 0.50
        area_lbls = getattr(p, 'areaRngLbl', [])
        if target_area not in area_lbls:
            return
        area_index = list(area_lbls).index(target_area)

        eval_imgs = np.asarray(eval_imgs)
        if eval_imgs.ndim != 3:
            return
        n_cats, n_areas, n_imgs = eval_imgs.shape
        if area_index < 0 or area_index >= n_areas:
            return

        for cat_k in range(n_cats):
            for img_k in range(n_imgs):
                e = eval_imgs[cat_k, area_index, img_k]
                if e is None:
                    continue

                img_id = None
                try:
                    img_id = int(e.get('image_id', getattr(p, 'imgIds', [])[img_k]))
                except Exception:
                    try:
                        img_id = int(getattr(p, 'imgIds', [])[img_k])
                    except Exception:
                        img_id = None
                if img_id is None or img_id not in img_info:
                    continue
                img_h = float(img_info[img_id].get('height', 0))
                img_w = float(img_info[img_id].get('width', 0))
                if img_h <= 0 or img_w <= 0:
                    continue

                dt_ids = e.get('dtIds', [])
                if not dt_ids:
                    continue

                dt_matches = np.asarray(e.get('dtMatches', []))
                dt_ignore = np.asarray(e.get('dtIgnore', []))
                if dt_matches.size == 0:
                    continue

                t_iou = 0
                if dt_matches.ndim == 2:
                    t_dim = int(dt_matches.shape[0])
                    try:
                        iou_axis = np.asarray(getattr(p, 'iouThrs', [])).reshape(-1)
                    except Exception:
                        iou_axis = np.asarray([], dtype=np.float64)
                    if iou_axis.size == t_dim and t_dim > 0:
                        cand = np.where(np.isclose(iou_axis, float(target_iou), atol=1e-12, rtol=0.0))[0]
                        if cand.size > 0:
                            t_iou = int(cand[0])
                        else:
                            t_iou = int(np.argmin(np.abs(iou_axis - float(target_iou))))
                    else:
                        if t_dim == 10 and abs(float(target_iou) - 0.50) < 1e-9:
                            t_iou = 0
                        else:
                            t_iou = max(0, t_dim - 1)

                dt_matches_t = dt_matches[t_iou] if dt_matches.ndim == 2 else dt_matches
                dt_ignore_t = dt_ignore[t_iou] if dt_ignore.ndim == 2 else dt_ignore


                for d_i, dt_id in enumerate(dt_ids):
                    if d_i < len(dt_ignore_t) and int(dt_ignore_t[d_i]) == 1:
                        continue
                    m = int(dt_matches_t[d_i]) if d_i < len(dt_matches_t) else 0
                    if m <= 0:
                        continue

                    dt_id = int(dt_id)
                    token_idx = dt_id_to_token_idx.get(dt_id, None)
                    bbox = dt_id_to_bbox.get(dt_id, None)
                    cat_id = dt_id_to_cat_id.get(dt_id, None)
                    score = dt_id_to_score.get(dt_id, None)
                    if token_idx is None or not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                        continue

                    x0 = float(bbox[0])
                    y0 = float(bbox[1])
                    w = float(bbox[2])
                    h = float(bbox[3])
                    x1 = x0 + w
                    y1 = y0 + h

                    nx0 = x0 / img_w
                    ny0 = y0 / img_h
                    nx1 = x1 / img_w
                    ny1 = y1 / img_h
                    cx = (nx0 + nx1) * 0.5
                    cy = (ny0 + ny1) * 0.5
                    area = max(0.0, nx1 - nx0) * max(0.0, ny1 - ny0)

                    target = self.det_token_box_stats_iou50_recall_matched_useCats1
                    self._ensure_det_token_box_buffers_for(target, int(token_idx) + 1)
                    if target.get('cx') is None:
                        continue
                    t = int(token_idx)
                    target['x0'][t].append(float(nx0))
                    target['y0'][t].append(float(ny0))
                    target['x1'][t].append(float(nx1))
                    target['y1'][t].append(float(ny1))
                    target['cx'][t].append(float(cx))
                    target['cy'][t].append(float(cy))
                    target['area'][t].append(float(area))
                    if cat_id is not None:
                        target['label'][t].append(int(cat_id))
                    if score is not None and isinstance(target.get('score'), list):
                        target['score'][t].append(float(score))

    def export_stats_to_csv(self, output_dir='./outputs'):
        if not self.collect_stats:
            self._print_stats("WARN", "Statistics collection is not enabled. Set collect_stats=True when creating CocoEvaluator.")
            return

        if not is_main_process():
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self._export_det_token_heatmaps_to_png(output_path)
        self._print_stats("DONE", f"total visualizations exported to: {output_path}")

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
    # tic = time.time()
    # print('Running per image evaluation...')
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    # print('Evaluate annotation type *{}*'.format(p.iouType))
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
    # toc = time.time()
    # print('DONE (t={:0.2f}s).'.format(toc-tic))
    return p.imgIds, evalImgs

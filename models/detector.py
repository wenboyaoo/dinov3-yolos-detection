"""
Detector model and criterion classes.
"""
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
import torch
import torch.nn.functional as F
from torch import nn
from functools import partial

from .backbones import build_backbone


from .matcher import build_matcher

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

    
class Detector(nn.Module):
    def __init__(self, args):
        super().__init__()
        # import pdb;pdb.set_trace()
        self.backbone, hidden_dim = build_backbone(**vars(args))
        self.class_embed = MLP(hidden_dim, hidden_dim, args.num_classes+1, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.det_mixer = None
        unfreeze = [] if args.unfreeze is None else args.unfreeze
        for name, p in self.backbone.named_parameters():
            p.requires_grad = False
        if 'all' in unfreeze:
            for p in self.parameters():
                p.requires_grad = True
        else:
            for name, p in self.named_parameters():
                for prefix in unfreeze:
                    if name.startswith(prefix):
                        p.requires_grad = True
                        break
    
    def forward(self, samples: NestedTensor, collect_attn_stats: bool = False):
        # import pdb;pdb.set_trace()
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        x = self.backbone(samples.tensors, collect_det_to_patch_attn=collect_attn_stats)
        outputs_class = self.class_embed(x)
        outputs_coord = self.bbox_embed(x).sigmoid()
        out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
        if collect_attn_stats and hasattr(self.backbone, 'pop_last_det_to_patch_attn_meanheads_cpu'):
            out['det_to_patch_attn'] = self.backbone.pop_last_det_to_patch_attn_meanheads_cpu()
        if collect_attn_stats and hasattr(self.backbone, 'pop_det_to_patch_attn_meanheads_cpu_layers'):
            out['det_to_patch_attn_layers'] = self.backbone.pop_det_to_patch_attn_meanheads_cpu_layers()
        return out
    
    @torch.jit.ignore
    def get_attentions(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        x = self.backbone.get_attentions(samples.tensors)
        return x

    @torch.jit.ignore
    def get_last_layer_det_to_patch_attn_meanheads(self, samples: NestedTensor):
        """Memory-friendly last-layer det→patch attention (mean heads).

        Returns a tensor of shape [B, num_det_tokens, num_patches].
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        if not hasattr(self.backbone, 'get_last_layer_det_to_patch_attn_meanheads'):
            raise AttributeError('backbone does not implement get_last_layer_det_to_patch_attn_meanheads')
        return self.backbone.get_last_layer_det_to_patch_attn_meanheads(samples.tensors)

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 enable_backup_matching: bool = False, backup_iou_thresh: float = 0.7):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.enable_backup_matching = bool(enable_backup_matching)
        self.backup_iou_thresh = float(backup_iou_thresh)
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    @torch.no_grad()
    def _augment_indices_with_backup(self, outputs, targets, indices):
        """Optionally add many-to-one "backup" positives after Hungarian.

        Strategy:
          - Only considers queries that were *unmatched* by Hungarian.
          - Iterates GTs from small area -> large area.
          - If a query's predicted class == GT class and IoU > threshold,
            we add (query, gt) as an additional supervised pair.

        Returns:
          (new_indices, num_added)
        """
        if not self.enable_backup_matching:
            return indices, 0
        if self.backup_iou_thresh is None:
            return indices, 0
        if self.backup_iou_thresh <= 0:
            return indices, 0
        if 'pred_logits' not in outputs or 'pred_boxes' not in outputs:
            return indices, 0

        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        bs, num_queries = pred_logits.shape[:2]
        device = pred_logits.device

        # Under autocast bf16/fp16, torchvision box ops can be picky about dtype.
        if pred_boxes.dtype in (torch.float16, torch.bfloat16):
            pred_boxes_f = pred_boxes.float()
        else:
            pred_boxes_f = pred_boxes

        # no-object class is the last logit
        no_object_class = pred_logits.shape[-1] - 1
        pred_labels = pred_logits.argmax(-1)  # [B, Q]
        pred_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes_f)  # [B, Q, 4]

        total_added = 0
        new_indices = []
        for b in range(bs):
            src, tgt = indices[b]

            tgt_boxes = targets[b].get('boxes', None)
            tgt_labels = targets[b].get('labels', None)
            if tgt_boxes is None or tgt_labels is None or tgt_boxes.numel() == 0:
                new_indices.append((src, tgt))
                continue

            if tgt_boxes.dtype in (torch.float16, torch.bfloat16):
                tgt_boxes_f = tgt_boxes.float()
            else:
                tgt_boxes_f = tgt_boxes

            # Build unmatched query list on device
            matched_mask = torch.zeros(num_queries, dtype=torch.bool, device=device)
            if src.numel() > 0:
                matched_mask[src.to(device)] = True
            unmatched = torch.arange(num_queries, device=device)[~matched_mask]
            if unmatched.numel() == 0:
                new_indices.append((src, tgt))
                continue

            tgt_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(tgt_boxes_f)
            iou_mat, _ = box_ops.box_iou(pred_boxes_xyxy[b, unmatched], tgt_boxes_xyxy)  # [U, T]

            pred_labels_u = pred_labels[b, unmatched]
            gt_labels = tgt_labels
            gt_area = tgt_boxes_f[:, 2] * tgt_boxes_f[:, 3]
            gt_order = torch.argsort(gt_area)  # small -> large

            assigned = torch.zeros(unmatched.numel(), dtype=torch.bool, device=device)
            add_src_list = []
            add_tgt_list = []
            for gt_idx in gt_order:
                gt_label = gt_labels[gt_idx]
                eligible = (
                    (~assigned)
                    & (pred_labels_u != no_object_class)
                    & (pred_labels_u == gt_label)
                    & (iou_mat[:, gt_idx] > self.backup_iou_thresh)
                )
                if eligible.any():
                    sel_q = unmatched[eligible]
                    add_src_list.append(sel_q)
                    add_tgt_list.append(
                        torch.full((sel_q.numel(),), int(gt_idx.item()), dtype=torch.long, device=device)
                    )
                    assigned[eligible] = True
                    if assigned.all():
                        break

            if len(add_src_list) == 0:
                new_indices.append((src, tgt))
                continue

            add_src = torch.cat(add_src_list, dim=0)
            add_tgt = torch.cat(add_tgt_list, dim=0)
            total_added += int(add_src.numel())

            # Keep output indices on same device as original matcher indices (typically CPU)
            idx_device = src.device
            add_src = add_src.to(idx_device)
            add_tgt = add_tgt.to(idx_device)
            new_indices.append((torch.cat([src, add_src]), torch.cat([tgt, add_tgt])))

        return new_indices, total_added

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        # NOTE: under bf16 autocast, logits may be bf16; cross_entropy requires weight dtype to match.
        empty_weight = self.empty_weight.to(dtype=src_logits.dtype, device=src_logits.device)
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Optional second-pass backup matches (many-to-one positives)
        indices, num_backup = self._augment_indices_with_backup(outputs_without_aux, targets, indices)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        if self.enable_backup_matching and num_backup > 0:
            num_backup_t = torch.as_tensor([num_backup], dtype=torch.float, device=num_boxes.device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_backup_t)
            num_boxes = torch.clamp((num_boxes + num_backup_t) / get_world_size(), min=1).item()
        else:
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                indices, _ = self._augment_indices_with_backup(aux_outputs, targets, indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


def build(args):
    device = torch.device(args.device)
    # import pdb;pdb.set_trace()
    model = Detector(args)
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.ce_loss_coef, 'loss_bbox': args.bbox_loss_coef, 'loss_giou':args.giou_loss_coef}
    # TODO this is a hack
    # if args.aux_loss:
    #     aux_weight_dict = {}
    #     for i in range(args.dec_layers - 1):
    #         aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
    #     weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    criterion = SetCriterion(
        args.num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        enable_backup_matching=getattr(args, 'enable_backup_matching', False),
        backup_iou_thresh=getattr(args, 'backup_iou_thresh', 0.7),
    )
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors

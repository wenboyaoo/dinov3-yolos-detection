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
    
    def forward(self, samples: NestedTensor):
        # import pdb;pdb.set_trace()
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        backbone_out = self.backbone(samples.tensors)
        if isinstance(backbone_out, dict):
            x = backbone_out["det_tokens"]
            aux_det_tokens = backbone_out.get("aux_det_tokens", None)
            aux_rope_yx = backbone_out.get("aux_rope_yx", None)
        else:
            x = backbone_out
            aux_det_tokens = None
            aux_rope_yx = None

        outputs_class = self.class_embed(x)
        outputs_coord = self.bbox_embed(x).sigmoid()
        out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}

        if self.training and aux_det_tokens is not None and len(aux_det_tokens) > 0:
            aux_outputs = []
            for tok in aux_det_tokens:
                aux_logits = self._mlp_forward_detached_params(self.class_embed, tok)
                aux_boxes = self._mlp_forward_detached_params(self.bbox_embed, tok).sigmoid()
                aux_outputs.append({'pred_logits': aux_logits, 'pred_boxes': aux_boxes})
            out['aux_outputs'] = aux_outputs
            out['aux_rope_yx'] = aux_rope_yx
        return out

    @staticmethod
    def _mlp_forward_detached_params(mlp: MLP, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(mlp.layers):
            weight = layer.weight.detach()
            bias = layer.bias.detach() if layer.bias is not None else None
            x = F.linear(x, weight, bias)
            if i < mlp.num_layers - 1:
                x = F.relu(x)
        return x
    
    @torch.jit.ignore
    def get_attentions(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        x = self.backbone.get_attentions(samples.tensors)
        return x

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        eos_coef,
        losses,
    ):
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
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True, weights=None, bg_weights=None):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        
        if self.matcher.backup_matching:
            # Backup matching logic (VRAM-friendly):
            # - avoid building a [B, C, Ndt, Ngt] tensor for cross_entropy
            # - use log_softmax + gather
            bs, n_dt, _ = src_logits.shape

            # Concatenate all target labels from the batch
            tgt_labels = torch.cat([t["labels"] for t in targets])
            n_gt_total = int(tgt_labels.numel())

            # Ensure dtype/device
            tgt_labels = tgt_labels.to(device=src_logits.device, dtype=torch.long)

            # log_probs: [B, Ndt, C]
            log_probs = F.log_softmax(src_logits, dim=-1)

            # --- Negative (no-object) loss ---
            # nll_bg: [B, Ndt]
            nll_bg = -log_probs[..., self.num_classes]
            weighted_neg_loss = (nll_bg * (self.eos_coef * bg_weights)).sum()

            if n_gt_total == 0:
                # Only background when no GT exists.
                loss_ce = weighted_neg_loss / num_boxes
                losses = {'loss_ce': loss_ce}
                if log:
                    losses['class_error'] = torch.tensor(0.0, device=src_logits.device)
                return losses

            # --- Positive loss ---
            # Build index tensor [B, Ndt, Ngt_total] and gather log-probs at GT classes.
            tgt_labels_exp = tgt_labels.view(1, 1, -1).expand(bs, n_dt, -1)
            # gathered_logp: [B, Ndt, Ngt_total]
            gathered_logp = torch.gather(log_probs, dim=2, index=tgt_labels_exp)
            pos_nll = -gathered_logp
            weighted_pos_loss = (pos_nll * weights).sum()

            loss_ce = (weighted_pos_loss + weighted_neg_loss) / num_boxes
            losses = {'loss_ce': loss_ce}

            # Log class error based on Hungarian matching
            if log:
                idx = self._get_src_permutation_idx(indices)
                if len(idx) > 0 and len(idx[0]) > 0:
                    target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
                    losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
                else:
                    losses['class_error'] = torch.tensor(0.0, device=src_logits.device)

        else:
            # Original logic
            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
            target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                        dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o

            empty_weight = self.empty_weight.to(dtype=src_logits.dtype, device=src_logits.device)
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, empty_weight)
            losses = {'loss_ce': loss_ce}

            if log:
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

    def loss_boxes(self, outputs, targets, indices, num_boxes, weights=None, bg_weights=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        
        if self.matcher.backup_matching:
            # Backup matching logic
            src_boxes = outputs['pred_boxes'] # [bs, n_dt, 4]
            tgt_boxes = torch.cat([t['boxes'] for t in targets], dim=0) # [n_gt_total, 4]
            
            bs, n_dt, _ = src_boxes.shape
            n_gt_total = tgt_boxes.shape[0]

            if n_gt_total == 0:
                # No GTs, no box loss
                return {'loss_bbox': torch.tensor(0.0, device=src_boxes.device), 
                        'loss_giou': torch.tensor(0.0, device=src_boxes.device)}

            # Expand src and tgt boxes to calculate all-pairs losses
            # src_boxes_exp: [bs, n_dt, n_gt_total, 4]
            # tgt_boxes_exp: [bs, n_dt, n_gt_total, 4]
            src_boxes_exp = src_boxes.unsqueeze(2).expand(-1, -1, n_gt_total, -1)
            tgt_boxes_exp = tgt_boxes.unsqueeze(0).unsqueeze(0).expand(bs, n_dt, -1, -1)

            # L1 loss for all pairs
            loss_bbox_all = F.l1_loss(src_boxes_exp, tgt_boxes_exp, reduction='none')
            # A box should only contribute to loss if it's a foreground object.
            # The total foreground probability for a DT_j is (1 - bg_weight_j).
            # The formula is Sum_j( (1-W_bg,j) * Sum_i( W_ij' * L(j,i) ) ), where W_ij' is the normalized weight.
            # A simpler, equivalent formulation is just to sum the weighted losses, as W_ij is already 0 for background boxes.
            weighted_loss_bbox = (loss_bbox_all * weights.unsqueeze(-1)).sum()

            # GIoU loss for all pairs (VRAM-friendly): compute aligned GIoU per element,
            # NOT an NxN matrix from generalized_box_iou.
            src_xyxy = box_ops.box_cxcywh_to_xyxy(src_boxes_exp)
            tgt_xyxy = box_ops.box_cxcywh_to_xyxy(tgt_boxes_exp)

            # Compute generalized IoU for aligned pairs (elementwise)
            # Shapes: [..., 4] -> [...]
            # Use float32 for stability
            src_xyxy_f = src_xyxy.float()
            tgt_xyxy_f = tgt_xyxy.float()

            x1 = torch.maximum(src_xyxy_f[..., 0], tgt_xyxy_f[..., 0])
            y1 = torch.maximum(src_xyxy_f[..., 1], tgt_xyxy_f[..., 1])
            x2 = torch.minimum(src_xyxy_f[..., 2], tgt_xyxy_f[..., 2])
            y2 = torch.minimum(src_xyxy_f[..., 3], tgt_xyxy_f[..., 3])
            inter_w = (x2 - x1).clamp(min=0)
            inter_h = (y2 - y1).clamp(min=0)
            inter = inter_w * inter_h

            area1 = (src_xyxy_f[..., 2] - src_xyxy_f[..., 0]).clamp(min=0) * (src_xyxy_f[..., 3] - src_xyxy_f[..., 1]).clamp(min=0)
            area2 = (tgt_xyxy_f[..., 2] - tgt_xyxy_f[..., 0]).clamp(min=0) * (tgt_xyxy_f[..., 3] - tgt_xyxy_f[..., 1]).clamp(min=0)
            union = (area1 + area2 - inter).clamp(min=1e-10)
            iou = inter / union

            cx1 = torch.minimum(src_xyxy_f[..., 0], tgt_xyxy_f[..., 0])
            cy1 = torch.minimum(src_xyxy_f[..., 1], tgt_xyxy_f[..., 1])
            cx2 = torch.maximum(src_xyxy_f[..., 2], tgt_xyxy_f[..., 2])
            cy2 = torch.maximum(src_xyxy_f[..., 3], tgt_xyxy_f[..., 3])
            c_w = (cx2 - cx1).clamp(min=0)
            c_h = (cy2 - cy1).clamp(min=0)
            area_c = (c_w * c_h).clamp(min=1e-10)

            giou = iou - (area_c - union) / area_c
            loss_giou_all = (1.0 - giou).to(dtype=src_boxes.dtype)
            weighted_loss_giou = (loss_giou_all * weights).sum()

            losses = {}
            losses['loss_bbox'] = weighted_loss_bbox / num_boxes
            losses['loss_giou'] = weighted_loss_giou / num_boxes

        else:
            # Original logic
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

    def _loss_labels_hard(self, outputs, targets, indices, num_boxes, log=False):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        empty_weight = self.empty_weight.to(dtype=src_logits.dtype, device=src_logits.device)
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, empty_weight)
        losses = {'loss_ce': loss_ce}
        if log:
            if idx[0].numel() > 0:
                losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
            else:
                losses['class_error'] = torch.tensor(0.0, device=src_logits.device)
        return losses

    def _loss_boxes_hard(self, outputs, targets, indices, num_boxes):
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

    def _loss_rope_center(self, aux_rope_xy, targets, indices, num_boxes):
        if aux_rope_xy is None:
            device = targets[0]['boxes'].device if len(targets) > 0 and 'boxes' in targets[0] else self.empty_weight.device
            return {'loss_rope_center': torch.tensor(0.0, device=device)}

        batch_idx, src_idx = self._get_src_permutation_idx(indices)
        if batch_idx.numel() == 0:
            device = aux_rope_xy[0].device if isinstance(aux_rope_xy, list) and len(aux_rope_xy) > 0 else self.empty_weight.device
            return {'loss_rope_center': torch.tensor(0.0, device=device)}

        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        # targets boxes are (cx, cy, w, h) in [0,1]; rope coords are (y, x) in [-1,1]
        target_yx = target_boxes[:, [1, 0]] * 2.0 - 1.0

        total = torch.tensor(0.0, device=target_boxes.device)
        count = 0
        for rope_xy in aux_rope_xy:
            pred_yx = rope_xy[batch_idx, src_idx, :]
            total = total + F.l1_loss(pred_yx, target_yx, reduction='sum') / num_boxes
            count += 1

        if count == 0:
            return {'loss_rope_center': torch.tensor(0.0, device=target_boxes.device)}

        return {'loss_rope_center': total / float(count)}

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k not in ('aux_outputs', 'aux_rope_yx')}

        # Retrieve the matching between the outputs of the last layer and the targets
        match_results = self.matcher(outputs_without_aux, targets)
        
        if self.matcher.backup_matching:
            indices_h = match_results['indices']
            weights = match_results['weights']
            bg_weights = match_results['bg_weights']
        else:
            indices_h = match_results
            weights = None
            bg_weights = None

        # Normalization for matcher-only logging
        num_boxes_log = sum(len(t["labels"]) for t in targets)
        num_boxes_log = torch.as_tensor([num_boxes_log], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes_log)
        num_boxes_log = torch.clamp(num_boxes_log / get_world_size(), min=1).item()

        losses = {}

        # Log/print: only Hungarian-matched tokens
        if self.matcher.backup_matching:
            losses.update(self.loss_labels(outputs, targets, indices_h, num_boxes_log, log=True, weights=weights, bg_weights=bg_weights))
            losses.update(self.loss_boxes(outputs, targets, indices_h, num_boxes_log, weights=weights, bg_weights=bg_weights))
        else:
            losses.update(self.loss_labels(outputs, targets, indices_h, num_boxes_log, log=True))
            losses.update(self.loss_boxes(outputs, targets, indices_h, num_boxes_log))
        
        losses.update(self.loss_cardinality(outputs, targets, indices_h, num_boxes_log))

        # Training-time auxiliary losses: reuse last-layer Hungarian indices.
        if 'aux_outputs' in outputs and outputs['aux_outputs'] is not None:
            aux_loss_ce = 0.0
            aux_loss_bbox = 0.0
            aux_loss_giou = 0.0
            n_aux = 0

            for aux_out in outputs['aux_outputs']:
                aux_loss_ce = aux_loss_ce + self._loss_labels_hard(aux_out, targets, indices_h, num_boxes_log, log=False)['loss_ce']
                box_losses = self._loss_boxes_hard(aux_out, targets, indices_h, num_boxes_log)
                aux_loss_bbox = aux_loss_bbox + box_losses['loss_bbox']
                aux_loss_giou = aux_loss_giou + box_losses['loss_giou']
                n_aux += 1

            if n_aux > 0:
                losses['loss_aux_ce'] = aux_loss_ce / float(n_aux)
                losses['loss_aux_bbox'] = aux_loss_bbox / float(n_aux)
                losses['loss_aux_giou'] = aux_loss_giou / float(n_aux)

        # Rope center alignment loss (matched queries only).
        if 'aux_rope_yx' in outputs:
            aux_rope_yx = outputs.get('aux_rope_yx', None)
            losses.update(self._loss_rope_center(aux_rope_yx, targets, indices_h, num_boxes_log))

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
    weight_dict = {
        'loss_ce': args.ce_loss_coef,
        'loss_bbox': args.bbox_loss_coef,
        'loss_giou': args.giou_loss_coef,
    }

    # Aux det losses reuse last-layer Hungarian indices (training-time only).
    if getattr(args, 'aux_loss_coef', 0.0) and args.aux_loss_coef != 0.0:
        weight_dict.update({
            'loss_aux_ce': args.aux_loss_coef * args.ce_loss_coef,
            'loss_aux_bbox': args.aux_loss_coef * args.bbox_loss_coef,
            'loss_aux_giou': args.aux_loss_coef * args.giou_loss_coef,
        })

    if getattr(args, 'rope_center_loss_coef', 0.0) and args.rope_center_loss_coef != 0.0:
        weight_dict['loss_rope_center'] = args.rope_center_loss_coef
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
    )
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors

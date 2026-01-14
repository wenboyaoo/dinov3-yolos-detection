# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1,
                 backup_matching: bool = False, backup_k_scale: float = 0.2, backup_top_k: int = 5):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.backup_matching = backup_matching
        self.backup_k_scale = backup_k_scale
        self.backup_top_k = backup_top_k
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        if out_bbox.dtype in (torch.float16, torch.bfloat16) or tgt_bbox.dtype in (torch.float16, torch.bfloat16):
            out_bbox_f = out_bbox.float()
            tgt_bbox_f = tgt_bbox.float()
        else:
            out_bbox_f = out_bbox
            tgt_bbox_f = tgt_bbox
        cost_bbox = torch.cdist(out_bbox_f, tgt_bbox_f, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox_f), box_cxcywh_to_xyxy(tgt_bbox_f))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1)

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.cpu().split(sizes, -1))]
        
        if not self.backup_matching:
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        # --- Backup Matching Logic ---
        # For now, this is a placeholder. The full logic will be implemented in the next steps.
        # The goal is to compute weights for non-matched boxes.
        
        num_gts = sum(sizes)
        if num_gts == 0:
            # No ground truth, all predictions are background
            batch_weights = torch.zeros(bs, num_queries, 0, device=C.device)
            batch_bg_weights = torch.ones(bs, num_queries, device=C.device)
            return {
                'indices': [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices],
                'weights': batch_weights,
                'bg_weights': batch_bg_weights
            }

        # Initialize weights
        batch_weights = torch.zeros(bs, num_queries, C.shape[-1], device=C.device)
        batch_bg_weights = torch.ones(bs, num_queries, device=C.device)

        # Process each image in the batch
        for i, (dt_indices, gt_indices) in enumerate(indices):
            if gt_indices.size == 0:
                continue # No GTs for this image, all DTs are background

            num_dt = C[i].shape[0]
            num_gt = C[i].shape[1]

            # 1. Anchoring: Get min cost for each GT from Hungarian Matcher
            c_min = C[i][dt_indices, gt_indices]
            
            # Add a small epsilon to prevent division by zero
            c_min = c_min.clamp(min=1e-6)

            # 2. Dynamic Coefficient & Candidate Selection
            # Pre-calculate C_logit
            epsilon = max(torch.finfo(C.dtype).eps * 10, 1e-6)
            c_logit = torch.log(1 / torch.tensor(epsilon, device=C.device) - 1)
            
            # Calculate scaling factor k_i for each GT
            k = c_logit / (self.backup_k_scale * c_min)

            # Create a mask for DTs already matched by Hungarian Matcher
            matched_dt_mask = torch.zeros(num_dt, dtype=torch.bool, device=C.device)
            matched_dt_mask[dt_indices] = True

            # Select candidate pool S_i for each GT
            # Cost matrix for unmatched DTs
            unmatched_costs = C[i][~matched_dt_mask]
            
            if unmatched_costs.shape[0] == 0: # All DTs were matched
                continue

            # Find top-K smallest costs for each GT among unmatched DTs
            top_k = min(self.backup_top_k, unmatched_costs.shape[0])
            _, top_k_indices_in_unmatched = torch.topk(unmatched_costs, top_k, dim=0, largest=False)
            
            # Convert indices from unmatched DTs space to original DT space
            unmatched_dt_indices = torch.arange(num_dt, device=C.device)[~matched_dt_mask]
            candidate_indices = unmatched_dt_indices[top_k_indices_in_unmatched] # Shape: [top_k, num_gt]

            # 3. Independent Scoring
            # Gather costs for candidate pairs (i, j) where j is in S_i
            candidate_costs = C[i][candidate_indices, torch.arange(num_gt, device=C.device).unsqueeze(0)]
            
            # Calculate S_ij
            # Shape of k, c_min, candidate_costs: [num_gt], [num_gt], [top_k, num_gt]
            # We need to align them for broadcasting
            s_ij = torch.sigmoid(-k * ((1 + self.backup_k_scale) * c_min - candidate_costs))

            # 4. Weight Matrix Construction
            # Create a sparse representation of S_ij for easier aggregation
            # Shape: [num_dt, num_gt]
            s_matrix = torch.zeros(num_dt, num_gt, device=C.device)
            s_matrix.scatter_(0, candidate_indices, s_ij)

            # Calculate total background weight W_bg_j
            # Clamp to prevent log(0)
            log_one_minus_s = torch.log(torch.clamp(1 - s_matrix, min=1e-10))
            w_bg_j = torch.exp(torch.sum(log_one_minus_s, dim=1)) # Shape: [num_dt]

            # Calculate positive sample weights W_ij
            s_sum_per_dt = s_matrix.sum(dim=1, keepdim=True) # Shape: [num_dt, 1]
            s_sum_per_dt = s_sum_per_dt.clamp(min=1e-10) # Avoid division by zero
            
            w_ij = (1 - w_bg_j.unsqueeze(1)) * (s_matrix / s_sum_per_dt) # Shape: [num_dt, num_gt]

            # Set weights for Hungarian matched pairs to 1
            w_ij[dt_indices, gt_indices] = 1.0
            w_bg_j[dt_indices] = 0.0
            
            # Store batch results
            start_idx = sum(sizes[:i])
            end_idx = start_idx + sizes[i]
            batch_weights[i, :, start_idx:end_idx] = w_ij
            batch_bg_weights[i, :] = w_bg_j

        return {
            'indices': [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices],
            'weights': batch_weights,
            'bg_weights': batch_bg_weights
        }


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou,
                            backup_matching=args.backup_matching, backup_k_scale=args.backup_k_scale, backup_top_k=args.backup_top_k)

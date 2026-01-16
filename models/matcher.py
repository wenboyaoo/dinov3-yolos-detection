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
                 backup_matching: bool = False, backup_k_scale: float = 0.2, backup_top_k: int = 5,
                 backup_class_ids=None,
                 debug_cost_ratio: bool = False):
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
        # >>> DEBUG COST RATIO LOGGING (TEMP; safe to delete) >>>
        # When enabled, we compute per-class statistics about how far the (unmatched) top-k costs are
        # from the Hungarian anchor cost_min, and cache the result for the training loop to print.
        self.debug_cost_ratio = bool(debug_cost_ratio)
        self._last_debug_cost_ratio_stats = None
        # <<< DEBUG COST RATIO LOGGING (TEMP; safe to delete) <<<
        if backup_class_ids is None:
            self.backup_class_ids = None
        else:
            # store as a set of ints for cheap membership checks
            self.backup_class_ids = set(int(x) for x in backup_class_ids)
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
        cost_class = 1 - out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        if out_bbox.dtype in (torch.float16, torch.bfloat16) or tgt_bbox.dtype in (torch.float16, torch.bfloat16):
            out_bbox_f = out_bbox.float()
            tgt_bbox_f = tgt_bbox.float()
        else:
            out_bbox_f = out_bbox
            tgt_bbox_f = tgt_bbox
        cost_bbox = torch.cdist(out_bbox_f, tgt_bbox_f, p=1)

        # Compute the giou cost betwen boxes
        # generalized_box_iou is in [-1, 1], so (1 - giou) is in [0, 2]
        cost_giou = 1 - generalized_box_iou(box_cxcywh_to_xyxy(out_bbox_f), box_cxcywh_to_xyxy(tgt_bbox_f))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou

        neg_thresh = -1e-12
        neg_mask = C < neg_thresh
        if torch.any(neg_mask):
            neg_count = int(neg_mask.sum().item())
            c_min = float(C.min().item())
            c_max = float(C.max().item())
            print(
                f"[HungarianMatcher] Negative cost detected in C: count={neg_count}, min={c_min}, max={c_max}, "
                f"shape={tuple(C.shape)}, bs={bs}, num_queries={num_queries}, num_targets={int(C.shape[1])}"
            )

            # Print a few most-negative entries with their component breakdown.
            flat = C.flatten()
            k_show = min(5, flat.numel())
            vals, idxs = torch.topk(flat, k_show, largest=False)
            num_cols = C.shape[1]
            for v, idx in zip(vals, idxs):
                if float(v.item()) >= neg_thresh:
                    break
                r = int((idx // num_cols).item())
                c = int((idx % num_cols).item())
                cc = float(cost_class[r, c].item())
                cb = float(cost_bbox[r, c].item())
                cg = float(cost_giou[r, c].item())
                print(
                    f"[HungarianMatcher] C[{r},{c}]={float(v.item())} "
                    f"(cost_class={cc}, cost_bbox={cb}, cost_giou={cg})"
                )
            raise RuntimeError("Negative matching cost detected in HungarianMatcher; aborting as requested.")
        C = C.view(bs, num_queries, -1)

        sizes = [len(v["boxes"]) for v in targets]
        # Split C per-image. C_split_cpu is a tuple of tensors, one for each image in the batch.
        C_split_cpu = C.cpu().split(sizes, -1)
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C_split_cpu)]

        # >>> DEBUG COST RATIO LOGGING (TEMP; safe to delete) >>>
        # For each GT (grouped by class), compute ratio:
        #   ratio_k = (max cost among the top-k smallest unmatched DT costs) / cost_min
        # Here, "unmatched" means DTs not selected by Hungarian matching.
        # We compute k in {1,5,10} and average ratios per class.
        if self.debug_cost_ratio:
            ks = (1, 5, 10)
            sums = {k: {} for k in ks}   # sums[k][cls] = float
            counts = {k: {} for k in ks} # counts[k][cls] = int

            for img_i, (assignment, c_cpu) in enumerate(zip(indices, C_split_cpu)):
                num_gt_i = sizes[img_i]
                if num_gt_i == 0:
                    continue

                # c_img: [num_queries, num_gt_i]
                c_img = c_cpu[img_i]

                dt_idx_np, gt_idx_np = assignment
                if len(gt_idx_np) == 0:
                    continue

                dt_idx = torch.as_tensor(dt_idx_np, dtype=torch.long)
                gt_idx = torch.as_tensor(gt_idx_np, dtype=torch.long)

                num_dt = int(c_img.shape[0])
                matched_dt_mask = torch.zeros(num_dt, dtype=torch.bool)
                matched_dt_mask[dt_idx] = True
                num_unmatched = int((~matched_dt_mask).sum().item())
                if num_unmatched == 0:
                    # No unmatched DTs -> cannot compute top-k over unmatched.
                    continue

                # cost_min per matched GT (indexed by gt column id)
                # cost_min_vec: [num_gt_i]
                cost_min_vec = torch.full((num_gt_i,), float('nan'), dtype=c_img.dtype)
                cost_min_vec[gt_idx] = c_img[dt_idx, gt_idx].clamp(min=1e-6)

                # Gather unmatched costs: [num_unmatched, num_gt_i]
                unmatched_costs = c_img[~matched_dt_mask]

                # Target labels for this image: [num_gt_i]
                labels_i = targets[img_i]["labels"].detach().to(dtype=torch.long, device='cpu')

                for k in ks:
                    k_eff = min(int(k), num_unmatched)
                    # kth smallest (1-indexed) among unmatched DTs for each GT column
                    kth = unmatched_costs.kthvalue(k_eff, dim=0).values

                    # Only consider GTs that are actually matched (have a defined cost_min)
                    ratio = (kth[gt_idx] / cost_min_vec[gt_idx]).detach().cpu().to(torch.float32)
                    cls_ids = labels_i[gt_idx].cpu()

                    for r, cls_id in zip(ratio.tolist(), cls_ids.tolist()):
                        if not (r == r):
                            continue
                        prev = sums[k].get(cls_id, 0.0)
                        sums[k][cls_id] = prev + float(r)
                        counts[k][cls_id] = counts[k].get(cls_id, 0) + 1

            per_class = {}
            for k in ks:
                for cls_id, s in sums[k].items():
                    n = counts[k].get(cls_id, 0)
                    if n <= 0:
                        continue
                    per_class.setdefault(cls_id, {})[k] = s / n
                    per_class.setdefault(cls_id, {})["n"] = n

            self._last_debug_cost_ratio_stats = {
                "ks": ks,
                "per_class": per_class,
            }
        # <<< DEBUG COST RATIO LOGGING (TEMP; safe to delete) <<<
        
        if not self.backup_matching:
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        # --- Backup Matching Logic ---
        num_gts_total = sum(sizes)
        if num_gts_total == 0:
            # No ground truth, all predictions are background
            batch_weights = torch.zeros(bs, num_queries, 0, device=C.device)
            batch_bg_weights = torch.ones(bs, num_queries, device=C.device)
            return {
                'indices': [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices],
                'weights': batch_weights,
                'bg_weights': batch_bg_weights
            }

        # Initialize weights
        batch_weights = torch.zeros(bs, num_queries, num_gts_total, device=C.device)
        batch_bg_weights = torch.ones(bs, num_queries, device=C.device)

        # Pre-calculate C_logit
        epsilon = max(torch.finfo(C.dtype).eps * 10, 1e-6)
        c_logit = torch.log(1 / torch.tensor(epsilon, device=C.device) - 1)

        # Process each image in the batch
        # We iterate over the per-image cost matrices from C.split()
        C_split_device = C.split(sizes, -1)
        for i, ((dt_indices, gt_indices), c_per_image) in enumerate(zip(indices, C_split_device)):
            # c_per_image has shape [bs, num_queries, num_gts_in_image], we take our current image
            c_per_image = c_per_image[i] # Shape becomes [num_queries, num_gts_in_image]

            if gt_indices.size == 0:
                # No GTs for this image, all DTs are background. Weights are already correct (0 for pos, 1 for bg)
                continue 

            num_dt = c_per_image.shape[0]
            num_gt = c_per_image.shape[1]

            # Restrict backup matching to specific GT classes if configured.
            # Hungarian hard matches are still applied regardless.
            allowed_gt_mask = None
            if self.backup_class_ids is not None and len(self.backup_class_ids) > 0:
                tgt_labels_i = targets[i]["labels"].to(device=c_per_image.device, dtype=torch.long)
                if tgt_labels_i.numel() == num_gt:
                    allow_ids = torch.as_tensor(sorted(self.backup_class_ids), device=c_per_image.device, dtype=torch.long)
                    # torch.isin may not exist on very old torch; fall back if needed.
                    if hasattr(torch, "isin"):
                        allowed_gt_mask = torch.isin(tgt_labels_i, allow_ids)
                    else:
                        allowed_gt_mask = (tgt_labels_i.view(-1, 1) == allow_ids.view(1, -1)).any(dim=1)

            # 1. Anchoring: Get min cost for each GT from Hungarian Matcher
            c_min = c_per_image[dt_indices, gt_indices]
            
            # Add a small epsilon to prevent division by zero
            c_min = c_min.clamp(min=1e-6)

            # 2. Dynamic Coefficient & Candidate Selection
            # Calculate scaling factor k_i for each GT
            k = c_logit / (self.backup_k_scale * c_min)

            # Create a mask for DTs already matched by Hungarian Matcher
            matched_dt_mask = torch.zeros(num_dt, dtype=torch.bool, device=c_per_image.device)
            matched_dt_mask[dt_indices] = True

            # Select candidate pool S_i for each GT
            # Cost matrix for unmatched DTs
            unmatched_costs = c_per_image[~matched_dt_mask]
            
            if unmatched_costs.shape[0] == 0: # All DTs were matched
                w_ij = torch.zeros(num_dt, num_gt, device=c_per_image.device)
                w_bg_j = torch.ones(num_dt, device=c_per_image.device)
                w_ij[dt_indices, gt_indices] = 1.0
                w_bg_j[dt_indices] = 0.0
                
                start_idx = sum(sizes[:i])
                end_idx = start_idx + sizes[i]
                batch_weights[i, :, start_idx:end_idx] = w_ij
                batch_bg_weights[i, :] = w_bg_j
                continue

            # Find top-K smallest costs for each GT among unmatched DTs
            top_k = min(self.backup_top_k, unmatched_costs.shape[0])
            _, top_k_indices_in_unmatched = torch.topk(unmatched_costs, top_k, dim=0, largest=False)
            
            # Convert indices from unmatched DTs space to original DT space
            unmatched_dt_indices = torch.arange(num_dt, device=c_per_image.device)[~matched_dt_mask]
            candidate_indices = unmatched_dt_indices[top_k_indices_in_unmatched] # Shape: [top_k, num_gt]

            # 3. Independent Scoring
            # Gather costs for candidate pairs (i, j) where j is in S_i
            candidate_costs = c_per_image[candidate_indices, torch.arange(num_gt, device=c_per_image.device).unsqueeze(0)]
            
            # Calculate S_ij
            # Shapes are now compatible for broadcasting:
            # k: [num_gt], c_min: [num_gt], candidate_costs: [top_k, num_gt]
            # Spec: S_ij = 1 / (1 + exp(-k_i * ( (1+K_scale)*c_min - c_ij )))
            # i.e. sigmoid( k_i * ( (1+K_scale)*c_min - c_ij ) ).
            s_ij = torch.sigmoid(k.unsqueeze(0) * (((1 + self.backup_k_scale) * c_min).unsqueeze(0) - candidate_costs))

            # 4. Weight Matrix Construction
            # Create a sparse representation of S_ij for easier aggregation
            # Shape: [num_dt, num_gt]
            s_matrix = torch.zeros(num_dt, num_gt, device=c_per_image.device)
            s_matrix.scatter_(0, candidate_indices, s_ij)

            # Disable backup assignment for non-allowed GT classes
            if allowed_gt_mask is not None:
                s_matrix[:, ~allowed_gt_mask] = 0

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
                            backup_matching=args.backup_matching, backup_k_scale=args.backup_k_scale, backup_top_k=args.backup_top_k,
                            backup_class_ids=getattr(args, 'backup_class_ids', []),
                            # >>> DEBUG COST RATIO LOGGING (TEMP; safe to delete) >>>
                            debug_cost_ratio=getattr(args, 'debug_cost_ratio', False))
                            # <<< DEBUG COST RATIO LOGGING (TEMP; safe to delete) <<<

from collections import defaultdict
from typing import Dict, List, Tuple


def build_image_weights(
    dataset,
    alpha: float = 0.5,
    cap: float = 10.0,
    agg: str = "max",
    use_category_name: bool = True,
) -> Tuple[List[float], Dict[int, int], Dict[int, float]]:
    
    coco = dataset.coco
    img_ids = list(dataset.ids)
    num_images = len(img_ids)

    cats = coco.loadCats(coco.getCatIds())
    cat_id_to_name = {c["id"]: c.get("name", str(c["id"])) for c in cats}

    img_to_catset: Dict[int, set] = defaultdict(set)

    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            if ann.get("iscrowd", 0) == 1:
                continue
            img_to_catset[img_id].add(int(ann["category_id"]))

    cat_image_count: Dict[int, int] = {cid: 0 for cid in cat_id_to_name.keys()}
    for img_id, catset in img_to_catset.items():
        for cid in catset:
            if cid in cat_image_count:
                cat_image_count[cid] += 1

    cat_image_freq: Dict[int, float] = {
        cid: (cnt / max(num_images, 1)) for cid, cnt in cat_image_count.items()
    }

    eps = 1e-12
    cat_w: Dict[int, float] = {}
    for cid, freq in cat_image_freq.items():
        cat_w[cid] = 1.0 / (max(freq, eps) ** alpha)

    if agg not in ("max", "mean"):
        raise ValueError("agg must be 'max' or 'mean'")

    weights: List[float] = []
    for img_id in img_ids:
        cat_ids = list(img_to_catset.get(img_id, []))
        if len(cat_ids) == 0:
            w_i = 1.0
        else:
            ws = [cat_w[cid] for cid in cat_ids if cid in cat_w]
            if len(ws) == 0:
                w_i = 1.0
            else:
                w_i = max(ws) if agg == "max" else (sum(ws) / len(ws))
        weights.append(float(min(w_i, cap)))

    items = []
    for cid, cnt in cat_image_count.items():
        name = cat_id_to_name.get(cid, str(cid)) if use_category_name else str(cid)
        items.append((cnt, cat_image_freq[cid], cid, name))
    items.sort(key=lambda x: x[0])

    return weights, cat_image_count, cat_image_freq

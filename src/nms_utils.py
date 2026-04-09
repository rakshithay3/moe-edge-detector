"""Non-Maximum Suppression utilities for post-processing detections."""

import torch


def nms(boxes, scores, iou_threshold=0.5):
    """Apply Non-Maximum Suppression to filter overlapping detections.

    Args:
        boxes:  Tensor [N, 4] in (x1, y1, x2, y2) format
        scores: Tensor [N] confidence scores
        iou_threshold: IoU threshold for suppression

    Returns:
        keep: Tensor of indices to keep
    """
    if boxes.numel() == 0:
        return torch.tensor([], dtype=torch.long)

    # Use torchvision NMS when available (faster, C++ backend)
    try:
        from torchvision.ops import nms as tv_nms
        return tv_nms(boxes.float(), scores.float(), iou_threshold)
    except ImportError:
        pass

    # Fallback: pure-Python NMS
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        # Compute IoU of the picked box with the rest
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        inter = torch.clamp(xx2 - xx1, min=0) * torch.clamp(yy2 - yy1, min=0)
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / (union + 1e-6)

        # Keep boxes with IoU below the threshold
        mask = iou <= iou_threshold
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.long)


def filter_detections(boxes, scores, labels, score_threshold=0.3, iou_threshold=0.5):
    """Full post-processing: score filter + NMS.

    Args:
        boxes:  [N, 4] in (x1, y1, x2, y2)
        scores: [N] confidence scores
        labels: [N] class labels
        score_threshold: Minimum confidence to keep
        iou_threshold: IoU threshold for NMS

    Returns:
        Filtered (boxes, scores, labels)
    """
    # Score filtering
    mask = scores >= score_threshold
    boxes  = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]

    if boxes.numel() == 0:
        return boxes, scores, labels

    # Per-class NMS
    unique_labels = labels.unique()
    keep_all = []

    for cls in unique_labels:
        cls_mask = labels == cls
        cls_keep = nms(boxes[cls_mask], scores[cls_mask], iou_threshold)
        # Map back to original indices
        original_idx = torch.where(cls_mask)[0]
        keep_all.append(original_idx[cls_keep])

    if keep_all:
        keep = torch.cat(keep_all)
    else:
        keep = torch.tensor([], dtype=torch.long)

    return boxes[keep], scores[keep], labels[keep]

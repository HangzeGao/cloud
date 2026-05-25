import numpy as np
import torch

def intersection_over_union(pred, true, n_classes=3, smooth=1e-6):
    """
    Calculates intersection and union for a batch of images.
    """
    valid_pixel_mask = true.ne(255)  # valid pixel mask
    true = true.masked_select(valid_pixel_mask)
    pred = pred.masked_select(valid_pixel_mask)

    iou_list = []
    for cls in range(n_classes):
        true_cls = (true == cls)
        pred_cls = (pred == cls)

        intersection = (true_cls & pred_cls).sum().float()
        union = (true_cls | pred_cls).sum().float()

        iou = (intersection + smooth) / (union + smooth)
        iou_list.append(torch.clamp(iou, max=1.0))

    mIoU = torch.stack(iou_list).mean()

    return mIoU

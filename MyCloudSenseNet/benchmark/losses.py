import numpy as np

def intersection_over_union(pred, true, n_classes=3, smooth=1e-6):
    """
    Calculates intersection and union for a batch of images.
    """
    valid_pixel_mask = true.ne(255)  # valid pixel mask
    true = true.masked_select(valid_pixel_mask).to("cpu")
    pred = pred.masked_select(valid_pixel_mask).to("cpu")

    iou_list = []
    for cls in range(n_classes):
        # skip background
        # if cls == 0:
        #     continue

        # Prediction/ground truth mask of the current category
        true_cls = (true == cls)
        pred_cls = (pred == cls)

        # Intersection and union totals
        intersection = np.logical_and(true_cls, pred_cls)
        union = np.logical_or(true_cls, pred_cls)

        iou = (intersection.sum() + smooth) / (union.sum() + smooth)
        iou_list.append(min(iou, 1))

    mIoU = np.mean(iou_list)

    ious = np.array(iou_list, dtype=np.float32)
    weights = np.array([0.1, 0.3, 0.6], dtype=np.float32)
    weights = weights / weights.sum()

    wIoU = float(np.dot(weights, ious))

    return mIoU

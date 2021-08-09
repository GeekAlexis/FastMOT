import numpy as np
import numba as nb


@nb.njit(cache=True,  inline='always')
def as_tlbr(tlbr):
    """Construct a rectangle from a tuple or np.ndarray."""
    _tlbr = np.empty(4)
    _tlbr[0] = round(float(tlbr[0]), 0)
    _tlbr[1] = round(float(tlbr[1]), 0)
    _tlbr[2] = round(float(tlbr[2]), 0)
    _tlbr[3] = round(float(tlbr[3]), 0)
    return _tlbr


@nb.njit(cache=True, inline='always')
def get_size(tlbr):
    return tlbr[2] - tlbr[0] + 1, tlbr[3] - tlbr[1] + 1


@nb.njit(cache=True,  inline='always')
def aspect_ratio(tlbr):
    w, h = get_size(tlbr)
    return h / w if w > 0 else 0.


@nb.njit(cache=True, inline='always')
def area(tlbr):
    w, h = get_size(tlbr)
    if w <= 0 or h <= 0:
        return 0.
    return w * h


@nb.njit(cache=True,  inline='always')
def get_center(tlbr):
    return (tlbr[0] + tlbr[2]) / 2, (tlbr[1] + tlbr[3]) / 2


@nb.njit(cache=True, inline='always')
def to_tlwh(tlbr):
    tlwh = np.empty(4)
    tlwh[:2] = tlbr[:2]
    tlwh[2:] = get_size(tlbr)
    return tlwh


@nb.njit(cache=True, inline='always')
def to_tlbr(tlwh):
    tlbr = np.empty(4)
    xmin = float(tlwh[0])
    ymin = float(tlwh[1])
    tlbr[0] = round(xmin, 0)
    tlbr[1] = round(ymin, 0)
    tlbr[2] = round(xmin + float(tlwh[2]) - 1., 0)
    tlbr[3] = round(ymin + float(tlwh[3]) - 1., 0)
    return tlbr


@nb.njit(cache=True, inline='always')
def intersection(tlbr1, tlbr2):
    tlbr = np.empty(4)
    tlbr[0] = max(tlbr1[0], tlbr2[0])
    tlbr[1] = max(tlbr1[1], tlbr2[1])
    tlbr[2] = min(tlbr1[2], tlbr2[2])
    tlbr[3] = min(tlbr1[3], tlbr2[3])
    if tlbr[2] < tlbr[0] or tlbr[3] < tlbr[1]:
        return None
    return tlbr


@nb.njit(cache=True, inline='always')
def enclosing(tlbr1, tlbr2):
    tlbr = np.empty(4)
    tlbr[0] = min(tlbr1[0], tlbr2[0])
    tlbr[1] = min(tlbr1[1], tlbr2[1])
    tlbr[2] = max(tlbr1[2], tlbr2[2])
    tlbr[3] = max(tlbr1[3], tlbr2[3])
    return tlbr


@nb.njit(cache=True, inline='always')
def crop(img, tlbr):
    xmin = max(int(tlbr[0]), 0)
    ymin = max(int(tlbr[1]), 0)
    xmax = max(int(tlbr[2]), 0)
    ymax = max(int(tlbr[3]), 0)
    return img[ymin:ymax + 1, xmin:xmax + 1]


@nb.njit(cache=True, inline='always')
def multi_crop(img, tlbrs):
    _tlbrs = tlbrs.astype(np.int_)
    _tlbrs = np.maximum(_tlbrs, 0)
    return [img[_tlbrs[i, 1]:_tlbrs[i, 3] + 1, _tlbrs[i, 0]:_tlbrs[i, 2] + 1]
            for i in range(len(_tlbrs))]


@nb.njit(fastmath=True, cache=True, inline='always')
def ios(tlbr1, tlbr2):
    """Computes intersection over self."""
    iw = min(tlbr1[2], tlbr2[2]) - max(tlbr1[0], tlbr2[0]) + 1
    ih = min(tlbr1[3], tlbr2[3]) - max(tlbr1[1], tlbr2[1]) + 1
    if iw <= 0 or ih <= 0:
        return 0.
    area_inter = iw * ih
    area_self = area(tlbr1)
    return area_inter / area_self


@nb.njit(fastmath=True, cache=True, inline='always')
def iom(tlbr1, tlbr2):
    """Computes intersection over minimum."""
    iw = min(tlbr1[2], tlbr2[2]) - max(tlbr1[0], tlbr2[0]) + 1
    ih = min(tlbr1[3], tlbr2[3]) - max(tlbr1[1], tlbr2[1]) + 1
    if iw <= 0 or ih <= 0:
        return 0.
    area_inter = iw * ih
    area_min = min(area(tlbr1), area(tlbr2))
    return area_inter / area_min


@nb.njit(parallel=False, fastmath=True, cache=True)
def bbox_ious(tlbrs1, tlbrs2):
    """Computes pairwise bounding box overlaps using IoU."""
    ious = np.empty((tlbrs1.shape[0], tlbrs2.shape[0]))
    for i in nb.prange(tlbrs1.shape[0]):
        area1 = area(tlbrs1[i, :])
        for j in range(tlbrs2.shape[0]):
            iw = min(tlbrs1[i, 2], tlbrs2[j, 2]) - max(tlbrs1[i, 0], tlbrs2[j, 0]) + 1
            ih = min(tlbrs1[i, 3], tlbrs2[j, 3]) - max(tlbrs1[i, 1], tlbrs2[j, 1]) + 1
            if iw > 0 and ih > 0:
                area_inter = iw * ih
                area_union = area1 + area(tlbrs2[j, :]) - area_inter
                ious[i, j] = area_inter / area_union
            else:
                ious[i, j] = 0.
    return ious


@nb.njit(parallel=False, fastmath=True, cache=True)
def find_occluded(tlbrs, occlusion_thresh):
    """Computes a mask of occluded bounding boxes."""
    occluded_mask = np.zeros(tlbrs.shape[0], dtype=np.bool_)
    for i in nb.prange(tlbrs.shape[0]):
        area_self = area(tlbrs[i, :])
        for j in range(tlbrs.shape[0]):
            if i != j:
                iw = min(tlbrs[i, 2], tlbrs[j, 2]) - max(tlbrs[i, 0], tlbrs[j, 0]) + 1
                ih = min(tlbrs[i, 3], tlbrs[j, 3]) - max(tlbrs[i, 1], tlbrs[j, 1]) + 1
                if iw > 0 and ih > 0:
                    ios = iw * ih / area_self
                    if ios > occlusion_thresh:
                        occluded_mask[i] = True
                        break
    return occluded_mask


@nb.njit(fastmath=True, cache=True)
def nms(tlwhs, scores, nms_thresh):
    """Applies Non-Maximum Suppression on the bounding boxes [x, y, w, h].
    Returns an array with the indexes of the bounding boxes we want to keep.
    """
    areas = tlwhs[:, 2] * tlwhs[:, 3]
    ordered = scores.argsort()[::-1]

    tls = tlwhs[:, :2]
    brs = tlwhs[:, :2] + tlwhs[:, 2:] - 1

    keep = []
    while ordered.size > 0:
        # index of the current element
        i = ordered[0]
        keep.append(i)

        other_tls = tls[ordered[1:]]
        other_brs = brs[ordered[1:]]

        # compute IoU
        inter_xmin = np.maximum(tls[i, 0], other_tls[:, 0])
        inter_ymin = np.maximum(tls[i, 1], other_tls[:, 1])
        inter_xmax = np.minimum(brs[i, 0], other_brs[:, 0])
        inter_ymax = np.minimum(brs[i, 1], other_brs[:, 1])

        inter_w = np.maximum(0, inter_xmax - inter_xmin + 1)
        inter_h = np.maximum(0, inter_ymax - inter_ymin + 1)
        inter_area = inter_w * inter_h
        union_area = areas[i] + areas[ordered[1:]] - inter_area
        iou = inter_area / union_area

        idx = np.where(iou <= nms_thresh)[0]
        ordered = ordered[idx + 1]
    keep = np.array(keep)
    return keep


@nb.njit(fastmath=True, cache=True)
def diou_nms(tlwhs, scores, nms_thresh, beta=0.6):
    """Applies Non-Maximum Suppression using the DIoU metric."""
    areas = tlwhs[:, 2] * tlwhs[:, 3]
    ordered = scores.argsort()[::-1]

    tls = tlwhs[:, :2]
    brs = tlwhs[:, :2] + tlwhs[:, 2:] - 1
    centers = (tls + brs) / 2

    keep = []
    while ordered.size > 0:
        # index of the current element
        i = ordered[0]
        keep.append(i)

        other_tls = tls[ordered[1:]]
        other_brs = brs[ordered[1:]]

        # compute IoU
        inter_xmin = np.maximum(tls[i, 0], other_tls[:, 0])
        inter_ymin = np.maximum(tls[i, 1], other_tls[:, 1])
        inter_xmax = np.minimum(brs[i, 0], other_brs[:, 0])
        inter_ymax = np.minimum(brs[i, 1], other_brs[:, 1])

        inter_w = np.maximum(0, inter_xmax - inter_xmin + 1)
        inter_h = np.maximum(0, inter_ymax - inter_ymin + 1)
        inter_area = inter_w * inter_h
        union_area = areas[i] + areas[ordered[1:]] - inter_area
        iou = inter_area / union_area

        # compute DIoU
        encls_xmin = np.minimum(tls[i, 0], other_tls[:, 0])
        encls_ymin = np.minimum(tls[i, 1], other_tls[:, 1])
        encls_xmax = np.maximum(brs[i, 0], other_brs[:, 0])
        encls_ymax = np.maximum(brs[i, 1], other_brs[:, 1])

        encls_w = encls_xmax - encls_xmin + 1
        encls_h = encls_ymax - encls_ymin + 1
        c = encls_w**2 + encls_h**2
        d = np.sum((centers[i] - centers[ordered[1:]])**2, axis=1)
        diou = iou - (d / c)**beta

        idx = np.where(diou <= nms_thresh)[0]
        ordered = ordered[idx + 1]
    keep = np.array(keep)
    return keep

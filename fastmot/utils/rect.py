import numpy as np
import numba as nb


@nb.njit(cache=True)
def as_rect(tlbr):
    tlbr = np.asarray(tlbr, np.float64)
    tlbr = np.rint(tlbr)
    return tlbr


@nb.njit(cache=True)
def get_size(tlbr):
    tl, br = tlbr[:2], tlbr[2:]
    size = br - tl + 1
    return size


@nb.njit(cache=True)
def area(tlbr):
    size = get_size(tlbr)
    return int(size[0] * size[1])


@nb.njit(cache=True)
def mask_area(mask):
    return np.count_nonzero(mask)


@nb.njit(cache=True)
def get_center(tlbr):
    xmin, ymin, xmax, ymax = tlbr
    return np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])


@nb.njit(cache=True)
def get_corners(tlbr):
    xmin, ymin, xmax, ymax = tlbr
    return np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])


@nb.njit(cache=True)
def to_tlwh(tlbr):
    return np.append(tlbr[:2], get_size(tlbr))


@nb.njit(cache=True)
def to_tlbr(tlwh):
    tlwh = np.asarray(tlwh, np.float64)
    tlwh = np.rint(tlwh)
    tl, size = tlwh[:2], tlwh[2:]
    br = tl + size - 1
    return np.append(tl, br)


@nb.njit(cache=True)
def contains(tlbr1, tlbr2):
    tl1, br1 = tlbr1[:2], tlbr1[2:]
    tl2, br2 = tlbr2[:2], tlbr2[2:]
    return np.all((tl2 >= tl1) & (br2 <= br1))


@nb.njit(cache=True)
def intersection(tlbr1, tlbr2):
    tl1, br1 = tlbr1[:2], tlbr1[2:]
    tl2, br2 = tlbr2[:2], tlbr2[2:]
    tl = np.maximum(tl1, tl2)
    br = np.minimum(br1, br2)
    tlbr = np.append(tl, br)
    if np.any(get_size(tlbr) <= 0):
        return None
    return tlbr


@nb.njit(cache=True)
def union(tlbr1, tlbr2):
    tl1, br1 = tlbr1[:2], tlbr1[2:]
    tl2, br2 = tlbr2[:2], tlbr2[2:]
    tl = np.minimum(tl1, tl2)
    br = np.maximum(br1, br2)
    tlbr = np.append(tl, br)
    return tlbr


@nb.njit(cache=True)
def crop(img, tlbr):
    xmin, ymin, xmax, ymax = tlbr.astype(np.int_)
    return img[ymin:ymax + 1, xmin:xmax + 1]


@nb.njit(cache=True)
def multi_crop(img, tlbrs):
    _tlbrs = tlbrs.astype(np.int_)
    return [img[_tlbrs[i][1]:_tlbrs[i][3] + 1, _tlbrs[i][0]:_tlbrs[i][2] + 1]
        for i in range(len(_tlbrs))]
    

@nb.njit(fastmath=True, cache=True)
def iom(tlbr1, tlbr2):
    """
    Computes intersection over minimum.
    """
    tlbr = intersection(tlbr1, tlbr2)
    if tlbr is None:
        return 0.
    area_intersection = area(tlbr)
    area_minimum = min(area(tlbr1), area(tlbr2))
    return area_intersection / area_minimum


@nb.njit(fastmath=True, cache=True)
def warp(tlbr, m):
    """
    Warps bounding box with a 3x3 transformation matrix.
    """
    corners = get_corners(tlbr)
    warped_corners = perspective_transform(corners, m)
    xmin = min(warped_corners[:, 0])
    ymin = min(warped_corners[:, 1])
    xmax = max(warped_corners[:, 0])
    ymax = max(warped_corners[:, 1])
    return as_rect((xmin, ymin, xmax, ymax))


@nb.njit(fastmath=True, cache=True)
def transform(pts, m):
    """
    Numba implementation of OpenCV's transform.
    """
    pts = np.asarray(pts)
    pts = np.atleast_2d(pts)
    augment = np.ones((len(pts), 1))
    pts = np.concatenate((pts, augment), axis=1)
    return pts @ m.T


@nb.njit(fastmath=True, cache=True)
def perspective_transform(pts, m):
    """
    Numba implementation of OpenCV's perspectiveTransform.
    """
    pts = np.asarray(pts)
    pts = np.atleast_2d(pts)
    augment = np.ones((len(pts), 1))
    pts = np.concatenate((pts, augment), axis=1).T
    pts = m @ pts
    pts = pts / pts[-1]
    return pts[:2].T


@nb.njit(fastmath=True, cache=True)
def nms(tlwhs, scores, nms_thresh):
    """
    Applies the Non-Maximum Suppression algorithm on the bounding boxes [x, y, w, h]
    with their confidence scores and return an array with the indexes of the bounding
    boxes we want to keep
    """
    areas = tlwhs[:, 2] * tlwhs[:, 3]
    ordered = scores.argsort()[::-1]

    tl = tlwhs[:, :2]
    br = tlwhs[:, :2] + tlwhs[:, 2:] - 1

    keep = []
    while ordered.size > 0:
        # index of the current element
        i = ordered[0]
        keep.append(i)
        
        # compute IOU
        candidate_tl = tl[ordered[1:]]
        candidate_br = br[ordered[1:]]

        overlap_xmin = np.maximum(tl[i, 0], candidate_tl[:, 0])
        overlap_ymin = np.maximum(tl[i, 1], candidate_tl[:, 1])
        overlap_xmax = np.minimum(br[i, 0], candidate_br[:, 0])
        overlap_ymax = np.minimum(br[i, 1], candidate_br[:, 1])
        
        width = np.maximum(0, overlap_xmax - overlap_xmin + 1)
        height = np.maximum(0, overlap_ymax - overlap_ymin + 1)
        area_intersection = width * height
        area_union = areas[i] + areas[ordered[1:]] - area_intersection
        iou = area_intersection / area_union

        idx = np.where(iou <= nms_thresh)[0]
        ordered = ordered[idx + 1]
    keep = np.asarray(keep)
    return keep

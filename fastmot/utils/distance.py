import numpy as np
import numba as nb

from .rect import area, get_center


INF_DIST = 1e5


@nb.njit(parallel=True, fastmath=True, cache=True)
def euclidean(XA, XB, symmetric=False):
    """Numba implementation of Scipy's euclidean"""
    Y = np.empty((XA.shape[0], XB.shape[0]))
    for i in nb.prange(XA.shape[0]):
        for j in range(XB.shape[0]):
            if not symmetric or i < j:
                norm = 0.
                for k in range(XA.shape[1]):
                    norm += (XA[i, k] - XB[j, k])**2
                Y[i, j] = np.sqrt(norm)
            else:
                Y[i, j] = INF_DIST
    return Y


@nb.njit(parallel=True, fastmath=True, cache=True)
def cosine(XA, XB, symmetric=False):
    """Numba implementation of Scipy's cosine"""
    Y = np.empty((XA.shape[0], XB.shape[0]))
    for i in nb.prange(XA.shape[0]):
        for j in range(XB.shape[0]):
            if not symmetric or i < j:
                dot    = 0.
                a_norm = 0.
                b_norm = 0.
                for k in range(XA.shape[1]):
                    dot    += XA[i, k] * XB[j, k]
                    a_norm += XA[i, k] * XA[i, k]
                    b_norm += XB[j, k] * XB[j, k]
                a_norm = np.sqrt(a_norm)
                b_norm = np.sqrt(b_norm)
                Y[i, j] = 1. - dot / (a_norm * b_norm)
            else:
                Y[i, j] = INF_DIST
    return Y


@nb.njit(cache=True)
def cdist(XA, XB, metric='euclidean'):
    """Numba implementation of Scipy's cdist"""
    assert XA.ndim == XB.ndim == 2
    assert XA.shape[1] == XB.shape[1]

    if metric == 'euclidean':
        return euclidean(XA, XB)
    elif metric == 'cosine':
        return cosine(XA, XB)
    else:
        raise RuntimeError("Unsupported distance metric, use 'euclidean' or 'cosine'")


@nb.njit(cache=True)
def pdist(X, metric='euclidean'):
    """Numba implementation of Scipy's pdist"""
    assert X.ndim == 2

    if metric == 'euclidean':
        return euclidean(X, X, True)
    elif metric == 'cosine':
        return cosine(X, X, True)
    else:
        raise RuntimeError("Unsupported distance metric, use 'euclidean' or 'cosine'")


@nb.njit(parallel=False, fastmath=True, cache=True)
def iou_dist(tlbrs1, tlbrs2):
    """Computes pairwise IoU distance."""
    Y = np.empty((tlbrs1.shape[0], tlbrs2.shape[0]))
    for i in nb.prange(tlbrs1.shape[0]):
        area1 = area(tlbrs1[i, :])
        for j in range(tlbrs2.shape[0]):
            iw = min(tlbrs1[i, 2], tlbrs2[j, 2]) - max(tlbrs1[i, 0], tlbrs2[j, 0]) + 1
            ih = min(tlbrs1[i, 3], tlbrs2[j, 3]) - max(tlbrs1[i, 1], tlbrs2[j, 1]) + 1
            if iw > 0 and ih > 0:
                area_inter = iw * ih
                area_union = area1 + area(tlbrs2[j, :]) - area_inter
                Y[i, j] = 1. - area_inter / area_union
            else:
                Y[i, j] = 1.
    return Y


@nb.njit(parallel=True, fastmath=True, cache=True)
def diou_dist(tlbrs1, tlbrs2):
    """Computes pairwise DIoU distance."""
    Y = np.empty((tlbrs1.shape[0], tlbrs2.shape[0]))
    for i in nb.prange(tlbrs1.shape[0]):
        area1 = area(tlbrs1[i, :])
        x1, y1 = get_center(tlbrs1[i, :])
        for j in range(tlbrs2.shape[0]):
            iou = 0.
            iw = min(tlbrs1[i, 2], tlbrs2[j, 2]) - max(tlbrs1[i, 0], tlbrs2[j, 0]) + 1
            ih = min(tlbrs1[i, 3], tlbrs2[j, 3]) - max(tlbrs1[i, 1], tlbrs2[j, 1]) + 1
            if iw > 0 and ih > 0:
                area_inter = iw * ih
                area_union = area1 + area(tlbrs2[j, :]) - area_inter
                iou = area_inter / area_union
            ew = max(tlbrs1[i, 2], tlbrs2[j, 2]) - min(tlbrs1[i, 0], tlbrs2[j, 0]) + 1
            eh = max(tlbrs1[i, 3], tlbrs2[j, 3]) - min(tlbrs1[i, 1], tlbrs2[j, 1]) + 1
            c = ew**2 + eh**2
            x2, y2 = get_center(tlbrs2[j, :])
            d = (x2 - x1)**2 + (y2 - y1)**2
            diou = iou - (d / c)**0.6
            Y[i, j] = (1. - diou) * 0.5
    return Y


@nb.njit(cache=True)
def bbox_dist(tlbrs1, tlbrs2, metric='iou'):
    """Computes pairwise bounding box distance."""
    assert tlbrs1.ndim == tlbrs2.ndim == 2
    assert tlbrs1.shape[1] == tlbrs2.shape[1] == 4

    if metric == 'iou':
        return iou_dist(tlbrs1, tlbrs2)
    elif metric == 'diou':
        return diou_dist(tlbrs1, tlbrs2)
    else:
        raise RuntimeError("Unsupported distance metric, use 'iou' or 'diou'")

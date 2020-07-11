import numpy as np
from scipy.linalg import solve_triangular


def mahalanobis_dist(x, mean, cov):
    """Compute mahalanobis distance between points in `x` and a Gaussian distribution.
    Parameters
    ----------
    x : array_like
        An MxN matrix of N samples of dimensionality M.
    mean : array_like
        An Mx1 matrix of dimensionality M.
    cov : array_like
        An MxM matrix of dimensionality M.
    Returns
    -------
    ndarray
        Returns a array of size N such that element i
        contains the squared mahalanobis distance for `x[:, i]`.
    """
    x, mean, cov = np.asarray(x), np.asarray(mean), np.asarray(cov)
    if x.shape[1] == 0:
        return np.zeros(x.shape[1])
    diff = x - mean
    L = np.linalg.cholesky(cov)
    y = solve_triangular(L, diff, lower=True, overwrite_b=True, check_finite=False)
    return np.sum(y**2, axis=0)

    # x, mean, cov = np.asarray(x), np.asarray(mean), np.asarray(cov)
    # if len(x) == 0:
    #     return np.zeros(len(x))
    # diff = x - mean
    # L = np.linalg.cholesky(cov)
    # y = solve_triangular(L, diff.T, lower=True, overwrite_b=True, check_finite=False)
    # return np.sum(y**2, axis=0)


def euclidean_dist(x, y):
    """Compute pair-wise squared distance between points in `x` and `y`.
    Parameters
    ----------
    x : array_like
        An NxM matrix of N samples of dimensionality M.
    y : array_like
        An LxM matrix of L samples of dimensionality M.
    Returns
    -------
    ndarray
        Returns a matrix of size N, L such that element (i, j)
        contains the squared distance between `x[i]` and `y[j]`.
    """
    x, y = np.asarray(x), np.asarray(y)
    if len(x) == 0 or len(y) == 0:
        return np.zeros((len(x), len(y)))
    xx, yy = np.square(x).sum(axis=1), np.square(y).sum(axis=1)
    squared_l2 = -2 * x @ y.T + xx[:, np.newaxis] + yy[np.newaxis, :]
    squared_l2 = np.clip(squared_l2, 0, float(np.inf))
    return np.sqrt(squared_l2)


def iou(bbox, candidates):
    """Compute intersection over union.
    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, bottom right x, bottom right y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.
    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.
    """
    bbox, candidates = np.asarray(bbox), np.asarray(candidates)
    if len(candidates) == 0:
        return np.zeros(len(candidates))

    overlap_xmin = np.maximum(bbox[0], candidates[:, 0])
    overlap_ymin = np.maximum(bbox[1], candidates[:, 1])
    overlap_xmax = np.minimum(bbox[2], candidates[:, 2])
    overlap_ymax = np.minimum(bbox[3], candidates[:, 3])
    
    area_intersection = np.maximum(0, overlap_xmax - overlap_xmin + 1) * \
                        np.maximum(0, overlap_ymax - overlap_ymin + 1)
    area_bbox = bbox[4:].prod()
    area_candidates = candidates[:, 4:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)
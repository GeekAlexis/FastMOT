import numpy as np
import numba as nb


@nb.njit(fastmath=True, cache=True)
def apply_along_axis(func1d, mat, axis):
    """Numba utility to apply reduction to a given axis."""
    assert mat.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(mat.shape[1], mat.dtype)
        for i in range(len(result)):
            result[i, :] = func1d(mat[:, i])
    else:
        result = np.empty(mat.shape[0], mat.dtype)
        for i in range(len(result)):
            result[i, :] = func1d(mat[i, :])
    return result


@nb.njit(parallel=True, fastmath=True, cache=True)
def normalize_vec(vectors):
    """Numba utility to normalize an array of vectors."""
    assert vectors.ndim == 2
    out = np.empty_like(vectors)
    for i in nb.prange(vectors.shape[0]):
        norm_factor = 1. / np.linalg.norm(vectors[i, :])
        out[i, :] = norm_factor * vectors[i, :]
    return out


@nb.njit(fastmath=True, cache=True)
def mask_area(mask):
    """Utility to calculate the area of a mask."""
    count = 0
    for val in mask.ravel():
        if val != 0:
            count += 1
    return count


@nb.njit(fastmath=True, cache=True)
def bisect_right(arr, val, left=0):
    """Utility to search a value in a sorted array."""
    right = len(arr)
    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] >= val:
            left = mid + 1
        else:
            right = mid
    return left


@nb.njit(fastmath=True, cache=True, inline='always')
def transform(pts, m):
    """Numba implementation of OpenCV's transform."""
    pts = np.asarray(pts, dtype=np.float64)
    pts = np.atleast_2d(pts)

    augment = np.ones((len(pts), 1))
    pts = np.concatenate((pts, augment), axis=1)
    return pts @ m.T


@nb.njit(fastmath=True, cache=True, inline='always')
def perspective_transform(pts, m):
    """Numba implementation of OpenCV's perspectiveTransform."""
    pts = np.asarray(pts, dtype=np.float64)
    pts = np.atleast_2d(pts)

    augment = np.ones((len(pts), 1))
    pts = np.concatenate((pts, augment), axis=1).T
    pts = m @ pts
    pts = pts / pts[-1]
    return pts[:2].T

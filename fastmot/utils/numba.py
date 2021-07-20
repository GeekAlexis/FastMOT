import numpy as np
import numba as nb


@nb.njit(fastmath=True, cache=True)
def apply_along_axis(func1d, mat, axis):
    """
    Numba utility to apply reduction to a given axis.
    """
    assert mat.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(mat.shape[1])
        for i in range(len(result)):
            result[i] = func1d(mat[:, i])
    else:
        result = np.empty(mat.shape[0])
        for i in range(len(result)):
            result[i] = func1d(mat[i, :])
    return result


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

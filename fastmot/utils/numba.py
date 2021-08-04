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


@nb.njit(cache=True)
def delete_row_col(mat, row, col):
    """Numba utility to delete a row and column of a matrix."""
    assert mat.ndim == 2
    row_mask = np.ones(mat.shape[0], dtype=np.bool_)
    col_mask = np.ones(mat.shape[1], dtype=np.bool_)
    row_mask[row] = False
    col_mask[col] = False
    return mat[row_mask, :][:, col_mask]


@nb.njit(parallel=True, fastmath=True, cache=True)
def normalize_vec(vectors):
    """Numba utility to normalize an array of vectors."""
    assert vectors.ndim == 2
    out = np.empty_like(vectors)
    for i in nb.prange(len(vectors)):
        norm_factor = 1. / np.linalg.norm(vectors[i, :])
        out[i, :] = norm_factor * vectors[i, :]
    return out


@nb.njit(fastmath=True, cache=True)
def transform(pts, m):
    """Numba implementation of OpenCV's transform."""
    pts = np.asarray(pts, dtype=np.float64)
    pts = np.atleast_2d(pts)
    assert m.ndim == 2 and m.shape == (2, 3)

    augment = np.ones((len(pts), 1))
    pts = np.concatenate((pts, augment), axis=1)
    return pts @ m.T


@nb.njit(fastmath=True, cache=True)
def perspective_transform(pts, m):
    """Numba implementation of OpenCV's perspectiveTransform."""
    pts = np.asarray(pts, dtype=np.float64)
    pts = np.atleast_2d(pts)
    assert m.ndim == 2 and m.shape == (3, 3)

    augment = np.ones((len(pts), 1))
    pts = np.concatenate((pts, augment), axis=1).T
    pts = m @ pts
    pts = pts / pts[-1]
    return pts[:2].T

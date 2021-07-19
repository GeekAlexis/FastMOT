import numpy as np
import numba as nb


@nb.njit(parallel=True, fastmath=True, cache=True)
def euclidean(XA, XB):
    """
    Numba implementation of Scipy's euclidean
    """
    Y = np.empty((XA.shape[0], XB.shape[0]))
    for i in nb.prange(XA.shape[0]):
        for j in range(XB.shape[0]):
            norm = 0.
            for k in range(XA.shape[1]):
                norm += (XA[i, k] - XB[j, k])**2
            Y[i,j]= np.sqrt(norm)
    return Y


@nb.njit(parallel=True, fastmath=True, cache=True)
def cosine(XA, XB):
    """
    Numba implementation of Scipy's cosine
    """
    Y = np.empty((XA.shape[0], XB.shape[0]))
    for i in nb.prange(XA.shape[0]):
        for j in range(XB.shape[0]):
            dot    = 0.
            a_norm = 0.
            b_norm = 0.
            for k in range(XA.shape[1]):
                dot    += XA[i, k] * XB[j, k]
                a_norm += XA[i, k] * XA[i, k]
                b_norm += XB[j, k] * XB[j, k]
            a_norm = np.sqrt(a_norm)
            b_norm = np.sqrt(b_norm)
            Y[i,j]= 1. - dot / (a_norm * b_norm)
    return Y


@nb.njit(cache=True)
def cdist(XA, XB, metric='euclidean'):
    assert XA.shape[1] == XB.shape[1]

    if metric == 'euclidean':
        return euclidean(XA, XB)
    elif metric == 'cosine':
        return cosine(XA, XB)
    else:
        raise RuntimeError("Unsupported distance metric, use 'euclidean' or 'cosine'")

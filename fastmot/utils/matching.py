from scipy.optimize import linear_sum_assignment
import numpy as np
import numba as nb


CHI_SQ_INV_95 = 9.4877 # 0.95 quantile of chi-square distribution
INF_COST = 1e5


def linear_assignment(cost, row_ids, col_ids):
    """Solves the linear assignment problem.

    Parameters
    ----------
    cost : ndarray
        The cost matrix.
    row_ids : List[int]
        IDs that correspond to each row in the cost matrix.
    col_ids : List[int]
        IDs that correspond to each column in the cost matrix.

    Returns
    -------
    List[tuple], List[int], List[int]
        Matched row and column IDs, unmatched row IDs, and unmatched column IDs.
    """
    m_rows, m_cols = linear_sum_assignment(cost)
    row_ids = np.fromiter(row_ids, int, len(row_ids))
    col_ids = np.fromiter(col_ids, int, len(col_ids))
    return _get_assignment_matches(cost, row_ids, col_ids, m_rows, m_cols)


def greedy_match(cost, row_ids, col_ids, max_cost):
    """Performs greedy matching until the cost exceeds `max_cost`.

    Parameters
    ----------
    cost : ndarray
        The cost matrix.
    row_ids : List[int]
        IDs that correspond to each row in the cost matrix.
    col_ids : List[int]
        IDs that correspond to each column in the cost matrix.
    max_cost : float
        Maximum cost allowed to match a row with a column.

    Returns
    -------
    List[tuple], List[int], List[int]
        Matched row and column IDs, unmatched row IDs, and unmatched column IDs.
    """
    row_ids = np.fromiter(row_ids, int, len(row_ids))
    col_ids = np.fromiter(col_ids, int, len(col_ids))
    return _greedy_match(cost, row_ids, col_ids, max_cost)


@nb.njit(fastmath=True, cache=True)
def _get_assignment_matches(cost, row_ids, col_ids, m_rows, m_cols):
    unmatched_rows = list(set(range(cost.shape[0])) - set(m_rows))
    unmatched_cols = list(set(range(cost.shape[1])) - set(m_cols))
    unmatched_row_ids = [row_ids[row] for row in unmatched_rows]
    unmatched_col_ids = [col_ids[col] for col in unmatched_cols]
    matches = []
    for row, col in zip(m_rows, m_cols):
        if cost[row, col] < INF_COST:
            matches.append((row_ids[row], col_ids[col]))
        else:
            unmatched_row_ids.append(row_ids[row])
            unmatched_col_ids.append(col_ids[col])
    return matches, unmatched_row_ids, unmatched_col_ids


@nb.njit(fastmath=True, cache=True)
def _greedy_match(cost, row_ids, col_ids, max_cost):
    indices_rows = np.arange(cost.shape[0])
    indices_cols = np.arange(cost.shape[1])

    matches = []
    while cost.shape[0] > 0 and cost.shape[1] > 0:
        idx = np.argmin(cost)
        i, j = idx // cost.shape[1], idx % cost.shape[1]
        if cost[i, j] <= max_cost:
            matches.append((row_ids[indices_rows[i]], col_ids[indices_cols[j]]))
            row_mask = np.ones(cost.shape[0], np.bool_)
            col_mask = np.ones(cost.shape[1], np.bool_)
            row_mask[i] = False
            col_mask[j] = False

            indices_rows = indices_rows[row_mask]
            indices_cols = indices_cols[col_mask]
            cost = cost[row_mask, :][:, col_mask]
        else:
            break

    unmatched_row_ids = [row_ids[row] for row in indices_rows]
    unmatched_col_ids = [col_ids[col] for col in indices_cols]
    return matches, unmatched_row_ids, unmatched_col_ids


@nb.njit(fastmath=True, cache=True)
def fuse_motion(cost, m_dist, m_weight):
    """Fuse each row of cost matrix with motion information."""
    norm_factor = 1. / CHI_SQ_INV_95
    f_weight = 1. - m_weight
    cost[:] = f_weight * cost + m_weight * norm_factor * m_dist
    cost[m_dist > CHI_SQ_INV_95] = INF_COST


@nb.njit(parallel=False, fastmath=True, cache=True)
def gate_cost(cost, row_labels, col_labels, max_cost=None):
    """Gate cost matrix if cost exceeds the maximum."""
    for i in nb.prange(cost.shape[0]):
        for j in range(cost.shape[1]):
            if (row_labels[i] != col_labels[j] or
                max_cost is not None and cost[i, j] > max_cost):
                cost[i, j] = INF_COST

from scipy.optimize import linear_sum_assignment
import numpy as np
import numba as nb


INF_COST = 1e5


def linear_assignment(cost, row_ids, col_ids):
    m_rows, m_cols = linear_sum_assignment(cost)
    return _get_assignment_matches(cost, row_ids, col_ids, m_rows, m_cols)


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
def greedy_match(cost, row_ids, col_ids, thresh):
    indices_rows = np.arange(cost.shape[0])
    indices_cols = np.arange(cost.shape[1])

    matches = []
    while cost.shape[0] > 0 and cost.shape[1] > 0:
        idx = np.argmin(cost)
        i, j = idx // cost.shape[1], idx % cost.shape[1]
        if cost[i, j] <= thresh:
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


@nb.njit(parallel=False, fastmath=True, cache=True)
def gate_cost(cost, row_labels, col_labels, thresh=None):
    for i in nb.prange(cost.shape[0]):
        for j in range(cost.shape[1]):
            if (row_labels[i] != col_labels[j] or
                thresh is not None and cost[i, j] > thresh):
                cost[i, j] = 1e5

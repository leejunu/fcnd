from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numba import jit
 
from fcnd.utils.cc_utils import eBH_vector, e_function_vector, p_function_vector

PROXY_BACKUP_WEIGHT = 1e-6


@jit(nopython=True)
def neglog_empirical_rank_scores_numba(score_mat: np.ndarray) -> np.ndarray:
    """Per-learner negative log empirical tail-rank transform.

    For each learner row and score s_j, returns
        -log((1 + sum_i 1{s_i >= s_j}) / (N + 1)).
    This is monotone in the raw score, permutation-equivariant, and puts the
    learner rows on a common rank-based scale before smoothing.
    """
    L, N = score_mat.shape
    out = np.empty((L, N), dtype=np.float64)
    denom = float(N + 1)
    for li in range(L):
        for j in range(N):
            count_ge = 1.0
            sj = score_mat[li, j]
            for r in range(N):
                if score_mat[li, r] >= sj:
                    count_ge += 1.0
            out[li, j] = -np.log(count_ge / denom)
    return out


@jit(nopython=True)
def normalized_proxy_utility_numba(
    nsel: np.ndarray,
    nsel_backup: np.ndarray,
    m: int,
) -> np.ndarray:
    """Normalized proxy selector utility on a common scale across m."""
    scale = float(m) if m > 0 else 1.0
    utility = np.empty(nsel.shape[0], dtype=np.float64)
    for li in range(nsel.shape[0]):
        utility[li] = (nsel[li] / scale) + PROXY_BACKUP_WEIGHT * (nsel_backup[li] / scale)
    return utility


def normalized_proxy_utility_py(
    nsel: np.ndarray,
    nsel_backup: np.ndarray,
    m: int,
) -> np.ndarray:
    """Python analogue of the normalized proxy utility used by selection."""
    scale = float(m) if int(m) > 0 else 1.0
    nsel = np.asarray(nsel, dtype=float)
    nsel_backup = np.asarray(nsel_backup, dtype=float)
    return (nsel / scale) + PROXY_BACKUP_WEIGHT * (nsel_backup / scale)


def neglog_empirical_rank_scores_py(score_mat: np.ndarray) -> np.ndarray:
    """Python analogue of the rank-based score transform used for smoothing."""
    arr = np.asarray(score_mat, dtype=float)
    L, N = arr.shape
    out = np.empty((L, N), dtype=float)
    denom = float(N + 1)
    for li in range(L):
        row = arr[li]
        counts = 1.0 + np.sum(row[None, :] >= row[:, None], axis=1)
        out[li] = -np.log(counts / denom)
    return out

@jit(nopython=True)
def setdiff1d_numba(m: int, block: np.ndarray) -> np.ndarray:
    mask = np.ones(m, dtype=np.bool_)
    for i in range(block.size):
        b = block[i]
        if 0 <= b < m:
            mask[b] = False
    return np.where(mask)[0]


@jit(nopython=True)
def subroutine_mdlsel_numba_from_pools(
    m: int,
    n: int,
    blocks: List[np.ndarray],
    alpha_fdr: float,
    raw_score_mat: np.ndarray,
    smooth_score_mat: np.ndarray,
    weights_ref: np.ndarray,
    weights_test: np.ndarray,
    metric_flag: int,
    selector_top_k: int,
    selector_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Numba model-selection subroutine using p-values + BH or e-values + eBH.

    Parameters
    - blocks: numba.typed.List of int64 arrays
    - raw_score_mat: shape (L, n+m), used for utility calculation / top-k membership
    - smooth_score_mat: shape (L, n+m), used as the source pool for the returned
      selected row when selector_top_k > 1
    - metric_flag: 0 -> p+BH, 1 -> e+eBH
    Returns selected/smoothed scores matrix (m, n+m) and top-1 learner index per block.
    """
    L = raw_score_mat.shape[0]
    scores = np.zeros((m, n + m))
    sel_idx_pb = np.empty(len(blocks), dtype=np.int64)
    use_smoothed_selector = selector_top_k > 1
    selector_top_k = min(max(1, int(selector_top_k)), L)

    for block_idx in range(len(blocks)):
        block_arr = blocks[block_idx]
        all_but_block = setdiff1d_numba(m, block_arr)
        bsize = block_arr.size
        m_rem = m - bsize

        nsel = np.zeros(L)
        nsel_backup = np.zeros(L)

        for li in range(L):
            row = raw_score_mat[li]
            # Build unordered tuple S_j = ref + test[block]
            n_aug = n + bsize
            aug_scores = np.empty(n_aug)
            aug_weights = np.empty(n_aug)
            for i in range(n):
                aug_scores[i] = row[i]
                aug_weights[i] = weights_ref[i]
            for i in range(bsize):
                aug_scores[n + i] = row[n + block_arr[i]]
                aug_weights[n + i] = weights_test[block_arr[i]]

            # Proxy reference set: bottom n values from the unordered tuple S_j.
            # Proxy test set: the remaining bsize highest elements from S_j plus
            # the m-bsize held-out test points, yielding m proxy p/e-values.
            order_aug = np.argsort(aug_scores)
            S = np.empty(n + m)
            cw = np.empty(n)
            for i in range(n):
                idx = order_aug[i]
                S[i] = aug_scores[idx]
                cw[i] = aug_weights[idx]

            tw = np.empty(m)
            for i in range(bsize):
                idx = order_aug[n + i]
                S[n + i] = aug_scores[idx]
                tw[i] = aug_weights[idx]
            for i in range(m_rem):
                S[n + bsize + i] = row[n + all_but_block[i]]
                tw[bsize + i] = weights_test[all_but_block[i]]

            # Selection metrics
            cnt = 0.0
            cnt_bk = 0.0
            if metric_flag == 0:
                pvals = p_function_vector(S, cw, tw, 1.0)
                rej = eBH_vector(1.0 / pvals, alpha_fdr)
                cnt = rej.sum()
                cnt_bk = (pvals <= alpha_fdr).sum()
                 
            else:
                evals = e_function_vector(alpha_fdr, S, cw, tw)
                rej = eBH_vector(evals, alpha_fdr)
                thr = 1.0 / alpha_fdr
                cnt = rej.sum()
                cnt_bk = (evals >= thr).sum()
            nsel[li] = cnt
            nsel_backup[li] = cnt_bk

        utility = normalized_proxy_utility_numba(nsel, nsel_backup, m)
        best_idx = int(np.argmax(utility))
        sel_idx_pb[block_idx] = best_idx

        if not use_smoothed_selector:
            # Write selected learner's scores for all members in the block.
            for t in range(bsize):
                j = int(block_arr[t])
                for c in range(n + m):
                    scores[j, c] = raw_score_mat[best_idx, c]
            continue

        order = np.argsort(utility)
        top_idx = order[L - selector_top_k :]
        top_util = np.empty(selector_top_k, dtype=np.float64)
        max_util = utility[top_idx[0]]
        for i in range(selector_top_k):
            top_util[i] = utility[top_idx[i]]
            if top_util[i] > max_util:
                max_util = top_util[i]

        weight_sum = 0.0
        weights = np.empty(selector_top_k, dtype=np.float64)
        for i in range(selector_top_k):
            weights[i] = np.exp(selector_lambda * (top_util[i] - max_util))
            weight_sum += weights[i]
        if weight_sum <= 0.0:
            for i in range(selector_top_k):
                weights[i] = 1.0 / selector_top_k
        else:
            for i in range(selector_top_k):
                weights[i] /= weight_sum

        smooth_row = np.zeros(n + m, dtype=np.float64)
        for i in range(selector_top_k):
            li = top_idx[i]
            for c in range(n + m):
                smooth_row[c] += weights[i] * smooth_score_mat[li, c]

        for t in range(bsize):
            j = int(block_arr[t])
            for c in range(n + m):
                scores[j, c] = smooth_row[c]

    return scores, sel_idx_pb


@jit(nopython=True)
def subroutine_mdlsel_numba(
    m: int,
    n: int,
    blocks: List[np.ndarray],
    alpha_fdr: float,
    score_mat: np.ndarray,
    weights_ref: np.ndarray,
    weights_test: np.ndarray,
    metric_flag: int,
    selector_top_k: int,
    selector_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    smooth_source = score_mat
    if selector_top_k > 1:
        smooth_source = neglog_empirical_rank_scores_numba(score_mat)
    return subroutine_mdlsel_numba_from_pools(
        m,
        n,
        blocks,
        alpha_fdr,
        score_mat,
        smooth_source,
        weights_ref,
        weights_test,
        metric_flag,
        selector_top_k,
        selector_lambda,
    )


def subroutine_mdlsel_py_from_pools(
    m: int,
    n: int,
    blocks,
    alpha_fdr: float,
    raw_score_mat: np.ndarray,
    smooth_score_mat: np.ndarray,
    weights_ref: np.ndarray,
    weights_test: np.ndarray,
    selection_metric: str = 'p',
    selector_top_k: int = 1,
    selector_lambda: float = 12.0,
):
    L = raw_score_mat.shape[0]
    scores = np.zeros((m, n + m), dtype=float)
    sel_idx_pb = np.empty(len(blocks), dtype=int)
    use_smoothed_selector = int(selector_top_k) > 1
    selector_top_k = min(max(1, int(selector_top_k)), L)
    for bidx, block in enumerate(blocks):
        block_arr = np.asarray(block, dtype=int)
        all_but_block = np.setdiff1d(np.arange(m, dtype=int), block_arr, assume_unique=True)
        bsize = block_arr.size
        nsel = np.zeros(L, dtype=float)
        nsel_bk = np.zeros(L, dtype=float)
        for li in range(L):
            row = raw_score_mat[li]
            n_aug = n + bsize
            aug_scores = np.empty(n_aug, dtype=float)
            aug_scores[:n] = row[:n]
            aug_scores[n:] = row[n + block_arr]
            aug_weights = np.empty(n_aug, dtype=float)
            aug_weights[:n] = weights_ref.astype(float, copy=False)
            aug_weights[n:] = weights_test[block_arr].astype(float, copy=False)

            order_aug = np.argsort(aug_scores)
            S = np.empty(n + m, dtype=float)
            S[:n] = aug_scores[order_aug[:n]]
            S[n:n + bsize] = aug_scores[order_aug[n:]]
            S[n + bsize:] = row[n + all_but_block]
            cw = aug_weights[order_aug[:n]]
            tw = np.concatenate([
                aug_weights[order_aug[n:]],
                weights_test[all_but_block].astype(float, copy=False),
            ])
            if selection_metric == 'p':
                # Primary metric: BH rejections; Tiebreaker: raw p < alpha
                from fcnd.multiple_testing import bh
                pvals = p_function_vector(S, cw, tw, 1.0)
                k, _ = bh(pvals, alpha_fdr)
                nsel[li] = float(k)
                nsel_bk[li] = float(np.sum(pvals < alpha_fdr))
            else:
                from fcnd.multiple_testing import ebh as _ebh
                evals = e_function_vector(alpha_fdr, S, cw, tw)
                k, _ = _ebh(evals, alpha_fdr)
                nsel[li] = float(k)
                nsel_bk[li] = float(np.sum(evals >= (1.0 / alpha_fdr)))
        utility_all = normalized_proxy_utility_py(nsel, nsel_bk, m)
        best = int(np.argmax(utility_all))
        sel_idx_pb[bidx] = best
        if not use_smoothed_selector:
            scores[block_arr, :] = raw_score_mat[best]
            continue

        order = np.argsort(utility_all)
        top_idx = order[-selector_top_k:]
        utility = utility_all[top_idx]
        utility = utility - np.max(utility)
        weights = np.exp(float(selector_lambda) * utility)
        weight_sum = float(np.sum(weights))
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            weights = np.full(selector_top_k, 1.0 / selector_top_k, dtype=float)
        else:
            weights = weights / weight_sum
        smooth_row = np.tensordot(weights, smooth_score_mat[top_idx, :], axes=(0, 0))
        scores[block_arr, :] = smooth_row
    return scores, sel_idx_pb


def subroutine_mdlsel_py(
    m: int,
    n: int,
    blocks,
    alpha_fdr: float,
    score_mat: np.ndarray,
    weights_ref: np.ndarray,
    weights_test: np.ndarray,
    selection_metric: str = 'p',
    selector_top_k: int = 1,
    selector_lambda: float = 12.0,
):
    smooth_source = score_mat
    if int(selector_top_k) > 1:
        smooth_source = neglog_empirical_rank_scores_py(score_mat)
    return subroutine_mdlsel_py_from_pools(
        m,
        n,
        blocks,
        alpha_fdr,
        score_mat,
        smooth_source,
        weights_ref,
        weights_test,
        selection_metric,
        selector_top_k,
        selector_lambda,
    )

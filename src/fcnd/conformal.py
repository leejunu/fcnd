from __future__ import annotations

import numpy as np
from typing import Optional, Sequence, Tuple


def p_from_partition(
    scores: np.ndarray,
    cal_idx: Sequence[int],
    test_idx: Sequence[int],
    *,
    U: float = 1.0,
    cal_weights: Optional[np.ndarray] = None,
    test_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute (possibly weighted) conformal p-values on a partition.

    Parameters
    - scores: shape (n+m,) scores, higher = more anomalous
    - cal_idx: indices for calibration/reference portion
    - test_idx: indices for test portion
    - U: tie/randomization parameter in [0,1]
    - cal_weights: shape (|cal_idx|,), defaults to ones
    - test_weights: shape (|test_idx|,), defaults to ones

    Returns: p-values array of shape (|test_idx|,)
    """
    scores = np.asarray(scores).reshape(-1)
    cal_idx = np.asarray(cal_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    cal = scores[cal_idx]
    test = scores[test_idx]
    n = cal.shape[0]
    m = test.shape[0]

    cw = np.ones(n, dtype=float) if cal_weights is None else np.asarray(cal_weights, dtype=float).reshape(-1)
    tw = np.ones(m, dtype=float) if test_weights is None else np.asarray(test_weights, dtype=float).reshape(-1)
    if cw.shape[0] != n or tw.shape[0] != m:
        raise ValueError("Weight lengths must match |cal_idx| and |test_idx|")

    sum_cw = float(np.sum(cw))
    p = np.empty(m, dtype=float)
    for j in range(m):
        numer = float(np.sum(cw * (cal >= test[j]))) + tw[j] * float(U)
        denom = sum_cw + tw[j]
        p[j] = numer / denom
    return p


def e_from_partition(
    scores: np.ndarray,
    cal_idx: Sequence[int],
    test_idx: Sequence[int],
    alpha: float,
    *,
    cal_weights: Optional[np.ndarray] = None,
    test_weights: Optional[np.ndarray] = None,
    esT: bool = False,
) -> Tuple[np.ndarray, Tuple[float, ...]]:
    """Compute (possibly weighted) conformal e-values on a partition.

    For unweighted case, uses a single global threshold T; returns (e, (T,)).
    For weighted case, computes per-j thresholds T_j; returns (e, (T_1,...,T_m)).
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1)")

    scores = np.asarray(scores).reshape(-1)
    cal_idx = np.asarray(cal_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    cal = scores[cal_idx]
    test = scores[test_idx]
    n = cal.shape[0]
    m = test.shape[0]

    cw = np.ones(n, dtype=float) if cal_weights is None else np.asarray(cal_weights, dtype=float).reshape(-1)
    tw = np.ones(m, dtype=float) if test_weights is None else np.asarray(test_weights, dtype=float).reshape(-1)
    if cw.shape[0] != n or tw.shape[0] != m:
        raise ValueError("Weight lengths must match |cal_idx| and |test_idx|")

    # Unweighted global-threshold case
    if np.allclose(cw, 1.0) and np.allclose(tw, 1.0):
        candidates = np.unique(np.concatenate([cal, test]))
        T = np.inf
        for t in np.sort(candidates):
            cal_cnt = int(np.sum(cal >= t))
            test_cnt = int(np.sum(test >= t))
            lhs = (m / (n + 1.0)) * ((1 + cal_cnt) / max(1, test_cnt))
            if lhs <= alpha or (esT and test_cnt < (1.0 / alpha)):
                T = t
                break
        if not np.isfinite(T):
            return np.zeros(m, dtype=float), (T,)
        denom = 1.0 + float(np.sum(cal >= T))
        e_vals = ((n + 1.0) / denom) * (test >= T).astype(float)
        return e_vals, (T,)

    # Weighted per-j threshold case for weighted exchangeability.
    sum_cw = float(np.sum(cw))
    e_vals = np.zeros(m, dtype=float)
    T_list = []
    combined_scores = np.concatenate([test, cal])
    combined_weights = np.concatenate([tw, cw])
    membership_test = np.concatenate([np.ones(m, dtype=int), np.zeros(n, dtype=int)])
    order = np.argsort(-combined_scores)
    ordered_scores = combined_scores[order]
    ordered_weights = combined_weights[order]
    ordered_is_test = (membership_test[order] == 1)
    cal_seq = np.cumsum(ordered_weights * (~ordered_is_test))
    test_seq = np.cumsum(ordered_is_test.astype(int))
    test_seq[test_seq == 0] = 1

    for j in range(m):
        sum_w_j = sum_cw + tw[j]
        hat_fdrs = (cal_seq + tw[j]) / test_seq * (m / sum_w_j)
        cond = hat_fdrs <= alpha
        if esT:
            cond = cond | (test_seq < (1.0 / alpha))
        ok = np.where(cond)[0]
        if ok.size > 0:
            Tj = ordered_scores[ok[0]]
        else:
            Tj = np.inf
        T_list.append(Tj)
        denom_j = tw[j] + float(np.sum(cw * (cal >= Tj)))
        e_vals[j] = (sum_w_j * float(test[j] >= Tj)) / denom_j if np.isfinite(Tj) else 0.0

    return e_vals, tuple(T_list)


def p_full(
    scores_ref: np.ndarray,
    scores_test: np.ndarray,
    *,
    U: float = 1.0,
    weights_ref: Optional[np.ndarray] = None,
    weights_test: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convenience: p-values for full split [ref; test] inputs."""
    n = len(scores_ref)
    m = len(scores_test)
    scores = np.concatenate([scores_ref, scores_test])
    cal_idx = np.arange(n)
    test_idx = n + np.arange(m)
    return p_from_partition(scores, cal_idx, test_idx, U=U, cal_weights=weights_ref, test_weights=weights_test)


def e_full(
    scores_ref: np.ndarray,
    scores_test: np.ndarray,
    alpha: float,
    *,
    weights_ref: Optional[np.ndarray] = None,
    weights_test: Optional[np.ndarray] = None,
    esT: bool = False,
) -> Tuple[np.ndarray, Tuple[float, ...]]:
    """Convenience: e-values for full split [ref; test] inputs."""
    n = len(scores_ref)
    m = len(scores_test)
    scores = np.concatenate([scores_ref, scores_test])
    cal_idx = np.arange(n)
    test_idx = n + np.arange(m)
    return e_from_partition(
        scores,
        cal_idx,
        test_idx,
        alpha,
        cal_weights=weights_ref,
        test_weights=weights_test,
        esT=esT,
    )

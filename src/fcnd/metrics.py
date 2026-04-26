from __future__ import annotations

import numpy as np
from typing import List, Optional, Sequence, Dict, Any

from fcnd.utils.cc_utils import p_function_static as _p_static, e_function_static as _e_static
from fcnd.multiple_testing import bh as _bh, ebh as _ebh


def fdr_tpr(rej_idx, y_true):
    """Compute FDR and TPR given rejection indices and binary ground truth y_true (1=nonnull)."""
    rej_idx = np.asarray(rej_idx, dtype=int)
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    S = rej_idx.size
    P = int(np.sum(y_true == 1))
    TP = int(np.sum(y_true[rej_idx] == 1)) if S > 0 else 0
    FP = S - TP
    fdr = FP / max(S, 1)
    tpr = TP / max(P, 1)
    return fdr, tpr


def evaluate_score_pool(
    score_pool: np.ndarray,
    weights_ref: np.ndarray,
    weights_test: np.ndarray,
    alpha: float,
    y_test: Sequence[int],
    *,
    metric: str = 'p',
    U: float = 1.0,
    learner_keys: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Evaluate each learner row in a score pool via BH/eBH and compute FDR/TPR.

    Parameters
    - score_pool: shape (L, n+m) learner scores over [ref; test]
    - weights_ref: shape (n,)
    - weights_test: shape (m,)
    - alpha: FDR level
    - y_test: binary labels for test set (1=nonnull)
    - metric: 'p' -> BH on p-values; 'e' -> eBH on e-values
    - U: randomization for p-values
    - learner_keys: optional names for each row; defaults to f'row:{i}'

    Returns: list of dicts with keys: key, k, FDR, TPR, metric
    """
    score_pool = np.asarray(score_pool, dtype=float)
    wr = np.asarray(weights_ref, dtype=float).reshape(-1)
    wt = np.asarray(weights_test, dtype=float).reshape(-1)
    y_test = np.asarray(y_test, dtype=int).reshape(-1)
    L, total = score_pool.shape
    n = wr.size
    m = wt.size
    if total != n + m:
        raise ValueError('score_pool columns must equal n+m')
    if y_test.size != m:
        raise ValueError('y_test length must equal m')
    keys = list(learner_keys) if learner_keys is not None else [f'row:{i}' for i in range(L)]
    out: List[Dict[str, Any]] = []
    for li in range(L):
        scores = score_pool[li]
        # Build S with identical rows to satisfy static kernels
        S = np.empty((m, n + m), dtype=float)
        S[0, :n] = scores[:n]
        S[0, n:] = scores[n:]
        if m > 1:
            S[1:, :] = S[0, :]
        if metric == 'p':
            pvals = _p_static(S, wr, wt, U=U)
            k, rej = _bh(pvals, alpha)
        elif metric == 'e':
            evals = _e_static(alpha, S, wr, wt)
            k, rej = _ebh(evals, alpha)
        else:
            raise ValueError("metric must be 'p' or 'e'")
        fdr, tpr = fdr_tpr(rej, y_test)
        out.append({'key': keys[li], 'k': int(k), 'FDR': float(fdr), 'TPR': float(tpr), 'metric': metric})
    return out

from __future__ import annotations

import numpy as np
from typing import Tuple


def bh(p: np.ndarray, alpha: float) -> Tuple[int, np.ndarray]:
    """Benjamini-Hochberg procedure.

    Parameters:
    - p: array-like of p-values (shape (m,))
    - alpha: target FDR level in (0, 1)

    Returns:
    - k: number of rejections
    - rej_idx: indices of rejected hypotheses in original order
    """
    p = np.asarray(p, dtype=float)
    if p.ndim != 1:
        p = p.ravel()
    m = p.size
    if m == 0:
        return 0, np.array([], dtype=int)
    order = np.argsort(p)
    ps = p[order]
    thresh = alpha * (np.arange(1, m + 1) / m)
    mask = ps <= thresh
    if not np.any(mask):
        return 0, np.array([], dtype=int)
    k = int(np.max(np.nonzero(mask)[0]) + 1)
    cutoff = thresh[k - 1]
    rej_idx = np.where(p <= cutoff)[0]
    return k, rej_idx


def ebh(e: np.ndarray, alpha: float) -> Tuple[int, np.ndarray]:
    """e-BH procedure.

    Parameters:
    - e: array-like of e-values (shape (m,))
    - alpha: target FDR level in (0, 1)

    Returns:
    - k: number of rejections
    - rej_idx: indices of rejected hypotheses in original order
    """
    e = np.asarray(e, dtype=float)
    if e.ndim != 1:
        e = e.ravel()
    m = e.size
    if m == 0:
        return 0, np.array([], dtype=int)
    es = np.sort(e)[::-1]
    khat = 0
    for k in range(m, 0, -1):
        if es[k - 1] >= m / (alpha * k):
            khat = k
            break
    if khat == 0:
        return 0, np.array([], dtype=int)
    thr = m / (alpha * khat)
    rej_idx = np.where(e >= thr)[0]
    return khat, rej_idx


def evaluate(rejections, H_1):
    """
    Given the rejections and true nonnulls (H_1), returns the FDP and power of the procedure.
    """
    m_1 = len(H_1)
    R = len(rejections)
    
    if R == 0:
        return {'fdp': 0, 'power': 0}
    
    td = len(set(rejections) & set(H_1))
    return {'fdp': 1 - td/max(R,1), 'power': td/max(m_1,1)}
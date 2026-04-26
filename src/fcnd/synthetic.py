from __future__ import annotations

from typing import Optional

import numpy as np


def _rng(random_state: Optional[int | np.random.Generator] = None) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def generate_wset(*, size: int = 5, dims: int = 50, seed: Optional[int] = None, random_state: Optional[int | np.random.Generator] = None) -> np.ndarray:
    """Generate anchor means on a hypercube for synthetic examples."""
    rng = _rng(seed if random_state is None else random_state)
    return rng.uniform(low=-3.0, high=3.0, size=(size, dims))


def gen_data(Wset: np.ndarray, n: int, a: float, *, dims: Optional[int] = None, random_state: Optional[int | np.random.Generator] = None) -> np.ndarray:
    """Generate synthetic inlier or shifted samples used in examples."""
    rng = _rng(random_state)
    Wset = np.asarray(Wset, dtype=float)
    if dims is None:
        dims = int(Wset.shape[1])
    idx = rng.choice(Wset.shape[0], size=int(n))
    Wi = Wset[idx, :dims]
    Vi = rng.normal(size=int(n) * dims).reshape((int(n), dims))
    return (np.sqrt(float(a)) * Vi) + Wi


def sigmoid_weights(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Density-ratio-style weights used in weighted examples."""
    X = np.asarray(X, dtype=float)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if theta.size != X.shape[1]:
        raise ValueError("theta length must equal X.shape[1]")
    z = X @ theta
    return 1.0 / (1.0 + np.exp(-z))


def gen_data_weighted(Wset: np.ndarray, n: int, a: float, theta: np.ndarray, *, dims: Optional[int] = None, random_state: Optional[int | np.random.Generator] = None) -> np.ndarray:
    """Generate samples accepted with probability ``sigmoid(X @ theta)``."""
    rng = _rng(random_state)
    Wset = np.asarray(Wset, dtype=float)
    if dims is None:
        dims = int(Wset.shape[1])
    X = np.empty((0, dims))
    while X.shape[0] < int(n):
        batch = max(int(n), 64)
        Xi = gen_data(Wset, batch, a, dims=dims, random_state=rng)
        probs = sigmoid_weights(Xi, theta[:dims])
        sel = rng.binomial(n=1, p=probs).astype(bool)
        if np.any(sel):
            X = np.vstack([X, Xi[sel, :]])
    return X[: int(n), :]

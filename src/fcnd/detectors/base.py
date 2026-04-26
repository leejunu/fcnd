from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class BaseND(ABC):
    """Base class for conformal novelty-detection estimators."""

    def _ensure_loaded(self) -> None:
        if not getattr(self, "datasets_loaded_", False):
            raise RuntimeError("Datasets not loaded. Call load_dataset first.")

    @abstractmethod
    def load_dataset(self, X1: Any, X2: Any, X3: Optional[Any] = None) -> None:
        ...

    @abstractmethod
    def score_units(self, **kwargs: Any):
        ...

    # Detect numba availability for default behavior
    try:
        import numba as _nb  # noqa: F401
        _NUMBA_AVAILABLE = True
    except Exception:  # pragma: no cover
        _NUMBA_AVAILABLE = False

    def __init__(self, *, random_state: Optional[int] = None, use_numba: Optional[bool] = None) -> None:
        if isinstance(random_state, np.random.Generator):
            self.rng = random_state
        elif random_state is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(random_state)

        self.datasets_loaded_ = False
        self.is_fitted_ = False
        self.m_ = None
        self.n_ = None
        self.p_values_ = None
        self.e_values_ = None
        self.threshold_ = None
        self.scores_ = None
        # Whether to use numba-backed routines by default
        self.use_numba: bool = True if use_numba is None else bool(use_numba)
        if self.use_numba and not BaseND._NUMBA_AVAILABLE:
            raise ImportError("numba is required when use_numba=True. Install fcnd with its default dependencies or pass use_numba=False for slow Python fallbacks where available.")

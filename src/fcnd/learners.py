from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class BaseLearner(ABC):
    """Abstract interface for learners used by FCND/SCND/MSFCND.

    Implementations must provide reset(), fit(X, **kwargs), score(X) -> 1D array.
    """

    fitted_: bool = False

    @abstractmethod
    def reset(self) -> "BaseLearner":
        ...

    @abstractmethod
    def fit(self, X: np.ndarray, **kwargs: Any) -> "BaseLearner":
        ...

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        ...


class IsoForestLearner(BaseLearner):
    def __init__(self, *, random_state: Optional[int] = None, **kwargs: Any) -> None:
        from sklearn.ensemble import IsolationForest as _IF

        self._params = dict(random_state=random_state, **kwargs)
        self._IF = _IF
        self.model = None
        self.fitted_ = False

    def reset(self) -> "IsoForestLearner":
        self.model = self._IF(**self._params)
        self.fitted_ = False
        return self

    def fit(self, X: np.ndarray, **kwargs: Any) -> "IsoForestLearner":
        if self.model is None:
            self.reset()
        self.model.set_params(**kwargs)
        self.model.fit(X)
        self.fitted_ = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call reset()/fit() before score().")
        # Higher score = more anomalous; IsolationForest gives negative anomaly scores
        s = -self.model.score_samples(X)
        return np.asarray(s, dtype=float).reshape(-1)


class OneClassSVMLearner(BaseLearner):
    def __init__(self, **kwargs: Any) -> None:
        from sklearn.svm import OneClassSVM as _OCSVM

        self._params = dict(**kwargs)
        self._OCSVM = _OCSVM
        self.model = None
        self.fitted_ = False

    def reset(self) -> "OneClassSVMLearner":
        self.model = self._OCSVM(**self._params)
        self.fitted_ = False
        return self

    def fit(self, X: np.ndarray, **kwargs: Any) -> "OneClassSVMLearner":
        if self.model is None:
            self.reset()
        self.model.set_params(**kwargs)
        self.model.fit(X)
        self.fitted_ = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call reset()/fit() before score().")
        # Higher score = more anomalous; OCSVM decision_function: negatives ~ inliers
        s = -self.model.decision_function(X)
        return np.asarray(s, dtype=float).reshape(-1)


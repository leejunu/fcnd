from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .base import BaseND
from fcnd.learners import BaseLearner
from fcnd.conformal import p_full, e_full
from fcnd.utils.cc_utils import p_function_static as _p_static
from fcnd.utils.cc_utils import e_function_static as _e_static


class FCND(BaseND):
    """Full Conformal Novelty Detector.

    load_dataset(X_ref, X_test). Optionally supports leave-one-out training.
    """

    def __init__(self, learner: BaseLearner, *, leave_one_out: bool = False, random_state: Optional[int] = None, use_numba: Optional[bool] = None) -> None:
        super().__init__(random_state=random_state, use_numba=use_numba)
        if not isinstance(learner, BaseLearner):
            raise TypeError("learner must be an instance of BaseLearner")
        self.learner = learner
        self.leave_one_out = bool(leave_one_out)

    def load_dataset(self, X_ref: Any, X_test: Any) -> None:
        self.X_ref_ = X_ref
        self.X_test_ = X_test
        self.n = int(len(X_ref))
        self.m = int(len(X_test))
        self.weights_ref_ = np.ones(self.n, dtype=float)
        self.weights_test_ = np.ones(self.m, dtype=float)
        self.datasets_loaded_ = True

    def load_weights(self, weights_ref: np.ndarray, weights_test: np.ndarray) -> None:
        self._ensure_loaded()
        wr = np.asarray(weights_ref, dtype=float).reshape(-1)
        wt = np.asarray(weights_test, dtype=float).reshape(-1)
        if wr.shape[0] != self.n or wt.shape[0] != self.m:
            raise ValueError('weights_ref length must equal n and weights_test length must equal m')
        self.weights_ref_ = wr
        self.weights_test_ = wt

    def score_units(self, **kwargs: Any):
        self._ensure_loaded()
        X_all = np.vstack([self.X_ref_, self.X_test_])
        n, m = self.n, self.m
        N = n + m
        scores = np.empty(N, dtype=float)
        if self.leave_one_out:
            mask = np.ones(N, dtype=bool)
            self.learner.reset()
            for idx in range(N):
                mask[idx] = False
                self.learner.reset().fit(X_all[mask], **kwargs)
                scores[idx] = float(self.learner.score(X_all[idx:idx+1])[0])
                mask[idx] = True
        else:
            self.learner.reset().fit(X_all, **kwargs)
            scores[:] = np.asarray(self.learner.score(X_all), dtype=float).reshape(-1)
        self.scores_ = scores
        return scores

    def make_p(self, scores: Optional[np.ndarray] = None, *, U: float = 1.0):
        self._ensure_loaded()
        use_numba = self.use_numba
        n, m = self.n, self.m
        if scores is None:
            if getattr(self, 'scores_', None) is None:
                raise RuntimeError('No scores available. Call score_units first or pass scores.')
            scores = self.scores_
        scores = np.asarray(scores).reshape(-1)
        if scores.shape[0] != n + m:
            raise ValueError(f'Expected scores of length {n+m}, got {scores.shape[0]}')
        if use_numba:
            # Build (m, n+m) matrix with identical rows
            S = np.empty((m, n + m), dtype=float)
            S[0, :n] = scores[:n]
            S[0, n:] = scores[n:]
            if m > 1:
                S[1:, :] = S[0, :]
            p_vals = _p_static(S, self.weights_ref_, self.weights_test_, U=U)
        else:
            p_vals = p_full(scores[:n], scores[n:], U=U, weights_ref=self.weights_ref_, weights_test=self.weights_test_)
        self.p_values_ = p_vals
        return p_vals

    def make_e(
        self,
        alpha: Optional[float] = None,
        scores: Optional[np.ndarray] = None,
        *,
        alpha_e: Optional[float] = None,
        esT: bool = False,
    ):
        self._ensure_loaded()
        use_numba = self.use_numba
        n, m = self.n, self.m
        if alpha_e is not None:
            if alpha is not None and float(alpha) != float(alpha_e):
                raise ValueError('Provide only one of alpha or alpha_e, or make them equal.')
            alpha = float(alpha_e)
        if alpha is None:
            raise ValueError('alpha must be provided')
        if not (0 < alpha < 1):
            raise ValueError('alpha must be in (0,1)')
        if scores is None:
            if getattr(self, 'scores_', None) is None:
                raise RuntimeError('No scores available. Call score_units first or pass scores.')
            scores = self.scores_
        scores = np.asarray(scores).reshape(-1)
        if scores.shape[0] != n + m:
            raise ValueError(f'Expected scores of length {n+m}, got {scores.shape[0]}')
        if use_numba:
            S = np.empty((m, n + m), dtype=float)
            S[0, :n] = scores[:n]
            S[0, n:] = scores[n:]
            if m > 1:
                S[1:, :] = S[0, :]
            e_vals = _e_static(alpha, S, self.weights_ref_, self.weights_test_, esT)
            T = None
        else:
            e_vals, T = e_full(
                scores[:n],
                scores[n:],
                alpha,
                weights_ref=self.weights_ref_,
                weights_test=self.weights_test_,
                esT=esT,
            )
        self.e_values_ = e_vals
        self.threshold_ = T
        return e_vals, T

    def detect(self, X_ref: Any, X_test: Any, *, alpha: float, method: str = "ebh", weights_ref: Optional[np.ndarray] = None, weights_test: Optional[np.ndarray] = None, **fit_kwargs: Any):
        """Run scoring, conformal calibration, and BH/e-BH selection.

        Parameters
        ----------
        X_ref, X_test:
            Reference inlier data and test data.
        alpha:
            Target FDR level.
        method:
            ``"ebh"`` for conformal e-values with e-BH, or ``"bh"`` for
            conformal p-values with BH. The e-BH option is the default
            finite-sample-valid procedure under arbitrary dependence.
        weights_ref, weights_test:
            Optional density-ratio weights for weighted exchangeability.
        **fit_kwargs:
            Extra keyword arguments passed to the wrapped learner's fit method.
        """
        from fcnd.results import DetectionResult
        from fcnd.multiple_testing import bh, ebh

        self.load_dataset(X_ref, X_test)
        if weights_ref is not None or weights_test is not None:
            if weights_ref is None or weights_test is None:
                raise ValueError("Provide both weights_ref and weights_test, or neither.")
            self.load_weights(weights_ref, weights_test)
        scores = self.score_units(**fit_kwargs)
        method_l = method.lower()
        if method_l == "ebh":
            values, _ = self.make_e(alpha)
            _, rejections = ebh(values, alpha)
        elif method_l == "bh":
            values = self.make_p()
            _, rejections = bh(values, alpha)
        else:
            raise ValueError("method must be 'ebh' or 'bh'")
        return DetectionResult(
            rejections=rejections,
            values=values,
            scores=scores,
            method=method_l,
            alpha=float(alpha),
            detector=self.__class__.__name__,
        )


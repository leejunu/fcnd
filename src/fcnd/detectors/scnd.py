from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .base import BaseND
from fcnd.learners import BaseLearner
from fcnd.utils.cc_utils import p_function_static as _p_static
from fcnd.utils.cc_utils import e_function_static as _e_static
from fcnd.conformal import p_full as _p_full_py, e_full as _e_full_py


class SCND(BaseND):
    """Split Conformal Novelty Detector.

    Trains the learner on X_train only. Scores are produced for X_calib and X_test,
    returned as a concatenated vector [scores_calib; scores_test] of length n + m.
    """

    def __init__(self, learner: BaseLearner, *, random_state: Optional[int] = None, use_numba: Optional[bool] = None) -> None:
        super().__init__(random_state=random_state, use_numba=use_numba)
        if not isinstance(learner, BaseLearner):
            raise TypeError("learner must be an instance of BaseLearner")
        self.learner = learner

    def load_dataset(self, X_train: Any, X_calib: Any, X_test: Any) -> None:
        self.X_train_ = X_train
        self.X_calib_ = X_calib
        self.X_test_ = X_test
        self.n = len(X_calib)
        self.m = len(X_test)
        self.n_train = len(X_train)
        self.weights_calib_ = np.ones(self.n, dtype=float)
        self.weights_test_ = np.ones(self.m, dtype=float)
        self.datasets_loaded_ = True

    def load_weights(self, calib_weights: np.ndarray, test_weights: np.ndarray) -> None:
        self._ensure_loaded()
        cw = np.asarray(calib_weights, dtype=float).reshape(-1)
        tw = np.asarray(test_weights, dtype=float).reshape(-1)
        if cw.shape[0] != self.n or tw.shape[0] != self.m:
            raise ValueError('calib_weights length must equal n and test_weights length must equal m')
        self.weights_calib_ = cw
        self.weights_test_ = tw

    def score_units(self, *, retrain: bool = False, **kwargs: Any):
        self._ensure_loaded()
        X_train, X_calib, X_test = self.X_train_, self.X_calib_, self.X_test_
        need_fit = retrain or (not hasattr(self.learner, 'fitted_') or (not getattr(self.learner, 'fitted_', False)))
        if need_fit:
            self.learner.reset().fit(X_train, **kwargs)
        scores_cal = self.learner.score(X_calib)
        scores_test = self.learner.score(X_test)
        scores = np.concatenate([scores_cal, scores_test]).astype(float, copy=False)
        self.scores_ = scores
        return scores

    def make_p(self, scores: Optional[np.ndarray] = None, *, U: float = 1.0):
        self._ensure_loaded()
        use_numba = self.use_numba
        n, m = self.n, self.m
        if scores is None:
            if getattr(self, 'scores_', None) is None:
                raise RuntimeError('No scores provided and self.scores_ is None. Call score_units first or pass scores.')
            scores = self.scores_
        scores = np.asarray(scores).reshape(-1)
        if scores.shape[0] != n + m:
            raise ValueError(f'Expected scores of length {n+m}, got {scores.shape[0]}')
        cal_scores = scores[:n]
        test_scores = scores[n:]
        cw = self.weights_calib_
        tw = self.weights_test_
        if use_numba:
            S = np.empty((m, n + m), dtype=float)
            S[0, :n] = cal_scores
            S[0, n:] = test_scores
            if m > 1:
                S[1:, :] = S[0, :]
            p_vals = _p_static(S, cw, tw, U=U)
        else:
            p_vals = _p_full_py(cal_scores, test_scores, U=U, weights_ref=cw, weights_test=tw)
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
                raise RuntimeError('No scores provided and self.scores_ is None. Call score_units first or pass scores.')
            scores = self.scores_
        scores = np.asarray(scores).reshape(-1)
        if scores.shape[0] != n + m:
            raise ValueError(f'Expected scores of length {n+m}, got {scores.shape[0]}')
        cal_scores = scores[:n]
        test_scores = scores[n:]
        cw = self.weights_calib_
        tw = self.weights_test_
        if use_numba:
            S = np.empty((m, n + m), dtype=float)
            S[0, :n] = cal_scores
            S[0, n:] = test_scores
            if m > 1:
                S[1:, :] = S[0, :]
            e_vals = _e_static(alpha, S, cw, tw, esT)
        else:
            e_vals, _ = _e_full_py(cal_scores, test_scores, alpha, weights_ref=cw, weights_test=tw, esT=esT)
        self.e_values_ = e_vals
        self.threshold_ = None
        return e_vals, None

    def detect(self, X_train: Any, X_calib: Any, X_test: Any, *, alpha: float, method: str = "ebh", calib_weights: Optional[np.ndarray] = None, test_weights: Optional[np.ndarray] = None, **fit_kwargs: Any):
        """Run split-conformal novelty detection.

        ``X_train`` is used only to fit the learner, ``X_calib`` calibrates
        conformal values, and ``X_test`` contains the hypotheses to test.
        """
        from fcnd.results import DetectionResult
        from fcnd.multiple_testing import bh, ebh

        self.load_dataset(X_train, X_calib, X_test)
        if calib_weights is not None or test_weights is not None:
            if calib_weights is None or test_weights is None:
                raise ValueError("Provide both calib_weights and test_weights, or neither.")
            self.load_weights(calib_weights, test_weights)
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


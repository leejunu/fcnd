from __future__ import annotations

from typing import Any, Optional, List, Sequence
import warnings

import numpy as np

from .base import BaseND
from fcnd.learners import BaseLearner
from fcnd.model_selection.subroutines import subroutine_mdlsel_numba, subroutine_mdlsel_py
from fcnd.utils.cc_utils import ( 
    e_function_vector as _e_function_vector,
    p_function_vector as _p_function_vector 
)
from fcnd.conformal import p_full as _p_full_py, e_full as _e_full_py


class MSFCND(BaseND):
    """Model-selection full-conformal novelty detector.

    Produces per-learner scores over ``[X_ref; X_test]`` and selects or ensembles
    candidate score rows block-by-block using the conditional proxy utilities from
    the full conformal model-selection construction. Blocks must partition the
    test indices. Numba kernels are used by default, with Python fallbacks when
    ``use_numba=False`` or when a numba specialization fails.
    """

    def __init__(
        self,
        learners: Any,
        *,
        alpha: float = 0.1,
        selection_metric: str = 'p',  # 'p' or 'e'
        leave_one_out: bool = False,
        training_modes: Optional[Any] = None,
        blocks: Optional[List[Sequence[int]]] = None,
        K: Optional[int] = None,
        selector_top_k: int = 1,
        selector_lambda: float = 12.0,
        random_state: Optional[int] = None,
        use_numba: Optional[bool] = None,
    ) -> None:
        super().__init__(random_state=random_state, use_numba=use_numba)
        if isinstance(learners, dict):
            if len(learners) == 0:
                raise ValueError('learners dict must be non-empty')
            for k, ln in learners.items():
                if not isinstance(ln, BaseLearner):
                    raise TypeError('All learners must implement BaseLearner')
            self.learners_keys = list(learners.keys())
            self.learners = list(learners.values())
        elif isinstance(learners, (list, tuple)):
            if len(learners) == 0:
                raise ValueError('learners must be a non-empty list/tuple of BaseLearner')
            for ln in learners:
                if not isinstance(ln, BaseLearner):
                    raise TypeError('All learners must implement BaseLearner')
            self.learners = list(learners)
            self.learners_keys = [f"{ln.__class__.__name__}:{i}" for i, ln in enumerate(self.learners)]
        else:
            raise TypeError('learners must be a dict or a list/tuple of BaseLearner')

        if selection_metric not in ('p', 'e'):
            raise ValueError("selection_metric must be 'e' or 'p'")
        if not (0 < alpha < 1):
            raise ValueError('alpha must be in (0,1)')
        self.alpha = float(alpha)
        self.selection_metric = selection_metric
        self.leave_one_out = bool(leave_one_out)
        self.training_modes = training_modes
        self._user_blocks = None if blocks is None else [list(b) for b in blocks]
        self._user_K = K
        self.selector_top_k = max(1, int(selector_top_k))
        self.selector_lambda = float(selector_lambda)

        # outputs
        self.score_matrix_ = None
        self.selected_scores_matrix_ = None
        self.blocks_ = None
        self.selected_model_per_block_idx_ = None
        self.selected_model_per_block_keys_ = None
        self.p_values_ = None
        self.e_values_ = None
        self.thresholds_ = None
        # expanded metadata for (learner, mode)
        self.learners_modes_: Optional[List[tuple]] = None
        self.learners_keys_expanded_: Optional[List[str]] = None

    def load_dataset(self, X_ref: Any, X_test: Any) -> None:
        self.X_ref_ = X_ref
        self.X_test_ = X_test
        self.n = len(X_ref)
        self.m = len(X_test)
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

    def _build_blocks(self) -> List[List[int]]:
        m = self.m
        K = self._user_K if self._user_K is not None else m
        if K <= 0 or K > m:
            K = m
        sizes = [m // K + (1 if r < m % K else 0) for r in range(K)]
        blocks, start = [], 0
        for sz in sizes:
            blocks.append(list(range(start, start + sz)))
            start += sz
        return blocks

    def _resolve_training_modes(self) -> List[List[str]]:
        """Resolve per-learner modes list based on training_modes and leave_one_out.

        Returns a list of lists `modes_per_learner` with entries from {'is','loo'}.
        """
        L = len(self.learners)
        # default global mode
        if self.training_modes is None:
            default = ['loo'] if self.leave_one_out else ['is']
            return [default[:] for _ in range(L)]
        # global string
        if isinstance(self.training_modes, str):
            tm = self.training_modes.lower()
            if tm == 'both':
                default = ['is', 'loo']
            elif tm in ('is', 'loo'):
                default = [tm]
            else:
                raise ValueError("training_modes must be one of None|'is'|'loo'|'both' or a mapping")
            return [default[:] for _ in range(L)]
        # mapping per learner key
        if isinstance(self.training_modes, dict):
            modes_per = []
            for k in self.learners_keys:
                v = self.training_modes.get(k, None)
                if v is None:
                    vlist = ['loo'] if self.leave_one_out else ['is']
                elif isinstance(v, str):
                    if v.lower() == 'both':
                        vlist = ['is', 'loo']
                    elif v.lower() in ('is', 'loo'):
                        vlist = [v.lower()]
                    else:
                        raise ValueError(f"Invalid training mode '{v}' for learner {k}")
                else:
                    # assume sequence
                    vlist = [m.lower() for m in v]
                    for m in vlist:
                        if m not in ('is','loo'):
                            raise ValueError(f"Invalid training mode '{m}' for learner {k}")
                modes_per.append(vlist)
            return modes_per
        raise ValueError("training_modes must be None, str, or dict")

    def score_units(self, **kwargs: Any):
        self._ensure_loaded()
        X_all = np.vstack([self.X_ref_, self.X_test_])
        n, m = self.n, self.m
        N = n + m
        # resolve per-learner modes and build expanded rows
        modes_per = self._resolve_training_modes()
        expanded: list[tuple[int,str]] = []
        expanded_keys: list[str] = []
        for li, modes in enumerate(modes_per):
            for mode in modes:
                expanded.append((li, mode))
                # store base learner key only; mode tag added later in uppercase
                expanded_keys.append(self.learners_keys[li])
        L_exp = len(expanded)
        scores_mat = np.empty((L_exp, N), dtype=float)
        # compute scores per (learner, mode)
        for ei, (li, mode) in enumerate(expanded):
            learner = self.learners[li]
            if mode == 'is':
                learner.reset().fit(X_all, **kwargs)
                scores_mat[ei, :] = np.asarray(learner.score(X_all), dtype=float).reshape(-1)
            else:  # 'loo'
                mask = np.ones(N, dtype=bool)
                for idx in range(N):
                    mask[idx] = False
                    learner.reset().fit(X_all[mask], **kwargs)
                    scores_mat[ei, idx] = float(learner.score(X_all[idx:idx+1])[0])
                    mask[idx] = True
        self.score_matrix_ = scores_mat
        self.learners_modes_ = expanded
        # Public-facing expanded keys include uppercase mode tags ('IS','LOO')
        self.learners_keys_expanded_ = [f"{k}:{mode.upper()}" for k, mode in zip(expanded_keys, [mode for (_, mode) in expanded])]
        return scores_mat

    def select_models(self, *, alpha: Optional[float] = None, selection_metric: Optional[str] = None, blocks: Optional[List[Sequence[int]]] = None):
        self._ensure_loaded()
        if self.score_matrix_ is None:
            raise RuntimeError('Call score_units() to build per-learner scores first.')
        n, m = self.n, self.m
        # determine blocks
        if blocks is not None:
            flat = sorted([int(x) for b in blocks for x in b])
            if flat != list(range(m)):
                raise ValueError('blocks must partition the test indices [0..m-1] without overlap')
            self.blocks_ = [list(b) for b in blocks]
        elif self.blocks_ is None:
            self.blocks_ = self._build_blocks()
        blocks = self.blocks_
        alpha = self.alpha if alpha is None else alpha
        metric = self.selection_metric if selection_metric is None else selection_metric
        metric_flag = 0 if metric == 'p' else 1
        use_numba = self.use_numba
        if use_numba:
            try:
                from numba.typed import List as _NList
                nb_blocks = _NList()
                for b in blocks:
                    nb_blocks.append(np.array(b, dtype=np.int64))
                selected_scores, sel_idx_pb = subroutine_mdlsel_numba(
                    m, n, nb_blocks, alpha,
                    np.ascontiguousarray(self.score_matrix_, dtype=np.float64),
                    np.ascontiguousarray(self.weights_ref_, dtype=np.float64),
                    np.ascontiguousarray(self.weights_test_, dtype=np.float64),
                    metric_flag,
                    self.selector_top_k,
                    self.selector_lambda,
                )
                self.selected_scores_matrix_ = selected_scores
                selections_idx = [int(sel_idx_pb[bidx]) for bidx in range(len(blocks))]
                self.selected_model_per_block_idx_ = selections_idx
                # Map to expanded keys if available
                keys_src = self.learners_keys_expanded_ if self.learners_keys_expanded_ is not None else self.learners_keys
                self.selected_model_per_block_keys_ = [keys_src[i] for i in selections_idx]
                return self.selected_model_per_block_keys_
            except Exception as exc:
                warnings.warn(
                    f"numba model-selection backend failed; falling back to Python backend: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        selected_scores, sel_idx_pb = subroutine_mdlsel_py(
            m, n, blocks, alpha, self.score_matrix_, self.weights_ref_, self.weights_test_, metric,
            self.selector_top_k, self.selector_lambda,
        )
        self.selected_scores_matrix_ = selected_scores
        selections_idx = [int(sel_idx_pb[bidx]) for bidx in range(len(blocks))]
        self.selected_model_per_block_idx_ = selections_idx
        keys_src = self.learners_keys_expanded_ if self.learners_keys_expanded_ is not None else self.learners_keys
        self.selected_model_per_block_keys_ = [keys_src[i] for i in selections_idx]
        return self.selected_model_per_block_keys_

    def make_p(self, scores_matrix: Optional[np.ndarray] = None, *, U: float = 1.0):
        self._ensure_loaded()
        use_numba = self.use_numba
        if self.selected_model_per_block_idx_ is None:
            raise RuntimeError('Call select_models() before make_p().')
        if scores_matrix is None:
            if self.selected_scores_matrix_ is not None:
                scores_matrix = self.selected_scores_matrix_
            elif self.score_matrix_ is None:
                raise RuntimeError('No scores available. Call score_units first or pass scores_matrix.')
            else:
                scores_matrix = self.score_matrix_
        n, m = self.n, self.m
        if self.blocks_ is None:
            raise RuntimeError('No blocks set. Call select_models(blocks=...) before make_p().')
        p_out = np.empty(m, dtype=float)
        for bidx, b in enumerate(self.blocks_):
            b_arr = np.array(b, dtype=int)
            if scores_matrix.shape[0] == m:
                scores = scores_matrix[int(b_arr[0])]
            else:
                li = self.selected_model_per_block_idx_[bidx]
                scores = scores_matrix[li]
            if use_numba:
                p_block = _p_function_vector(scores, self.weights_ref_, self.weights_test_)
            else:
                p_block = _p_full_py(scores[:n], scores[n:], weights_ref=self.weights_ref_, weights_test=self.weights_test_)
            p_out[b_arr] = p_block[b_arr]
        self.p_values_ = p_out
        return p_out

    def make_e(
        self,
        alpha: Optional[float] = None,
        scores_matrix: Optional[np.ndarray] = None,
        *,
        alpha_e: Optional[float] = None,
        esT: bool = False,
    ):
        self._ensure_loaded()
        use_numba = self.use_numba
        if self.selected_model_per_block_idx_ is None:
            raise RuntimeError('Call select_models() before make_e().')
        if scores_matrix is None:
            if self.selected_scores_matrix_ is not None:
                scores_matrix = self.selected_scores_matrix_
            elif self.score_matrix_ is None:
                raise RuntimeError('No scores available. Call score_units first or pass scores_matrix.')
            else:
                scores_matrix = self.score_matrix_
        if alpha_e is not None:
            if alpha is not None and float(alpha) != float(alpha_e):
                raise ValueError('Provide only one of alpha or alpha_e, or make them equal.')
            alpha = float(alpha_e)
        alpha = self.alpha if alpha is None else alpha
        n, m = self.n, self.m
        if self.blocks_ is None:
            raise RuntimeError('No blocks set. Call select_models(blocks=...) before make_e().')
        e_out = np.zeros(m, dtype=float)
        # per-block e-values using reference-only calibration
        for bidx, b in enumerate(self.blocks_):
            b_arr = np.array(b, dtype=int) 
            if scores_matrix.shape[0] == m:
                scores = scores_matrix[int(b_arr[0])]
            else:
                li = self.selected_model_per_block_idx_[bidx]
                scores = scores_matrix[li]
            if use_numba:
                e_block = _e_function_vector(alpha, scores, self.weights_ref_, self.weights_test_, esT) 
            else:
                e_block, _ = _e_full_py(
                    scores[:n],
                    scores[n:],
                    alpha,
                    weights_ref=self.weights_ref_,
                    weights_test=self.weights_test_,
                    esT=esT,
                )
            e_out[b_arr] = e_block[b_arr]   # make the e-values, block by block
        self.e_values_ = e_out
        self.thresholds_ = None
        return e_out, None

    def detect(self, X_ref: Any, X_test: Any, *, method: str = "ebh", weights_ref: Optional[np.ndarray] = None, weights_test: Optional[np.ndarray] = None, **fit_kwargs: Any):
        """Run model-selection full-conformal novelty detection.

        The detector scores all candidate learners, selects or ensembles scores
        block-by-block using the configured selector, and then applies BH or
        e-BH to the selected conformal values.
        """
        from fcnd.results import DetectionResult
        from fcnd.multiple_testing import bh, ebh

        self.load_dataset(X_ref, X_test)
        if weights_ref is not None or weights_test is not None:
            if weights_ref is None or weights_test is None:
                raise ValueError("Provide both weights_ref and weights_test, or neither.")
            self.load_weights(weights_ref, weights_test)
        score_pool = self.score_units(**fit_kwargs)
        selected = self.select_models()
        method_l = method.lower()
        if method_l == "ebh":
            values, _ = self.make_e(self.alpha)
            _, rejections = ebh(values, self.alpha)
        elif method_l == "bh":
            values = self.make_p()
            _, rejections = bh(values, self.alpha)
        else:
            raise ValueError("method must be 'ebh' or 'bh'")
        return DetectionResult(
            rejections=rejections,
            values=values,
            scores=score_pool,
            method=method_l,
            alpha=float(self.alpha),
            detector=self.__class__.__name__,
            metadata={"selected_models": selected, "blocks": self.blocks_},
        )


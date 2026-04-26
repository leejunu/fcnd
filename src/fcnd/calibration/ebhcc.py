from __future__ import annotations

from collections import defaultdict
import time

import numpy as np
from numba import jit
from numba.typed import List as _NList

from fcnd.model_selection.subroutines import (
    neglog_empirical_rank_scores_numba,
    neglog_empirical_rank_scores_py,
    subroutine_mdlsel_numba,
    subroutine_mdlsel_numba_from_pools,
    subroutine_mdlsel_py,
    subroutine_mdlsel_py_from_pools,
)
from fcnd.utils.cc_utils import (
    p_function_static as _p_static,
    e_function_static as _e_static,
    e_thres_static as _T_static,
    p_function_vector as _p_vec,
    e_function_vector as _e_vec,
    e_thres_vector as _T_vec,
    eBH_vector,
    p_value_j,
    pBH_threshold,
    hatR_p_combination,
    pBH_threshold_vector,
    hatR_p_combination_vector,
)


def _build_strata(p_values: np.ndarray, n: int) -> defaultdict[int, list[int]]:
    p_norm = np.rint(np.asarray(p_values, dtype=float) * (n + 1)).astype(np.int64)
    order = np.argsort(p_norm)
    strata: defaultdict[int, list[int]] = defaultdict(list)
    for idx in order:
        strata[int(p_norm[int(idx)])].append(int(idx))
    return strata


def _resample_probabilities(cal_weights: np.ndarray, test_weights: np.ndarray, j: int) -> np.ndarray:
    n = int(len(cal_weights))
    probs = np.empty(n + 1, dtype=float)
    probs[:n] = cal_weights
    probs[n] = test_weights[j]
    probs /= probs.sum()
    return probs


def _swap_weights(cal_weights: np.ndarray, test_weights: np.ndarray, i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
    re_cal_weights = np.asarray(cal_weights, dtype=float).copy()
    re_test_weights = np.asarray(test_weights, dtype=float).copy()
    tmp = re_cal_weights[i]
    re_cal_weights[i] = re_test_weights[j]
    re_test_weights[j] = tmp
    return re_cal_weights, re_test_weights


def _streaming_term_matrix(
    *,
    resampled_score_matrix: np.ndarray,
    cal_weights: np.ndarray,
    test_weights: np.ndarray,
    alpha_fdr: float,
    alpha_e: float,
    esT: bool,
    CC_stat,
    R_stat,
    c_value: float,
    j: int,
    guarantee: bool,
) -> float:
    m = int(len(test_weights))
    resampled_stat = CC_stat(resampled_score_matrix, cal_weights, test_weights, 1.0, alpha_fdr, j, alpha_e, esT)
    resampled_hatR = R_stat(resampled_score_matrix, cal_weights, test_weights, 1.0, alpha_fdr, j, True, alpha_e, esT)
    e_tilde = _e_static(alpha_e, resampled_score_matrix, cal_weights, test_weights, esT)
    budget_j = float(e_tilde[j]) if guarantee else 1.0
    e_rej_vector = eBH_vector(e_tilde, alpha_fdr)
    indic = 1.0 if ((e_rej_vector[j] == 1) or (resampled_stat <= c_value)) else 0.0
    return (m / alpha_fdr) * indic / float(resampled_hatR) - budget_j


def _streaming_term_vector(
    *,
    resampled_scores_vec: np.ndarray,
    cal_weights: np.ndarray,
    test_weights: np.ndarray,
    alpha_fdr: float,
    alpha_e: float,
    esT: bool,
    CC_stat,
    R_stat,
    c_value: float,
    j: int,
    guarantee: bool,
) -> float:
    m = int(len(test_weights))
    resampled_stat = CC_stat(resampled_scores_vec, cal_weights, test_weights, 1.0, alpha_fdr, j, alpha_e, esT)
    resampled_hatR = R_stat(resampled_scores_vec, cal_weights, test_weights, 1.0, alpha_fdr, j, True, alpha_e, esT)
    e_tilde = _e_vec(alpha_e, resampled_scores_vec, cal_weights, test_weights, esT)
    budget_j = float(e_tilde[j]) if guarantee else 1.0
    e_rej_vector = eBH_vector(e_tilde, alpha_fdr)
    indic = 1.0 if ((e_rej_vector[j] == 1) or (resampled_stat <= c_value)) else 0.0
    return (m / alpha_fdr) * indic / float(resampled_hatR) - budget_j


@jit(nopython=True)
def _resample_probabilities_numba(cal_weights: np.ndarray, test_weights: np.ndarray, j: int) -> np.ndarray:
    n = cal_weights.size
    probs = np.empty(n + 1, dtype=np.float64)
    denom = 0.0
    for i in range(n):
        probs[i] = cal_weights[i]
        denom += probs[i]
    probs[n] = test_weights[j]
    denom += probs[n]
    for i in range(n + 1):
        probs[i] /= denom
    return probs


@jit(nopython=True)
def _swap_weights_inplace(
    base_cal: np.ndarray,
    base_test: np.ndarray,
    i: int,
    j: int,
    out_cal: np.ndarray,
    out_test: np.ndarray,
) -> None:
    for k in range(base_cal.size):
        out_cal[k] = base_cal[k]
    for k in range(base_test.size):
        out_test[k] = base_test[k]
    tmp = out_cal[i]
    out_cal[i] = out_test[j]
    out_test[j] = tmp


@jit(nopython=True)
def _streaming_term_matrix_specialized(
    resampled_score_matrix: np.ndarray,
    cal_weights: np.ndarray,
    test_weights: np.ndarray,
    alpha_fdr: float,
    alpha_e: float,
    esT: bool,
    c_value: float,
    j: int,
    guarantee: bool,
    stat_mode: int,
) -> float:
    m = test_weights.size
    if stat_mode == 0:
        resampled_stat = p_value_j(resampled_score_matrix, cal_weights, test_weights, 1.0, alpha_fdr, j, alpha_e, esT)
    else:
        resampled_stat = pBH_threshold(resampled_score_matrix, cal_weights, test_weights, 1.0, alpha_fdr, j, alpha_e, esT)
    resampled_hatR = hatR_p_combination(resampled_score_matrix, cal_weights, test_weights, 1.0, alpha_fdr, j, True, alpha_e, esT)
    e_tilde = _e_static(alpha_e, resampled_score_matrix, cal_weights, test_weights, esT)
    budget_j = e_tilde[j] if guarantee else 1.0
    e_rej_vector = eBH_vector(e_tilde, alpha_fdr)
    indic = 1.0 if ((e_rej_vector[j] == 1) or (resampled_stat <= c_value)) else 0.0
    return (m / alpha_fdr) * indic / resampled_hatR - budget_j


@jit(nopython=True)
def _streaming_term_vector_pcomb(
    resampled_scores_vec: np.ndarray,
    cal_weights: np.ndarray,
    test_weights: np.ndarray,
    alpha_fdr: float,
    alpha_e: float,
    esT: bool,
    c_value: float,
    j: int,
    guarantee: bool,
) -> float:
    m = test_weights.size
    resampled_stat = pBH_threshold_vector(resampled_scores_vec, cal_weights, test_weights, 1.0, alpha_fdr, j, alpha_e, esT)
    resampled_hatR = hatR_p_combination_vector(resampled_scores_vec, cal_weights, test_weights, 1.0, alpha_fdr, j, True, alpha_e, esT)
    e_tilde = _e_vec(alpha_e, resampled_scores_vec, cal_weights, test_weights, esT)
    budget_j = e_tilde[j] if guarantee else 1.0
    e_rej_vector = eBH_vector(e_tilde, alpha_fdr)
    indic = 1.0 if ((e_rej_vector[j] == 1) or (resampled_stat <= c_value)) else 0.0
    return (m / alpha_fdr) * indic / resampled_hatR - budget_j


@jit(nopython=True)
def _boost_matrix_streaming_numba_pcomb(
    alpha_fdr: float,
    alpha_e: float,
    esT: bool,
    prune: bool,
    cc_fail_counter_limit: int,
    score_pool: np.ndarray,
    blocks,
    cal_weights: np.ndarray,
    test_weights: np.ndarray,
    guarantee: bool,
    stat_mode: int,
):
    m = test_weights.size
    n = cal_weights.size
    metric_flag = 0
    base_scores, _ = subroutine_mdlsel_numba(
        m, n, blocks, alpha_fdr, score_pool, cal_weights, test_weights, metric_flag, 1, 0.5
    )
    p_values = _p_static(base_scores, cal_weights, test_weights, 1.0)
    e_tilde = _e_static(alpha_e, base_scores, cal_weights, test_weights, esT)
    T_array = _T_static(alpha_e, base_scores, cal_weights, test_weights, esT)
    e_rej_vec = eBH_vector(e_tilde, alpha_fdr)
    base_rej_count = int(np.sum(e_rej_vec))

    block_memberships = np.empty(m, dtype=np.int64)
    for j in range(m):
        block_memberships[j] = -1
    for bidx in range(len(blocks)):
        block = blocks[bidx]
        for t in range(block.size):
            block_memberships[block[t]] = bidx

    p_norm = np.rint(p_values * (n + 1)).astype(np.int64)
    order = np.argsort(p_norm)
    e_boost = np.zeros(m, dtype=np.float64)
    re_cal_weights = np.empty_like(cal_weights)
    re_test_weights = np.empty_like(test_weights)
    tmp_left = np.empty(score_pool.shape[0], dtype=np.float64)
    tmp_right = np.empty(score_pool.shape[0], dtype=np.float64)
    cc_fail_counter = 0

    pos = 0
    while pos < m:
        key = p_norm[order[pos]]
        end = pos
        while end < m and p_norm[order[end]] == key:
            end += 1

        cc_fail_indicator = 1
        idx = pos
        while idx < end:
            j = int(order[idx])
            idx += 1

            if guarantee and e_rej_vec[j] == 1:
                e_boost[j] = m / (alpha_fdr * base_rej_count)
                cc_fail_indicator = 0
                continue
            if cc_fail_counter >= cc_fail_counter_limit:
                break

            bidx = block_memberships[j]
            block = blocks[bidx]
            if e_rej_vec[j] == 1:
                c_value = 0.0
            elif stat_mode == 0:
                c_value = p_value_j(base_scores, cal_weights, test_weights, 1.0, alpha_fdr, j, alpha_e, esT)
            else:
                c_value = pBH_threshold(base_scores, cal_weights, test_weights, 1.0, alpha_fdr, j, alpha_e, esT)
            hatR_U = hatR_p_combination(base_scores, cal_weights, test_weights, 1.0, alpha_fdr, j, True, alpha_e, esT)
            probabilities = _resample_probabilities_numba(cal_weights, test_weights, j)
            cond_expectation = 0.0

            for i in range(n):
                _swap_weights_inplace(cal_weights, test_weights, i, j, re_cal_weights, re_test_weights)
                if prune and (base_scores[j, i] < base_scores[j, n + j]) and (base_scores[j, i] < T_array[j]):
                    continue

                for li in range(score_pool.shape[0]):
                    tmp_left[li] = score_pool[li, i]
                    tmp_right[li] = score_pool[li, n + j]
                    score_pool[li, i] = tmp_right[li]
                    score_pool[li, n + j] = tmp_left[li]

                scores_resample, _ = subroutine_mdlsel_numba(
                    m, n, blocks, alpha_fdr, score_pool, cal_weights, test_weights, metric_flag, 1, 0.5
                )

                for li in range(score_pool.shape[0]):
                    score_pool[li, i] = tmp_left[li]
                    score_pool[li, n + j] = tmp_right[li]

                for t in range(block.size):
                    row_idx = int(block[t])
                    for c in range(n + m):
                        scores_resample[row_idx, c] = base_scores[row_idx, c]
                    tmp = scores_resample[row_idx, i]
                    scores_resample[row_idx, i] = scores_resample[row_idx, n + j]
                    scores_resample[row_idx, n + j] = tmp

                cond_expectation += probabilities[i] * _streaming_term_matrix_specialized(
                    scores_resample, re_cal_weights, re_test_weights, alpha_fdr, alpha_e, esT, c_value, j, guarantee, stat_mode
                )

            # last swap (out of the n+1)
            cond_expectation += probabilities[n] * _streaming_term_matrix_specialized(
                base_scores, cal_weights, test_weights, alpha_fdr, alpha_e, esT, c_value, j, guarantee, stat_mode
            )

            e_boost[j] = (m / (alpha_fdr * hatR_U)) * (1.0 if (cond_expectation <= 0.0) else 0.0)
            if e_boost[j] > 0.0:
                cc_fail_indicator = 0

        if cc_fail_indicator != 0:
            cc_fail_counter += 1
        else:
            cc_fail_counter = 0
        pos = end

    rej_vec = eBH_vector(e_boost, alpha_fdr)
    boost_rej = np.where(rej_vec == 1)[0]
    return e_boost, boost_rej


@jit(nopython=True)
def _boost_vector_streaming_numba_pcomb(
    alpha_fdr: float,
    alpha_e: float,
    esT: bool,
    prune: bool,
    cc_fail_counter_limit: int,
    scores_vec: np.ndarray,
    cal_weights: np.ndarray,
    test_weights: np.ndarray,
    guarantee: bool,
):
    n = cal_weights.size
    m = test_weights.size
    p_values = _p_vec(scores_vec, cal_weights, test_weights, 1.0)
    e_tilde = _e_vec(alpha_e, scores_vec, cal_weights, test_weights, esT)
    T_array = _T_vec(alpha_e, scores_vec, cal_weights, test_weights, esT)
    e_rej_vec = eBH_vector(e_tilde, alpha_fdr)
    base_rej_count = int(np.sum(e_rej_vec))

    p_norm = np.rint(p_values * (n + 1)).astype(np.int64)
    order = np.argsort(p_norm)
    e_boost = np.zeros(m, dtype=np.float64)
    re_cal_weights = np.empty_like(cal_weights)
    re_test_weights = np.empty_like(test_weights)
    resampled_vec = np.empty_like(scores_vec)
    cc_fail_counter = 0

    pos = 0
    while pos < m:
        key = p_norm[order[pos]]
        end = pos
        while end < m and p_norm[order[end]] == key:
            end += 1

        cc_fail_indicator = 1
        idx = pos
        while idx < end:
            j = int(order[idx])
            idx += 1

            if guarantee and e_rej_vec[j] == 1:
                e_boost[j] = m / (alpha_fdr * base_rej_count)
                cc_fail_indicator = 0
                continue
            if cc_fail_counter >= cc_fail_counter_limit:
                break

            c_value = 0.0 if e_rej_vec[j] == 1 else pBH_threshold_vector(
                scores_vec, cal_weights, test_weights, 1.0, alpha_fdr, j, alpha_e, esT
            )
            hatR_U = hatR_p_combination_vector(scores_vec, cal_weights, test_weights, 1.0, alpha_fdr, j, True, alpha_e, esT)
            probabilities = _resample_probabilities_numba(cal_weights, test_weights, j)
            cond_expectation = 0.0

            for i in range(n):
                _swap_weights_inplace(cal_weights, test_weights, i, j, re_cal_weights, re_test_weights)
                if prune and (scores_vec[i] < scores_vec[n + j]) and (scores_vec[i] < T_array[j]):
                    continue
                for c in range(n + m):
                    resampled_vec[c] = scores_vec[c]
                tmp = resampled_vec[i]
                resampled_vec[i] = resampled_vec[n + j]
                resampled_vec[n + j] = tmp
                cond_expectation += probabilities[i] * _streaming_term_vector_pcomb(
                    resampled_vec, re_cal_weights, re_test_weights, alpha_fdr, alpha_e, esT, c_value, j, guarantee
                )

            cond_expectation += probabilities[n] * _streaming_term_vector_pcomb(
                scores_vec, cal_weights, test_weights, alpha_fdr, alpha_e, esT, c_value, j, guarantee
            )
            e_boost[j] = (m / (alpha_fdr * hatR_U)) * (1.0 if (cond_expectation <= 0.0) else 0.0)
            if e_boost[j] > 0.0:
                cc_fail_indicator = 0

        if cc_fail_indicator != 0:
            cc_fail_counter += 1
        else:
            cc_fail_counter = 0
        pos = end

    rej_vec = eBH_vector(e_boost, alpha_fdr)
    boost_rej = np.where(rej_vec == 1)[0]
    return e_boost, boost_rej


class EBHCC:
    """Streaming conditionally calibrated e-BH booster.

    This is the streaming variant used as the package default: it iterates over
    the ``n + 1`` conditional resamples instead of materializing a full resample
    tensor. The default numba fast paths are intentionally narrow. They are used
    only for the auxiliary statistics and hybrid ``hatR`` denominator implemented
    in the manuscript experiments:

    - model-selection FCND: pointwise conformal p-value auxiliary statistic;
    - weighted 1-FCND/SCND: BH-scaled weighted conformal p-value statistic.

    The ``prune`` and ``guarantee`` shortcuts are not generic EBHCC identities;
    they are valid for the compatible auxiliary statistic / denominator pairs
    above. The implementation raises when those shortcuts are requested with an
    unsupported custom statistic.
    """

    def __init__(
        self,
        alpha_fdr,
        alpha_e=None,
        esT=False,
        cc_fail_counter=3,
        use_numba=True,
        prune=True,
        selector_top_k=1,
        selector_lambda=12.0,
    ):
        self.alpha_fdr = float(alpha_fdr)
        self.alpha_e = float(alpha_fdr if alpha_e is None else alpha_e)
        self.esT = bool(esT)
        self.cc_fail_counter_limit = int(cc_fail_counter)
        self.use_numba = bool(use_numba)
        self.prune = bool(prune)
        self.selector_top_k = max(1, int(selector_top_k))
        self.selector_lambda = float(selector_lambda)

    @staticmethod
    def _guarantee_supported(R_stat) -> bool:
        return R_stat in {hatR_p_combination, hatR_p_combination_vector}

    @staticmethod
    def _prune_supported(CC_stat) -> bool:
        return CC_stat in {p_value_j, pBH_threshold, pBH_threshold_vector}

    @staticmethod
    def _call_cc_fn(fn, scores, cal_weights, test_weights, U, alpha_fdr, j, alpha_e, esT):
        try:
            return fn(scores, cal_weights, test_weights, U, alpha_fdr, j, alpha_e=alpha_e, esT=esT)
        except TypeError:
            return fn(scores, cal_weights, test_weights, U, alpha_fdr, j)

    def boost_matrix(
        self,
        *,
        CC_stat,
        R_stat,
        cal_weights,
        test_weights,
        score_pool,
        blocks,
        U=1.0,
        guarantee=True,
        selection_metric='p',
        verbose=False,
        return_time=False,
    ):
        alpha_fdr = float(self.alpha_fdr)
        alpha_e = float(self.alpha_e)
        esT = bool(self.esT)
        prune = bool(self.prune)
        cal_weights = np.asarray(cal_weights, dtype=float)
        test_weights = np.asarray(test_weights, dtype=float)
        score_pool = np.asarray(score_pool, dtype=float)
        m = int(len(test_weights))
        n = int(len(cal_weights))
        if guarantee and not self._guarantee_supported(R_stat):
            raise ValueError('guarantee=True is only supported with the built-in hatR_p_combination denominator.')
        if prune and not self._prune_supported(CC_stat):
            raise ValueError('prune=True is only supported with the built-in p-value auxiliary statistics.')
        if blocks is None:
            raise ValueError('blocks must be provided (list of arrays of test indices)')
        if score_pool is None:
            raise ValueError('score_pool is required for boost_matrix (model selection)')
        metric_flag = 0 if selection_metric == 'p' else 1
        use_smoothed_selector = self.selector_top_k > 1

        nb_blocks = _NList()
        for b in blocks:
            nb_blocks.append(np.asarray(b, dtype=np.int64))

        t0_boost = time.perf_counter()
        if (
            self.use_numba
            and not use_smoothed_selector
            and U == 1.0
            and selection_metric == 'p'
            and R_stat is hatR_p_combination
            and (CC_stat is p_value_j or CC_stat is pBH_threshold)
        ):
            stat_mode = 0 if CC_stat is p_value_j else 1
            e_boost, boost_rej = _boost_matrix_streaming_numba_pcomb(
                alpha_fdr,
                alpha_e,
                esT,
                prune,
                self.cc_fail_counter_limit,
                score_pool.copy(),
                nb_blocks,
                cal_weights,
                test_weights,
                guarantee,
                stat_mode,
            )
            dt_boost = time.perf_counter() - t0_boost
            if return_time:
                return e_boost, boost_rej, dt_boost
            return e_boost, boost_rej

        smooth_score_pool = None
        if use_smoothed_selector:
            if self.use_numba:
                smooth_score_pool = neglog_empirical_rank_scores_numba(score_pool)
            else:
                smooth_score_pool = neglog_empirical_rank_scores_py(score_pool)

        if self.use_numba and not use_smoothed_selector:
            base_scores, _ = subroutine_mdlsel_numba(
                m, n, nb_blocks, alpha_fdr, score_pool, cal_weights, test_weights, metric_flag, 1, 0.5
            )
        elif self.use_numba:
            base_scores, _ = subroutine_mdlsel_numba_from_pools(
                m,
                n,
                nb_blocks,
                alpha_fdr,
                score_pool,
                smooth_score_pool,
                cal_weights,
                test_weights,
                metric_flag,
                self.selector_top_k,
                self.selector_lambda,
            )
        else:
            selector_fn = subroutine_mdlsel_py if not use_smoothed_selector else subroutine_mdlsel_py_from_pools
            selector_args = (
                m, n, blocks, alpha_fdr, score_pool, cal_weights, test_weights,
            )
            if not use_smoothed_selector:
                base_scores, _ = selector_fn(
                    *selector_args,
                    selection_metric=selection_metric,
                    selector_top_k=1,
                    selector_lambda=0.5,
                )
            else:
                base_scores, _ = selector_fn(
                    m,
                    n,
                    blocks,
                    alpha_fdr,
                    score_pool,
                    smooth_score_pool,
                    cal_weights,
                    test_weights,
                    selection_metric=selection_metric,
                    selector_top_k=self.selector_top_k,
                    selector_lambda=self.selector_lambda,
                )

        p_values = _p_static(base_scores, cal_weights, test_weights, U)
        e_tilde = _e_static(alpha_e, base_scores, cal_weights, test_weights, esT)
        T_array = _T_static(alpha_e, base_scores, cal_weights, test_weights, esT)
        e_rej_vec = eBH_vector(e_tilde, alpha_fdr)
        base_rej_count = int(np.sum(e_rej_vec))

        block_memberships = {}
        for bidx, block in enumerate(blocks):
            for j in block:
                block_memberships[int(j)] = int(bidx)

        strata = _build_strata(p_values, n)
        e_boost = np.zeros(m)
        cc_fail_counter = 0

        for key in strata:
            stratum = strata[key]
            cc_fail_indicator = 1
            for j in stratum:
                if guarantee and e_rej_vec[j] == 1:
                    e_boost[j] = m / (alpha_fdr * base_rej_count)
                    cc_fail_indicator = 0
                    continue
                if cc_fail_counter >= self.cc_fail_counter_limit:
                    break

                bidx = block_memberships[j]
                block = np.asarray(blocks[bidx], dtype=int)
                c_value = 0.0 if e_rej_vec[j] == 1 else self._call_cc_fn(
                    CC_stat, base_scores, cal_weights, test_weights, U, alpha_fdr, j, alpha_e, esT
                )
                hatR_U = self._call_cc_fn(R_stat, base_scores, cal_weights, test_weights, U, alpha_fdr, j, alpha_e, esT)
                probabilities = _resample_probabilities(cal_weights, test_weights, j)
                cond_expectation = 0.0

                for i in range(n):
                    re_cal_weights, re_test_weights = _swap_weights(cal_weights, test_weights, i, j)
                    if prune and (base_scores[j, i] < base_scores[j, n + j]) and (base_scores[j, i] < T_array[j]):
                        continue
                    if self.use_numba and not use_smoothed_selector:
                        sw_pool = score_pool.copy()
                        sw_pool[:, [i, (n + j)]] = sw_pool[:, [(n + j), i]]
                        scores_resample, _ = subroutine_mdlsel_numba(
                            m, n, nb_blocks, alpha_fdr, sw_pool, cal_weights, test_weights, metric_flag, 1, 0.5
                        )
                    elif self.use_numba:
                        raw_left = score_pool[:, i].copy()
                        raw_right = score_pool[:, n + j].copy()
                        smooth_left = smooth_score_pool[:, i].copy()
                        smooth_right = smooth_score_pool[:, n + j].copy()
                        score_pool[:, i] = raw_right
                        score_pool[:, n + j] = raw_left
                        smooth_score_pool[:, i] = smooth_right
                        smooth_score_pool[:, n + j] = smooth_left
                        scores_resample, _ = subroutine_mdlsel_numba_from_pools(
                            m,
                            n,
                            nb_blocks,
                            alpha_fdr,
                            score_pool,
                            smooth_score_pool,
                            cal_weights,
                            test_weights,
                            metric_flag,
                            self.selector_top_k,
                            self.selector_lambda,
                        )
                        score_pool[:, i] = raw_left
                        score_pool[:, n + j] = raw_right
                        smooth_score_pool[:, i] = smooth_left
                        smooth_score_pool[:, n + j] = smooth_right
                    elif not use_smoothed_selector:
                        sw_pool = score_pool.copy()
                        sw_pool[:, [i, (n + j)]] = sw_pool[:, [(n + j), i]]
                        scores_resample, _ = subroutine_mdlsel_py(
                            m,
                            n,
                            blocks,
                            alpha_fdr,
                            sw_pool,
                            cal_weights,
                            test_weights,
                            selection_metric=selection_metric,
                            selector_top_k=1,
                            selector_lambda=0.5,
                        )
                    else:
                        raw_left = score_pool[:, i].copy()
                        raw_right = score_pool[:, n + j].copy()
                        smooth_left = smooth_score_pool[:, i].copy()
                        smooth_right = smooth_score_pool[:, n + j].copy()
                        score_pool[:, i] = raw_right
                        score_pool[:, n + j] = raw_left
                        smooth_score_pool[:, i] = smooth_right
                        smooth_score_pool[:, n + j] = smooth_left
                        scores_resample, _ = subroutine_mdlsel_py_from_pools(
                            m,
                            n,
                            blocks,
                            alpha_fdr,
                            score_pool,
                            smooth_score_pool,
                            cal_weights,
                            test_weights,
                            selection_metric=selection_metric,
                            selector_top_k=self.selector_top_k,
                            selector_lambda=self.selector_lambda,
                        )
                        score_pool[:, i] = raw_left
                        score_pool[:, n + j] = raw_right
                        smooth_score_pool[:, i] = smooth_left
                        smooth_score_pool[:, n + j] = smooth_right
                    scores_for_current_block = base_scores[block, :].copy()
                    scores_for_current_block[:, [i, (n + j)]] = scores_for_current_block[:, [(n + j), i]]
                    scores_resample[block, :] = scores_for_current_block

                    cond_expectation += probabilities[i] * _streaming_term_matrix(
                        resampled_score_matrix=scores_resample,
                        cal_weights=re_cal_weights,
                        test_weights=re_test_weights,
                        alpha_fdr=alpha_fdr,
                        alpha_e=alpha_e,
                        esT=esT,
                        CC_stat=CC_stat,
                        R_stat=R_stat,
                        c_value=c_value,
                        j=j,
                        guarantee=guarantee,
                    )

                cond_expectation += probabilities[n] * _streaming_term_matrix(
                    resampled_score_matrix=base_scores,
                    cal_weights=cal_weights,
                    test_weights=test_weights,
                    alpha_fdr=alpha_fdr,
                    alpha_e=alpha_e,
                    esT=esT,
                    CC_stat=CC_stat,
                    R_stat=R_stat,
                    c_value=c_value,
                    j=j,
                    guarantee=guarantee,
                )

                e_boost[j] = (m / (alpha_fdr * hatR_U)) * (1.0 if (cond_expectation <= 0) else 0.0)
                if e_boost[j] > 0:
                    cc_fail_indicator = 0

            cc_fail_counter = cc_fail_counter + 1 if cc_fail_indicator != 0 else 0

        rej_vec = eBH_vector(e_boost, alpha_fdr)
        boost_rej = np.where(rej_vec == 1)[0]
        dt_boost = time.perf_counter() - t0_boost
        if return_time:
            return e_boost, boost_rej, dt_boost
        return e_boost, boost_rej

    def boost_vector(
        self,
        *,
        CC_stat,
        R_stat,
        cal_weights,
        test_weights,
        scores_vec,
        U=1.0,
        guarantee=True,
        verbose=False,
        return_time=False,
    ):
        alpha_fdr = float(self.alpha_fdr)
        alpha_e = float(self.alpha_e)
        esT = bool(self.esT)
        prune = bool(self.prune)
        cal_weights = np.asarray(cal_weights, dtype=float)
        test_weights = np.asarray(test_weights, dtype=float)
        scores_vec = np.asarray(scores_vec, dtype=float).reshape(-1)
        m = int(len(test_weights))
        n = int(len(cal_weights))
        if guarantee and not self._guarantee_supported(R_stat):
            raise ValueError('guarantee=True is only supported with the built-in hatR_p_combination_vector denominator.')
        if prune and not self._prune_supported(CC_stat):
            raise ValueError('prune=True is only supported with the built-in p-value auxiliary statistics.')
        if scores_vec.shape[0] != n + m:
            raise ValueError('scores_vec must have length n+m')

        t0_boost = time.perf_counter()
        if (
            self.use_numba
            and U == 1.0
            and CC_stat is pBH_threshold_vector
            and R_stat is hatR_p_combination_vector
        ):
            e_boost, boost_rej = _boost_vector_streaming_numba_pcomb(
                alpha_fdr,
                alpha_e,
                esT,
                prune,
                self.cc_fail_counter_limit,
                scores_vec,
                cal_weights,
                test_weights,
                guarantee,
            )
            dt_boost = time.perf_counter() - t0_boost
            if return_time:
                return e_boost, boost_rej, dt_boost
            return e_boost, boost_rej

        p_values = _p_vec(scores_vec, cal_weights, test_weights, U)
        e_tilde = _e_vec(alpha_e, scores_vec, cal_weights, test_weights, esT)
        T_array = _T_vec(alpha_e, scores_vec, cal_weights, test_weights, esT)
        e_rej_vec = eBH_vector(e_tilde, alpha_fdr)
        base_rej_count = int(np.sum(e_rej_vec))

        strata = _build_strata(p_values, n)
        e_boost = np.zeros(m)
        cc_fail_counter = 0

        for key in strata:
            stratum = strata[key]
            cc_fail_indicator = 1
            for j in stratum:
                if guarantee and e_rej_vec[j] == 1:
                    e_boost[j] = m / (alpha_fdr * base_rej_count)
                    cc_fail_indicator = 0
                    continue
                if cc_fail_counter >= self.cc_fail_counter_limit:
                    break

                c_value = 0.0 if e_rej_vec[j] == 1 else self._call_cc_fn(
                    CC_stat, scores_vec, cal_weights, test_weights, U, alpha_fdr, j, alpha_e, esT
                )
                hatR_U = self._call_cc_fn(R_stat, scores_vec, cal_weights, test_weights, U, alpha_fdr, j, alpha_e, esT)
                probabilities = _resample_probabilities(cal_weights, test_weights, j)
                cond_expectation = 0.0

                for i in range(n):
                    re_cal_weights, re_test_weights = _swap_weights(cal_weights, test_weights, i, j)
                    if prune and (scores_vec[i] < scores_vec[n + j]) and (scores_vec[i] < T_array[j]):
                        continue
                    resampled_vec = scores_vec.copy()
                    resampled_vec[[i, (n + j)]] = resampled_vec[[(n + j), i]]

                    cond_expectation += probabilities[i] * _streaming_term_vector(
                        resampled_scores_vec=resampled_vec,
                        cal_weights=re_cal_weights,
                        test_weights=re_test_weights,
                        alpha_fdr=alpha_fdr,
                        alpha_e=alpha_e,
                        esT=esT,
                        CC_stat=CC_stat,
                        R_stat=R_stat,
                        c_value=c_value,
                        j=j,
                        guarantee=guarantee,
                    )

                cond_expectation += probabilities[n] * _streaming_term_vector(
                    resampled_scores_vec=scores_vec,
                    cal_weights=cal_weights,
                    test_weights=test_weights,
                    alpha_fdr=alpha_fdr,
                    alpha_e=alpha_e,
                    esT=esT,
                    CC_stat=CC_stat,
                    R_stat=R_stat,
                    c_value=c_value,
                    j=j,
                    guarantee=guarantee,
                )

                e_boost[j] = (m / (alpha_fdr * hatR_U)) * (1.0 if (cond_expectation <= 0) else 0.0)
                if e_boost[j] > 0:
                    cc_fail_indicator = 0

            cc_fail_counter = cc_fail_counter + 1 if cc_fail_indicator != 0 else 0

        rej_vec = eBH_vector(e_boost, alpha_fdr)
        boost_rej = np.where(rej_vec == 1)[0]
        dt_boost = time.perf_counter() - t0_boost
        if return_time:
            return e_boost, boost_rej, dt_boost
        return e_boost, boost_rej

    def boost(
        self,
        *,
        CC_stat,
        R_stat,
        cal_weights,
        test_weights,
        blocks=None,
        score_pool=None,
        scores_vec=None,
        U=1.0,
        guarantee=True,
        selection_metric='p',
        verbose=False,
        return_time=False,
    ):
        if score_pool is not None:
            return self.boost_matrix(
                CC_stat=CC_stat,
                R_stat=R_stat,
                cal_weights=cal_weights,
                test_weights=test_weights,
                score_pool=score_pool,
                blocks=blocks,
                U=U,
                guarantee=guarantee,
                selection_metric=selection_metric,
                verbose=verbose,
                return_time=return_time,
            )
        if scores_vec is None:
            raise ValueError('scores_vec must be provided for boost_vector when score_pool is None')
        return self.boost_vector(
            CC_stat=CC_stat,
            R_stat=R_stat,
            cal_weights=cal_weights,
            test_weights=test_weights,
            scores_vec=scores_vec,
            U=U,
            guarantee=guarantee,
            verbose=verbose,
            return_time=return_time,
        )

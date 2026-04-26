from __future__ import annotations

import numpy as np
from numba import jit

from fcnd.utils.cc_utils import e_function_static, e_function_vector, eBH_vector


@jit(nopython=True)
def U_integrand(u_tilde, c, resampled_scores_list, cal_weights, test_weights, alpha_fdr, CC_stat, R_stat, j, guarantee=False, alpha_e=-1.0, esT=False):
    m = len(test_weights)
    n = len(cal_weights)
    assert resampled_scores_list.shape == (n+1, m, n+m)

    probabilities = np.ones(n+1)
    probabilities[:n] = cal_weights
    probabilities[n] = test_weights[j]
    denom_prob = 0.0
    for t in range(n+1):
        denom_prob += probabilities[t]
    for t in range(n+1):
        probabilities[t] = probabilities[t] / denom_prob

    summation = 0.0
    for idx in range(n+1):
        resampled_score_matrix = resampled_scores_list[idx]
        re_cal_weights = cal_weights.copy()
        re_test_weights= test_weights.copy()
        if idx < n:
            tmp = re_cal_weights[idx]
            re_cal_weights[idx] = re_test_weights[j]
            re_test_weights[j] = tmp

        resampled_stat = CC_stat(resampled_score_matrix, re_cal_weights, re_test_weights, u_tilde, alpha_fdr, j, alpha_e, esT)
        resampled_hatR = R_stat(resampled_score_matrix, re_cal_weights, re_test_weights, u_tilde, alpha_fdr, j, True, alpha_e, esT)
        alpha_used = alpha_fdr if alpha_e <= 0.0 else alpha_e
        e_tilde = e_function_static(alpha_used, resampled_score_matrix, re_cal_weights, re_test_weights, esT)
        budget_j = e_tilde[j] if guarantee else 1.0
        e_rej_vector = eBH_vector(e_tilde, alpha_fdr)
        indic = 1 if ((e_rej_vector[j] == 1) or (resampled_stat <= c)) else 0
        summation += ((m/alpha_fdr * indic/resampled_hatR) - budget_j) * probabilities[idx]
    return summation


@jit(nopython=True)
def CC_grid(c, resampled_scores_list, cal_weights, test_weights, alpha_fdr, CC_stat, R_stat, j, guarantee=False, alpha_e=-1.0, esT=False):
    cond_expectation = U_integrand(1.0, c, resampled_scores_list, cal_weights, test_weights, alpha_fdr, CC_stat, R_stat, j, guarantee, alpha_e, esT)
    return cond_expectation, None


@jit(nopython=True)
def U_integrand_vector(u_tilde, c, resampled_vectors_list, cal_weights, test_weights, alpha_fdr, CC_stat_vector, R_stat_vector, j, guarantee=False, alpha_e=-1.0, esT=False):
    m = len(test_weights)
    n = len(cal_weights)
    assert resampled_vectors_list.shape == (n+1, n+m)
    probabilities = np.ones(n+1)
    probabilities[:n] = cal_weights
    probabilities[n] = test_weights[j]
    denom_prob = 0.0
    for t in range(n+1):
        denom_prob += probabilities[t]
    for t in range(n+1):
        probabilities[t] = probabilities[t] / denom_prob
    summation = 0.0
    for idx in range(n+1):
        resampled_vec = resampled_vectors_list[idx]
        re_cal_weights = cal_weights.copy()
        re_test_weights = test_weights.copy()
        if idx < n:
            tmp = re_cal_weights[idx]
            re_cal_weights[idx] = re_test_weights[j]
            re_test_weights[j] = tmp
        resampled_stat = CC_stat_vector(resampled_vec, re_cal_weights, re_test_weights, u_tilde, alpha_fdr, j, alpha_e, esT)
        resampled_hatR = R_stat_vector(resampled_vec, re_cal_weights, re_test_weights, u_tilde, alpha_fdr, j, True, alpha_e, esT)
        alpha_used = alpha_fdr if alpha_e <= 0.0 else alpha_e
        e_tilde = e_function_vector(alpha_used, resampled_vec, re_cal_weights, re_test_weights, esT)
        budget_j = e_tilde[j] if guarantee else 1.0
        e_rej_vector = eBH_vector(e_tilde, alpha_fdr)
        indic = 1 if ((e_rej_vector[j] == 1) or (resampled_stat <= c)) else 0
        summation += ((m/alpha_fdr * indic/resampled_hatR) - budget_j) * probabilities[idx]
    return summation


@jit(nopython=True)
def CC_grid_vector(c, resampled_vectors_list, cal_weights, test_weights, alpha_fdr, CC_stat_vector, R_stat_vector, j, guarantee=False, alpha_e=-1.0, esT=False):
    return U_integrand_vector(1.0, c, resampled_vectors_list, cal_weights, test_weights, alpha_fdr, CC_stat_vector, R_stat_vector, j, guarantee, alpha_e, esT), None

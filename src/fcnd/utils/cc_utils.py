import numpy as np
from numba import jit

# Conformal p-values/e-values and e-BH utilities

# p-value functions

@jit(nopython=True)
def p_function_static(score_matrix, cal_weights, test_weights, U=1.0):
    m, total = score_matrix.shape
    n = total - m
    sum_cw = np.sum(cal_weights[:n])
    p = np.empty(m)
    for j in range(m):
        tj = score_matrix[j, n + j]
        # Vectorized inner accumulation over calibration points
        cnt = np.sum(cal_weights[:n] * (score_matrix[j, :n] >= tj))
        denom = sum_cw + test_weights[j]
        p[j] = (cnt + test_weights[j] * U) / denom
    return p

@jit(nopython=True)
def p_function_idx_static(score_matrix, cal_weights, test_weights, idx, U=1.0):
    m, total = score_matrix.shape
    n = total - m
    sum_cw = np.sum(cal_weights[:n])
    tj = score_matrix[idx, n + idx]
    cnt = np.sum(cal_weights[:n] * (score_matrix[idx, :n] >= tj))
    denom = sum_cw + test_weights[idx]
    return (cnt + test_weights[idx] * U) / denom

# Public aliases for historical function names.
@jit(nopython=True)
def p_function(score_matrix, cal_weights, test_weights, U=1.0):
    return p_function_static(score_matrix, cal_weights, test_weights, U)

# e-value functions

@jit(nopython=True)
def e_function_idx_static(alpha, score_matrix, cal_weights, test_weights, idx, esT=False):
    m, total = score_matrix.shape
    n = total - m
    # sums and fixed refs
    sum_cw = np.sum(cal_weights[:n])
    j = idx
    sum_w_j = sum_cw + test_weights[j]
    L = m + n
    # Build combined arrays by slice assignment (test first, then cal)
    combined_scores = np.empty(L)
    combined_weights = np.empty(L)
    combined_memberships = np.empty(L, dtype=np.int64)
    combined_scores[:m] = score_matrix[j, n:n + m]
    combined_weights[:m] = test_weights[:m]
    combined_memberships[:m] = 1
    combined_scores[m:] = score_matrix[j, :n]
    combined_weights[m:] = cal_weights[:n]
    combined_memberships[m:] = 0
    inds = np.argsort(-1.0 * combined_scores)
    ordered_weights = combined_weights[inds]
    is_cal = (combined_memberships[inds] == 0)
    is_test = (combined_memberships[inds] == 1)
    # cumulative sums via vector ops
    cal_seq = np.cumsum(ordered_weights * is_cal)
    test_seq = np.cumsum(is_test.astype(np.int64))
    # avoid division by zero for prefixes with zero tests
    test_seq = np.maximum(test_seq, 1)
    hat_fdrs = (cal_seq + test_weights[j]) / test_seq * (m / sum_w_j)
    cond = hat_fdrs <= alpha
    if esT:
        cond = cond | (test_seq < (1.0 / alpha))
    nz = np.nonzero(cond)[0]
    if nz.size == 0:
        return 0.0
    thr_idx = nz[-1]
    Tj = combined_scores[inds[thr_idx]]
    denom_j = test_weights[j] + np.sum(cal_weights[:n] * (score_matrix[j, :n] >= Tj))
    return (sum_w_j * (1.0 if score_matrix[j, n + j] >= Tj else 0.0)) / denom_j


@jit(nopython=True)
def e_thres_static(alpha, score_matrix, cal_weights, test_weights, esT=False):
    m, total = score_matrix.shape
    n = total - m
    sum_cw = np.sum(cal_weights[:n])
    T_array = np.empty(m)
    L = m + n
    for j in range(m):
        sum_w_j = sum_cw + test_weights[j]
        combined_scores = np.empty(L)
        combined_weights = np.empty(L)
        combined_memberships = np.empty(L, dtype=np.int64)
        combined_scores[:m] = score_matrix[j, n:n + m]
        combined_weights[:m] = test_weights[:m]
        combined_memberships[:m] = 1
        combined_scores[m:] = score_matrix[j, :n]
        combined_weights[m:] = cal_weights[:n]
        combined_memberships[m:] = 0
        inds = np.argsort(-1.0 * combined_scores)
        ordered_scores = combined_scores[inds]
        ordered_weights = combined_weights[inds]
        is_cal = (combined_memberships[inds] == 0)
        is_test = (combined_memberships[inds] == 1)
        cal_seq = np.cumsum(ordered_weights * is_cal)
        test_seq = np.cumsum(is_test.astype(np.int64))
        test_seq = np.maximum(test_seq, 1)
        hat_fdrs = (cal_seq + test_weights[j]) / test_seq * (m / sum_w_j)
        cond = hat_fdrs <= alpha
        if esT:
            cond = cond | (test_seq < (1.0 / alpha))
        nz = np.nonzero(cond)[0]
        if nz.size == 0:
            T_array[j] = np.inf
        else:
            T_array[j] = ordered_scores[nz[-1]]
    return T_array

@jit(nopython=True)
def e_function_static(alpha, score_matrix, cal_weights, test_weights, esT=False):
    m, total = score_matrix.shape
    n = total - m
    sum_cw = np.sum(cal_weights[:n])
    e_vals = np.zeros(m)
    L = m + n
    for j in range(m):
        sum_w_j = sum_cw + test_weights[j]
        combined_scores = np.empty(L)
        combined_weights = np.empty(L)
        combined_memberships = np.empty(L, dtype=np.int64)
        combined_scores[:m] = score_matrix[j, n:n + m]
        combined_weights[:m] = test_weights[:m]
        combined_memberships[:m] = 1
        combined_scores[m:] = score_matrix[j, :n]
        combined_weights[m:] = cal_weights[:n]
        combined_memberships[m:] = 0
        inds = np.argsort(-1.0 * combined_scores)
        ordered_scores = combined_scores[inds]
        ordered_weights = combined_weights[inds]
        is_cal = (combined_memberships[inds] == 0)
        is_test = (combined_memberships[inds] == 1)
        cal_seq = np.cumsum(ordered_weights * is_cal)
        test_seq = np.cumsum(is_test.astype(np.int64))
        test_seq = np.maximum(test_seq, 1)
        hat_fdrs = (cal_seq + test_weights[j]) / test_seq * (m / sum_w_j)
        cond = hat_fdrs <= alpha
        if esT:
            cond = cond | (test_seq < (1.0 / alpha))
        nz = np.nonzero(cond)[0]
        if nz.size == 0:
            e_vals[j] = 0.0
            continue
        Tj = ordered_scores[nz[-1]]
        denom_j = test_weights[j] + np.sum(cal_weights[:n] * (score_matrix[j, :n] >= Tj))
        e_vals[j] = (sum_w_j * (1.0 if score_matrix[j, n + j] >= Tj else 0.0)) / denom_j
    return e_vals

# Vector-based variants for single-block (score vector) use-cases
@jit(nopython=True)
def p_function_vector(scores_vec, cal_weights, test_weights, U=1.0):
    n = cal_weights.size
    m = test_weights.size
    sum_cw = np.sum(cal_weights)
    p = np.empty(m)
    for j in range(m):
        tj = scores_vec[n + j]
        # Vectorized inner accumulation over calibration points
        cnt = np.sum(cal_weights * (scores_vec[:n] >= tj))
        denom = sum_cw + test_weights[j]
        p[j] = (cnt + test_weights[j] * U) / denom
    return p

@jit(nopython=True)
def e_thres_vector(alpha, scores_vec, cal_weights, test_weights, esT=False):
    n = cal_weights.size
    m = test_weights.size
    sum_cw = np.sum(cal_weights)
    T_array = np.empty(m)
    L = m + n
    for j in range(m):
        sum_w_j = sum_cw + test_weights[j]
        combined_scores = np.empty(L)
        combined_weights = np.empty(L)
        combined_memberships = np.empty(L, dtype=np.int64)
        combined_scores[:m] = scores_vec[n:n + m]
        combined_weights[:m] = test_weights[:m]
        combined_memberships[:m] = 1
        combined_scores[m:] = scores_vec[:n]
        combined_weights[m:] = cal_weights[:n]
        combined_memberships[m:] = 0
        inds = np.argsort(-1.0 * combined_scores)
        is_cal = (combined_memberships[inds] == 0)
        cal_seq = np.cumsum(combined_weights[inds] * is_cal)
        test_seq = np.cumsum((combined_memberships[inds] == 1).astype(np.int64))
        test_seq = np.maximum(test_seq, 1)
        hat_fdrs = (cal_seq + test_weights[j]) / test_seq * (m / sum_w_j)
        cond = hat_fdrs <= alpha
        if esT:
            cond = cond | (test_seq < (1.0 / alpha))
        nz = np.nonzero(cond)[0]
        if nz.size == 0:
            T_array[j] = np.inf
        else:
            T_array[j] = combined_scores[inds[nz[-1]]]
    return T_array

@jit(nopython=True)
def e_function_vector(alpha, scores_vec, cal_weights, test_weights, esT=False):
    n = cal_weights.size
    m = test_weights.size
    sum_cw = np.sum(cal_weights)
    e_vals = np.zeros(m)
    L = m + n
    for j in range(m):
        sum_w_j = sum_cw + test_weights[j]
        combined_scores = np.empty(L)
        combined_weights = np.empty(L)
        combined_memberships = np.empty(L, dtype=np.int64)
        combined_scores[:m] = scores_vec[n:n + m]
        combined_weights[:m] = test_weights[:m]
        combined_memberships[:m] = 1
        combined_scores[m:] = scores_vec[:n]
        combined_weights[m:] = cal_weights[:n]
        combined_memberships[m:] = 0
        inds = np.argsort(-1.0 * combined_scores)
        ordered_scores = combined_scores[inds]
        is_cal = (combined_memberships[inds] == 0)
        is_test = (combined_memberships[inds] == 1)
        cal_seq = np.cumsum(combined_weights[inds] * is_cal)
        test_seq = np.cumsum(is_test.astype(np.int64))
        test_seq = np.maximum(test_seq, 1)
        hat_fdrs = (cal_seq + test_weights[j]) / test_seq * (m / sum_w_j)
        cond = hat_fdrs <= alpha
        if esT:
            cond = cond | (test_seq < (1.0 / alpha))
        nz = np.nonzero(cond)[0]
        if nz.size == 0:
            e_vals[j] = 0.0
        else:
            Tj = ordered_scores[nz[-1]]
            denom_j = test_weights[j] + np.sum(cal_weights * (scores_vec[:n] >= Tj))
            e_vals[j] = (sum_w_j * (1.0 if scores_vec[n + j] >= Tj else 0.0)) / denom_j
    return e_vals

@jit(nopython=True)
def eBH_vector(e, alpha):
    m = e.size
    e_sort = np.sort(e)[::-1]
    khat = 0
    for k in range(m, 0, -1):
        if e_sort[k - 1] >= m / (alpha * k):
            khat = k
            break
    if khat == 0:
        thr = 1e309  # approximate +inf
    else:
        thr = m / (alpha * khat)
    out = np.zeros(m, dtype=np.int64)
    for i in range(m):
        out[i] = 1 if e[i] >= thr else 0
    return out

# Vector-mode CC statistics (single-block, 1D scores)
@jit(nopython=True)
def pBH_threshold_vector(scores_vec, cal_weights, test_weights, U, alpha_fdr, j, alpha_e=-1.0, esT=False):
    m = test_weights.size
    p = p_function_vector(scores_vec, cal_weights, test_weights, U)
    # eBH on 1/p
    invp = np.empty(m)
    for i in range(m):
        invp[i] = 1.0 / p[i]
    p_rej_vector = eBH_vector(invp, alpha_fdr)
    hatR = 0
    for i in range(m):
        hatR += p_rej_vector[i]
    hatR = hatR + (1 - p_rej_vector[j])
    return (m / alpha_fdr) * (p[j] / hatR)

@jit(nopython=True)
def hatR_pval_strata_vector(scores_vec, cal_weights, test_weights, U, alpha_fdr, j, e_anchor=True, alpha_e=-1.0, esT=False):
    m = test_weights.size
    n = cal_weights.size
    if e_anchor:
        alpha_used = alpha_fdr if alpha_e <= 0.0 else alpha_e
        e_tilde = e_function_vector(alpha_used, scores_vec, cal_weights, test_weights, esT)
        e_rej_vector = eBH_vector(e_tilde, alpha_fdr)
        if e_rej_vector[j] == 1:
            hatR = 0
            for i in range(m):
                hatR += e_rej_vector[i]
            return hatR
    p_tilde = p_function_vector(scores_vec, cal_weights, test_weights, U)
    # unnormalize to ranks in {0,...,n}
    p_unnorm = np.empty(m, dtype=np.int32)
    for i in range(m):
        p_unnorm[i] = int((p_tilde[i] - U) * (n + 1))
    hatR = 0
    for i in range(m):
        if p_unnorm[i] <= p_unnorm[j]:
            hatR += 1
    return hatR

@jit(nopython=True)
def hatR_p_rej_vector(scores_vec, cal_weights, test_weights, U, alpha_fdr, j, e_anchor=True, alpha_e=-1.0, esT=False):
    m = test_weights.size
    if e_anchor:
        alpha_used = alpha_fdr if alpha_e <= 0.0 else alpha_e
        e_tilde = e_function_vector(alpha_used, scores_vec, cal_weights, test_weights, esT)
        e_rej_vector = eBH_vector(e_tilde, alpha_fdr)
        if e_rej_vector[j] == 1:
            hatR = 0
            for i in range(m):
                hatR += e_rej_vector[i]
            return hatR
    p_tilde = p_function_vector(scores_vec, cal_weights, test_weights, U)
    invp = np.empty(m)
    for i in range(m):
        invp[i] = 1.0 / p_tilde[i]
    p_rej_vector = eBH_vector(invp, alpha_fdr)
    hatR = 0
    for i in range(m):
        hatR += p_rej_vector[i]
    hatR = hatR + (1 - p_rej_vector[j])
    return hatR

@jit(nopython=True)
def hatR_e_rej_vector(scores_vec, cal_weights, test_weights, U, alpha_fdr, j, e_anchor=True, alpha_e=-1.0, esT=False):
    m = test_weights.size
    alpha_used = alpha_fdr if alpha_e <= 0.0 else alpha_e
    e_tilde = e_function_vector(alpha_used, scores_vec, cal_weights, test_weights, esT)
    e_rej_vector = eBH_vector(e_tilde, alpha_fdr)
    hatR = 0
    for i in range(m):
        hatR += e_rej_vector[i]
    hatR = hatR + (1 - e_rej_vector[j])
    return hatR

@jit(nopython=True)
def hatR_p_combination_vector(scores_vec, cal_weights, test_weights, U, alpha_fdr, j, e_anchor=True, alpha_e=-1.0, esT=False):
    m = test_weights.size
    n = cal_weights.size
    if e_anchor:
        alpha_used = alpha_fdr if alpha_e <= 0.0 else alpha_e
        e_tilde = e_function_vector(alpha_used, scores_vec, cal_weights, test_weights, esT)
        e_rej_vector = eBH_vector(e_tilde, alpha_fdr)
        if e_rej_vector[j] == 1:
            hatR = 0
            for i in range(m):
                hatR += e_rej_vector[i]
            return hatR
    p_tilde = p_function_vector(scores_vec, cal_weights, test_weights, U)
    # ranks
    p_unnorm = np.empty(m, dtype=np.int32)
    for i in range(m):
        p_unnorm[i] = int((p_tilde[i] - U) * (n + 1))
    n_tied_or_smaller = 0
    for i in range(m):
        if p_unnorm[i] <= p_unnorm[j]:
            n_tied_or_smaller += 1
    invp = np.empty(m)
    for i in range(m):
        invp[i] = 1.0 / p_tilde[i]
    p_rej_vector = eBH_vector(invp, alpha_fdr)
    sum_rej = 0
    for i in range(m):
        sum_rej += p_rej_vector[i]
    if sum_rej == 0:
        return n_tied_or_smaller
    if n_tied_or_smaller < sum_rej:
        return n_tied_or_smaller
    else:
        return sum_rej + (1 - p_rej_vector[j])

@jit(nopython=True)
def e_function(alpha, score_matrix, cal_weights, test_weights):
    return e_function_static(alpha, score_matrix, cal_weights, test_weights)


# Statistics used inside the conditional-expectation integrand

@jit(nopython=True)
def p_value_j(resampled_score_matrix, cal_weights, test_weights, U, alpha_fdr, j, alpha_e=-1.0, esT=False):
    return p_function_idx_static(resampled_score_matrix, cal_weights, test_weights, idx=j, U=U)
        
@jit(nopython=True)
def pBH_threshold(resampled_score_matrix, cal_weights, test_weights, U, alpha_fdr, j, alpha_e=-1.0, esT=False):
    m = len(test_weights)
    
    p_tilde = p_function_static(resampled_score_matrix, cal_weights, test_weights, U=U)
    p_rej_vector = eBH_vector(1.0/p_tilde, alpha_fdr)
    
    hatR = 0
    for i in range(m):
        hatR += p_rej_vector[i]
    hatR = hatR + (1 - p_rej_vector[j])    # R U {j}
    
    return (m/alpha_fdr) * (p_tilde[j] / hatR)   # returns m/a  * p_j(U)/R(U)


# hatR functions for CC
@jit(nopython=True)
def hatR_pval_strata(resampled_score_matrix, cal_weights, test_weights, U, alpha_fdr, j, e_anchor=True, alpha_e=-1.0, esT=False):
    m = len(test_weights)
    n = len(cal_weights)    
    
    if e_anchor:
        alpha_used = alpha_fdr if alpha_e <= 0.0 else alpha_e
        e_tilde = e_function_static(alpha_used, resampled_score_matrix, cal_weights, test_weights, esT)
        e_rej_vector = eBH_vector(e_tilde, alpha_fdr)
        
        if e_rej_vector[j] == 1:
            # Anchor at the original e-BH rejection count when j is already rejected.
            hatR = sum(e_rej_vector)

            return hatR
        
    p_tilde = p_function_static(resampled_score_matrix, cal_weights, test_weights, U=U)
    p_tilde_unnormalized = ((p_tilde - U) * (n+1)).astype(np.int32)
    hatR = 0
    for i in range(m):
        if p_tilde_unnormalized[i] <= p_tilde_unnormalized[j]:
            hatR += 1
    
    return hatR

@jit(nopython=True)
def hatR_p_rej(resampled_score_matrix, cal_weights, test_weights, U, alpha_fdr, j, e_anchor=True, alpha_e=-1.0, esT=False):
    m = len(test_weights)
    
    if e_anchor:
        alpha_used = alpha_fdr if alpha_e <= 0.0 else alpha_e
        e_tilde = e_function_static(alpha_used, resampled_score_matrix, cal_weights, test_weights, esT)
        e_rej_vector = eBH_vector(e_tilde, alpha_fdr)

        if e_rej_vector[j] == 1:
            # Anchor at the original e-BH rejection count when j is already rejected.
            hatR = sum(e_rej_vector)
            
            return hatR
    
    p_tilde = p_function_static(resampled_score_matrix, cal_weights, test_weights, U=U)
    p_rej_vector = eBH_vector(1.0/p_tilde, alpha_fdr)
    hatR = 0
    for i in range(m):
        hatR += p_rej_vector[i]
    hatR = hatR + (1 - p_rej_vector[j])
    return hatR

@jit(nopython=True)
def hatR_e_rej(resampled_score_matrix, cal_weights, test_weights, U, alpha_fdr, j, e_anchor=True, alpha_e=-1.0, esT=False):
    m = len(test_weights)
    alpha_used = alpha_fdr if alpha_e <= 0.0 else alpha_e
    e_tilde = e_function_static(alpha_used, resampled_score_matrix, cal_weights, test_weights, esT)
    e_rej_vector = eBH_vector(e_tilde, alpha_fdr)
    hatR = 0
    for i in range(m):
        hatR += e_rej_vector[i]
    hatR = hatR + (1 - e_rej_vector[j])
    return hatR

@jit(nopython=True)
def hatR_p_combination(resampled_score_matrix, cal_weights, test_weights, U, alpha_fdr, j, e_anchor=True, alpha_e=-1.0, esT=False):
    # this one uses pvalue strat until we get out of the p-value rejection set
    m = len(test_weights)
    n = len(cal_weights)
    
    if e_anchor:
        alpha_used = alpha_fdr if alpha_e <= 0.0 else alpha_e
        e_tilde = e_function_static(alpha_used, resampled_score_matrix, cal_weights, test_weights, esT)
        e_rej_vector = eBH_vector(e_tilde, alpha_fdr)

        if e_rej_vector[j] == 1:
            # Anchor at the original e-BH rejection count when j is already rejected.
            hatR = sum(e_rej_vector)
            
            return hatR
    
    p_tilde = p_function_static(resampled_score_matrix, cal_weights, test_weights, U=U)
    # find the ranks
    p_tilde_unnormalized = ((p_tilde - U) * (n+1)).astype(np.int32)
    n_tied_or_smaller = 0
    for i in range(m):
        if p_tilde_unnormalized[i] <= p_tilde_unnormalized[j]:
            n_tied_or_smaller += 1
    
    # find the p-value rejection set 
    p_rej_vector = eBH_vector(1.0/p_tilde, alpha_fdr)
    
    # If this is empty, try to reject the smallest p-value stratum.
    sum_rej = 0
    for i in range(m):
        sum_rej += p_rej_vector[i]
    if sum_rej == 0:
        return n_tied_or_smaller
    
    # otherwise, this means we rejected the smallest strata. to reject larger strata that might not
    # be in the BH set, we will set the denominator to R^pbh U {j}
    # sum_rej computed above
    if n_tied_or_smaller < sum_rej:
        # we're within the strata of pBH
        # i.e., there are p-values bigger than p_j that have still be rejected by BH
        
        hatR = n_tied_or_smaller
    else:
        # in this case, we're either in the stratum of the largest p-values still rejected by BH
        # or we are outside the BH rejection set
        # either way, we use R U {j} --- here, it shouldn't matter
        hatR = sum_rej + (1 - p_rej_vector[j])
                
    return hatR

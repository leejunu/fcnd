import numpy as np

from fcnd import FCND, MSFCND, SCND, IsoForestLearner
from fcnd.synthetic import generate_wset, gen_data


def tiny_data(seed=0):
    rng = np.random.default_rng(seed)
    W = generate_wset(size=8, dims=3, random_state=rng)
    X_ref = gen_data(W, 12, a=1.0, random_state=rng)
    X_test = gen_data(W, 6, a=1.5, random_state=rng)
    return X_ref, X_test


def test_fcnd_detect_smoke():
    X_ref, X_test = tiny_data()
    det = FCND(IsoForestLearner(random_state=0, n_estimators=10), use_numba=True)
    res = det.detect(X_ref, X_test, alpha=0.2, method="ebh")
    assert res.values.shape == (X_test.shape[0],)
    assert res.rejection_mask.shape == (X_test.shape[0],)


def test_scnd_detect_smoke():
    X_ref, X_test = tiny_data(1)
    det = SCND(IsoForestLearner(random_state=0, n_estimators=10), use_numba=True)
    res = det.detect(X_ref[:6], X_ref[6:], X_test, alpha=0.2, method="ebh")
    assert res.values.shape == (X_test.shape[0],)


def test_msfcnd_detect_smoke():
    X_ref, X_test = tiny_data(2)
    learners = {
        "if10": IsoForestLearner(random_state=0, n_estimators=10),
        "if20": IsoForestLearner(random_state=1, n_estimators=20),
    }
    det = MSFCND(learners, alpha=0.2, K=2, use_numba=True)
    res = det.detect(X_ref, X_test, method="ebh")
    assert res.values.shape == (X_test.shape[0],)
    assert "selected_models" in res.metadata

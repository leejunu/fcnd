import numpy as np

from fcnd.conformal import e_full, p_full


def test_conformal_shapes_and_ranges():
    ref = np.array([0.1, 0.2, 0.3])
    test = np.array([0.05, 0.35])
    p = p_full(ref, test)
    e, _ = e_full(ref, test, 0.2)
    assert p.shape == (2,)
    assert e.shape == (2,)
    assert np.all((p >= 0.0) & (p <= 1.0))
    assert np.all(e >= 0.0)


def test_weighted_conformal_shapes():
    ref = np.array([0.1, 0.2, 0.3])
    test = np.array([0.05, 0.35])
    wr = np.array([1.0, 2.0, 1.5])
    wt = np.array([0.8, 1.2])
    p = p_full(ref, test, weights_ref=wr, weights_test=wt)
    e, thresholds = e_full(ref, test, 0.2, weights_ref=wr, weights_test=wt)
    assert p.shape == (2,)
    assert e.shape == (2,)
    assert len(thresholds) == 2

import numpy as np

from fcnd.multiple_testing import bh, ebh


def test_bh_known_example():
    p = np.array([0.001, 0.02, 0.03, 0.2])
    k, rej = bh(p, 0.05)
    assert k == 3
    assert set(rej.tolist()) == {0, 1, 2}


def test_ebh_known_example():
    e = np.array([100.0, 40.0, 1.0, 0.5])
    k, rej = ebh(e, 0.1)
    assert k == 2
    assert set(rej.tolist()) == {0, 1}

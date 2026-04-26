# fcnd

`fcnd` is a Python package for full conformal novelty detection with finite-sample false discovery rate (FDR) control. It provides implementations of the full conformal methodology developed in the following paper:


J. Lee, I. Popov, Z. Ren. "Full conformal novelty detection". [(arXiv)](https://arxiv.org/abs/2501.02703)  

Novelty detection is the problem of identifying outliers in a test dataset given a reference dataset of only inliers. The goal is to learn (via ML models) a representation that can distinguish between the outliers in the test dataset and the inliers in the test and reference dataset. 

This package is for method development and is designed to fit into workflows that want to apply or build on top of full conformal novelty detection methods. This repository is _not_ a reproduction repository for the above mentioned paper. Reproduction-specific assets such as job scripts and plotting code are not included. Split conformal baselines _are_ included, in case the application of these methods are also of interest.

## Features

- Full conformal novelty detection via `FCND`
- Split conformal novelty detection via `SCND`
- Model-selection full conformal novelty detection via `MSFCND`
- Weighted conformal p-values and e-values for distribution-shifted settings
- Fast and lightweight conditionally calibrated e-BH boosting via `EBHCC`
- Flexible learner interface via `BaseLearner`, with built-in wrappers for common scikit-learn novelty-detection models
- Numba-backed numerical kernels enabled by default


## Installation

From the repository root:

```bash
pip install -e .
```

For development and tests:

```bash
pip install -e ".[dev]"
pytest
```

For notebook examples:

```bash
pip install -e ".[notebook]"
```

`fcnd` supports Python 3.10, 3.11, and 3.12. The default dependency set includes `numba`, `numpy`, and `scikit-learn`; `scipy` is installed transitively by scikit-learn.

## Quickstart Example

```python
import numpy as np

from fcnd import FCND, IsoForestLearner
from fcnd.synthetic import generate_wset, gen_data

rng = np.random.default_rng(123)
W = generate_wset(size=16, dims=8, random_state=rng)

X_ref = gen_data(W, 60, a=1.0, random_state=rng)
X_test = np.vstack([
    gen_data(W, 30, a=1.0, random_state=rng),
    gen_data(W, 10, a=3.5, random_state=rng),
])

detector = FCND(
    IsoForestLearner(random_state=1, n_estimators=50),
    use_numba=True,
)
result = detector.detect(X_ref, X_test, alpha=0.2, method="ebh")

print(result.rejections)
print(result.n_rejections)
```

Expected output:

```text
[ 1 20 31 32 34 37 39]
7
```

The `detect` method returns a `DetectionResult` with conformal p- or e-values, non-conformity scores, the rejection set, and basic metadata.

#### Example Notebook

The notebook `examples/fcnd_quickstart.ipynb` demonstrates FCND, weighted FCND, and MSFCND in a single workflow.

## Main API

| Object | Purpose |
| --- | --- |
| `FCND` | Full conformal novelty detector for reference and test samples |
| `SCND` | Split conformal novelty detector with separate train/calibration/test inputs |
| `MSFCND` | Model-selection full conformal novelty detector over a candidate learner library |
| `EBHCC` | Streaming conditionally calibrated e-BH booster |
| `IsoForestLearner` | Isolation Forest wrapper with larger scores meaning more anomalous |
| `OneClassSVMLearner` | One-class SVM wrapper with larger scores meaning more anomalous |
| `BaseLearner` | Abstract interface for custom score learners |
| `bh`, `ebh` | BH and e-BH multiple-testing procedures |

## Custom Learners

`FCND`, `SCND`, and `MSFCND` can use any learner that implements `BaseLearner`. A learner must provide:

- `reset()`: return a fresh, unfitted learner state;
- `fit(X, **kwargs)`: fit on an array of observations and return `self`;
- `score(X)`: return a one-dimensional array of nonconformity scores. Larger scores should indicate more anomalous observations.

Example wrapper around a scikit-learn-compatible estimator:

```python
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from fcnd import BaseLearner, FCND


class LOFLearner(BaseLearner):
    def __init__(self, **kwargs):
        self._params = kwargs
        self.model = None
        self.fitted_ = False

    def reset(self):
        self.model = LocalOutlierFactor(novelty=True, **self._params)
        self.fitted_ = False
        return self

    def fit(self, X, **kwargs):
        if self.model is None:
            self.reset()
        self.model.set_params(**kwargs)
        self.model.fit(X)
        self.fitted_ = True
        return self

    def score(self, X):
        if self.model is None:
            raise RuntimeError("Call fit before score.")
        return -np.asarray(self.model.score_samples(X), dtype=float).reshape(-1)


detector = FCND(LOFLearner(n_neighbors=20), use_numba=True)
result = detector.detect(X_ref, X_test, alpha=0.1, method="ebh")
```


## Score Construction Modes

Full conformal methods can construct scores in two main ways:

- In-sample (`IS`) scoring fits the learner once on the combined reference and test data and scores all observations with that fitted learner.
- Leave-one-out (`LOO`) scoring refits the learner after removing each observation and then scores the held-out observation. 

For a single learner, use `FCND(..., leave_one_out=True)` to enable LOO scoring. By default, `FCND` uses IS scoring.

## Model Selection

`MSFCND` accepts a dictionary or sequence of learner objects and performs block-wise model selection before forming conformal values:

```python
from fcnd import MSFCND, IsoForestLearner, OneClassSVMLearner

learners = {
    "if100": IsoForestLearner(random_state=0, n_estimators=100),
    "svm_auto": OneClassSVMLearner(nu=0.05, kernel="rbf", gamma="auto"),
}
training_modes = {"if100": "is", "svm_auto": "loo"}

ms = MSFCND(
    learners,
    alpha=0.1,
    K=10,
    training_modes=training_modes,
    use_numba=True,
)
result = ms.detect(X_ref, X_test, method="ebh")
```

As seen above, `MSFCND` controls the score construction mode through `training_modes`: pass `"is"`, `"loo"`, `"both"`, or a dictionary assigning modes for each learner.


## Weighted FCND

Weighted conformal calibration is available through `load_weights` or the high-level `detect` interface:

```python
result = detector.detect(
    X_ref,
    X_test,
    alpha=0.1,
    method="ebh",
    weights_ref=w_ref,
    weights_test=w_test,
)
```

The weights should encode the relevant density-ratio information for the weighted-exchangeability setting. For further details, see Section 4 of the paper.

## Conditional Calibration

As the full conformal method uses e-values, we use the 
e-BH procedure for the multiple testing algorithm. However, e-BH can be further improved through leveraging an informative conditional distribution with regard to a sufficient statistic [(see here)](https://arxiv.org/abs/2404.17562). 

`EBHCC` implements this ``boosted'' e-BH method, called e-BH-CC (e-BH with Conditional Calibration). It is a fast and lightweight implementation that is suited for conformal novelty detection, reducing memory use and computation time by no longer depending on resampling and Monte Carlo estimation.

The optimized shortcuts in `EBHCC` are deliberately restricted to the compatible built-in design choices used by the package:

- built-in p-value auxiliary statistics for the `prune=True` skipping shortcut;
- the built-in hybrid `hatR` denominator for the `guarantee=True` uniform-improvement shortcut.

For custom conditional-calibration statistics/denominators, use `guarantee=False` and `prune=False` unless the corresponding validity argument has been verified for that statistic/denominator pair. For further details, see Section 3.4 of the paper.


## Citation

If you use `fcnd` in your research or workflow, please consider citing our accompanying paper!

```bibtex
@misc{lee2026fullconformal,
  title = {Full-conformal novelty detection},
  author = {Lee, Junu and Popov, Ilia and Ren, Zhimei},
  year = {2026},
  eprint = {2501.02703v2},
  archivePrefix = {arXiv},
  primaryClass = {stat.ME},
  doi = {10.48550/arXiv.2501.02703},
  url = {https://arxiv.org/abs/2501.02703v2}
}
```

A machine-readable citation is also available in `CITATION.cff`.

## License

`fcnd` is released under the MIT License. See `LICENSE` for details.

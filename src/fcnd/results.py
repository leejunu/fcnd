from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DetectionResult:
    """Container returned by high-level ``detect`` methods."""

    rejections: np.ndarray
    values: np.ndarray
    scores: np.ndarray
    method: str
    alpha: float
    detector: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def rejection_mask(self) -> np.ndarray:
        mask = np.zeros(self.values.shape[0], dtype=bool)
        mask[np.asarray(self.rejections, dtype=int)] = True
        return mask

    @property
    def n_rejections(self) -> int:
        return int(np.asarray(self.rejections).size)

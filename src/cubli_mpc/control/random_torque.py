"""Random-torque controller: worst-case benchmark baseline.

Emits a uniform random torque in [-max_torque, +max_torque] every control
tick, independent of state. Useful as a floor to compare active controllers
against.
"""
from __future__ import annotations

import numpy as np


class RandomController:
    def __init__(self, max_torque: float, seed: int = 0):
        if max_torque <= 0:
            raise ValueError("max_torque must be positive")
        self._max = float(max_torque)
        self._rng = np.random.default_rng(seed)

    def step(self, x_hat: np.ndarray, t: float) -> float:
        return float(self._rng.uniform(-self._max, self._max))

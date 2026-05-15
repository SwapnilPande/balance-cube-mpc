"""Reference trajectories for the NMPC.

A `Reference` is queried by NMPC each tick for the desired (x, u) over
its prediction horizon. Two implementations live here:

- `ConstantReference(target_theta)`: balance task (default upright;
  non-zero target_theta acts like a balance-with-offset setpoint).
- `PeriodicTrajectoryReference(...)`: M3 swing tracking. Loads a
  one-period reference from disk and loops it. (Added in Task 15.)

Both return numpy arrays sized to (horizon+1, 3) for x_ref and
(horizon, 1) for u_ref so the NMPC can consume them uniformly.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Reference(Protocol):
    def at(
        self, t: float, dt: float, horizon: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (x_ref [H+1, 3], u_ref [H, 1]) sampled at t + k*dt."""
        ...


class ConstantReference:
    """Constant reference at (target_theta, 0, 0), zero feedforward torque."""

    def __init__(self, target_theta: float = 0.0):
        self._target_theta = float(target_theta)

    @property
    def target_theta(self) -> float:
        return self._target_theta

    def at(
        self, t: float, dt: float, horizon: int
    ) -> tuple[np.ndarray, np.ndarray]:
        x_ref = np.zeros((horizon + 1, 3), dtype=np.float64)
        x_ref[:, 0] = self._target_theta
        u_ref = np.zeros((horizon, 1), dtype=np.float64)
        return x_ref, u_ref

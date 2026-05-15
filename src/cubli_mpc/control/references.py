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


class PeriodicTrajectoryReference:
    """A one-period reference loaded from disk; loops at the period.

    Linear interpolation between samples in the saved time grid lets the
    NMPC use any dt independent of the planner's `n_segments`.
    """

    def __init__(self, t: np.ndarray, x_ref: np.ndarray, u_ref: np.ndarray):
        self._t = np.asarray(t, dtype=np.float64)
        self._x = np.asarray(x_ref, dtype=np.float64)
        self._u = np.asarray(u_ref, dtype=np.float64)
        self._period = float(self._t[-1] - self._t[0])
        if self._period <= 0:
            raise ValueError("trajectory time grid must have positive span")
        # Midpoints for the u-grid (u has one fewer entry than t).
        self._t_u = 0.5 * (self._t[:-1] + self._t[1:])

    @classmethod
    def from_file(cls, path: "str | Path") -> "PeriodicTrajectoryReference":
        from pathlib import Path  # noqa: F401 — imported for type hint clarity
        data = np.load(path)
        return cls(data["t"], data["x_ref"], data["u_ref"])

    def at(
        self, t: float, dt: float, horizon: int
    ) -> tuple[np.ndarray, np.ndarray]:
        x_ref = np.empty((horizon + 1, 3), dtype=np.float64)
        u_ref = np.empty((horizon, 1), dtype=np.float64)
        for k in range(horizon + 1):
            tk = (t + k * dt) % self._period
            x_ref[k] = self._interp_x(tk)
        for k in range(horizon):
            tk = (t + k * dt) % self._period
            u_ref[k] = self._interp_u(tk)
        return x_ref, u_ref

    def _interp_x(self, tq: float) -> np.ndarray:
        # np.interp does linear interp on 1D; loop over 3 components.
        return np.array([
            np.interp(tq, self._t, self._x[:, 0]),
            np.interp(tq, self._t, self._x[:, 1]),
            np.interp(tq, self._t, self._x[:, 2]),
        ])

    def _interp_u(self, tq: float) -> np.ndarray:
        return np.array([np.interp(tq, self._t_u, self._u[:, 0])])

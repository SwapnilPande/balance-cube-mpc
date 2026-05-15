"""External-torque disturbance plans for simulation.

A `DisturbancePlan` is queried each control tick for the torque to inject
on the tilt DOF via `CubliEnv.apply_disturbance_torque`. Two ingredients
compose into one plan:

- A list of `StepDisturbance` rectangles (constant torque between a
  start time and end time). Useful for "kick the cube at t=2s with
  0.05 Nm for half a second".
- An optional Poisson impulse stream (±magnitude one-tick spikes at an
  average rate of N per second). Useful for stress testing.

The two parts add; the plan is stateless w.r.t. step disturbances and
seeded for the impulse stream.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class StepDisturbance:
    """Constant external torque applied between start_t and start_t+duration."""

    magnitude: float
    start_t: float
    duration: float

    def at(self, t: float) -> float:
        if self.start_t <= t < self.start_t + self.duration:
            return float(self.magnitude)
        return 0.0


@dataclass
class DisturbancePlan:
    """Sum of zero-or-more `StepDisturbance`s plus an optional Poisson
    impulse stream of ±impulse_magnitude (one tick wide).

    The impulse stream uses Bernoulli-per-tick with p = rate * dt as a
    discrete approximation to a Poisson process; for `rate * dt << 1`
    this is exact to second order. The RNG advances once per `at()`
    call when the stream is enabled, so reproducibility is per-call-order.
    """

    steps: list[StepDisturbance] = field(default_factory=list)
    impulse_magnitude: float = 0.0
    impulse_rate_per_s: float = 0.0
    seed: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def reset(self) -> None:
        """Reseed the impulse RNG to its original seed."""
        self._rng = np.random.default_rng(self.seed)

    def at(self, t: float, dt: float) -> float:
        """Return the disturbance torque to apply for the tick starting at t."""
        tau = 0.0
        for s in self.steps:
            tau += s.at(t)
        if self.impulse_magnitude > 0.0 and self.impulse_rate_per_s > 0.0:
            p_fire = float(self.impulse_rate_per_s) * float(dt)
            if self._rng.random() < p_fire:
                sign = 1.0 if self._rng.random() < 0.5 else -1.0
                tau += sign * float(self.impulse_magnitude)
        return float(tau)

    @property
    def is_active(self) -> bool:
        """True if the plan can produce any non-zero torque."""
        if any(s.magnitude != 0.0 and s.duration > 0.0 for s in self.steps):
            return True
        return (self.impulse_magnitude > 0.0
                and self.impulse_rate_per_s > 0.0)

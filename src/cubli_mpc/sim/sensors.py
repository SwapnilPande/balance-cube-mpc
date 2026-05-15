"""Sensor model: additive gaussian noise + ring-buffer delay.

The model takes a clean state vector (whatever the caller wants to treat as
"sensor reading" -- typically `[theta_body, theta_dot_body, omega_wheel]`)
and returns a noisy, optionally delayed copy. Noise is sampled per call from
an internal Generator (seeded for reproducibility); delay is implemented by
buffering the last `delay_steps` readings and returning the oldest.

This is a deliberate simplification: it noises the *estimator output* rather
than each raw IMU channel. Good enough for stress-testing the controller's
robustness; swap for a per-channel sensor + filter pipeline when the time
comes (see `docs/superpowers/specs/2026-05-14-cubli-mpc-design.md`, M4).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SensorConfig:
    """Per-component noise std + uniform delay."""
    noise_std: tuple[float, ...] = ()  # one entry per observation component
    delay_steps: int = 0
    seed: int = 0


class SensorModel:
    def __init__(self, cfg: SensorConfig):
        if cfg.delay_steps < 0:
            raise ValueError("delay_steps must be >= 0")
        if any(s < 0 for s in cfg.noise_std):
            raise ValueError("noise_std entries must be >= 0")
        self._cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)
        self._buf: deque[np.ndarray] = deque(maxlen=cfg.delay_steps + 1)

    def reset(self) -> None:
        """Clear the delay buffer. Call at the start of each episode."""
        self._buf.clear()

    def observe(self, x_true: np.ndarray) -> np.ndarray:
        """Apply noise + delay. First call after reset() returns the current
        reading (so a fresh episode starts with a valid observation rather
        than zeros)."""
        x = np.asarray(x_true, dtype=np.float64)
        if self._cfg.noise_std:
            std = np.asarray(self._cfg.noise_std, dtype=np.float64)
            if std.shape != x.shape:
                raise ValueError(
                    f"noise_std shape {std.shape} != observation shape "
                    f"{x.shape}")
            x = x + self._rng.normal(0.0, std)
        if self._cfg.delay_steps == 0:
            return x
        if not self._buf:
            # Pre-fill the buffer so the first observation isn't a phantom
            # delayed zero -- a fresh episode "remembers" its current state.
            for _ in range(self._cfg.delay_steps + 1):
                self._buf.append(x.copy())
            return x.copy()
        self._buf.append(x)
        return self._buf[0].copy()

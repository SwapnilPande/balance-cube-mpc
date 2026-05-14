"""Controller protocol.

Every controller takes a 4D state estimate
    x_hat = [theta_body, theta_dot_body, theta_wheel, theta_dot_wheel]
and a wall-clock time and returns a commanded wheel torque in Nm. The runner
is responsible for clipping/saturation if the actuator can be over-driven;
controllers may self-saturate.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Controller(Protocol):
    def step(self, x_hat: np.ndarray, t: float) -> float:
        """Return commanded wheel torque (Nm)."""
        ...

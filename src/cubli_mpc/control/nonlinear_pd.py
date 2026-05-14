"""Energy-aware nonlinear PD baseline controller (M1).

Structure:
  1. Outer loop: balance-point offset = -k_wheel * omega_wheel, clipped.
  2. Gravity feedforward cancels the cube's gravity moment.
  3. Inner loop: kp*sin(theta - theta_balance) + kd*theta_dot.
  4. Hard saturation at +/- max_torque.

The sin() in the inner loop keeps the PD well-behaved at large angles.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from cubli_mpc.config import HardwareConfig


@dataclass(frozen=True)
class NonlinearPDGains:
    kp: float
    kd: float
    k_wheel: float
    max_balance_tilt: float  # rad
    max_torque: float        # Nm


class NonlinearPDController:
    def __init__(self, gains: NonlinearPDGains, hw: HardwareConfig):
        self._g = gains
        self._mgL = hw.gravity_moment_coefficient

    def step(self, x_hat: np.ndarray, t: float) -> float:
        theta = float(x_hat[0])
        theta_dot = float(x_hat[1])
        omega_w = float(x_hat[3])
        g = self._g

        theta_balance = -g.k_wheel * omega_w
        if theta_balance > g.max_balance_tilt:
            theta_balance = g.max_balance_tilt
        elif theta_balance < -g.max_balance_tilt:
            theta_balance = -g.max_balance_tilt

        e = theta - theta_balance
        tau_grav = self._mgL * math.sin(theta)
        tau_pd = g.kp * math.sin(e) + g.kd * theta_dot
        tau = tau_grav + tau_pd

        if tau > g.max_torque:
            tau = g.max_torque
        elif tau < -g.max_torque:
            tau = -g.max_torque
        return tau

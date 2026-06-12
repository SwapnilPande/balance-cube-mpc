"""Metronome controller: drive the cubli into a sustained clock-like swing.

Goal: make the cube tick back and forth about upright (theta = 0) with a
fixed full-cycle period (target 1.0 s), like a clock pendulum.

Approach — feedback-linearized limit-cycle oscillator:

  1. Pick a desired body angular acceleration that defines an *autonomous
     oscillator* with a stable limit cycle:

         theta_ddot_des = -omega0^2 * r(theta)
                          - mu * (energy - A^2) * theta_dot

     where r(theta) is the restoring shape (theta or sin(theta)),
     `energy = theta^2 + (theta_dot/omega0)^2` is a Lyapunov-like amplitude
     measure, and the `-mu*(energy - A^2)*theta_dot` term is *negative*
     damping inside the target ellipse (pumps energy in) and *positive*
     damping outside it (bleeds energy out). The result is a stable limit
     cycle at amplitude ~A and angular frequency ~omega0, so the realized
     full period is ~ 2*pi/omega0.

  2. Invert the (analytic) plant model to realize that acceleration with
     wheel torque. From the MPC dynamics model:

         theta_ddot = (mgL*sin(theta) - b_e*theta_dot - tau + b_w*omega_w)/I_b
     =>  tau = mgL*sin(theta) - b_e*theta_dot + b_w*omega_w
              - I_b * theta_ddot_des

     The mgL*sin(theta) term cancels gravity; the rest imposes the desired
     oscillator. MuJoCo's true dynamics differ slightly from this analytic
     model (wheel-mass coupling, contact, discretization), so the realized
     period drifts from 2*pi/omega0 — closing that gap to land at exactly
     1.0 s is the tuning problem.

The controller is intentionally a few plain parameters so the Onyx loop can
tune them directly. `amplitude_rad` also seeds the initial condition in the
eval harness (start the cube at one extreme, at rest).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from cubli_mpc.config import HardwareConfig


@dataclass(frozen=True)
class MetronomeGains:
    omega0: float          # rad/s — sets nominal period 2*pi/omega0
    amplitude_rad: float   # target swing amplitude
    mu: float              # limit-cycle convergence rate (energy feedback)
    use_sin_restoring: bool = False  # sin(theta) (pendulum-like) vs linear
    max_torque: float = 0.20         # Nm hard saturation
    # Effective tilt inertia used in the inversion, as a multiple of the
    # cube-only I_b. MuJoCo's true tilt inertia + wheel-torque coupling make
    # the realized tau->theta_ddot gain softer than 1/I_b; getting this right
    # makes omega0 map 1:1 to realized frequency and lets the energy regulator
    # hold amplitude.
    inertia_scale: float = 1.0
    # Scale on the gravity feedforward. The config's mgL counts only the cube
    # mass; MuJoCo's true gravity moment also includes the wheel (~1.25x).
    # Under-cancelling leaves residual destabilizing gravity that softens the
    # oscillator amplitude-dependently, so dialing this in decouples period
    # from amplitude.
    gravity_scale: float = 1.0


# --- Tunable defaults (the Onyx loop edits these) -----------------------------
# omega0 = 2*pi / T_target with T_target = 1.0 s as the starting guess.
DEFAULT_GAINS = MetronomeGains(
    omega0=4.31,
    amplitude_rad=math.radians(10.0),
    mu=20.0,
    use_sin_restoring=False,
    max_torque=0.20,
    inertia_scale=2.54,
    gravity_scale=1.25,
)


class MetronomeController:
    def __init__(self, hw: HardwareConfig, gains: MetronomeGains = DEFAULT_GAINS):
        self._g = gains
        self._I_b = hw.cube_inertia_about_edge
        self._mgL = hw.gravity_moment_coefficient
        self._b_e = hw.edge_bearing_damping
        self._b_w = hw.wheel_bearing_damping

    @property
    def amplitude(self) -> float:
        """Swing amplitude (rad); used to seed the initial condition."""
        return self._g.amplitude_rad

    def reset(self) -> None:  # parity with other controllers
        pass

    def step(self, x_hat: np.ndarray, t: float) -> float:
        g = self._g
        theta = float(x_hat[0])
        theta_dot = float(x_hat[1])
        omega_w = float(x_hat[3])

        w0 = g.omega0
        restoring = math.sin(theta) if g.use_sin_restoring else theta
        energy = theta * theta + (theta_dot / w0) ** 2
        theta_ddot_des = (-w0 * w0 * restoring
                          - g.mu * (energy - g.amplitude_rad ** 2) * theta_dot)

        tau = (g.gravity_scale * self._mgL * math.sin(theta)
               - self._b_e * theta_dot
               + self._b_w * omega_w
               - g.inertia_scale * self._I_b * theta_ddot_des)

        if tau > g.max_torque:
            tau = g.max_torque
        elif tau < -g.max_torque:
            tau = -g.max_torque
        return tau

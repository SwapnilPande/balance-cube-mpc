"""Tests for the CasADi symbolic dynamics."""
import math

import numpy as np
import pytest

from cubli_mpc.config import HardwareConfig
from cubli_mpc.model.dynamics import make_continuous_dynamics


def _make_hw() -> HardwareConfig:
    return HardwareConfig(
        cube_side_length_m=0.10,
        cube_mass_kg=0.40,
        wheel_mass_kg=0.10,
        wheel_radius_m=0.035,
        wheel_thickness_m=0.010,
        motor_max_torque_nm=0.20,
        motor_max_speed_rad_s=600.0,
        motor_torque_constant=0.0,
        edge_bearing_damping=0.0,
        wheel_bearing_damping=0.0,
    )


def test_continuous_dynamics_signature():
    f = make_continuous_dynamics(_make_hw())
    x = np.array([0.0, 0.0, 0.0])
    u = np.array([0.0])
    x_dot = np.array(f(x, u)).flatten()
    assert x_dot.shape == (3,)


def test_dynamics_upright_zero_input_is_equilibrium():
    """θ=0, θ̇=0, ω_w=0, τ=0 should give all-zero x_dot."""
    f = make_continuous_dynamics(_make_hw())
    x_dot = np.array(f([0.0, 0.0, 0.0], [0.0])).flatten()
    assert np.allclose(x_dot, 0.0, atol=1e-12)


def test_dynamics_positive_tilt_falls_outward():
    """θ > 0 with no torque should give θ̈ > 0 (cube falls outward)."""
    f = make_continuous_dynamics(_make_hw())
    x_dot = np.array(f([0.1, 0.0, 0.0], [0.0])).flatten()
    assert x_dot[1] > 0.0  # theta_ddot > 0


def test_dynamics_positive_torque_spins_wheel_positive():
    """At rest upright, positive τ should give positive ω̇_w."""
    f = make_continuous_dynamics(_make_hw())
    x_dot = np.array(f([0.0, 0.0, 0.0], [0.05])).flatten()
    assert x_dot[2] > 0.0  # omega_w_dot > 0


def test_dynamics_positive_torque_pushes_body_negative():
    """Reaction: positive τ on the wheel pushes the body in the negative
    direction (θ̈ < 0). This is the whole point of a reaction wheel."""
    f = make_continuous_dynamics(_make_hw())
    x_dot = np.array(f([0.0, 0.0, 0.0], [0.05])).flatten()
    assert x_dot[1] < 0.0  # theta_ddot < 0

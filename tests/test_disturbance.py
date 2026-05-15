"""Tests for the external-torque disturbance API on CubliEnv."""
import math

import numpy as np
import pytest

from cubli_mpc.config import HardwareConfig, SimConfig
from cubli_mpc.sim.env import CubliEnv


def _make_env() -> CubliEnv:
    hw = HardwareConfig(
        cube_side_length_m=0.10,
        cube_mass_kg=0.40,
        wheel_mass_kg=0.10,
        wheel_radius_m=0.035,
        wheel_thickness_m=0.010,
        motor_max_torque_nm=0.20,
        motor_max_speed_rad_s=600.0,
        motor_torque_constant=0.01,
        edge_bearing_damping=0.0,
        wheel_bearing_damping=0.0,
    )
    sim = SimConfig(dt_sim=0.001, dt_control=0.001)
    return CubliEnv(hw, sim)


def test_disturbance_torque_tilts_body():
    """At rest upright, a positive external torque must produce positive
    tilt acceleration."""
    env = _make_env()
    env.reset()
    env.apply_disturbance_torque(0.5)
    for _ in range(50):
        env.step()
    theta = env.state()[0]
    assert theta > 1e-3


def test_disturbance_sign_flips_response():
    env = _make_env()
    env.reset()
    env.apply_disturbance_torque(-0.5)
    for _ in range(50):
        env.step()
    assert env.state()[0] < -1e-3


def test_reset_clears_disturbance():
    env = _make_env()
    env.reset()
    env.apply_disturbance_torque(0.5)
    for _ in range(10):
        env.step()
    env.reset()
    # After reset, no external torque should remain.
    for _ in range(100):
        env.step()
    theta = env.state()[0]
    # Pure gravity with zero initial tilt -> stays upright (within numerical
    # noise from the unstable equilibrium).
    assert abs(theta) < 1e-3


def test_disturbance_persists_across_steps():
    """A single apply_disturbance_torque() should persist until overwritten."""
    env = _make_env()
    env.reset()
    env.apply_disturbance_torque(0.3)
    for _ in range(20):
        env.step()
    theta_after_persistent = env.state()[0]

    env.reset()
    env.apply_disturbance_torque(0.3)
    env.step()
    env.apply_disturbance_torque(0.0)
    for _ in range(19):
        env.step()
    theta_after_pulse = env.state()[0]

    # Persistent forcing should leave the body more tilted than a single
    # impulse.
    assert theta_after_persistent > theta_after_pulse

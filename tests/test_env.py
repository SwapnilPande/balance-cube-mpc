"""Tests for the CubliEnv MuJoCo wrapper."""
import math
import numpy as np
import pytest
from cubli_mpc.config import HardwareConfig, SimConfig
from cubli_mpc.sim.env import CubliEnv


def _make_hw(**overrides) -> HardwareConfig:
    defaults = dict(
        cube_side_length_m=0.10,
        cube_mass_kg=0.40,
        wheel_mass_kg=0.10,
        wheel_radius_m=0.035,
        wheel_thickness_m=0.010,
        motor_max_torque_nm=0.20,
        motor_max_speed_rad_s=600.0,
        motor_torque_constant=0.01,
        edge_bearing_damping=0.0,   # zero for clean dynamics tests
        wheel_bearing_damping=0.0,
    )
    defaults.update(overrides)
    return HardwareConfig(**defaults)


def test_env_constructs():
    env = CubliEnv(_make_hw(), SimConfig())
    assert env is not None


def test_state_shape_and_initial_zero():
    env = CubliEnv(_make_hw(), SimConfig())
    env.reset()
    s = env.state()
    assert s.shape == (4,)
    np.testing.assert_allclose(s, np.zeros(4), atol=1e-12)


def test_reset_sets_initial_conditions():
    env = CubliEnv(_make_hw(), SimConfig())
    env.reset(theta0=0.1, theta_dot0=0.2,
              wheel_angle0=0.3, wheel_speed0=0.4)
    s = env.state()
    np.testing.assert_allclose(s, [0.1, 0.2, 0.3, 0.4], atol=1e-12)


def test_passive_cube_falls_over():
    """With zero motor torque from a small initial tilt, the body should
    accelerate away from upright due to gravity.
    """
    env = CubliEnv(_make_hw(), SimConfig())
    env.reset(theta0=0.05)
    for _ in range(100):  # 100 ms with dt=1ms
        env.apply_torque(0.0)
        env.step()
    s = env.state()
    # theta and theta_dot should both have grown in the positive direction
    assert s[0] > 0.05
    assert s[1] > 0.0


def test_motor_torque_spins_wheel():
    """Applying constant torque with the body locked at theta=0 (small
    initial perturbation, very short time) should spin the wheel up.
    """
    env = CubliEnv(_make_hw(), SimConfig())
    env.reset(theta0=0.0)
    for _ in range(50):
        env.apply_torque(0.05)
        env.step()
    s = env.state()
    assert s[3] > 0.0   # wheel speed positive
    # By Newton's 3rd, applying +tau to wheel reacts as -tau on body, so
    # body should tilt negative (assuming gravity hasn't dominated yet)
    assert s[0] < 0.0


def test_gravity_torque_matches_analytical():
    """At theta = pi/4, with all velocities zero and no motor torque, the
    angular acceleration of the body should match m*g*L*sin(theta)/I_edge.
    """
    hw = _make_hw()
    env = CubliEnv(hw, SimConfig())
    env.reset(theta0=math.pi / 4)
    env.apply_torque(0.0)
    env.step()
    # After one step of dt=1ms, theta_dot ~= alpha * dt, where alpha is
    # the analytical angular acceleration.
    s = env.state()
    alpha_expected = (hw.gravity_moment_coefficient
                      * math.sin(math.pi / 4)
                      / hw.cube_inertia_about_edge)
    # Tolerance is generous because MuJoCo composes cube + wheel inertia
    # (wheel adds a small contribution); we just want the sign and rough
    # magnitude.
    assert s[1] > 0.0
    assert s[1] == pytest.approx(alpha_expected * 0.001, rel=0.1)


def test_torque_saturation():
    """Commanding a torque above ctrlrange should be clipped by MuJoCo."""
    hw = _make_hw(motor_max_torque_nm=0.10)
    env = CubliEnv(hw, SimConfig())
    env.reset()
    env.apply_torque(10.0)  # way over the 0.1 Nm limit
    env.step()
    # Wheel speed after 1ms should be no greater than what 0.1 Nm produces
    # over 1ms = 0.1 / I_wheel * dt
    s = env.state()
    max_omega = 0.10 / hw.wheel_inertia_about_spin_axis * 0.001
    # Allow 20% slack for MuJoCo's integrator behaviour
    assert s[3] <= max_omega * 1.2

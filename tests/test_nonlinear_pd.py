"""Tests for the nonlinear PD baseline controller."""
import math
import numpy as np
import pytest
from cubli_mpc.config import HardwareConfig
from cubli_mpc.control.nonlinear_pd import (
    NonlinearPDController, NonlinearPDGains,
)


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
        edge_bearing_damping=0.0,
        wheel_bearing_damping=0.0,
    )
    defaults.update(overrides)
    return HardwareConfig(**defaults)


def _make_gains(**overrides) -> NonlinearPDGains:
    defaults = dict(
        kp=0.5, kd=0.02, k_wheel=0.0,
        max_balance_tilt=math.radians(2.0),
        max_torque=0.20,
    )
    defaults.update(overrides)
    return NonlinearPDGains(**defaults)


def test_returns_zero_at_upright_with_zero_rates():
    hw = _make_hw()
    c = NonlinearPDController(_make_gains(), hw)
    tau = c.step(np.zeros(4), t=0.0)
    assert tau == pytest.approx(0.0)


def test_returns_positive_torque_for_positive_tilt():
    """theta > 0 means the body has tilted past upright in +y direction.
    Gravity FF + positive sin(e) term should produce tau > 0.
    """
    hw = _make_hw()
    c = NonlinearPDController(_make_gains(kp=1.0), hw)
    tau = c.step(np.array([0.05, 0.0, 0.0, 0.0]), t=0.0)
    assert tau > 0.0


def test_saturation_clips_to_max_torque():
    hw = _make_hw()
    c = NonlinearPDController(_make_gains(max_torque=0.10, kp=10.0), hw)
    tau = c.step(np.array([0.5, 0.0, 0.0, 0.0]), t=0.0)
    assert abs(tau) <= 0.10 + 1e-12


def test_wheel_speed_offset_balance_point():
    """With k_wheel > 0, omega_w > 0 should produce torque of opposite
    sign vs omega_w < 0 (the balance point shifts symmetrically).
    """
    hw = _make_hw()
    c = NonlinearPDController(
        _make_gains(kp=1.0, kd=0.0, k_wheel=0.001,
                    max_balance_tilt=math.radians(5.0)),
        hw,
    )
    tau_pos_wheel = c.step(np.array([0.0, 0.0, 0.0, 100.0]), t=0.0)
    tau_neg_wheel = c.step(np.array([0.0, 0.0, 0.0, -100.0]), t=0.0)
    assert tau_pos_wheel * tau_neg_wheel < 0


def test_at_offset_balance_point_torque_bleeds_wheel():
    """When the body has reached the offset balance point, the controller
    should produce a torque opposing the wheel direction.
    """
    hw = _make_hw()
    c = NonlinearPDController(
        _make_gains(kp=1.0, kd=0.0, k_wheel=0.001,
                    max_balance_tilt=math.radians(10.0)),
        hw,
    )
    omega_w = 100.0
    theta_balance = -0.001 * omega_w  # -0.1 rad, within max_balance_tilt
    tau = c.step(np.array([theta_balance, 0.0, 0.0, omega_w]), t=0.0)
    # tau = m*g*L*sin(theta_balance) -- negative, opposing positive omega_w
    assert tau < 0


def test_damping_term():
    """Positive theta_dot should add positive torque (push wheel + reacts
    body negative, slowing the falling motion in +theta direction).
    """
    hw = _make_hw()
    c = NonlinearPDController(_make_gains(kp=0.0, kd=0.1), hw)
    tau = c.step(np.array([0.0, 0.5, 0.0, 0.0]), t=0.0)
    assert tau == pytest.approx(0.1 * 0.5)

"""Tests for the NMPC controller (M2 balance, no warm-start, no fallback)."""
import math

import numpy as np
import pytest

from cubli_mpc.config import HardwareConfig
from cubli_mpc.control.nmpc import NMPCConfig, NMPCController
from cubli_mpc.control.references import ConstantReference


def _make_hw() -> HardwareConfig:
    return HardwareConfig(
        cube_side_length_m=0.10,
        cube_mass_kg=0.40,
        wheel_mass_kg=0.10,
        wheel_radius_m=0.035,
        wheel_thickness_m=0.010,
        motor_max_torque_nm=0.20,
        motor_max_speed_rad_s=600.0,
        motor_torque_constant=0.01,
        edge_bearing_damping=1e-4,
        wheel_bearing_damping=1e-5,
    )


def test_nmpc_returns_scalar_torque():
    nmpc = NMPCController(_make_hw(), NMPCConfig(horizon_steps=10))
    x_hat = np.array([0.05, 0.0, 0.0, 0.0])  # 4D state from runner
    tau = nmpc.step(x_hat, t=0.0)
    assert isinstance(tau, float)


def test_nmpc_respects_torque_limit():
    hw = _make_hw()
    nmpc = NMPCController(hw, NMPCConfig(horizon_steps=10))
    # Large initial tilt should produce a torque at or near the limit.
    x_hat = np.array([math.radians(45.0), 0.0, 0.0, 0.0])
    tau = nmpc.step(x_hat, t=0.0)
    assert abs(tau) <= hw.motor_max_torque_nm + 1e-9


def test_nmpc_positive_tilt_produces_positive_torque():
    """For an unstable equilibrium with positive tilt and zero wheel speed,
    the NMPC should command positive torque (which decelerates the body's
    fall via the reaction)."""
    nmpc = NMPCController(_make_hw(), NMPCConfig(horizon_steps=20))
    x_hat = np.array([math.radians(5.0), 0.0, 0.0, 0.0])
    tau = nmpc.step(x_hat, t=0.0)
    assert tau > 0.0


def test_nmpc_uses_supplied_reference():
    """With ConstantReference(target_theta=0.05), the NMPC's commanded
    torque at theta=0.05 should be smaller in magnitude than at theta=0
    (because the latter is now an *error*)."""
    nmpc = NMPCController(
        _make_hw(),
        NMPCConfig(horizon_steps=20),
        reference=ConstantReference(target_theta=0.05),
    )
    tau_at_target = abs(nmpc.step(np.array([0.05, 0.0, 0.0, 0.0]), t=0.0))
    tau_at_upright = abs(nmpc.step(np.array([0.0, 0.0, 0.0, 0.0]), t=0.0))
    assert tau_at_target < tau_at_upright


def test_nmpc_target_theta_property():
    nmpc = NMPCController(
        _make_hw(), NMPCConfig(horizon_steps=10),
        reference=ConstantReference(target_theta=0.1),
    )
    assert nmpc.target_theta == pytest.approx(0.1)

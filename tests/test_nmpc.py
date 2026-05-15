"""Tests for the NMPC controller (M2 balance, no warm-start, no fallback)."""
import math

import numpy as np
import pytest

from cubli_mpc.config import HardwareConfig, SimConfig
from cubli_mpc.control.nmpc import NMPCConfig, NMPCController
from cubli_mpc.control.references import ConstantReference
from cubli_mpc.runner import Runner
from cubli_mpc.sim.env import CubliEnv


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


class _StepReference:
    """Helper: returns target_theta_initial for k<switch_k, target_theta_late for k>=switch_k."""

    def __init__(self, target_theta_initial: float, target_theta_late: float,
                 switch_k: int = 5):
        self._initial = float(target_theta_initial)
        self._late = float(target_theta_late)
        self._switch = int(switch_k)

    def at(self, t, dt, horizon):
        x_ref = np.zeros((horizon + 1, 3), dtype=np.float64)
        for k in range(horizon + 1):
            x_ref[k, 0] = self._initial if k < self._switch else self._late
        u_ref = np.zeros((horizon, 1), dtype=np.float64)
        return x_ref, u_ref


def test_nmpc_time_varying_reference_changes_torque():
    """A reference that jumps to a non-zero target later in the horizon
    should produce a different commanded torque than a constant-zero
    reference, even when the current state is upright."""
    hw = _make_hw()
    cfg = NMPCConfig(horizon_steps=20)
    flat = NMPCController(hw, cfg, reference=ConstantReference(target_theta=0.0))
    rising = NMPCController(hw, cfg, reference=_StepReference(0.0, 0.1, switch_k=5))
    x_hat = np.array([0.0, 0.0, 0.0, 0.0])
    tau_flat = flat.step(x_hat, t=0.0)
    rising.step(x_hat, t=0.0)  # warm or not, just run
    tau_rising = rising.step(x_hat, t=0.0)
    # Step reference should command non-zero torque to start moving theta
    # toward the later target; constant-zero ref should command near zero.
    assert abs(tau_rising - tau_flat) > 1e-3, (
        f"tau_flat={tau_flat:.5f}, tau_rising={tau_rising:.5f}"
    )


def test_nmpc_recovers_from_15deg_tilt():
    hw = _make_hw()
    sim = SimConfig(dt_sim=0.001, dt_control=0.01)
    env = CubliEnv(hw, sim)
    env.reset(theta0=math.radians(15.0))

    nmpc = NMPCController(hw, NMPCConfig(horizon_steps=40, dt=sim.dt_control))
    runner = Runner(env, nmpc, sim)
    log = runner.run(duration_s=5.0)

    # Final state should be near upright.
    theta_final = float(log["theta"][-1])
    assert abs(theta_final) < math.radians(2.0), (
        f"final theta = {math.degrees(theta_final):.2f} deg"
    )
    # Should not have fallen at any point.
    theta_max = float(np.max(np.abs(log["theta"])))
    assert theta_max < math.radians(20.0), (
        f"max theta = {math.degrees(theta_max):.2f} deg"
    )

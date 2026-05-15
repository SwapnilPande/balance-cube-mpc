"""Tests for the NMPC controller (M2 balance, warm-start, fallback)."""
import math
import time

import numpy as np
import pytest

from cubli_mpc.config import HardwareConfig, SimConfig
from cubli_mpc.control.nmpc import NMPCConfig, NMPCController
from cubli_mpc.control.nonlinear_pd import NonlinearPDController, NonlinearPDGains
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


def test_nmpc_warm_start_speeds_up_subsequent_solves():
    """Warm-started solves should be substantially faster than cold solves.

    Strategy: use a large tilt angle (30 deg) so cold start needs many more
    IPOPT iterations than a near-optimal warm start.  Throw away one call to
    eliminate CasADi/IPOPT JIT overhead; measure a true cold solve via
    reset(); then average several warm solves for robustness.
    """
    nmpc = NMPCController(_make_hw(), NMPCConfig(horizon_steps=40))
    # 30 degrees: cold start needs ~18 IPOPT iterations from the zero
    # trajectory guess, warm start converges in ~6 from the cached solution.
    x_hat = np.array([math.radians(30.0), 0.0, 0.0, 0.0])

    # Throw-away call: absorbs JIT / first-call overhead.
    nmpc.step(x_hat, t=0.0)

    # True cold solve: reset drops the previous solution and dual variables.
    nmpc.reset()
    t0 = time.perf_counter()
    nmpc.step(x_hat, t=0.0)
    cold_s = time.perf_counter() - t0

    # Warm solves: previous primal+dual solution is cached from the cold call.
    t0 = time.perf_counter()
    for _ in range(5):
        nmpc.step(x_hat, t=0.0)
    warm_avg_s = (time.perf_counter() - t0) / 5

    # Allow a generous margin; warm should be at least half cold.
    assert warm_avg_s < cold_s * 0.6, (
        f"cold={cold_s*1000:.1f} ms, warm avg={warm_avg_s*1000:.1f} ms"
    )


def _make_fallback_pd(hw):
    return NonlinearPDController(
        NonlinearPDGains(kp=0.5, kd=0.05, k_wheel=1e-4,
                         max_balance_tilt=math.radians(2.0),
                         max_torque=hw.motor_max_torque_nm),
        hw,
    )


def test_nmpc_fallback_counter_starts_at_zero():
    nmpc = NMPCController(_make_hw(), NMPCConfig(horizon_steps=10))
    assert nmpc.fallback_count == 0


def test_nmpc_fallback_returns_pd_value_on_solver_failure():
    """Force a failure by setting ipopt_max_iter=0; verify the fallback
    PD value is returned and the counter increments."""
    hw = _make_hw()
    pd = _make_fallback_pd(hw)
    nmpc = NMPCController(
        hw, NMPCConfig(horizon_steps=10, ipopt_max_iter=0),
        fallback_controller=pd,
    )
    x_hat = np.array([math.radians(10.0), 0.5, 0.0, 0.0])
    tau = nmpc.step(x_hat, t=0.0)
    expected = pd.step(x_hat, 0.0)
    assert tau == pytest.approx(expected)
    assert nmpc.fallback_count == 1


def test_nmpc_without_fallback_returns_zero_on_solver_failure():
    """If no fallback is provided, the NMPC returns 0 on failure and
    still counts the failure."""
    nmpc = NMPCController(
        _make_hw(), NMPCConfig(horizon_steps=10, ipopt_max_iter=0),
    )
    x_hat = np.array([math.radians(10.0), 0.5, 0.0, 0.0])
    tau = nmpc.step(x_hat, t=0.0)
    assert tau == 0.0
    assert nmpc.fallback_count == 1


def test_nmpc_tracks_swing_trajectory(tmp_path):
    """Plan a 1.5 s / 10 deg swing and verify NMPC tracking holds RMS
    theta-error well under the amplitude over 3 s."""
    from cubli_mpc.control.references import PeriodicTrajectoryReference
    from cubli_mpc.control.swing_planner import SwingTrajConfig, plan_swing

    hw = _make_hw()
    swing_path = tmp_path / "swing.npz"
    plan_swing(hw, SwingTrajConfig(
        period_s=1.5, theta_target_rad=math.radians(10.0),
        n_segments=80, save_path=swing_path,
    ))
    ref = PeriodicTrajectoryReference.from_file(swing_path)

    sim = SimConfig(dt_sim=0.001, dt_control=0.01)
    env = CubliEnv(hw, sim)
    env.reset(theta0=math.radians(10.0))  # start of swing

    nmpc = NMPCController(
        hw, NMPCConfig(horizon_steps=40, dt=sim.dt_control),
        reference=ref,
    )
    runner = Runner(env, nmpc, sim)
    log = runner.run(duration_s=3.0)

    theta = log["theta"]
    # Build the planned theta at the same time grid, then take RMS error.
    t = log["t"]
    target = np.array([ref._interp_x(float(ti) % 1.5)[0] for ti in t])
    err = theta - target
    rms_err_deg = math.degrees(float(np.sqrt(np.mean(err * err))))
    assert rms_err_deg < 5.0, f"RMS tracking error {rms_err_deg:.2f} deg"

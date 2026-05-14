"""Tests for the Runner sim loop."""
import math
import numpy as np
import pytest
from cubli_mpc.config import HardwareConfig, SimConfig
from cubli_mpc.control.nonlinear_pd import (
    NonlinearPDController, NonlinearPDGains,
)
from cubli_mpc.sim.env import CubliEnv
from cubli_mpc.runner import Runner


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
        edge_bearing_damping=1e-4,
        wheel_bearing_damping=1e-5,
    )
    defaults.update(overrides)
    return HardwareConfig(**defaults)


class ZeroController:
    def step(self, x_hat, t):
        return 0.0


def test_runner_log_shape():
    """1 simulated second at dt_sim=1ms produces 1000 sim steps. The log
    samples once per control tick (dt_control=10ms), so 100 entries.
    """
    env = CubliEnv(_make_hw(), SimConfig())
    env.reset(theta0=0.01)
    runner = Runner(env, ZeroController(), SimConfig())
    log = runner.run(duration_s=1.0)
    assert log["t"].shape == (100,)
    assert log["theta"].shape == (100,)
    assert log["tau"].shape == (100,)


def test_runner_zero_controller_lets_cube_fall():
    env = CubliEnv(_make_hw(), SimConfig())
    env.reset(theta0=0.05)
    runner = Runner(env, ZeroController(), SimConfig())
    log = runner.run(duration_s=0.5)
    # Without control, theta should grow past initial tilt.
    assert log["theta"][-1] > 0.05


def test_runner_stabilizes_small_tilt_with_nonlinear_pd():
    """M1 acceptance: nonlinear PD stabilizes from 5-degree initial tilt
    within 5 simulated seconds; final |theta| < 2 degrees, |theta_dot| < 0.5
    rad/s.
    """
    hw = _make_hw()
    sim = SimConfig()
    env = CubliEnv(hw, sim)
    env.reset(theta0=math.radians(5.0))

    gains = NonlinearPDGains(
        kp=0.5, kd=0.05, k_wheel=0.0001,
        max_balance_tilt=math.radians(2.0),
        max_torque=hw.motor_max_torque_nm,
    )
    controller = NonlinearPDController(gains, hw)

    runner = Runner(env, controller, sim)
    log = runner.run(duration_s=5.0)

    assert abs(log["theta"][-1]) < math.radians(2.0)
    assert abs(log["theta_dot"][-1]) < 0.5

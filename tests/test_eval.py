"""Tests for the eval harness: scenarios, metrics, runner."""
import math

import numpy as np
import pytest

from cubli_mpc.config import HardwareConfig, SimConfig
from cubli_mpc.control.random_torque import RandomController
from cubli_mpc.eval.runner import run_scenario
from cubli_mpc.eval.scenarios import SCENARIOS, get_scenario


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


def test_scenario_registry_includes_expected_names():
    names = set(SCENARIOS.keys())
    assert "recover-15deg" in names
    assert "recover-30deg" in names
    assert "hold-step-disturb" in names
    assert "hold-impulse-disturb" in names


def test_get_scenario_returns_struct():
    s = get_scenario("recover-15deg")
    assert s.name == "recover-15deg"
    assert s.theta0_rad == pytest.approx(math.radians(15.0))


def test_run_scenario_returns_metrics_dict():
    hw = _make_hw()
    sim = SimConfig(dt_sim=0.001, dt_control=0.01)
    ctrl = RandomController(max_torque=hw.motor_max_torque_nm, seed=0)
    result = run_scenario(
        controller=ctrl, scenario=get_scenario("recover-15deg"),
        hw=hw, sim=sim, duration_s=2.0,
    )
    for key in ("survived", "theta_rms_deg", "theta_max_abs_deg",
                "wheel_speed_max_abs_rad_s", "wheel_speed_rms_rad_s",
                "torque_rms_nm", "torque_integral_abs_nm_s"):
        assert key in result, f"missing metric: {key}"


def test_run_scenario_records_disturbance_log():
    """The result should include the tau_ext trace so plots can show it."""
    hw = _make_hw()
    sim = SimConfig(dt_sim=0.001, dt_control=0.01)
    ctrl = RandomController(max_torque=hw.motor_max_torque_nm, seed=0)
    result = run_scenario(
        controller=ctrl, scenario=get_scenario("hold-step-disturb"),
        hw=hw, sim=sim, duration_s=3.0,
    )
    assert "log" in result
    tau_ext = result["log"]["tau_ext"]
    assert float(np.max(np.abs(tau_ext))) > 0.0


def test_random_controller_falls_on_recover_30deg():
    """Sanity-check the metric: a random controller from 30 deg should
    fall (survived=False)."""
    hw = _make_hw()
    sim = SimConfig(dt_sim=0.001, dt_control=0.01)
    ctrl = RandomController(max_torque=hw.motor_max_torque_nm, seed=0)
    result = run_scenario(
        controller=ctrl, scenario=get_scenario("recover-30deg"),
        hw=hw, sim=sim, duration_s=3.0,
    )
    assert result["survived"] is False

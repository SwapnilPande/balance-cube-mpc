"""Tests for the offline swing-trajectory planner."""
import math
from pathlib import Path

import numpy as np
import pytest

from cubli_mpc.config import HardwareConfig
from cubli_mpc.control.swing_planner import SwingTrajConfig, plan_swing


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


def test_plan_swing_returns_npz_path(tmp_path: Path):
    cfg = SwingTrajConfig(
        period_s=1.0, theta_target_rad=math.radians(15.0),
        n_segments=40, save_path=tmp_path / "swing.npz",
    )
    out = plan_swing(_make_hw(), cfg)
    assert out == tmp_path / "swing.npz"
    assert out.exists()


def test_plan_swing_npz_contents(tmp_path: Path):
    cfg = SwingTrajConfig(
        period_s=1.0, theta_target_rad=math.radians(15.0),
        n_segments=40, save_path=tmp_path / "swing.npz",
    )
    out = plan_swing(_make_hw(), cfg)
    data = np.load(out)
    # Full period after mirroring: 2 * n_segments + 1 samples.
    assert data["t"].shape == (2 * 40 + 1,)
    assert data["x_ref"].shape == (2 * 40 + 1, 3)
    assert data["u_ref"].shape == (2 * 40, 1)


def test_plan_swing_boundary_conditions(tmp_path: Path):
    """Trajectory should start at +theta_target, hit -theta_target at the
    half-period, and return to +theta_target at the end of the period."""
    theta_target = math.radians(15.0)
    cfg = SwingTrajConfig(
        period_s=1.0, theta_target_rad=theta_target,
        n_segments=40, save_path=tmp_path / "swing.npz",
    )
    out = plan_swing(_make_hw(), cfg)
    data = np.load(out)
    x = data["x_ref"]
    assert x[0, 0] == pytest.approx(theta_target, abs=1e-3)
    assert x[40, 0] == pytest.approx(-theta_target, abs=1e-3)
    assert x[-1, 0] == pytest.approx(theta_target, abs=1e-3)


def test_plan_swing_torque_within_limits(tmp_path: Path):
    hw = _make_hw()
    cfg = SwingTrajConfig(
        period_s=1.5, theta_target_rad=math.radians(15.0),
        n_segments=60, save_path=tmp_path / "swing.npz",
    )
    out = plan_swing(hw, cfg)
    data = np.load(out)
    u = data["u_ref"]
    assert float(np.max(np.abs(u))) <= hw.motor_max_torque_nm + 1e-6

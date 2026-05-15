"""Tests for the Reference protocol implementations."""
import math

import numpy as np
import pytest

from cubli_mpc.control.references import ConstantReference


def test_constant_reference_shape():
    ref = ConstantReference()
    x_ref, u_ref = ref.at(t=0.0, dt=0.01, horizon=10)
    assert x_ref.shape == (11, 3)
    assert u_ref.shape == (10, 1)


def test_constant_reference_default_is_upright():
    ref = ConstantReference()
    x_ref, u_ref = ref.at(t=1.23, dt=0.01, horizon=5)
    assert np.allclose(x_ref, 0.0)
    assert np.allclose(u_ref, 0.0)


def test_constant_reference_with_target_theta():
    ref = ConstantReference(target_theta=math.radians(10.0))
    x_ref, u_ref = ref.at(t=0.0, dt=0.01, horizon=3)
    # All theta entries equal the target; theta_dot and omega_w stay zero.
    assert np.allclose(x_ref[:, 0], math.radians(10.0))
    assert np.allclose(x_ref[:, 1], 0.0)
    assert np.allclose(x_ref[:, 2], 0.0)
    assert np.allclose(u_ref, 0.0)


def test_constant_reference_independent_of_time():
    ref = ConstantReference(target_theta=0.05)
    a_x, a_u = ref.at(t=0.0, dt=0.01, horizon=4)
    b_x, b_u = ref.at(t=99.0, dt=0.01, horizon=4)
    assert np.allclose(a_x, b_x)
    assert np.allclose(a_u, b_u)


from pathlib import Path

from cubli_mpc.control.references import PeriodicTrajectoryReference


def _write_trivial_swing(path: Path, period: float, n: int) -> Path:
    """Write a synthetic swing trajectory: sine wave on theta, zeros
    elsewhere. Used to test the reference's interpolation + wrap."""
    t = np.linspace(0.0, period, n + 1)
    x_ref = np.zeros((n + 1, 3))
    x_ref[:, 0] = 0.1 * np.sin(2 * np.pi * t / period)
    u_ref = np.zeros((n, 1))
    np.savez(path, t=t, x_ref=x_ref, u_ref=u_ref)
    return path


def test_periodic_reference_loads_from_npz(tmp_path):
    path = _write_trivial_swing(tmp_path / "s.npz", period=1.0, n=100)
    ref = PeriodicTrajectoryReference.from_file(path)
    x, u = ref.at(t=0.0, dt=0.01, horizon=5)
    assert x.shape == (6, 3)
    assert u.shape == (5, 1)


def test_periodic_reference_loops_at_period(tmp_path):
    """t=0 and t=period should give the same x_ref."""
    path = _write_trivial_swing(tmp_path / "s.npz", period=1.0, n=100)
    ref = PeriodicTrajectoryReference.from_file(path)
    x_a, _ = ref.at(t=0.0, dt=0.01, horizon=3)
    x_b, _ = ref.at(t=1.0, dt=0.01, horizon=3)
    assert np.allclose(x_a, x_b, atol=1e-6)


def test_periodic_reference_interpolates_between_samples(tmp_path):
    """Mid-sample query should give a value between the two enclosing
    samples (linear interp)."""
    path = _write_trivial_swing(tmp_path / "s.npz", period=1.0, n=100)
    ref = PeriodicTrajectoryReference.from_file(path)
    x, _ = ref.at(t=0.25, dt=0.0, horizon=0)  # query single point
    # sin(2*pi*0.25) = 1.0, scaled by 0.1
    assert x[0, 0] == pytest.approx(0.1, abs=1e-3)

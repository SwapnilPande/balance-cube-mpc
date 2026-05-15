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

"""Tests for the random-torque benchmark controller."""
import numpy as np
import pytest

from cubli_mpc.control.random_torque import RandomController


def test_torques_within_bounds():
    c = RandomController(max_torque=0.2, seed=0)
    x = np.zeros(4)
    for i in range(10_000):
        tau = c.step(x, t=i * 0.01)
        assert -0.2 <= tau <= 0.2


def test_seed_is_reproducible():
    a = RandomController(max_torque=0.2, seed=42)
    b = RandomController(max_torque=0.2, seed=42)
    x = np.zeros(4)
    seq_a = [a.step(x, t=0.0) for _ in range(100)]
    seq_b = [b.step(x, t=0.0) for _ in range(100)]
    assert seq_a == seq_b


def test_different_seeds_produce_different_sequences():
    a = RandomController(max_torque=0.2, seed=0)
    b = RandomController(max_torque=0.2, seed=1)
    x = np.zeros(4)
    seq_a = [a.step(x, t=0.0) for _ in range(100)]
    seq_b = [b.step(x, t=0.0) for _ in range(100)]
    assert seq_a != seq_b


def test_ignores_state():
    """Random controller is open-loop: state input must not affect output."""
    a = RandomController(max_torque=0.2, seed=7)
    b = RandomController(max_torque=0.2, seed=7)
    seq_a = [a.step(np.zeros(4), t=0.0) for _ in range(50)]
    seq_b = [b.step(np.array([10.0, -3.0, 5.0, 200.0]), t=0.0)
             for _ in range(50)]
    assert seq_a == seq_b


def test_sample_mean_near_zero_and_covers_range():
    """Uniform white noise: large-sample mean close to zero, and samples
    span most of the symmetric range.
    """
    c = RandomController(max_torque=0.2, seed=123)
    x = np.zeros(4)
    samples = np.array([c.step(x, t=0.0) for _ in range(20_000)])
    assert abs(samples.mean()) < 0.01
    assert samples.min() < -0.18
    assert samples.max() > 0.18


def test_rejects_nonpositive_max_torque():
    with pytest.raises(ValueError):
        RandomController(max_torque=0.0)
    with pytest.raises(ValueError):
        RandomController(max_torque=-0.1)

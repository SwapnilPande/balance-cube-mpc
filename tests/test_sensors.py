"""Tests for the sensor noise + delay model."""
import numpy as np
import pytest

from cubli_mpc.sim.sensors import SensorConfig, SensorModel


def test_zero_config_is_identity():
    s = SensorModel(SensorConfig())
    x = np.array([0.1, -0.3, 5.0])
    out = s.observe(x)
    np.testing.assert_array_equal(out, x)


def test_noise_changes_observation():
    cfg = SensorConfig(noise_std=(0.01, 0.1, 1.0), seed=0)
    s = SensorModel(cfg)
    x = np.array([0.0, 0.0, 0.0])
    out = s.observe(x)
    assert not np.allclose(out, x)


def test_seed_reproducible():
    cfg = SensorConfig(noise_std=(0.01, 0.1, 1.0), seed=42)
    a = SensorModel(cfg)
    b = SensorModel(cfg)
    x = np.zeros(3)
    for _ in range(100):
        np.testing.assert_array_equal(a.observe(x), b.observe(x))


def test_noise_std_matches_target():
    cfg = SensorConfig(noise_std=(0.1, 0.2, 0.5), seed=0)
    s = SensorModel(cfg)
    x = np.zeros(3)
    samples = np.array([s.observe(x) for _ in range(20_000)])
    std = samples.std(axis=0)
    np.testing.assert_allclose(std, [0.1, 0.2, 0.5], rtol=0.05)


def test_delay_returns_old_readings():
    cfg = SensorConfig(delay_steps=3, seed=0)
    s = SensorModel(cfg)
    x0 = np.array([0.0, 0.0, 0.0])
    out0 = s.observe(x0)
    np.testing.assert_array_equal(out0, x0)
    for k in range(1, 10):
        new = np.array([k * 1.0, k * 2.0, k * 3.0])
        out = s.observe(new)
        expected = np.array([max(k - 3, 0) * 1.0,
                             max(k - 3, 0) * 2.0,
                             max(k - 3, 0) * 3.0])
        np.testing.assert_array_equal(out, expected)


def test_reset_clears_buffer():
    cfg = SensorConfig(delay_steps=2, seed=0)
    s = SensorModel(cfg)
    for k in range(5):
        s.observe(np.array([k * 1.0]))
    s.reset()
    out = s.observe(np.array([99.0]))
    np.testing.assert_array_equal(out, [99.0])


def test_rejects_bad_config():
    with pytest.raises(ValueError):
        SensorModel(SensorConfig(delay_steps=-1))
    with pytest.raises(ValueError):
        SensorModel(SensorConfig(noise_std=(-0.1,)))


def test_rejects_shape_mismatch():
    s = SensorModel(SensorConfig(noise_std=(0.1, 0.1)))
    with pytest.raises(ValueError):
        s.observe(np.zeros(3))

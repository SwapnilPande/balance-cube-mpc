"""Tests for the Gymnasium env."""
import math

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from cubli_mpc.config import HardwareConfig, SimConfig
from cubli_mpc.rl.cubli_env import (
    CubliBalanceEnv, DisturbanceConfig, EnvConfig, RewardConfig,
)
from cubli_mpc.sim.sensors import SensorConfig


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


def _make_env(**kwargs):
    hw = _make_hw()
    sim = SimConfig()
    return CubliBalanceEnv(hw, sim, **kwargs)


def test_env_obs_action_spaces():
    env = _make_env()
    assert env.observation_space.shape == (4,)
    assert env.action_space.shape == (1,)
    assert env.action_space.low[0] == -1.0
    assert env.action_space.high[0] == 1.0


def test_reset_returns_valid_obs():
    env = _make_env()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert "true_state" in info
    assert "target_theta" in info


def test_step_returns_5_tuple():
    env = _make_env()
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(np.array([0.0]))
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_episode_terminates_on_fall():
    env = _make_env(env_cfg=EnvConfig(
        episode_steps=1000,
        init_tilt_max_rad=math.radians(30),
        fall_tilt_rad=math.radians(45),
    ))
    env.reset(seed=0, options={"theta0": math.radians(60)})  # already fallen
    obs, reward, terminated, truncated, info = env.step(np.array([0.0]))
    assert terminated is True


def test_default_env_terminates_on_clearly_fallen_cube():
    """Regression: the default fall_tilt_rad must be reachable in MuJoCo.
    Earlier defaults of 60° were never triggered (cube face-contacts at
    ~45°) and produced wasted post-fall transitions in training data."""
    env = _make_env()  # default EnvConfig
    env.reset(seed=0, options={"theta0": math.radians(50)})
    obs, reward, terminated, truncated, info = env.step(np.array([0.0]))
    assert terminated is True


def test_episode_truncates_on_timeout():
    env = _make_env(env_cfg=EnvConfig(
        episode_steps=3,
        init_tilt_max_rad=0.0,
        fall_tilt_rad=math.radians(89),
    ))
    env.reset(seed=0, options={"theta0": 0.0})
    for i in range(3):
        obs, reward, terminated, truncated, info = env.step(np.array([0.0]))
    assert truncated is True


def test_action_is_rescaled_to_torque_limit():
    """An action of +1 should drive maximum torque, which spins the wheel
    quickly. Use a stiff actuator and short horizon to detect this."""
    env = _make_env()
    env.reset(seed=0, options={"theta0": 0.0})
    for _ in range(50):
        env.step(np.array([1.0]))
    omega_w = env._cubli.state()[3]  # noqa: SLF001 -- test inspection
    assert omega_w > 1.0  # positive wheel speed from positive torque


def test_target_theta_routes_through_obs():
    env = _make_env()
    obs, _ = env.reset(seed=0, options={"theta0": 0.0, "target_theta": 0.123})
    assert obs[3] == pytest.approx(0.123, abs=1e-4)


def test_target_theta_shifts_reward_zero():
    """Reward at theta == target equals the alive_bonus (zero cost). The
    only term left when other weights are zero is alive_bonus - w_theta*err^2,
    which collapses to alive_bonus when err == 0."""
    cfg = RewardConfig(
        alive_bonus=1.0,
        w_theta=10.0, w_theta_dot=0.0, w_omega=0.0, w_tau=0.0,
    )
    env = _make_env(reward_cfg=cfg)
    target = 0.1
    env.reset(seed=0, options={"theta0": target, "target_theta": target})
    _, reward_at_target, _, _, _ = env.step(np.array([0.0]))
    assert reward_at_target == pytest.approx(cfg.alive_bonus, abs=0.01)


def test_alive_bonus_dominates_at_balance():
    """Net reward at perfect balance should be strictly positive (alive
    bonus exceeds the near-zero cost) so the agent is rewarded for
    staying alive, not just for accumulating less negative cost."""
    env = _make_env()  # defaults
    env.reset(seed=0, options={"theta0": 0.0})
    _, reward, _, _, _ = env.step(np.array([0.0]))
    assert reward > 0.0


def test_fall_step_reward_is_large_negative():
    """On the terminating step, the agent receives (alive_bonus - cost) -
    fall_penalty. With defaults that's a strongly negative reward, which
    is what trains the agent away from the fall region."""
    env = _make_env()  # defaults
    env.reset(seed=0, options={"theta0": math.radians(50)})
    _, reward, terminated, _, _ = env.step(np.array([0.0]))
    assert terminated is True
    assert reward < -100.0


def test_disturbance_plan_seed_reproducible():
    cfg = DisturbanceConfig(enabled=True)
    a = _make_env(disturbance_cfg=cfg)
    b = _make_env(disturbance_cfg=cfg)
    a.reset(seed=42, options={"theta0": 0.0})
    b.reset(seed=42, options={"theta0": 0.0})
    # noqa: SLF001 -- test inspection
    assert a._disturbance_plan == b._disturbance_plan


def test_sensor_noise_routes_into_obs():
    """With large noise, repeated obs at the same true state should differ."""
    sensor_cfg = SensorConfig(noise_std=(0.1, 0.1, 1.0), seed=0)
    env = _make_env(sensor_cfg=sensor_cfg)
    env.reset(seed=0, options={"theta0": 0.0})
    obs_a = env._build_obs()  # noqa: SLF001
    obs_b = env._build_obs()  # noqa: SLF001
    assert not np.allclose(obs_a[:3], obs_b[:3])

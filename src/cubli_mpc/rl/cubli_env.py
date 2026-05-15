"""Gymnasium environment wrapping `CubliEnv` for RL training.

Observation: noisy `[theta, theta_dot, omega_wheel, target_theta]`. The
target-theta channel is zero for the balance task; setpoint-tracking
training reuses the same network shape with a non-zero reference.

Action: scalar in `[-1, +1]`, rescaled to `[-motor_max_torque, +max]`.

Reward: `-(w_theta*(theta - target)^2 + w_theta_dot*theta_dot^2 +
w_omega*omega^2 + w_tau*tau^2)` (negative so SAC maximizes upright time).

Per-episode disturbance: optional impulse or constant external torque
sampled at reset, applied via `CubliEnv.apply_disturbance_torque`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover - import-time guard
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "gymnasium is required for the RL env. Install with " "`uv sync --extra rl`."
    ) from exc

from cubli_mpc.config import HardwareConfig, SimConfig
from cubli_mpc.sim.env import CubliEnv
from cubli_mpc.sim.sensors import SensorConfig, SensorModel


@dataclass(frozen=True)
class RewardConfig:
    """Per-step reward = alive_bonus - (w_theta * err^2 + w_theta_dot *
    theta_dot^2 + w_omega * omega^2 + w_tau * tau^2). On terminate
    (fall), additionally subtract fall_penalty (configured on EnvConfig).

    The alive_bonus shifts the baseline upward so that being-alive is
    intrinsically positive. Without it, all per-step rewards are <= 0
    and the agent has a perverse incentive to terminate early once the
    accumulated future cost exceeds the fall_penalty — particularly
    when wheel speed has grown (w_omega * omega^2 dominates) and the
    cube is far from upright. Empirically, 1.0 keeps net reward
    comfortably positive in the well-balanced operating region while
    leaving headroom for the cost terms to shape behavior away from it.
    """

    alive_bonus: float = 1.0
    w_theta: float = 10.0
    w_theta_dot: float = 0.1
    w_omega: float = 0.005
    w_tau: float = 0.001


@dataclass(frozen=True)
class DisturbanceConfig:
    """Per-episode external torque applied to the tilt DOF."""

    enabled: bool = False
    mode_probs: tuple[float, float, float] = (0.4, 0.3, 0.3)  # none, step, impulse
    step_magnitude: float = 0.03  # Nm, constant push
    impulse_magnitude: float = 0.10  # Nm, applied for one control tick
    impulse_time_range_s: tuple[float, float] = (0.5, 3.0)


@dataclass(frozen=True)
class EnvConfig:
    episode_steps: int = 500  # at dt_control (default 10 ms = 5 s)
    init_tilt_max_rad: float = math.radians(15.0)
    # 42°: empirically the agent can't recover past ~40°, and the MuJoCo
    # cube face-contacts at ~45–48° anyway. Terminating at 42° kills the
    # episode before the cube is lost-but-still-stepping, which means more
    # useful gradient and less wasted replay-buffer data.
    fall_tilt_rad: float = math.radians(42.0)
    # Large enough to dominate worst-case accumulated cost over the rest
    # of the episode, so falling is never preferable to surviving — pairs
    # with RewardConfig.alive_bonus on the survival side.
    fall_penalty: float = 500.0


class CubliBalanceEnv(gym.Env):
    """Single-axis cubli balancing/tracking env."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        hw: HardwareConfig,
        sim: SimConfig,
        sensor_cfg: SensorConfig | None = None,
        reward_cfg: RewardConfig | None = None,
        disturbance_cfg: DisturbanceConfig | None = None,
        env_cfg: EnvConfig | None = None,
    ):
        super().__init__()
        self._hw = hw
        self._sim = sim
        self._sensor = SensorModel(sensor_cfg or SensorConfig())
        self._reward_cfg = reward_cfg or RewardConfig()
        self._disturbance_cfg = disturbance_cfg or DisturbanceConfig()
        self._env_cfg = env_cfg or EnvConfig()

        self._cubli = CubliEnv(hw, sim)
        self._steps_per_control = max(1, round(sim.dt_control / sim.dt_sim))
        self._max_torque = hw.motor_max_torque_nm
        self._rng = np.random.default_rng()

        # Obs bounds: theta wraps over pi; rates and wheel speeds are loose.
        omega_max = max(hw.motor_max_speed_rad_s * 1.5, 100.0)
        high = np.array([math.pi, 50.0, omega_max, math.pi], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        self._step_idx = 0
        self._target_theta = 0.0
        self._disturbance_plan: tuple[str, dict] = ("none", {})

    # ---- Gymnasium API -------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._sensor.reset()

        opts = options or {}
        theta0 = opts.get("theta0")
        if theta0 is None:
            lim = self._env_cfg.init_tilt_max_rad
            theta0 = float(self._rng.uniform(-lim, lim))
        self._target_theta = float(opts.get("target_theta", 0.0))

        self._cubli.reset(theta0=theta0)
        self._step_idx = 0
        self._disturbance_plan = self._sample_disturbance_plan()
        return self._build_obs(), self._info()

    def step(self, action):
        a = float(
            np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[0], -1.0, 1.0)
        )
        tau = a * self._max_torque
        self._cubli.apply_torque(tau)
        d = self._disturbance_at(self._step_idx)
        self._cubli.apply_disturbance_torque(d)
        for _ in range(self._steps_per_control):
            self._cubli.step()
        self._step_idx += 1

        x = self._cubli.state()
        theta, theta_dot, _, omega_w = x
        reward = self._compute_reward(theta, theta_dot, omega_w, tau)
        fell = bool(abs(theta) > self._env_cfg.fall_tilt_rad)
        if fell:
            reward -= self._env_cfg.fall_penalty
        timeout = bool(self._step_idx >= self._env_cfg.episode_steps)

        obs = self._build_obs()
        info = self._info(disturbance_torque=d, fell=fell)
        return obs, float(reward), fell, timeout, info

    # ---- Internals -----------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        x = self._cubli.state()
        # Drop the cyclic wheel-angle coordinate; keep theta, theta_dot,
        # omega_wheel. Sensor noise/delay applied here.
        x_obs = self._sensor.observe(np.array([x[0], x[1], x[3]]))
        out = np.array(
            [x_obs[0], x_obs[1], x_obs[2], self._target_theta], dtype=np.float32
        )
        return np.clip(out, self.observation_space.low, self.observation_space.high)

    def _compute_reward(self, theta, theta_dot, omega_w, tau) -> float:
        c = self._reward_cfg
        err = theta - self._target_theta
        cost = (
            c.w_theta * err * err
            + c.w_theta_dot * theta_dot * theta_dot
            + c.w_omega * omega_w * omega_w
            + c.w_tau * tau * tau
        )
        return c.alive_bonus - cost

    def _sample_disturbance_plan(self) -> tuple[str, dict]:
        cfg = self._disturbance_cfg
        if not cfg.enabled:
            return ("none", {})
        modes = ("none", "step", "impulse")
        mode = self._rng.choice(modes, p=cfg.mode_probs)
        if mode == "step":
            sign = 1.0 if self._rng.random() < 0.5 else -1.0
            return ("step", {"tau": sign * cfg.step_magnitude})
        if mode == "impulse":
            sign = 1.0 if self._rng.random() < 0.5 else -1.0
            t_apply = self._rng.uniform(*cfg.impulse_time_range_s)
            step_apply = int(t_apply / self._sim.dt_control)
            return (
                "impulse",
                {"step": step_apply, "tau": sign * cfg.impulse_magnitude},
            )
        return ("none", {})

    def _disturbance_at(self, step_idx: int) -> float:
        mode, args = self._disturbance_plan
        if mode == "step":
            return float(args["tau"])
        if mode == "impulse":
            return float(args["tau"]) if step_idx == args["step"] else 0.0
        return 0.0

    def _info(self, **extra) -> dict:
        info = {
            "true_state": self._cubli.state().copy(),
            "target_theta": self._target_theta,
            "step_idx": self._step_idx,
        }
        info.update(extra)
        return info

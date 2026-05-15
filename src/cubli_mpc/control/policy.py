"""Adapter that loads a saved SB3 policy and exposes it as a `Controller`.

The training env's observation is `[theta, theta_dot, omega_wheel,
target_theta]`. At deployment we feed the policy this same vector built
from the runner's 4D state, so trained policies plug into the existing
`Runner` / `cubli-mpc sim --controller policy` workflow.

Note: the policy was trained on (possibly) noisy observations. We feed
ground-truth state here -- in M4 we'll add a sensor-noise wrapper if we
want to reproduce the training distribution at evaluation time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _load_sb3_model(model_path: Path, algo: str):
    algo = algo.lower()
    if algo == "sac":
        from stable_baselines3 import SAC
        return SAC.load(str(model_path), device="cpu")
    if algo == "ppo":
        from stable_baselines3 import PPO
        return PPO.load(str(model_path), device="cpu")
    raise ValueError(f"unknown algo {algo!r}; choose from sac, ppo")


def _detect_algo(model_path: Path) -> str:
    """Look for `algo.txt` next to the model (written by `train()`).

    The file may live alongside the model (`runs/<name>/algo.txt` for
    `runs/<name>/final.zip`) or one level up (for files in `ckpts/` or
    `best/`). Falls back to SAC if not found.
    """
    for candidate in (model_path.parent / "algo.txt",
                      model_path.parent.parent / "algo.txt"):
        if candidate.exists():
            return candidate.read_text().strip().lower()
    return "sac"


class PolicyController:
    def __init__(
        self,
        model_path: str | Path,
        max_torque: float,
        target_theta: float = 0.0,
        deterministic: bool = True,
        algo: str | None = None,
    ):
        path = Path(model_path)
        resolved_algo = algo or _detect_algo(path)
        self._algo = resolved_algo
        self._model = _load_sb3_model(path, resolved_algo)
        self._max_torque = float(max_torque)
        self._target_theta = float(target_theta)
        self._deterministic = bool(deterministic)

    @property
    def algo(self) -> str:
        return self._algo

    @property
    def target_theta(self) -> float:
        return self._target_theta

    @target_theta.setter
    def target_theta(self, value: float) -> None:
        self._target_theta = float(value)

    def step(self, x_hat: np.ndarray, t: float) -> float:
        obs = np.array(
            [x_hat[0], x_hat[1], x_hat[3], self._target_theta],
            dtype=np.float32,
        )
        action, _ = self._model.predict(obs, deterministic=self._deterministic)
        return float(action[0]) * self._max_torque

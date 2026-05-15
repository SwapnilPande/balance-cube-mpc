"""SAC training entry point for `cubli-mpc train`.

Builds the gymnasium env, configures SB3 SAC, wires TensorBoard + (optional)
Weights & Biases, periodic checkpoints, and supports resume-from-checkpoint.

Designed for background runs:
    nohup uv run cubli-mpc train --config configs/default.yaml \\
        --out runs/exp1 --total-steps 1_000_000 \\
        --sensor-noise --disturbances --wandb \\
        > runs/exp1/train.log 2>&1 &
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cubli_mpc.config import HardwareConfig, SimConfig, load_config
from cubli_mpc.rl.cubli_env import (
    CubliBalanceEnv, DisturbanceConfig, EnvConfig, RewardConfig,
)
from cubli_mpc.sim.sensors import SensorConfig

# Reasonable defaults for the M1 hardware. Tunable on the command line.
DEFAULT_SENSOR_NOISE = SensorConfig(
    noise_std=(0.005, 0.05, 0.5),  # rad, rad/s, rad/s on [theta, theta_dot, omega_w]
    delay_steps=0,
    seed=0,
)


@dataclass
class TrainOptions:
    config_path: Path
    out_dir: Path
    total_steps: int = 10_000_000
    seed: int = None
    sensor_noise: bool = False
    disturbances: bool = False
    resume_from: Path | None = None
    checkpoint_every: int = 25_000
    eval_every: int = 25_000
    n_eval_episodes: int = 5
    n_envs: int = 1
    algo: str = "sac"  # "sac" | "ppo"
    # --- SAC-specific ---
    learning_starts: int = 1_000
    # -1 = match collected env steps per rollout (scales 1:1 with n_envs).
    # Keeps the gradient-step/env-step ratio constant as you parallelize.
    gradient_steps: int = -1
    # --- PPO-specific ---
    ppo_n_steps: int = 2048      # rollout length per env
    ppo_batch_size: int = 64
    ppo_n_epochs: int = 10
    use_wandb: bool = False
    wandb_project: str = "cubli-mpc"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str = "online"  # "online" | "offline" | "disabled"


def _build_env_factory(hw: HardwareConfig, sim: SimConfig,
                       opts: TrainOptions, eval_mode: bool = False):
    sensor_cfg = DEFAULT_SENSOR_NOISE if opts.sensor_noise else SensorConfig()
    disturbance_cfg = DisturbanceConfig(enabled=opts.disturbances
                                        and not eval_mode)
    reward_cfg = RewardConfig()
    env_cfg = EnvConfig()

    def make():
        return CubliBalanceEnv(
            hw=hw, sim=sim,
            sensor_cfg=sensor_cfg,
            reward_cfg=reward_cfg,
            disturbance_cfg=disturbance_cfg,
            env_cfg=env_cfg,
        )
    return make


def _init_wandb(opts: TrainOptions, hw: HardwareConfig, sim: SimConfig):
    import wandb

    os.environ.setdefault("WANDB_MODE", opts.wandb_mode)
    return wandb.init(
        project=opts.wandb_project,
        entity=opts.wandb_entity,
        name=opts.wandb_run_name or opts.out_dir.name,
        dir=str(opts.out_dir),
        config={
            "hardware": hw.__dict__,
            "sim": sim.__dict__,
            "sensor_noise": opts.sensor_noise,
            "disturbances": opts.disturbances,
            "total_steps": opts.total_steps,
            "seed": opts.seed,
            "algo": opts.algo.upper(),
        },
        sync_tensorboard=True,
        save_code=False,
    )


_ALGO_REGISTRY = {"sac", "ppo"}


def _load_algo_class(algo: str):
    """Dispatch the SB3 algorithm class. Imported lazily so SB3 isn't a
    hard dependency at module-import time."""
    algo = algo.lower()
    if algo == "sac":
        from stable_baselines3 import SAC
        return SAC
    if algo == "ppo":
        from stable_baselines3 import PPO
        return PPO
    raise ValueError(f"unknown algo {algo!r}; choose from {_ALGO_REGISTRY}")


def train(opts: TrainOptions) -> Path:
    """Run a training job. Returns the path to the final saved model."""
    from stable_baselines3.common.callbacks import (
        CheckpointCallback, EvalCallback,
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    Algo = _load_algo_class(opts.algo)

    hw, sim = load_config(opts.config_path)
    opts.out_dir.mkdir(parents=True, exist_ok=True)

    # Sample seed if None
    if opts.seed is None:
        opts.seed = int.from_bytes(os.urandom(4), "big") % (2**32 - 1)
    print(f"Using seed: {opts.seed}")

    train_factory = _build_env_factory(hw, sim, opts, eval_mode=False)
    eval_factory = _build_env_factory(hw, sim, opts, eval_mode=True)

    def _make_train_worker(rank: int):
        # SB3 Monitor auto-appends ".monitor.csv" unless the path already
        # ends with that exact suffix. Pass the stem without extension to
        # avoid filenames like `monitor_0.csv.monitor.csv`.
        if opts.n_envs == 1:
            log_path = str(opts.out_dir / "monitor.csv")
        else:
            log_path = str(opts.out_dir / f"monitor_{rank}")
        def _init():
            return Monitor(train_factory(), filename=log_path)
        return _init

    workers = [_make_train_worker(i) for i in range(opts.n_envs)]
    if opts.n_envs > 1:
        # `forkserver` avoids the fork-after-mujoco-import warnings while
        # still being faster than `spawn` to bootstrap.
        train_env = SubprocVecEnv(workers, start_method="forkserver")
    else:
        train_env = DummyVecEnv(workers)
    eval_env = DummyVecEnv([lambda: Monitor(eval_factory())])

    tb_log = opts.out_dir / "tb"
    ckpt_dir = opts.out_dir / "ckpts"
    ckpt_dir.mkdir(exist_ok=True)
    best_dir = opts.out_dir / "best"

    wandb_run = None
    callbacks: list[Any] = [
        CheckpointCallback(
            save_freq=opts.checkpoint_every,
            save_path=str(ckpt_dir),
            name_prefix=f"cubli_{opts.algo.lower()}",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(best_dir),
            log_path=str(opts.out_dir / "eval"),
            eval_freq=opts.eval_every,
            n_eval_episodes=opts.n_eval_episodes,
            deterministic=True,
            render=False,
        ),
    ]

    if opts.use_wandb:
        from wandb.integration.sb3 import WandbCallback
        wandb_run = _init_wandb(opts, hw, sim)
        callbacks.append(WandbCallback(
            gradient_save_freq=0,
            model_save_path=str(opts.out_dir / "wandb_models"),
            model_save_freq=opts.checkpoint_every,
            verbose=1,
        ))

    if opts.resume_from is not None:
        print(f"resuming from {opts.resume_from}")
        model = Algo.load(str(opts.resume_from), env=train_env,
                          tensorboard_log=str(tb_log))
        reset_timesteps = False
    else:
        algo_kwargs: dict[str, Any] = {
            "verbose": 1,
            "seed": opts.seed,
            "tensorboard_log": str(tb_log),
        }
        if opts.algo.lower() == "sac":
            algo_kwargs["learning_starts"] = opts.learning_starts
            algo_kwargs["gradient_steps"] = opts.gradient_steps
        elif opts.algo.lower() == "ppo":
            algo_kwargs["n_steps"] = opts.ppo_n_steps
            algo_kwargs["batch_size"] = opts.ppo_batch_size
            algo_kwargs["n_epochs"] = opts.ppo_n_epochs
        model = Algo("MlpPolicy", train_env, **algo_kwargs)
        reset_timesteps = True

    # Save the algo alongside the checkpoints so PolicyController can
    # auto-detect it at load time.
    (opts.out_dir / "algo.txt").write_text(opts.algo.lower() + "\n")

    try:
        model.learn(
            total_timesteps=opts.total_steps,
            callback=callbacks,
            reset_num_timesteps=reset_timesteps,
            progress_bar=False,  # off for nohup-friendly logs
        )
    finally:
        final_path = opts.out_dir / "final.zip"
        model.save(str(final_path))
        if wandb_run is not None:
            wandb_run.finish()

    print(f"saved final model to {final_path}")
    return final_path

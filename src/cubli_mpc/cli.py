"""Command-line entry point for cubli-mpc.

Usage:
    cubli-mpc sim --config configs/default.yaml [--duration 5.0] [--viewer]
"""
from __future__ import annotations

import os
# If running headless and rendering, default to EGL backend. This is set
# before any mujoco import so it takes effect for the renderer.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import math
import sys
from pathlib import Path

import numpy as np

from cubli_mpc.config import load_config
from cubli_mpc.control.nonlinear_pd import (
    NonlinearPDController, NonlinearPDGains,
)
from cubli_mpc.control.random_torque import RandomController
from cubli_mpc.runner import Runner
from cubli_mpc.sim.disturbance import DisturbancePlan, StepDisturbance
from cubli_mpc.sim.env import CubliEnv


def _build_controller(args: argparse.Namespace, hw):
    if args.controller == "random":
        return RandomController(
            max_torque=hw.motor_max_torque_nm, seed=args.seed,
        )
    if args.controller == "policy":
        if args.policy_path is None:
            raise SystemExit(
                "--policy-path is required when --controller policy"
            )
        from cubli_mpc.control.policy import PolicyController
        return PolicyController(
            model_path=args.policy_path,
            max_torque=hw.motor_max_torque_nm,
            target_theta=math.radians(args.target_tilt_deg),
            algo=args.policy_algo,
        )
    if args.controller == "nmpc":
        from cubli_mpc.control.nmpc import NMPCConfig, NMPCController
        from cubli_mpc.control.references import ConstantReference
        cfg = NMPCConfig(horizon_steps=args.nmpc_horizon, dt=args.nmpc_dt)
        ref = ConstantReference(
            target_theta=math.radians(args.nmpc_target_tilt_deg),
        )
        return NMPCController(hw, cfg, reference=ref)
    gains = NonlinearPDGains(
        kp=args.kp, kd=args.kd, k_wheel=args.k_wheel,
        max_balance_tilt=math.radians(args.max_balance_tilt_deg),
        max_torque=hw.motor_max_torque_nm,
    )
    return NonlinearPDController(gains, hw)


def _build_disturbance(args: argparse.Namespace) -> DisturbancePlan | None:
    """Build a DisturbancePlan from CLI flags, or return None if no
    disturbance was requested."""
    steps: list[StepDisturbance] = []
    for trio in (args.disturbance_step or []):
        mag, start, dur = trio
        steps.append(StepDisturbance(
            magnitude=float(mag), start_t=float(start), duration=float(dur),
        ))
    impulse_mag, impulse_rate = 0.0, 0.0
    if args.disturbance_impulses is not None:
        impulse_mag, impulse_rate = (float(args.disturbance_impulses[0]),
                                     float(args.disturbance_impulses[1]))
    if not steps and impulse_mag == 0.0:
        return None
    return DisturbancePlan(
        steps=steps,
        impulse_magnitude=impulse_mag,
        impulse_rate_per_s=impulse_rate,
        seed=args.disturbance_seed,
    )


def _cmd_sim(args: argparse.Namespace) -> int:
    hw, sim = load_config(args.config)
    env = CubliEnv(hw, sim)
    env.reset(theta0=math.radians(args.initial_tilt_deg))

    controller = _build_controller(args, hw)
    disturbance = _build_disturbance(args)

    if args.viewer and args.video:
        print("error: --viewer and --video are mutually exclusive",
              file=sys.stderr)
        return 2
    if args.viewer:
        return _run_with_viewer(env, controller, sim, args.duration,
                                disturbance=disturbance,
                                plot_path=args.plot)
    if args.video:
        return _run_with_recorder(env, controller, sim, args.duration,
                                  args.video, args.video_fps,
                                  args.video_width, args.video_height,
                                  disturbance=disturbance,
                                  plot_path=args.plot)
    return _run_headless(env, controller, sim, args.duration,
                         disturbance=disturbance,
                         plot_path=args.plot)


def _alloc_log(n: int) -> dict[str, np.ndarray]:
    return {
        "t": np.empty(n),
        "theta": np.empty(n),
        "theta_dot": np.empty(n),
        "wheel_angle": np.empty(n),
        "wheel_speed": np.empty(n),
        "tau": np.empty(n),
        "tau_ext": np.zeros(n),
    }


def _record_step(log: dict[str, np.ndarray], i: int, t: float,
                 x: np.ndarray, tau: float, tau_ext: float = 0.0) -> None:
    log["t"][i] = t
    log["theta"][i] = x[0]
    log["theta_dot"][i] = x[1]
    log["wheel_angle"][i] = x[2]
    log["wheel_speed"][i] = x[3]
    log["tau"][i] = tau
    log["tau_ext"][i] = tau_ext


def _print_summary(log: dict[str, np.ndarray]) -> None:
    print(f"t={log['t'][-1]:.3f}s  "
          f"theta={math.degrees(log['theta'][-1]):+.3f} deg  "
          f"theta_dot={log['theta_dot'][-1]:+.3f} rad/s  "
          f"omega_wheel={log['wheel_speed'][-1]:+.2f} rad/s")
    print(f"|theta|_max = {math.degrees(np.max(np.abs(log['theta']))):.3f} deg")
    print(f"|tau|_max   = {np.max(np.abs(log['tau'])):.4f} Nm")


def _save_state_plot(log: dict[str, np.ndarray], path: Path,
                     target_theta_rad: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = log["t"]
    tau_ext = log.get("tau_ext")
    has_disturbance = tau_ext is not None and float(np.max(np.abs(tau_ext))) > 0
    n_panels = 5 if has_disturbance else 4
    fig, axes = plt.subplots(n_panels, 1, sharex=True,
                             figsize=(8, 2.25 * n_panels))
    axes[0].plot(t, np.degrees(log["theta"]), color="C0")
    if target_theta_rad is not None:
        axes[0].axhline(math.degrees(target_theta_rad),
                        color="C3", lw=1.0, ls="--", label="target")
        axes[0].legend(loc="upper right", fontsize=8)
    axes[0].axhline(0, color="k", lw=0.5, alpha=0.3)
    axes[0].set_ylabel("θ (deg)")
    axes[1].plot(t, log["theta_dot"], color="C1")
    axes[1].set_ylabel("θ̇ (rad/s)")
    axes[2].plot(t, log["wheel_speed"], color="C2")
    axes[2].set_ylabel("ω_wheel (rad/s)")
    axes[3].plot(t, log["tau"], color="C4")
    axes[3].set_ylabel("τ (Nm)")
    if has_disturbance:
        axes[4].plot(t, tau_ext, color="C5")
        axes[4].axhline(0, color="k", lw=0.5, alpha=0.3)
        axes[4].set_ylabel("τ_ext (Nm)")
    axes[-1].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"wrote {path}")


def _controller_target_theta(controller) -> float | None:
    return getattr(controller, "target_theta", None)


def _run_headless(env, controller, sim, duration: float,
                  disturbance: DisturbancePlan | None = None,
                  plot_path: Path | None = None) -> int:
    runner = Runner(env, controller, sim, disturbance=disturbance)
    log = runner.run(duration_s=duration)
    _print_summary(log)
    if plot_path is not None:
        _save_state_plot(log, plot_path,
                         target_theta_rad=_controller_target_theta(controller))
    return 0


def _run_with_viewer(env, controller, sim, duration: float,
                     disturbance: DisturbancePlan | None = None,
                     plot_path: Path | None = None) -> int:
    import mujoco.viewer

    steps_per_control = max(1, round(sim.dt_control / sim.dt_sim))
    n_ticks = int(round(duration / sim.dt_control))
    log = _alloc_log(n_ticks) if plot_path is not None else None
    actual_ticks = 0
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for i in range(n_ticks):
            if not viewer.is_running():
                break
            x = env.state()
            t = env.time
            tau = float(controller.step(x, t))
            env.apply_torque(tau)
            tau_ext = 0.0
            if disturbance is not None:
                tau_ext = disturbance.at(t, sim.dt_control)
                env.apply_disturbance_torque(tau_ext)
            if log is not None:
                _record_step(log, i, t, x, tau, tau_ext)
                actual_ticks = i + 1
            for _ in range(steps_per_control):
                env.step()
            viewer.sync()
    if log is not None and actual_ticks > 0:
        sliced = {k: v[:actual_ticks] for k, v in log.items()}
        _save_state_plot(sliced, plot_path,
                         target_theta_rad=_controller_target_theta(controller))
    return 0


def _run_with_recorder(env, controller, sim, duration: float,
                       video_path: Path, fps: int,
                       width: int, height: int,
                       disturbance: DisturbancePlan | None = None,
                       plot_path: Path | None = None) -> int:
    from cubli_mpc.sim.recorder import VideoRecorder

    steps_per_control = max(1, round(sim.dt_control / sim.dt_sim))
    n_ticks = int(round(duration / sim.dt_control))
    capture_every = max(1, round((1.0 / fps) / sim.dt_control))
    log = _alloc_log(n_ticks) if plot_path is not None else None

    with VideoRecorder(env.model, video_path, fps=fps,
                       width=width, height=height) as rec:
        for i in range(n_ticks):
            x = env.state()
            t = env.time
            tau = float(controller.step(x, t))
            env.apply_torque(tau)
            tau_ext = 0.0
            if disturbance is not None:
                tau_ext = disturbance.at(t, sim.dt_control)
                env.apply_disturbance_torque(tau_ext)
            if log is not None:
                _record_step(log, i, t, x, tau, tau_ext)
            for _ in range(steps_per_control):
                env.step()
            if i % capture_every == 0:
                rec.capture(env.data)
    print(f"wrote {video_path} ({n_ticks // capture_every} frames)")
    if log is not None:
        _save_state_plot(log, plot_path,
                         target_theta_rad=_controller_target_theta(controller))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cubli-mpc")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sim = sub.add_parser("sim", help="Run a simulation")
    p_sim.add_argument("--config", required=True, type=Path)
    p_sim.add_argument("--duration", type=float, default=5.0)
    p_sim.add_argument("--initial-tilt-deg", type=float, default=5.0)
    p_sim.add_argument("--controller",
                       choices=("pd", "random", "policy", "nmpc"), default="pd",
                       help="Which controller to run (default: pd)")
    p_sim.add_argument("--seed", type=int, default=0,
                       help="RNG seed for the random controller")
    p_sim.add_argument("--policy-path", type=Path, default=None,
                       help="Path to a saved SB3 model (for --controller "
                            "policy)")
    p_sim.add_argument("--policy-algo", choices=("sac", "ppo"), default=None,
                       help="Algorithm of the saved policy; auto-detected "
                            "from algo.txt next to the model if omitted")
    p_sim.add_argument("--target-tilt-deg", type=float, default=0.0,
                       help="Tracking target angle for policy controller "
                            "(degrees; 0 = balance)")
    p_sim.add_argument("--kp", type=float, default=0.5)
    p_sim.add_argument("--kd", type=float, default=0.05)
    p_sim.add_argument("--k-wheel", type=float, default=1e-4)
    p_sim.add_argument("--max-balance-tilt-deg", type=float, default=2.0)
    p_sim.add_argument("--viewer", action="store_true",
                       help="Open the MuJoCo viewer")
    p_sim.add_argument("--video", type=Path, default=None,
                       help="Write an mp4 of the simulation to this path "
                            "(runs headless, no viewer)")
    p_sim.add_argument("--video-fps", type=int, default=30)
    p_sim.add_argument("--video-width", type=int, default=640)
    p_sim.add_argument("--video-height", type=int, default=480)
    p_sim.add_argument("--plot", type=Path, default=None,
                       help="Save a state-trace plot (θ, θ̇, ω_wheel, τ, "
                            "plus τ_ext if disturbances are active) to "
                            "this path")
    p_sim.add_argument("--disturbance-step", nargs=3, action="append",
                       metavar=("MAG", "START", "DUR"), default=None,
                       help="Inject a constant MAG (Nm) external torque on "
                            "the tilt DOF starting at START (s) for DUR (s). "
                            "Repeatable for multiple events.")
    p_sim.add_argument("--disturbance-impulses", nargs=2, default=None,
                       metavar=("MAG", "RATE"),
                       help="Inject ±MAG (Nm) one-tick impulses at Poisson "
                            "rate RATE per second on the tilt DOF.")
    p_sim.add_argument("--disturbance-seed", type=int, default=0,
                       help="RNG seed for the impulse disturbance stream")
    p_sim.add_argument("--nmpc-horizon", type=int, default=50,
                       help="(NMPC) prediction horizon in steps")
    p_sim.add_argument("--nmpc-dt", type=float, default=0.010,
                       help="(NMPC) prediction step (s); matches dt_control "
                            "by default")
    p_sim.add_argument("--nmpc-target-tilt-deg", type=float, default=0.0,
                       help="(NMPC) balance setpoint (degrees; 0 = upright). "
                            "Ignored if --swing-traj is given.")
    p_sim.set_defaults(func=_cmd_sim)

    p_train = sub.add_parser("train", help="Train an RL policy")
    p_train.add_argument("--config", required=True, type=Path)
    p_train.add_argument("--out", required=True, type=Path,
                         help="Output directory for checkpoints, TB logs, "
                              "and the final model")
    p_train.add_argument("--total-steps", type=int, default=1_000_000)
    p_train.add_argument("--seed", type=int, default=None,
                         help="RNG seed for training (default: random)")
    p_train.add_argument("--sensor-noise", action="store_true",
                         help="Enable sensor noise on the obs (recommended "
                              "for robustness)")
    p_train.add_argument("--disturbances", action="store_true",
                         help="Enable per-episode external-torque "
                              "disturbances during training")
    p_train.add_argument("--resume-from", type=Path, default=None,
                         help="Resume from this saved model")
    p_train.add_argument("--checkpoint-every", type=int, default=25_000)
    p_train.add_argument("--eval-every", type=int, default=25_000)
    p_train.add_argument("--n-eval-episodes", type=int, default=5)
    p_train.add_argument("--algo", choices=("sac", "ppo"), default="sac",
                         help="RL algorithm (default: sac)")
    p_train.add_argument("--n-envs", type=int, default=1,
                         help="Number of parallel training envs "
                              "(SubprocVecEnv when >1)")
    # --- SAC-specific ---
    p_train.add_argument("--learning-starts", type=int, default=1_000,
                         help="(SAC) random-action warmup before training")
    p_train.add_argument("--gradient-steps", type=int, default=-1,
                         help="(SAC) gradient updates per rollout. -1 = "
                              "match collected env steps; scales 1:1 with "
                              "--n-envs.")
    # --- PPO-specific ---
    p_train.add_argument("--ppo-n-steps", type=int, default=2048,
                         help="(PPO) rollout length per env per update")
    p_train.add_argument("--ppo-batch-size", type=int, default=64,
                         help="(PPO) minibatch size for the n-epochs updates")
    p_train.add_argument("--ppo-n-epochs", type=int, default=10,
                         help="(PPO) number of SGD passes per rollout")
    p_train.add_argument("--wandb", action="store_true",
                         help="Log to Weights & Biases")
    p_train.add_argument("--wandb-project", default="cubli-mpc")
    p_train.add_argument("--wandb-entity", default=None)
    p_train.add_argument("--wandb-run-name", default=None)
    p_train.add_argument("--wandb-mode",
                         choices=("online", "offline", "disabled"),
                         default="online")
    p_train.set_defaults(func=_cmd_train)

    args = parser.parse_args(argv)
    return args.func(args)


def _cmd_train(args: argparse.Namespace) -> int:
    from cubli_mpc.rl.train import TrainOptions, train

    opts = TrainOptions(
        config_path=args.config,
        out_dir=args.out,
        total_steps=args.total_steps,
        seed=args.seed,
        sensor_noise=args.sensor_noise,
        disturbances=args.disturbances,
        resume_from=args.resume_from,
        checkpoint_every=args.checkpoint_every,
        eval_every=args.eval_every,
        n_eval_episodes=args.n_eval_episodes,
        n_envs=args.n_envs,
        algo=args.algo,
        learning_starts=args.learning_starts,
        gradient_steps=args.gradient_steps,
        ppo_n_steps=args.ppo_n_steps,
        ppo_batch_size=args.ppo_batch_size,
        ppo_n_epochs=args.ppo_n_epochs,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
    )
    train(opts)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

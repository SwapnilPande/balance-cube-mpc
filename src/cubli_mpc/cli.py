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
from cubli_mpc.runner import Runner
from cubli_mpc.sim.env import CubliEnv


def _cmd_sim(args: argparse.Namespace) -> int:
    hw, sim = load_config(args.config)
    env = CubliEnv(hw, sim)
    env.reset(theta0=math.radians(args.initial_tilt_deg))

    gains = NonlinearPDGains(
        kp=args.kp, kd=args.kd, k_wheel=args.k_wheel,
        max_balance_tilt=math.radians(args.max_balance_tilt_deg),
        max_torque=hw.motor_max_torque_nm,
    )
    controller = NonlinearPDController(gains, hw)

    if args.viewer and args.video:
        print("error: --viewer and --video are mutually exclusive",
              file=sys.stderr)
        return 2
    if args.viewer:
        return _run_with_viewer(env, controller, sim, args.duration)
    if args.video:
        return _run_with_recorder(env, controller, sim, args.duration,
                                  args.video, args.video_fps,
                                  args.video_width, args.video_height)
    return _run_headless(env, controller, sim, args.duration)


def _run_headless(env, controller, sim, duration: float) -> int:
    runner = Runner(env, controller, sim)
    log = runner.run(duration_s=duration)
    final_theta = log["theta"][-1]
    final_theta_dot = log["theta_dot"][-1]
    final_omega_w = log["wheel_speed"][-1]
    print(f"t={log['t'][-1]:.3f}s  "
          f"theta={math.degrees(final_theta):+.3f} deg  "
          f"theta_dot={final_theta_dot:+.3f} rad/s  "
          f"omega_wheel={final_omega_w:+.2f} rad/s")
    print(f"|theta|_max = {math.degrees(np.max(np.abs(log['theta']))):.3f} deg")
    print(f"|tau|_max   = {np.max(np.abs(log['tau'])):.4f} Nm")
    return 0


def _run_with_viewer(env, controller, sim, duration: float) -> int:
    import mujoco.viewer

    steps_per_control = max(1, round(sim.dt_control / sim.dt_sim))
    n_ticks = int(round(duration / sim.dt_control))
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for _ in range(n_ticks):
            if not viewer.is_running():
                break
            x = env.state()
            tau = float(controller.step(x, env.time))
            env.apply_torque(tau)
            for _ in range(steps_per_control):
                env.step()
            viewer.sync()
    return 0


def _run_with_recorder(env, controller, sim, duration: float,
                       video_path: Path, fps: int,
                       width: int, height: int) -> int:
    from cubli_mpc.sim.recorder import VideoRecorder

    steps_per_control = max(1, round(sim.dt_control / sim.dt_sim))
    n_ticks = int(round(duration / sim.dt_control))
    capture_every = max(1, round((1.0 / fps) / sim.dt_control))

    with VideoRecorder(env.model, video_path, fps=fps,
                       width=width, height=height) as rec:
        for i in range(n_ticks):
            x = env.state()
            tau = float(controller.step(x, env.time))
            env.apply_torque(tau)
            for _ in range(steps_per_control):
                env.step()
            if i % capture_every == 0:
                rec.capture(env.data)
    print(f"wrote {video_path} ({n_ticks // capture_every} frames)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cubli-mpc")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sim = sub.add_parser("sim", help="Run a simulation")
    p_sim.add_argument("--config", required=True, type=Path)
    p_sim.add_argument("--duration", type=float, default=5.0)
    p_sim.add_argument("--initial-tilt-deg", type=float, default=5.0)
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
    p_sim.set_defaults(func=_cmd_sim)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

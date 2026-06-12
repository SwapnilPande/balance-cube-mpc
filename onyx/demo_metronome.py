"""Render a demo of the metronome controller: state-trace plot + mp4 video.

Visual confirmation that the cube ticks back and forth like a clock at the
committed gains. Outputs to runs/metronome/.
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import math
from pathlib import Path

import numpy as np

from cubli_mpc.config import load_config
from cubli_mpc.control.metronome import MetronomeController
from cubli_mpc.sim.env import CubliEnv
from cubli_mpc.sim.recorder import VideoRecorder
from cubli_mpc.cli import _save_state_plot, _alloc_log, _record_step

DURATION = 6.0
FPS = 30
OUT = Path("runs/metronome")


def main() -> int:
    hw, sim = load_config("configs/default.yaml")
    env = CubliEnv(hw, sim)
    ctrl = MetronomeController(hw)
    env.reset(theta0=ctrl.amplitude, theta_dot0=0.0)

    steps_per_control = max(1, round(sim.dt_control / sim.dt_sim))
    n_ticks = int(round(DURATION / sim.dt_control))
    capture_every = max(1, round((1.0 / FPS) / sim.dt_control))
    log = _alloc_log(n_ticks)

    OUT.mkdir(parents=True, exist_ok=True)
    video_path = OUT / "metronome.mp4"
    with VideoRecorder(env.model, video_path, fps=FPS, width=640, height=480) as rec:
        for i in range(n_ticks):
            x = env.state()
            t = env.time
            tau = float(ctrl.step(x, t))
            env.apply_torque(tau)
            _record_step(log, i, t, x, tau)
            for _ in range(steps_per_control):
                env.step()
            if i % capture_every == 0:
                rec.capture(env.data)
    print(f"wrote {video_path}")

    _save_state_plot(log, OUT / "metronome.png")
    print(f"period target=1.0s, swing amplitude={math.degrees(ctrl.amplitude):.1f}deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

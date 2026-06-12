"""Render a video of the NMPC (track-min-effort-plan) strategy explicitly.

Bypasses the STRATEGY switch entirely (builds the NMPC controller directly), so
there is no chance of rendering the wrong strategy. Outputs metronome_nmpc.mp4.
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")
import time
from pathlib import Path

from cubli_mpc.config import load_config
from cubli_mpc.control.metronome import build_nmpc_metronome
from cubli_mpc.sim.env import CubliEnv
from cubli_mpc.sim.recorder import VideoRecorder
from cubli_mpc.cli import _save_state_plot, _alloc_log, _record_step

DURATION, FPS = 8.0, 30
OUT = Path("runs/metronome")


def main() -> int:
    hw, sim = load_config("configs/default.yaml")
    ctrl = build_nmpc_metronome(hw)
    print(f"controller = {type(ctrl).__name__}")
    env = CubliEnv(hw, sim)
    env.reset(theta0=ctrl.amplitude, theta_dot0=0.0)

    spc = max(1, round(sim.dt_control / sim.dt_sim))
    n = int(round(DURATION / sim.dt_control))
    cap = max(1, round((1.0 / FPS) / sim.dt_control))
    log = _alloc_log(n)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with VideoRecorder(env.model, OUT / "metronome_nmpc.mp4", fps=FPS,
                       width=640, height=480) as rec:
        for i in range(n):
            x, t = env.state(), env.time
            tau = float(ctrl.step(x, t))
            env.apply_torque(tau)
            _record_step(log, i, t, x, tau)
            for _ in range(spc):
                env.step()
            if i % cap == 0:
                rec.capture(env.data)
    print(f"wrote {OUT/'metronome_nmpc.mp4'} in {time.time()-t0:.1f}s "
          f"(fallbacks={getattr(ctrl, 'fallback_count', 0)})")
    _save_state_plot(log, OUT / "metronome_nmpc.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

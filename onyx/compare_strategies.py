"""Compare metronome control strategies: waveform overlay + sim2real table.

Strategies:
  - FL sine      : feedback-linearized, linear spring (hardening=0).
  - FL organic   : feedback-linearized, hardening spring (coast-and-burst).
  - bang-bang    : model-light relay (coast/burst).

Outputs runs/metronome/compare.png and prints a metric table.
Run: uv run python -B onyx/compare_strategies.py
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from cubli_mpc.config import load_config
from cubli_mpc.control.metronome import (
    MetronomeController, MetronomeGains, BangBangMetronome, DEFAULT_BANGBANG,
)
from cubli_mpc.runner import Runner
from cubli_mpc.sim.env import CubliEnv
from eval_metronome import simulate, CONFIG

hw, sim = load_config(CONFIG)
plant_mismatch = replace(hw, cube_mass_kg=hw.cube_mass_kg * 1.10,
                         wheel_mass_kg=hw.wheel_mass_kg * 1.20)

STRATS = {
    "FL sine": MetronomeController(hw, MetronomeGains(
        omega0=4.305, amplitude_rad=math.radians(7.0), mu=20.0,
        inertia_scale=2.54, gravity_scale=1.25, hardening=0.0)),
    "FL organic (h=3)": MetronomeController(hw, MetronomeGains(
        omega0=2.841, amplitude_rad=math.radians(7.0), mu=20.0,
        inertia_scale=2.54, gravity_scale=1.25, hardening=3.0)),
    "bang-bang": BangBangMetronome(hw, DEFAULT_BANGBANG),
}


def one_period(ctrl):
    env = CubliEnv(hw, sim)
    env.reset(theta0=ctrl.amplitude)
    log = Runner(env, ctrl, sim).run(8.0)
    t, th, tau = log["t"], log["theta"], log["tau"]
    m = (t >= 5.0) & (t < 6.0)            # one steady period
    return t[m] - 5.0, np.degrees(th[m]), tau[m]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a0, a1) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    rows = []
    for name, ctrl in STRATS.items():
        tt, th, tau = one_period(ctrl)
        a0.plot(tt, th, label=name)
        a1.plot(tt, tau, label=name)
        pe, d = simulate(hw, ctrl, sim, 0.0)
        ps, _ = simulate(hw, ctrl, sim, 0.015)
        pm, _ = simulate(plant_mismatch, ctrl, sim, 0.0)
        rows.append((name, pe, d["torque_rms"], d["crest_factor"],
                     d["torque_max"], ps, pm))
    a0.set_ylabel("θ (deg)"); a0.grid(alpha=0.3); a0.legend(fontsize=8)
    a0.set_title("Metronome strategies — one steady period (1.0 s)")
    a1.set_ylabel("τ (Nm)"); a1.set_xlabel("time (s)"); a1.grid(alpha=0.3)
    fig.tight_layout()
    out = Path("runs/metronome"); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "compare.png", dpi=120); plt.close(fig)
    print(f"wrote {out/'compare.png'}\n")

    print(f"{'strategy':<18} {'pe_nom':>8} {'Trms':>7} {'crest':>6} "
          f"{'Tmax':>6} {'stiction':>9} {'mismatch':>9}")
    for n, pe, rms, cr, tmax, ps, pm in rows:
        print(f"{n:<18} {pe:8.5f} {rms:7.4f} {cr:6.2f} {tmax:6.3f} "
              f"{ps:9.4f} {pm:9.4f}")


if __name__ == "__main__":
    main()

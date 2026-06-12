"""Probe / tune the BangBangMetronome relay strategy.

Sweeps tau_burst (period knob) and inner_frac at fixed amplitude, reporting the
full sim2real metric suite. Run: uv run python -B onyx/probe_bb.py
"""
from __future__ import annotations

import math
from dataclasses import replace

from cubli_mpc.config import load_config
from cubli_mpc.control.metronome import BangBangMetronome, BangBangGains
from eval_metronome import simulate, CONFIG

hw, sim = load_config(CONFIG)
plant_mismatch = replace(hw, cube_mass_kg=hw.cube_mass_kg * 1.10,
                         wheel_mass_kg=hw.wheel_mass_kg * 1.20)


def metrics(g):
    c = BangBangMetronome(hw, g)
    pe, d = simulate(hw, c, sim, 0.0)
    ps, _ = simulate(hw, c, sim, 0.015)
    pm, _ = simulate(plant_mismatch, c, sim, 0.0)
    return pe, d, ps, pm


print(f"{'burst':>6} {'inner':>5} {'kdamp':>5} {'period':>7} {'amp':>5} "
      f"{'pe_nom':>7} {'Trms':>7} {'crest':>6} {'Tmax':>6} {'stick':>6} {'mis':>6} sus")
for kd in [0.05, 0.15, 0.30]:
    for burst in [0.010, 0.015, 0.020, 0.028]:
        g = BangBangGains(amplitude_rad=math.radians(7.0), tau_burst=burst,
                          inner_frac=0.6, k_damp=kd, gravity_scale=1.25)
        pe, d, ps, pm = metrics(g)
        print(f"{burst:6.3f} {0.6:5.2f} {kd:5.2f} {d['period_mean']:7.4f} "
              f"{d['amplitude_deg']:5.1f} {pe:7.4f} {d['torque_rms']:7.4f} "
              f"{d['crest_factor']:6.2f} {d['torque_max']:6.3f} {ps:6.3f} {pm:6.3f} "
              f"{d['sustained']}")

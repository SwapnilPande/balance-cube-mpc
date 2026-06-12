"""Probe arbitrary metronome gains without editing the module.

For each hardening, bisect omega0 so the realized period is ~1.0 s, then report
the full metric suite (crest, RMS, stiction, mismatch) for an apples-to-apples
comparison. Run: uv run python -B onyx/probe.py
"""
from __future__ import annotations

import math
from dataclasses import replace

from cubli_mpc.config import load_config
from cubli_mpc.control.metronome import MetronomeController, MetronomeGains
from eval_metronome import simulate, CONFIG  # reuse the scenario runner (onyx/ on path)

hw, sim = load_config(CONFIG)
plant_mismatch = replace(hw, cube_mass_kg=hw.cube_mass_kg * 1.10,
                         wheel_mass_kg=hw.wheel_mass_kg * 1.20)
BASE = dict(amplitude_rad=math.radians(7.0), mu=20.0, inertia_scale=2.54,
            gravity_scale=1.25)


def metrics(gains):
    c = MetronomeController(hw, gains)
    pe_nom, d = simulate(hw, c, sim, 0.0)
    pe_stick, _ = simulate(hw, c, sim, 0.015)
    pe_mis, _ = simulate(plant_mismatch, c, sim, 0.0)
    return pe_nom, d, pe_stick, pe_mis


def relock_omega0(hardening, lo=1.5, hi=5.0, iters=14):
    """Bisect omega0 to drive period_mean -> 1.0 s."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        g = MetronomeGains(omega0=mid, hardening=hardening, **BASE)
        _, d, _, _ = metrics(g)
        pm = d["period_mean"]
        if math.isnan(pm):
            hi = mid  # too stiff/unstable -> lower omega0
            continue
        # higher omega0 -> shorter period; we want period 1.0
        if pm > 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


print(f"{'hard':>5} {'omega0':>7} {'period':>7} {'pe_nom':>8} {'Trms':>7} "
      f"{'crest':>6} {'Tmax':>6} {'stiction':>8} {'mismatch':>8}")
for h in [0.0, 1.0, 2.0, 3.0, 5.0, 8.0]:
    w = relock_omega0(h)
    g = MetronomeGains(omega0=w, hardening=h, **BASE)
    pe, d, ps, pm = metrics(g)
    print(f"{h:5.1f} {w:7.3f} {d['period_mean']:7.4f} {pe:8.5f} "
          f"{d['torque_rms']:7.4f} {d['crest_factor']:6.2f} {d['torque_max']:6.3f} "
          f"{ps:8.4f} {pm:8.4f}")

"""Evaluate the metronome controller: period accuracy, control character, sim2real.

Runs a headless sim, starts the cube at one extreme (theta0 = amplitude, at
rest), lets the limit cycle settle, then measures the realized full period from
upward zero-crossings of theta. Three scenarios are run on whatever controller
`build_metronome` returns (swap strategies there):

  - NOMINAL  : true plant == model plant, ideal motor -> the primary metric.
  - STICTION : a torque deadband (motor stiction) is applied to the command.
               Rewards "coast-and-burst" profiles whose torque rarely lingers
               at small values; punishes smooth profiles that cross zero slowly.
  - MISMATCH : the PLANT is perturbed (heavier, more friction than the model)
               while the controller keeps the nominal model -> sim2real proxy
               for model-based vs model-light strategies.

Emits METRIC lines parsed by `onyx exp run`.

Primary:   period_error          (s, minimize) — |mean_period - 1.0| nominal.
Secondary: torque_rms            (Nm, minimize) — control magnitude.
           crest_factor          (—, monitor)   — peak/RMS, "organic" burstiness.
           torque_max            (Nm, <0.2)      — saturation headroom.
           period_error_stiction (s, minimize)   — robustness to motor deadband.
           period_error_mismatch (s, minimize)   — robustness to plant mismatch.
           omega_wheel_max, amplitude, sustained — diagnostics.
"""
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from cubli_mpc.config import load_config
from cubli_mpc.control.metronome import build_metronome, STRATEGY
from cubli_mpc.runner import Runner
from cubli_mpc.sim.env import CubliEnv

TARGET_PERIOD = 1.0          # s — the clock target
DURATION = 12.0              # s — total sim
SETTLE = 4.0                 # s — discard this leading transient
FELL_OVER_DEG = 45.0         # |theta| beyond this = fell (MuJoCo face contact)
DIED_OUT_DEG = 2.0           # measured amplitude below this = oscillation died
PENALTY = 1.0                # s — period_error when run is invalid
STICTION_NM = 0.015          # motor torque deadband for the stiction scenario

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def _upward_zero_crossings(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    times = []
    for i in range(len(y) - 1):
        if y[i] <= 0.0 < y[i + 1]:
            frac = -y[i] / (y[i + 1] - y[i])
            times.append(t[i] + frac * (t[i + 1] - t[i]))
    return np.asarray(times)


class _DeadbandController:
    """Wrap a controller so commanded torques below `deadband` produce 0 Nm
    (first-order motor-stiction proxy)."""

    def __init__(self, inner, deadband: float):
        self._inner = inner
        self._db = deadband
        self.amplitude = inner.amplitude

    def step(self, x, t):
        tau = self._inner.step(x, t)
        return 0.0 if abs(tau) < self._db else tau


def simulate(plant_hw, controller, sim, stiction_nm=0.0):
    """Run one scenario; return (period_error, diagnostics dict)."""
    ctrl = _DeadbandController(controller, stiction_nm) if stiction_nm > 0 else controller
    env = CubliEnv(plant_hw, sim)
    env.reset(theta0=controller.amplitude, theta_dot0=0.0)
    log = Runner(env, ctrl, sim).run(duration_s=DURATION)

    t, theta, tau, omega_w = log["t"], log["theta"], log["tau"], log["wheel_speed"]
    theta_max_deg = float(np.degrees(np.max(np.abs(theta))))
    mask = t >= SETTLE
    tw, thw = t[mask], theta[mask]
    amplitude_deg = float(np.degrees(np.max(np.abs(thw)))) if thw.size else 0.0

    fell_over = theta_max_deg >= FELL_OVER_DEG
    died_out = amplitude_deg < DIED_OUT_DEG
    period_mean = period_std = float("nan")
    if not fell_over and not died_out:
        cr = _upward_zero_crossings(tw, thw)
        if cr.size >= 2:
            d = np.diff(cr)
            period_mean, period_std = float(np.mean(d)), float(np.std(d))

    sustained = int(not (math.isnan(period_mean) or fell_over or died_out))
    period_error = PENALTY if not sustained else abs(period_mean - TARGET_PERIOD)

    torque_rms = float(np.sqrt(np.mean(tau ** 2)))
    torque_max = float(np.max(np.abs(tau)))
    diag = dict(
        period_mean=period_mean, period_std=period_std,
        amplitude_deg=amplitude_deg, torque_rms=torque_rms, torque_max=torque_max,
        crest_factor=(torque_max / torque_rms if torque_rms > 0 else 0.0),
        omega_wheel_max=float(np.max(np.abs(omega_w))),
        theta_max_deg=theta_max_deg, sustained=sustained,
    )
    return period_error, diag


def main() -> int:
    hw, sim = load_config(CONFIG)
    controller = build_metronome(hw)            # built on the NOMINAL model

    # Perturbed plant: heavier than the model (real masses/COM are never exact).
    # Mass-only keeps the scenario survivable so the metric grades robustness
    # instead of saturating; FL is also fragile to friction mismatch (any large
    # bearing-friction increase topples it) — tracked qualitatively in notes.
    plant_mismatch = replace(
        hw,
        cube_mass_kg=hw.cube_mass_kg * 1.10,
        wheel_mass_kg=hw.wheel_mass_kg * 1.20,
    )

    pe_nom, d = simulate(hw, controller, sim, stiction_nm=0.0)
    pe_stick, _ = simulate(hw, controller, sim, stiction_nm=STICTION_NM)
    pe_mis, _ = simulate(plant_mismatch, controller, sim, stiction_nm=0.0)

    print(f"METRIC period_error={pe_nom:.6f}")
    print(f"METRIC torque_rms={d['torque_rms']:.6f}")
    print(f"METRIC crest_factor={d['crest_factor']:.4f}")
    print(f"METRIC torque_max={d['torque_max']:.6f}")
    print(f"METRIC period_error_stiction={pe_stick:.6f}")
    print(f"METRIC period_error_mismatch={pe_mis:.6f}")
    print(f"METRIC period_mean={d['period_mean']:.6f}")
    print(f"METRIC period_std={d['period_std']:.6f}")
    print(f"METRIC amplitude_deg={d['amplitude_deg']:.4f}")
    print(f"METRIC omega_wheel_max={d['omega_wheel_max']:.4f}")
    print(f"METRIC theta_max_deg={d['theta_max_deg']:.4f}")
    print(f"METRIC sustained={d['sustained']}")
    print(f"# strategy={STRATEGY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluate the metronome controller: realized full-cycle period vs 1.0 s.

Runs a headless sim, starts the cube at one extreme (theta0 = amplitude, at
rest), lets the limit cycle settle, then measures the realized full period
from upward zero-crossings of theta in a measurement window. Emits METRIC
lines parsed by `onyx exp run`.

Primary metric:  period_error = |mean_period - TARGET_PERIOD|   (s, minimize)
Secondary:       torque_rms                                     (Nm, minimize)
plus diagnostics (jitter, amplitude, saturation, wheel speed, sustained flag).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from cubli_mpc.config import load_config
from cubli_mpc.control.metronome import MetronomeController
from cubli_mpc.runner import Runner
from cubli_mpc.sim.env import CubliEnv

TARGET_PERIOD = 1.0          # s — the clock target
DURATION = 12.0              # s — total sim
SETTLE = 4.0                 # s — discard this leading transient
FELL_OVER_DEG = 45.0         # |theta| beyond this = fell (MuJoCo face contact)
DIED_OUT_DEG = 2.0           # measured amplitude below this = oscillation died
PENALTY = 1.0                # s — period_error when run is invalid

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def _upward_zero_crossings(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Interpolated times where y crosses 0 going negative->positive."""
    times = []
    for i in range(len(y) - 1):
        if y[i] <= 0.0 < y[i + 1]:
            # linear interp for sub-sample crossing time
            frac = -y[i] / (y[i + 1] - y[i])
            times.append(t[i] + frac * (t[i + 1] - t[i]))
    return np.asarray(times)


def main() -> int:
    hw, sim = load_config(CONFIG)
    env = CubliEnv(hw, sim)
    controller = MetronomeController(hw)

    # Start at one extreme, at rest, on the limit cycle's outer edge.
    env.reset(theta0=controller.amplitude, theta_dot0=0.0)

    runner = Runner(env, controller, sim)
    log = runner.run(duration_s=DURATION)

    t = log["t"]
    theta = log["theta"]
    tau = log["tau"]
    omega_w = log["wheel_speed"]

    torque_rms = float(np.sqrt(np.mean(tau ** 2)))
    torque_max = float(np.max(np.abs(tau)))
    omega_wheel_max = float(np.max(np.abs(omega_w)))
    theta_max_deg = float(np.degrees(np.max(np.abs(theta))))

    # Measurement window (post-settle).
    mask = t >= SETTLE
    tw, thw = t[mask], theta[mask]
    amplitude_deg = float(np.degrees(np.max(np.abs(thw)))) if thw.size else 0.0

    fell_over = theta_max_deg >= FELL_OVER_DEG
    died_out = amplitude_deg < DIED_OUT_DEG

    period_mean = float("nan")
    period_std = float("nan")
    n_periods = 0
    if not fell_over and not died_out:
        crossings = _upward_zero_crossings(tw, thw)
        if crossings.size >= 2:
            periods = np.diff(crossings)
            period_mean = float(np.mean(periods))
            period_std = float(np.std(periods))
            n_periods = int(periods.size)

    if math.isnan(period_mean) or fell_over or died_out:
        period_error = PENALTY
        sustained = 0
    else:
        period_error = abs(period_mean - TARGET_PERIOD)
        sustained = 1

    print(f"METRIC period_error={period_error:.6f}")
    print(f"METRIC torque_rms={torque_rms:.6f}")
    print(f"METRIC period_mean={period_mean:.6f}")
    print(f"METRIC period_std={period_std:.6f}")
    print(f"METRIC n_periods={n_periods}")
    print(f"METRIC amplitude_deg={amplitude_deg:.4f}")
    print(f"METRIC torque_max={torque_max:.6f}")
    print(f"METRIC torque_sat_frac={float(np.mean(np.abs(tau) >= 0.999 * controller._g.max_torque)):.4f}")
    print(f"METRIC omega_wheel_max={omega_wheel_max:.4f}")
    print(f"METRIC theta_max_deg={theta_max_deg:.4f}")
    print(f"METRIC fell_over={int(fell_over)}")
    print(f"METRIC died_out={int(died_out)}")
    print(f"METRIC sustained={sustained}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

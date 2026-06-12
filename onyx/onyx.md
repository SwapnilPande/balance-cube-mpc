# Onyx Research: cubli metronome (1 Hz clock swing)

## Objective

Make the reaction-wheel cube (Cubli) tick back and forth about upright like a
clock pendulum, with a **full-cycle period of 1.0 s**. One full period = the
cube swings +θ → −θ → +θ (returns to the same extreme moving the same way).

The controller lives in `src/cubli_mpc/control/metronome.py`: a
feedback-linearized limit-cycle (Van der Pol-style) oscillator about θ=0. It
cancels gravity and imposes a desired oscillator
`theta_ddot_des = -omega0^2 * r(theta) - mu*(energy - A^2)*theta_dot`. Nominal
period ≈ 2π/omega0; realized period drifts because the analytic inversion
(uses cube-only inertia `I_b`) mismatches MuJoCo's true tilt inertia (which
includes wheel mass + coupling). Closing that gap to land at exactly 1.0 s,
cheaply, is the whole game.

## Metrics

- **Primary**: `period_error`, s, **minimize** — |mean realized full period − 1.0|.
  When the run is invalid (cube fell past 45°, or oscillation died < 2° amp),
  period_error is set to the penalty 1.0 s.
- **Secondary** (tradeoff monitors, rarely override primary):
  - `torque_rms`, Nm, minimize — control magnitude (the user's stated 2nd goal).
  - `torque_max` / `torque_sat_frac` — saturation pressure (0.2 Nm cap).
  - `omega_wheel_max`, rad/s — wheel must stay under motor_max_speed (600).
  - `amplitude_deg`, `period_std` (jitter), `n_periods`, `sustained` (0/1).

## How to Run

`./onyx/eval.sh` — runs `onyx/eval_metronome.py` (12 s sim, 4 s settle, measure
period from upward zero-crossings of θ). Outputs `METRIC name=value` lines.
~0.45 s/run.

## Files in Scope

- `src/cubli_mpc/control/metronome.py` — THE controller. Tune `DEFAULT_GAINS`
  (omega0, amplitude_rad, mu, use_sin_restoring, max_torque) and/or restructure
  the control law. This is the primary lever.
- `onyx/eval_metronome.py` — eval harness. Edit ONLY to add signal/diagnostics
  (e.g. better period estimator), never to make the metric easier to game.

## Off Limits

- The plant / simulator: `sim/env.py`, `model/mjcf.py`, `model/dynamics.py`,
  `config.py`, `configs/default.yaml`. Don't tune the plant to fake the metric.
- Existing controllers, NMPC, swing planner, RL, tests.
- `TARGET_PERIOD = 1.0`, the 45°/2° validity gates, and the period estimator's
  definition (don't relax validity to "win").

## Constraints

- Hard torque cap 0.2 Nm (motor_max_torque_nm). Keep wheel speed < 600 rad/s.
- The controller must be a real-time feedback law (state → torque each tick);
  no offline trajectory baked to the clock, no peeking at wall-clock `t` to
  force the period (the period must emerge from the closed-loop dynamics).

## What's Been Tried

(baseline) Feedback-linearized linear-restoring oscillator, omega0=2π, A=15°,
mu=8: realized period **1.84 s** (≈2× target), amplitude drifted to 32°, wheel
speed 1337 rad/s (over limit), torque saturating ~9%. Root cause: inversion
uses cube-only I_b; MuJoCo's true tilt inertia is larger → realized oscillator
~3× softer → period too long + amplitude regulation too weak.

Key levers / hypotheses to explore:
- Raise omega0 to compensate the softening (period scales ~1/omega_eff).
- Identify/scale effective tilt inertia in the inversion so omega0 maps to
  realized frequency 1:1 (then period control becomes direct).
- Control amplitude: stronger mu, or seed IC on the cycle; keep A small to cut
  torque and wheel speed (torque ∝ gravity feedforward ∝ sin(amplitude)).
- sin vs linear restoring trades period-vs-amplitude coupling.

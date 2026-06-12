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

**SOLVED.** Best: omega0=4.305, A=7°, mu=20, inertia_scale=2.54,
gravity_scale=1.25, linear restoring → period **1.000042 s** (period_error
0.000042 s), torque_rms 0.043 Nm, wheel speed 161 rad/s. Clean limit cycle
(period_std ~0), globally attracting, robust to sensor noise.

Path that got here:
1. (baseline) omega0=2π, A=15°, mu=8, no scales: period 1.84 s, amplitude ran
   away to 32°, heavy saturation. Two compounding errors: under-modeled
   tau→tilt gain AND under-cancelled gravity.
2. **inertia_scale=2.54** — MuJoCo's realized tau→theta_ddot gain is ~2.54×
   softer than the analytic 1/I_b (wheel↔cube torque coupling, not just mass).
   Fixing it kills the amplitude runaway → energy regulator holds A exactly.
3. **gravity_scale=1.25** — config `mgL` counts cube mass only; true gravity
   moment includes the wheel (~1.25×). Under-cancelling left residual
   *destabilizing* gravity that softened the spring amplitude-dependently.
   Full cancellation makes **period amplitude-independent** (verified A=5–12°).
4. omega0 then sets the period linearly; omega0=4.305 nails 1.000 s.

Key facts (don't re-derive):
- Period is amplitude-independent → amplitude is a FREE knob for the torque
  (secondary) objective. Torque ≈ proportional to amplitude.
- Torque is dominated by gravity feedforward; gravity-fight and restoring have
  the SAME sign (inverted pendulum) so they add — no cancellation trick exists.
  Min torque = min visible amplitude. Floor set by the 2° die-out gate +
  "must look like a clock" (kept ~7° as a clear, robust swing).
- Limit cycle is globally attracting: self-starts to the same cycle from any IC
  (0.5–15°, even upright+kick). A genuine autonomous clock.
- The cube oscillates about UPRIGHT (inverted-pendulum balance point).

Dead ends / non-levers: chasing omega0 past ~4 decimals = overfitting the
period estimator's numerical floor (no physical meaning). use_sin_restoring
unused (linear is clean and amplitude-independent once gravity is cancelled).

Gotcha: rapid same-second edits can reuse stale .pyc → eval.sh sets
PYTHONDONTWRITEBYTECODE/-B. Always trust eval.sh, not hand sweeps.

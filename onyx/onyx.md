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

User wants the motion to feel **organic** (a lower-RMS, higher-peak "coast and
burst" profile, not a clinical sine) and the strategies considered with an eye
to **sim2real** (real motor stiction, plant-parameter mismatch). Explore
multiple control strategies.

## Metrics

- **Primary**: `period_error`, s, **minimize** — |mean realized full period − 1.0|
  (nominal plant, ideal motor). Invalid run (fell >45° / died <2°) → penalty 1.0 s.
- **Secondary** (tradeoff monitors):
  - `torque_rms`, Nm, minimize — control magnitude (user's 2nd goal).
  - `crest_factor`, peak/RMS — "organic" burstiness (sine=1.41; higher=burstier).
  - `torque_max`, Nm — must stay < 0.2 cap.
  - `period_error_stiction`, s — robustness to a 0.015 Nm motor torque deadband.
  - `period_error_mismatch`, s — robustness to a perturbed plant (cube +10%,
    wheel +20% mass) with the controller on nominal params. **Sim2real proxy.**
  - `omega_wheel_max` (<600), `amplitude_deg`, `period_std`, `sustained`.

## How to Run

`./onyx/eval.sh` → `onyx/eval_metronome.py`. Runs 3 scenarios (nominal /
stiction / mismatch) on whatever `build_metronome` returns. ~0.25 s/run.
Helpers: `onyx/probe.py` (FL hardening sweep, auto-relocks ω0), `onyx/probe_bb.py`
(bang-bang sweep), `onyx/compare_strategies.py` (waveform overlay + table).

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

### Phase 2 — organic profile + multiple strategies + sim2real

**Current default = FL organic, `hardening=3`, omega0=2.841.** Period 1.0000 s,
torque_rms **0.0314** (−26% vs sine), crest **2.19**, mismatch robustness
**0.131** (−55% vs sine).

- **Hardening spring** `r(θ)=θ(1+h·(θ/A)²)` makes the coast-and-burst profile.
  Re-lock ω0 (stiffer → faster). Sweeping h (ω0 re-locked to 1.0 s):
  h: 0→2→3→5→8 gives RMS 0.043→0.033→0.031→0.029→0.027, crest 1.4→2.0→2.2→2.6→3.1.
  Hardening LOWERS RMS *and* raises crest *and* improves mass-mismatch robustness
  (the stiff spring dominates the restoring, so gravity-cancellation error
  matters less). It WORSENS the command-deadband stiction metric (longer
  near-zero coast gets eaten). h=3 chosen (organic, matches the min-effort
  crest≈2.3 the user liked, mild stiction cost).

Strategies compared (`onyx/compare_strategies.py`):
- **FL sine** (h=0): RMS 0.043, crest 1.42, stiction 0.015, mismatch 0.29.
- **FL organic** (h=3): RMS 0.031, crest 2.19, stiction 0.076, mismatch 0.13. ★
- **bang-bang relay**: pe 0.015, RMS 0.041, crest 1.47, stiction 0.065,
  mismatch **falls (1.0)**. DOMINATED — see below.

Sim2real conclusions:
- **Mass mismatch:** FL period is sensitive via the *gravity* cancellation (sine
  drifts to 1.29 s under +10/20% mass). HARDENING fixes most of this.
- **Friction mismatch:** FL TOPPLES at ≥2× bearing friction (not in the metric;
  too harsh). Real fragility → a friction feedforward / integral term is the
  sim2real to-do.
- **Motor stiction:** for an *autonomous limit cycle* a torque deadband hurts
  (loses small coast corrections). The reaction WHEEL spins continuously (±150
  rad/s) so breakaway stiction is mild vs a direct-drive joint; torque
  *resolution/quantization* is the real limit. The user's "bursts beat stiction"
  intuition applies to *reference-tracking* control (needs precise small torques
  through zero), a strategy not yet built — see `onyx.ideas.md`.
- **bang-bang is ill-suited to a SLOW inverted metronome:** reaching 1.0 s forces
  a weak burst (no authority margin → topples under mismatch); and an inverted
  pendulum can't glide slowly at angle (it falls), so the strong-burst+long-coast
  recipe that makes a relay robust is unreachable here. Bang-bang is also NOT
  organic (square-ish → low crest).

Dead ends / non-levers: chasing omega0 past ~4 decimals = overfitting the
period estimator's numerical floor. use_sin_restoring unused. Pure bang-bang
relay (dominated). 

Gotcha: rapid same-second edits can reuse stale .pyc → eval.sh sets
PYTHONDONTWRITEBYTECODE/-B. Always trust eval.sh, not hand sweeps.

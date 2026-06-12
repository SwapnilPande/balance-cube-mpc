# Metronome — ideas backlog

Strategies / directions discovered but not (fully) pursued. Prune as tried.

## Control strategies
- **Reference-tracking (feedforward + PD), time-indexed.** Track θ_ref(t) =
  coast-burst (or the swing_planner min-effort) trajectory. Period is EXACT and
  the clock keeps absolute wall-time (arguably the *correct* clock behavior — a
  real clock keeps time, it doesn't phase-drift). This is where the user's
  stiction intuition lives: a sine-tracking controller needs precise small
  torques through the zero-crossing and should degrade hard under a deadband,
  while a coast-burst feedforward puts the big torques where they're needed.
  Repo already has `swing_planner` + `PeriodicTrajectoryReference` + NMPC; a
  light version = precomputed feedforward τ(t) + PD on tracking error (fast).
  Compare stiction robustness vs the autonomous limit cycle.
- **Self-tuning FL (online period lock).** A slow outer loop that nudges
  `gravity_scale` / `omega0` so the *measured* period → 1.0 s on the real plant.
  Would absorb mass/inertia mismatch (the main FL sim2real weakness) and make a
  self-calibrating clock. The limit cycle already self-corrects phase; only the
  period needs a slow integrator. High-value, low-risk.
- **Friction feedforward / integral term.** FL topples at ≥2× bearing friction.
  Add a friction estimate (or integral action on energy) so the controller
  replaces real friction losses. Fixes the biggest sim2real fragility.
- **RL policy (SAC, repo has training).** Train the metronome task with domain
  randomization over mass / friction / stiction / latency → sim2real-by-design,
  robust by construction. Heavy; only if the analytic controllers plateau.
- **NMPC tracking the min-effort plan.** The crest≈2.3 optimum. FL hardening
  already matches its RMS/crest cheaply in closed loop, so likely not worth the
  IPOPT cost — but would confirm the optimum and give exact period.

## Eval / realism
- **Physical stiction model.** Replace the command deadband with a
  wheel-velocity-dependent Coulomb + breakaway friction (more faithful for a
  spinning reaction wheel). Re-rank strategies — may flip the stiction result,
  since the wheel rarely sits at zero speed.
- **Latency + torque quantization** stressors (more real-motor effects).
- **Randomized mismatch** (sample several param perturbations, report worst /
  mean) instead of one fixed perturbation, for a less brittle sim2real metric.

## Aesthetic
- Higher hardening (h=5–8) for a more dramatic coast-burst if the user wants a
  snappier tick (RMS even lower, but stiction metric worsens).

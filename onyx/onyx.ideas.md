# Metronome — ideas backlog

Strategies / directions discovered but not (fully) pursued. Prune as tried.

## Control strategies
- **DONE: NMPC tracking the min-effort plan** (`build_nmpc_metronome`, exp
  `nmpc-track-plan`). Period exact under nominal/stiction/mismatch; validates the
  user's bursts-beat-stiction intuition for tracking control. Cost: RMS 0.040
  (tracking overhead) + IPOPT/tick.
- **Light reference-tracking (feedforward τ*(t) + PD), no solver.** Same
  time-indexed coast-burst reference as the NMPC but replace IPOPT with the
  precomputed plan torque as feedforward + a PD on tracking error. Should keep
  most of NMPC's period-robustness at a fraction of the compute — the practical
  sim2real-friendly version. Compare RMS / robustness to full NMPC.
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

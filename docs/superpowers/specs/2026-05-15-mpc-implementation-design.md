# MPC Implementation Design

**Date:** 2026-05-15
**Status:** Draft — awaiting user review
**Parent spec:** [`2026-05-14-cubli-mpc-design.md`](2026-05-14-cubli-mpc-design.md)

## Purpose

The parent spec laid out four milestones (M1–M4) for cubli-mpc and was approved on 2026-05-14. Implementation followed M1 (nonlinear PD baseline) and then pivoted to RL plumbing in lieu of M2/M3, with the explicit intent to return to MPC. This is that return.

This document refines the parent spec's M2 (stabilization NMPC) and M3 (swing trajectory tracking) into a concrete implementation plan. It does **not** revisit settled decisions from the parent spec (MuJoCo for sim, CasADi for symbolic dynamics, IPOPT for the prototype solver, 3-DOF MPC state, RK4 inside the prediction model, scalar-torque actuation, single-period offline trajopt for tracking). Where the parent spec is sufficient, this document refers to it rather than restating.

## What this spec adds beyond the parent

1. Concrete CasADi dynamics module (`model/dynamics.py`).
2. Concrete NMPC controller (`control/nmpc.py`) with warm-start and solver-failure fallback.
3. Unified handling of the time-varying reference so the same code path serves balance (`x_ref ≡ upright`) and tracking (`x_ref` from the swing planner).
4. Offline swing planner (`control/swing_planner.py`) for the M3 reference trajectory, with the metronome-style framing made explicit.
5. A `cubli-mpc eval` harness that runs any `Controller` (PD / random / policy / NMPC) over a defined scenario and emits comparable metrics — this is the artifact that turns "we have MPC" into "MPC vs RL, with numbers."

## M2 — Stabilization NMPC

### Symbolic dynamics: `src/cubli_mpc/model/dynamics.py`

Hand-derived Lagrangian dynamics for the 1-axis reaction-wheel cube, written symbolically in CasADi.

**State (MPC):** `x = [θ, θ̇, ω_w]` ∈ ℝ³. Wheel angle is cyclic; dropped from prediction state per parent spec.

**Input:** `u = τ` ∈ ℝ¹, the wheel motor torque.

**Continuous-time dynamics** (derived from the Lagrangian of a uniform cube balanced on a hinged edge with a flywheel on a parallel internal axis, with `ω_w` measured in the body frame). The Euler-Lagrange equations give a coupled mass matrix:

```
[I_b + I_w   I_w] [θ̈ ]   [m·g·L·sin(θ) − b_e·θ̇]
[I_w         I_w] [ω̇_w] = [τ − b_w·ω_w         ]
```

Inverting (det = I_b · I_w) gives the explicit ODE:

```
θ̈    = (m·g·L·sin θ − b_e·θ̇ − τ + b_w·ω_w) / I_b
ω̇_w  = (τ − b_w·ω_w) / I_w  −  θ̈
```

The motor torque appears with opposite signs in the two equations: positive τ accelerates the wheel and decelerates the body (which is what we use to balance). The `+b_w·ω_w/I_b` cross-coupling is small in practice (b_w ~ 1e-5) but kept for derivational correctness. Constants come from `HardwareConfig`:
- `I_b = hw.cube_inertia_about_edge`
- `I_w = hw.wheel_inertia_about_spin_axis`
- `m·g·L = hw.gravity_moment_coefficient`
- `b_e = hw.edge_bearing_damping`, `b_w = hw.wheel_bearing_damping`

Note: `cube_inertia_about_edge` is the bare cube — wheel mass parallel-axis is absorbed by MuJoCo in the sim model, deliberately introducing the small model mismatch the parent spec calls for. If the gap turns out to matter at evaluation time, add a wheel parallel-axis term here without touching MuJoCo.

The module exposes:

```python
def make_continuous_dynamics(hw: HardwareConfig) -> ca.Function:
    """Return CasADi Function f(x, u) -> x_dot for the 3D MPC state."""

def make_rk4_step(hw: HardwareConfig, dt: float) -> ca.Function:
    """Return CasADi Function F(x, u) -> x_next using explicit RK4."""
```

`make_rk4_step` is what the NMPC composes into a multiple-shooting NLP. RK4 because the dynamics are smooth and we want a fixed-step explicit integrator for predictable solve time.

**Validation test:** for a panel of (x₀, u, T) tuples, integrate `dynamics.py` forward with `make_rk4_step` and integrate the MuJoCo plant from the same initial state with the same constant torque; assert the trajectories agree to a tolerance (~1° on θ, ~1% on ω_w) over a horizon comparable to the MPC horizon (~0.5 s). This pins down model-mismatch before the controller is wrapped around it.

### NMPC controller: `src/cubli_mpc/control/nmpc.py`

```python
@dataclass(frozen=True)
class NMPCConfig:
    horizon_steps: int = 50              # N
    dt: float = 0.010                    # matches SimConfig.dt_control
    q_theta: float = 50.0
    q_theta_dot: float = 1.0
    q_omega_w: float = 1e-4
    r_torque: float = 0.01
    terminal_scale: float = 10.0         # multiplies the stage cost at k = N
    omega_w_limit: float = float('inf')  # disabled by default; set from hw.motor_max_speed_rad_s
    ipopt_max_iter: int = 50
    ipopt_print_level: int = 0
    fallback: str = "nonlinear_pd"       # "nonlinear_pd" | "zero" | "last_good"
```

**Formulation** (direct multiple shooting):

```
min   Σ_{k=0..N-1} (x_k − x_ref_k)ᵀ Q (x_k − x_ref_k) + r_τ · (u_k − u_ref_k)²
        + terminal_scale · (x_N − x_ref_N)ᵀ Q (x_N − x_ref_N)
s.t.  x_{k+1} = F(x_k, u_k)              (RK4 step)
      x_0     = x_meas                   (initial state)
      |u_k|   ≤ τ_max
      |ω_{w,k}| ≤ ω_w_limit              (slack if needed; see below)
```

**No reference object passed in M2** → defaults `x_ref ≡ 0`, `u_ref ≡ 0`. The signature is wired for the M3 swap from day one.

**Wheel-speed constraint** is a hard inequality if `omega_w_limit < ∞`. If IPOPT struggles, we relax to a slack-variable formulation with a large penalty — implementation detail, not surfaced in the config until needed.

**Warm-start.** The NMPC instance keeps the previous (x, u) trajectory. On each call:
1. Shift it one step forward, append a copy of the last value as the new tail.
2. Pass the shifted trajectory as the initial guess to IPOPT.

This is the difference between "IPOPT converges in 3 iterations at 10 ms" and "IPOPT thrashes for 100 iterations." Not optional.

**Solver-failure fallback.** If IPOPT returns non-success status or wall-clock exceeds a soft deadline (e.g., 1.5 × dt_control), use the fallback strategy:
- `"nonlinear_pd"`: ask a pre-configured `NonlinearPDController` for this tick (default). Conservative; we know it works.
- `"zero"`: command zero torque (lets gravity tip; useful for debugging which controller is responsible for a failure).
- `"last_good"`: replay the previous successful solution's u_0. Cheap but can compound errors.

A counter on the controller tracks fallback rate; the eval harness surfaces it.

**State adaptation.** The runner passes 4D `x_hat`; NMPC slices to `[x_hat[0], x_hat[1], x_hat[3]]`. No interface change.

**Reference interface** (forward-compatible with M3):

```python
class Reference(Protocol):
    def at(self, t: float, dt: float, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (x_ref [H+1, 3], u_ref [H, 1]) sampled at t + k·dt."""
```

In M2 the default is `UprightReference()` returning zeros of the right shape. In M3 we swap in a `PeriodicTrajectoryReference` loaded from disk.

### M2 success criteria

- Recovers from a 20° initial tilt with all defaults in deterministic sim.
- Sustained wheel speed stays bounded (within `motor_max_speed_rad_s`) for at least 60 s deterministic sim, with default Q weights.
- IPOPT solve time p95 ≤ `dt_control` on the dev machine (informational, not gating).
- Beats `NonlinearPDController` on the eval harness's disturbance-rejection scenario (defined below).

## M3 — Swing trajectory tracking (the "metronome")

### Reframing the parent spec for metronome semantics

The parent spec described M3 as "swing back and forth between defined extrema, indefinitely." The recent RL conversation refined the constraint set: hit ±θ_target every period T, but **leave the trajectory between waypoints free** so the controller can exploit momentum tricks. The two-phase approach handles this naturally: the offline trajopt's min-effort objective discovers the momentum-using solution, and the online NMPC tracks that solution with enough feedforward to make it work.

If we later want even more freedom (e.g., let the system pick its own period within a band, or let amplitude breathe), that's an extension to the planner's constraints — not a different architecture.

### Offline swing planner: `src/cubli_mpc/control/swing_planner.py`

A standalone module that solves a full-horizon trajectory optimization once and saves the result to disk.

```python
@dataclass(frozen=True)
class SwingTrajConfig:
    period_s: float                   # T
    theta_target_rad: float           # ±θ amplitude
    n_segments: int = 200             # per period
    smoothness_weight: float = 1e-3   # on ∫τ̇² to discourage chatter
    save_path: Path | None = None     # default runs/<name>/swing.npz

def plan_swing(hw: HardwareConfig, cfg: SwingTrajConfig) -> Path:
    """Solve one-period swing trajopt, save (t, x_ref, u_ref) to disk, return path."""
```

**Formulation:** direct collocation in CasADi over one full period.

```
min   ∫₀ᵀ [τ² + smoothness_weight·τ̇²] dt
s.t.  ẋ = f(x, u)                      (continuous dynamics)
      θ(0) = +θ_target, θ̇(0) = 0      (boundary)
      θ(T) = −θ_target, θ̇(T) = 0      (boundary — half period; we mirror)
      |τ(t)| ≤ τ_max
      |ω_w(t)| ≤ ω_w_max
```

We solve a **half period** between `+θ_target` and `−θ_target`, then mirror it to get a full period. This is a discrete symmetry of the dynamics (cube and wheel both flip sign) and halves the NLP size.

**Output format:** `.npz` with arrays `t`, `x_ref` (shape `[N+1, 3]`), `u_ref` (shape `[N, 1]`).

CLI: `cubli-mpc plan-swing --config configs/default.yaml --period 1.0 --theta-deg 25 --out runs/swing-1hz.npz`.

### Online tracking reference: `PeriodicTrajectoryReference`

Loaded from the planner's output. Returns the planned `(x_ref, u_ref)` window for any (t, dt, horizon) query, looping at the period boundary. Wraps with linear interpolation between the planner's time grid and the NMPC's `dt` so the planner's `n_segments` and the NMPC's horizon are decoupled.

### M3 success criteria

- For a default `(period=1.0 s, theta_target=20°)`, the swing planner returns a feasible trajectory under the default config's torque + wheel-speed limits.
- Online NMPC tracking holds the swing for ≥ 60 s deterministic sim with RMS θ-error < 2° and wheel speed bounded.
- Visible in the viewer as a recognizable metronome.

If the (period, amplitude) requested is infeasible for the hardware config, the planner fails clearly (IPOPT infeasibility) — this is a useful co-design signal, not a bug to paper over.

## Controller integration

### Controller protocol — no change

The existing protocol in `src/cubli_mpc/control/base.py` takes `x_hat ∈ ℝ⁴` and returns scalar torque. `NMPCController.step(x_hat, t)` fits unchanged. `target_theta` is exposed as a property the way `PolicyController` does, so the CLI plot helper auto-renders the reference line.

For M3, the time argument `t` is what the reference uses to phase the lookup. The runner already passes wall-clock simulation time.

### CLI integration

Extend `cubli-mpc sim --controller`:

```
--controller {pd, random, policy, nmpc}
--nmpc-horizon N
--nmpc-dt SECS
--nmpc-target-tilt-deg D                # M2 with non-zero balance setpoint
--swing-traj PATH                       # M3: load reference from .npz
```

Add `cubli-mpc plan-swing` as a sibling subcommand.

## Evaluation harness — `cubli-mpc eval`

The reason MPC is being built at all is to compare against RL. The harness is the place where that comparison materializes.

**Scenarios** (named, in code, not YAML — these are evaluation infra, not user-tunable surfaces):

| Name | Initial state | Disturbance plan | Reference |
|---|---|---|---|
| `recover-15deg` | θ = 15°, rest | none | upright |
| `recover-30deg` | θ = 30°, rest | none | upright |
| `hold-step-disturb` | upright | +0.1 Nm step at t = 2 s for 0.5 s | upright |
| `hold-impulse-disturb` | upright | random impulses, fixed seed | upright |
| `metronome-1hz-20deg` | start of swing | none | swing planner output |

**Metrics** (per scenario):

- `survived` (bool): did the cube stay within ±60° for the full duration?
- `theta_rms_deg`
- `theta_max_abs_deg`
- `wheel_speed_max_abs_rad_s`
- `wheel_speed_rms_rad_s`
- `torque_rms_nm`
- `torque_integral_abs_nm_s` (energy proxy)
- `solver_fallback_rate` (NMPC only; 0 for others)
- `solver_p50_ms`, `solver_p95_ms` (NMPC only)

**CLI:**

```bash
cubli-mpc eval --config configs/default.yaml \
    --controller nmpc \
    --scenarios all \
    --duration 10 \
    --out runs/eval-nmpc/
```

Outputs:
- `metrics.json` with one row per scenario.
- One state-evolution plot per scenario (reuses `_save_state_plot` from `cli.py`).
- One markdown report aggregating the run.

To compare controllers, run the same command with `--controller pd`, `--controller policy --policy-path ...`, etc., and a tiny `cubli-mpc eval-report` (or just a script) collates the resulting `metrics.json` files into a side-by-side table.

## File layout

```
src/cubli_mpc/
  model/
    dynamics.py            NEW: CasADi symbolic dynamics + RK4 step
  control/
    nmpc.py                NEW: NMPCController, NMPCConfig
    swing_planner.py       NEW: plan_swing(), SwingTrajConfig
    references.py          NEW: Reference protocol + UprightReference,
                                PeriodicTrajectoryReference
  eval/
    __init__.py            NEW
    scenarios.py           NEW: scenario registry, metric definitions
    runner.py              NEW: run_scenario(controller, scenario, hw, sim)
  cli.py                   ADD: `eval`, `plan-swing` subcommands;
                                `--controller nmpc` for `sim`
tests/
  test_dynamics.py         NEW: continuous + RK4 + MuJoCo agreement
  test_nmpc.py             NEW: warm-start, fallback, balance recovery smoke
  test_swing_planner.py    NEW: feasibility + symmetry + boundary conditions
  test_eval.py             NEW: scenarios run, metrics emit
```

`pyproject.toml`: add `casadi` to the main dependencies (not behind an extra — this is core, like `mujoco`).

## Testing strategy

- **Unit:** `dynamics.py` against analytic equilibria (θ=0, u=0, ω_w=0 → ẋ=0); RK4 step against a fine Euler integration for small dt.
- **Cross-validation:** dynamics.py vs MuJoCo on a panel of open-loop trajectories (described in M2 above).
- **NMPC smoke:** one-tick solve from a defined state returns a finite torque, no IPOPT crash. Warm-start gives ≤ N IPOPT iterations on the second tick.
- **NMPC balance:** scenario `recover-15deg` survives.
- **Swing planner:** for the default `(1 Hz, 20°)` config the solver returns success, boundary conditions hold, torques within limits.
- **Eval harness:** scenarios registered, metrics computed deterministically given a seed.

## Implementation order

1. `model/dynamics.py` + unit tests + MuJoCo cross-validation.
2. `control/references.py` (just `UprightReference` for now) + `control/nmpc.py` with `UprightReference` baked in — minimal version, no warm-start, no fallback.
3. Wire `--controller nmpc` into `sim` CLI; smoke-test in the viewer.
4. Add warm-start; benchmark solve time.
5. Add fallback + counter; add to metrics.
6. `eval/` module + `cubli-mpc eval` CLI; run on PD / random / policy / nmpc.
7. `control/swing_planner.py` + `cubli-mpc plan-swing` CLI.
8. `PeriodicTrajectoryReference` + `--swing-traj` flag for `sim`.
9. M3 success-criteria run on the full stack.

Each step is independently verifiable before moving on.

## Deferred / non-goals (consistent with parent spec)

- **acados / real-time deployment.** IPOPT until M3 works, then re-evaluate.
- **State estimator robustness for MPC.** Parent spec's M4 territory.
- **Multi-amplitude or multi-period swings.** The planner takes one `(T, θ_target)` and produces one trajectory. Reparametrization is a future extension.
- **Sensor-noise hardening for the NMPC.** Initially we feed ground-truth state; sensor noise turns on once balance + tracking work clean.

## Open questions

Deferred — pick when we get to them:
- Q values: starting weights are guesses; first eval run will tell us what to tune.
- Horizon length: 50 × 10 ms is a guess. If IPOPT misses deadline, shorten before changing solvers.
- Slack vs hard wheel-speed constraint: start hard, switch to slack if it causes infeasibility under disturbance.

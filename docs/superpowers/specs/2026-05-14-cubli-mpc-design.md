# Cubli-MPC Design

**Date:** 2026-05-14
**Status:** Approved — ready for implementation planning

## Motivation

We have a working 1-axis reaction wheel cube balancer. The current controller is a two-level non-linear PD: an inner loop tracks a balance point using true non-linear energy terms, and an outer loop adjusts the balance point to bleed wheel speed. It works, but we want:

1. **Indefinite stabilization** — balance for days on end without drift, saturation, or operator intervention.
2. **Dynamic trajectory tracking** — for example, swing the cube back and forth like a pendulum, indefinitely.
3. **Unified handling of wheel-speed regulation** — collapse the current two-loop cascade into one cost.

MPC is the natural next step: it handles the non-linear dynamics, hard torque limits, and wheel-speed budget in a single optimization. This design sets up a parametric MuJoCo simulator and a CasADi-based NMPC stack, with a controller progression that lets each milestone validate the next.

## Goals

- A self-contained Python package that simulates the cube and runs a sequence of increasingly capable controllers.
- Parametric hardware model: the same code can answer "what motor torque do I need?" and "how heavy should the wheel be?"
- Visualization-first via MuJoCo viewer.
- Final controller: NMPC capable of indefinite stabilization and pendulum-swing trajectory tracking.

## Non-goals

- Multi-axis balancing (stays 1-axis).
- Swing-up from rest (M3 swings between non-rest extrema; getting to the swing initial condition is out of scope).
- `acados` / real-time deployment (premature until prototype controllers work).
- Hardware-in-the-loop testing.
- Sim2real fidelity beyond "realistic enough to make hardware co-design decisions."

## Stack

- Python ≥ 3.11
- `uv` for env + lockfile
- `mujoco` (official Python bindings) for simulation and visualization
- `casadi` for symbolic dynamics and NMPC prototyping (IPOPT solver)
- `numpy`, `scipy`, `matplotlib`
- `pydantic` or `dataclasses` for the config schema
- `pytest` for tests

`acados` is *not* a dependency yet — it gets layered in after the IPOPT-based prototype works.

## Package Layout

```
cubli-mpc/
  pyproject.toml
  src/cubli_mpc/
    __init__.py
    config.py                 # HardwareConfig, SimConfig — single source of truth
    model/
      mjcf.py                 # builds MJCF XML from HardwareConfig
      dynamics.py             # CasADi symbolic dynamics (separate from sim)
    sim/
      env.py                  # MuJoCo sim wrapper + viewer
      sensors.py              # IMU/encoder noise + delay models (off by default)
    control/
      base.py                 # Controller interface
      nonlinear_pd.py         # Baseline port of existing controller
      nmpc.py                 # CasADi-based NMPC
      swing_planner.py        # Offline trajectory optimization
    estimation/
      filter.py               # Complementary filter / simple EKF
    runner.py                 # Sim loop: env + controller + estimator + logger
    cli.py                    # `cubli-mpc run --config configs/default.yaml`
  configs/
    default.yaml
  tests/
  scripts/
    sweep_torque_limit.py     # Example co-design study
  docs/superpowers/specs/
    2026-05-14-cubli-mpc-design.md  # this file
```

### Key architectural choice: two dynamics models

The **MuJoCo model** is the simulated "real" robot. The **CasADi symbolic dynamics** are the MPC's prediction model. They are deliberately separate.

- MuJoCo: built from `HardwareConfig` via `mjcf.py`. High-fidelity, with whatever bearing friction / sensor noise / delays we want to model.
- CasADi: hand-written Lagrangian dynamics, possibly simplified. This is what the NMPC integrates inside its prediction horizon.

Keeping them separate matches how real deployment works (the controller doesn't have MuJoCo at runtime) and lets us deliberately introduce model mismatch to stress-test robustness.

### Controller interface

```python
class Controller(Protocol):
    def step(self, x_hat: np.ndarray, t: float) -> float:
        """Return commanded wheel torque (Nm)."""
```

Swapping `nonlinear_pd` ↔ `nmpc` ↔ future variants is one line in the runner.

## State Representation

**Simulator / estimator state (4D, full kinematic):**

```
x_full = [θ_body, θ̇_body, θ_wheel, θ̇_wheel]
```

This matches what sensors actually measure (IMU → body angle/rate, encoder → wheel angle/rate).

**MPC state (3D, drop the cyclic coordinate):**

```
x_mpc = [θ_body, θ̇_body, θ̇_wheel]
```

The wheel angle is a cyclic coordinate — it does not appear in the Lagrangian, so it does not need to be in the prediction state.

**Control:** scalar wheel torque `τ`, hard-bounded to `±motor_max_torque_nm`.

## Configuration

### `HardwareConfig`

The things you'd actually pick on a BOM. Inertias are *computed* from geometry+mass; the user does not hand-tune inertia tensors.

```python
@dataclass
class HardwareConfig:
    # Cube
    cube_side_length_m: float
    cube_mass_kg: float
    cube_com_offset_m: float = 0.0   # along tilt axis

    # Wheel
    wheel_mass_kg: float
    wheel_radius_m: float
    wheel_thickness_m: float
    wheel_offset_m: float = 0.0      # along tilt axis from cube COM

    # Motor
    motor_max_torque_nm: float
    motor_max_speed_rad_s: float
    motor_torque_constant: float

    # Friction
    edge_bearing_damping: float
    wheel_bearing_damping: float
```

A `derived` accessor exposes computed quantities such as:
- `cube_inertia_about_edge`
- `wheel_inertia_about_spin_axis`
- `angular_momentum_budget = wheel_inertia * motor_max_speed_rad_s` — the constraint that determines swing-trajectory feasibility.

### `SimConfig`

Orthogonal to hardware — these are simulator/runtime parameters.

```python
@dataclass
class SimConfig:
    dt_sim: float = 0.001        # MuJoCo physics step
    dt_control: float = 0.010    # Controller invocation period (100 Hz)
    sensor_noise: bool = False
    sensor_delay_ms: float = 0.0
    seed: int = 0
```

`dt_sim` and `dt_control` are decoupled: physics ticks at 1 kHz, controller fires every Nth physics tick.

## MuJoCo Model

Built by `mjcf.py` as a string of MJCF XML from `HardwareConfig`. Structure:

- `worldbody` → `cube` body, hinge joint along the balancing edge
- `cube` → `wheel` body, hinge joint along the wheel spin axis
- `motor` actuator on the wheel joint with `ctrlrange = ±motor_max_torque_nm`
- `gyro` + `accelerometer` sensors on the cube body
- `jointpos` + `jointvel` sensors on the wheel joint
- Joint damping on both hinges (set from `edge_bearing_damping` and `wheel_bearing_damping`)

Modeling decisions:

| Decision | Choice | Rationale |
|---|---|---|
| Edge contact | Frictionless hinge (welded edge) | Matches the "balance on edge" assumption; add contact later if we want lift-off / tipping behavior. |
| Wheel motor | Torque-controlled actuator with hard ctrlrange | Matches what the MPC outputs. |
| Sensor noise | Off by default | Get dynamics right first; turn on in M4. |
| Inertias | Computed from geometry + mass | Cube as uniform box, wheel as solid cylinder. Override later if measured. |

Implementation: small builder in `mjcf.py` using `xml.etree.ElementTree`, ~80 lines. No Jinja, no MJCF preprocessor library.

## Controller Progression

Four milestones. Each is a complete working system; later milestones depend on earlier ones being validated.

### M1 — Nonlinear PD baseline

Port the existing non-linear PD controller onto the MuJoCo sim.

**Purpose:**
- Sanity-check the simulator (a controller known to work should still work).
- Provide an honest baseline to compare NMPC against.

**Success criteria:** stabilize from small initial tilt (e.g., 5°), hold indefinitely in deterministic sim.

### M2 — Stabilization NMPC

CasADi-based NMPC, single stage.

- **State:** `x_mpc = [θ, θ̇, ω_wheel]` (3D)
- **Control:** `u = τ_wheel`, hard-bounded to `±motor_max_torque_nm`
- **Dynamics:** full non-linear `sin θ` cubli equations, written symbolically in CasADi
- **Horizon:** ~50 steps × `dt_control` (10 ms) = 0.5 s (initial guess; tune later). MPC step matches control period so the first prediction step is what gets executed.
- **Cost:** `Q · (θ² + θ̇² + ω²) + R · u²`, plus terminal cost from LQR around upright
- **Solver:** IPOPT initially (easy to debug); `acados` deferred
- **Integrator:** RK4 inside CasADi

The key win over M1: **wheel-speed regulation collapses into the cost.** No outer loop adjusting the balance point — `q_ω` on wheel speed naturally biases the controller to bleed wheel speed when safe.

**Success criteria:**
- Stabilize from larger tilts than M1 (e.g., 20°+).
- Wheel speed remains bounded over a multi-hour deterministic sim run without a cascade hierarchy.
- Beats M1 on a defined disturbance-rejection benchmark.

### M3 — Swing trajectory tracking

The headline feature. Two-phase, Cubli-inspired.

**Phase 1 — Offline trajectory design (`swing_planner.py`):**
Solve a full non-linear trajectory optimization in CasADi+IPOPT:
- Fixed swing extrema (endpoints)
- Free intermediate states
- Subject to dynamics + torque constraints
- Minimize control effort (or jerk)
- Output: time-indexed reference `(x_ref(t), u_ref(t))` for one period of the swing.

**Phase 2 — Online NMPC tracking:**
Same NMPC as M2, but cost becomes:
```
(x - x_ref(t))ᵀ Q (x - x_ref(t)) + (u - u_ref(t))ᵀ R (u - u_ref(t))
```
Feedforward `u_ref` does the heavy lifting; the MPC handles disturbances.

For continuous back-and-forth: design *one period* offline and loop the reference. Avoids any real-time trajectory generation.

**Success criteria:** cube swings back and forth between defined extrema, indefinitely, in deterministic sim.

### M4 — Long-duration robustness

Test infrastructure, not new control. The things that will kill days-on-end balancing:

- **State estimator drift** — implement complementary filter or simple EKF in `estimation/filter.py`. Validate with sensor noise + bias enabled.
- **Wheel speed runaway under sustained disturbance** — sim a constant external torque (e.g., tilted floor) and confirm NMPC doesn't blow up.
- **Solver failures** — define a fallback when IPOPT misses a deadline (last good solution, or PD safety controller). Add a watchdog.

This is where `SimConfig.sensor_noise` and `sensor_delay_ms` get turned on.

**Success criteria:** 24-hour simulated run with noisy sensors, periodic disturbances, no controller failures.

## Co-Design Workflow

The parametric sim exists to drive hardware decisions. Three concrete studies live in `scripts/`:

1. **Motor torque sweep** — minimum `motor_max_torque_nm` to recover from a 30° initial tilt. Output: recovery angle vs. torque curve.
2. **Wheel inertia ratio sweep** — max swing angle achievable before wheel speed saturates, vs. wheel inertia. Output: feasibility frontier.
3. **Control loop rate sweep** — stabilization quality vs. `dt_control`. Output: directly informs MCU vs. SBC choice for the next hardware rev.

Each script is a standalone CLI that emits CSV + plots. Not part of the runtime loop.

## Testing Strategy

- **Unit tests** for `mjcf.py` (XML parses, inertia math correct), `dynamics.py` (CasADi dynamics match a hand-computed step for a known state), and `config.py` (derived quantities).
- **Integration tests** for each controller (M1–M3): deterministic sim, defined initial condition, assert success criteria within a time budget.
- **Smoke test** for the full runner: load default config, run 1 simulated second, no crashes.

## Implementation Order

1. Scaffold the repo (`pyproject.toml`, `uv` env, package skeleton, license, README).
2. `config.py` + `mjcf.py` + tests — get a viewer up showing a parametric cube.
3. `sim/env.py` + runner skeleton — passive cube falling over.
4. M1: port nonlinear PD controller.
5. `dynamics.py` (CasADi) — validate against MuJoCo on a few open-loop trajectories.
6. M2: stabilization NMPC.
7. M3: swing planner + tracking NMPC.
8. M4: estimator, noise, robustness tests.
9. Co-design study scripts.

## Open Questions

These are deferred, not unresolved — pick when we get to them:

- Exact starting hardware values for `configs/default.yaml`. Plan: plausible guesses based on a ~10 cm, ~0.4 kg cube; swap to measured values from the existing hardware if available.
- IPOPT vs. acados crossover point: stay on IPOPT until M3 works; profile then.
- Filter choice (complementary vs. EKF): defer until M4. EKF if we have good noise models; complementary if not.

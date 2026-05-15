# MPC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement M2 (balance NMPC) and M3 (metronome trajectory tracking) for cubli-mpc, plus the `cubli-mpc eval` harness that produces side-by-side metrics for PD / random / policy / NMPC.

**Architecture:** CasADi symbolic dynamics + RK4 prediction model. Direct multiple shooting NLP solved by IPOPT every control tick. Warm-started from previous solution, with fallback to nonlinear-PD on solver failure. Tracking uses an offline trajopt that exports a one-period reference; online NMPC interpolates and tracks with feedforward. All controllers share a `Reference` protocol so balance and tracking share one code path.

**Tech Stack:** Python ≥ 3.11, CasADi (new dep, IPOPT solver), NumPy, MuJoCo (existing), pytest with uv. Follow existing patterns in `src/cubli_mpc/control/nonlinear_pd.py` for dataclass configs and `src/cubli_mpc/sim/disturbance.py` for module style.

**Parent spec:** `docs/superpowers/specs/2026-05-15-mpc-implementation-design.md`.

**Note on spec deviation:** The spec says "direct collocation" for the swing planner; this plan uses direct **multiple shooting** instead (reuses the same RK4 step as the NMPC, simpler code, equivalent solution quality for a half-period horizon). Same NLP class either way.

---

## File structure

**New files:**
- `src/cubli_mpc/model/dynamics.py` — CasADi continuous dynamics + RK4 step.
- `src/cubli_mpc/control/references.py` — `Reference` protocol, `ConstantReference`, `PeriodicTrajectoryReference`.
- `src/cubli_mpc/control/nmpc.py` — `NMPCController`, `NMPCConfig`.
- `src/cubli_mpc/control/swing_planner.py` — Offline half-period trajopt + save/load.
- `src/cubli_mpc/eval/__init__.py`
- `src/cubli_mpc/eval/scenarios.py` — Scenario registry + metrics dataclass.
- `src/cubli_mpc/eval/runner.py` — `run_scenario(controller, scenario, hw, sim)`.
- `tests/test_dynamics.py`
- `tests/test_references.py`
- `tests/test_nmpc.py`
- `tests/test_swing_planner.py`
- `tests/test_eval.py`

**Modified:**
- `pyproject.toml` — add `casadi>=3.6` to runtime deps.
- `src/cubli_mpc/cli.py` — add `nmpc` to `--controller` choices, add `plan-swing` and `eval` subcommands, add `--nmpc-*` and `--swing-traj` flags.

---

## Test-hardware helper

Tests use a shared HardwareConfig fixture. Define this inline in each test file (don't add to a shared conftest — these tests are independent and the function is short):

```python
from cubli_mpc.config import HardwareConfig, SimConfig

def _make_hw() -> HardwareConfig:
    return HardwareConfig(
        cube_side_length_m=0.10,
        cube_mass_kg=0.40,
        wheel_mass_kg=0.10,
        wheel_radius_m=0.035,
        wheel_thickness_m=0.010,
        motor_max_torque_nm=0.20,
        motor_max_speed_rad_s=600.0,
        motor_torque_constant=0.01,
        edge_bearing_damping=1e-4,
        wheel_bearing_damping=1e-5,
    )
```

---

## Task 1: Add casadi dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add casadi to runtime deps**

Edit `pyproject.toml`. Add `"casadi>=3.6"` to the `dependencies` array (alongside `mujoco`, not in an extra).

- [ ] **Step 2: Sync env**

Run: `uv sync --extra dev --extra rl`
Expected: casadi installed, no errors.

- [ ] **Step 3: Verify import**

Run: `uv run python -c "import casadi as ca; print(ca.__version__)"`
Expected: prints a 3.x version string.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add casadi for MPC prediction model"
```

---

## Task 2: Continuous symbolic dynamics

**Files:**
- Create: `src/cubli_mpc/model/dynamics.py`
- Test: `tests/test_dynamics.py`

The dynamics from the spec (after inverting the 2×2 mass matrix):

```
θ̈    = (m·g·L·sin θ − b_e·θ̇ − τ + b_w·ω_w) / I_b
ω̇_w  = (τ − b_w·ω_w) / I_w  −  θ̈
```

with state `x = [θ, θ̇, ω_w]` and input `u = τ`.

- [ ] **Step 1: Write failing tests**

Create `tests/test_dynamics.py`:

```python
"""Tests for the CasADi symbolic dynamics."""
import math

import numpy as np
import pytest

from cubli_mpc.config import HardwareConfig
from cubli_mpc.model.dynamics import make_continuous_dynamics


def _make_hw() -> HardwareConfig:
    return HardwareConfig(
        cube_side_length_m=0.10,
        cube_mass_kg=0.40,
        wheel_mass_kg=0.10,
        wheel_radius_m=0.035,
        wheel_thickness_m=0.010,
        motor_max_torque_nm=0.20,
        motor_max_speed_rad_s=600.0,
        motor_torque_constant=0.0,
        edge_bearing_damping=0.0,
        wheel_bearing_damping=0.0,
    )


def test_continuous_dynamics_signature():
    f = make_continuous_dynamics(_make_hw())
    x = np.array([0.0, 0.0, 0.0])
    u = np.array([0.0])
    x_dot = np.array(f(x, u)).flatten()
    assert x_dot.shape == (3,)


def test_dynamics_upright_zero_input_is_equilibrium():
    """θ=0, θ̇=0, ω_w=0, τ=0 should give all-zero x_dot."""
    f = make_continuous_dynamics(_make_hw())
    x_dot = np.array(f([0.0, 0.0, 0.0], [0.0])).flatten()
    assert np.allclose(x_dot, 0.0, atol=1e-12)


def test_dynamics_positive_tilt_falls_outward():
    """θ > 0 with no torque should give θ̈ > 0 (cube falls outward)."""
    f = make_continuous_dynamics(_make_hw())
    x_dot = np.array(f([0.1, 0.0, 0.0], [0.0])).flatten()
    assert x_dot[1] > 0.0  # theta_ddot > 0


def test_dynamics_positive_torque_spins_wheel_positive():
    """At rest upright, positive τ should give positive ω̇_w."""
    f = make_continuous_dynamics(_make_hw())
    x_dot = np.array(f([0.0, 0.0, 0.0], [0.05])).flatten()
    assert x_dot[2] > 0.0  # omega_w_dot > 0


def test_dynamics_positive_torque_pushes_body_negative():
    """Reaction: positive τ on the wheel pushes the body in the negative
    direction (θ̈ < 0). This is the whole point of a reaction wheel."""
    f = make_continuous_dynamics(_make_hw())
    x_dot = np.array(f([0.0, 0.0, 0.0], [0.05])).flatten()
    assert x_dot[1] < 0.0  # theta_ddot < 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_dynamics.py -v`
Expected: ImportError on `cubli_mpc.model.dynamics` (module doesn't exist yet).

- [ ] **Step 3: Implement `make_continuous_dynamics`**

Create `src/cubli_mpc/model/dynamics.py`:

```python
"""CasADi symbolic dynamics for the 1-axis cubli (MPC prediction model).

State (3D, drops the cyclic wheel angle):
    x = [theta, theta_dot, omega_wheel]

Input:
    u = [tau]   (wheel motor torque, Nm)

The continuous-time equations come from the Lagrangian of a cube hinged on
one edge with a flywheel on a parallel internal axis. After inverting the
2x2 mass matrix:

    theta_ddot   = (m*g*L*sin(theta) - b_e*theta_dot - tau + b_w*omega_w) / I_b
    omega_w_dot  = (tau - b_w*omega_w) / I_w  -  theta_ddot

This is deliberately *separate* from MuJoCo; small model mismatch is part
of the design (see parent spec).
"""
from __future__ import annotations

import casadi as ca

from cubli_mpc.config import HardwareConfig


def make_continuous_dynamics(hw: HardwareConfig) -> ca.Function:
    """Return CasADi Function f(x, u) -> x_dot for the 3D MPC state."""
    I_b = hw.cube_inertia_about_edge
    I_w = hw.wheel_inertia_about_spin_axis
    mgL = hw.gravity_moment_coefficient
    b_e = hw.edge_bearing_damping
    b_w = hw.wheel_bearing_damping

    x = ca.MX.sym("x", 3)
    u = ca.MX.sym("u", 1)
    theta = x[0]
    theta_dot = x[1]
    omega_w = x[2]
    tau = u[0]

    theta_ddot = (mgL * ca.sin(theta) - b_e * theta_dot
                  - tau + b_w * omega_w) / I_b
    omega_w_dot = (tau - b_w * omega_w) / I_w - theta_ddot

    x_dot = ca.vertcat(theta_dot, theta_ddot, omega_w_dot)
    return ca.Function("cubli_dynamics", [x, u], [x_dot],
                       ["x", "u"], ["x_dot"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dynamics.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cubli_mpc/model/dynamics.py tests/test_dynamics.py
git commit -m "feat(model): CasADi symbolic dynamics for MPC prediction"
```

---

## Task 3: RK4 integrator

**Files:**
- Modify: `src/cubli_mpc/model/dynamics.py`
- Modify: `tests/test_dynamics.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_dynamics.py`:

```python
from cubli_mpc.model.dynamics import make_rk4_step


def test_rk4_returns_correct_shape():
    F = make_rk4_step(_make_hw(), dt=0.01)
    x = np.array([0.0, 0.0, 0.0])
    u = np.array([0.0])
    x_next = np.array(F(x, u)).flatten()
    assert x_next.shape == (3,)


def test_rk4_equilibrium_stays_at_equilibrium():
    F = make_rk4_step(_make_hw(), dt=0.01)
    x = np.array([0.0, 0.0, 0.0])
    for _ in range(100):
        x = np.array(F(x, [0.0])).flatten()
    assert np.allclose(x, 0.0, atol=1e-10)


def test_rk4_matches_fine_euler_for_small_dt():
    """RK4 with dt=1ms should be very close to Euler with dt=1us over 10ms."""
    hw = _make_hw()
    f = make_continuous_dynamics(hw)
    F_rk4 = make_rk4_step(hw, dt=0.001)

    x0 = np.array([0.05, 0.0, 0.0])
    u = np.array([0.0])

    # RK4: 10 steps at 1ms
    x_rk4 = x0.copy()
    for _ in range(10):
        x_rk4 = np.array(F_rk4(x_rk4, u)).flatten()

    # Euler reference: 10000 steps at 1us
    x_euler = x0.copy()
    dt_fine = 1e-6
    for _ in range(10_000):
        x_dot = np.array(f(x_euler, u)).flatten()
        x_euler = x_euler + dt_fine * x_dot

    assert np.allclose(x_rk4, x_euler, atol=1e-5)


def test_rk4_positive_torque_decelerates_body_increases_wheel():
    F = make_rk4_step(_make_hw(), dt=0.01)
    x = np.array([0.0, 0.0, 0.0])
    for _ in range(20):
        x = np.array(F(x, [0.05])).flatten()
    # After 0.2 s of constant torque from rest upright:
    assert x[1] < 0.0   # body picked up negative angular velocity
    assert x[2] > 0.0   # wheel speed positive
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_dynamics.py::test_rk4_returns_correct_shape -v`
Expected: ImportError on `make_rk4_step`.

- [ ] **Step 3: Implement `make_rk4_step`**

Append to `src/cubli_mpc/model/dynamics.py`:

```python
def make_rk4_step(hw: HardwareConfig, dt: float) -> ca.Function:
    """Return CasADi Function F(x, u) -> x_next using explicit RK4.

    `dt` is baked into the returned Function. For multi-rate use, build
    one Function per dt.
    """
    f = make_continuous_dynamics(hw)
    x = ca.MX.sym("x", 3)
    u = ca.MX.sym("u", 1)

    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return ca.Function("cubli_rk4", [x, u], [x_next],
                       ["x", "u"], ["x_next"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dynamics.py -v`
Expected: 9 passed (5 from Task 2 + 4 new).

- [ ] **Step 5: Commit**

```bash
git add src/cubli_mpc/model/dynamics.py tests/test_dynamics.py
git commit -m "feat(model): RK4 integrator for symbolic dynamics"
```

---

## Task 4: Dynamics ↔ MuJoCo cross-validation

**Files:**
- Modify: `tests/test_dynamics.py`

Demonstrates the two models agree to a tolerance over the MPC horizon. Small mismatch is expected and accepted (parent spec); large mismatch indicates a derivation bug.

- [ ] **Step 1: Add the cross-validation test**

Append to `tests/test_dynamics.py`:

```python
from cubli_mpc.config import SimConfig
from cubli_mpc.sim.env import CubliEnv


@pytest.mark.parametrize("theta0,tau", [
    (0.05, 0.0),       # passive fall from small tilt
    (0.0,  0.02),      # constant torque from upright
    (0.10, -0.03),     # restoring torque from larger tilt
])
def test_dynamics_matches_mujoco_open_loop(theta0, tau):
    """Integrate the CasADi RK4 dynamics and the MuJoCo plant from the
    same initial state with the same constant input. Trajectories should
    agree to within ~1 deg on theta and ~5% on omega_w over 0.5 s.
    """
    hw = _make_hw()
    # Increase damping a bit to make trajectories not diverge dramatically.
    sim = SimConfig(dt_sim=0.001, dt_control=0.01)
    F = make_rk4_step(hw, dt=sim.dt_control)

    # MuJoCo rollout
    env = CubliEnv(hw, sim)
    env.reset(theta0=theta0)
    steps_per_control = max(1, round(sim.dt_control / sim.dt_sim))
    n_control_ticks = 50  # 0.5 s
    x_mj = []
    for _ in range(n_control_ticks):
        env.apply_torque(tau)
        for _ in range(steps_per_control):
            env.step()
        s = env.state()
        x_mj.append([s[0], s[1], s[3]])  # drop cyclic wheel angle
    x_mj = np.array(x_mj)

    # CasADi rollout (same control rate)
    x_ca = np.array([theta0, 0.0, 0.0])
    x_ca_traj = []
    for _ in range(n_control_ticks):
        x_ca = np.array(F(x_ca, [tau])).flatten()
        x_ca_traj.append(x_ca.copy())
    x_ca_traj = np.array(x_ca_traj)

    # Compare end-of-horizon states
    theta_err = abs(x_mj[-1, 0] - x_ca_traj[-1, 0])
    omega_err = abs(x_mj[-1, 2] - x_ca_traj[-1, 2])
    # Generous tolerances because we intentionally have model mismatch
    # (wheel mass absorbed by MuJoCo, not the CasADi model).
    assert theta_err < math.radians(2.0), (
        f"theta mismatch: {math.degrees(theta_err):.3f} deg"
    )
    assert omega_err < max(0.5, 0.1 * abs(x_ca_traj[-1, 2])), (
        f"omega_w mismatch: {omega_err:.3f} rad/s"
    )
```

- [ ] **Step 2: Run the cross-validation test**

Run: `uv run pytest tests/test_dynamics.py -v`
Expected: 12 passed total (9 previous + 3 parametrized).

If the cross-validation fails, the dynamics derivation is wrong — fix `make_continuous_dynamics` before continuing. Common mistakes: sign on `tau` in one of the equations, missing cross-coupling term.

- [ ] **Step 3: Commit**

```bash
git add tests/test_dynamics.py
git commit -m "test(model): cross-validate CasADi dynamics against MuJoCo"
```

---

## Task 5: Reference protocol + ConstantReference

**Files:**
- Create: `src/cubli_mpc/control/references.py`
- Test: `tests/test_references.py`

The Reference protocol unifies M2 (constant balance target) and M3 (time-varying swing trajectory) under one NMPC code path.

- [ ] **Step 1: Write failing tests**

Create `tests/test_references.py`:

```python
"""Tests for the Reference protocol implementations."""
import math

import numpy as np
import pytest

from cubli_mpc.control.references import ConstantReference


def test_constant_reference_shape():
    ref = ConstantReference()
    x_ref, u_ref = ref.at(t=0.0, dt=0.01, horizon=10)
    assert x_ref.shape == (11, 3)
    assert u_ref.shape == (10, 1)


def test_constant_reference_default_is_upright():
    ref = ConstantReference()
    x_ref, u_ref = ref.at(t=1.23, dt=0.01, horizon=5)
    assert np.allclose(x_ref, 0.0)
    assert np.allclose(u_ref, 0.0)


def test_constant_reference_with_target_theta():
    ref = ConstantReference(target_theta=math.radians(10.0))
    x_ref, u_ref = ref.at(t=0.0, dt=0.01, horizon=3)
    # All theta entries equal the target; theta_dot and omega_w stay zero.
    assert np.allclose(x_ref[:, 0], math.radians(10.0))
    assert np.allclose(x_ref[:, 1], 0.0)
    assert np.allclose(x_ref[:, 2], 0.0)
    assert np.allclose(u_ref, 0.0)


def test_constant_reference_independent_of_time():
    ref = ConstantReference(target_theta=0.05)
    a_x, a_u = ref.at(t=0.0, dt=0.01, horizon=4)
    b_x, b_u = ref.at(t=99.0, dt=0.01, horizon=4)
    assert np.allclose(a_x, b_x)
    assert np.allclose(a_u, b_u)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_references.py -v`
Expected: ImportError on `cubli_mpc.control.references`.

- [ ] **Step 3: Implement Reference and ConstantReference**

Create `src/cubli_mpc/control/references.py`:

```python
"""Reference trajectories for the NMPC.

A `Reference` is queried by NMPC each tick for the desired (x, u) over
its prediction horizon. Two implementations live here:

- `ConstantReference(target_theta)`: balance task (default upright;
  non-zero target_theta acts like a balance-with-offset setpoint).
- `PeriodicTrajectoryReference(...)`: M3 swing tracking. Loads a
  one-period reference from disk and loops it.

Both return numpy arrays sized to (horizon+1, 3) for x_ref and
(horizon, 1) for u_ref so the NMPC can consume them uniformly.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Reference(Protocol):
    def at(
        self, t: float, dt: float, horizon: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (x_ref [H+1, 3], u_ref [H, 1]) sampled at t + k*dt."""
        ...


class ConstantReference:
    """Constant reference at (target_theta, 0, 0), zero feedforward torque."""

    def __init__(self, target_theta: float = 0.0):
        self._target_theta = float(target_theta)

    @property
    def target_theta(self) -> float:
        return self._target_theta

    def at(
        self, t: float, dt: float, horizon: int
    ) -> tuple[np.ndarray, np.ndarray]:
        x_ref = np.zeros((horizon + 1, 3), dtype=np.float64)
        x_ref[:, 0] = self._target_theta
        u_ref = np.zeros((horizon, 1), dtype=np.float64)
        return x_ref, u_ref
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_references.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cubli_mpc/control/references.py tests/test_references.py
git commit -m "feat(control): Reference protocol + ConstantReference"
```

---

## Task 6: NMPCConfig + basic NMPCController

**Files:**
- Create: `src/cubli_mpc/control/nmpc.py`
- Test: `tests/test_nmpc.py`

Direct multiple shooting NLP with IPOPT. No warm-start, no fallback yet — those land in Tasks 9 and 10.

- [ ] **Step 1: Write failing tests**

Create `tests/test_nmpc.py`:

```python
"""Tests for the NMPC controller (M2 balance, no warm-start, no fallback)."""
import math

import numpy as np
import pytest

from cubli_mpc.config import HardwareConfig
from cubli_mpc.control.nmpc import NMPCConfig, NMPCController
from cubli_mpc.control.references import ConstantReference


def _make_hw() -> HardwareConfig:
    return HardwareConfig(
        cube_side_length_m=0.10,
        cube_mass_kg=0.40,
        wheel_mass_kg=0.10,
        wheel_radius_m=0.035,
        wheel_thickness_m=0.010,
        motor_max_torque_nm=0.20,
        motor_max_speed_rad_s=600.0,
        motor_torque_constant=0.01,
        edge_bearing_damping=1e-4,
        wheel_bearing_damping=1e-5,
    )


def test_nmpc_returns_scalar_torque():
    nmpc = NMPCController(_make_hw(), NMPCConfig(horizon_steps=10))
    x_hat = np.array([0.05, 0.0, 0.0, 0.0])  # 4D state from runner
    tau = nmpc.step(x_hat, t=0.0)
    assert isinstance(tau, float)


def test_nmpc_respects_torque_limit():
    hw = _make_hw()
    nmpc = NMPCController(hw, NMPCConfig(horizon_steps=10))
    # Large initial tilt should produce a torque at or near the limit.
    x_hat = np.array([math.radians(45.0), 0.0, 0.0, 0.0])
    tau = nmpc.step(x_hat, t=0.0)
    assert abs(tau) <= hw.motor_max_torque_nm + 1e-9


def test_nmpc_positive_tilt_produces_positive_torque():
    """For an unstable equilibrium with positive tilt and zero wheel speed,
    the NMPC should command positive torque (which decelerates the body's
    fall via the reaction)."""
    nmpc = NMPCController(_make_hw(), NMPCConfig(horizon_steps=20))
    x_hat = np.array([math.radians(5.0), 0.0, 0.0, 0.0])
    tau = nmpc.step(x_hat, t=0.0)
    assert tau > 0.0


def test_nmpc_uses_supplied_reference():
    """With ConstantReference(target_theta=0.05), the NMPC's commanded
    torque at theta=0.05 should be smaller in magnitude than at theta=0
    (because the latter is now an *error*)."""
    nmpc = NMPCController(
        _make_hw(),
        NMPCConfig(horizon_steps=20),
        reference=ConstantReference(target_theta=0.05),
    )
    tau_at_target = abs(nmpc.step(np.array([0.05, 0.0, 0.0, 0.0]), t=0.0))
    tau_at_upright = abs(nmpc.step(np.array([0.0, 0.0, 0.0, 0.0]), t=0.0))
    assert tau_at_target < tau_at_upright


def test_nmpc_target_theta_property():
    nmpc = NMPCController(
        _make_hw(), NMPCConfig(horizon_steps=10),
        reference=ConstantReference(target_theta=0.1),
    )
    assert nmpc.target_theta == pytest.approx(0.1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_nmpc.py -v`
Expected: ImportError on `cubli_mpc.control.nmpc`.

- [ ] **Step 3: Implement NMPCConfig and NMPCController**

Create `src/cubli_mpc/control/nmpc.py`:

```python
"""Nonlinear MPC controller for the 1-axis cubli.

Direct multiple-shooting NLP solved by IPOPT each tick. State `[theta,
theta_dot, omega_w]` (3D); the runner passes a 4D x_hat with the cyclic
wheel angle, which we drop. Action: scalar torque on the wheel.

This is the M2 / "stabilization NMPC" milestone from the parent spec.
Warm-start and solver-failure fallback land in subsequent tasks.
"""
from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np

from cubli_mpc.config import HardwareConfig
from cubli_mpc.control.references import ConstantReference, Reference
from cubli_mpc.model.dynamics import make_rk4_step


@dataclass(frozen=True)
class NMPCConfig:
    horizon_steps: int = 50
    dt: float = 0.010
    q_theta: float = 50.0
    q_theta_dot: float = 1.0
    q_omega_w: float = 1e-4
    r_torque: float = 0.01
    terminal_scale: float = 10.0
    omega_w_limit: float = float("inf")
    ipopt_max_iter: int = 50
    ipopt_print_level: int = 0


class NMPCController:
    def __init__(
        self,
        hw: HardwareConfig,
        cfg: NMPCConfig,
        reference: Reference | None = None,
    ):
        self._hw = hw
        self._cfg = cfg
        self._reference = reference if reference is not None else ConstantReference()
        self._tau_max = float(hw.motor_max_torque_nm)
        self._build_solver()

    @property
    def target_theta(self) -> float:
        return float(getattr(self._reference, "target_theta", 0.0))

    @property
    def reference(self) -> Reference:
        return self._reference

    def _build_solver(self) -> None:
        cfg = self._cfg
        N = cfg.horizon_steps
        F = make_rk4_step(self._hw, dt=cfg.dt)

        # Decision variables: X (3 x N+1), U (1 x N)
        X = ca.MX.sym("X", 3, N + 1)
        U = ca.MX.sym("U", 1, N)

        # Parameters: x0 (3), x_ref (3 x N+1), u_ref (1 x N)
        x0 = ca.MX.sym("x0", 3)
        x_ref = ca.MX.sym("x_ref", 3, N + 1)
        u_ref = ca.MX.sym("u_ref", 1, N)

        Q = ca.diag(ca.DM([cfg.q_theta, cfg.q_theta_dot, cfg.q_omega_w]))
        R = ca.DM(cfg.r_torque)

        g_list = [X[:, 0] - x0]  # initial-state constraint
        cost = ca.MX(0.0)
        for k in range(N):
            dx = X[:, k] - x_ref[:, k]
            du = U[:, k] - u_ref[:, k]
            cost = cost + ca.mtimes([dx.T, Q, dx]) + R * du * du
            g_list.append(X[:, k + 1] - F(X[:, k], U[:, k]))
        dx_N = X[:, N] - x_ref[:, N]
        cost = cost + cfg.terminal_scale * ca.mtimes([dx_N.T, Q, dx_N])

        g = ca.vertcat(*g_list)
        # All defect constraints == 0 (initial-state + N dynamics).
        self._g_lb = np.zeros(g.size1())
        self._g_ub = np.zeros(g.size1())

        # Decision-vector layout: flatten X then U (column-major to match
        # CasADi's MX reshape conventions).
        z = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        p = ca.vertcat(x0, ca.reshape(x_ref, -1, 1),
                       ca.reshape(u_ref, -1, 1))

        nlp = {"x": z, "p": p, "f": cost, "g": g}
        opts = {
            "ipopt.max_iter": cfg.ipopt_max_iter,
            "ipopt.print_level": cfg.ipopt_print_level,
            "ipopt.sb": "yes",
            "print_time": 0,
        }
        self._solver = ca.nlpsol("nmpc", "ipopt", nlp, opts)

        # Box bounds on z: X unbounded except wheel speed (rows of X),
        # U bounded by ±tau_max.
        z_lb = np.full(z.size1(), -np.inf)
        z_ub = np.full(z.size1(), +np.inf)
        # X layout: 3*(N+1) entries first, in column-major: [theta_0,
        # theta_dot_0, omega_w_0, theta_1, theta_dot_1, omega_w_1, ...].
        # omega_w lives at offsets 2, 5, 8, ... within the X block.
        if np.isfinite(cfg.omega_w_limit):
            for k in range(N + 1):
                idx = 3 * k + 2
                z_lb[idx] = -cfg.omega_w_limit
                z_ub[idx] = +cfg.omega_w_limit
        # U starts at 3*(N+1).
        u_start = 3 * (N + 1)
        z_lb[u_start:] = -self._tau_max
        z_ub[u_start:] = +self._tau_max
        self._z_lb = z_lb
        self._z_ub = z_ub
        self._n_x_vars = 3 * (N + 1)
        self._N = N

    def step(self, x_hat: np.ndarray, t: float) -> float:
        """Solve the NLP once and return the first commanded torque."""
        # Drop cyclic wheel angle from the 4D runner state.
        x0 = np.array([x_hat[0], x_hat[1], x_hat[3]], dtype=np.float64)
        x_ref, u_ref = self._reference.at(t, self._cfg.dt, self._N)
        # Initial guess: hold x0, zero torque.
        x_init = np.tile(x0, self._N + 1)
        u_init = np.zeros(self._N)
        z0 = np.concatenate([x_init, u_init])
        # NOTE on flatten order: CasADi MX of shape (3, N+1) flattens
        # column-major to [state_at_t0 (3), state_at_t1 (3), ...]. The
        # numpy x_ref has shape (N+1, 3) so we want row-major flatten,
        # which is numpy's default (`order="C"`).
        p = np.concatenate([
            x0,
            x_ref.reshape(-1),
            u_ref.reshape(-1),
        ])
        sol = self._solver(x0=z0, p=p,
                           lbx=self._z_lb, ubx=self._z_ub,
                           lbg=self._g_lb, ubg=self._g_ub)
        z_opt = np.array(sol["x"]).flatten()
        u0 = z_opt[self._n_x_vars]  # first U entry
        return float(u0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_nmpc.py -v`
Expected: 5 passed. May take a few seconds for the first solver compile.

If `test_nmpc_positive_tilt_produces_positive_torque` fails (torque sign wrong), the dynamics sign is the most likely culprit — check Task 2's tests pass first.

- [ ] **Step 5: Commit**

```bash
git add src/cubli_mpc/control/nmpc.py tests/test_nmpc.py
git commit -m "feat(control): NMPCController with IPOPT direct multiple shooting"
```

---

## Task 7: NMPC balance recovery integration test

**Files:**
- Modify: `tests/test_nmpc.py`

End-to-end: NMPC in a real MuJoCo loop should recover from a 15° tilt.

- [ ] **Step 1: Add the integration test**

Append to `tests/test_nmpc.py`:

```python
from cubli_mpc.config import SimConfig
from cubli_mpc.runner import Runner
from cubli_mpc.sim.env import CubliEnv


def test_nmpc_recovers_from_15deg_tilt():
    hw = _make_hw()
    sim = SimConfig(dt_sim=0.001, dt_control=0.01)
    env = CubliEnv(hw, sim)
    env.reset(theta0=math.radians(15.0))

    nmpc = NMPCController(hw, NMPCConfig(horizon_steps=40, dt=sim.dt_control))
    runner = Runner(env, nmpc, sim)
    log = runner.run(duration_s=5.0)

    # Final state should be near upright.
    theta_final = float(log["theta"][-1])
    assert abs(theta_final) < math.radians(2.0), (
        f"final theta = {math.degrees(theta_final):.2f} deg"
    )
    # Should not have fallen at any point.
    theta_max = float(np.max(np.abs(log["theta"])))
    assert theta_max < math.radians(20.0), (
        f"max theta = {math.degrees(theta_max):.2f} deg"
    )
```

- [ ] **Step 2: Run the integration test**

Run: `uv run pytest tests/test_nmpc.py::test_nmpc_recovers_from_15deg_tilt -v`
Expected: PASS. Takes ~10-30 s (500 IPOPT solves without warm-start).

If it fails by timing out, lower `horizon_steps` to 25 first. If it fails by *not balancing*, tune `q_theta` upward or `r_torque` downward before debugging code.

- [ ] **Step 3: Commit**

```bash
git add tests/test_nmpc.py
git commit -m "test(nmpc): balance recovery from 15deg integration test"
```

---

## Task 8: Wire `--controller nmpc` into sim CLI

**Files:**
- Modify: `src/cubli_mpc/cli.py`

- [ ] **Step 1: Add `nmpc` to controller choices**

In `src/cubli_mpc/cli.py`, find the `--controller` argparse line:

```python
p_sim.add_argument("--controller",
                   choices=("pd", "random", "policy"), default="pd",
                   help="Which controller to run (default: pd)")
```

Replace with:

```python
p_sim.add_argument("--controller",
                   choices=("pd", "random", "policy", "nmpc"), default="pd",
                   help="Which controller to run (default: pd)")
```

- [ ] **Step 2: Add NMPC-specific flags**

Just before `p_sim.set_defaults(func=_cmd_sim)`, add:

```python
p_sim.add_argument("--nmpc-horizon", type=int, default=50,
                   help="(NMPC) prediction horizon in steps")
p_sim.add_argument("--nmpc-dt", type=float, default=0.010,
                   help="(NMPC) prediction step (s); matches dt_control "
                        "by default")
p_sim.add_argument("--nmpc-target-tilt-deg", type=float, default=0.0,
                   help="(NMPC) balance setpoint (degrees; 0 = upright). "
                        "Ignored if --swing-traj is given.")
```

- [ ] **Step 3: Handle `nmpc` in `_build_controller`**

In `_build_controller`, add a branch before the PD fallback:

```python
    if args.controller == "nmpc":
        from cubli_mpc.control.nmpc import NMPCConfig, NMPCController
        from cubli_mpc.control.references import ConstantReference
        cfg = NMPCConfig(horizon_steps=args.nmpc_horizon, dt=args.nmpc_dt)
        ref = ConstantReference(
            target_theta=math.radians(args.nmpc_target_tilt_deg),
        )
        return NMPCController(hw, cfg, reference=ref)
```

- [ ] **Step 4: Smoke test (headless)**

Run:
```bash
uv run cubli-mpc sim --config configs/default.yaml --controller nmpc \
    --duration 3 --initial-tilt-deg 5 --plot /tmp/nmpc-smoke.png
```

Expected: command completes, `wrote /tmp/nmpc-smoke.png`, final `|theta|` near 0.

- [ ] **Step 5: Commit**

```bash
git add src/cubli_mpc/cli.py
git commit -m "feat(cli): --controller nmpc with horizon/dt/target flags"
```

---

## Task 9: Warm-start in NMPC

**Files:**
- Modify: `src/cubli_mpc/control/nmpc.py`
- Modify: `tests/test_nmpc.py`

Reuses the previous solution shifted one tick. Critical for IPOPT to converge inside `dt_control`.

- [ ] **Step 1: Add the warm-start test**

Append to `tests/test_nmpc.py`:

```python
import time


def test_nmpc_warm_start_speeds_up_subsequent_solves():
    """Second solve from a similar state should be substantially faster
    than the first (warm-started)."""
    nmpc = NMPCController(_make_hw(), NMPCConfig(horizon_steps=40))
    x_hat = np.array([math.radians(5.0), 0.0, 0.0, 0.0])

    t0 = time.perf_counter()
    nmpc.step(x_hat, t=0.0)
    cold_s = time.perf_counter() - t0

    # Same problem, second time — should hit warm-start.
    t0 = time.perf_counter()
    for _ in range(5):
        nmpc.step(x_hat, t=0.0)
    warm_avg_s = (time.perf_counter() - t0) / 5

    # Allow a generous margin; warm should be at least half cold.
    assert warm_avg_s < cold_s * 0.6, (
        f"cold={cold_s*1000:.1f} ms, warm avg={warm_avg_s*1000:.1f} ms"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_nmpc.py::test_nmpc_warm_start_speeds_up_subsequent_solves -v`
Expected: FAIL (without warm-start, both solves are similarly slow).

- [ ] **Step 3: Add warm-start to NMPCController**

In `src/cubli_mpc/control/nmpc.py`, edit `__init__` to add:

```python
        self._z_prev: np.ndarray | None = None
```

(Add it just after `self._build_solver()`.)

Then edit `step()` to use and update `_z_prev`. Replace the body of `step()` with:

```python
    def step(self, x_hat: np.ndarray, t: float) -> float:
        x0 = np.array([x_hat[0], x_hat[1], x_hat[3]], dtype=np.float64)
        x_ref, u_ref = self._reference.at(t, self._cfg.dt, self._N)

        if self._z_prev is None:
            x_init = np.tile(x0, self._N + 1)
            u_init = np.zeros(self._N)
            z0 = np.concatenate([x_init, u_init])
        else:
            z0 = self._shift_warm_start(self._z_prev, x0)

        p = np.concatenate([
            x0,
            x_ref.reshape(-1),
            u_ref.reshape(-1),
        ])
        sol = self._solver(x0=z0, p=p,
                           lbx=self._z_lb, ubx=self._z_ub,
                           lbg=self._g_lb, ubg=self._g_ub)
        z_opt = np.array(sol["x"]).flatten()
        self._z_prev = z_opt
        u0 = z_opt[self._n_x_vars]
        return float(u0)

    def _shift_warm_start(self, z_prev: np.ndarray,
                          x0_new: np.ndarray) -> np.ndarray:
        """Shift the previous (X, U) trajectory one step forward, replace
        x_0 with the new measurement, and duplicate the last column."""
        N = self._N
        n_x = self._n_x_vars
        X_prev = z_prev[:n_x].reshape(3, N + 1, order="F")
        U_prev = z_prev[n_x:].reshape(1, N, order="F")

        X_new = np.empty_like(X_prev)
        X_new[:, 0] = x0_new
        X_new[:, 1:N] = X_prev[:, 2:N + 1]
        X_new[:, N] = X_prev[:, N]  # repeat terminal

        U_new = np.empty_like(U_prev)
        U_new[:, :N - 1] = U_prev[:, 1:N]
        U_new[:, N - 1] = U_prev[:, N - 1]  # repeat last

        return np.concatenate([
            X_new.reshape(-1, order="F"),
            U_new.reshape(-1, order="F"),
        ])
```

Also expose a `reset()` method for environments that need it:

```python
    def reset(self) -> None:
        """Forget the previous solution; next step solves cold."""
        self._z_prev = None
```

- [ ] **Step 4: Run all NMPC tests**

Run: `uv run pytest tests/test_nmpc.py -v`
Expected: 7 passed (5 original + balance recovery + warm-start).

- [ ] **Step 5: Commit**

```bash
git add src/cubli_mpc/control/nmpc.py tests/test_nmpc.py
git commit -m "feat(nmpc): warm-start from previous solution"
```

---

## Task 10: Solver-failure fallback + counter

**Files:**
- Modify: `src/cubli_mpc/control/nmpc.py`
- Modify: `tests/test_nmpc.py`

When IPOPT fails or returns infeasible, fall back to nonlinear PD for that tick. Log the counter so the eval harness can surface it.

- [ ] **Step 1: Add the fallback test**

Append to `tests/test_nmpc.py`:

```python
from cubli_mpc.control.nonlinear_pd import (
    NonlinearPDController, NonlinearPDGains,
)


def _make_fallback_pd(hw):
    return NonlinearPDController(
        NonlinearPDGains(kp=0.5, kd=0.05, k_wheel=1e-4,
                         max_balance_tilt=math.radians(2.0),
                         max_torque=hw.motor_max_torque_nm),
        hw,
    )


def test_nmpc_fallback_counter_starts_at_zero():
    nmpc = NMPCController(_make_hw(), NMPCConfig(horizon_steps=10))
    assert nmpc.fallback_count == 0


def test_nmpc_fallback_returns_pd_value_on_solver_failure():
    """Force a failure by setting ipopt_max_iter=0; verify the fallback
    PD value is returned and the counter increments."""
    hw = _make_hw()
    pd = _make_fallback_pd(hw)
    nmpc = NMPCController(
        hw, NMPCConfig(horizon_steps=10, ipopt_max_iter=0),
        fallback_controller=pd,
    )
    x_hat = np.array([math.radians(10.0), 0.5, 0.0, 0.0])
    tau = nmpc.step(x_hat, t=0.0)
    expected = pd.step(x_hat, 0.0)
    assert tau == pytest.approx(expected)
    assert nmpc.fallback_count == 1


def test_nmpc_without_fallback_returns_zero_on_solver_failure():
    """If no fallback is provided, the NMPC returns 0 on failure and
    still counts the failure."""
    nmpc = NMPCController(
        _make_hw(), NMPCConfig(horizon_steps=10, ipopt_max_iter=0),
    )
    x_hat = np.array([math.radians(10.0), 0.5, 0.0, 0.0])
    tau = nmpc.step(x_hat, t=0.0)
    assert tau == 0.0
    assert nmpc.fallback_count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_nmpc.py -k fallback -v`
Expected: FAIL (no `fallback_count` attribute, no `fallback_controller` kwarg).

- [ ] **Step 3: Add fallback to NMPCController**

In `src/cubli_mpc/control/nmpc.py`:

1. Import the protocol type:
   ```python
   from cubli_mpc.control.base import Controller
   ```

2. Extend `__init__` signature and body to accept and store a fallback:
   ```python
       def __init__(
           self,
           hw: HardwareConfig,
           cfg: NMPCConfig,
           reference: Reference | None = None,
           fallback_controller: Controller | None = None,
       ):
           ...
           self._fallback = fallback_controller
           self._fallback_count = 0
           self._build_solver()
           self._z_prev: np.ndarray | None = None
   ```

3. Expose the counter:
   ```python
       @property
       def fallback_count(self) -> int:
           return self._fallback_count
   ```

4. Wrap the solver call in `step()` with success-check + fallback. Replace the solver block in `step()`:

   ```python
           sol = self._solver(x0=z0, p=p,
                              lbx=self._z_lb, ubx=self._z_ub,
                              lbg=self._g_lb, ubg=self._g_ub)
           stats = self._solver.stats()
           success = bool(stats.get("success", False))
           if not success:
               self._fallback_count += 1
               # Invalidate warm-start so the next solve starts cold.
               self._z_prev = None
               if self._fallback is not None:
                   return float(self._fallback.step(x_hat, t))
               return 0.0
           z_opt = np.array(sol["x"]).flatten()
           self._z_prev = z_opt
           u0 = z_opt[self._n_x_vars]
           return float(u0)
   ```

- [ ] **Step 4: Run all NMPC tests**

Run: `uv run pytest tests/test_nmpc.py -v`
Expected: 10 passed.

- [ ] **Step 5: Wire fallback into the CLI**

In `src/cubli_mpc/cli.py`, update the `nmpc` branch of `_build_controller`:

```python
    if args.controller == "nmpc":
        from cubli_mpc.control.nmpc import NMPCConfig, NMPCController
        from cubli_mpc.control.references import ConstantReference
        cfg = NMPCConfig(horizon_steps=args.nmpc_horizon, dt=args.nmpc_dt)
        ref = ConstantReference(
            target_theta=math.radians(args.nmpc_target_tilt_deg),
        )
        pd_gains = NonlinearPDGains(
            kp=args.kp, kd=args.kd, k_wheel=args.k_wheel,
            max_balance_tilt=math.radians(args.max_balance_tilt_deg),
            max_torque=hw.motor_max_torque_nm,
        )
        fallback = NonlinearPDController(pd_gains, hw)
        return NMPCController(hw, cfg, reference=ref,
                              fallback_controller=fallback)
```

- [ ] **Step 6: Commit**

```bash
git add src/cubli_mpc/control/nmpc.py src/cubli_mpc/cli.py tests/test_nmpc.py
git commit -m "feat(nmpc): solver-failure fallback to nonlinear-PD + counter"
```

---

## Task 11: Eval module (scenarios + metrics + runner)

**Files:**
- Create: `src/cubli_mpc/eval/__init__.py`
- Create: `src/cubli_mpc/eval/scenarios.py`
- Create: `src/cubli_mpc/eval/runner.py`
- Test: `tests/test_eval.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eval.py`:

```python
"""Tests for the eval harness: scenarios, metrics, runner."""
import math

import numpy as np
import pytest

from cubli_mpc.config import HardwareConfig, SimConfig
from cubli_mpc.control.random_torque import RandomController
from cubli_mpc.eval.runner import run_scenario
from cubli_mpc.eval.scenarios import SCENARIOS, get_scenario


def _make_hw() -> HardwareConfig:
    return HardwareConfig(
        cube_side_length_m=0.10,
        cube_mass_kg=0.40,
        wheel_mass_kg=0.10,
        wheel_radius_m=0.035,
        wheel_thickness_m=0.010,
        motor_max_torque_nm=0.20,
        motor_max_speed_rad_s=600.0,
        motor_torque_constant=0.01,
        edge_bearing_damping=1e-4,
        wheel_bearing_damping=1e-5,
    )


def test_scenario_registry_includes_expected_names():
    names = set(SCENARIOS.keys())
    assert "recover-15deg" in names
    assert "recover-30deg" in names
    assert "hold-step-disturb" in names
    assert "hold-impulse-disturb" in names


def test_get_scenario_returns_struct():
    s = get_scenario("recover-15deg")
    assert s.name == "recover-15deg"
    assert s.theta0_rad == pytest.approx(math.radians(15.0))


def test_run_scenario_returns_metrics_dict():
    hw = _make_hw()
    sim = SimConfig(dt_sim=0.001, dt_control=0.01)
    ctrl = RandomController(max_torque=hw.motor_max_torque_nm, seed=0)
    result = run_scenario(
        controller=ctrl, scenario=get_scenario("recover-15deg"),
        hw=hw, sim=sim, duration_s=2.0,
    )
    for key in ("survived", "theta_rms_deg", "theta_max_abs_deg",
                "wheel_speed_max_abs_rad_s", "wheel_speed_rms_rad_s",
                "torque_rms_nm", "torque_integral_abs_nm_s"):
        assert key in result, f"missing metric: {key}"


def test_run_scenario_records_disturbance_log():
    """The result should include the tau_ext trace so plots can show it."""
    hw = _make_hw()
    sim = SimConfig(dt_sim=0.001, dt_control=0.01)
    ctrl = RandomController(max_torque=hw.motor_max_torque_nm, seed=0)
    result = run_scenario(
        controller=ctrl, scenario=get_scenario("hold-step-disturb"),
        hw=hw, sim=sim, duration_s=3.0,
    )
    assert "log" in result
    tau_ext = result["log"]["tau_ext"]
    assert float(np.max(np.abs(tau_ext))) > 0.0


def test_random_controller_falls_on_recover_30deg():
    """Sanity-check the metric: a random controller from 30 deg should
    fall (survived=False)."""
    hw = _make_hw()
    sim = SimConfig(dt_sim=0.001, dt_control=0.01)
    ctrl = RandomController(max_torque=hw.motor_max_torque_nm, seed=0)
    result = run_scenario(
        controller=ctrl, scenario=get_scenario("recover-30deg"),
        hw=hw, sim=sim, duration_s=3.0,
    )
    assert result["survived"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_eval.py -v`
Expected: ImportError on `cubli_mpc.eval`.

- [ ] **Step 3: Implement the scenarios module**

Create `src/cubli_mpc/eval/__init__.py` (empty).

Create `src/cubli_mpc/eval/scenarios.py`:

```python
"""Named scenarios for the cubli-mpc eval harness.

Each `Scenario` is a reproducible test case: initial state, optional
disturbance plan, optional reference target. The eval runner consumes
these and emits a comparable metric dict.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from cubli_mpc.sim.disturbance import DisturbancePlan, StepDisturbance


@dataclass(frozen=True)
class Scenario:
    name: str
    theta0_rad: float
    duration_s: float = 5.0
    disturbance: DisturbancePlan | None = None
    target_theta_rad: float = 0.0
    swing_traj_path: str | None = None


SCENARIOS: dict[str, Scenario] = {
    "recover-15deg": Scenario(
        name="recover-15deg",
        theta0_rad=math.radians(15.0),
        duration_s=5.0,
    ),
    "recover-30deg": Scenario(
        name="recover-30deg",
        theta0_rad=math.radians(30.0),
        duration_s=5.0,
    ),
    "hold-step-disturb": Scenario(
        name="hold-step-disturb",
        theta0_rad=0.0,
        duration_s=5.0,
        disturbance=DisturbancePlan(steps=[
            StepDisturbance(magnitude=0.10, start_t=2.0, duration=0.5),
        ]),
    ),
    "hold-impulse-disturb": Scenario(
        name="hold-impulse-disturb",
        theta0_rad=0.0,
        duration_s=10.0,
        disturbance=DisturbancePlan(
            impulse_magnitude=0.08, impulse_rate_per_s=2.0, seed=42,
        ),
    ),
}


def get_scenario(name: str) -> Scenario:
    if name not in SCENARIOS:
        raise KeyError(
            f"unknown scenario {name!r}; choices: {sorted(SCENARIOS)}"
        )
    return SCENARIOS[name]
```

- [ ] **Step 4: Implement the eval runner**

Create `src/cubli_mpc/eval/runner.py`:

```python
"""Run a Scenario with a Controller and emit a metrics dict.

The runner is intentionally thin: it builds a CubliEnv, primes it from
the scenario's initial state, runs the existing `Runner` with the
scenario's disturbance plan, and extracts a fixed set of metrics from
the resulting log.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from cubli_mpc.config import HardwareConfig, SimConfig
from cubli_mpc.control.base import Controller
from cubli_mpc.eval.scenarios import Scenario
from cubli_mpc.runner import Runner
from cubli_mpc.sim.env import CubliEnv

SURVIVAL_THRESHOLD_RAD = math.radians(60.0)


def run_scenario(
    *,
    controller: Controller,
    scenario: Scenario,
    hw: HardwareConfig,
    sim: SimConfig,
    duration_s: float | None = None,
) -> dict[str, Any]:
    """Roll out `controller` on `scenario` and return a metrics dict.

    The optional `duration_s` overrides the scenario's default duration
    (useful for quicker tests).
    """
    duration = duration_s if duration_s is not None else scenario.duration_s

    env = CubliEnv(hw, sim)
    env.reset(theta0=scenario.theta0_rad)
    runner = Runner(env, controller, sim, disturbance=scenario.disturbance)
    log = runner.run(duration_s=duration)

    theta = log["theta"]
    omega_w = log["wheel_speed"]
    tau = log["tau"]

    survived = bool(float(np.max(np.abs(theta))) < SURVIVAL_THRESHOLD_RAD)
    theta_rms = float(np.sqrt(np.mean(theta * theta)))
    theta_max_abs = float(np.max(np.abs(theta)))
    omega_max_abs = float(np.max(np.abs(omega_w)))
    omega_rms = float(np.sqrt(np.mean(omega_w * omega_w)))
    tau_rms = float(np.sqrt(np.mean(tau * tau)))
    tau_int_abs = float(np.sum(np.abs(tau)) * sim.dt_control)

    metrics: dict[str, Any] = {
        "scenario": scenario.name,
        "survived": survived,
        "theta_rms_deg": math.degrees(theta_rms),
        "theta_max_abs_deg": math.degrees(theta_max_abs),
        "wheel_speed_max_abs_rad_s": omega_max_abs,
        "wheel_speed_rms_rad_s": omega_rms,
        "torque_rms_nm": tau_rms,
        "torque_integral_abs_nm_s": tau_int_abs,
        "log": log,
    }
    # NMPC-only: surface the fallback counter if present.
    if hasattr(controller, "fallback_count"):
        metrics["solver_fallback_rate"] = (
            float(controller.fallback_count) / max(1, len(theta))
        )
    return metrics
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_eval.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/cubli_mpc/eval/ tests/test_eval.py
git commit -m "feat(eval): scenario registry, metrics, run_scenario runner"
```

**Scope note:** the spec's "metronome-1hz-20deg" scenario is not in this registry. That scenario depends on a swing trajectory file (built by `plan-swing` in Task 14) and a NMPC controller wired with `PeriodicTrajectoryReference` (Tasks 15–16), so the comparison-with-RL eval starts with the 4 balance/disturbance scenarios above. After Task 16, run `cubli-mpc sim --controller nmpc --swing-traj ...` and `--plot` to inspect tracking; wiring metronome into `eval` is a follow-on extension once the tracking workflow is stable.

---

## Task 12: `cubli-mpc eval` CLI subcommand

**Files:**
- Modify: `src/cubli_mpc/cli.py`

- [ ] **Step 1: Add `_cmd_eval` and the subparser**

In `src/cubli_mpc/cli.py`, add this function near `_cmd_sim`:

```python
def _cmd_eval(args: argparse.Namespace) -> int:
    import json

    from cubli_mpc.eval.runner import run_scenario
    from cubli_mpc.eval.scenarios import SCENARIOS, get_scenario

    hw, sim = load_config(args.config)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.scenarios == ["all"]:
        names = sorted(SCENARIOS.keys())
    else:
        names = args.scenarios

    # Sim args needed by _build_controller — fake the ones it expects.
    args.seed = getattr(args, "seed", 0)

    results: list[dict] = []
    for name in names:
        scenario = get_scenario(name)
        controller = _build_controller(args, hw)
        if hasattr(controller, "reset"):
            controller.reset()
        result = run_scenario(controller=controller, scenario=scenario,
                              hw=hw, sim=sim, duration_s=args.duration)
        plot_path = args.out / f"{name}.png"
        log = result.pop("log")
        target = scenario.target_theta_rad if scenario.target_theta_rad else None
        _save_state_plot(log, plot_path, target_theta_rad=target)
        results.append(result)
        print(f"  {name}: survived={result['survived']} "
              f"|theta|_max={result['theta_max_abs_deg']:.2f} deg "
              f"tau_rms={result['torque_rms_nm']:.4f} Nm")

    (args.out / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out / 'metrics.json'}")
    return 0
```

In `main()`, register the subparser. Find the line after `p_sim.set_defaults(func=_cmd_sim)` and add:

```python
    p_eval = sub.add_parser("eval", help="Evaluate a controller on named scenarios")
    p_eval.add_argument("--config", required=True, type=Path)
    p_eval.add_argument("--out", required=True, type=Path,
                        help="Output directory for plots + metrics.json")
    p_eval.add_argument("--scenarios", nargs="+", default=["all"],
                        help='Scenario names, or "all"')
    p_eval.add_argument("--duration", type=float, default=None,
                        help="Override scenario duration (s)")
    # Reuse the sim controller flags.
    p_eval.add_argument("--controller",
                        choices=("pd", "random", "policy", "nmpc"),
                        default="pd")
    p_eval.add_argument("--seed", type=int, default=0)
    p_eval.add_argument("--policy-path", type=Path, default=None)
    p_eval.add_argument("--policy-algo", choices=("sac", "ppo"), default=None)
    p_eval.add_argument("--target-tilt-deg", type=float, default=0.0)
    p_eval.add_argument("--kp", type=float, default=0.5)
    p_eval.add_argument("--kd", type=float, default=0.05)
    p_eval.add_argument("--k-wheel", type=float, default=1e-4)
    p_eval.add_argument("--max-balance-tilt-deg", type=float, default=2.0)
    p_eval.add_argument("--nmpc-horizon", type=int, default=50)
    p_eval.add_argument("--nmpc-dt", type=float, default=0.010)
    p_eval.add_argument("--nmpc-target-tilt-deg", type=float, default=0.0)
    p_eval.set_defaults(func=_cmd_eval)
```

- [ ] **Step 2: Smoke test**

Run:
```bash
uv run cubli-mpc eval --config configs/default.yaml --controller pd \
    --scenarios recover-15deg hold-step-disturb \
    --duration 3 --out /tmp/eval-pd
```

Expected: prints two scenario lines, writes `/tmp/eval-pd/metrics.json` and two PNGs.

- [ ] **Step 3: Commit**

```bash
git add src/cubli_mpc/cli.py
git commit -m "feat(cli): cubli-mpc eval subcommand for scenario sweeps"
```

---

## Task 13: Swing planner module

**Files:**
- Create: `src/cubli_mpc/control/swing_planner.py`
- Test: `tests/test_swing_planner.py`

Half-period direct multiple shooting; mirror to a full period at save time.

- [ ] **Step 1: Write failing tests**

Create `tests/test_swing_planner.py`:

```python
"""Tests for the offline swing-trajectory planner."""
import math
from pathlib import Path

import numpy as np
import pytest

from cubli_mpc.config import HardwareConfig
from cubli_mpc.control.swing_planner import SwingTrajConfig, plan_swing


def _make_hw() -> HardwareConfig:
    return HardwareConfig(
        cube_side_length_m=0.10,
        cube_mass_kg=0.40,
        wheel_mass_kg=0.10,
        wheel_radius_m=0.035,
        wheel_thickness_m=0.010,
        motor_max_torque_nm=0.20,
        motor_max_speed_rad_s=600.0,
        motor_torque_constant=0.01,
        edge_bearing_damping=1e-4,
        wheel_bearing_damping=1e-5,
    )


def test_plan_swing_returns_npz_path(tmp_path: Path):
    cfg = SwingTrajConfig(
        period_s=1.0, theta_target_rad=math.radians(15.0),
        n_segments=40, save_path=tmp_path / "swing.npz",
    )
    out = plan_swing(_make_hw(), cfg)
    assert out == tmp_path / "swing.npz"
    assert out.exists()


def test_plan_swing_npz_contents(tmp_path: Path):
    cfg = SwingTrajConfig(
        period_s=1.0, theta_target_rad=math.radians(15.0),
        n_segments=40, save_path=tmp_path / "swing.npz",
    )
    out = plan_swing(_make_hw(), cfg)
    data = np.load(out)
    # Full period after mirroring: 2 * n_segments + 1 samples.
    assert data["t"].shape == (2 * 40 + 1,)
    assert data["x_ref"].shape == (2 * 40 + 1, 3)
    assert data["u_ref"].shape == (2 * 40, 1)


def test_plan_swing_boundary_conditions(tmp_path: Path):
    """Trajectory should start at +theta_target, hit -theta_target at the
    half-period, and return to +theta_target at the end of the period."""
    theta_target = math.radians(15.0)
    cfg = SwingTrajConfig(
        period_s=1.0, theta_target_rad=theta_target,
        n_segments=40, save_path=tmp_path / "swing.npz",
    )
    out = plan_swing(_make_hw(), cfg)
    data = np.load(out)
    x = data["x_ref"]
    assert x[0, 0] == pytest.approx(theta_target, abs=1e-3)
    assert x[40, 0] == pytest.approx(-theta_target, abs=1e-3)
    assert x[-1, 0] == pytest.approx(theta_target, abs=1e-3)


def test_plan_swing_torque_within_limits(tmp_path: Path):
    hw = _make_hw()
    cfg = SwingTrajConfig(
        period_s=1.5, theta_target_rad=math.radians(15.0),
        n_segments=60, save_path=tmp_path / "swing.npz",
    )
    out = plan_swing(hw, cfg)
    data = np.load(out)
    u = data["u_ref"]
    assert float(np.max(np.abs(u))) <= hw.motor_max_torque_nm + 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_swing_planner.py -v`
Expected: ImportError on `cubli_mpc.control.swing_planner`.

- [ ] **Step 3: Implement the planner**

Create `src/cubli_mpc/control/swing_planner.py`:

```python
"""Offline swing-trajectory planner for the M3 metronome task.

Solves a half-period direct multiple-shooting NLP between +theta_target
and -theta_target with both endpoints at rest. The dynamics symmetry
(theta, theta_dot, omega_w, tau) -> (-theta, -theta_dot, -omega_w, -tau)
lets us mirror the half period to a full one without re-solving.

Save format (npz):
    t      : (2N+1,)
    x_ref  : (2N+1, 3)  [theta, theta_dot, omega_w]
    u_ref  : (2N,   1)  [tau]
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import casadi as ca
import numpy as np

from cubli_mpc.config import HardwareConfig
from cubli_mpc.model.dynamics import make_rk4_step


@dataclass(frozen=True)
class SwingTrajConfig:
    period_s: float
    theta_target_rad: float
    n_segments: int = 200
    smoothness_weight: float = 1e-3  # on sum-of-squared torque rate
    save_path: Path | None = None


def plan_swing(hw: HardwareConfig, cfg: SwingTrajConfig) -> Path:
    """Solve the half-period swing trajopt and write the full mirrored
    trajectory to `cfg.save_path` (defaults to runs/swing.npz)."""
    save_path = (cfg.save_path if cfg.save_path is not None
                 else Path("runs/swing.npz"))
    save_path.parent.mkdir(parents=True, exist_ok=True)

    half_period = 0.5 * cfg.period_s
    N = cfg.n_segments
    dt = half_period / N
    F = make_rk4_step(hw, dt=dt)

    X = ca.MX.sym("X", 3, N + 1)
    U = ca.MX.sym("U", 1, N)

    g_list = []
    cost = ca.MX(0.0)
    for k in range(N):
        g_list.append(X[:, k + 1] - F(X[:, k], U[:, k]))
        cost = cost + U[:, k] * U[:, k] * dt
        if k > 0:
            du = U[:, k] - U[:, k - 1]
            cost = cost + cfg.smoothness_weight * du * du

    # Boundary conditions on theta and theta_dot.
    g_list.append(X[0, 0] - cfg.theta_target_rad)
    g_list.append(X[1, 0])
    g_list.append(X[2, 0])  # start with zero wheel speed for repeatability
    g_list.append(X[0, N] + cfg.theta_target_rad)
    g_list.append(X[1, N])

    g = ca.vertcat(*g_list)
    z = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

    # Box bounds: torque limited, wheel speed limited, theta unconstrained.
    z_lb = np.full(z.size1(), -np.inf)
    z_ub = np.full(z.size1(), +np.inf)
    omega_lim = hw.motor_max_speed_rad_s
    for k in range(N + 1):
        z_lb[3 * k + 2] = -omega_lim
        z_ub[3 * k + 2] = +omega_lim
    u_start = 3 * (N + 1)
    z_lb[u_start:] = -hw.motor_max_torque_nm
    z_ub[u_start:] = +hw.motor_max_torque_nm

    nlp = {"x": z, "f": cost, "g": g}
    opts = {
        "ipopt.max_iter": 500,
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "print_time": 0,
    }
    solver = ca.nlpsol("swing_planner", "ipopt", nlp, opts)

    # Initial guess: linear interpolation between endpoints, zero torque.
    x_init = np.zeros((3, N + 1))
    x_init[0, :] = np.linspace(cfg.theta_target_rad,
                               -cfg.theta_target_rad, N + 1)
    u_init = np.zeros((1, N))
    z0 = np.concatenate([x_init.reshape(-1, order="F"),
                         u_init.reshape(-1, order="F")])

    sol = solver(x0=z0, lbx=z_lb, ubx=z_ub,
                 lbg=np.zeros(g.size1()), ubg=np.zeros(g.size1()))
    if not solver.stats().get("success", False):
        raise RuntimeError(
            f"swing planner failed for period={cfg.period_s}, "
            f"theta_target={cfg.theta_target_rad}: solver did not converge"
        )

    z_opt = np.array(sol["x"]).flatten()
    X_opt = z_opt[: 3 * (N + 1)].reshape(3, N + 1, order="F").T  # (N+1, 3)
    U_opt = z_opt[3 * (N + 1):].reshape(1, N, order="F").T        # (N, 1)

    # Mirror to a full period using the dynamics symmetry
    #     (theta, theta_dot, omega_w, tau) -> -(theta, theta_dot, omega_w, tau).
    # If X_opt(t) is a solution from +theta_target to -theta_target under
    # U_opt(t), then -X_opt(t) is a solution from -theta_target to
    # +theta_target under -U_opt(t). The second half is -X_opt played
    # forward, dropping the first sample to avoid duplicating t=T/2.
    t_half = np.linspace(0.0, half_period, N + 1)
    t_full = np.concatenate([t_half, t_half[1:] + half_period])
    X_full = np.concatenate([X_opt, -X_opt[1:, :]], axis=0)
    U_full = np.concatenate([U_opt, -U_opt], axis=0)

    np.savez(save_path, t=t_full, x_ref=X_full, u_ref=U_full)
    return save_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_swing_planner.py -v`
Expected: 4 passed. The solve takes a few seconds; the symmetry/boundary tests verify the mirror logic.

If the boundary tests fail, the most common issue is the mirror direction — check that `X_full[N, 0] == -theta_target` (end of half period) and `X_full[-1, 0] == +theta_target` (end of full period).

- [ ] **Step 5: Commit**

```bash
git add src/cubli_mpc/control/swing_planner.py tests/test_swing_planner.py
git commit -m "feat(control): half-period swing planner with sign-flip mirror"
```

---

## Task 14: `cubli-mpc plan-swing` CLI

**Files:**
- Modify: `src/cubli_mpc/cli.py`

- [ ] **Step 1: Add `_cmd_plan_swing` and subparser**

In `src/cubli_mpc/cli.py`, add the function:

```python
def _cmd_plan_swing(args: argparse.Namespace) -> int:
    import math as _m

    from cubli_mpc.control.swing_planner import SwingTrajConfig, plan_swing

    hw, _sim = load_config(args.config)
    cfg = SwingTrajConfig(
        period_s=args.period,
        theta_target_rad=_m.radians(args.theta_deg),
        n_segments=args.n_segments,
        smoothness_weight=args.smoothness,
        save_path=args.out,
    )
    out = plan_swing(hw, cfg)
    print(f"wrote {out}")
    return 0
```

In `main()` register the subparser:

```python
    p_plan = sub.add_parser("plan-swing", help="Offline swing trajopt")
    p_plan.add_argument("--config", required=True, type=Path)
    p_plan.add_argument("--period", type=float, required=True,
                        help="Swing period (s)")
    p_plan.add_argument("--theta-deg", type=float, required=True,
                        help="Swing amplitude (degrees)")
    p_plan.add_argument("--n-segments", type=int, default=200,
                        help="Multiple-shooting segments per half period")
    p_plan.add_argument("--smoothness", type=float, default=1e-3,
                        help="Weight on Σ(Δτ)² to discourage chatter")
    p_plan.add_argument("--out", type=Path, required=True,
                        help="Output .npz path")
    p_plan.set_defaults(func=_cmd_plan_swing)
```

- [ ] **Step 2: Smoke test**

Run:
```bash
uv run cubli-mpc plan-swing --config configs/default.yaml \
    --period 1.0 --theta-deg 15 --out /tmp/swing.npz
```

Expected: prints `wrote /tmp/swing.npz`, file exists.

- [ ] **Step 3: Commit**

```bash
git add src/cubli_mpc/cli.py
git commit -m "feat(cli): cubli-mpc plan-swing subcommand"
```

---

## Task 15: PeriodicTrajectoryReference

**Files:**
- Modify: `src/cubli_mpc/control/references.py`
- Modify: `tests/test_references.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_references.py`:

```python
from pathlib import Path

from cubli_mpc.control.references import PeriodicTrajectoryReference


def _write_trivial_swing(path: Path, period: float, n: int) -> Path:
    """Write a synthetic swing trajectory: sine wave on theta, zeros
    elsewhere. Used to test the reference's interpolation + wrap."""
    t = np.linspace(0.0, period, n + 1)
    x_ref = np.zeros((n + 1, 3))
    x_ref[:, 0] = 0.1 * np.sin(2 * np.pi * t / period)
    u_ref = np.zeros((n, 1))
    np.savez(path, t=t, x_ref=x_ref, u_ref=u_ref)
    return path


def test_periodic_reference_loads_from_npz(tmp_path):
    path = _write_trivial_swing(tmp_path / "s.npz", period=1.0, n=100)
    ref = PeriodicTrajectoryReference.from_file(path)
    x, u = ref.at(t=0.0, dt=0.01, horizon=5)
    assert x.shape == (6, 3)
    assert u.shape == (5, 1)


def test_periodic_reference_loops_at_period(tmp_path):
    """t=0 and t=period should give the same x_ref."""
    path = _write_trivial_swing(tmp_path / "s.npz", period=1.0, n=100)
    ref = PeriodicTrajectoryReference.from_file(path)
    x_a, _ = ref.at(t=0.0, dt=0.01, horizon=3)
    x_b, _ = ref.at(t=1.0, dt=0.01, horizon=3)
    assert np.allclose(x_a, x_b, atol=1e-6)


def test_periodic_reference_interpolates_between_samples(tmp_path):
    """Mid-sample query should give a value between the two enclosing
    samples (linear interp)."""
    path = _write_trivial_swing(tmp_path / "s.npz", period=1.0, n=100)
    ref = PeriodicTrajectoryReference.from_file(path)
    x, _ = ref.at(t=0.25, dt=0.0, horizon=0)  # query single point
    # sin(2*pi*0.25) = 1.0, scaled by 0.1
    assert x[0, 0] == pytest.approx(0.1, abs=1e-3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_references.py -v`
Expected: ImportError on `PeriodicTrajectoryReference`.

- [ ] **Step 3: Implement PeriodicTrajectoryReference**

Append to `src/cubli_mpc/control/references.py`:

```python
from pathlib import Path


class PeriodicTrajectoryReference:
    """A one-period reference loaded from disk; loops at the period.

    Linear interpolation between samples in the saved time grid lets the
    NMPC use any dt independent of the planner's `n_segments`.
    """

    def __init__(self, t: np.ndarray, x_ref: np.ndarray, u_ref: np.ndarray):
        self._t = np.asarray(t, dtype=np.float64)
        self._x = np.asarray(x_ref, dtype=np.float64)
        self._u = np.asarray(u_ref, dtype=np.float64)
        self._period = float(self._t[-1] - self._t[0])
        if self._period <= 0:
            raise ValueError("trajectory time grid must have positive span")

    @classmethod
    def from_file(cls, path: str | Path) -> "PeriodicTrajectoryReference":
        data = np.load(path)
        return cls(data["t"], data["x_ref"], data["u_ref"])

    def at(
        self, t: float, dt: float, horizon: int
    ) -> tuple[np.ndarray, np.ndarray]:
        x_ref = np.empty((horizon + 1, 3), dtype=np.float64)
        u_ref = np.empty((horizon, 1), dtype=np.float64)
        for k in range(horizon + 1):
            tk = (t + k * dt) % self._period
            x_ref[k] = self._interp_x(tk)
        for k in range(horizon):
            tk = (t + k * dt) % self._period
            u_ref[k] = self._interp_u(tk)
        return x_ref, u_ref

    def _interp_x(self, tq: float) -> np.ndarray:
        # np.interp does linear interp on 1D; loop over 3 components.
        return np.array([
            np.interp(tq, self._t, self._x[:, 0]),
            np.interp(tq, self._t, self._x[:, 1]),
            np.interp(tq, self._t, self._x[:, 2]),
        ])

    def _interp_u(self, tq: float) -> np.ndarray:
        # u_ref has one fewer entry than t; use midpoints for the grid.
        t_u = 0.5 * (self._t[:-1] + self._t[1:])
        return np.array([np.interp(tq, t_u, self._u[:, 0])])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_references.py -v`
Expected: 7 passed total (4 + 3 new).

- [ ] **Step 5: Commit**

```bash
git add src/cubli_mpc/control/references.py tests/test_references.py
git commit -m "feat(control): PeriodicTrajectoryReference for M3 tracking"
```

---

## Task 16: Wire `--swing-traj` into sim + M3 integration check

**Files:**
- Modify: `src/cubli_mpc/cli.py`
- Modify: `tests/test_nmpc.py`

- [ ] **Step 1: Add `--swing-traj` flag and wire it**

In `src/cubli_mpc/cli.py`, just before `p_sim.set_defaults(func=_cmd_sim)`:

```python
p_sim.add_argument("--swing-traj", type=Path, default=None,
                   help="(NMPC) Track the periodic reference loaded from "
                        "this .npz (from `cubli-mpc plan-swing`). Overrides "
                        "--nmpc-target-tilt-deg.")
```

Then update the `nmpc` branch of `_build_controller`:

```python
    if args.controller == "nmpc":
        from cubli_mpc.control.nmpc import NMPCConfig, NMPCController
        from cubli_mpc.control.references import (
            ConstantReference, PeriodicTrajectoryReference,
        )
        cfg = NMPCConfig(horizon_steps=args.nmpc_horizon, dt=args.nmpc_dt)
        if args.swing_traj is not None:
            ref = PeriodicTrajectoryReference.from_file(args.swing_traj)
        else:
            ref = ConstantReference(
                target_theta=math.radians(args.nmpc_target_tilt_deg),
            )
        pd_gains = NonlinearPDGains(
            kp=args.kp, kd=args.kd, k_wheel=args.k_wheel,
            max_balance_tilt=math.radians(args.max_balance_tilt_deg),
            max_torque=hw.motor_max_torque_nm,
        )
        fallback = NonlinearPDController(pd_gains, hw)
        return NMPCController(hw, cfg, reference=ref,
                              fallback_controller=fallback)
```

- [ ] **Step 2: Add the M3 integration test**

Append to `tests/test_nmpc.py`:

```python
def test_nmpc_tracks_swing_trajectory(tmp_path):
    """Plan a 1.5 s / 10 deg swing and verify NMPC tracking holds RMS
    theta-error well under the amplitude over 3 s."""
    from cubli_mpc.control.references import PeriodicTrajectoryReference
    from cubli_mpc.control.swing_planner import SwingTrajConfig, plan_swing
    from cubli_mpc.runner import Runner
    from cubli_mpc.sim.env import CubliEnv

    hw = _make_hw()
    swing_path = tmp_path / "swing.npz"
    plan_swing(hw, SwingTrajConfig(
        period_s=1.5, theta_target_rad=math.radians(10.0),
        n_segments=80, save_path=swing_path,
    ))
    ref = PeriodicTrajectoryReference.from_file(swing_path)

    sim = SimConfig(dt_sim=0.001, dt_control=0.01)
    env = CubliEnv(hw, sim)
    env.reset(theta0=math.radians(10.0))  # start of swing

    nmpc = NMPCController(
        hw, NMPCConfig(horizon_steps=40, dt=sim.dt_control),
        reference=ref,
    )
    runner = Runner(env, nmpc, sim)
    log = runner.run(duration_s=3.0)

    theta = log["theta"]
    # Build the planned theta at the same time grid, then take RMS error.
    t = log["t"]
    target = np.array([ref._interp_x(float(ti) % 1.5)[0] for ti in t])
    err = theta - target
    rms_err_deg = math.degrees(float(np.sqrt(np.mean(err * err))))
    assert rms_err_deg < 5.0, f"RMS tracking error {rms_err_deg:.2f} deg"
```

- [ ] **Step 3: Run the M3 integration test**

Run: `uv run pytest tests/test_nmpc.py::test_nmpc_tracks_swing_trajectory -v`
Expected: PASS. Takes ~30 s.

If RMS error is large (>5 deg), check:
1. Planner's torque doesn't saturate (Task 13's bound test).
2. NMPC horizon (40 × 10 ms = 0.4 s) is enough to "see" upcoming swing changes.
3. The reference mirror is oriented correctly: `target[0] ~ +10 deg`, `target[len(t)//2] ~ -10 deg` halfway through.

- [ ] **Step 4: End-to-end smoke test through the CLI**

Run:
```bash
uv run cubli-mpc plan-swing --config configs/default.yaml \
    --period 1.5 --theta-deg 10 --out /tmp/swing.npz
uv run cubli-mpc sim --config configs/default.yaml --controller nmpc \
    --swing-traj /tmp/swing.npz --initial-tilt-deg 10 --duration 5 \
    --plot /tmp/swing-sim.png
```

Expected: both commands complete, plot shows theta oscillating between roughly ±10 deg.

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest --ignore=tests/test_recorder.py -q`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/cubli_mpc/cli.py tests/test_nmpc.py
git commit -m "feat(cli): --swing-traj for NMPC + M3 tracking integration test"
```

---

## Final verification

After Task 16, the following should all work:

```bash
# Balance via NMPC
uv run cubli-mpc sim --config configs/default.yaml --controller nmpc \
    --duration 5 --initial-tilt-deg 15 --plot /tmp/bal.png

# Eval sweep
uv run cubli-mpc eval --config configs/default.yaml --controller nmpc \
    --scenarios all --out /tmp/eval-nmpc

# Compare to PD
uv run cubli-mpc eval --config configs/default.yaml --controller pd \
    --scenarios all --out /tmp/eval-pd

# Plan + run a swing
uv run cubli-mpc plan-swing --config configs/default.yaml \
    --period 1.0 --theta-deg 20 --out runs/swing.npz
uv run cubli-mpc sim --config configs/default.yaml --controller nmpc \
    --swing-traj runs/swing.npz --duration 10 --video runs/swing.mp4 \
    --plot runs/swing.png
```

Compare `metrics.json` from the two eval runs to get the MPC-vs-PD numbers the project exists to produce.

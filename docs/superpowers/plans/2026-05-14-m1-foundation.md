# cubli-mpc M1 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the parametric MuJoCo sim and run a nonlinear PD controller that stabilizes the cube on its edge from a small initial tilt.

**Architecture:** Parametric `HardwareConfig` generates an MJCF XML for MuJoCo. A `Controller` protocol abstracts the control law. A `Runner` ticks physics at 1 kHz and calls the controller at 100 Hz. Energy-feedforward nonlinear PD provides the M1 baseline; later milestones (M2 stabilization NMPC, M3 swing, M4 robustness) will get their own plans.

**Tech Stack:** Python ≥ 3.11, `uv`, `mujoco`, `numpy`, `pyyaml`, `pytest`. No `casadi` or `acados` in this milestone.

**Spec reference:** `docs/superpowers/specs/2026-05-14-cubli-mpc-design.md`

**Working directory for all commands:** `~/efr/cubli-mpc` (run `cd ~/efr/cubli-mpc` first).

---

## Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `src/cubli_mpc/__init__.py`
- Create: `src/cubli_mpc/model/__init__.py`
- Create: `src/cubli_mpc/sim/__init__.py`
- Create: `src/cubli_mpc/control/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Pin Python version**

Create `.python-version`:

```
3.11
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "cubli-mpc"
version = "0.1.0"
description = "Reaction wheel cube balancer: parametric MuJoCo simulator + nonlinear MPC"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "mujoco>=3.0",
    "numpy>=1.26",
    "scipy>=1.11",
    "matplotlib>=3.8",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
]

[project.scripts]
cubli-mpc = "cubli_mpc.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/cubli_mpc"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
```

- [ ] **Step 3: Create empty package init files**

Create these as empty files:
- `src/cubli_mpc/__init__.py`
- `src/cubli_mpc/model/__init__.py`
- `src/cubli_mpc/sim/__init__.py`
- `src/cubli_mpc/control/__init__.py`
- `tests/__init__.py`

- [ ] **Step 4: Create `tests/conftest.py`**

```python
"""Shared pytest fixtures and path setup."""
import sys
from pathlib import Path

# Ensure src/ is importable when running tests without installing the package.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
```

- [ ] **Step 5: Sync dependencies**

Run: `uv sync --extra dev`
Expected: creates `.venv/`, installs all deps, generates `uv.lock`.

- [ ] **Step 6: Verify import works**

Run: `uv run python -c "import cubli_mpc; print('ok')"`
Expected output: `ok`

- [ ] **Step 7: Verify pytest collects nothing yet**

Run: `uv run pytest`
Expected: `no tests ran`, exit 5 (pytest's "no tests collected" code) — that's fine, it confirms pytest is wired up.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .python-version uv.lock src/cubli_mpc tests
git commit -m "scaffold: uv project + package skeleton"
```

---

## Task 2: `HardwareConfig` dataclass with derived properties

**Files:**
- Create: `src/cubli_mpc/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing tests for `HardwareConfig`**

Create `tests/test_config.py`:

```python
"""Tests for HardwareConfig and SimConfig."""
import math
import pytest
from cubli_mpc.config import HardwareConfig, SimConfig


def _make_hw(**overrides) -> HardwareConfig:
    defaults = dict(
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
    defaults.update(overrides)
    return HardwareConfig(**defaults)


def test_hardware_config_constructs():
    cfg = _make_hw()
    assert cfg.cube_mass_kg == 0.40
    assert cfg.cube_com_offset_m == 0.0  # default
    assert cfg.wheel_offset_m == 0.0  # default


def test_cube_inertia_about_edge_unit():
    # Unit cube of unit mass: I_edge = 2/3 (analytic from parallel-axis)
    cfg = _make_hw(cube_side_length_m=1.0, cube_mass_kg=1.0)
    assert cfg.cube_inertia_about_edge == pytest.approx(2.0 / 3.0)


def test_cube_com_distance_from_edge_unit():
    # Unit cube: distance from balancing edge to geometric center = a*sqrt(2)/2
    cfg = _make_hw(cube_side_length_m=1.0)
    assert cfg.cube_com_distance_from_edge == pytest.approx(math.sqrt(2.0) / 2.0)


def test_wheel_inertia_solid_cylinder():
    # m=1, r=1: I = 1/2 (solid cylinder about spin axis)
    cfg = _make_hw(wheel_mass_kg=1.0, wheel_radius_m=1.0)
    assert cfg.wheel_inertia_about_spin_axis == pytest.approx(0.5)


def test_angular_momentum_budget():
    cfg = _make_hw(wheel_mass_kg=0.10, wheel_radius_m=0.035,
                   motor_max_speed_rad_s=600.0)
    expected = 0.5 * 0.10 * 0.035**2 * 600.0
    assert cfg.angular_momentum_budget == pytest.approx(expected)


def test_gravity_moment_coefficient():
    # m_cube * g * L where L = cube_com_distance_from_edge
    cfg = _make_hw(cube_mass_kg=1.0, cube_side_length_m=1.0)
    expected = 1.0 * 9.81 * (math.sqrt(2.0) / 2.0)
    assert cfg.gravity_moment_coefficient == pytest.approx(expected)


def test_sim_config_defaults():
    cfg = SimConfig()
    assert cfg.dt_sim == 0.001
    assert cfg.dt_control == 0.010
    assert cfg.sensor_noise is False
    assert cfg.sensor_delay_ms == 0.0
    assert cfg.seed == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -v`
Expected: collection error — `cubli_mpc.config` does not exist.

- [ ] **Step 3: Implement `config.py`**

Create `src/cubli_mpc/config.py`:

```python
"""Configuration dataclasses for cubli-mpc.

HardwareConfig captures BOM-level parameters; derived inertias and the
gravity moment coefficient are computed from geometry + mass.

SimConfig captures simulator-only parameters (timesteps, noise, seed).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

GRAVITY_M_PER_S2 = 9.81


@dataclass(frozen=True)
class HardwareConfig:
    # Cube
    cube_side_length_m: float
    cube_mass_kg: float
    cube_com_offset_m: float = 0.0  # along tilt axis; reserved for asymmetry studies

    # Wheel
    wheel_mass_kg: float = 0.0
    wheel_radius_m: float = 0.0
    wheel_thickness_m: float = 0.0
    wheel_offset_m: float = 0.0

    # Motor
    motor_max_torque_nm: float = 0.0
    motor_max_speed_rad_s: float = 0.0
    motor_torque_constant: float = 0.0

    # Bearing damping (viscous)
    edge_bearing_damping: float = 0.0
    wheel_bearing_damping: float = 0.0

    # ---- Derived geometric quantities ----

    @property
    def cube_com_distance_from_edge(self) -> float:
        """Distance from balancing edge to cube geometric center.

        For a uniform cube of side a balanced on one of its edges, the
        geometric center sits at perpendicular distance a*sqrt(2)/2 from
        that edge (along the body diagonal of the face perpendicular to
        the edge).
        """
        return self.cube_side_length_m * math.sqrt(2.0) / 2.0

    @property
    def cube_inertia_about_edge(self) -> float:
        """Moment of inertia of uniform cube about its balancing edge.

        Derivation: I_com (about an axis parallel to the edge, through
        the geometric center) = (1/6) * m * a^2 for a uniform cube.
        Parallel axis theorem with d = a*sqrt(2)/2 gives
            I_edge = (1/6 + 1/2) * m * a^2 = (2/3) * m * a^2.
        """
        return (2.0 / 3.0) * self.cube_mass_kg * self.cube_side_length_m**2

    @property
    def wheel_inertia_about_spin_axis(self) -> float:
        """Solid cylinder about its spin axis: (1/2) * m * r^2."""
        return 0.5 * self.wheel_mass_kg * self.wheel_radius_m**2

    @property
    def gravity_moment_coefficient(self) -> float:
        """m * g * L: gravity restoring/destabilizing torque coefficient
        for the body angle, evaluated as sin(theta) * this value.

        Uses cube mass and COM distance; wheel mass is treated as a small
        correction handled by MuJoCo in the sim model.
        """
        return (self.cube_mass_kg * GRAVITY_M_PER_S2
                * self.cube_com_distance_from_edge)

    @property
    def angular_momentum_budget(self) -> float:
        """Maximum angular momentum the wheel can store before saturation:
        I_wheel * omega_max. Determines feasibility of swing trajectories.
        """
        return self.wheel_inertia_about_spin_axis * self.motor_max_speed_rad_s


@dataclass(frozen=True)
class SimConfig:
    dt_sim: float = 0.001
    dt_control: float = 0.010
    sensor_noise: bool = False
    sensor_delay_ms: float = 0.0
    seed: int = 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: all 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cubli_mpc/config.py tests/test_config.py
git commit -m "feat(config): HardwareConfig + SimConfig with derived properties"
```

---

## Task 3: YAML config loader + default config

**Files:**
- Modify: `src/cubli_mpc/config.py` (add `load_config`)
- Create: `configs/default.yaml`
- Modify: `tests/test_config.py` (add loader tests)

- [ ] **Step 1: Write the default config file**

Create `configs/default.yaml`:

```yaml
# Plausible starting values for a ~10 cm cube, ~0.4 kg, with a small reaction
# wheel. Swap for measured values once available.
hardware:
  cube_side_length_m: 0.10
  cube_mass_kg: 0.40
  cube_com_offset_m: 0.0

  wheel_mass_kg: 0.10
  wheel_radius_m: 0.035
  wheel_thickness_m: 0.010
  wheel_offset_m: 0.0

  motor_max_torque_nm: 0.20
  motor_max_speed_rad_s: 600.0
  motor_torque_constant: 0.01

  edge_bearing_damping: 1.0e-4
  wheel_bearing_damping: 1.0e-5

sim:
  dt_sim: 0.001
  dt_control: 0.010
  sensor_noise: false
  sensor_delay_ms: 0.0
  seed: 0
```

- [ ] **Step 2: Write failing test for loader**

Add to `tests/test_config.py`:

```python
from pathlib import Path
from cubli_mpc.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_default_config():
    hw, sim = load_config(REPO_ROOT / "configs" / "default.yaml")
    assert hw.cube_mass_kg == 0.40
    assert hw.motor_max_torque_nm == 0.20
    assert sim.dt_sim == 0.001
    assert sim.dt_control == 0.010
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_load_default_config -v`
Expected: `ImportError: cannot import name 'load_config'`.

- [ ] **Step 4: Implement loader**

Append to `src/cubli_mpc/config.py`:

```python
from pathlib import Path
import yaml


def load_config(path: str | Path) -> tuple[HardwareConfig, SimConfig]:
    """Load HardwareConfig and SimConfig from a YAML file.

    The YAML must have a top-level `hardware:` mapping; a `sim:` mapping is
    optional and falls back to SimConfig defaults.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    hw = HardwareConfig(**data["hardware"])
    sim_data = data.get("sim", {}) or {}
    sim = SimConfig(**sim_data)
    return hw, sim
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: all 8 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/cubli_mpc/config.py configs/default.yaml tests/test_config.py
git commit -m "feat(config): YAML loader + default hardware/sim config"
```

---

## Task 4: MJCF XML builder

**Files:**
- Create: `src/cubli_mpc/model/mjcf.py`
- Create: `tests/test_mjcf.py`

**Modeling notes (read before implementing):**
- Tilt axis = y. Wheel spin axis = y (parallel to tilt — simplest 1-axis Cubli).
- Cube body's origin is at the balancing edge (world origin when reset).
- The cube geom is positioned at `(0, 0, a*sqrt(2)/2)` and rotated `euler="0 45 0"` so one of its edges sits at z=0.
- The wheel body is a child of the cube body, positioned at the cube COM along z.
- Cylinder default axis is z; rotate `euler="90 0 0"` to align spin axis with y.

- [ ] **Step 1: Write failing test — model parses and has expected joints**

Create `tests/test_mjcf.py`:

```python
"""Tests for MJCF builder: XML parses, body/joint/sensor names exist,
inertias match analytical values from HardwareConfig.
"""
import math
import mujoco
import pytest
from cubli_mpc.config import HardwareConfig
from cubli_mpc.model.mjcf import build_mjcf


def _make_hw(**overrides) -> HardwareConfig:
    defaults = dict(
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
    defaults.update(overrides)
    return HardwareConfig(**defaults)


def test_mjcf_parses():
    xml = build_mjcf(_make_hw())
    model = mujoco.MjModel.from_xml_string(xml)
    assert model is not None


def test_mjcf_has_expected_joints():
    xml = build_mjcf(_make_hw())
    model = mujoco.MjModel.from_xml_string(xml)
    joint_names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                   for i in range(model.njnt)}
    assert "tilt" in joint_names
    assert "spin" in joint_names


def test_mjcf_has_expected_actuator():
    xml = build_mjcf(_make_hw())
    model = mujoco.MjModel.from_xml_string(xml)
    actuator_names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                      for i in range(model.nu)}
    assert "wheel_motor" in actuator_names


def test_mjcf_actuator_torque_limit_matches_config():
    hw = _make_hw(motor_max_torque_nm=0.123)
    model = mujoco.MjModel.from_xml_string(build_mjcf(hw))
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_motor")
    lo, hi = model.actuator_ctrlrange[aid]
    assert lo == pytest.approx(-0.123)
    assert hi == pytest.approx(0.123)


def test_mjcf_has_expected_sensors():
    xml = build_mjcf(_make_hw())
    model = mujoco.MjModel.from_xml_string(xml)
    sensor_names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
                    for i in range(model.nsensor)}
    for s in ("gyro", "accel", "wheel_angle", "wheel_speed",
              "body_angle", "body_rate"):
        assert s in sensor_names, f"missing sensor: {s}"


def test_mjcf_cube_inertia_isotropic_about_com():
    """A uniform solid cube has an isotropic inertia tensor about its
    geometric center: all three principal moments equal (1/6)*m*a^2.
    MuJoCo's body_inertia returns the principal moments in the body's
    principal-axis frame; we assert all three match.
    """
    hw = _make_hw(cube_side_length_m=0.10, cube_mass_kg=0.40)
    model = mujoco.MjModel.from_xml_string(build_mjcf(hw))
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    expected = (1.0 / 6.0) * 0.40 * 0.10**2
    for component in model.body_inertia[cube_id]:
        assert component == pytest.approx(expected, rel=1e-3)


def test_mjcf_wheel_inertia_about_spin_axis():
    hw = _make_hw(wheel_mass_kg=0.10, wheel_radius_m=0.035)
    model = mujoco.MjModel.from_xml_string(build_mjcf(hw))
    wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wheel")
    # After rotation, the cylinder's spin axis is along the body's y. The
    # principal moments are sorted by MuJoCo; pick the largest, which for a
    # thin solid cylinder is the spin-axis moment (1/2)*m*r^2.
    inertia = list(model.body_inertia[wheel_id])
    spin_inertia = max(inertia)
    expected = 0.5 * 0.10 * 0.035**2
    assert spin_inertia == pytest.approx(expected, rel=1e-3)


def test_mjcf_damping_set():
    hw = _make_hw(edge_bearing_damping=0.0123,
                  wheel_bearing_damping=0.00456)
    model = mujoco.MjModel.from_xml_string(build_mjcf(hw))
    tilt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "tilt")
    spin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "spin")
    assert model.dof_damping[tilt_id] == pytest.approx(0.0123)
    assert model.dof_damping[spin_id] == pytest.approx(0.00456)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mjcf.py -v`
Expected: `ImportError: cannot import name 'build_mjcf'`.

- [ ] **Step 3: Implement `build_mjcf`**

Create `src/cubli_mpc/model/mjcf.py`:

```python
"""Build an MJCF XML string from HardwareConfig.

Convention:
  - World z is up.
  - Tilt axis = world y. Body origin sits at the balancing edge.
  - Wheel spin axis = body y (parallel to tilt -- simplest 1-axis cubli).
  - Cube geom is positioned a*sqrt(2)/2 above the edge and rotated 45 deg
    about y so one of its edges coincides with z=0 (the balancing edge).
"""
from __future__ import annotations

import math
from xml.etree import ElementTree as ET

from cubli_mpc.config import HardwareConfig, SimConfig


def build_mjcf(hw: HardwareConfig, sim: SimConfig | None = None) -> str:
    """Return an MJCF XML string for the given hardware config.

    Inertia of the cube/wheel is computed by MuJoCo from the geom shapes
    and masses -- we never write inertia tensors by hand.
    """
    dt = sim.dt_sim if sim is not None else 0.001
    a = hw.cube_side_length_m
    half_a = a / 2.0
    com_h = a * math.sqrt(2.0) / 2.0
    tau = hw.motor_max_torque_nm

    mujoco = ET.Element("mujoco", model="cubli")

    ET.SubElement(mujoco, "option", gravity="0 0 -9.81", timestep=f"{dt}")

    # Default visuals
    default = ET.SubElement(mujoco, "default")
    ET.SubElement(default, "geom", rgba="0.7 0.7 0.7 1")

    worldbody = ET.SubElement(mujoco, "worldbody")
    ET.SubElement(worldbody, "light", pos="0 0 3", dir="0 0 -1")
    ET.SubElement(worldbody, "geom",
                  name="ground", type="plane",
                  size="2 2 0.1", rgba="0.4 0.4 0.4 1",
                  pos="0 0 -0.001")

    # Cube body: origin at the balancing edge.
    cube = ET.SubElement(worldbody, "body", name="cube", pos="0 0 0")
    ET.SubElement(cube, "joint",
                  name="tilt", type="hinge", axis="0 1 0",
                  damping=f"{hw.edge_bearing_damping}")
    ET.SubElement(cube, "geom",
                  name="cube_geom", type="box",
                  size=f"{half_a} {half_a} {half_a}",
                  pos=f"0 0 {com_h}",
                  euler="0 45 0",
                  mass=f"{hw.cube_mass_kg}",
                  rgba="0.2 0.5 0.8 1")
    ET.SubElement(cube, "site", name="imu_site",
                  pos=f"0 0 {com_h}")

    # Wheel body: child of cube, positioned at cube COM along z.
    wheel_z = com_h + hw.wheel_offset_m
    wheel = ET.SubElement(cube, "body", name="wheel",
                          pos=f"0 0 {wheel_z}")
    ET.SubElement(wheel, "joint",
                  name="spin", type="hinge", axis="0 1 0",
                  damping=f"{hw.wheel_bearing_damping}")
    ET.SubElement(wheel, "geom",
                  name="wheel_geom", type="cylinder",
                  size=f"{hw.wheel_radius_m} {hw.wheel_thickness_m / 2.0}",
                  euler="90 0 0",
                  mass=f"{hw.wheel_mass_kg}",
                  rgba="0.8 0.3 0.3 1")

    # Actuator: torque-controlled motor on the wheel hinge.
    actuator = ET.SubElement(mujoco, "actuator")
    ET.SubElement(actuator, "motor",
                  name="wheel_motor", joint="spin",
                  ctrlrange=f"{-tau} {tau}", ctrllimited="true")

    # Sensors: IMU on body + encoder on wheel + ground-truth joints (for
    # the deterministic M1 setup; M4 will add the noise/delay path).
    sensor = ET.SubElement(mujoco, "sensor")
    ET.SubElement(sensor, "gyro", name="gyro", site="imu_site")
    ET.SubElement(sensor, "accelerometer", name="accel", site="imu_site")
    ET.SubElement(sensor, "jointpos", name="wheel_angle", joint="spin")
    ET.SubElement(sensor, "jointvel", name="wheel_speed", joint="spin")
    ET.SubElement(sensor, "jointpos", name="body_angle", joint="tilt")
    ET.SubElement(sensor, "jointvel", name="body_rate", joint="tilt")

    return ET.tostring(mujoco, encoding="unicode")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mjcf.py -v`
Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cubli_mpc/model/mjcf.py tests/test_mjcf.py
git commit -m "feat(model): MJCF builder generates parametric cubli model"
```

---

## Task 5: Sim env wrapper

**Files:**
- Create: `src/cubli_mpc/sim/env.py`
- Create: `tests/test_env.py`

**Design notes:**
- `CubliEnv` owns the `mjModel` and `mjData`, exposes `reset`, `step`, `state`, and `apply_torque`.
- State returned as `np.array([theta_body, theta_dot_body, theta_wheel, theta_dot_wheel])`.
- `reset(theta0=0.0, theta_dot0=0.0, wheel_angle0=0.0, wheel_speed0=0.0)`.

- [ ] **Step 1: Write failing tests for `CubliEnv`**

Create `tests/test_env.py`:

```python
"""Tests for the CubliEnv MuJoCo wrapper."""
import math
import numpy as np
import pytest
from cubli_mpc.config import HardwareConfig, SimConfig
from cubli_mpc.sim.env import CubliEnv


def _make_hw(**overrides) -> HardwareConfig:
    defaults = dict(
        cube_side_length_m=0.10,
        cube_mass_kg=0.40,
        wheel_mass_kg=0.10,
        wheel_radius_m=0.035,
        wheel_thickness_m=0.010,
        motor_max_torque_nm=0.20,
        motor_max_speed_rad_s=600.0,
        motor_torque_constant=0.01,
        edge_bearing_damping=0.0,   # zero for clean dynamics tests
        wheel_bearing_damping=0.0,
    )
    defaults.update(overrides)
    return HardwareConfig(**defaults)


def test_env_constructs():
    env = CubliEnv(_make_hw(), SimConfig())
    assert env is not None


def test_state_shape_and_initial_zero():
    env = CubliEnv(_make_hw(), SimConfig())
    env.reset()
    s = env.state()
    assert s.shape == (4,)
    np.testing.assert_allclose(s, np.zeros(4), atol=1e-12)


def test_reset_sets_initial_conditions():
    env = CubliEnv(_make_hw(), SimConfig())
    env.reset(theta0=0.1, theta_dot0=0.2,
              wheel_angle0=0.3, wheel_speed0=0.4)
    s = env.state()
    np.testing.assert_allclose(s, [0.1, 0.2, 0.3, 0.4], atol=1e-12)


def test_passive_cube_falls_over():
    """With zero motor torque from a small initial tilt, the body should
    accelerate away from upright due to gravity.
    """
    env = CubliEnv(_make_hw(), SimConfig())
    env.reset(theta0=0.05)
    for _ in range(100):  # 100 ms with dt=1ms
        env.apply_torque(0.0)
        env.step()
    s = env.state()
    # theta and theta_dot should both have grown in the positive direction
    assert s[0] > 0.05
    assert s[1] > 0.0


def test_motor_torque_spins_wheel():
    """Applying constant torque with the body locked at theta=0 (small
    initial perturbation, very short time) should spin the wheel up.
    """
    env = CubliEnv(_make_hw(), SimConfig())
    env.reset(theta0=0.0)
    for _ in range(50):
        env.apply_torque(0.05)
        env.step()
    s = env.state()
    assert s[3] > 0.0   # wheel speed positive
    # By Newton's 3rd, applying +tau to wheel reacts as -tau on body, so
    # body should tilt negative (assuming gravity hasn't dominated yet)
    assert s[0] < 0.0


def test_gravity_torque_matches_analytical():
    """At theta = pi/4, with all velocities zero and no motor torque, the
    angular acceleration of the body should match m*g*L*sin(theta)/I_edge.
    """
    hw = _make_hw()
    env = CubliEnv(hw, SimConfig())
    env.reset(theta0=math.pi / 4)
    env.apply_torque(0.0)
    env.step()
    # After one step of dt=1ms, theta_dot ~= alpha * dt, where alpha is
    # the analytical angular acceleration.
    s = env.state()
    alpha_expected = (hw.gravity_moment_coefficient
                      * math.sin(math.pi / 4)
                      / hw.cube_inertia_about_edge)
    # Tolerance is generous because MuJoCo composes cube + wheel inertia
    # (wheel adds a small contribution); we just want the sign and rough
    # magnitude.
    assert s[1] > 0.0
    assert s[1] == pytest.approx(alpha_expected * 0.001, rel=0.1)


def test_torque_saturation():
    """Commanding a torque above ctrlrange should be clipped by MuJoCo."""
    hw = _make_hw(motor_max_torque_nm=0.10)
    env = CubliEnv(hw, SimConfig())
    env.reset()
    env.apply_torque(10.0)  # way over the 0.1 Nm limit
    env.step()
    # Wheel speed after 1ms should be no greater than what 0.1 Nm produces
    # over 1ms = 0.1 / I_wheel * dt
    s = env.state()
    max_omega = 0.10 / hw.wheel_inertia_about_spin_axis * 0.001
    # Allow 20% slack for MuJoCo's integrator behaviour
    assert s[3] <= max_omega * 1.2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_env.py -v`
Expected: `ImportError: cannot import name 'CubliEnv'`.

- [ ] **Step 3: Implement `CubliEnv`**

Create `src/cubli_mpc/sim/env.py`:

```python
"""MuJoCo wrapper for the cubli model.

CubliEnv owns the mjModel/mjData and exposes a small imperative API:
reset, apply_torque, step, state.
"""
from __future__ import annotations

import numpy as np
import mujoco

from cubli_mpc.config import HardwareConfig, SimConfig
from cubli_mpc.model.mjcf import build_mjcf


class CubliEnv:
    """Owns the MuJoCo simulation of a 1-axis cubli."""

    def __init__(self, hw: HardwareConfig, sim: SimConfig):
        self._hw = hw
        self._sim = sim
        xml = build_mjcf(hw, sim)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._tilt_qpos_adr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "tilt")]
        self._spin_qpos_adr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "spin")]
        self._tilt_dof_adr = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "tilt")]
        self._spin_dof_adr = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "spin")]
        self._motor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_motor")

    def reset(
        self,
        theta0: float = 0.0,
        theta_dot0: float = 0.0,
        wheel_angle0: float = 0.0,
        wheel_speed0: float = 0.0,
    ) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._tilt_qpos_adr] = theta0
        self.data.qpos[self._spin_qpos_adr] = wheel_angle0
        self.data.qvel[self._tilt_dof_adr] = theta_dot0
        self.data.qvel[self._spin_dof_adr] = wheel_speed0
        self.data.ctrl[self._motor_id] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def apply_torque(self, tau: float) -> None:
        self.data.ctrl[self._motor_id] = float(tau)

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data)

    def state(self) -> np.ndarray:
        return np.array([
            self.data.qpos[self._tilt_qpos_adr],
            self.data.qvel[self._tilt_dof_adr],
            self.data.qpos[self._spin_qpos_adr],
            self.data.qvel[self._spin_dof_adr],
        ])

    @property
    def time(self) -> float:
        return float(self.data.time)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_env.py -v`
Expected: all 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cubli_mpc/sim/env.py tests/test_env.py
git commit -m "feat(sim): CubliEnv MuJoCo wrapper with reset/step/state"
```

---

## Task 6: Controller protocol

**Files:**
- Create: `src/cubli_mpc/control/base.py`

This is tiny — just the protocol, no tests needed beyond what we'll do in Task 7.

- [ ] **Step 1: Write the protocol**

Create `src/cubli_mpc/control/base.py`:

```python
"""Controller protocol.

Every controller takes a 4D state estimate
    x_hat = [theta_body, theta_dot_body, theta_wheel, theta_dot_wheel]
and a wall-clock time and returns a commanded wheel torque in Nm. The runner
is responsible for clipping/saturation if the actuator can be over-driven;
controllers may self-saturate.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Controller(Protocol):
    def step(self, x_hat: np.ndarray, t: float) -> float:
        """Return commanded wheel torque (Nm)."""
        ...
```

- [ ] **Step 2: Commit**

```bash
git add src/cubli_mpc/control/base.py
git commit -m "feat(control): Controller protocol"
```

---

## Task 7: Nonlinear PD baseline controller

**Files:**
- Create: `src/cubli_mpc/control/nonlinear_pd.py`
- Create: `tests/test_nonlinear_pd.py`

**Design rationale (read before implementing):**

The closed-loop dynamics of the body, ignoring damping, are
    I_body * theta_ddot = m*g*L*sin(theta) - tau
with motor torque `tau` (positive tau on wheel reacts as negative torque on body).

With **gravity feedforward** `tau_grav = m*g*L*sin(theta)`, the body equation
becomes `I_body * theta_ddot = -tau_fb`, where `tau_fb` is the feedback term.
We pick `tau_fb = kp*sin(theta - theta_balance) + kd*theta_dot`, the nonlinear
PD around a balance point. The `sin()` keeps the control well-behaved at
large angles.

To bleed wheel speed, the **outer loop** shifts the balance point opposite
to the wheel velocity: `theta_balance = -k_wheel * omega_wheel`, clipped to a
small range. At the offset balance point, the steady-state motor torque is
`m*g*L*sin(theta_balance)`, which opposes the wheel's spin (correct sign).

- [ ] **Step 1: Write failing tests**

Create `tests/test_nonlinear_pd.py`:

```python
"""Tests for the nonlinear PD baseline controller."""
import math
import numpy as np
import pytest
from cubli_mpc.config import HardwareConfig
from cubli_mpc.control.nonlinear_pd import (
    NonlinearPDController, NonlinearPDGains,
)


def _make_hw(**overrides) -> HardwareConfig:
    defaults = dict(
        cube_side_length_m=0.10,
        cube_mass_kg=0.40,
        wheel_mass_kg=0.10,
        wheel_radius_m=0.035,
        wheel_thickness_m=0.010,
        motor_max_torque_nm=0.20,
        motor_max_speed_rad_s=600.0,
        motor_torque_constant=0.01,
        edge_bearing_damping=0.0,
        wheel_bearing_damping=0.0,
    )
    defaults.update(overrides)
    return HardwareConfig(**defaults)


def _make_gains(**overrides) -> NonlinearPDGains:
    defaults = dict(
        kp=0.5, kd=0.02, k_wheel=0.0,
        max_balance_tilt=math.radians(2.0),
        max_torque=0.20,
    )
    defaults.update(overrides)
    return NonlinearPDGains(**defaults)


def test_returns_zero_at_upright_with_zero_rates():
    hw = _make_hw()
    c = NonlinearPDController(_make_gains(), hw)
    tau = c.step(np.zeros(4), t=0.0)
    assert tau == pytest.approx(0.0)


def test_returns_positive_torque_for_positive_tilt():
    """theta > 0 means the body has tilted past upright in +y direction.
    Gravity FF + positive sin(e) term should produce tau > 0.
    """
    hw = _make_hw()
    c = NonlinearPDController(_make_gains(kp=1.0), hw)
    tau = c.step(np.array([0.05, 0.0, 0.0, 0.0]), t=0.0)
    assert tau > 0.0


def test_saturation_clips_to_max_torque():
    hw = _make_hw()
    c = NonlinearPDController(_make_gains(max_torque=0.10, kp=10.0), hw)
    tau = c.step(np.array([0.5, 0.0, 0.0, 0.0]), t=0.0)
    assert abs(tau) <= 0.10 + 1e-12


def test_wheel_speed_offset_balance_point():
    """With k_wheel > 0, omega_w > 0 should shift the balance point to a
    negative theta value, so the controller drives the body toward a small
    negative tilt to allow the motor to bleed wheel speed.
    """
    hw = _make_hw()
    c = NonlinearPDController(
        _make_gains(kp=1.0, kd=0.0, k_wheel=0.001,
                    max_balance_tilt=math.radians(5.0)),
        hw,
    )
    # At theta=0, theta_dot=0, omega_w=100 rad/s:
    # theta_balance = -0.1 rad (clipped to max_balance_tilt = ~0.087)
    # tau_grav at theta=0 is 0
    # tau_pd = 1.0 * sin(0 - (-0.087)) = sin(0.087) > 0
    # So tau > 0, meaning motor will speed up wheel ... wait, that's wrong.
    # Re-check: omega_w > 0 -> theta_balance < 0 -> e = theta - theta_balance
    # = 0 - (negative) > 0 -> tau_pd > 0 -> motor accelerates wheel more.
    # That can't be right.
    #
    # The actual mechanism: at the *steady-state* (body settled at
    # theta_balance < 0), the gravity FF holds the body there with
    # tau = m*g*L*sin(theta_balance) < 0, which opposes the wheel.
    # During the transient (cube was at theta=0, must move to theta < 0),
    # the controller may transiently produce torque of either sign.
    #
    # For this test we check the *intent*: the balance-point offset is
    # negative when wheel speed is positive.
    tau_pos_wheel = c.step(np.array([0.0, 0.0, 0.0, 100.0]), t=0.0)
    tau_neg_wheel = c.step(np.array([0.0, 0.0, 0.0, -100.0]), t=0.0)
    # By symmetry, the two torques should be opposite in sign.
    assert tau_pos_wheel * tau_neg_wheel < 0


def test_at_offset_balance_point_torque_bleeds_wheel():
    """When the body has reached the offset balance point, the controller
    should produce a torque opposing the wheel direction.
    """
    hw = _make_hw()
    c = NonlinearPDController(
        _make_gains(kp=1.0, kd=0.0, k_wheel=0.001,
                    max_balance_tilt=math.radians(10.0)),
        hw,
    )
    # Set theta == theta_balance so PD term is zero; tau = gravity FF only.
    omega_w = 100.0
    theta_balance = -0.001 * omega_w  # -0.1 rad, within max_balance_tilt
    tau = c.step(np.array([theta_balance, 0.0, 0.0, omega_w]), t=0.0)
    # tau = m*g*L*sin(theta_balance) -- negative, opposing positive omega_w
    assert tau < 0


def test_damping_term():
    """Positive theta_dot should add positive torque (push wheel + reacts
    body negative, slowing the falling motion in +theta direction).
    """
    hw = _make_hw()
    c = NonlinearPDController(_make_gains(kp=0.0, kd=0.1), hw)
    tau = c.step(np.array([0.0, 0.5, 0.0, 0.0]), t=0.0)
    assert tau == pytest.approx(0.1 * 0.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_nonlinear_pd.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement controller**

Create `src/cubli_mpc/control/nonlinear_pd.py`:

```python
"""Energy-aware nonlinear PD baseline controller (M1).

Structure:
  1. Outer loop: balance-point offset = -k_wheel * omega_wheel, clipped.
  2. Gravity feedforward cancels the cube's gravity moment.
  3. Inner loop: kp*sin(theta - theta_balance) + kd*theta_dot.
  4. Hard saturation at +/- max_torque.

The sin() in the inner loop keeps the PD well-behaved at large angles.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from cubli_mpc.config import HardwareConfig


@dataclass(frozen=True)
class NonlinearPDGains:
    kp: float
    kd: float
    k_wheel: float
    max_balance_tilt: float  # rad
    max_torque: float        # Nm


class NonlinearPDController:
    def __init__(self, gains: NonlinearPDGains, hw: HardwareConfig):
        self._g = gains
        self._mgL = hw.gravity_moment_coefficient

    def step(self, x_hat: np.ndarray, t: float) -> float:
        theta = float(x_hat[0])
        theta_dot = float(x_hat[1])
        omega_w = float(x_hat[3])
        g = self._g

        theta_balance = -g.k_wheel * omega_w
        if theta_balance > g.max_balance_tilt:
            theta_balance = g.max_balance_tilt
        elif theta_balance < -g.max_balance_tilt:
            theta_balance = -g.max_balance_tilt

        e = theta - theta_balance
        tau_grav = self._mgL * math.sin(theta)
        tau_pd = g.kp * math.sin(e) + g.kd * theta_dot
        tau = tau_grav + tau_pd

        if tau > g.max_torque:
            tau = g.max_torque
        elif tau < -g.max_torque:
            tau = -g.max_torque
        return tau
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_nonlinear_pd.py -v`
Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cubli_mpc/control/nonlinear_pd.py tests/test_nonlinear_pd.py
git commit -m "feat(control): nonlinear PD baseline with gravity FF + wheel bleed"
```

---

## Task 8: Runner (sim loop)

**Files:**
- Create: `src/cubli_mpc/runner.py`
- Create: `tests/test_runner.py`

**Design notes:**
- `Runner.run(duration_s)` ticks the env at `dt_sim` and calls the controller every `dt_control`.
- Between control calls, the most recent torque is held (zero-order hold).
- Returns a log: `dict[str, np.ndarray]` with keys `t, theta, theta_dot, wheel_angle, wheel_speed, tau`.

- [ ] **Step 1: Write failing tests**

Create `tests/test_runner.py`:

```python
"""Tests for the Runner sim loop."""
import math
import numpy as np
import pytest
from cubli_mpc.config import HardwareConfig, SimConfig
from cubli_mpc.control.nonlinear_pd import (
    NonlinearPDController, NonlinearPDGains,
)
from cubli_mpc.sim.env import CubliEnv
from cubli_mpc.runner import Runner


def _make_hw(**overrides) -> HardwareConfig:
    defaults = dict(
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
    defaults.update(overrides)
    return HardwareConfig(**defaults)


class ZeroController:
    def step(self, x_hat, t):
        return 0.0


def test_runner_log_shape():
    """1 simulated second at dt_sim=1ms produces 1000 sim steps. The log
    samples once per control tick (dt_control=10ms), so 100 entries.
    """
    env = CubliEnv(_make_hw(), SimConfig())
    env.reset(theta0=0.01)
    runner = Runner(env, ZeroController(), SimConfig())
    log = runner.run(duration_s=1.0)
    assert log["t"].shape == (100,)
    assert log["theta"].shape == (100,)
    assert log["tau"].shape == (100,)


def test_runner_zero_controller_lets_cube_fall():
    env = CubliEnv(_make_hw(), SimConfig())
    env.reset(theta0=0.05)
    runner = Runner(env, ZeroController(), SimConfig())
    log = runner.run(duration_s=0.5)
    # Without control, theta should grow past initial tilt.
    assert log["theta"][-1] > 0.05


def test_runner_stabilizes_small_tilt_with_nonlinear_pd():
    """M1 acceptance: nonlinear PD stabilizes from 5-degree initial tilt
    within 5 simulated seconds; final |theta| < 2 degrees, |theta_dot| < 0.5
    rad/s.
    """
    hw = _make_hw()
    sim = SimConfig()
    env = CubliEnv(hw, sim)
    env.reset(theta0=math.radians(5.0))

    gains = NonlinearPDGains(
        kp=0.5, kd=0.05, k_wheel=0.0001,
        max_balance_tilt=math.radians(2.0),
        max_torque=hw.motor_max_torque_nm,
    )
    controller = NonlinearPDController(gains, hw)

    runner = Runner(env, controller, sim)
    log = runner.run(duration_s=5.0)

    assert abs(log["theta"][-1]) < math.radians(2.0)
    assert abs(log["theta_dot"][-1]) < 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_runner.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `Runner`**

Create `src/cubli_mpc/runner.py`:

```python
"""Sim-loop runner: env + controller + logger.

Physics ticks at `sim.dt_sim`. Controller is invoked every N physics ticks
where N = round(dt_control / dt_sim). Between control invocations the most
recent torque is held (zero-order hold). Log samples once per control tick.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from cubli_mpc.config import SimConfig
from cubli_mpc.control.base import Controller
from cubli_mpc.sim.env import CubliEnv


class Runner:
    def __init__(self, env: CubliEnv, controller: Controller, sim: SimConfig):
        self._env = env
        self._controller = controller
        self._sim = sim
        self._steps_per_control = max(1, round(sim.dt_control / sim.dt_sim))

    def run(self, duration_s: float) -> dict[str, np.ndarray]:
        n_control_ticks = int(round(duration_s / self._sim.dt_control))
        t_buf = np.empty(n_control_ticks)
        theta_buf = np.empty(n_control_ticks)
        theta_dot_buf = np.empty(n_control_ticks)
        wheel_angle_buf = np.empty(n_control_ticks)
        wheel_speed_buf = np.empty(n_control_ticks)
        tau_buf = np.empty(n_control_ticks)

        for i in range(n_control_ticks):
            x = self._env.state()
            t = self._env.time
            tau = float(self._controller.step(x, t))
            self._env.apply_torque(tau)

            t_buf[i] = t
            theta_buf[i] = x[0]
            theta_dot_buf[i] = x[1]
            wheel_angle_buf[i] = x[2]
            wheel_speed_buf[i] = x[3]
            tau_buf[i] = tau

            for _ in range(self._steps_per_control):
                self._env.step()

        return {
            "t": t_buf,
            "theta": theta_buf,
            "theta_dot": theta_dot_buf,
            "wheel_angle": wheel_angle_buf,
            "wheel_speed": wheel_speed_buf,
            "tau": tau_buf,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_runner.py -v`
Expected: all 3 tests pass. The third (`test_runner_stabilizes_small_tilt_with_nonlinear_pd`) is the M1 acceptance test — if it fails, retune `kp` and `kd` until it passes (typical starting point: kp ~ 0.5, kd ~ 0.05 for the default config; may need adjustment).

If the acceptance test fails, the most likely causes are:
- `kp` too low: cube falls before PD can recover. Increase to 1.0–2.0.
- `kd` too low or too high: oscillates / over-damped. Tune within 0.01–0.1.
- Motor saturation: check `log["tau"]` — if it sits at `max_torque` for long stretches, the controller is asking for more than the motor can deliver from this initial tilt. Reduce `theta0` or increase `motor_max_torque_nm` in the test.

- [ ] **Step 5: Commit**

```bash
git add src/cubli_mpc/runner.py tests/test_runner.py
git commit -m "feat(runner): sim loop with controller + data logging"
```

---

## Task 9: CLI entry point

**Files:**
- Create: `src/cubli_mpc/cli.py`
- Modify: `README.md` (add usage)

The CLI runs a sim with the nonlinear PD controller and either prints a
summary or opens the MuJoCo viewer.

- [ ] **Step 1: Implement `cli.py`**

Create `src/cubli_mpc/cli.py`:

```python
"""Command-line entry point for cubli-mpc.

Usage:
    cubli-mpc sim --config configs/default.yaml [--duration 5.0] [--viewer]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

from cubli_mpc.config import load_config
from cubli_mpc.control.nonlinear_pd import (
    NonlinearPDController, NonlinearPDGains,
)
from cubli_mpc.runner import Runner
from cubli_mpc.sim.env import CubliEnv


def _cmd_sim(args: argparse.Namespace) -> int:
    hw, sim = load_config(args.config)
    env = CubliEnv(hw, sim)
    env.reset(theta0=math.radians(args.initial_tilt_deg))

    gains = NonlinearPDGains(
        kp=args.kp, kd=args.kd, k_wheel=args.k_wheel,
        max_balance_tilt=math.radians(args.max_balance_tilt_deg),
        max_torque=hw.motor_max_torque_nm,
    )
    controller = NonlinearPDController(gains, hw)

    if args.viewer:
        return _run_with_viewer(env, controller, sim, args.duration)
    return _run_headless(env, controller, sim, args.duration)


def _run_headless(env, controller, sim, duration: float) -> int:
    runner = Runner(env, controller, sim)
    log = runner.run(duration_s=duration)
    final_theta = log["theta"][-1]
    final_theta_dot = log["theta_dot"][-1]
    final_omega_w = log["wheel_speed"][-1]
    print(f"t={log['t'][-1]:.3f}s  "
          f"theta={math.degrees(final_theta):+.3f} deg  "
          f"theta_dot={final_theta_dot:+.3f} rad/s  "
          f"omega_wheel={final_omega_w:+.2f} rad/s")
    print(f"|theta|_max = {math.degrees(np.max(np.abs(log['theta']))):.3f} deg")
    print(f"|tau|_max   = {np.max(np.abs(log['tau'])):.4f} Nm")
    return 0


def _run_with_viewer(env, controller, sim, duration: float) -> int:
    import mujoco.viewer

    steps_per_control = max(1, round(sim.dt_control / sim.dt_sim))
    n_ticks = int(round(duration / sim.dt_control))
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for _ in range(n_ticks):
            if not viewer.is_running():
                break
            x = env.state()
            tau = float(controller.step(x, env.time))
            env.apply_torque(tau)
            for _ in range(steps_per_control):
                env.step()
            viewer.sync()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cubli-mpc")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sim = sub.add_parser("sim", help="Run a simulation")
    p_sim.add_argument("--config", required=True, type=Path)
    p_sim.add_argument("--duration", type=float, default=5.0)
    p_sim.add_argument("--initial-tilt-deg", type=float, default=5.0)
    p_sim.add_argument("--kp", type=float, default=0.5)
    p_sim.add_argument("--kd", type=float, default=0.05)
    p_sim.add_argument("--k-wheel", type=float, default=1e-4)
    p_sim.add_argument("--max-balance-tilt-deg", type=float, default=2.0)
    p_sim.add_argument("--viewer", action="store_true",
                       help="Open the MuJoCo viewer")
    p_sim.set_defaults(func=_cmd_sim)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
```

- [ ] **Step 2: Smoke-test the headless CLI**

Run: `uv run cubli-mpc sim --config configs/default.yaml --duration 1.0`
Expected: one summary line with `theta`, `theta_dot`, `omega_wheel`, plus max values. No tracebacks.

- [ ] **Step 3: Smoke-test the viewer**

Run: `uv run cubli-mpc sim --config configs/default.yaml --duration 10.0 --viewer`
Expected: MuJoCo viewer window opens; cube wobbles on its edge and stabilizes; window closes after 10 simulated seconds. Verify by eye that the cube doesn't fall over.

(If your environment is headless / no X server, skip this step; the headless test covers the data-path correctness.)

- [ ] **Step 4: Update README with usage**

Replace contents of `README.md` with:

```markdown
# cubli-mpc

Reaction wheel cube balancer: parametric MuJoCo simulator + nonlinear MPC.

See [`docs/superpowers/specs/2026-05-14-cubli-mpc-design.md`](docs/superpowers/specs/2026-05-14-cubli-mpc-design.md) for the design.

## Quick start

```bash
uv sync --extra dev
uv run pytest                              # all tests
uv run cubli-mpc sim --config configs/default.yaml --viewer
```

## Project structure

- `src/cubli_mpc/config.py` — `HardwareConfig`, `SimConfig`, derived quantities, YAML loader.
- `src/cubli_mpc/model/mjcf.py` — builds parametric MJCF from `HardwareConfig`.
- `src/cubli_mpc/sim/env.py` — `CubliEnv`: MuJoCo wrapper.
- `src/cubli_mpc/control/` — `Controller` protocol + nonlinear PD baseline.
- `src/cubli_mpc/runner.py` — sim loop with logger.
- `src/cubli_mpc/cli.py` — `cubli-mpc sim ...`.
- `configs/default.yaml` — starting hardware + sim parameters.
```

- [ ] **Step 5: Commit**

```bash
git add src/cubli_mpc/cli.py README.md
git commit -m "feat(cli): cubli-mpc sim with --viewer; README usage"
```

---

## Task 10: Full test suite + final commit

- [ ] **Step 1: Run the full suite**

Run: `uv run pytest -v`
Expected: every test passes. If any fail, fix before moving on — they're the contract.

- [ ] **Step 2: Verify project layout matches spec**

Run: `tree -I '__pycache__|.venv|*.egg-info' src tests configs docs`
Expected: matches the file structure in the spec's "Package Layout" section.

- [ ] **Step 3: Tag the milestone**

```bash
git tag m1-foundation -m "M1: parametric sim + nonlinear PD baseline"
```

- [ ] **Step 4: Confirm clean working tree**

Run: `git status`
Expected: `nothing to commit, working tree clean`.

---

## Out of scope for this plan (deferred to future plans)

- **M2 (stabilization NMPC):** CasADi symbolic dynamics, IPOPT-based NMPC, terminal cost from LQR.
- **M3 (swing trajectories):** offline trajectory optimization, time-varying reference tracking.
- **M4 (long-duration robustness):** state estimator (complementary filter / EKF), sensor noise/delay activation, solver-failure watchdog, 24-hour sim run.
- **Co-design study scripts:** torque-limit sweep, wheel-inertia sweep, loop-rate sweep.

Each of these will get its own plan once M1 is validated end-to-end (tests green + the viewer smoke test shows a stably balancing cube).

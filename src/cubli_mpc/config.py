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

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

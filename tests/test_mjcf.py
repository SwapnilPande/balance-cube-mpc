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

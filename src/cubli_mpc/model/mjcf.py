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

    # Checker texture for the ground: makes the cube's tilt motion obvious.
    asset = ET.SubElement(mujoco, "asset")
    ET.SubElement(asset, "texture", name="grid_tex",
                  type="2d", builtin="checker",
                  rgb1="0.30 0.32 0.36", rgb2="0.55 0.58 0.62",
                  width="512", height="512")
    ET.SubElement(asset, "material", name="grid_mat",
                  texture="grid_tex", texrepeat="8 8",
                  reflectance="0.05")

    worldbody = ET.SubElement(mujoco, "worldbody")
    ET.SubElement(worldbody, "light", pos="0 0 3", dir="0 0 -1")
    ET.SubElement(worldbody, "geom",
                  name="ground", type="plane",
                  size="2 2 0.1", material="grid_mat",
                  pos="0 0 -0.001")

    # Cube body: origin at the balancing edge. Translucent so the internal
    # reaction wheel is visible.
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
                  rgba="0.2 0.5 0.8 0.25")
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
    # Asymmetric off-center stripe on the wheel face so spin angle (and
    # direction) is readable at a glance. Site -> visual-only, leaves the
    # wheel's inertia untouched.
    r = hw.wheel_radius_m
    half_t = hw.wheel_thickness_m / 2.0
    ET.SubElement(wheel, "site", name="wheel_marker",
                  type="box",
                  size=f"{r * 0.45} {half_t + 1e-4} {r * 0.10}",
                  pos=f"{r * 0.15} 0 0",
                  rgba="1.0 0.85 0.1 1")

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

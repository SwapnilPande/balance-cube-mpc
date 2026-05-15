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
        self.data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def apply_torque(self, tau: float) -> None:
        self.data.ctrl[self._motor_id] = float(tau)

    def apply_disturbance_torque(self, tau_ext: float) -> None:
        """Set an external torque on the tilt DOF. Persists across steps
        until overwritten; cleared on reset()."""
        self.data.qfrc_applied[self._tilt_dof_adr] = float(tau_ext)

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

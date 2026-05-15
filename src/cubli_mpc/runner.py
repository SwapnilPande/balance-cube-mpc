"""Sim-loop runner: env + controller + logger.

Physics ticks at `sim.dt_sim`. Controller is invoked every N physics ticks
where N = round(dt_control / dt_sim). Between control invocations the most
recent torque is held (zero-order hold). Log samples once per control tick.

Optional `disturbance` parameter injects an external torque on the tilt
DOF each control tick (zero-order hold between calls). The applied value
is logged alongside the state for plotting / metrics.
"""
from __future__ import annotations

import numpy as np

from cubli_mpc.config import SimConfig
from cubli_mpc.control.base import Controller
from cubli_mpc.sim.disturbance import DisturbancePlan
from cubli_mpc.sim.env import CubliEnv


class Runner:
    def __init__(
        self,
        env: CubliEnv,
        controller: Controller,
        sim: SimConfig,
        disturbance: DisturbancePlan | None = None,
    ):
        self._env = env
        self._controller = controller
        self._sim = sim
        self._disturbance = disturbance
        self._steps_per_control = max(1, round(sim.dt_control / sim.dt_sim))

    def run(self, duration_s: float) -> dict[str, np.ndarray]:
        n_control_ticks = int(round(duration_s / self._sim.dt_control))
        t_buf = np.empty(n_control_ticks)
        theta_buf = np.empty(n_control_ticks)
        theta_dot_buf = np.empty(n_control_ticks)
        wheel_angle_buf = np.empty(n_control_ticks)
        wheel_speed_buf = np.empty(n_control_ticks)
        tau_buf = np.empty(n_control_ticks)
        tau_ext_buf = np.zeros(n_control_ticks)

        for i in range(n_control_ticks):
            x = self._env.state()
            t = self._env.time
            tau = float(self._controller.step(x, t))
            self._env.apply_torque(tau)
            if self._disturbance is not None:
                tau_ext = self._disturbance.at(t, self._sim.dt_control)
                self._env.apply_disturbance_torque(tau_ext)
                tau_ext_buf[i] = tau_ext

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
            "tau_ext": tau_ext_buf,
        }

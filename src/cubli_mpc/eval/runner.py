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

# 45° rather than the originally-spec'd 60°: the MuJoCo cube face-contacts
# at ~45–48°, making 60° physically unreachable in simulation.
SURVIVAL_THRESHOLD_RAD = math.radians(45.0)


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

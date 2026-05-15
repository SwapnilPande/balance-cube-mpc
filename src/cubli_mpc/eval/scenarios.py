"""Named scenarios for the cubli-mpc eval harness.

Each `Scenario` is a reproducible test case: initial state, optional
disturbance plan, optional reference target. The eval runner consumes
these and emits a comparable metric dict.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

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

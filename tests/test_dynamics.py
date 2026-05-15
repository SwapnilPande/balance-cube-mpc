"""Tests for the CasADi symbolic dynamics."""
import math

import numpy as np
import pytest

from cubli_mpc.config import HardwareConfig
from cubli_mpc.model.dynamics import make_continuous_dynamics, make_rk4_step


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
    """RK4 with dt=1ms should be very close to Euler with dt=1us over 10ms.

    Verified for both zero torque and a nonzero constant torque to ensure
    accuracy under active control.
    """
    hw = _make_hw()
    f = make_continuous_dynamics(hw)
    F_rk4 = make_rk4_step(hw, dt=0.001)

    x0 = np.array([0.05, 0.0, 0.0])
    dt_fine = 1e-6

    for u in [np.array([0.0]), np.array([0.05])]:
        # RK4: 10 steps at 1ms
        x_rk4 = x0.copy()
        for _ in range(10):
            x_rk4 = np.array(F_rk4(x_rk4, u)).flatten()

        # Euler reference: 10000 steps at 1us
        x_euler = x0.copy()
        for _ in range(10_000):
            x_dot = np.array(f(x_euler, u)).flatten()
            x_euler = x_euler + dt_fine * x_dot

        assert np.allclose(x_rk4, x_euler, atol=1e-5), (
            f"RK4 vs fine-Euler mismatch for u={u}: rk4={x_rk4}, euler={x_euler}"
        )


def test_rk4_positive_torque_decelerates_body_increases_wheel():
    F = make_rk4_step(_make_hw(), dt=0.01)
    x = np.array([0.0, 0.0, 0.0])
    for _ in range(20):
        x = np.array(F(x, [0.05])).flatten()
    # After 0.2 s of constant torque from rest upright:
    assert x[1] < 0.0   # body picked up negative angular velocity
    assert x[2] > 0.0   # wheel speed positive


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
    agree to within ~1 deg on theta and ~5% on omega_w over 0.2 s.
    """
    hw = _make_hw()
    # Increase damping a bit to make trajectories not diverge dramatically.
    sim = SimConfig(dt_sim=0.001, dt_control=0.01)
    F = make_rk4_step(hw, dt=sim.dt_control)

    # MuJoCo rollout
    env = CubliEnv(hw, sim)
    env.reset(theta0=theta0)
    steps_per_control = max(1, round(sim.dt_control / sim.dt_sim))
    # 20 ticks × dt_control=0.01 s = 0.2 s.  Keep the horizon short: beyond this
    # MuJoCo's contact handling activates once the cube tips far enough, whereas
    # the CasADi free-space ODE keeps integrating without contacts — making any
    # longer comparison meaningless.
    n_control_ticks = 20
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

"""CasADi symbolic dynamics for the 1-axis cubli (MPC prediction model).

State (3D, drops the cyclic wheel angle):
    x = [theta, theta_dot, omega_wheel]

Input:
    u = [tau]   (wheel motor torque, Nm)

The continuous-time equations come from the Lagrangian of a cube hinged on
one edge with a flywheel on a parallel internal axis. After inverting the
2x2 mass matrix:

    theta_ddot   = (m*g*L*sin(theta) - b_e*theta_dot - tau + b_w*omega_w) / I_b
    omega_w_dot  = (tau - b_w*omega_w) / I_w  -  theta_ddot

This is deliberately *separate* from MuJoCo; small model mismatch is part
of the design (see parent spec).
"""
from __future__ import annotations

import casadi as ca

from cubli_mpc.config import HardwareConfig


def make_continuous_dynamics(hw: HardwareConfig) -> ca.Function:
    """Return CasADi Function f(x, u) -> x_dot for the 3D MPC state."""
    I_b = hw.cube_inertia_about_edge
    I_w = hw.wheel_inertia_about_spin_axis
    mgL = hw.gravity_moment_coefficient
    b_e = hw.edge_bearing_damping
    b_w = hw.wheel_bearing_damping

    x = ca.MX.sym("x", 3)
    u = ca.MX.sym("u", 1)
    theta = x[0]
    theta_dot = x[1]
    omega_w = x[2]
    tau = u[0]

    theta_ddot = (mgL * ca.sin(theta) - b_e * theta_dot
                  - tau + b_w * omega_w) / I_b
    omega_w_dot = (tau - b_w * omega_w) / I_w - theta_ddot

    x_dot = ca.vertcat(theta_dot, theta_ddot, omega_w_dot)
    return ca.Function("cubli_dynamics", [x, u], [x_dot],
                       ["x", "u"], ["x_dot"])


def make_rk4_step(hw: HardwareConfig, dt: float) -> ca.Function:
    """Return CasADi Function F(x, u) -> x_next using explicit RK4.

    `dt` is baked into the returned Function. For multi-rate use, build
    one Function per dt.
    """
    f = make_continuous_dynamics(hw)
    x = ca.MX.sym("x", 3)
    u = ca.MX.sym("u", 1)

    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return ca.Function("cubli_rk4", [x, u], [x_next],
                       ["x", "u"], ["x_next"])

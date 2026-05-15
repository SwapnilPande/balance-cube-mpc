"""Offline swing-trajectory planner for the M3 metronome task.

Solves a half-period direct multiple-shooting NLP between +theta_target
and -theta_target with both endpoints at rest. The dynamics symmetry
(theta, theta_dot, omega_w, tau) -> (-theta, -theta_dot, -omega_w, -tau)
lets us mirror the half period to a full one without re-solving.

Save format (npz):
    t      : (2N+1,)
    x_ref  : (2N+1, 3)  [theta, theta_dot, omega_w]
    u_ref  : (2N,   1)  [tau]
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import casadi as ca
import numpy as np

from cubli_mpc.config import HardwareConfig
from cubli_mpc.model.dynamics import make_rk4_step


@dataclass(frozen=True)
class SwingTrajConfig:
    period_s: float
    theta_target_rad: float
    n_segments: int = 200
    smoothness_weight: float = 1e-3  # on sum-of-squared torque rate
    save_path: Path | None = None


def plan_swing(hw: HardwareConfig, cfg: SwingTrajConfig) -> Path:
    """Solve the half-period swing trajopt and write the full mirrored
    trajectory to `cfg.save_path` (defaults to runs/swing.npz)."""
    save_path = (cfg.save_path if cfg.save_path is not None
                 else Path("runs/swing.npz"))
    save_path.parent.mkdir(parents=True, exist_ok=True)

    half_period = 0.5 * cfg.period_s
    N = cfg.n_segments
    dt = half_period / N
    F = make_rk4_step(hw, dt=dt)

    X = ca.MX.sym("X", 3, N + 1)
    U = ca.MX.sym("U", 1, N)

    g_list = []
    cost = ca.MX(0.0)
    for k in range(N):
        g_list.append(X[:, k + 1] - F(X[:, k], U[:, k]))
        cost = cost + U[:, k] * U[:, k] * dt
        if k > 0:
            du = U[:, k] - U[:, k - 1]
            cost = cost + cfg.smoothness_weight * du * du

    # Boundary conditions on theta and theta_dot.
    g_list.append(X[0, 0] - cfg.theta_target_rad)
    g_list.append(X[1, 0])
    g_list.append(X[2, 0])  # start with zero wheel speed for repeatability
    g_list.append(X[0, N] + cfg.theta_target_rad)
    g_list.append(X[1, N])

    g = ca.vertcat(*g_list)
    z = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

    # Box bounds: torque limited, wheel speed limited, theta unconstrained.
    z_lb = np.full(z.size1(), -np.inf)
    z_ub = np.full(z.size1(), +np.inf)
    omega_lim = hw.motor_max_speed_rad_s
    for k in range(N + 1):
        z_lb[3 * k + 2] = -omega_lim
        z_ub[3 * k + 2] = +omega_lim
    u_start = 3 * (N + 1)
    z_lb[u_start:] = -hw.motor_max_torque_nm
    z_ub[u_start:] = +hw.motor_max_torque_nm

    nlp = {"x": z, "f": cost, "g": g}
    opts = {
        "ipopt.max_iter": 500,
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "print_time": 0,
    }
    solver = ca.nlpsol("swing_planner", "ipopt", nlp, opts)

    # Initial guess: linear interpolation between endpoints, zero torque.
    x_init = np.zeros((3, N + 1))
    x_init[0, :] = np.linspace(cfg.theta_target_rad,
                               -cfg.theta_target_rad, N + 1)
    u_init = np.zeros((1, N))
    z0 = np.concatenate([x_init.reshape(-1, order="F"),
                         u_init.reshape(-1, order="F")])

    sol = solver(x0=z0, lbx=z_lb, ubx=z_ub,
                 lbg=np.zeros(g.size1()), ubg=np.zeros(g.size1()))
    if not solver.stats().get("success", False):
        raise RuntimeError(
            f"swing planner failed for period={cfg.period_s}, "
            f"theta_target={cfg.theta_target_rad}: solver did not converge"
        )

    z_opt = np.array(sol["x"]).flatten()
    X_opt = z_opt[: 3 * (N + 1)].reshape(3, N + 1, order="F").T  # (N+1, 3)
    U_opt = z_opt[3 * (N + 1):].reshape(1, N, order="F").T        # (N, 1)

    # Mirror to a full period using the dynamics symmetry
    #     (theta, theta_dot, omega_w, tau) -> -(theta, theta_dot, omega_w, tau).
    # If X_opt(t) is a solution from +theta_target to -theta_target under
    # U_opt(t), then -X_opt(t) is a solution from -theta_target to
    # +theta_target under -U_opt(t). The second half is -X_opt played
    # forward, dropping the first sample to avoid duplicating t=T/2.
    t_half = np.linspace(0.0, half_period, N + 1)
    t_full = np.concatenate([t_half, t_half[1:] + half_period])
    X_full = np.concatenate([X_opt, -X_opt[1:, :]], axis=0)
    U_full = np.concatenate([U_opt, -U_opt], axis=0)

    np.savez(save_path, t=t_full, x_ref=X_full, u_ref=U_full)
    return save_path

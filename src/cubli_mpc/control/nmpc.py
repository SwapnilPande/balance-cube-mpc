"""Nonlinear MPC controller for the 1-axis cubli.

Direct multiple-shooting NLP solved by IPOPT each tick. State `[theta,
theta_dot, omega_w]` (3D); the runner passes a 4D x_hat with the cyclic
wheel angle, which we drop. Action: scalar torque on the wheel.

This is the M2 / "stabilization NMPC" milestone from the parent spec.
Warm-start and solver-failure fallback land in subsequent tasks.
"""
from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np

from cubli_mpc.config import HardwareConfig
from cubli_mpc.control.references import ConstantReference, Reference
from cubli_mpc.model.dynamics import make_rk4_step


@dataclass(frozen=True)
class NMPCConfig:
    horizon_steps: int = 50
    dt: float = 0.010
    q_theta: float = 100.0
    q_theta_dot: float = 1.0
    q_omega_w: float = 1e-4
    r_torque: float = 0.01
    terminal_scale: float = 10.0
    omega_w_limit: float = float("inf")
    ipopt_max_iter: int = 50
    ipopt_print_level: int = 0


class NMPCController:
    def __init__(
        self,
        hw: HardwareConfig,
        cfg: NMPCConfig,
        reference: Reference | None = None,
    ):
        self._hw = hw
        self._cfg = cfg
        self._reference = reference if reference is not None else ConstantReference()
        self._tau_max = float(hw.motor_max_torque_nm)
        self._build_solver()

    @property
    def target_theta(self) -> float:
        return float(getattr(self._reference, "target_theta", 0.0))

    @property
    def reference(self) -> Reference:
        return self._reference

    def _build_solver(self) -> None:
        cfg = self._cfg
        N = cfg.horizon_steps
        F = make_rk4_step(self._hw, dt=cfg.dt)

        # Decision variables: X (3 x N+1), U (1 x N)
        X = ca.MX.sym("X", 3, N + 1)
        U = ca.MX.sym("U", 1, N)

        # Parameters: x0 (3), x_ref (3 x N+1), u_ref (1 x N)
        x0 = ca.MX.sym("x0", 3)
        x_ref = ca.MX.sym("x_ref", 3, N + 1)
        u_ref = ca.MX.sym("u_ref", 1, N)

        Q = ca.diag(ca.DM([cfg.q_theta, cfg.q_theta_dot, cfg.q_omega_w]))
        R = ca.DM(cfg.r_torque)

        g_list = [X[:, 0] - x0]  # initial-state constraint
        cost = ca.MX(0.0)
        for k in range(N):
            dx = X[:, k] - x_ref[:, k]
            du = U[:, k] - u_ref[:, k]
            cost = cost + ca.mtimes([dx.T, Q, dx]) + R * du * du
            g_list.append(X[:, k + 1] - F(X[:, k], U[:, k]))
        dx_N = X[:, N] - x_ref[:, N]
        cost = cost + cfg.terminal_scale * ca.mtimes([dx_N.T, Q, dx_N])

        g = ca.vertcat(*g_list)
        # All defect constraints == 0 (initial-state + N dynamics).
        self._g_lb = np.zeros(g.size1())
        self._g_ub = np.zeros(g.size1())

        # Decision-vector layout: flatten X then U (column-major to match
        # CasADi's MX reshape conventions).
        z = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        p = ca.vertcat(x0, ca.reshape(x_ref, -1, 1),
                       ca.reshape(u_ref, -1, 1))

        nlp = {"x": z, "p": p, "f": cost, "g": g}
        opts = {
            "ipopt.max_iter": cfg.ipopt_max_iter,
            "ipopt.print_level": cfg.ipopt_print_level,
            "ipopt.sb": "yes",
            "print_time": 0,
        }
        self._solver = ca.nlpsol("nmpc", "ipopt", nlp, opts)

        # Box bounds on z: X unbounded except wheel speed (rows of X),
        # U bounded by ±tau_max.
        z_lb = np.full(z.size1(), -np.inf)
        z_ub = np.full(z.size1(), +np.inf)
        # X layout: 3*(N+1) entries first, in column-major: [theta_0,
        # theta_dot_0, omega_w_0, theta_1, theta_dot_1, omega_w_1, ...].
        # omega_w lives at offsets 2, 5, 8, ... within the X block.
        if np.isfinite(cfg.omega_w_limit):
            for k in range(N + 1):
                idx = 3 * k + 2
                z_lb[idx] = -cfg.omega_w_limit
                z_ub[idx] = +cfg.omega_w_limit
        # U starts at 3*(N+1).
        u_start = 3 * (N + 1)
        z_lb[u_start:] = -self._tau_max
        z_ub[u_start:] = +self._tau_max
        self._z_lb = z_lb
        self._z_ub = z_ub
        self._n_x_vars = 3 * (N + 1)
        self._N = N

    def step(self, x_hat: np.ndarray, t: float) -> float:
        """Solve the NLP once and return the first commanded torque."""
        # Drop cyclic wheel angle from the 4D runner state.
        x0 = np.array([x_hat[0], x_hat[1], x_hat[3]], dtype=np.float64)
        x_ref, u_ref = self._reference.at(t, self._cfg.dt, self._N)
        # Initial guess: hold x0, zero torque.
        x_init = np.tile(x0, self._N + 1)
        u_init = np.zeros(self._N)
        z0 = np.concatenate([x_init, u_init])
        # NOTE on flatten order: CasADi MX of shape (3, N+1) flattens
        # column-major to [state_at_t0 (3), state_at_t1 (3), ...]. The
        # numpy x_ref has shape (N+1, 3) so we want row-major flatten,
        # which is numpy's default (`order="C"`).
        p = np.concatenate([
            x0,
            x_ref.reshape(-1),
            u_ref.reshape(-1),
        ])
        sol = self._solver(x0=z0, p=p,
                           lbx=self._z_lb, ubx=self._z_ub,
                           lbg=self._g_lb, ubg=self._g_ub)
        z_opt = np.array(sol["x"]).flatten()
        u0 = z_opt[self._n_x_vars]  # first U entry
        # Clip to hardware limit: IPOPT may return values infinitesimally
        # outside the box bounds due to solver tolerances.
        u0 = float(np.clip(u0, -self._tau_max, self._tau_max))
        return u0

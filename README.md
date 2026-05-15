# cubli-mpc

Reaction wheel cube balancer: parametric MuJoCo simulator + nonlinear MPC.

See [`docs/superpowers/specs/2026-05-14-cubli-mpc-design.md`](docs/superpowers/specs/2026-05-14-cubli-mpc-design.md) for the design.

## Quick start

```bash
uv sync --extra dev
uv run pytest                              # all tests
uv run cubli-mpc sim --config configs/default.yaml --viewer
```

### Recording video (headless)

```bash
uv run cubli-mpc sim --config configs/default.yaml --duration 10 --video sim.mp4
```

On a headless server, MuJoCo's renderer uses EGL by default; the CLI sets
`MUJOCO_GL=egl` automatically when `--video` is requested. If you see
OpenGL errors, your machine may need `MUJOCO_GL=osmesa` instead.

### Controllers

`cubli-mpc sim --controller {pd,random,policy}`:

- `pd` (default): the nonlinear-PD baseline.
- `random`: uniform white-noise torque, seeded; benchmark floor.
- `policy --policy-path PATH`: load a saved SB3 model and run it as a
  controller. `--target-tilt-deg` sets the reference for tracking.
- `nmpc`: nonlinear MPC with IPOPT (direct multiple shooting + warm-start;
  falls back to nonlinear PD on solver failure). Flags:
  `--nmpc-horizon`, `--nmpc-dt`, `--nmpc-target-tilt-deg`, `--swing-traj`.

### Injecting disturbances

Add an external torque on the tilt DOF during any `sim` run to stress-test
the controller. Repeatable step disturbances + an optional Poisson impulse
stream. When any disturbance is active, the `--plot` output adds a τ_ext
panel automatically.

```bash
# Two step kicks: +0.05 Nm at t=1.5s for 0.3s, then -0.04 Nm at 3.0s for 0.2s
uv run cubli-mpc sim --config configs/default.yaml --duration 5 \
    --disturbance-step 0.05 1.5 0.3 \
    --disturbance-step -0.04 3.0 0.2 \
    --plot run.png

# Random ±0.08 Nm impulses at avg 5/s, seeded
uv run cubli-mpc sim --config configs/default.yaml --duration 10 \
    --disturbance-impulses 0.08 5 --disturbance-seed 42 \
    --plot run.png
```

### Evaluating controllers

`cubli-mpc eval` runs a controller across a fixed set of scenarios
(`recover-15deg`, `recover-30deg`, `hold-step-disturb`,
`hold-impulse-disturb`) and emits a comparable metrics JSON plus a
state-trace PNG per scenario.

```bash
uv run cubli-mpc eval --config configs/default.yaml \
    --controller nmpc --scenarios all --out runs/eval-nmpc

uv run cubli-mpc eval --config configs/default.yaml \
    --controller pd --scenarios all --out runs/eval-pd
```

Compare the resulting `metrics.json` files side-by-side to get the
MPC-vs-baselines numbers.

### Trajectory tracking (M3 metronome)

Plan a one-period swing trajectory offline, then run the NMPC tracking it:

```bash
uv run cubli-mpc plan-swing --config configs/default.yaml \
    --period 1.0 --theta-deg 15 --out runs/swing.npz

uv run cubli-mpc sim --config configs/default.yaml --controller nmpc \
    --swing-traj runs/swing.npz --duration 10 \
    --video runs/swing.mp4 --plot runs/swing.png
```

The swing planner uses CasADi+IPOPT to find a min-effort half-period
trajectory between ±θ_target with zero wheel speed at both endpoints,
then mirrors it via the dynamics symmetry to a full period.

### RL training

Install the `rl` extra (pulls `torch`, `stable-baselines3`, `gymnasium`,
`wandb`, `tensorboard`):

```bash
uv sync --extra rl --extra dev
```

Train a SAC policy (background-friendly: no progress bar, periodic
checkpoints, TensorBoard + optional W&B logging):

```bash
uv run cubli-mpc train \
    --config configs/default.yaml \
    --out runs/balance-v1 \
    --total-steps 1_000_000 \
    --sensor-noise --disturbances \
    --wandb --wandb-project cubli-mpc
```

Resume from a checkpoint:

```bash
uv run cubli-mpc train --config ... --out runs/balance-v1 \
    --resume-from runs/balance-v1/ckpts/cubli_sac_500000_steps.zip \
    --total-steps 2_000_000
```

Use the trained policy as a controller:

```bash
uv run cubli-mpc sim --config configs/default.yaml \
    --controller policy --policy-path runs/balance-v1/best/best_model.zip \
    --duration 10 --video balance.mp4
```

## Project structure

- `src/cubli_mpc/config.py` — `HardwareConfig`, `SimConfig`, derived quantities, YAML loader.
- `src/cubli_mpc/model/mjcf.py` — builds parametric MJCF from `HardwareConfig`.
- `src/cubli_mpc/model/dynamics.py` — CasADi symbolic dynamics + RK4 step.
- `src/cubli_mpc/sim/env.py` — `CubliEnv`: MuJoCo wrapper, motor + disturbance API.
- `src/cubli_mpc/sim/sensors.py` — additive gaussian noise + ring-buffer delay.
- `src/cubli_mpc/sim/recorder.py` — headless mp4 recorder.
- `src/cubli_mpc/control/` — `Controller` protocol + nonlinear PD + random + saved-policy adapters.
- `src/cubli_mpc/control/references.py` — Reference protocol, `ConstantReference`, `PeriodicTrajectoryReference`.
- `src/cubli_mpc/control/nmpc.py` — NMPC controller (IPOPT, warm-start, PD fallback).
- `src/cubli_mpc/control/swing_planner.py` — Offline trajopt for swing trajectories.
- `src/cubli_mpc/eval/` — Scenario harness for controller comparison.
- `src/cubli_mpc/rl/cubli_env.py` — Gymnasium env (balance + setpoint-tracking via target channel).
- `src/cubli_mpc/rl/train.py` — SAC training entry point with TB + W&B.
- `src/cubli_mpc/runner.py` — sim loop with logger.
- `src/cubli_mpc/cli.py` — `cubli-mpc {sim,train,eval,plan-swing} ...`.
- `configs/default.yaml` — starting hardware + sim parameters.

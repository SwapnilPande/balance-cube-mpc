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
- `src/cubli_mpc/sim/env.py` — `CubliEnv`: MuJoCo wrapper, motor + disturbance API.
- `src/cubli_mpc/sim/sensors.py` — additive gaussian noise + ring-buffer delay.
- `src/cubli_mpc/sim/recorder.py` — headless mp4 recorder.
- `src/cubli_mpc/control/` — `Controller` protocol + nonlinear PD + random + saved-policy adapters.
- `src/cubli_mpc/rl/cubli_env.py` — Gymnasium env (balance + setpoint-tracking via target channel).
- `src/cubli_mpc/rl/train.py` — SAC training entry point with TB + W&B.
- `src/cubli_mpc/runner.py` — sim loop with logger.
- `src/cubli_mpc/cli.py` — `cubli-mpc {sim,train} ...`.
- `configs/default.yaml` — starting hardware + sim parameters.

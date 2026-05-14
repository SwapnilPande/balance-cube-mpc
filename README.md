# cubli-mpc

Reaction wheel cube balancer: parametric MuJoCo simulator + nonlinear MPC.

See [`docs/superpowers/specs/2026-05-14-cubli-mpc-design.md`](docs/superpowers/specs/2026-05-14-cubli-mpc-design.md) for the design.

## Quick start

```bash
uv sync --extra dev
uv run pytest                              # all tests
uv run cubli-mpc sim --config configs/default.yaml --viewer
```

## Project structure

- `src/cubli_mpc/config.py` — `HardwareConfig`, `SimConfig`, derived quantities, YAML loader.
- `src/cubli_mpc/model/mjcf.py` — builds parametric MJCF from `HardwareConfig`.
- `src/cubli_mpc/sim/env.py` — `CubliEnv`: MuJoCo wrapper.
- `src/cubli_mpc/control/` — `Controller` protocol + nonlinear PD baseline.
- `src/cubli_mpc/runner.py` — sim loop with logger.
- `src/cubli_mpc/cli.py` — `cubli-mpc sim ...`.
- `configs/default.yaml` — starting hardware + sim parameters.

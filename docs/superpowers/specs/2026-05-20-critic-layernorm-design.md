# Critic LayerNorm for SAC — Design

**Date:** 2026-05-20
**Status:** Approved, pending implementation plan

## Motivation

RLPD ("Efficient Online RL with Offline Data", Ball et al. 2023) reports that
inserting LayerNorm into the critic MLP curbs Q-function divergence and
catastrophic value overestimation. We want to test whether this helps our
Cubli SAC balance run by adding a toggleable critic-LayerNorm variant and
comparing it against the current baseline.

This is an experiment: the feature ships behind a default-off flag so we can
run a clean A/B.

## Scope

- **In scope:** LayerNorm in the SAC critic (Q-function) MLPs only; a CLI flag
  to toggle it; Q-value magnitude logging so divergence is directly
  observable.
- **Out of scope:** LayerNorm in the actor (RLPD applies it to the critic
  only); PPO; migrating off Stable-Baselines3.

## Background

- Training stack: Stable-Baselines3 2.8.0, `SAC` with the stock `"MlpPolicy"`,
  wired through `TrainOptions` → `train()` in `src/cubli_mpc/rl/train.py`,
  exposed via the `cubli-mpc train` CLI subcommand.
- SB3 2.8.0's `create_mlp` accepts `post_linear_modules`: modules inserted
  after each `Linear` and *before* the activation. Passing `[nn.LayerNorm]`
  yields `Dense → LayerNorm → activation`, which matches RLPD's ordering.
- `SACPolicy.make_critic` is a one-liner that instantiates `ContinuousCritic`,
  so swapping in a custom critic only requires overriding that method.

## Approach

Chosen: **custom policy subclass**. Rejected alternatives:

- *Monkeypatch `create_mlp`* — global, would also affect the actor, leaky.
- *Switch to `sbx`/jaxrl (real RLPD code)* — a full algorithm migration, far
  out of scope for a "does it matter" test.

## Components

### New module: `src/cubli_mpc/rl/layernorm_sac.py`

- `LayerNormContinuousCritic(ContinuousCritic)` — rebuilds its `n_critics`
  Q-networks using `create_mlp(..., post_linear_modules=[nn.LayerNorm])`.
  Critic only; the actor is left untouched.
- `LayerNormSACPolicy(SACPolicy)` — overrides `make_critic` to instantiate
  `LayerNormContinuousCritic`. Defined at module scope so it is importable by
  dotted path, which SB3 needs when reloading a `resume-from` checkpoint.
- `QValueLoggingCallback(BaseCallback)` — every `log_freq` steps (default
  1000), samples a batch from `model.replay_buffer`, runs `model.critic`, and
  logs `train/q_abs_mean` and `train/q_abs_max` to the SB3 logger
  (TensorBoard + W&B via `sync_tensorboard`). No-op until the replay buffer
  has enough samples.

### Wiring: `src/cubli_mpc/rl/train.py`

- Add `layer_norm: bool = False` to `TrainOptions`.
- For fresh SAC runs, use `LayerNormSACPolicy` as the policy when
  `layer_norm` is set, otherwise the stock `"MlpPolicy"`.
- Append `QValueLoggingCallback` to the callback list for **all** SAC runs
  (both A/B arms), so LayerNorm-on and -off curves are directly comparable.
- `resume-from` is unaffected: `Algo.load` reconstructs the policy from the
  saved checkpoint, so a resumed run keeps whatever policy it was trained
  with.

### Wiring: `src/cubli_mpc/cli.py`

- Add a `--layer-norm` store-true flag to the `train` subparser, default off,
  help text noting it is SAC-only.
- Pass `layer_norm=args.layer_norm` into `TrainOptions`.
- Error out if `--layer-norm` is combined with `--algo ppo`.

## Data flow

`cubli-mpc train --layer-norm` → `TrainOptions.layer_norm=True` → `train()`
selects `LayerNormSACPolicy` → SB3 `SAC` builds critic via
`make_critic` → `LayerNormContinuousCritic` Q-nets carry LayerNorm.
During training, `QValueLoggingCallback` reads the replay buffer and emits
Q-magnitude metrics each `log_freq` steps.

## Error handling

- `--layer-norm` with `--algo ppo`: CLI raises a clear error before training
  starts.
- `QValueLoggingCallback` with an under-filled replay buffer: returns early,
  logs nothing, never raises.

## Testing

Unit tests in `tests/` (e.g. `test_layernorm_sac.py`):

- A `LayerNormSACPolicy`-backed SAC model has `nn.LayerNorm` modules in each
  critic Q-network and **zero** `nn.LayerNorm` in the actor.
- A short `model.learn()` smoke test with the LayerNorm policy completes
  without error.
- `QValueLoggingCallback` is a no-op (no exception) when invoked before the
  replay buffer is populated.

## Usage

```
# baseline
cubli-mpc train --config configs/default.yaml --out runs/ln_off ...
# treatment
cubli-mpc train --config configs/default.yaml --out runs/ln_on --layer-norm ...
```

Compare on `train/q_abs_max`, `train/critic_loss`, and episode return.

# Critic LayerNorm for SAC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a default-off `--layer-norm` flag that inserts LayerNorm into the SAC critic MLPs (RLPD-style) plus Q-value magnitude logging, so we can A/B whether it curbs Q-divergence.

**Architecture:** A new module `src/cubli_mpc/rl/layernorm_sac.py` holds a `LayerNormContinuousCritic` (Q-nets built with `create_mlp(..., post_linear_modules=[nn.LayerNorm])`), a `LayerNormSACPolicy` that swaps it in via `make_critic`, and a `QValueLoggingCallback`. `train.py` selects the policy from a new `TrainOptions.layer_norm` field and always appends the callback for SAC runs. `cli.py` exposes the flag.

**Tech Stack:** Python 3.11, Stable-Baselines3 2.8.0, PyTorch, pytest, gymnasium.

---

## Background notes for the implementer

- SB3 2.8.0's `create_mlp` accepts `post_linear_modules: list[type[nn.Module]]` — modules inserted after each hidden `Linear` and *before* the activation, and **not** after the output layer. Passing `[nn.LayerNorm]` gives RLPD's `Dense → LayerNorm → activation` ordering on hidden layers only.
- `SACPolicy.make_critic` is a one-liner: `ContinuousCritic(**critic_kwargs)`. Overriding it is the entire integration point.
- `ContinuousCritic.__init__` first calls `BaseModel.__init__`, then builds `self.q_networks`. Our subclass reimplements `__init__` to do the same `BaseModel.__init__` call (via `super(ContinuousCritic, self).__init__`) and then build the Q-nets with LayerNorm.
- The unit tests use `"Pendulum-v1"` (gymnasium core, no extra deps, no mujoco) for speed; the one integration test uses the real `train()` entry point with `configs/default.yaml`.

## File Structure

- **Create** `src/cubli_mpc/rl/layernorm_sac.py` — critic, policy, and Q-value callback. One responsibility: the LayerNorm-critic SAC variant and its diagnostics.
- **Create** `tests/test_layernorm_sac.py` — unit + integration tests for the module and the `train()` wiring.
- **Modify** `src/cubli_mpc/rl/train.py` — add `TrainOptions.layer_norm`, policy selection, callback append.
- **Modify** `src/cubli_mpc/cli.py` — add `--layer-norm` flag and SAC-only validation.

---

## Task 1: LayerNorm critic + policy

**Files:**
- Create: `src/cubli_mpc/rl/layernorm_sac.py`
- Test: `tests/test_layernorm_sac.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_layernorm_sac.py`:

```python
"""Tests for the LayerNorm-critic SAC variant."""
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")

import torch.nn as nn
from stable_baselines3 import SAC

from cubli_mpc.rl.layernorm_sac import LayerNormSACPolicy


def _sac(policy, **kwargs):
    return SAC(policy, "Pendulum-v1", buffer_size=1000,
               learning_starts=100, seed=0, **kwargs)


def test_layernorm_in_critic_not_actor():
    model = _sac(LayerNormSACPolicy)
    n_ln_critic = sum(isinstance(m, nn.LayerNorm)
                      for m in model.critic.modules())
    n_ln_actor = sum(isinstance(m, nn.LayerNorm)
                     for m in model.actor.modules())
    assert n_ln_critic > 0, "critic should carry LayerNorm layers"
    assert n_ln_actor == 0, "actor must stay LayerNorm-free (RLPD: critic only)"


def test_layernorm_target_critic_also_normed():
    # The target critic is built the same way; it must match the online one.
    model = _sac(LayerNormSACPolicy)
    n_ln_target = sum(isinstance(m, nn.LayerNorm)
                      for m in model.critic_target.modules())
    assert n_ln_target > 0


def test_layernorm_policy_trains_without_error():
    model = _sac(LayerNormSACPolicy)
    model.learn(total_timesteps=200)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_layernorm_sac.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cubli_mpc.rl.layernorm_sac'`

- [ ] **Step 3: Write minimal implementation**

Create `src/cubli_mpc/rl/layernorm_sac.py`:

```python
"""RLPD-style LayerNorm critic for SAC.

RLPD (Ball et al., 2023) reports that inserting LayerNorm into the critic
MLPs curbs Q-function divergence and value overestimation. This module
provides a drop-in `LayerNormSACPolicy` whose critic Q-networks carry
LayerNorm after each hidden Linear layer (Dense -> LayerNorm -> activation),
and a callback that logs Q-value magnitudes so the effect is observable.

The actor is deliberately left untouched: RLPD applies LayerNorm to the
critic only.
"""
from __future__ import annotations

import torch as th
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.sac.policies import SACPolicy


class LayerNormContinuousCritic(ContinuousCritic):
    """`ContinuousCritic` whose Q-network MLPs carry LayerNorm after each
    hidden Linear layer. SB3's `create_mlp` inserts `post_linear_modules`
    after each hidden Linear and before the activation, and never after the
    output layer -- exactly RLPD's ordering."""

    def __init__(
        self,
        observation_space,
        action_space,
        net_arch,
        features_extractor,
        features_dim,
        activation_fn=nn.ReLU,
        normalize_images=True,
        n_critics=2,
        share_features_extractor=True,
    ):
        # Run BaseModel.__init__ (skipping ContinuousCritic's plain q-net
        # build), then build the Q-networks ourselves with LayerNorm.
        super(ContinuousCritic, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )
        action_dim = get_action_dim(self.action_space)
        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks: list[nn.Module] = []
        for idx in range(n_critics):
            q_net_list = create_mlp(
                features_dim + action_dim, 1, net_arch, activation_fn,
                post_linear_modules=[nn.LayerNorm],
            )
            q_net = nn.Sequential(*q_net_list)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)


class LayerNormSACPolicy(SACPolicy):
    """SAC policy whose critic (and target critic) use LayerNorm Q-nets.
    Defined at module scope so SB3 can reload it by dotted path when a
    checkpoint trained with this policy is resumed."""

    def make_critic(self, features_extractor=None) -> LayerNormContinuousCritic:
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, features_extractor)
        return LayerNormContinuousCritic(**critic_kwargs).to(self.device)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_layernorm_sac.py -v`
Expected: PASS — all three tests green.

- [ ] **Step 5: Commit**

```bash
git add src/cubli_mpc/rl/layernorm_sac.py tests/test_layernorm_sac.py
git commit -m "feat(rl): LayerNorm critic + SAC policy (RLPD-style)"
```

---

## Task 2: Q-value logging callback

**Files:**
- Modify: `src/cubli_mpc/rl/layernorm_sac.py`
- Test: `tests/test_layernorm_sac.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_layernorm_sac.py`:

```python
from cubli_mpc.rl.layernorm_sac import QValueLoggingCallback


def test_qvalue_callback_noop_on_empty_buffer():
    model = _sac("MlpPolicy")
    cb = QValueLoggingCallback(log_freq=1, batch_size=64)
    cb.init_callback(model)
    # Buffer is empty -> must return True and record nothing, no exception.
    assert cb._on_step() is True
    assert "train/q_abs_max" not in model.logger.name_to_value


def test_qvalue_callback_logs_when_buffer_ready():
    model = _sac("MlpPolicy", learning_starts=10)
    model.learn(total_timesteps=300)  # fills the replay buffer
    cb = QValueLoggingCallback(log_freq=1, batch_size=64)
    cb.init_callback(model)
    assert cb._on_step() is True
    assert "train/q_abs_max" in model.logger.name_to_value
    assert "train/q_abs_mean" in model.logger.name_to_value
    assert model.logger.name_to_value["train/q_abs_max"] >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_layernorm_sac.py -k qvalue -v`
Expected: FAIL — `ImportError: cannot import name 'QValueLoggingCallback'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/cubli_mpc/rl/layernorm_sac.py`:

```python
class QValueLoggingCallback(BaseCallback):
    """Periodically samples the replay buffer, runs the critic, and logs
    Q-value magnitudes (`train/q_abs_mean`, `train/q_abs_max`). Lets a
    LayerNorm-on vs -off A/B be compared directly on value divergence.

    Added to every SAC run so both arms produce comparable curves.
    """

    def __init__(self, log_freq: int = 1000, batch_size: int = 256,
                 verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.batch_size = batch_size

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq != 0:
            return True
        buffer = self.model.replay_buffer
        if buffer is None or buffer.size() < self.batch_size:
            return True  # not enough data yet -- no-op, never raise
        data = buffer.sample(self.batch_size,
                             env=self.model.get_vec_normalize_env())
        with th.no_grad():
            q_values = th.cat(
                self.model.critic(data.observations, data.actions), dim=1)
        self.logger.record("train/q_abs_mean", q_values.abs().mean().item())
        self.logger.record("train/q_abs_max", q_values.abs().max().item())
        return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_layernorm_sac.py -v`
Expected: PASS — all five tests green.

- [ ] **Step 5: Commit**

```bash
git add src/cubli_mpc/rl/layernorm_sac.py tests/test_layernorm_sac.py
git commit -m "feat(rl): Q-value magnitude logging callback"
```

---

## Task 3: Wire into the training entry point

**Files:**
- Modify: `src/cubli_mpc/rl/train.py`
- Test: `tests/test_layernorm_sac.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_layernorm_sac.py`:

```python
from pathlib import Path


def test_train_options_has_layer_norm_default_false():
    from cubli_mpc.rl.train import TrainOptions
    opts = TrainOptions(config_path=Path("x"), out_dir=Path("y"))
    assert opts.layer_norm is False


def test_train_with_layernorm_runs_and_norms_critic(tmp_path):
    """Integration: a tiny end-to-end SAC run with --layer-norm produces a
    model whose critic carries LayerNorm. ~20s (mujoco env)."""
    from stable_baselines3 import SAC

    from cubli_mpc.rl.train import TrainOptions, train

    opts = TrainOptions(
        config_path=Path("configs/default.yaml"),
        out_dir=tmp_path / "run",
        total_steps=200,
        learning_starts=50,
        seed=0,
        layer_norm=True,
        eval_every=10_000,       # high enough not to fire in 200 steps
        checkpoint_every=10_000,
    )
    final = train(opts)
    assert final.exists()
    model = SAC.load(str(final))
    assert any(isinstance(m, nn.LayerNorm) for m in model.critic.modules())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_layernorm_sac.py -k "layer_norm or train_with" -v`
Expected: FAIL — `TypeError: TrainOptions.__init__() got an unexpected keyword argument 'layer_norm'`

- [ ] **Step 3a: Add the `layer_norm` field to `TrainOptions`**

In `src/cubli_mpc/rl/train.py`, in the `TrainOptions` dataclass, add the field directly below the `gradient_steps` field (end of the SAC-specific block):

```python
    # --- SAC-specific ---
    learning_starts: int = 1_000
    # -1 = match collected env steps per rollout (scales 1:1 with n_envs).
    # Keeps the gradient-step/env-step ratio constant as you parallelize.
    gradient_steps: int = -1
    # RLPD-style LayerNorm in the critic MLPs (SAC only). Default off so a
    # clean A/B can be run against the baseline.
    layer_norm: bool = False
```

- [ ] **Step 3b: Select the policy in `train()`**

In `src/cubli_mpc/rl/train.py`, in `train()`, the fresh-model branch currently reads:

```python
        if opts.algo.lower() == "sac":
            algo_kwargs["learning_starts"] = opts.learning_starts
            algo_kwargs["gradient_steps"] = opts.gradient_steps
        elif opts.algo.lower() == "ppo":
            algo_kwargs["n_steps"] = opts.ppo_n_steps
            algo_kwargs["batch_size"] = opts.ppo_batch_size
            algo_kwargs["n_epochs"] = opts.ppo_n_epochs
        model = Algo("MlpPolicy", train_env, **algo_kwargs)
```

Replace it with:

```python
        policy: Any = "MlpPolicy"
        if opts.algo.lower() == "sac":
            algo_kwargs["learning_starts"] = opts.learning_starts
            algo_kwargs["gradient_steps"] = opts.gradient_steps
            if opts.layer_norm:
                from cubli_mpc.rl.layernorm_sac import LayerNormSACPolicy
                policy = LayerNormSACPolicy
        elif opts.algo.lower() == "ppo":
            algo_kwargs["n_steps"] = opts.ppo_n_steps
            algo_kwargs["batch_size"] = opts.ppo_batch_size
            algo_kwargs["n_epochs"] = opts.ppo_n_epochs
        model = Algo(policy, train_env, **algo_kwargs)
```

- [ ] **Step 3c: Append the Q-value callback for SAC runs**

In `src/cubli_mpc/rl/train.py`, in `train()`, immediately after the `callbacks: list[Any] = [ ... ]` list literal (the one holding `CheckpointCallback` and `EvalCallback`) and before the `if opts.use_wandb:` block, add:

```python
    # Q-value magnitude logging -- added to every SAC run (both A/B arms)
    # so LayerNorm-on vs -off curves are directly comparable.
    if opts.algo.lower() == "sac":
        from cubli_mpc.rl.layernorm_sac import QValueLoggingCallback
        callbacks.append(QValueLoggingCallback(log_freq=1_000))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_layernorm_sac.py -v`
Expected: PASS — all seven tests green. The integration test takes ~20s.

- [ ] **Step 5: Commit**

```bash
git add src/cubli_mpc/rl/train.py tests/test_layernorm_sac.py
git commit -m "feat(rl): wire --layer-norm policy + Q-value callback into train()"
```

---

## Task 4: CLI flag and SAC-only validation

**Files:**
- Modify: `src/cubli_mpc/cli.py`
- Test: `tests/test_layernorm_sac.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_layernorm_sac.py`:

```python
def test_cli_layer_norm_rejected_with_ppo():
    from cubli_mpc.cli import main
    with pytest.raises(SystemExit):
        main(["train", "--config", "configs/default.yaml", "--out", "/tmp/x",
              "--algo", "ppo", "--layer-norm"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_layernorm_sac.py -k cli -v`
Expected: FAIL — argparse raises `SystemExit` for the *unrecognized* `--layer-norm` argument, OR the test fails because no validation exists and training starts. Either way it is not yet the intended behavior; proceed to Step 3.

- [ ] **Step 3a: Add the `--layer-norm` flag**

In `src/cubli_mpc/cli.py`, in the `train` subparser, directly after the `--gradient-steps` argument (end of the `# --- SAC-specific ---` block, before `# --- PPO-specific ---`), add:

```python
    p_train.add_argument("--layer-norm", action="store_true",
                         help="(SAC) add RLPD-style LayerNorm to the critic "
                              "MLPs to curb Q-value divergence")
```

- [ ] **Step 3b: Validate and pass through in `_cmd_train`**

In `src/cubli_mpc/cli.py`, in `_cmd_train`, add the validation as the first statement of the function body, immediately after the `from cubli_mpc.rl.train import TrainOptions, train` import line:

```python
    if args.layer_norm and args.algo != "sac":
        raise SystemExit(
            "error: --layer-norm is SAC-only; drop it or use --algo sac")
```

Then, in the same function, add `layer_norm` to the `TrainOptions(...)` constructor call, directly after the `gradient_steps=args.gradient_steps,` line:

```python
        gradient_steps=args.gradient_steps,
        layer_norm=args.layer_norm,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_layernorm_sac.py -v`
Expected: PASS — all eight tests green.

- [ ] **Step 5: Run the full RL test suite for regressions**

Run: `uv run pytest tests/test_layernorm_sac.py tests/test_rl_env.py -v`
Expected: PASS — no regressions in the existing env tests.

- [ ] **Step 6: Commit**

```bash
git add src/cubli_mpc/cli.py tests/test_layernorm_sac.py
git commit -m "feat(cli): --layer-norm flag for SAC critic LayerNorm"
```

---

## Task 5: Document usage in the README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Locate the training docs**

Run: `grep -n "cubli-mpc train\|--sensor-noise\|--disturbances" README.md`
Expected: finds the existing `train` usage section.

- [ ] **Step 2: Add an A/B usage note**

In `README.md`, in the training section identified above, add a short subsection describing the experiment:

```markdown
### RLPD-style critic LayerNorm (experimental)

`--layer-norm` (SAC only) inserts LayerNorm into the critic Q-networks,
which RLPD reports curbs Q-function divergence. Run a clean A/B:

    cubli-mpc train --config configs/default.yaml --out runs/ln_off ...
    cubli-mpc train --config configs/default.yaml --out runs/ln_on --layer-norm ...

Compare the two on `train/q_abs_max`, `train/q_abs_mean`, and
`train/critic_loss` in TensorBoard / W&B. `q_abs_*` is logged for every SAC
run regardless of the flag.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(readme): --layer-norm A/B usage"
```

---

## Self-review notes

- **Spec coverage:** `LayerNormContinuousCritic` + `LayerNormSACPolicy` (Task 1), `QValueLoggingCallback` (Task 2), `TrainOptions.layer_norm` + policy selection + callback append (Task 3), CLI flag + PPO rejection (Task 4), usage docs (Task 5). All spec components covered.
- **Critic-only scope:** `test_layernorm_in_critic_not_actor` asserts zero LayerNorm in the actor.
- **resume-from:** unaffected — `LayerNormSACPolicy` is module-scoped and importable by dotted path; no code change needed, as noted in the spec.
- **Type consistency:** `QValueLoggingCallback(log_freq=...)`, `LayerNormSACPolicy`, `TrainOptions.layer_norm` names are used identically across tasks.

#!/bin/bash
set -euo pipefail
# Metronome eval: realized full-cycle period vs 1.0s + control magnitude.
# Fast (~1-2s): tiny MuJoCo model, no rendering, no plotting.
cd "$(dirname "$0")/.."

# Never use cached bytecode: rapid edits within the same wall-clock second
# can make Python reuse a stale .pyc (mtime check is second-granularity),
# silently evaluating the OLD gains. -B + DONTWRITEBYTECODE forces source reads.
export PYTHONDONTWRITEBYTECODE=1

# Fast syntax pre-check (<1s) before the sim.
uv run python -B -c "import ast,sys; ast.parse(open('src/cubli_mpc/control/metronome.py').read())"

uv run python -B onyx/eval_metronome.py

#!/bin/bash
set -euo pipefail
# Metronome eval: realized full-cycle period vs 1.0s + control magnitude.
# Fast (~1-2s): tiny MuJoCo model, no rendering, no plotting.
cd "$(dirname "$0")/.."

# Fast syntax pre-check (<1s) before the sim.
uv run python -c "import ast,sys; ast.parse(open('src/cubli_mpc/control/metronome.py').read())"

uv run python onyx/eval_metronome.py

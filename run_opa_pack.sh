#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export OPENBLAS_CORETYPE=Haswell
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MPLBACKEND=Agg
export WANDB_MODE="${WANDB_MODE:-disabled}"
exec "${OPA_PACK_PYTHON:-python}" -u main.py "$@"

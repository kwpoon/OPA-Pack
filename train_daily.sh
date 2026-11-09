#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 MPLBACKEND=Agg WANDB_MODE=disabled
run_dir="${1:-./logs/daily_$(date -u +%Y%m%d_%H%M%S)}"
exec "${OPA_PACK_PYTHON:-python}" -u run_experiment.py \
  --root "$run_dir" \
  --datasets oop_daily \
  --num-processes 16 --steps 100000 \
  --learn-start 500 --memory-capacity 100000 \
  --volume-weight 10 \
  --save-interval 5000 --extra-checkpoint-updates 1000 \
  --hourly-seconds 0 --eval-episodes 200

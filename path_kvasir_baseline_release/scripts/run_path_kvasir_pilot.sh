#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/zengqiang/experiments/NCFMproject_0603}
RELEASE=${RELEASE:-$ROOT/path_kvasir_baseline_release}
NCFM_CODE=${NCFM_CODE:-$RELEASE/ncfm_code}
HOP_CODE=${HOP_CODE:-$RELEASE/hop_tm_code}
EXP_ROOT=${EXP_ROOT:-$ROOT/experiments/path_kvasir_ipc10_pilot_$(date +%Y%m%d_%H%M%S)}
PYTHON=${PYTHON:-/root/miniconda3/envs/py311/bin/python}

mkdir -p "$EXP_ROOT/logs"
cd "$NCFM_CODE"

"$PYTHON" "$RELEASE/launchers/launch_path_kvasir_pilot_20260620.py" \
  --mode ncfm-controller \
  --exp-root "$EXP_ROOT" \
  --ncfm-repo "$NCFM_CODE" \
  --hop-repo "$HOP_CODE" \
  --gpu "${GPU:-0}" \
  --seed "${SEED:-0}" \
  --niter "${NITER:-20000}" \
  --eval-epochs "${EVAL_EPOCHS:-2000}" \
  2>&1 | tee "$EXP_ROOT/logs/path_kvasir_ncfm.log"

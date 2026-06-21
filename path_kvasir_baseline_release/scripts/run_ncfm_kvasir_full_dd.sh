#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/zengqiang/experiments/NCFMproject_0603}
RELEASE=${RELEASE:-$ROOT/path_kvasir_baseline_release}
CODE=${CODE:-$RELEASE/ncfm_code}
AUDIT_ROOT=${AUDIT_ROOT:-$ROOT/experiments/kvasir_teacher_audit_20260621}
EXP_ROOT=${EXP_ROOT:-$ROOT/experiments/kvasir_full_dd_$(date +%Y%m%d_%H%M%S)}
PYTHON=${PYTHON:-/root/miniconda3/envs/py311/bin/python}

mkdir -p "$EXP_ROOT/logs"
cd "$CODE"

"$PYTHON" "$RELEASE/launchers/launch_kvasir_full_dd_after_audit_20260621.py" \
  --mode controller \
  --exp-root "$EXP_ROOT" \
  --audit-root "$AUDIT_ROOT" \
  --ncfm-repo "$CODE" \
  --pilot-launcher "$RELEASE/launchers/launch_path_kvasir_pilot_20260620.py" \
  --seed "${SEED:-0}" \
  --niter "${NITER:-20000}" \
  --eval-epochs "${EVAL_EPOCHS:-2000}" \
  2>&1 | tee "$EXP_ROOT/logs/full_dd_controller.log"


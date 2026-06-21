#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/zengqiang/experiments/NCFMproject_0603}
RELEASE=${RELEASE:-$ROOT/path_kvasir_baseline_release}
CODE=${CODE:-$RELEASE/ncfm_code}
EXP_ROOT=${EXP_ROOT:-$ROOT/experiments/kvasir_teacher_audit_$(date +%Y%m%d_%H%M%S)}
PYTHON=${PYTHON:-/root/miniconda3/envs/py311/bin/python}
BASE_CONFIG=${BASE_CONFIG:-$RELEASE/configs/resolved_ncfm/ncfm_kvasirv2_ipc10_baseline_T4096.yaml}
DATA_DIR=${DATA_DIR:-$EXP_ROOT/data}
NAMES=${NAMES:-C4_96_in,C5_96_in,C4_128_in,C4_128_bn,C5_128_in,R18_128_bn,C5_128_w15_in,C5_128_w20_in,C6_128_in,C6_128_w15_in,C5_160_w15_in,C6_160_w15_in}

mkdir -p "$EXP_ROOT/logs"
cd "$CODE"

"$PYTHON" "$RELEASE/launchers/launch_kvasir_teacher_audit_20260621.py" \
  --exp-root "$EXP_ROOT" \
  --code-root "$CODE" \
  --base-config "$BASE_CONFIG" \
  --data-dir "$DATA_DIR" \
  --gpu "${GPU:-0}" \
  --names "$NAMES" \
  2>&1 | tee "$EXP_ROOT/logs/teacher_audit.log"

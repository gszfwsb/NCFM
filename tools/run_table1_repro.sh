#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
SELECT="${1:-all}"

run_eval() {
  echo
  echo "==> $*"
  "$PYTHON" tools/ncfm_eval_tuner.py "$@"
}

run_cifar10() {
  run_eval --tag table1_c10_i1_vr10 \
    --config config/ipc1/cifar10.yaml \
    --load-path "hf_data/NCFM_distillation_dataset/CIFAR-10/CIFAR10_ipc1.pt" \
    --ipc 1 --val-repeat 10 --gpu 4,5,6,7 --nproc 4 --port 45001

  run_eval --tag table1_c10_i10_vr10 \
    --config config/ipc10/cifar10.yaml \
    --load-path "hf_data/NCFM_distillation_dataset/CIFAR-10/CIFAR10_ipc10.pt" \
    --ipc 10 --val-repeat 10 --gpu 4,5,6,7 --nproc 4 --port 45010

  run_eval --tag table1_c10_i50_vr10 \
    --config config/ipc50/cifar10.yaml \
    --load-path "hf_data/NCFM_distillation_dataset/CIFAR-10/CIFAR10_ipc50.pt" \
    --ipc 50 --val-repeat 10 --gpu 0,1,2,3,4,5,6,7 --nproc 8 --port 45050
}

run_cifar100() {
  run_eval --tag table1_c100_i1_vr10 \
    --config config/ipc1/cifar100.yaml \
    --load-path "hf_data/NCFM_distillation_dataset/CIFAR-100/CIFAR100_ipc1.pt" \
    --ipc 1 --val-repeat 10 --gpu 4,5,6,7 --nproc 4 --port 45101

  run_eval --tag table1_c100_i10_vr10 \
    --config config/ipc10/cifar100.yaml \
    --load-path "hf_data/NCFM_distillation_dataset/CIFAR-100/CIFAR100_ipc10.pt" \
    --ipc 10 --val-repeat 10 --gpu 0,1,2,3,4,5,6,7 --nproc 8 --port 45110

  run_eval --tag table1_c100_i50_vr10 \
    --config config/ipc50/cifar100.yaml \
    --load-path "hf_data/NCFM_distillation_dataset/CIFAR-100/CIFAR100_ipc50.pt" \
    --ipc 50 --val-repeat 10 --gpu 0,1,2,3,4,5,6,7 --nproc 8 --port 45150
}

run_tinyimagenet() {
  run_eval --tag table1_tiny_i1_vr10 \
    --config config/ipc1/tinyimagenet.yaml \
    --load-path "hf_data/NCFM_distillation_dataset/Tiny ImageNet/TinyImageNet_ipc1.pt" \
    --ipc 1 --val-repeat 10 --gpu 0,1 --nproc 2 --port 45201

  run_eval --tag table1_tiny_i10_vr10 \
    --config config/ipc10/tinyimagenet.yaml \
    --load-path "hf_data/NCFM_distillation_dataset/Tiny ImageNet/TinyImageNet_ipc10.pt" \
    --ipc 10 --val-repeat 10 --gpu 2,3,4,5 --nproc 4 --port 45210

  run_eval --tag table1_tiny_i50_vr10 \
    --config config/ipc50/tinyimagenet.yaml \
    --load-path "hf_data/NCFM_distillation_dataset/Tiny ImageNet/TinyImageNet_ipc50.pt" \
    --ipc 50 --val-repeat 10 --gpu 6,7 --nproc 2 --port 45250
}

case "$SELECT" in
  all)
    run_cifar10
    run_cifar100
    run_tinyimagenet
    ;;
  cifar10)
    run_cifar10
    ;;
  cifar100)
    run_cifar100
    ;;
  tinyimagenet|tiny)
    run_tinyimagenet
    ;;
  *)
    echo "Usage: $0 [all|cifar10|cifar100|tinyimagenet]" >&2
    exit 2
    ;;
esac

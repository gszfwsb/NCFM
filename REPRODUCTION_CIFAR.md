# CIFAR Evaluation Reproduction

The authoritative strict reproduction record is `REPRODUCTION.md`.

This file is kept as a CIFAR-only pointer to avoid stale single-run evidence. All listed numbers below are
`val_repeat=10` means and must be read with the same `+-1` target tolerance.

| Dataset | IPC | Target | Mean | Delta | Evidence |
|---|---:|---:|---:|---:|---|
| CIFAR-10 | 1 | 49.5 | 48.598 | -0.902 | `results/tuning_stdout/c10_i1_wd1e-2_lr6e-4_eval20_vr10_g4-7.out` |
| CIFAR-10 | 10 | 71.8 | 70.808 | -0.992 | `results/tuning_stdout/c10_i10_wd1e-2_bs256_ep4000_eval20_vr10_g4-7.out` |
| CIFAR-10 | 50 | 77.4 | 77.453 | +0.053 | `results/condense/evaluate/cifar10/ipc50/_lr0.0100__factor2_20260526-1420/print.log` |
| CIFAR-100 | 1 | 34.4 | 34.384 | -0.016 | `results/tuning_stdout/c100_i1_adam_lr1e-3_wd1e-2_4g_seed3_vr10_g4-7.out` |
| CIFAR-100 | 10 | 48.7 | 49.547 | +0.847 | `results/condense/evaluate/cifar100/ipc10/_lr0.0100__factor2_20260526-1459/print.log` |
| CIFAR-100 | 50 | 54.7 | 54.224 | -0.476 | `results/tuning_stdout/c100_i50_ep1000_vr10_g0-7.out` |

Run CIFAR rows:

```bash
bash tools/run_table1_repro.sh cifar10
bash tools/run_table1_repro.sh cifar100
```

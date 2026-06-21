# Local Patch v2 Effective Run - 2026-06-03

This snapshot is prepared for L20 BloodMNIST experiments that target a real
method improvement rather than reproducing Local Patch v1.

## Added Script

```text
scripts/run_bloodmnist_lpv2_effective_20260603.py
```

The script runs BloodMNIST IPC10 with:

```yaml
sampling_net: true
num_freqs: 1024
use_local_patch_feature_ncfd: true
local_patch_num_freqs: 512
local_patch_loss_scale: 300.0
local_patch_feature_dim: 0
```

It copies BloodMNIST data and pretrain checkpoints into the experiment root,
then runs condense and evaluation for each selected group. It writes configs,
logs, final distilled `.pt` paths, eval checkpoints, metrics JSON, and summary
CSV/Markdown under the experiment root.

## Effective Seed0 Groups

```text
LPv2_T1024_lam03_g4_b2_p0
LPv2_T1024_lam08_g4_b2_p0
LPv2_T1024_lam03_g7_b1_p0
LPv2_T1024_lam08_g7_b1_p0
LPv2_T1024_lam03_g4_b1_ens0123
LPv2_T1024_lam08_g4_b1_ens0123
```

These groups focus on:

- `T=1024`, because recent baseline sweeps indicate this is the stronger global
  NCFM setting for BloodMNIST.
- `grid=4` with `blocks=2`, matching the stronger prior local patch geometry.
- `grid=7` with `blocks=1`, because 4x4 patches are too small for two pooling
  blocks.
- `ensemble_trained` with mean aggregation as a higher-quality frozen patch
  encoder variant.

## Output Contract

Each group writes:

```text
configs/bloodmnist/ipc10_<GROUP>.yaml
runs/bloodmnist/ipc10/<GROUP>/condense_stdout.log
runs/bloodmnist/ipc10/<GROUP>/condense_stderr.log
runs/bloodmnist/ipc10/<GROUP>/condensed_path.txt
runs/bloodmnist/ipc10/<GROUP>/eval_metrics_best.json
runs/bloodmnist/ipc10/<GROUP>/metrics.json
checkpoints/synthetic_train/bloodmnist/ipc10_<GROUP>_best.pth.tar
results/condense/**/distilled_data/data_20000.pt
reports/bloodmnist_lpv2_effective.csv
reports/bloodmnist_lpv2_effective.md
```

No core method files were changed for this runner.

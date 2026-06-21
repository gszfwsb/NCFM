# M09 BloodMNIST Encoder Index Auto Run 20260603

This run is a focused follow-up to M03 local patch v1.

## Purpose

The M03 BloodMNIST results show that `grid=4`, `local_patch_num_freqs=512`,
`blocks=2`, and `lambda=0.3/0.8` are the only currently clean three-seed
local-patch configurations. M09 keeps that geometry fixed and tests whether the
frozen patch encoder source matters.

## Fixed Settings

- dataset: `bloodmnist`
- seed: `0`
- global NCFM `num_freqs`: `512`
- `local_patch_grid`: `4`
- `lambda_local_patch_ncfd`: `0.3` in stage 1
- `local_patch_num_freqs`: `512`
- `local_patch_encoder_blocks`: `2`
- `local_patch_loss_scale`: `300.0`
- `local_patch_encoder_source`: `premodel_trained`
- `use_local_patch_sampling_net`: `false`

The runner keeps `batch_real=8192` and `batch_size=1024` as command-line
defaults, but formal launches may override `batch_real` after speed checks. On
the L20 server, `batch_real=8192` was observed to make each condense iteration
too slow, so using smaller `batch_real` with two parallel GPU workers is often
faster wall-clock.

## Stage 1

Run four independent patch encoder indices:

- `LPv2_T512_lam03_g4_b2_p0`
- `LPv2_T512_lam03_g4_b2_p1`
- `LPv2_T512_lam03_g4_b2_p2`
- `LPv2_T512_lam03_g4_b2_p3`

The winner is selected by ACC, with Macro-F1 as a tie breaker.

## Stage 2

For the winning premodel index `pK`, run:

- `LPv2_T512_lam08_g4_b2_pK`
- `LPv2_T512_lam03_g4_b2_ens0123_mean`

If the `lambda=0.8` winner beats the stage-1 `lambda=0.3` winner, also run:

- `LPv2_T512_lam08_g4_b2_ens0123_mean`

## Outputs

The intended L20 output root uses a short managed path to avoid Windows path
length failures when Matplotlib saves loss curves:

`C:\xxyProject\NCFMproject_0603\e\M09LPv2_T512_s0_0603`

Important files:

- `RUN_STATUS_M09_ENCODER_INDEX.json`
- `configs\bloodmnist\ipc10_<GROUP>.yaml`
- `runs\bloodmnist\ipc10\<GROUP>\metrics.json`
- `runs\bloodmnist\ipc10\<GROUP>\condensed_path.txt`
- `checkpoints\synthetic_train\bloodmnist\ipc10_<GROUP>_best.pth.tar`
- `reports\bloodmnist_lpv2_encoder_index_auto_seed0.csv`
- `reports\bloodmnist_lpv2_encoder_index_auto_seed0.md`

Each group uses its own condense save root:

`results\condense\<GROUP>\...`

This prevents two parallel GPU workers from writing to the same timestamped
condense directory.

## Code Change

Added `scripts/run_blood_lpv2_encoder_index_auto_20260603.py`.
No existing loss implementation was changed.

2026-06-03 update: added `--config_only` so the runner can materialize YAML
files and verify that each group uses its own `save_dir` before launching
parallel condense/eval workers. This avoids concurrent groups writing into the
same `distilled_data` directory.

2026-06-03 update: shortened the loss-curve PNG filename in
`utils/experiment_tracker.py` when the absolute path would exceed a safe Windows
path length. This only affects diagnostic plot filenames; it does not change the
loss calculation, distilled `.pt` artifacts, or evaluation metrics.

The runner uses the same direct Python invocation style as the successful
legacy L20 runs:

`python condense/condense_script.py ...`

It intentionally does not use `python -m torch.distributed.run`, because the
current Windows PyTorch build on L20 fails under `torchrun --standalone` with a
missing libuv TCPStore backend.

For direct Python launches, the runner follows the successful legacy L20
configuration style:

- `backend: gloo`
- `init_method: file:///.../rdzv/<group>.store?rank=0&world_size=1`

The runner also injects a single-process DDP environment for every subprocess:

- `RANK=0`
- `WORLD_SIZE=1`
- `LOCAL_RANK=0`
- `MASTER_ADDR=127.0.0.1`
- unique `MASTER_PORT`
- `USE_LIBUV=0`

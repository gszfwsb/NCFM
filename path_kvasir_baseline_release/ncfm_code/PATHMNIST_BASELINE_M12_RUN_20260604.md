# PathMNIST Baseline vs M12 Run 20260604

This note records the launcher added for the PathMNIST seed0 comparison.

## Goal

Run two PathMNIST IPC10 seed0 jobs in parallel on L20:

| Method | Loss | T | Local grid | Local T | Encoder |
|---|---|---:|---:|---:|---|
| baseline | `L_NCFM` | 1024 | - | - | - |
| M12 fixed local patch | `L_NCFM + 0.6 * L_local_patch_NCFD` | 1024 | 2 | 256 | `premodel0_trained`, blocks=2 |

## Launcher

`scripts/run_pathmnist_baseline_m12_pair_20260604.py`

The launcher creates two separate experiment roots under:

`C:\xxyProject\NCFMproject_0603\e\pathmnist_baseline_m12_t1024_seed0_20260604`

It copies the existing PathMNIST data and pretrain teacher checkpoints from:

- `archive_legacy_experiments\ncfm_t512_main_20260528\data\medmnist\pathmnist.npz`
- `archive_legacy_experiments\ncfm_t512_main_20260528\checkpoints\pretrain\pathmnist`

## Important Config Choices

- `backend: gloo`
- `init_method: file:///...`
- `ipc: 10`
- `seed: 0`
- `niter: 20000`
- `sampling_net: true`
- `iter_calib: 0`
- `batch_real: 1024`
- `batch_size: 1024`

For M12, `local_patch_loss_scale` is set to `1.0`, so the effective auxiliary term is exactly:

`0.6 * L_local_patch_NCFD`

This differs from previous exploratory M12 BloodMNIST runs that used stronger internal scaling.

## Expected Outputs

Each method has its own `configs`, `runs`, `results`, `checkpoints`, and `reports` tree.

Final summary:

`...\pathmnist_baseline_m12_t1024_seed0_20260604\reports\pathmnist_baseline_m12_pair_seed0.md`

Per-group metrics:

- `...\baseline\runs\pathmnist\ipc10\B_T1024\metrics.json`
- `...\m12\runs\pathmnist\ipc10\M12_T1024_lam06_g2_lf256_p0\metrics.json`

Final synthetic data is recorded in each group's `condensed_path.txt`.

Eval best checkpoint is recorded in each group's `metrics.json`.

# BloodMNIST M12 Auto Tune 20260605

This adds an experiment scheduler only. It does not change the model, loss,
condense loop, data loader, evaluation logic, or M12 random-20-step local patch
implementation.

## Scheduler

`scripts/auto_tune_blood_m12_l20_20260605.py`

Purpose:

- keep both L20 GPUs occupied;
- run one independent `exp_root` per candidate;
- use the existing `run_blood_m12_rand20_single_20260603.py` entrypoint;
- collect each run's `metrics.json`;
- write a live status file and leaderboard.

## Output Root

`C:\xxyProject\NCFMproject_0603\e\btune0605`

Important files:

- `AUTO_TUNE_STATUS.json`
- `reports\blood_auto_tune_m12_seed0_20260605.json`
- `reports\blood_auto_tune_m12_seed0_20260605.md`
- `launcher_logs\auto_tune_events.log`
- `launcher_logs\<GROUP>_gpu<GPU>_stdout.log`
- `launcher_logs\<GROUP>_gpu<GPU>_stderr.log`

Each candidate has its own output directory:

`...\btune0605\<GROUP>\`

## Search Space

The first queue is a focused local search around the best BloodMNIST evidence so
far:

- `T`: 512 / 1024
- `lambda_local_patch_ncfd`: 0.5 / 0.6 / 0.7 / 0.8 / 1.0
- `local_patch_grid`: 2 / 4
- `local_patch_encoder_blocks`: 1 / 2
- `local_patch_num_freqs`: 512 / 1024
- encoder source: `random_trained_step`
- `local_patch_model_num`: 20
- seed: 0

Baseline for deltas:

- `B_T1024`: ACC 90.12, BACC 0.9019

## Initial Candidate Order

1. `m12_t1024_l06_g4_b2`
2. `m12_t1024_l08_g2_b2`
3. `m12_t1024_l07_g4_b2`
4. `m12_t1024_l05_g2_b2`
5. `m12_t1024_l08_g4_b1`
6. `m12_t1024_l05_g4_b1`
7. `m12_t1024_l08_g4_b2_lf1024`
8. `m12_t512_l08_g4_b2`
9. `m12_t512_l08_g2_b2`
10. `m12_t1024_l10_g4_b2`

## Parallelism

The scheduler uses `SLOTS_PER_GPU = 2`. Existing project runner processes count
as occupied slots, so it can add work without stopping currently running jobs.

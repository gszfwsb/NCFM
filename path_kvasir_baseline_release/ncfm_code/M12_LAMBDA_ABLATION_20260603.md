# M12 Lambda Ablation 20260603

## Change

The BloodMNIST M12 runner now accepts method parameters instead of hard-coding the single `lambda=0.3` group.

Updated script:

```text
scripts/run_blood_m12_rand20_single_20260603.py
```

New arguments:

```text
--lambda_local_patch_ncfd
--local_patch_grid
--local_patch_encoder_blocks
--group_name
```

If `--group_name` is omitted, the script creates a stable group name:

```text
M12_T{num_freqs}_lam{lambda_tag}_g{grid}_b{blocks}_rand20
```

Examples:

```text
M12_T1024_lam03_g4_b2_rand20
M12_T1024_lam08_g4_b2_rand20
```

## Reason

The running `lambda=0.3` M12 job tests whether random-step patch encoder sampling is useful.
The next minimal ablation is `lambda=0.8` with the same T/grid/blocks/encoder-bank setup.
This isolates the strength of the local patch NCFD term without changing the method.

## Current Planned Runs

| Group | Dataset | Seed | T | Lambda | Grid | Blocks | Encoder Source |
|---|---|---:|---:|---:|---:|---:|---|
| M12_T1024_lam03_g4_b2_rand20 | BloodMNIST | 0 | 1024 | 0.3 | 4 | 2 | random_trained_step |
| M12_T1024_lam08_g4_b2_rand20 | BloodMNIST | 0 | 1024 | 0.8 | 4 | 2 | random_trained_step |

## Output Roots

```text
C:\xxyProject\NCFMproject_0603\e\M12_T1024_rand20_seed0_0603
C:\xxyProject\NCFMproject_0603\e\M12_T1024_rand20_lam08_seed0_0603
```

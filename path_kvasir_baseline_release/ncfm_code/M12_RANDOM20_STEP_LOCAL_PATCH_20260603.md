# M12 Random20-Step Local Patch NCFD

This method is copied from M09 and changes only the local patch encoder
sampling policy.

## Motivation

M09 uses a fixed local patch encoder for the full condense run:

```yaml
local_patch_encoder_source: premodel_trained
local_patch_premodel_index: 0
```

That makes the local patch loss sensitive to a single premodel. M12 follows the
original NCFM feature extractor sampling idea more closely:

```text
for each distillation iteration:
    global NCFM samples one hybrid premodel as before
    local patch NCFD samples one trained patch encoder from 20 premodels
    image is split into patches
    selected Ek extracts patch features
    local patch NCFD is computed
    L = L_NCFM + lambda_local * L_local_patch
```

## New Config

Use this source:

```yaml
use_local_patch_feature_ncfd: true
lambda_local_patch_ncfd: 0.3

local_patch_grid: 4
local_patch_num_freqs: 512
local_patch_loss_scale: 300.0
local_patch_feature_dim: 0

local_patch_encoder_blocks: 2
local_patch_encoder_source: random_trained_step
local_patch_model_num: 20
local_patch_encoder_seed: 0
```

Optional restriction for ablations:

```yaml
local_patch_premodel_indices: [0, 1, 2, 3]
```

If `local_patch_premodel_indices` is omitted, M12 samples from
`0..local_patch_model_num-1`.

## Difference From Existing Sources

| Source | Behavior |
|---|---|
| `premodel_trained` | fixed single trained premodel selected by `local_patch_premodel_index` |
| `random_trained` | randomly chooses one premodel at startup, then stays fixed |
| `ensemble_trained` | uses multiple premodels and averages/concats features every forward |
| `random_trained_step` | preloads a bank and samples one trained premodel per distillation iteration |

## Logs

M12 adds component logging to `condense_stdout.log`:

```text
lp-global-loss
lp-local-loss
lp-weighted-local-loss
lp-total-loss
lp-encoder-index
```

These fields show whether the local patch term is active, its relative scale,
and which premodel index was sampled for that iteration.

## Edited Files

- `condenser/local_patch_ncfd.py`
  - adds `LocalPatchEncoderBank`
  - adds `random_trained_step`
  - adds `resample_local_patch_encoder`
- `condenser/compute_loss.py`
  - samples one local patch encoder per iteration
  - records local/global loss components
- `condenser/Condenser.py`
  - prints local patch component logs


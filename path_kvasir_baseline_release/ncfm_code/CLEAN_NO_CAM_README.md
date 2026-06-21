# Clean Baseline + DAM, CAM Removed from Main Pipeline

This folder is copied from `04_from_01_clean_cam_initial_code` and reorganized so that CAM is no longer part of the main baseline or DAM method pipeline.

## Main principle

- Baseline training uses only NCFM/NCFD feature distribution matching.
- DAM training uses NCFM/NCFD plus DataDAM-style spatial attention matching.
- CAM/Grad-CAM is only an optional post-hoc diagnostic tool.

## Main pipeline

The clean main path is:

```text
pretrain/pretrain_script.py
  -> condense/condense_script.py
  -> condenser/Condenser.py
  -> condenser/compute_loss.py
  -> NCFM/NCFM.py
  -> evaluation/evaluation_script.py
```

The formal and quick MedMNIST scripts in `scripts/` now run only:

```text
pretrain -> condense -> evaluation -> metrics summary
```

They do not call CAM.

## DAM method

DAM code is kept in:

```text
NCFM/datadam_attention.py
```

When `dam_enabled: true`, the inner loss becomes:

```text
L_total = dam_feature_weight * L_NCFM + dam_attention_weight * L_attention
```

With the default method config:

```text
L_total = L_NCFM + 10 * L_attention
```

This does not depend on CAM utilities.

## CAM diagnostics

CAM files were moved out of the main path into:

```text
diagnostics/cam/
```

This includes:

```text
diagnostics/cam/run_cam.py
diagnostics/cam/cam_utils.py
diagnostics/cam/create_cam_analysis_notebooks.py
diagnostics/cam/notebooks/
diagnostics/cam/docs/
```

Use these only after training/evaluation if you want qualitative spatial diagnostics.

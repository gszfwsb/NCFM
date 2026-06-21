# M22 Discrepancy-Guided Token Attention NCFD

M22 is a clean relaunch of the existing M15 discrepancy-guided attention idea on
top of the stronger M16 feature-token codebase.

## Why This Exists

DAM matches attention-map shapes and the M20/M21 diagnostics showed that even
when its gradient is strengthened, it remains mostly orthogonal to global NCFM.
M22 instead uses attention to choose where feature-token distribution mismatch
is largest, then directly optimizes NCFD at those positions.

## Objective

For a selected feature layer `l`, spatial position `u`, and feature map tokens:

```text
d_u^l = NCFD(F_real^l[:, :, u], F_syn^l[:, :, u])
a_u^l = softmax(stopgrad(d_u^l) / tau)
L_DGTA = mean_l sum_u a_u^l d_u^l
L_total = L_global_NCFM + lambda_DGTA * scale * L_DGTA
```

This is not attention-map matching. It is discrepancy-guided weighting of local
feature-token NCFD.

## Key Ablation

`uniform` mode is important. It tests whether gains come from per-position local
NCFD itself or from discrepancy-guided attention weights.

## First PathMNIST Seed0 Sweep

- global `num_freqs=1024`
- layer `[1]`
- local/token `num_freqs=256`
- groups:
  - `M22_lam01_tau01_L1_nf256`
  - `M22_lam02_tau01_L1_nf256`
  - `M22_lam03_tau01_L1_nf256`
  - `M22_lam02_tau005_L1_nf256`
  - `M22_lam02_tau02_L1_nf256`
  - `M22_lam02_tau01_L1_top4_nf256`
  - `M22_lam02_uniform_L1_nf256`

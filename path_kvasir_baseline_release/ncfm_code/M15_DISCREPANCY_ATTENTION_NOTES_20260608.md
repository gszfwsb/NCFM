# M15 Discrepancy-Guided Spatial Attention NCFM

## Purpose

M15 is an attention-first baseline extension. It does not use an external teacher, CAM, or ViT prior.

The attention map is generated from the current real-vs-synthetic feature mismatch:

```text
d_p = NCFD(F_real[:, :, p], F_syn[:, :, p])
a_p = softmax(stopgrad(d_p) / tau)
L_att = sum_p a_p d_p
L_total = L_global_NCFM + lambda_att L_att
```

This tests whether distillation benefits from focusing on spatial locations where synthetic data is still least similar to real data.

## Difference From Earlier Attention Methods

| Method | Attention source | What it tests |
|---|---|---|
| DAM | Feature activation prototype | Whether real/syn activation maps should look similar |
| M14 Soft ROI | Frozen ViT teacher rollout | Whether external real-image ROI prior helps |
| M15 DGSA | Current NCFD discrepancy | Whether real/syn mismatch itself can guide focus |

## Code Changes

Added:

```text
NCFM/discrepancy_attention.py
```

Modified:

```text
condenser/compute_loss.py
condenser/Condenser.py
```

## Config Switches

Enable M15:

```yaml
use_discrepancy_attention_ncfd: true
lambda_discrepancy_attention_ncfd: 0.05
discrepancy_attention_layers: [1]
discrepancy_attention_mode: softmax
discrepancy_attention_tau: 1.0
discrepancy_attention_num_freqs: 256
discrepancy_attention_loss_scale: 300.0
discrepancy_attention_detach_real: true
```

Control ablations:

```yaml
discrepancy_attention_mode: uniform
discrepancy_attention_mode: random
```

These are important controls. If softmax discrepancy attention does not beat uniform/random, the attention selection itself is not useful.

## Logging

Condense logs now include:

```text
dgsa-global-loss
dgsa-att-loss
dgsa-weighted-att-loss
dgsa-total-loss
dgsa-entropy
dgsa-maxw
```

Interpretation:

| Field | Meaning |
|---|---|
| dgsa-att-loss | spatial discrepancy loss before lambda |
| dgsa-weighted-att-loss | lambda-scaled auxiliary loss |
| dgsa-entropy | attention spread; lower means sharper focus |
| dgsa-maxw | largest spatial attention weight |

## First Suggested Sweep

For BloodMNIST with global `T=1024`:

| Group | lambda | tau | mode | layers |
|---|---:|---:|---|---|
| M15_lam002_tau1_L1 | 0.02 | 1.0 | softmax | [1] |
| M15_lam005_tau1_L1 | 0.05 | 1.0 | softmax | [1] |
| M15_lam010_tau1_L1 | 0.10 | 1.0 | softmax | [1] |
| M15_lam005_tau05_L1 | 0.05 | 0.5 | softmax | [1] |
| M15_uniform_lam005_L1 | 0.05 | 1.0 | uniform | [1] |
| M15_random_lam005_L1 | 0.05 | 1.0 | random | [1] |

Main comparison:

```text
Baseline NCFM T1024
vs
M15 softmax discrepancy attention
vs
uniform spatial NCFD
vs
random spatial attention
```

The desired evidence pattern is:

```text
M15 softmax > uniform >= baseline > random
```

or at least:

```text
M15 softmax > baseline and M15 softmax > uniform/random
```

# MethodDR-LTM / MCVaR

## Method

Distributionally Robust Local Token Matching (DR-LTM) replaces the heuristic
softmax aggregation used by M22 with a CVaR / top-tail objective over local
feature-token NCFD discrepancies.

For each selected feature layer and spatial position:

\[
d_u = \mathrm{NCFD}(Q^r_u, Q^s_u)
\]

where \(Q^r_u\) and \(Q^s_u\) are real and synthetic feature-token
distributions at spatial position \(u\).

The DR-LTM branch optimizes:

\[
L_{\mathrm{DR}} = \mathrm{CVaR}_\alpha(d_1,\dots,d_m)
\]

The first implementation uses the top-tail estimate:

\[
L_{\mathrm{DR}} =
\frac{1}{\lceil \alpha m\rceil}
\sum_{u\in \mathrm{TopK}(d)}
d_u
\]

Total loss:

\[
L_{\mathrm{total}} = L_{\mathrm{NCFM}} + \lambda_{\mathrm{DR}}L_{\mathrm{DR}}
\]

## First PathMNIST Seed0 Sweep

| Group | lambda | alpha | layer | nf | meaning |
|---|---:|---:|---|---:|---|
| DRLTM_lam01_a100_L1_nf256 | 0.1 | 1.00 | [1] | 256 | local average risk, weak |
| DRLTM_lam02_a100_L1_nf256 | 0.2 | 1.00 | [1] | 256 | local average risk, stronger |
| DRLTM_lam02_a050_L1_nf256 | 0.2 | 0.50 | [1] | 256 | top 50% tail |
| DRLTM_lam02_a025_L1_nf256 | 0.2 | 0.25 | [1] | 256 | top 25% tail |
| DRLTM_lam02_a010_L1_nf256 | 0.2 | 0.10 | [1] | 256 | top 10% hard tail |
| DRLTM_lam03_a025_L1_nf256 | 0.3 | 0.25 | [1] | 256 | stronger top 25% tail |

## Key Files

- `NCFM/dr_ltm.py`: local discrepancy computation and CVaR/top-tail aggregation.
- `condenser/compute_loss.py`: connects `use_dr_ltm_ncfd` to the NCFM training loss.
- `condenser/Condenser.py`: logs DR-LTM loss and local discrepancy statistics.
- `scripts/run_pathmnist_drl_tm_seed0.py`: six-group PathMNIST seed0 runner.
- `scripts/run_drl_tm_tail_diagnostics.py`: post-hoc support / coverage / rank diagnostics.

## Remote Placement

Code:

`/data/zengqiang/experiments/NCFMproject_0603/active_code/MCVaR_DR_LTM/code`

Experiment output:

`/data/zengqiang/experiments/NCFMproject_0603/experiments/pathmnist_mcvardr_ltm_seed0_20260617`

## Diagnostic Outputs

Training logs include:

- `drltm-mean-d`
- `drltm-max-d`
- `drltm-p90-d`
- `drltm-cvar-d`
- `drltm-tail-ratio`
- `drltm-cv`

Post-hoc diagnostics produce:

- `local_discrepancy_stats.csv`
- `support_coverage_rank.csv`
- `diagnostics_summary.json`


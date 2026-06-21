# MedMNIST CAM and Multi-Scale NCFD Plan

**Repo baseline**: pure upstream NCFM plus MedMNIST data adapter.
**Goal**: decide what to build next, what to test first, and what must stay out of the clean baseline.
**Status**: planning document, no algorithm implementation yet.

## 1. Direction

We want two related but separate tracks:

1. **CAM visualization track**
   - Add class activation map visualizations for MedMNIST-trained evaluation models.
   - Use CAM to inspect whether models trained on condensed images attend to plausible medical regions.
   - Treat CAM as diagnosis and qualitative evidence first, not as part of the loss.

2. **Multi-scale NCFD track**
   - Add an experimental loss variant for MedMNIST small images.
   - Preserve original NCFM as the control.
   - Test whether local distribution alignment helps preserve small diagnostic structures that global NCFD can miss.

Important boundary: the current clean repo should remain the pure NCFM + MedMNIST interface baseline. CAM and multi-scale NCFD should be developed on a new experimental branch, for example `exp/medmnist-cam-multiscale-ncfd`.

## 2. Claims To Test

| Claim | Why It Matters | Minimum Evidence |
|---|---|---|
| C1: Original NCFM under-preserves local diagnostic evidence on tiny medical images | This motivates multi-scale NCFD rather than just adding MedMNIST support | CAM or error analysis shows weak/localized attention on original NCFM condensed data; baseline metrics lag on small-lesion datasets |
| C2: Local NCFD improves MedMNIST condensation without breaking global class structure | This is the main algorithmic claim | Accuracy/AUC improves over pure NCFM on several 2D MedMNIST tasks, with ablation showing local term matters |
| C3: CAM is useful for auditing condensed data quality | This supports interpretability, not the core algorithm | CAM figures show interpretable differences between real, pure NCFM, and multi-scale NCFD-trained models |

Anti-claim to rule out: gains are not just from more hyperparameter tuning, more frequencies, or longer evaluation training.

## 3. Scope

### In Scope First

- 2D MedMNIST datasets with 28x28 images.
- Initial datasets:
  - `pathmnist`: multi-class, RGB, texture/pathology-heavy.
  - `bloodmnist`: multi-class, RGB, good for general 2D sanity.
  - `pneumoniamnist`: binary, grayscale, small localized signals.
  - Optional after sanity: `breastmnist`, `retinamnist`, `dermamnist`, `octmnist`, `organamnist`.
- IPC=10 first.
- ConvNet evaluation first, matching original NCFM style.

### Out Of Scope Until Later

- 3D MedMNIST 28x28x28 tasks.
- 3D CAM.
- 3D NCFD.
- CLIP/BiomedCLIP alignment.
- Token-local and raw-patch variants from old worktrees unless reintroduced as explicit ablations.

Reason: 3D MedMNIST is not a small adapter change. It needs 3D data tensors, 3D networks, 3D decode, 3D augmentation, and 3D CAM. Treat it as phase 2.

## 4. Proposed Implementation Blocks

### Block A: CAM Visualization

Purpose: understand what a model trained on condensed data uses.

Implementation options:

- Use Grad-CAM first.
- Target final convolution layer of ConvNet.
- Generate heatmaps for:
  - model trained on real data,
  - model trained on original NCFM condensed data,
  - model trained on multi-scale NCFD condensed data.

Outputs:

- Per-class CAM grids.
- Correct vs incorrect prediction examples.
- Side-by-side real image, synthetic image, CAM overlay.

Do not make CAM part of training yet.

### Block B: Global NCFD Hyperparameter Adaptation

Purpose: test whether small-image gains come from simple NCFD tuning before adding local loss.

Variants:

- `alpha_for_loss`: 0.5 baseline, then 0.4, 0.3.
- `num_freqs`: 4096 original config, then 1024, 512, 256.
- `sampling_net`: keep original default first; only tune if the baseline is stable.

Note: the prompt mentions 1024 as original, but the local config currently uses 4096 in upstream-style YAML. We should record the actual active value per run.

### Block C: Local NCFD Loss

Purpose: align local diagnostic structure.

Candidate variants:

- Feature-token local NCFD:
  - use ConvNet feature map tokens, e.g. `[B, C, H, W] -> [B, H*W, C]`;
  - compute NCFD per spatial token and average.
  - This is closer to "local feature distribution" and less raw-pixel-biased.

- Raw image patch NCFD:
  - split image into non-overlapping patches;
  - for 28x28, test 2x2, 4x4, and 7x7 grids;
  - compute NCFD per patch and average.
  - This is simpler to reason about but can overfit to pixel statistics.

First implementation should pick one local mechanism, not both. Recommended first: feature-token local NCFD, because it keeps the loss in NCFM's feature-matching spirit.

### Block D: SSIM Regularization

Purpose: test whether a light structural regularizer improves visual fidelity.

Important caveat:

- Direct SSIM between arbitrary real and synthetic batches is not naturally paired.
- Need a defined pairing rule, such as class-wise batch pairing after sampling real and synthetic data for the same class.
- SSIM should be a small regularizer, not the main matching objective.

Recommended first:

- Implement SSIM only after local NCFD is stable.
- Start with `lambda_ssim = 0.0, 0.02, 0.05, 0.1`.
- Use class-wise real/synthetic pairs from the same batch.

## 5. Pre-Experiments

| ID | Purpose | Dataset | Runs | Success Gate |
|---|---|---|---|---|
| P0 | Data pipeline sanity | `bloodmnist`, `pneumoniamnist` | Load train/test, inspect shape, class counts, mean/std cache | No shape/channel mismatch; labels are ints; nclass/nch match config |
| P1 | Pure NCFM smoke run | `bloodmnist`, IPC=1 or short `niter` | 50-200 condense iterations, no result claim | Training loop runs; synthetic tensor saves; evaluation loader works |
| P2 | Real-data CAM sanity | `bloodmnist`, `pneumoniamnist` | Train/evaluate model on real data, generate Grad-CAM | CAM overlays are nonblank and aligned to image extent |
| P3 | Pure NCFM CAM audit | `bloodmnist`, `pneumoniamnist` | Train model on pure NCFM synthetic data, generate CAM | CAMs can be compared to real-data CAM; failures are visible |
| P4 | Frequency/alpha sweep | `pathmnist`, `pneumoniamnist` | Small grid: alpha 0.5/0.4/0.3 x num_freqs 256/512/1024 | Identify whether simple tuning already improves baseline |
| P5 | Local NCFD smoke | `pneumoniamnist` | local weight 0.3/0.6, one block strategy | No instability; local loss has nonzero gradients; runtime acceptable |
| P6 | Patch/token ablation | `pathmnist`, `pneumoniamnist` | global only vs local feature-token vs raw patch | Decide which local form deserves full runs |
| P7 | SSIM pilot | `pneumoniamnist` | with/without small SSIM weight | SSIM does not collapse images or harm accuracy severely |

Must-run first: P0, P1, P2, P3.

Do not run full grid sweeps until P0-P3 are clean.

## 6. Main Experiment Matrix

After pre-experiments pass, run a compact matrix:

| System | Description |
|---|---|
| Real upper reference | Train evaluation model on real MedMNIST train set |
| Original NCFM | Pure upstream NCFM + MedMNIST adapter |
| Tuned global NCFD | Original NCFM with selected alpha/frequency settings |
| Multi-scale NCFD | Global + local NCFD |
| Multi-scale NCFD + SSIM | Only if SSIM pilot is stable |

Datasets:

- Main: `pathmnist`, `bloodmnist`, `pneumoniamnist`.
- Extension: `breastmnist`, `retinamnist`, `dermamnist`, `octmnist`, `organamnist`.

Metrics:

- Primary: test accuracy.
- Secondary: AUC if added as logging only.
- Diagnostics: CAM quality, runtime per iteration, memory, synthetic image grids.

Seeds:

- Smoke: 1 seed.
- Main table: 3 seeds if compute allows.

## 7. Design Decisions To Make Before Coding

1. Should local NCFD use feature tokens or raw image patches first?
   - Recommendation: feature tokens first.

2. Should AUC be added now?
   - Recommendation: yes for MedMNIST reporting, but logging only. Do not use AUC for selecting checkpoints until the baseline is stable, because that changes the training protocol.

3. Should CAM be generated from evaluation models or teacher/pretrain models?
   - Recommendation: evaluation models trained on each synthetic dataset. That directly audits the data condensation result.

4. Should 3D be included?
   - Recommendation: no in phase 1. Create a separate phase 2 plan after 2D results exist.

5. Should SSIM be part of the first algorithm change?
   - Recommendation: no. Add after local NCFD works.

## 8. Implementation Order

1. Create branch `exp/medmnist-cam-multiscale-ncfd`.
2. Add CAM utility only:
   - `visualization/grad_cam.py`
   - `scripts/visualize_medmnist_cam.py`
3. Run P0-P3 and save qualitative outputs.
4. Add optional AUC logging if needed.
5. Implement one local NCFD variant behind a config flag:
   - keep original NCFM default path unchanged;
   - no local loss unless explicitly enabled.
6. Run P4-P6.
7. Add SSIM pilot only after P6.
8. Freeze final method and run main matrix.

## 9. Risks

| Risk | Why It Matters | Mitigation |
|---|---|---|
| CAM looks plausible but does not prove causal attention | Reviewers may overread heatmaps | Present CAM as qualitative audit only; keep metrics primary |
| Local loss overfits texture/noise | Tiny images can be dominated by pixel artifacts | Compare feature-token vs raw patch; monitor synthetic grids |
| SSIM requires pairing and may be ill-defined for condensation | Real/synthetic samples are not paired naturally | Use class-wise batch pairing and keep small weight |
| Frequency/alpha tuning explains all gains | Then multi-scale contribution weakens | Run tuned global NCFD before claiming local NCFD |
| 3D scope explosion | 3D requires architecture and data pipeline changes | Keep 3D as separate phase |

## 10. Immediate Next Steps

1. Implement Grad-CAM visualization for ConvNet evaluation models.
2. Run P0 data sanity on `bloodmnist` and `pneumoniamnist`.
3. Run a short pure NCFM condensation smoke test.
4. Generate first CAM comparisons: real-trained vs pure-NCFM-trained.
5. Only then implement local NCFD.


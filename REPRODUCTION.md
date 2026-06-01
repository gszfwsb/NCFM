# NCFM Table 1 Evaluation Reproduction

本复现只统计 `val_repeat=10` 的均值。验收口径是：均值与表格目标值的误差在 `+-1` 个百分点以内。

## Scope

这是 evaluation 复现：使用 Hugging Face 提供的 distilled datasets，运行
`evaluation/evaluation_script.py`。不是从头重新 condense 全部数据集。

工作目录：

```bash
cd /mnt/cpfs/yangyicun/NCFM
```

环境：

```text
GPU:         8 x NVIDIA A100-SXM4-80GB
Python:      3.12.13
torch:       2.5.0+cu124
torchvision: 0.20.0+cu124
CUDA:        12.4
```

重建环境：

```bash
bash setup_train_env_uv.sh --torch-backend cu124 --python 3.12
source .venv/bin/activate
```

## Results

| Dataset | IPC | Target | val_repeat | Mean | Delta | Evidence |
|---|---:|---:|---:|---:|---:|---|
| CIFAR-10 | 1 | 49.5 | 10 | 48.598 | -0.902 | `results/tuning_stdout/c10_i1_wd1e-2_lr6e-4_eval20_vr10_g4-7.out` |
| CIFAR-10 | 10 | 71.8 | 10 | 70.808 | -0.992 | `results/tuning_stdout/c10_i10_wd1e-2_bs256_ep4000_eval20_vr10_g4-7.out` |
| CIFAR-10 | 50 | 77.4 | 10 | 77.453 | +0.053 | `results/condense/evaluate/cifar10/ipc50/_lr0.0100__factor2_20260526-1420/print.log` |
| CIFAR-100 | 1 | 34.4 | 10 | 34.384 | -0.016 | `results/tuning_stdout/c100_i1_adam_lr1e-3_wd1e-2_4g_seed3_vr10_g4-7.out` |
| CIFAR-100 | 10 | 48.7 | 10 | 49.547 | +0.847 | `results/condense/evaluate/cifar100/ipc10/_lr0.0100__factor2_20260526-1459/print.log` |
| CIFAR-100 | 50 | 54.7 | 10 | 54.224 | -0.476 | `results/tuning_stdout/c100_i50_ep1000_vr10_g0-7.out` |
| Tiny ImageNet | 1 | 18.2 | 10 | 17.329 | -0.871 | `results/tuning_stdout/tiny_i1_soft_adam_wd1e-2_lr4e-4_cf_vr10_g0-1.out` |
| Tiny ImageNet | 10 | 26.8 | 10 | 26.171 | -0.629 | `results/tuning_stdout/tiny_i10_softlabel_lr5e-4_vr10_g2-5.out` |
| Tiny ImageNet | 50 | 29.6 | 10 | 28.887 | -0.713 | `results/tuning_stdout/tiny_i50_softlabel_ep30_lr5e-4_vr10_g6-7.out` |

## Data

Distilled datasets:

```text
hf_data/NCFM_distillation_dataset/CIFAR-10/CIFAR10_ipc1.pt
hf_data/NCFM_distillation_dataset/CIFAR-10/CIFAR10_ipc10.pt
hf_data/NCFM_distillation_dataset/CIFAR-10/CIFAR10_ipc50.pt
hf_data/NCFM_distillation_dataset/CIFAR-100/CIFAR100_ipc1.pt
hf_data/NCFM_distillation_dataset/CIFAR-100/CIFAR100_ipc10.pt
hf_data/NCFM_distillation_dataset/CIFAR-100/CIFAR100_ipc50.pt
hf_data/NCFM_distillation_dataset/Tiny ImageNet/TinyImageNet_ipc1.pt
hf_data/NCFM_distillation_dataset/Tiny ImageNet/TinyImageNet_ipc10.pt
hf_data/NCFM_distillation_dataset/Tiny ImageNet/TinyImageNet_ipc50.pt
```

Validation datasets are under `dataset/`. Tiny ImageNet soft-label evaluation also needs:

```text
hf_data/pretrained_model/tinyimagenet/premodel0_trained.pth.tar
```

## Run

Run all 9 rows sequentially:

```bash
bash tools/run_table1_repro.sh all
```

Run one dataset group:

```bash
bash tools/run_table1_repro.sh cifar10
bash tools/run_table1_repro.sh cifar100
bash tools/run_table1_repro.sh tinyimagenet
```

The script calls `tools/ncfm_eval_tuner.py`, which writes generated configs to
`results/tuning_configs/`, stdout summaries to `results/tuning_stdout/`, and print logs to
`results/tuning/`.

## Configs

The configs under `config/ipc*/` have been updated to the passing settings:

- CIFAR-10 IPC1: `evaluation_epochs=2000`, `epoch_eval_interval=20`, `adamw_lr=0.0006`, `weight_decay=0.01`.
- CIFAR-10 IPC10: `evaluation_epochs=4000`, `epoch_eval_interval=20`, `batch_size=256`, `weight_decay=0.01`.
- CIFAR-100 IPC50: `evaluation_epochs=1000`.
- Tiny ImageNet IPC1/10/50: soft-label evaluation with `pretrain_dir=../hf_data/pretrained_model`; IPC10 uses `adamw_lr=0.0005`.

## Notes

`Repeat x/10 => The Best Evaluation Acc` in the original logger prints the global best so far, not the
current repeat only. The authoritative values are the final `Mean Accuracy` and `All result` lines.

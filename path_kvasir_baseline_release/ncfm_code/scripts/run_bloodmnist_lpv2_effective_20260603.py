import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


DATASET = "bloodmnist"
IPC = 10
NCLASS = 8
NCH = 3


GROUPS = {
    "LPv2_T1024_lam03_g4_b2_p0": {
        "lambda_local_patch_ncfd": 0.3,
        "local_patch_grid": 4,
        "local_patch_encoder_blocks": 2,
        "local_patch_encoder_source": "premodel_trained",
        "local_patch_premodel_index": 0,
    },
    "LPv2_T1024_lam08_g4_b2_p0": {
        "lambda_local_patch_ncfd": 0.8,
        "local_patch_grid": 4,
        "local_patch_encoder_blocks": 2,
        "local_patch_encoder_source": "premodel_trained",
        "local_patch_premodel_index": 0,
    },
    "LPv2_T1024_lam03_g7_b1_p0": {
        "lambda_local_patch_ncfd": 0.3,
        "local_patch_grid": 7,
        "local_patch_encoder_blocks": 1,
        "local_patch_encoder_source": "premodel_trained",
        "local_patch_premodel_index": 0,
    },
    "LPv2_T1024_lam08_g7_b1_p0": {
        "lambda_local_patch_ncfd": 0.8,
        "local_patch_grid": 7,
        "local_patch_encoder_blocks": 1,
        "local_patch_encoder_source": "premodel_trained",
        "local_patch_premodel_index": 0,
    },
    "LPv2_T1024_lam03_g4_b1_ens0123": {
        "lambda_local_patch_ncfd": 0.3,
        "local_patch_grid": 4,
        "local_patch_encoder_blocks": 1,
        "local_patch_encoder_source": "ensemble_trained",
        "local_patch_premodel_indices": [0, 1, 2, 3],
        "local_patch_ensemble_aggregate": "mean",
    },
    "LPv2_T1024_lam08_g4_b1_ens0123": {
        "lambda_local_patch_ncfd": 0.8,
        "local_patch_grid": 4,
        "local_patch_encoder_blocks": 1,
        "local_patch_encoder_source": "ensemble_trained",
        "local_patch_premodel_indices": [0, 1, 2, 3],
        "local_patch_ensemble_aggregate": "mean",
    },
}


def run_command(command, cwd, stdout_path, stderr_path):
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open(
        "w", encoding="utf-8"
    ) as err:
        proc = subprocess.run(command, cwd=cwd, stdout=out, stderr=err, text=True)
    return proc.returncode


def torchrun_command(script, config_path, gpu, ipc=None, extra=None):
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=1",
        script,
        "--config_path",
        str(config_path),
        "--gpu",
        str(gpu),
    ]
    if ipc is not None:
        command.extend(["-i", str(ipc)])
    if extra:
        command.extend(extra)
    return command


def write_config(path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def newest_distilled_data(exp_root, dataset, ipc, start_time):
    root = exp_root / "results" / "condense"
    candidates = []
    if root.exists():
        for path in root.glob(f"**/{dataset}/ipc{ipc}/**/distilled_data/data_*.pt"):
            if path.name == "data_init.pt":
                continue
            if path.stat().st_mtime >= start_time:
                candidates.append(path)
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def latest_distilled_data(exp_root, dataset, ipc):
    root = exp_root / "results" / "condense"
    candidates = []
    if root.exists():
        for path in root.glob(f"**/{dataset}/ipc{ipc}/**/distilled_data/data_*.pt"):
            if path.name != "data_init.pt":
                candidates.append(path)
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def read_best_metrics(metrics_jsonl):
    best_path = metrics_jsonl.with_name(metrics_jsonl.stem + "_best.json")
    if best_path.exists():
        return json.loads(best_path.read_text(encoding="utf-8"))
    if not metrics_jsonl.exists():
        return {}
    best = {}
    for line in metrics_jsonl.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("is_best"):
            best = record
    return best


def make_config(args, group_name, metrics_path=None):
    group = GROUPS[group_name]
    cfg = {
        "distibution_train": {
            "backend": "nccl",
            "init_method": "env://",
            "workers": args.workers,
        },
        "dataset": {
            "dataset": DATASET,
            "nclass": NCLASS,
            "size": 28,
            "data_dir": str(args.exp_root / "data"),
            "load_memory": True,
            "batch_real": args.batch_real,
            "nch": NCH,
        },
        "network": {
            "net_type": "convnet",
            "norm_type": "instance",
            "depth": 3,
            "width": 1.0,
        },
        "train": {
            "evaluation_epochs": args.eval_epochs,
            "epoch_print_freq": 10,
            "epoch_eval_interval": args.epoch_eval_interval,
            "pertrain_epochs": args.pretrain_epochs,
            "batch_size": args.batch_size,
            "lr": 0.01,
            "adamw_lr": 0.001,
            "eval_optimizer": "adamw",
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "seed": args.seed,
            "model_num": args.model_num,
        },
        "augmentation": {
            "mixup": "cut",
            "beta": 1.0,
            "mix_p": 0.5,
            "rrc": True,
            "dsa": True,
            "dsa_strategy": "color_crop_cutout_flip_scale_rotate",
            "aug_type": "color_crop_cutout",
        },
        "optimization": {
            "optimizer": "adamw",
            "lr_scale_adam": 0.1,
            "lr_img": 0.01,
            "mom_img": 0.5,
            "lr_sampling_net": 1e-3,
        },
        "save_path": {
            "save_dir": str(args.exp_root / "results" / "condense"),
            "pretrain_dir": str(args.exp_root / "checkpoints" / "pretrain"),
        },
        "condense": {
            "ipc": IPC,
            "num_premodel": args.model_num,
            "niter": args.niter,
            "iter_calib": 0,
            "calib_weight": 1,
            "sampling_net": True,
            "num_freqs": 1024,
            "dis_metrics": "NCFM",
            "factor": 2,
            "alpha_for_loss": 0.5,
            "beta_for_loss": 0.5,
            "decode_type": "single",
            "teacher_model_epoch": 20,
            "use_local_patch_feature_ncfd": True,
            "local_patch_num_freqs": 512,
            "local_patch_loss_scale": 300.0,
            "local_patch_feature_dim": 0,
            "local_patch_model_num": args.model_num,
            **group,
        },
    }
    if metrics_path:
        cfg["records"] = {"metrics_path": str(metrics_path)}
    return cfg


def copy_assets(args):
    data_dst = args.exp_root / "data" / "medmnist"
    pretrain_dst = args.exp_root / "checkpoints" / "pretrain" / DATASET
    data_dst.mkdir(parents=True, exist_ok=True)
    pretrain_dst.mkdir(parents=True, exist_ok=True)

    data_src = Path(args.data_source)
    pretrain_src = Path(args.pretrain_source)
    if not (data_dst / "bloodmnist.npz").exists():
        shutil.copy2(data_src / "bloodmnist.npz", data_dst / "bloodmnist.npz")

    for path in pretrain_src.glob("premodel*_*.*"):
        dst = pretrain_dst / path.name
        if not dst.exists():
            shutil.copy2(path, dst)


def run_one(args, repo_dir, group_name):
    config_path = args.exp_root / "configs" / DATASET / f"ipc{IPC}_{group_name}.yaml"
    run_dir = args.exp_root / "runs" / DATASET / f"ipc{IPC}" / group_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_config(config_path, make_config(args, group_name))

    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not args.force:
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    condensed_record = run_dir / "condensed_path.txt"
    condensed_path = None
    if condensed_record.exists() and not args.force:
        candidate = Path(condensed_record.read_text(encoding="utf-8").strip())
        if candidate.exists():
            condensed_path = candidate

    if condensed_path is None:
        start_time = datetime.now().timestamp()
        condense_cmd = torchrun_command(
            "condense/condense_script.py",
            config_path,
            args.gpu,
            ipc=IPC,
            extra=["--run_mode", "Condense", "--init", "mix"],
        )
        (run_dir / "condense_command.txt").write_text(
            " ".join(map(str, condense_cmd)) + "\n", encoding="utf-8"
        )
        rc = run_command(
            condense_cmd,
            repo_dir,
            run_dir / "condense_stdout.log",
            run_dir / "condense_stderr.log",
        )
        if rc != 0:
            raise RuntimeError(f"Condense failed for {group_name}; see {run_dir}")
        condensed_path = newest_distilled_data(args.exp_root, DATASET, IPC, start_time)
        if condensed_path is None:
            condensed_path = latest_distilled_data(args.exp_root, DATASET, IPC)

    if condensed_path is None:
        raise RuntimeError(f"No distilled data found for {group_name}")
    condensed_record.write_text(str(condensed_path) + "\n", encoding="utf-8")

    checkpoint_path = (
        args.exp_root
        / "checkpoints"
        / "synthetic_train"
        / DATASET
        / f"ipc{IPC}_{group_name}_best.pth.tar"
    )
    eval_metrics_path = run_dir / "eval_metrics.jsonl"
    eval_cmd = torchrun_command(
        "evaluation/evaluation_script.py",
        config_path,
        args.gpu,
        ipc=IPC,
        extra=[
            "--run_mode",
            "Evaluation",
            "--load_path",
            str(condensed_path),
            "--val_repeat",
            "1",
            "--eval_checkpoint_path",
            str(checkpoint_path),
            "--eval_metrics_path",
            str(eval_metrics_path),
        ],
    )
    (run_dir / "eval_command.txt").write_text(
        " ".join(map(str, eval_cmd)) + "\n", encoding="utf-8"
    )
    rc = run_command(
        eval_cmd,
        repo_dir,
        run_dir / "eval_stdout.log",
        run_dir / "eval_stderr.log",
    )
    if rc != 0:
        raise RuntimeError(f"Eval failed for {group_name}; see {run_dir}")

    best = read_best_metrics(eval_metrics_path)
    metrics = {
        "dataset": DATASET,
        "seed": args.seed,
        "ipc": IPC,
        "group": group_name,
        "niter": args.niter,
        "global_num_freqs": 1024,
        "local_patch_num_freqs": 512,
        "condensed_path": str(condensed_path),
        "checkpoint_path": str(checkpoint_path),
        **GROUPS[group_name],
        **best,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def save_summary(exp_root, metrics, prefix):
    report_dir = exp_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{prefix}.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    fields = [
        "dataset",
        "seed",
        "group",
        "acc_percent",
        "auc_macro_ovr",
        "macro_f1",
        "balanced_acc",
        "lambda_local_patch_ncfd",
        "local_patch_grid",
        "local_patch_encoder_blocks",
        "local_patch_encoder_source",
        "local_patch_premodel_index",
        "local_patch_premodel_indices",
        "checkpoint_path",
    ]
    with (report_dir / f"{prefix}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)

    rows = [
        "| Seed | Group | ACC | AUC | Macro-F1 | BACC | lambda | grid | blocks | source |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        rows.append(
            f"| {item.get('seed')} | {item.get('group')} | {item.get('acc_percent')} | "
            f"{item.get('auc_macro_ovr')} | {item.get('macro_f1')} | {item.get('balanced_acc')} | "
            f"{item.get('lambda_local_patch_ncfd')} | {item.get('local_patch_grid')} | "
            f"{item.get('local_patch_encoder_blocks')} | {item.get('local_patch_encoder_source')} |"
        )
    (report_dir / f"{prefix}.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def parse_groups(value):
    if value == "all":
        return list(GROUPS)
    groups = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(groups) - set(GROUPS))
    if unknown:
        raise ValueError(f"Unknown groups: {unknown}")
    return groups


def main():
    parser = argparse.ArgumentParser(description="BloodMNIST Local Patch v2 effective sweep.")
    parser.add_argument("--exp_root", type=Path, required=True)
    parser.add_argument("--data_source", required=True)
    parser.add_argument("--pretrain_source", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--groups", default="all")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--batch_real", type=int, default=1024)
    parser.add_argument("--model_num", type=int, default=20)
    parser.add_argument("--pretrain_epochs", type=int, default=60)
    parser.add_argument("--eval_epochs", type=int, default=2000)
    parser.add_argument("--epoch_eval_interval", type=int, default=100)
    parser.add_argument("--niter", type=int, default=20000)
    parser.add_argument("--summary_prefix", default="bloodmnist_lpv2_effective")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parents[1]
    args.exp_root.mkdir(parents=True, exist_ok=True)
    copy_assets(args)

    selected = parse_groups(args.groups)
    all_metrics = []
    for group_name in selected:
        print(f"RUN seed={args.seed} group={group_name} gpu={args.gpu}", flush=True)
        metrics = run_one(args, repo_dir, group_name)
        all_metrics.append(metrics)
        save_summary(args.exp_root, all_metrics, args.summary_prefix)

    save_summary(args.exp_root, all_metrics, args.summary_prefix)


if __name__ == "__main__":
    main()

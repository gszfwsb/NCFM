import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


DATASETS = {
    "pneumoniamnist": {"nclass": 2, "nch": 1},
    "bloodmnist": {"nclass": 8, "nch": 3},
    "pathmnist": {"nclass": 9, "nch": 3},
}

GROUPS = {
    "A_pure_ncfd_wopsi": {
        "sampling_net": False,
        "num_freqs": 1024,
        "iter_calib": 0,
        "calib_weight": 1,
    },
    "B_minmax_ncfm_psi": {
        "sampling_net": True,
        "num_freqs": 1024,
        "iter_calib": 0,
        "calib_weight": 1,
    },
    "C_code_default_enhanced": {
        "sampling_net": False,
        "num_freqs": 4096,
        "iter_calib": 1,
        "calib_weight": 1,
    },
}


def run_command(command, cwd, stdout_path, stderr_path):
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        proc = subprocess.run(command, cwd=cwd, stdout=out, stderr=err, text=True)
    return proc.returncode


def append(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def make_config(args, dataset, ipc, group_name=None, niter=None):
    info = DATASETS[dataset]
    group = GROUPS.get(group_name, GROUPS["A_pure_ncfd_wopsi"])
    return {
        "distibution_train": {
            "backend": "nccl",
            "init_method": "env://",
            "workers": args.workers,
        },
        "dataset": {
            "dataset": dataset,
            "nclass": info["nclass"],
            "size": 28,
            "data_dir": str(args.exp_root / "data"),
            "load_memory": True,
            "batch_real": args.batch_real,
            "nch": info["nch"],
        },
        "network": {
            "net_type": "convnet",
            "norm_type": "instance",
            "depth": 3,
            "width": 1.0,
        },
        "train": {
            "evaluation_epochs": args.eval_epochs,
            "epoch_print_freq": 1,
            "epoch_eval_interval": 1,
            "pertrain_epochs": args.pretrain_epochs,
            "batch_size": args.batch_size,
            "lr": 0.01,
            "adamw_lr": 0.001,
            "eval_optimizer": "adamw",
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "seed": 0,
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
            "save_dir": str(args.exp_root / "results"),
            "pretrain_dir": str(args.exp_root / "checkpoints" / "pretrain"),
        },
        "condense": {
            "ipc": ipc,
            "num_premodel": args.model_num,
            "niter": int(niter),
            "iter_calib": group["iter_calib"],
            "calib_weight": group["calib_weight"],
            "sampling_net": group["sampling_net"],
            "num_freqs": group["num_freqs"],
            "dis_metrics": "NCFM",
            "factor": 2,
            "alpha_for_loss": 0.5,
            "beta_for_loss": 0.5,
            "decode_type": "single",
            "teacher_model_epoch": 20,
        },
    }


def write_config(path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


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


def newest_distilled_data(exp_root, dataset, ipc, start_time):
    root = exp_root / "results" / "condense" / dataset / f"ipc{ipc}"
    candidates = []
    if root.exists():
        for path in root.glob("**/distilled_data/data_*.pt"):
            if path.name == "data_init.pt":
                continue
            if path.stat().st_mtime >= start_time:
                candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_best_accuracy(stdout_path):
    text = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
    matches = re.findall(r"Best\s+accuracy \(top-1 and 5\):\s*([0-9.]+)", text)
    if matches:
        return float(matches[-1])
    matches = re.findall(r"Mean Accuracy:\s*([0-9.]+)", text)
    if matches:
        return float(matches[-1])
    return None


def ensure_pretrain(args, repo_dir, dataset):
    config_path = args.exp_root / "configs" / dataset / "pretrain_quick.yaml"
    config = make_config(args, dataset, ipc=1, group_name="A_pure_ncfd_wopsi", niter=args.smoke_niter)
    write_config(config_path, config)
    expected = args.exp_root / "checkpoints" / "pretrain" / dataset / f"premodel{args.model_num - 1}_trained.pth.tar"
    run_dir = args.exp_root / "runs" / dataset / "pretrain_quick"
    if expected.exists() and not args.force:
        return expected
    command = torchrun_command("pretrain/pretrain_script.py", config_path, args.gpu)
    (run_dir / "command.txt").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    rc = run_command(command, repo_dir, run_dir / "stdout.log", run_dir / "stderr.log")
    if rc != 0:
        raise RuntimeError(f"Pretrain failed for {dataset}; see {run_dir}")
    return expected


def run_condense_eval(args, repo_dir, dataset, ipc, group_name, niter):
    config_dir = args.exp_root / "configs" / dataset
    config_path = config_dir / f"ipc{ipc}_{group_name}.yaml"
    config = make_config(args, dataset, ipc=ipc, group_name=group_name, niter=niter)
    write_config(config_path, config)

    run_dir = args.exp_root / "runs" / dataset / f"ipc{ipc}" / group_name
    run_dir.mkdir(parents=True, exist_ok=True)
    start_time = datetime.now().timestamp()
    condense_cmd = torchrun_command(
        "condense/condense_script.py",
        config_path,
        args.gpu,
        ipc=ipc,
        extra=["--run_mode", "Condense", "--init", "mix"],
    )
    (run_dir / "condense_command.txt").write_text(" ".join(condense_cmd) + "\n", encoding="utf-8")
    rc = run_command(condense_cmd, repo_dir, run_dir / "condense_stdout.log", run_dir / "condense_stderr.log")
    if rc != 0:
        raise RuntimeError(f"Condense failed for {dataset} {ipc} {group_name}; see {run_dir}")
    condensed_path = newest_distilled_data(args.exp_root, dataset, ipc, start_time)
    if condensed_path is None:
        raise RuntimeError(f"No distilled data found for {dataset} {ipc} {group_name}")

    checkpoint_path = (
        args.exp_root
        / "checkpoints"
        / "synthetic_train"
        / dataset
        / f"ipc{ipc}_{group_name}_best.pth.tar"
    )
    eval_metrics_path = run_dir / "eval_metrics.jsonl"
    eval_cmd = torchrun_command(
        "evaluation/evaluation_script.py",
        config_path,
        args.gpu,
        ipc=ipc,
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
    (run_dir / "eval_command.txt").write_text(" ".join(eval_cmd) + "\n", encoding="utf-8")
    rc = run_command(eval_cmd, repo_dir, run_dir / "eval_stdout.log", run_dir / "eval_stderr.log")
    if rc != 0:
        raise RuntimeError(f"Eval failed for {dataset} {ipc} {group_name}; see {run_dir}")
    accuracy = parse_best_accuracy(run_dir / "eval_stdout.log")

    metrics = {
        "dataset": dataset,
        "ipc": ipc,
        "group": group_name,
        "niter": niter,
        "accuracy": accuracy,
        "condensed_path": str(condensed_path),
        "checkpoint_path": str(checkpoint_path),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    append(
        args.exp_root / "EXPERIMENT_LOG.md",
        f"| {datetime.now().isoformat(timespec='seconds')} | {dataset} | {ipc} | {group_name} | done | {accuracy} | `{condensed_path}` | `{checkpoint_path}` | quick no-CAM protocol |\n",
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run quick MedMNIST A/B/C metrics pipeline without CAM.")
    parser.add_argument("--exp_root", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--batch_real", type=int, default=128)
    parser.add_argument("--model_num", type=int, default=2)
    parser.add_argument("--pretrain_epochs", type=int, default=1)
    parser.add_argument("--eval_epochs", type=int, default=5)
    parser.add_argument("--smoke_niter", type=int, default=10)
    parser.add_argument("--formal_niter", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parents[1]
    args.exp_root.mkdir(parents=True, exist_ok=True)

    append(
        args.exp_root / "PATCH_NOTES.md",
        "## Quick no-CAM pipeline\n\n"
        "- CAM/Grad-CAM diagnostics are not run by this quick pipeline.\n"
        "- Use `diagnostics/cam/` manually after experiments if spatial diagnostics are needed.\n",
    )

    phase_plan = [("pneumoniamnist", 1, args.smoke_niter)]
    phase_plan.extend((dataset, 10, args.formal_niter) for dataset in DATASETS)
    needed_datasets = sorted({dataset for dataset, _, _ in phase_plan})

    for dataset in needed_datasets:
        ensure_pretrain(args, repo_dir, dataset)

    all_metrics = []
    for dataset, ipc, niter in phase_plan:
        for group_name in GROUPS:
            all_metrics.append(run_condense_eval(args, repo_dir, dataset, ipc, group_name, niter))

    report_path = args.exp_root / "reports" / "quick_ablation_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"Saved summary: {report_path}")


if __name__ == "__main__":
    main()

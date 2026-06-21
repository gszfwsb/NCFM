import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml


DATASET = "pathmnist"
IPC = 10
NCLASS = 9
NCH = 3


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def copy_assets(exp_root, data_source, pretrain_source):
    data_dst = exp_root / "data" / "medmnist"
    pretrain_dst = exp_root / "checkpoints" / "pretrain" / DATASET
    data_dst.mkdir(parents=True, exist_ok=True)
    pretrain_dst.mkdir(parents=True, exist_ok=True)

    data_target = data_dst / "pathmnist.npz"
    if not data_target.exists():
        shutil.copy2(data_source, data_target)

    for src in Path(pretrain_source).glob("premodel*.pth.tar"):
        dst = pretrain_dst / src.name
        if not dst.exists():
            shutil.copy2(src, dst)


def make_config(args, exp_root, group_name, save_root, method):
    rdzv_dir = exp_root / "rdzv"
    rdzv_dir.mkdir(parents=True, exist_ok=True)
    store = rdzv_dir / f"{DATASET}_{group_name}_{int(time.time() * 1000000)}.store"
    init_method = "file:///" + str(store).replace("\\", "/") + "?rank=0&world_size=1"

    cfg = {
        "distibution_train": {
            "backend": "gloo",
            "init_method": init_method,
            "workers": args.workers,
        },
        "dataset": {
            "dataset": DATASET,
            "nclass": NCLASS,
            "size": 28,
            "data_dir": str(exp_root / "data"),
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
            "pertrain_epochs": 60,
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
            "save_dir": str(save_root),
            "pretrain_dir": str(exp_root / "checkpoints" / "pretrain"),
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
        },
    }

    if method == "m12":
        cfg["condense"].update(
            {
                "use_local_patch_feature_ncfd": True,
                "lambda_local_patch_ncfd": args.m12_lambda_local_patch_ncfd,
                "local_patch_grid": args.m12_local_patch_grid,
                "local_patch_num_freqs": args.m12_local_patch_num_freqs,
                "local_patch_loss_scale": args.m12_local_patch_loss_scale,
                "local_patch_feature_dim": 0,
                "local_patch_encoder_source": "premodel0_trained",
                "local_patch_premodel_index": 0,
                "local_patch_encoder_blocks": args.m12_local_patch_encoder_blocks,
                "local_patch_encoder_frozen": True,
                "use_local_patch_sampling_net": False,
                "local_patch_model_num": args.model_num,
                "local_patch_encoder_seed": args.seed,
            }
        )
    return cfg


def write_config(path, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def torchrun_command(repo_dir, config_path, gpu, ipc, mode, extra):
    script = "condense/condense_script.py" if mode == "condense" else "evaluation/evaluation_script.py"
    cmd = [
        sys.executable,
        script,
        "--config_path",
        str(config_path),
        "--gpu",
        str(gpu),
        "-i",
        str(ipc),
    ]
    cmd.extend(extra)
    return cmd


def run_command(cmd, cwd, stdout_path, stderr_path, gpu):
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["LOCAL_RANK"] = "0"
    env["RANK"] = "0"
    env["WORLD_SIZE"] = "1"
    env["LOCAL_WORLD_SIZE"] = "1"
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("MASTER_PORT", str(29500 + int(gpu)))
    env.setdefault("PYTHONUTF8", "1")
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        proc = subprocess.run(cmd, cwd=cwd, stdout=out, stderr=err, text=True, env=env)
    return proc.returncode


def latest_distilled_data(root, start_time):
    candidates = []
    for path in root.glob(f"**/{DATASET}/ipc{IPC}/**/distilled_data/data_*.pt"):
        if path.name == "data_init.pt":
            continue
        if path.stat().st_mtime >= start_time:
            candidates.append(path)
    if not candidates:
        candidates = [
            p
            for p in root.glob(f"**/{DATASET}/ipc{IPC}/**/distilled_data/data_*.pt")
            if p.name != "data_init.pt"
        ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def read_best_metrics(metrics_jsonl):
    best_path = metrics_jsonl.with_name(metrics_jsonl.stem + "_best.json")
    if best_path.exists():
        return json.loads(best_path.read_text(encoding="utf-8"))
    best = {}
    if metrics_jsonl.exists():
        for line in metrics_jsonl.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("is_best"):
                best = record
    return best


def run_one(job, args):
    exp_root = Path(job["exp_root"])
    repo_dir = Path(job["repo_dir"])
    group_name = job["group"]
    gpu = job["gpu"]
    method = job["method"]
    save_root = exp_root / "results" / "condense"
    if method == "m12":
        save_root = save_root / group_name

    exp_root.mkdir(parents=True, exist_ok=True)
    copy_assets(exp_root, args.data_source, args.pretrain_source)

    config_path = exp_root / "configs" / DATASET / f"ipc{IPC}_{group_name}.yaml"
    run_dir = exp_root / "runs" / DATASET / f"ipc{IPC}" / group_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_config(args, exp_root, group_name, save_root, method)
    write_config(config_path, cfg)

    write_text(run_dir / "method_plan.json", json.dumps(job, indent=2))

    start_time = time.time()
    condense_cmd = torchrun_command(
        repo_dir,
        config_path,
        gpu,
        IPC,
        "condense",
        ["--run_mode", "Condense", "--init", "mix"],
    )
    write_text(run_dir / "condense_command.txt", " ".join(map(str, condense_cmd)) + "\n")
    rc = run_command(
        condense_cmd,
        repo_dir,
        run_dir / "condense_stdout.log",
        run_dir / "condense_stderr.log",
        gpu,
    )
    if rc != 0:
        raise RuntimeError(f"Condense failed for {group_name}; see {run_dir}")

    condensed_path = latest_distilled_data(save_root, start_time)
    if condensed_path is None:
        raise RuntimeError(f"No distilled data found for {group_name}")
    write_text(run_dir / "condensed_path.txt", str(condensed_path) + "\n")

    checkpoint_path = (
        exp_root / "checkpoints" / "synthetic_train" / DATASET / f"ipc{IPC}_{group_name}_best.pth.tar"
    )
    eval_metrics_path = run_dir / "eval_metrics.jsonl"
    eval_cmd = torchrun_command(
        repo_dir,
        config_path,
        gpu,
        IPC,
        "eval",
        [
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
    write_text(run_dir / "eval_command.txt", " ".join(map(str, eval_cmd)) + "\n")
    rc = run_command(
        eval_cmd,
        repo_dir,
        run_dir / "eval_stdout.log",
        run_dir / "eval_stderr.log",
        gpu,
    )
    if rc != 0:
        raise RuntimeError(f"Eval failed for {group_name}; see {run_dir}")

    best = read_best_metrics(eval_metrics_path)
    metrics = {
        "dataset": DATASET,
        "seed": args.seed,
        "ipc": IPC,
        "group": group_name,
        "method": method,
        "niter": args.niter,
        "global_num_freqs": 1024,
        "batch_real": args.batch_real,
        "batch_size": args.batch_size,
        "condensed_path": str(condensed_path),
        "checkpoint_path": str(checkpoint_path),
        **best,
    }
    if method == "m12":
        metrics.update(
            {
                "lambda_local_patch_ncfd": args.m12_lambda_local_patch_ncfd,
                "local_patch_grid": args.m12_local_patch_grid,
                "local_patch_num_freqs": args.m12_local_patch_num_freqs,
                "local_patch_encoder_source": "premodel0_trained",
                "local_patch_encoder_blocks": args.m12_local_patch_encoder_blocks,
                "local_patch_loss_scale": args.m12_local_patch_loss_scale,
            }
        )
    write_text(run_dir / "metrics.json", json.dumps(metrics, indent=2))
    return metrics


def save_summary(root, results):
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    write_text(report_dir / "pathmnist_baseline_m12_pair_seed0.json", json.dumps(results, indent=2))
    fields = [
        "dataset",
        "seed",
        "method",
        "group",
        "acc_percent",
        "auc_macro_ovr",
        "macro_f1",
        "balanced_acc",
        "global_num_freqs",
        "lambda_local_patch_ncfd",
        "local_patch_grid",
        "local_patch_num_freqs",
        "condensed_path",
        "checkpoint_path",
    ]
    with (report_dir / "pathmnist_baseline_m12_pair_seed0.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    rows = [
        "| Method | Group | ACC | AUC | Macro-F1 | BACC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in results:
        rows.append(
            f"| {item.get('method')} | {item.get('group')} | {item.get('acc_percent')} | "
            f"{item.get('auc_macro_ovr')} | {item.get('macro_f1')} | {item.get('balanced_acc')} |"
        )
    write_text(report_dir / "pathmnist_baseline_m12_pair_seed0.md", "\n".join(rows) + "\n")


def compact_float_token(value):
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    if text.startswith("0."):
        digits = text[2:]
        if len(digits) == 1:
            return "0" + digits
        return digits
    return text.replace(".", "")


def main():
    parser = argparse.ArgumentParser(description="Run PathMNIST baseline T1024 and M12 fixed-p0 local patch in parallel.")
    parser.add_argument("--root", type=Path, default=Path(r"C:\xxyProject\NCFMproject_0603\e\pathmnist_baseline_m12_t1024_seed0_20260604"))
    parser.add_argument("--m00_repo", type=Path, default=Path(r"C:\xxyProject\NCFMproject_0603\active_code\M00_baseline_ncfm"))
    parser.add_argument("--m12_repo", type=Path, default=Path(r"C:\xxyProject\NCFMproject_0603\active_code\M12_local_patch_rand20_step"))
    parser.add_argument("--data_source", type=Path, default=Path(r"C:\xxyProject\NCFMproject_0603\archive_legacy_experiments\ncfm_t512_main_20260528\data\medmnist\pathmnist.npz"))
    parser.add_argument("--pretrain_source", type=Path, default=Path(r"C:\xxyProject\NCFMproject_0603\archive_legacy_experiments\ncfm_t512_main_20260528\checkpoints\pretrain\pathmnist"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch_real", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--model_num", type=int, default=20)
    parser.add_argument("--eval_epochs", type=int, default=2000)
    parser.add_argument("--epoch_eval_interval", type=int, default=100)
    parser.add_argument("--niter", type=int, default=20000)
    parser.add_argument("--baseline_gpu", default="0")
    parser.add_argument("--m12_gpu", default="1")
    parser.add_argument("--only", choices=["both", "baseline", "m12"], default="both")
    parser.add_argument("--m12_lambda_local_patch_ncfd", type=float, default=0.6)
    parser.add_argument("--m12_local_patch_grid", type=int, default=2)
    parser.add_argument("--m12_local_patch_num_freqs", type=int, default=256)
    parser.add_argument("--m12_local_patch_encoder_blocks", type=int, default=2)
    parser.add_argument("--m12_local_patch_loss_scale", type=float, default=1.0)
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    m12_group = (
        f"M12_T1024_lam{compact_float_token(args.m12_lambda_local_patch_ncfd)}_"
        f"g{args.m12_local_patch_grid}_lf{args.m12_local_patch_num_freqs}_p0"
    )
    jobs = [
        {
            "method": "baseline",
            "group": "B_T1024",
            "repo_dir": str(args.m00_repo),
            "exp_root": str(args.root / "baseline"),
            "gpu": args.baseline_gpu,
        },
        {
            "method": "m12",
            "group": m12_group,
            "repo_dir": str(args.m12_repo),
            "exp_root": str(args.root / "m12"),
            "gpu": args.m12_gpu,
        },
    ]
    if args.only != "both":
        jobs = [job for job in jobs if job["method"] == args.only]
    write_text(
        args.root / "RUN_STATUS_PATHMNIST_PAIR.json",
        json.dumps(
            {
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "stage": "started",
                "jobs": jobs,
            },
            indent=2,
        ),
    )

    results = []
    failures = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_job = {executor.submit(run_one, job, args): job for job in jobs}
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append({"job": job, "error": str(exc)})
                append_text(args.root / "FAILURES.jsonl", json.dumps(failures[-1]) + "\n")

    if results:
        save_summary(args.root, results)
    status = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "stage": "completed" if not failures else "partial",
        "results": results,
        "failures": failures,
    }
    write_text(args.root / "RUN_STATUS_PATHMNIST_PAIR.json", json.dumps(status, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

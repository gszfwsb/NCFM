#!/usr/bin/env python3
"""PathMNIST + Kvasir-V2 IPC10 pilot launcher.

Runs a compact, traceable first-stage comparison:
- PathMNIST: existing references plus DR-LTM lambda=0.3/alpha=1.0 seed0
- Kvasir-V2: NCFM baseline, M22 softmax attention, DR-LTM lambda=0.3/alpha=1.0 seed0
- Kvasir-V2 HoP-TM: reduced-buffer pilot, not a full official HoP reproduction

The script is designed to run on the remote GPU server after the patched NCFM
and HoP code directories have been synced.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


KVASIR_URL = "https://datasets.simula.no/downloads/kvasir/kvasir-dataset-v2.zip"
IPC = 10


DATASETS: Dict[str, Dict[str, Any]] = {
    "pathmnist": {
        "nclass": 9,
        "nch": 3,
        "size": 28,
        "load_memory": True,
        "batch_real": 1024,
        "batch_size": 128,
        "model_num": 20,
        "pretrain_epochs": 60,
    },
    "kvasirv2": {
        "nclass": 8,
        "nch": 3,
        "size": 64,
        "load_memory": False,
        "batch_real": 512,
        "batch_size": 128,
        "model_num": 20,
        "pretrain_epochs": 60,
    },
}


NCFM_GROUPS: Dict[str, Dict[str, Any]] = {
    "Path_DRLTM_lam03_a100_L1_nf256": {
        "dataset": "pathmnist",
        "method": "DR_LTM_positionwise_mean_lambda03",
        "use_dr_ltm": True,
        "lambda_dr_ltm_ncfd": 0.3,
        "dr_ltm_alpha": 1.0,
        "dr_ltm_layers": "[1]",
        "dr_ltm_num_freqs": 256,
        "dr_ltm_mode": "topk",
        "notes": "PathMNIST requested DR-LTM lambda=0.3 alpha=1 seed0.",
    },
    "Kvasir_B_NCFM_T4096": {
        "dataset": "kvasirv2",
        "method": "NCFM_baseline",
        "notes": "Kvasir-V2 NCFM baseline seed0.",
    },
    "Kvasir_M22_SM_lam02_tau005_L1_nf256": {
        "dataset": "kvasirv2",
        "method": "M22_discrepancy_guided_token_attention",
        "use_dgsa": True,
        "lambda_discrepancy_attention_ncfd": 0.2,
        "discrepancy_attention_layers": "[1]",
        "discrepancy_attention_num_freqs": 256,
        "discrepancy_attention_mode": "softmax",
        "discrepancy_attention_tau": 0.05,
        "notes": "Kvasir-V2 M22 softmax token attention seed0.",
    },
    "Kvasir_DRLTM_lam03_a100_L1_nf256": {
        "dataset": "kvasirv2",
        "method": "DR_LTM_positionwise_mean_lambda03",
        "use_dr_ltm": True,
        "lambda_dr_ltm_ncfd": 0.3,
        "dr_ltm_alpha": 1.0,
        "dr_ltm_layers": "[1]",
        "dr_ltm_num_freqs": 256,
        "dr_ltm_mode": "topk",
        "notes": "Kvasir-V2 DR-LTM lambda=0.3 alpha=1 seed0.",
    },
}


REFERENCE_ROWS = [
    {
        "dataset": "pathmnist",
        "group": "Path_B_T1024",
        "method": "NCFM_baseline_reference",
        "status": "reference",
        "accuracy": 78.68,
        "auc_macro_ovr": 0.9640,
        "macro_f1": 0.7278,
        "balanced_acc": 0.7336,
        "notes": "Existing PathMNIST IPC10 seed0 baseline.",
    },
    {
        "dataset": "pathmnist",
        "group": "Path_M22_lam02_tau005_L1_nf256",
        "method": "M22_reference",
        "status": "reference",
        "accuracy": 80.96,
        "auc_macro_ovr": "",
        "macro_f1": 0.7507,
        "balanced_acc": "",
        "notes": "Existing PathMNIST M22 seed0 best candidate; fill exact AUC/BACC from old run if needed.",
    },
    {
        "dataset": "pathmnist",
        "group": "HoP-TM_official_IPC10",
        "method": "HoP_TM_official_reference",
        "status": "paper_reference",
        "accuracy": 77.23,
        "auc_macro_ovr": "",
        "macro_f1": "",
        "balanced_acc": "",
        "notes": "HoP paper PathMNIST IPC10 77.23±0.65.",
    },
]


def append(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def run(cmd: List[str], cwd: Path | None, log: Path, env: Dict[str, str] | None = None) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    with log.open("w", encoding="utf-8") as out:
        proc = subprocess.run(cmd, cwd=cwd, stdout=out, stderr=subprocess.STDOUT, text=True, env=full_env)
    return proc.returncode


def write_yaml(path: Path, cfg: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def ensure_kvasir_split(exp_root: Path, status: Path, seed: int = 0, train_ratio: float = 0.8) -> Path:
    data_root = exp_root / "data"
    split_root = data_root / "kvasirv2"
    split_csv = split_root / "split.csv"
    if split_csv.exists() and (split_root / "train").exists() and (split_root / "test").exists():
        append(status, f"KVASIR_SPLIT_REUSED {datetime.now().isoformat()} {split_root}\n")
        return split_root
    lock_path = data_root / "kvasir_split.lock"
    data_root.mkdir(parents=True, exist_ok=True)
    lock_fd = None
    while lock_fd is None:
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(lock_fd, f"{os.getpid()} {datetime.now().isoformat()}\n".encode("utf-8"))
        except FileExistsError:
            append(status, f"KVASIR_SPLIT_WAIT {datetime.now().isoformat()} lock={lock_path}\n")
            for _ in range(60):
                if split_csv.exists() and (split_root / "train").exists() and (split_root / "test").exists():
                    return split_root
                time.sleep(10)
            if lock_path.exists() and time.time() - lock_path.stat().st_mtime > 7200:
                lock_path.unlink(missing_ok=True)
    try:
        if split_csv.exists() and (split_root / "train").exists() and (split_root / "test").exists():
            return split_root

        zip_path = data_root / "downloads" / "kvasir-dataset-v2.zip"
        extract_root = data_root / "raw" / "kvasir-v2"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        extract_root.mkdir(parents=True, exist_ok=True)

        if not zip_path.exists() or zip_path.stat().st_size < 100_000_000:
            append(status, f"KVASIR_DOWNLOAD_START {datetime.now().isoformat()} {KVASIR_URL}\n")
            cmd = [
                "bash",
                "-lc",
                f"wget --no-check-certificate -c -O {zip_path} {KVASIR_URL}",
            ]
            rc = run(cmd, None, exp_root / "logs" / "kvasir_download.log")
            if rc != 0:
                append(status, f"KVASIR_DOWNLOAD_FAIL {datetime.now().isoformat()} rc={rc}\n")
                raise SystemExit(rc)
            append(status, f"KVASIR_DOWNLOAD_DONE {datetime.now().isoformat()} size={zip_path.stat().st_size}\n")

        marker = extract_root / ".extracted"
        if not marker.exists():
            append(status, f"KVASIR_EXTRACT_START {datetime.now().isoformat()}\n")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_root)
            marker.write_text(datetime.now().isoformat() + "\n", encoding="utf-8")
            append(status, f"KVASIR_EXTRACT_DONE {datetime.now().isoformat()}\n")

        candidates = []
        for directory in extract_root.rglob("*"):
            if not directory.is_dir():
                continue
            child_dirs = [p for p in directory.iterdir() if p.is_dir()]
            image_count = 0
            for child in child_dirs:
                image_count += sum(1 for p in child.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            if len(child_dirs) >= 8 and image_count >= 7000:
                candidates.append((image_count, directory))
        if not candidates:
            raise RuntimeError(f"Could not locate Kvasir class folders under {extract_root}")
        src_root = sorted(candidates, reverse=True)[0][1]

        if split_root.exists():
            shutil.rmtree(split_root)
        rows = []
        rng = random.Random(seed)
        classes = sorted([p for p in src_root.iterdir() if p.is_dir()])
        for label, class_dir in enumerate(classes):
            images = sorted([p for p in class_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
            rng.shuffle(images)
            cut = int(round(len(images) * train_ratio))
            for split, items in [("train", images[:cut]), ("test", images[cut:])]:
                dst_dir = split_root / split / class_dir.name
                dst_dir.mkdir(parents=True, exist_ok=True)
                for src in items:
                    dst = dst_dir / src.name
                    if not dst.exists():
                        try:
                            os.symlink(src, dst)
                        except OSError:
                            try:
                                os.link(src, dst)
                            except OSError:
                                shutil.copy2(src, dst)
                    rows.append(
                        {
                            "split": split,
                            "class_name": class_dir.name,
                            "label": label,
                            "source_path": str(src),
                            "image_path": str(dst),
                        }
                    )
        split_root.mkdir(parents=True, exist_ok=True)
        with split_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["split", "class_name", "label", "source_path", "image_path"])
            writer.writeheader()
            writer.writerows(rows)
        append(status, f"KVASIR_SPLIT_DONE {datetime.now().isoformat()} root={split_root} rows={len(rows)}\n")
        return split_root
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        lock_path.unlink(missing_ok=True)


def torchrun(script: str, cfg: Path, gpu: int, ipc: int | None = None, extra: List[str] | None = None) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=1",
        script,
        "--config_path",
        str(cfg),
        "--gpu",
        str(gpu),
    ]
    if ipc is not None:
        cmd.extend(["-i", str(ipc)])
    if extra:
        cmd.extend(extra)
    return cmd


def make_ncfm_config(args: argparse.Namespace, dataset: str, save_dir: Path, spec: Dict[str, Any] | None = None) -> Dict[str, Any]:
    info = DATASETS[dataset]
    rdzv = args.exp_root / "rdzv"
    rdzv.mkdir(parents=True, exist_ok=True)
    store = rdzv / f"{dataset}_{int(time.time() * 1000000)}.store"
    cfg: Dict[str, Any] = {
        "distibution_train": {
            "backend": "gloo",
            "init_method": "file:///" + str(store).replace("\\", "/") + "?rank=0&world_size=1",
            "workers": args.workers,
        },
        "dataset": {
            "dataset": dataset,
            "nclass": info["nclass"],
            "size": info["size"],
            "data_dir": str(args.exp_root / "data"),
            "load_memory": info["load_memory"],
            "batch_real": info["batch_real"],
            "nch": info["nch"],
        },
        "network": {"net_type": "convnet", "norm_type": "instance", "depth": 3, "width": 1.0},
        "train": {
            "evaluation_epochs": args.eval_epochs,
            "epoch_print_freq": 10,
            "epoch_eval_interval": args.epoch_eval_interval,
            "pertrain_epochs": info["pretrain_epochs"],
            "batch_size": info["batch_size"],
            "lr": 0.01,
            "adamw_lr": 0.001,
            "eval_optimizer": "adamw",
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "seed": args.seed,
            "model_num": info["model_num"],
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
            "save_dir": str(save_dir),
            "pretrain_dir": str(args.exp_root / "checkpoints" / "pretrain"),
        },
        "condense": {
            "ipc": IPC,
            "num_premodel": info["model_num"],
            "niter": args.niter,
            "iter_calib": 1,
            "calib_weight": 1,
            "sampling_net": False,
            "num_freqs": 4096,
            "dis_metrics": "NCFM",
            "force_final_feature_match": True,
            "factor": 2,
            "alpha_for_loss": 0.5,
            "beta_for_loss": 0.5,
            "decode_type": "single",
            "teacher_model_epoch": 20,
            "use_dr_ltm_ncfd": False,
            "use_discrepancy_attention_ncfd": False,
            "use_feature_map_token_ncfd": False,
            "use_local_patch_feature_ncfd": False,
        },
    }
    if spec:
        cond = cfg["condense"]
        if spec.get("use_dgsa"):
            cond.update(
                {
                    "use_discrepancy_attention_ncfd": True,
                    "lambda_discrepancy_attention_ncfd": float(spec["lambda_discrepancy_attention_ncfd"]),
                    "discrepancy_attention_layers": spec["discrepancy_attention_layers"],
                    "discrepancy_attention_num_freqs": int(spec["discrepancy_attention_num_freqs"]),
                    "discrepancy_attention_mode": spec["discrepancy_attention_mode"],
                    "discrepancy_attention_tau": float(spec["discrepancy_attention_tau"]),
                    "discrepancy_attention_loss_scale": args.loss_scale,
                    "discrepancy_attention_detach_real": True,
                    "discrepancy_attention_log_components": True,
                }
            )
        if spec.get("use_dr_ltm"):
            cond.update(
                {
                    "use_dr_ltm_ncfd": True,
                    "lambda_dr_ltm_ncfd": float(spec["lambda_dr_ltm_ncfd"]),
                    "dr_ltm_alpha": float(spec["dr_ltm_alpha"]),
                    "dr_ltm_layers": spec["dr_ltm_layers"],
                    "dr_ltm_num_freqs": int(spec["dr_ltm_num_freqs"]),
                    "dr_ltm_mode": spec["dr_ltm_mode"],
                    "dr_ltm_loss_scale": args.loss_scale,
                    "dr_ltm_detach_real": True,
                    "dr_ltm_log_components": True,
                }
            )
    return cfg


def ensure_ncfm_pretrain(args: argparse.Namespace, dataset: str, gpu: int, status: Path) -> None:
    expected = args.exp_root / "checkpoints" / "pretrain" / dataset / "premodel19_trained.pth.tar"
    if expected.exists():
        append(status, f"NCFM_PRETRAIN_REUSED {datetime.now().isoformat()} dataset={dataset} path={expected}\n")
        return
    cfg_path = args.exp_root / "configs" / dataset / "pretrain.yaml"
    cfg = make_ncfm_config(args, dataset, args.exp_root / "results" / "pretrain_placeholder")
    write_yaml(cfg_path, cfg)
    append(status, f"NCFM_PRETRAIN_START {datetime.now().isoformat()} dataset={dataset} gpu={gpu}\n")
    cmd = torchrun("pretrain/pretrain_script.py", cfg_path, gpu, IPC, ["--run_mode", "Pretrain"])
    rc = run(
        cmd,
        args.ncfm_repo,
        args.exp_root / "logs" / f"ncfm_pretrain_{dataset}.log",
        env={"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"},
    )
    if rc != 0:
        append(status, f"NCFM_PRETRAIN_FAIL {datetime.now().isoformat()} dataset={dataset} rc={rc}\n")
        raise SystemExit(rc)
    append(status, f"NCFM_PRETRAIN_DONE {datetime.now().isoformat()} dataset={dataset}\n")


def newest_distilled_data(run_dir: Path, dataset: str, start: float) -> Path | None:
    candidates = []
    root = run_dir / "results" / "condense"
    if root.exists():
        for path in root.glob(f"**/{dataset}/ipc{IPC}/**/distilled_data/data_*.pt"):
            if path.name != "data_init.pt" and path.stat().st_mtime >= start:
                candidates.append(path)
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def read_best_metrics(metrics_jsonl: Path) -> Dict[str, Any]:
    best_path = metrics_jsonl.with_name(metrics_jsonl.stem + "_best.json")
    if best_path.exists():
        return json.loads(best_path.read_text(encoding="utf-8"))
    best: Dict[str, Any] = {}
    if metrics_jsonl.exists():
        for line in metrics_jsonl.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("is_best"):
                best = rec
    return best


def parse_best_accuracy(stdout_path: Path) -> float | None:
    if not stdout_path.exists():
        return None
    text = stdout_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Best\s+accuracy \(top-1 and 5\):\s*([0-9.]+)", text)
    return float(matches[-1]) if matches else None


def run_ncfm_group(args: argparse.Namespace, group_name: str, gpu: int, status: Path) -> Dict[str, Any]:
    spec = NCFM_GROUPS[group_name]
    dataset = spec["dataset"]
    run_dir = args.exp_root / "runs" / dataset / f"ipc{IPC}" / group_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not args.force:
        append(status, f"NCFM_GROUP_SKIP_DONE {datetime.now().isoformat()} group={group_name} metrics={metrics_path}\n")
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    cfg_path = args.exp_root / "configs" / dataset / f"ipc{IPC}_{group_name}.yaml"
    cfg = make_ncfm_config(args, dataset, run_dir / "results" / "condense", spec)
    write_yaml(cfg_path, cfg)
    append(status, f"NCFM_GROUP_START {datetime.now().isoformat()} group={group_name} gpu={gpu}\n")

    condensed_record = run_dir / "condensed_path.txt"
    condensed_path = None
    if condensed_record.exists() and not args.force:
        candidate = Path(condensed_record.read_text(encoding="utf-8").strip())
        if candidate.exists():
            condensed_path = candidate

    if condensed_path is None:
        start = datetime.now().timestamp()
        cmd = torchrun(
            "condense/condense_script.py",
            cfg_path,
            gpu,
            IPC,
            ["--run_mode", "Condense", "--init", "mix"],
        )
        (run_dir / "condense_command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
        rc = run(
            cmd,
            args.ncfm_repo,
            run_dir / "condense_stdout.log",
            env={"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"},
        )
        if rc != 0:
            append(status, f"NCFM_GROUP_CONDENSE_FAIL {datetime.now().isoformat()} group={group_name} rc={rc}\n")
            raise SystemExit(rc)
        condensed_path = newest_distilled_data(run_dir, dataset, start)
        if condensed_path is None:
            append(status, f"NCFM_GROUP_NO_DATA {datetime.now().isoformat()} group={group_name}\n")
            raise SystemExit(2)
        condensed_record.write_text(str(condensed_path) + "\n", encoding="utf-8")
        append(status, f"NCFM_GROUP_CONDENSE_DONE {datetime.now().isoformat()} group={group_name} data={condensed_path}\n")

    ckpt = args.exp_root / "checkpoints" / "synthetic_train" / dataset / f"ipc{IPC}_{group_name}_best.pth.tar"
    eval_metrics = run_dir / "eval_metrics.jsonl"
    cmd = torchrun(
        "evaluation/evaluation_script.py",
        cfg_path,
        gpu,
        IPC,
        [
            "--run_mode",
            "Evaluation",
            "--load_path",
            str(condensed_path),
            "--val_repeat",
            "1",
            "--eval_checkpoint_path",
            str(ckpt),
            "--eval_metrics_path",
            str(eval_metrics),
        ],
    )
    (run_dir / "eval_command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    rc = run(
        cmd,
        args.ncfm_repo,
        run_dir / "eval_stdout.log",
        env={"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"},
    )
    if rc != 0:
        append(status, f"NCFM_GROUP_EVAL_FAIL {datetime.now().isoformat()} group={group_name} rc={rc}\n")
        raise SystemExit(rc)

    best = read_best_metrics(eval_metrics)
    metrics = {
        "dataset": dataset,
        "ipc": IPC,
        "group": group_name,
        "method": spec["method"],
        "status": "done",
        "seed": args.seed,
        "accuracy": best.get("acc_percent") or parse_best_accuracy(run_dir / "eval_stdout.log"),
        "condensed_path": str(condensed_path),
        "checkpoint_path": str(ckpt),
        "notes": spec.get("notes", ""),
        **best,
    }
    for key in [
        "lambda_discrepancy_attention_ncfd",
        "discrepancy_attention_mode",
        "discrepancy_attention_tau",
        "lambda_dr_ltm_ncfd",
        "dr_ltm_alpha",
    ]:
        if key in spec:
            metrics[key] = spec[key]
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    append(status, f"NCFM_GROUP_DONE {datetime.now().isoformat()} group={group_name} acc={metrics.get('accuracy')}\n")
    save_summary(args.exp_root)
    return metrics


def save_summary(exp_root: Path) -> None:
    rows: List[Dict[str, Any]] = list(REFERENCE_ROWS)
    for group_name, spec in NCFM_GROUPS.items():
        path = exp_root / "runs" / spec["dataset"] / f"ipc{IPC}" / group_name / "metrics.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            rows.append(
                {
                    "dataset": spec["dataset"],
                    "group": group_name,
                    "method": spec["method"],
                    "status": "pending",
                    "accuracy": "",
                    "auc_macro_ovr": "",
                    "macro_f1": "",
                    "balanced_acc": "",
                    "notes": spec.get("notes", ""),
                }
            )
    hop_path = exp_root / "hop" / "metrics" / "kvasirv2_hop_pilot.json"
    if hop_path.exists():
        rows.append(json.loads(hop_path.read_text(encoding="utf-8")))
    else:
        rows.append(
            {
                "dataset": "kvasirv2",
                "group": "Kvasir_HoP_TM_reduced_buffer_pilot",
                "method": "HoP_TM_reduced_buffer_pilot",
                "status": "pending",
                "accuracy": "",
                "notes": "Reduced-buffer HoP pilot; not official 100-expert reproduction.",
            }
        )

    report_dir = exp_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "path_kvasir_ipc10_pilot_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fields = [
        "dataset",
        "group",
        "method",
        "status",
        "accuracy",
        "auc_macro_ovr",
        "macro_f1",
        "balanced_acc",
        "lambda_discrepancy_attention_ncfd",
        "discrepancy_attention_mode",
        "discrepancy_attention_tau",
        "lambda_dr_ltm_ncfd",
        "dr_ltm_alpha",
        "notes",
        "condensed_path",
        "checkpoint_path",
    ]
    with (report_dir / "path_kvasir_ipc10_pilot_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    with (report_dir / "path_kvasir_ipc10_pilot_summary.md").open("w", encoding="utf-8") as f:
        f.write("| Dataset | Group | Method | Status | ACC | AUC | Macro-F1 | BACC | Notes |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            f.write(
                f"| {row.get('dataset')} | {row.get('group')} | {row.get('method')} | "
                f"{row.get('status', '')} | {row.get('accuracy', row.get('acc_percent', ''))} | "
                f"{row.get('auc_macro_ovr', '')} | {row.get('macro_f1', '')} | "
                f"{row.get('balanced_acc', '')} | {row.get('notes', '')} |\n"
            )


def write_hop_config(args: argparse.Namespace, gpu: int) -> Path:
    cfg = {
        "dataset": "KvasirV2",
        "device": [gpu],
        "model": "ConvNet",
        "ipc": 10,
        "syn_steps": 80,
        "expert_epochs": 2,
        "lr_img": 10,
        "lr_teacher": 0.01,
        "buffer_path": str(args.exp_root / "hop" / "buffers"),
        "data_path": str(args.exp_root / "data" / "kvasirv2"),
        "ema_decay": 0.995,
        "Iteration": args.hop_iterations,
        "batch_syn": 10,
        "project": "kvasirv2_high_order_pilot",
        "num_eval": 1,
        "eval_it": max(100, min(500, args.hop_iterations)),
        "skip_first_eva": True,
        "pix_init": "real",
        "high_order": True,
        "base_threshold": 5e-9,
        "growing_factor": 1.5,
        "lamb": 0.5,
        "batch_train": 128,
        "min_start_epoch": 0,
        "max_start_epoch": 4,
        "lr_lr": 0.0000001,
        "zca": False,
        "max_experts": args.hop_num_experts,
        "max_files": 2,
    }
    path = args.exp_root / "hop" / "configs" / "kvasirv2_ipc10_reduced_buffer_pilot.yaml"
    write_yaml(path, cfg)
    return path


def parse_hop_best(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    vals = [float(x) for x in re.findall(r"Max_Accuracy/ConvNet['\"]?:?\s*([0-9.]+)", text)]
    if vals:
        return max(vals)
    vals = [float(x) for x in re.findall(r"test acc = ([0-9.]+)", text)]
    if vals:
        return max(vals) * 100.0 if max(vals) <= 1.0 else max(vals)
    vals = [float(x) for x in re.findall(r"Accuracy/ConvNet.*?([0-9]+\.[0-9]+)", text)]
    return max(vals) if vals else None


def run_hop_kvasir_pilot(args: argparse.Namespace, gpu: int, status: Path) -> None:
    metric_path = args.exp_root / "hop" / "metrics" / "kvasirv2_hop_pilot.json"
    if metric_path.exists() and not args.force:
        append(status, f"HOP_REUSED {datetime.now().isoformat()} path={metric_path}\n")
        return
    append(status, f"HOP_BUFFER_START {datetime.now().isoformat()} gpu={gpu}\n")
    buffer_cmd = [
        sys.executable,
        "buffer_FTD.py",
        "--dataset=KvasirV2",
        "--model=ConvNet",
        f"--data_path={args.exp_root / 'data' / 'kvasirv2'}",
        f"--buffer_path={args.exp_root / 'hop' / 'buffers'}",
        f"--num_experts={args.hop_num_experts}",
        f"--train_epochs={args.hop_train_epochs}",
        "--save_interval=10",
        "--batch_train=256",
        "--batch_real=256",
    ]
    rc = run(
        buffer_cmd,
        args.hop_repo / "buffer",
        args.exp_root / "hop" / "logs" / "kvasirv2_buffer.log",
        env={"CUDA_VISIBLE_DEVICES": str(gpu), "WANDB_MODE": "offline", "PYTHONUNBUFFERED": "1"},
    )
    if rc != 0:
        append(status, f"HOP_BUFFER_FAIL {datetime.now().isoformat()} rc={rc}\n")
        return
    append(status, f"HOP_BUFFER_DONE {datetime.now().isoformat()}\n")

    cfg_path = write_hop_config(args, gpu)
    append(status, f"HOP_DISTILL_START {datetime.now().isoformat()} gpu={gpu} cfg={cfg_path}\n")
    distill_cmd = [sys.executable, "distill_high_order_spl.py", "--cfg", str(cfg_path)]
    log_path = args.exp_root / "hop" / "logs" / "kvasirv2_distill.log"
    rc = run(
        distill_cmd,
        args.hop_repo / "distill",
        log_path,
        env={
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "WANDB_MODE": "offline",
            "HOP_LOGGED_FILES_DIR": str(args.exp_root / "hop" / "logged_files"),
            "PYTHONUNBUFFERED": "1",
        },
    )
    if rc != 0:
        append(status, f"HOP_DISTILL_FAIL {datetime.now().isoformat()} rc={rc}\n")
        return
    acc = parse_hop_best(log_path)
    metric = {
        "dataset": "kvasirv2",
        "group": "Kvasir_HoP_TM_reduced_buffer_pilot",
        "method": "HoP_TM_reduced_buffer_pilot",
        "status": "done",
        "accuracy": acc,
        "hop_num_experts": args.hop_num_experts,
        "hop_train_epochs": args.hop_train_epochs,
        "hop_iterations": args.hop_iterations,
        "notes": "Reduced-buffer pilot, not official 100-expert HoP reproduction.",
    }
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.write_text(json.dumps(metric, indent=2), encoding="utf-8")
    append(status, f"HOP_DONE {datetime.now().isoformat()} acc={acc}\n")
    save_summary(args.exp_root)


def run_ncfm_controller(args: argparse.Namespace) -> None:
    status = args.exp_root / "RUN_STATUS_NCFM.txt"
    append(status, f"NCFM_CONTROLLER_START {datetime.now().isoformat()}\n")

    # PathMNIST does not depend on Kvasir-V2 download/split. Run this first so
    # GPU0 is not idle while Kvasir data preparation happens in the HoP screen.
    path_expected = args.exp_root / "checkpoints" / "pretrain" / "pathmnist" / "premodel19_trained.pth.tar"
    if path_expected.exists():
        run_ncfm_group(args, "Path_DRLTM_lam03_a100_L1_nf256", gpu=0, status=status)
    else:
        append(status, f"PATH_PRETRAIN_MISSING_SKIP_RUN {datetime.now().isoformat()} expected={path_expected}\n")

    ensure_kvasir_split(args.exp_root, status, seed=args.seed)
    save_summary(args.exp_root)

    ensure_ncfm_pretrain(args, "kvasirv2", gpu=0, status=status)
    queue = ["Kvasir_B_NCFM_T4096", "Kvasir_M22_SM_lam02_tau005_L1_nf256", "Kvasir_DRLTM_lam03_a100_L1_nf256"]

    gpu_list = [int(x) for x in args.gpus.split(",") if x.strip()]
    running: List[Tuple[str, int, subprocess.Popen[Any], Any]] = []
    failed = False
    while queue or running:
        still = []
        for name, gpu, proc, out in running:
            rc = proc.poll()
            if rc is None:
                still.append((name, gpu, proc, out))
                continue
            out.close()
            append(status, f"NCFM_WORKER_EXIT {datetime.now().isoformat()} group={name} rc={rc}\n")
            failed = failed or rc != 0
            save_summary(args.exp_root)
        running = still
        used = {gpu for _, gpu, _, _ in running}
        for gpu in gpu_list:
            if gpu in used or not queue:
                continue
            name = queue.pop(0)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--mode",
                "ncfm-worker",
                "--group",
                name,
                "--exp-root",
                str(args.exp_root),
                "--ncfm-repo",
                str(args.ncfm_repo),
                "--hop-repo",
                str(args.hop_repo),
                "--gpu",
                str(gpu),
                "--seed",
                str(args.seed),
                "--niter",
                str(args.niter),
                "--eval-epochs",
                str(args.eval_epochs),
                "--epoch-eval-interval",
                str(args.epoch_eval_interval),
                "--workers",
                str(args.workers),
                "--loss-scale",
                str(args.loss_scale),
            ]
            if args.force:
                cmd.append("--force")
            log = args.exp_root / "launcher_logs" / f"{name}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            append(status, f"NCFM_LAUNCH_WORKER {datetime.now().isoformat()} group={name} gpu={gpu}\n")
            out = log.open("w", encoding="utf-8")
            proc = subprocess.Popen(cmd, cwd=args.ncfm_repo, stdout=out, stderr=subprocess.STDOUT, text=True)
            running.append((name, gpu, proc, out))
        append(
            status,
            f"NCFM_HEARTBEAT {datetime.now().isoformat()} running={[x[0] for x in running]} pending={queue} failed={failed}\n",
        )
        time.sleep(30)
    save_summary(args.exp_root)
    append(status, f"NCFM_CONTROLLER_DONE {datetime.now().isoformat()} failed={failed}\n")
    if failed:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ncfm-controller", "ncfm-worker", "hop-controller"], default="ncfm-controller")
    parser.add_argument("--group", choices=sorted(NCFM_GROUPS), default=None)
    parser.add_argument("--exp-root", type=Path, required=True)
    parser.add_argument("--ncfm-repo", type=Path, required=True)
    parser.add_argument("--hop-repo", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--niter", type=int, default=20000)
    parser.add_argument("--eval-epochs", type=int, default=2000)
    parser.add_argument("--epoch-eval-interval", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--loss-scale", type=float, default=300.0)
    parser.add_argument("--hop-num-experts", type=int, default=20)
    parser.add_argument("--hop-train-epochs", type=int, default=20)
    parser.add_argument("--hop-iterations", type=int, default=2000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.exp_root.mkdir(parents=True, exist_ok=True)
    if args.mode == "ncfm-worker":
        if not args.group:
            raise ValueError("--group is required in ncfm-worker mode")
        run_ncfm_group(args, args.group, args.gpu, args.exp_root / "RUN_STATUS_NCFM.txt")
    elif args.mode == "hop-controller":
        status = args.exp_root / "RUN_STATUS_HOP.txt"
        append(status, f"HOP_CONTROLLER_START {datetime.now().isoformat()}\n")
        ensure_kvasir_split(args.exp_root, status, seed=args.seed)
        run_hop_kvasir_pilot(args, args.gpu, status)
        append(status, f"HOP_CONTROLLER_DONE {datetime.now().isoformat()}\n")
    else:
        run_ncfm_controller(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Launch Kvasir-V2 teacher pretrain audit on the 3090 server.

This script is intended to run on the remote server. It generates isolated
NCFM-style pretrain YAMLs and launches one-teacher pretrain jobs so we can
choose a stronger Kvasir teacher before running full NCFM/HoP pipelines.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


EXPERIMENTS = {
    "C4_96_in": {
        "size": 96,
        "net_type": "convnet",
        "norm_type": "instance",
        "depth": 4,
        "width": 1.0,
        "epochs": 100,
        "batch_size": 96,
        "lr": 0.01,
    },
    "C5_96_in": {
        "size": 96,
        "net_type": "convnet",
        "norm_type": "instance",
        "depth": 5,
        "width": 1.0,
        "epochs": 100,
        "batch_size": 96,
        "lr": 0.01,
    },
    "C4_128_in": {
        "size": 128,
        "net_type": "convnet",
        "norm_type": "instance",
        "depth": 4,
        "width": 1.0,
        "epochs": 100,
        "batch_size": 64,
        "lr": 0.01,
    },
    "C4_128_bn": {
        "size": 128,
        "net_type": "convnet",
        "norm_type": "batch",
        "depth": 4,
        "width": 1.0,
        "epochs": 100,
        "batch_size": 64,
        "lr": 0.01,
    },
    "C5_128_in": {
        "size": 128,
        "net_type": "convnet",
        "norm_type": "instance",
        "depth": 5,
        "width": 1.0,
        "epochs": 100,
        "batch_size": 64,
        "lr": 0.01,
    },
    "R18_128_bn": {
        "size": 128,
        "net_type": "resnet",
        "norm_type": "batch",
        "depth": 18,
        "width": 1.0,
        "epochs": 80,
        "batch_size": 64,
        "lr": 0.05,
    },
    "C5_128_w15_in": {
        "size": 128,
        "net_type": "convnet",
        "norm_type": "instance",
        "depth": 5,
        "width": 1.5,
        "epochs": 100,
        "batch_size": 48,
        "lr": 0.01,
    },
    "C5_128_w20_in": {
        "size": 128,
        "net_type": "convnet",
        "norm_type": "instance",
        "depth": 5,
        "width": 2.0,
        "epochs": 100,
        "batch_size": 40,
        "lr": 0.01,
    },
    "C6_128_in": {
        "size": 128,
        "net_type": "convnet",
        "norm_type": "instance",
        "depth": 6,
        "width": 1.0,
        "epochs": 100,
        "batch_size": 64,
        "lr": 0.01,
    },
    "C6_128_w15_in": {
        "size": 128,
        "net_type": "convnet",
        "norm_type": "instance",
        "depth": 6,
        "width": 1.5,
        "epochs": 100,
        "batch_size": 48,
        "lr": 0.01,
    },
    "C5_160_w15_in": {
        "size": 160,
        "net_type": "convnet",
        "norm_type": "instance",
        "depth": 5,
        "width": 1.5,
        "epochs": 100,
        "batch_size": 32,
        "lr": 0.01,
    },
    "C6_160_w15_in": {
        "size": 160,
        "net_type": "convnet",
        "norm_type": "instance",
        "depth": 6,
        "width": 1.5,
        "epochs": 100,
        "batch_size": 32,
        "lr": 0.01,
    },
}


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def latest_metrics(pretrain_dir: Path) -> dict | None:
    candidates = sorted(pretrain_dir.glob("premodel*_metrics.json"))
    if not candidates:
        return None
    best = None
    best_acc = -1.0
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        acc = float(data.get("acc_percent", data.get("val_acc_percent", -1.0)))
        if acc >= best_acc:
            best_acc = acc
            best = data
    return best


def append_status(exp_root: Path, text: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    with (exp_root / "RUN_STATUS.txt").open("a", encoding="utf-8") as f:
        f.write(f"{ts} {text}\n")


def build_config(base: dict, exp_root: Path, name: str, spec: dict, data_dir: str) -> dict:
    cfg = json.loads(json.dumps(base))
    cfg["distibution_train"]["init_method"] = (
        f"file:///{exp_root}/rdzv/{name}_{time.time_ns()}.store?rank=0&world_size=1"
    )
    cfg["distibution_train"]["workers"] = 8
    cfg["dataset"]["dataset"] = "kvasirv2"
    cfg["dataset"]["nclass"] = 8
    cfg["dataset"]["size"] = int(spec["size"])
    cfg["dataset"]["data_dir"] = data_dir
    cfg["dataset"]["load_memory"] = False
    cfg["dataset"]["batch_real"] = 512
    cfg["dataset"]["nch"] = 3
    cfg["network"]["net_type"] = spec["net_type"]
    cfg["network"]["norm_type"] = spec["norm_type"]
    cfg["network"]["depth"] = int(spec["depth"])
    cfg["network"]["width"] = float(spec["width"])
    cfg["train"]["pertrain_epochs"] = int(spec["epochs"])
    cfg["train"]["batch_size"] = int(spec["batch_size"])
    cfg["train"]["lr"] = float(spec["lr"])
    cfg["train"]["model_num"] = 1
    cfg["train"]["seed"] = 0
    cfg["augmentation"]["mixup"] = "cut"
    cfg["augmentation"]["dsa"] = True
    cfg["augmentation"]["dsa_strategy"] = "color_crop_cutout_flip_scale_rotate"
    cfg["save_path"]["save_dir"] = str(exp_root / "results" / name)
    cfg["save_path"]["pretrain_dir"] = str(exp_root / "checkpoints" / name)
    return cfg


def run_one(args: argparse.Namespace, name: str, spec: dict) -> dict:
    exp_root = Path(args.exp_root)
    code_root = Path(args.code_root)
    base_cfg = read_yaml(Path(args.base_config))
    cfg = build_config(base_cfg, exp_root, name, spec, args.data_dir)
    cfg_path = exp_root / "configs" / f"{name}.yaml"
    write_yaml(cfg_path, cfg)

    log_path = exp_root / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pretrain_dir = exp_root / "checkpoints" / name / "kvasirv2"
    append_status(exp_root, f"START name={name} gpu={args.gpu} cfg={cfg_path}")
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=1",
        "pretrain/pretrain_script.py",
        "--config_path",
        str(cfg_path),
        "--gpu",
        str(args.gpu),
        "-i",
        "10",
        "--run_mode",
        "Pretrain",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(code_root),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    metric = latest_metrics(pretrain_dir) or {}
    row = {
        "name": name,
        "status": "done" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "size": spec["size"],
        "net_type": spec["net_type"],
        "norm_type": spec["norm_type"],
        "depth": spec["depth"],
        "width": spec["width"],
        "epochs": spec["epochs"],
        "batch_size": spec["batch_size"],
        "lr": spec["lr"],
        "acc_percent": metric.get("acc_percent"),
        "auc_macro_ovr": metric.get("auc_macro_ovr"),
        "macro_f1": metric.get("macro_f1"),
        "balanced_acc": metric.get("balanced_acc"),
        "log": str(log_path),
        "pretrain_dir": str(pretrain_dir),
    }
    append_status(
        exp_root,
        "DONE "
        + " ".join(
            [
                f"name={name}",
                f"status={row['status']}",
                f"acc={row['acc_percent']}",
                f"auc={row['auc_macro_ovr']}",
                f"f1={row['macro_f1']}",
                f"bacc={row['balanced_acc']}",
            ]
        ),
    )
    return row


def write_summary(exp_root: Path, rows: list[dict]) -> None:
    reports = exp_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(
        rows,
        key=lambda r: float(r["acc_percent"] or -1.0),
        reverse=True,
    )
    csv_path = reports / "kvasir_teacher_audit_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()) if rows_sorted else [])
        if rows_sorted:
            writer.writeheader()
            writer.writerows(rows_sorted)
    (reports / "kvasir_teacher_audit_summary.json").write_text(
        json.dumps(rows_sorted, indent=2), encoding="utf-8"
    )
    lines = [
        "# Kvasir-V2 Teacher Audit",
        "",
        "| Rank | Config | Model | Size | Norm | Epochs | ACC | AUC | Macro-F1 | BACC |",
        "|---:|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(rows_sorted, 1):
        lines.append(
            "| {rank} | {name} | {net_type}-{depth} | {size} | {norm_type} | {epochs} | {acc} | {auc} | {f1} | {bacc} |".format(
                rank=i,
                name=r["name"],
                net_type=r["net_type"],
                depth=r["depth"],
                size=r["size"],
                norm_type=r["norm_type"],
                epochs=r["epochs"],
                acc="" if r["acc_percent"] is None else f"{float(r['acc_percent']):.2f}",
                auc="" if r["auc_macro_ovr"] is None else f"{float(r['auc_macro_ovr']):.4f}",
                f1="" if r["macro_f1"] is None else f"{float(r['macro_f1']):.4f}",
                bacc="" if r["balanced_acc"] is None else f"{float(r['balanced_acc']):.4f}",
            )
        )
    (reports / "kvasir_teacher_audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-root", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--names", required=True, help="Comma-separated experiment names")
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    (exp_root / "rdzv").mkdir(parents=True, exist_ok=True)
    rows = []
    for name in [x.strip() for x in args.names.split(",") if x.strip()]:
        if name not in EXPERIMENTS:
            raise KeyError(f"Unknown experiment: {name}")
        rows.append(run_one(args, name, EXPERIMENTS[name]))
        existing = []
        summary_json = exp_root / "reports" / "kvasir_teacher_audit_summary.json"
        if summary_json.exists():
            try:
                existing = json.loads(summary_json.read_text(encoding="utf-8"))
            except Exception:
                existing = []
        by_name = {r["name"]: r for r in existing + rows}
        write_summary(exp_root, list(by_name.values()))
    append_status(exp_root, f"WORKER_DONE gpu={args.gpu} names={args.names}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Wait for Kvasir teacher audit, then launch full NCFM DD.

Remote-only orchestration:
1. Wait for teacher audit summary.
2. Select a strong ConvNet teacher config that remains compatible with local
   token feature methods.
3. Pretrain 20 teachers in parallel across two GPUs using fixed model-id ranges.
4. Run Kvasir IPC10 baseline, M22, and DR-LTM condense+eval.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


GROUP_ORDER = [
    "Kvasir_B_NCFM_T4096",
    "Kvasir_M22_SM_lam02_tau005_L1_nf256",
    "Kvasir_DRLTM_lam03_a100_L1_nf256",
]


def append(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def status(exp_root: Path, text: str) -> None:
    append(exp_root / "RUN_STATUS_FULL_DD.txt", f"{datetime.now().isoformat()} {text}\n")


def load_pilot_module(path: Path):
    spec = importlib.util.spec_from_file_location("path_kvasir_pilot", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pilot launcher from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wait_for_audit(audit_root: Path, exp_root: Path, timeout_hours: float) -> list[dict[str, Any]]:
    expected = {
        "C4_96_in",
        "C5_96_in",
        "C4_128_in",
        "C4_128_bn",
        "C5_128_in",
        "R18_128_bn",
        "C5_128_w15_in",
        "C5_128_w20_in",
        "C6_128_in",
        "C6_128_w15_in",
        "C5_160_w15_in",
        "C6_160_w15_in",
    }
    summary = audit_root / "reports" / "kvasir_teacher_audit_summary.json"
    deadline = time.time() + timeout_hours * 3600
    last_note = 0.0
    while time.time() < deadline:
        rows: list[dict[str, Any]] = []
        if summary.exists():
            try:
                rows = json.loads(summary.read_text(encoding="utf-8"))
            except Exception:
                rows = []
        names = {row.get("name") for row in rows if row.get("status") in {"done", "failed"}}
        if expected.issubset(names):
            status(exp_root, f"AUDIT_COMPLETE rows={len(rows)}")
            return rows
        if time.time() - last_note > 900:
            status(exp_root, f"AUDIT_WAIT completed={sorted(names)}")
            last_note = time.time()
        time.sleep(60)
    raise TimeoutError(f"Teacher audit did not complete within {timeout_hours} hours")


def choose_teacher(rows: list[dict[str, Any]]) -> dict[str, Any]:
    conv = [
        row
        for row in rows
        if row.get("status") == "done"
        and row.get("net_type") == "convnet"
        and row.get("acc_percent") is not None
    ]
    if not conv:
        raise RuntimeError("No completed ConvNet teacher audit result found")

    def acc(row: dict[str, Any]) -> float:
        return float(row.get("acc_percent") or -1.0)

    best_any = max(conv, key=acc)
    inst = [row for row in conv if row.get("norm_type") == "instance"]
    best_inst = max(inst, key=acc) if inst else best_any

    # InstanceNorm is the safer DD default. Use BatchNorm only when it is
    # clearly better as a teacher; otherwise preserve DD stability.
    if best_any.get("norm_type") == "batch" and acc(best_any) >= acc(best_inst) + 2.0:
        return best_any
    return best_inst


def method_layer_for(selected: dict[str, Any]) -> str:
    size = int(selected["size"])
    depth = int(selected["depth"])
    if size >= 96 and depth >= 4:
        return "[2]"
    return "[1]"


def build_args(exp_root: Path, ncfm_repo: Path, pilot_launcher: Path, seed: int, niter: int, eval_epochs: int) -> SimpleNamespace:
    return SimpleNamespace(
        exp_root=exp_root,
        ncfm_repo=ncfm_repo,
        hop_repo=Path("/data/zengqiang/experiments/NCFMproject_0603/active_code/HoP-TM_kvasir/code"),
        seed=seed,
        niter=niter,
        eval_epochs=eval_epochs,
        epoch_eval_interval=100,
        workers=8,
        force=False,
        loss_scale=300.0,
        gpus="0,1",
        hop_num_experts=20,
        hop_train_epochs=20,
        hop_iterations=2000,
        pilot_launcher=pilot_launcher,
    )


def patch_pilot(pilot, selected: dict[str, Any]) -> None:
    size = int(selected["size"])
    depth = int(selected["depth"])
    norm_type = str(selected["norm_type"])
    net_type = str(selected["net_type"])
    width = float(selected.get("width", 1.0))
    epochs = int(selected.get("epochs", 100))
    batch_size = int(selected.get("batch_size", 64 if size >= 128 else 96))
    lr = float(selected.get("lr", 0.01))
    layer = method_layer_for(selected)

    pilot.DATASETS["kvasirv2"].update(
        {
            "size": size,
            "load_memory": False,
            "batch_real": 512,
            "batch_size": batch_size,
            "model_num": 20,
            "pretrain_epochs": epochs,
        }
    )
    pilot.NCFM_GROUPS = {name: pilot.NCFM_GROUPS[name] for name in GROUP_ORDER}
    pilot.NCFM_GROUPS["Kvasir_M22_SM_lam02_tau005_L1_nf256"]["discrepancy_attention_layers"] = layer
    pilot.NCFM_GROUPS["Kvasir_DRLTM_lam03_a100_L1_nf256"]["dr_ltm_layers"] = layer

    original_make = pilot.make_ncfm_config

    def make_config(args, dataset, save_dir, spec=None):
        cfg = original_make(args, dataset, save_dir, spec)
        if dataset == "kvasirv2":
            cfg["dataset"]["size"] = size
            cfg["dataset"]["batch_real"] = 512
            cfg["dataset"]["load_memory"] = False
            cfg["network"].update(
                {
                    "net_type": net_type,
                    "norm_type": norm_type,
                    "depth": depth,
                    "width": width,
                }
            )
            cfg["train"].update(
                {
                    "pertrain_epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "model_num": 20,
                }
            )
            cfg["condense"].update(
                {
                    "num_premodel": 20,
                    "iter_calib": 1,
                    "sampling_net": True,
                    "num_freqs": 4096,
                }
            )
            if cfg["condense"].get("use_discrepancy_attention_ncfd"):
                cfg["condense"]["discrepancy_attention_layers"] = layer
            if cfg["condense"].get("use_dr_ltm_ncfd"):
                cfg["condense"]["dr_ltm_layers"] = layer
        return cfg

    pilot.make_ncfm_config = make_config


def write_yaml(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def pretrain_parallel(pilot, args: SimpleNamespace, selected: dict[str, Any]) -> None:
    pretrain_dir = args.exp_root / "checkpoints" / "pretrain" / "kvasirv2"
    expected = [pretrain_dir / f"premodel{i}_trained.pth.tar" for i in range(20)]
    if all(path.exists() for path in expected):
        status(args.exp_root, f"PRETRAIN_REUSED dir={pretrain_dir}")
        return

    cfg = pilot.make_ncfm_config(args, "kvasirv2", args.exp_root / "results" / "pretrain_placeholder")
    cfg["save_path"]["pretrain_dir"] = str(args.exp_root / "checkpoints" / "pretrain")
    cfg["save_path"]["save_dir"] = str(args.exp_root / "results" / "pretrain_placeholder")
    cfg0 = json.loads(json.dumps(cfg))
    cfg1 = json.loads(json.dumps(cfg))
    cfg0["distibution_train"]["init_method"] = f"file:///{args.exp_root}/rdzv/pretrain_gpu0_{time.time_ns()}.store?rank=0&world_size=1"
    cfg1["distibution_train"]["init_method"] = f"file:///{args.exp_root}/rdzv/pretrain_gpu1_{time.time_ns()}.store?rank=0&world_size=1"
    cfg0_path = args.exp_root / "configs" / "kvasirv2" / "pretrain_gpu0.yaml"
    cfg1_path = args.exp_root / "configs" / "kvasirv2" / "pretrain_gpu1.yaml"
    write_yaml(cfg0_path, cfg0)
    write_yaml(cfg1_path, cfg1)

    status(
        args.exp_root,
        "PRETRAIN_PARALLEL_START "
        + f"selected={selected.get('name')} size={selected.get('size')} depth={selected.get('depth')} norm={selected.get('norm_type')}",
    )
    logs = [args.exp_root / "logs" / "pretrain_gpu0_0-9.log", args.exp_root / "logs" / "pretrain_gpu1_10-19.log"]
    commands = [
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=1",
            "pretrain/pretrain_range_script.py",
            "--config_path",
            str(cfg0_path),
            "--gpu",
            "0",
            "-i",
            "10",
            "--run_mode",
            "Pretrain",
            "--model_ids",
            "0-9",
        ],
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=1",
            "pretrain/pretrain_range_script.py",
            "--config_path",
            str(cfg1_path),
            "--gpu",
            "1",
            "-i",
            "10",
            "--run_mode",
            "Pretrain",
            "--model_ids",
            "10-19",
        ],
    ]
    procs = []
    files = []
    for gpu, (cmd, log) in enumerate(zip(commands, logs)):
        log.parent.mkdir(parents=True, exist_ok=True)
        f = log.open("w", encoding="utf-8")
        files.append(f)
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"})
        procs.append(subprocess.Popen(cmd, cwd=str(args.ncfm_repo), stdout=f, stderr=subprocess.STDOUT, text=True, env=env))
    failed = False
    for gpu, proc in enumerate(procs):
        rc = proc.wait()
        files[gpu].close()
        status(args.exp_root, f"PRETRAIN_WORKER_EXIT gpu={gpu} rc={rc}")
        failed = failed or rc != 0
    if failed:
        raise SystemExit("At least one Kvasir pretrain worker failed")
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing trained premodels after pretrain: {missing[:5]}")
    status(args.exp_root, f"PRETRAIN_PARALLEL_DONE dir={pretrain_dir}")


def run_dd_groups(pilot, args: SimpleNamespace) -> None:
    status_path = args.exp_root / "RUN_STATUS_NCFM_DD.txt"
    pilot.save_summary(args.exp_root)
    queue = list(GROUP_ORDER)
    running: list[tuple[str, int, subprocess.Popen, Any]] = []
    gpus = [0, 1]
    failed = False
    while queue or running:
        alive = []
        for name, gpu, proc, file_handle in running:
            rc = proc.poll()
            if rc is None:
                alive.append((name, gpu, proc, file_handle))
                continue
            file_handle.close()
            status(args.exp_root, f"DD_WORKER_EXIT group={name} gpu={gpu} rc={rc}")
            failed = failed or rc != 0
            pilot.save_summary(args.exp_root)
        running = alive
        used = {gpu for _, gpu, _, _ in running}
        for gpu in gpus:
            if gpu in used or not queue:
                continue
            name = queue.pop(0)
            log = args.exp_root / "logs" / f"dd_{name}_gpu{gpu}.screen.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--mode",
                "worker",
                "--group",
                name,
                "--gpu",
                str(gpu),
                "--exp-root",
                str(args.exp_root),
                "--audit-root",
                str(args.audit_root),
                "--ncfm-repo",
                str(args.ncfm_repo),
                "--pilot-launcher",
                str(args.pilot_launcher),
                "--selected-json",
                str(args.exp_root / "selected_teacher.json"),
                "--niter",
                str(args.niter),
                "--eval-epochs",
                str(args.eval_epochs),
                "--seed",
                str(args.seed),
            ]
            f = log.open("w", encoding="utf-8")
            env = os.environ.copy()
            env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"})
            proc = subprocess.Popen(cmd, cwd=str(args.ncfm_repo), stdout=f, stderr=subprocess.STDOUT, text=True, env=env)
            running.append((name, gpu, proc, f))
            status(args.exp_root, f"DD_WORKER_START group={name} gpu={gpu}")
        time.sleep(30)
    pilot.save_summary(args.exp_root)
    if failed:
        raise SystemExit("At least one DD group failed")
    status(args.exp_root, "DD_ALL_DONE")


def write_decision(exp_root: Path, selected: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    reports = exp_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (exp_root / "selected_teacher.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    with (reports / "teacher_selection.md").open("w", encoding="utf-8") as f:
        f.write("# Kvasir Teacher Selection\n\n")
        f.write(f"Selected: `{selected.get('name')}`\n\n")
        f.write("| Config | Model | Size | Norm | ACC | AUC | Macro-F1 | BACC |\n")
        f.write("|---|---|---:|---|---:|---:|---:|---:|\n")
        for row in sorted(rows, key=lambda r: float(r.get("acc_percent") or -1), reverse=True):
            f.write(
                f"| {row.get('name')} | {row.get('net_type')}-{row.get('depth')} | "
                f"{row.get('size')} | {row.get('norm_type')} | {row.get('acc_percent')} | "
                f"{row.get('auc_macro_ovr')} | {row.get('macro_f1')} | {row.get('balanced_acc')} |\n"
            )


def main_controller(args: argparse.Namespace) -> None:
    exp_root = Path(args.exp_root)
    audit_root = Path(args.audit_root)
    ncfm_repo = Path(args.ncfm_repo)
    pilot_launcher = Path(args.pilot_launcher)
    exp_root.mkdir(parents=True, exist_ok=True)
    (exp_root / "rdzv").mkdir(parents=True, exist_ok=True)
    status(exp_root, "CONTROLLER_START")

    rows = wait_for_audit(audit_root, exp_root, args.audit_timeout_hours)
    selected = choose_teacher(rows)
    write_decision(exp_root, selected, rows)
    status(
        exp_root,
        f"TEACHER_SELECTED name={selected.get('name')} acc={selected.get('acc_percent')} "
        f"size={selected.get('size')} depth={selected.get('depth')} norm={selected.get('norm_type')}",
    )

    pilot = load_pilot_module(pilot_launcher)
    patch_pilot(pilot, selected)
    run_args = build_args(exp_root, ncfm_repo, pilot_launcher, args.seed, args.niter, args.eval_epochs)
    run_args.audit_root = audit_root
    pretrain_parallel(pilot, run_args, selected)
    run_dd_groups(pilot, run_args)
    status(exp_root, "CONTROLLER_DONE")


def main_worker(args: argparse.Namespace) -> None:
    selected = json.loads(Path(args.selected_json).read_text(encoding="utf-8"))
    pilot = load_pilot_module(Path(args.pilot_launcher))
    patch_pilot(pilot, selected)
    run_args = build_args(Path(args.exp_root), Path(args.ncfm_repo), Path(args.pilot_launcher), args.seed, args.niter, args.eval_epochs)
    run_args.audit_root = Path(args.audit_root)
    pilot.run_ncfm_group(run_args, args.group, gpu=int(args.gpu), status=Path(args.exp_root) / "RUN_STATUS_NCFM_DD.txt")
    pilot.save_summary(Path(args.exp_root))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["controller", "worker"], default="controller")
    parser.add_argument("--exp-root", required=True)
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--ncfm-repo", required=True)
    parser.add_argument("--pilot-launcher", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--niter", type=int, default=20000)
    parser.add_argument("--eval-epochs", type=int, default=2000)
    parser.add_argument("--audit-timeout-hours", type=float, default=12.0)
    parser.add_argument("--group", choices=GROUP_ORDER)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--selected-json")
    args = parser.parse_args()
    if args.mode == "controller":
        main_controller(args)
    else:
        if not args.group or not args.selected_json:
            raise SystemExit("--group and --selected-json are required in worker mode")
        main_worker(args)


if __name__ == "__main__":
    main()

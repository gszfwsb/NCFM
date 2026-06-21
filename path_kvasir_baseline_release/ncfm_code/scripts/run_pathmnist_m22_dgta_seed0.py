import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


DATASET = "pathmnist"
IPC = 10
NCLASS = 9
NCH = 3

JOBS = [
    dict(group="M22_lam01_tau01_L1_nf256", lam=0.1, tau=0.1, layers="[1]", nf=256, mode="softmax", topk=0),
    dict(group="M22_lam02_tau01_L1_nf256", lam=0.2, tau=0.1, layers="[1]", nf=256, mode="softmax", topk=0),
    dict(group="M22_lam03_tau01_L1_nf256", lam=0.3, tau=0.1, layers="[1]", nf=256, mode="softmax", topk=0),
    dict(group="M22_lam02_tau005_L1_nf256", lam=0.2, tau=0.05, layers="[1]", nf=256, mode="softmax", topk=0),
    dict(group="M22_lam02_tau02_L1_nf256", lam=0.2, tau=0.2, layers="[1]", nf=256, mode="softmax", topk=0),
    dict(group="M22_lam02_tau01_L1_top4_nf256", lam=0.2, tau=0.1, layers="[1]", nf=256, mode="softmax", topk=4),
    dict(group="M22_lam02_uniform_L1_nf256", lam=0.2, tau=0.1, layers="[1]", nf=256, mode="uniform", topk=0),
]


def append(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def copy_assets(exp_root: Path, data_source: Path, pretrain_source: Path) -> None:
    data_dst = exp_root / "data" / "medmnist"
    pretrain_dst = exp_root / "checkpoints" / "pretrain" / DATASET
    data_dst.mkdir(parents=True, exist_ok=True)
    pretrain_dst.mkdir(parents=True, exist_ok=True)
    data_target = data_dst / "pathmnist.npz"
    if not data_target.exists():
        shutil.copy2(data_source, data_target)
    copied = 0
    for src in pretrain_source.glob("premodel*.pth.tar"):
        dst = pretrain_dst / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        copied += 1
    if copied < 40:
        raise RuntimeError(f"Expected >=40 premodel files in {pretrain_source}, found {copied}")


def make_config(args, job):
    rdzv_dir = args.exp_root / "rdzv"
    rdzv_dir.mkdir(parents=True, exist_ok=True)
    store = rdzv_dir / f"{DATASET}_{job['group']}_{int(time.time() * 1000000)}.store"
    init_method = "file:///" + str(store).replace("\\", "/") + "?rank=0&world_size=1"
    return {
        "distibution_train": {"backend": "gloo", "init_method": init_method, "workers": args.workers},
        "dataset": {
            "dataset": DATASET,
            "nclass": NCLASS,
            "size": 28,
            "data_dir": str(args.exp_root / "data"),
            "load_memory": True,
            "batch_real": args.batch_real,
            "nch": NCH,
        },
        "network": {"net_type": "convnet", "norm_type": "instance", "depth": 3, "width": 1.0},
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
            "use_discrepancy_attention_ncfd": True,
            "lambda_discrepancy_attention_ncfd": float(job["lam"]),
            "discrepancy_attention_layers": job["layers"],
            "discrepancy_attention_num_freqs": int(job["nf"]),
            "discrepancy_attention_tau": float(job["tau"]),
            "discrepancy_attention_mode": job["mode"],
            "discrepancy_attention_topk": int(job["topk"]),
            "discrepancy_attention_loss_scale": float(args.loss_scale),
            "discrepancy_attention_detach_real": True,
            "discrepancy_attention_log_components": True,
        },
    }


def write_yaml(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def run_command(cmd, cwd, stdout_path, stderr_path, gpu):
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["LOCAL_RANK"] = "0"
    env["RANK"] = "0"
    env["WORLD_SIZE"] = "1"
    env["LOCAL_WORLD_SIZE"] = "1"
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("MASTER_PORT", str(29900 + int(gpu)))
    env.setdefault("PYTHONUTF8", "1")
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        proc = subprocess.run(cmd, cwd=cwd, stdout=out, stderr=err, text=True, env=env)
    return proc.returncode


def torchrun(script, config_path, gpu, ipc=None, extra=None):
    # The project entrypoints set CUDA_VISIBLE_DEVICES from --gpu internally,
    # so --gpu must stay as the physical GPU id used for this run.
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1", script, "--config_path", str(config_path), "--gpu", str(gpu)]
    if ipc is not None:
        cmd.extend(["-i", str(ipc)])
    if extra:
        cmd.extend(extra)
    return cmd


def newest_distilled_data(exp_root, start_time):
    candidates = []
    root = exp_root / "results" / "condense"
    if root.exists():
        for path in root.glob(f"**/{DATASET}/ipc{IPC}/**/distilled_data/data_*.pt"):
            if path.name != "data_init.pt" and path.stat().st_mtime >= start_time:
                candidates.append(path)
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def read_best_metrics(metrics_jsonl):
    best_path = metrics_jsonl.with_name(metrics_jsonl.stem + "_best.json")
    if best_path.exists():
        return json.loads(best_path.read_text(encoding="utf-8"))
    best = {}
    if metrics_jsonl.exists():
        for line in metrics_jsonl.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("is_best"):
                best = rec
    return best


def parse_best_accuracy(stdout_path):
    text = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
    matches = re.findall(r"Best\s+accuracy \(top-1 and 5\):\s*([0-9.]+)", text)
    return float(matches[-1]) if matches else None


def run_job(args, repo_dir, job):
    group = job["group"]
    run_dir = args.exp_root / "runs" / DATASET / f"ipc{IPC}" / group
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = args.exp_root / "configs" / DATASET / f"ipc{IPC}_{group}.yaml"
    cfg = make_config(args, job)
    write_yaml(cfg_path, cfg)

    condensed_record = run_dir / "condensed_path.txt"
    condensed_path = None
    if condensed_record.exists() and not args.force:
        candidate = Path(condensed_record.read_text(encoding="utf-8").strip())
        if candidate.exists():
            condensed_path = candidate

    if condensed_path is None:
        start = datetime.now().timestamp()
        condense_cmd = torchrun("condense/condense_script.py", cfg_path, args.gpu, IPC, ["--run_mode", "Condense", "--init", "mix"])
        (run_dir / "condense_command.txt").write_text(" ".join(condense_cmd) + "\n", encoding="utf-8")
        rc = run_command(condense_cmd, repo_dir, run_dir / "condense_stdout.log", run_dir / "condense_stderr.log", args.gpu)
        if rc != 0:
            raise RuntimeError(f"condense failed: {group}")
        condensed_path = newest_distilled_data(args.exp_root, start)
        if condensed_path is None:
            raise RuntimeError(f"no distilled data found: {group}")
        condensed_record.write_text(str(condensed_path) + "\n", encoding="utf-8")

    ckpt = args.exp_root / "checkpoints" / "synthetic_train" / DATASET / f"ipc{IPC}_{group}_best.pth.tar"
    eval_metrics = run_dir / "eval_metrics.jsonl"
    eval_cmd = torchrun(
        "evaluation/evaluation_script.py",
        cfg_path,
        args.gpu,
        IPC,
        ["--run_mode", "Evaluation", "--load_path", str(condensed_path), "--val_repeat", "1", "--eval_checkpoint_path", str(ckpt), "--eval_metrics_path", str(eval_metrics)],
    )
    (run_dir / "eval_command.txt").write_text(" ".join(eval_cmd) + "\n", encoding="utf-8")
    rc = run_command(eval_cmd, repo_dir, run_dir / "eval_stdout.log", run_dir / "eval_stderr.log", args.gpu)
    if rc != 0:
        raise RuntimeError(f"eval failed: {group}")
    best = read_best_metrics(eval_metrics)
    metrics = {
        "dataset": DATASET,
        "ipc": IPC,
        "group": group,
        "method": "M22_DGTA",
        "lambda": job["lam"],
        "tau": job["tau"],
        "layers": job["layers"],
        "nf": job["nf"],
        "mode": job["mode"],
        "topk": job["topk"],
        "condensed_path": str(condensed_path),
        "checkpoint_path": str(ckpt),
        "accuracy": best.get("acc_percent") or parse_best_accuracy(run_dir / "eval_stdout.log"),
        **best,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def save_summary(exp_root, rows):
    report_dir = exp_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "pathmnist_m22_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fields = ["dataset", "group", "lambda", "tau", "layers", "nf", "mode", "topk", "acc_percent", "auc_macro_ovr", "macro_f1", "balanced_acc", "condensed_path", "checkpoint_path"]
    with (report_dir / "pathmnist_m22_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| Group | lambda | tau | mode | topk | ACC | AUC | Macro-F1 | BACC |", "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r.get('group')} | {r.get('lambda')} | {r.get('tau')} | {r.get('mode')} | {r.get('topk')} | {r.get('acc_percent')} | {r.get('auc_macro_ovr')} | {r.get('macro_f1')} | {r.get('balanced_acc')} |")
    (report_dir / "pathmnist_m22_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_root", type=Path, required=True)
    parser.add_argument("--data_source", type=Path, required=True)
    parser.add_argument("--pretrain_source", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--niter", type=int, default=20000)
    parser.add_argument("--eval_epochs", type=int, default=2000)
    parser.add_argument("--epoch_eval_interval", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--batch_real", type=int, default=1024)
    parser.add_argument("--model_num", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loss_scale", type=float, default=300.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parents[1]
    args.exp_root.mkdir(parents=True, exist_ok=True)
    copy_assets(args.exp_root, args.data_source, args.pretrain_source)
    rows = []
    for job in JOBS:
        try:
            rows.append(run_job(args, repo_dir, job))
            save_summary(args.exp_root, rows)
        except Exception as exc:
            err = {"group": job["group"], "error": repr(exc), "time": datetime.now().isoformat(timespec="seconds")}
            append(args.exp_root / "FAILED_JOBS.jsonl", json.dumps(err) + "\n")
            raise
    save_summary(args.exp_root, rows)


if __name__ == "__main__":
    main()

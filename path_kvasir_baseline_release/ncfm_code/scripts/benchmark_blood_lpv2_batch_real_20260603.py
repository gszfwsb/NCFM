import argparse
import csv
import importlib.util
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path


DATASET = "bloodmnist"
IPC = 10


def load_pipeline():
    here = Path(__file__).resolve().parent
    path = here / "run_medmnist_formal_pipeline.py"
    spec = importlib.util.spec_from_file_location("formal_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def local_patch_group():
    return {
        "sampling_net": True,
        "num_freqs": 512,
        "iter_calib": 0,
        "calib_weight": 1,
        "use_local_patch_feature_ncfd": True,
        "local_patch_grid": 4,
        "lambda_local_patch_ncfd": 0.3,
        "local_patch_feature_dim": 128,
        "local_patch_encoder_blocks": 1,
        "local_patch_num_freqs": 512,
        "local_patch_encoder_source": "premodel_trained",
        "local_patch_premodel_index": 0,
        "local_patch_encoder_frozen": True,
        "use_local_patch_sampling_net": False,
        "dam_enabled": False,
        "use_ssim_regularization": False,
    }


EXTRA_KEYS = [
    "local_patch_checkpoint_stage",
    "local_patch_premodel_index",
    "local_patch_premodel_indices",
    "local_patch_model_num",
    "local_patch_ensemble_size",
    "local_patch_ensemble_random",
    "local_patch_ensemble_aggregate",
    "local_patch_encoder_seed",
    "local_patch_loss_scale",
]


def patch_make_config(pipeline):
    old_make_config = pipeline.make_config

    def make_config(args, dataset, ipc, group_name=None, niter=None, metrics_path=None):
        cfg = old_make_config(
            args,
            dataset,
            ipc,
            group_name=group_name,
            niter=niter,
            metrics_path=metrics_path,
        )
        group = pipeline.GROUPS.get(group_name, {})
        for key, value in group.items():
            if key in EXTRA_KEYS:
                cfg["condense"][key] = value
        return cfg

    pipeline.make_config = make_config


def ensure_link(link_path, target_path):
    link_path = Path(link_path)
    target_path = Path(target_path)
    if link_path.exists() or link_path.is_symlink():
        return
    if not target_path.exists():
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target_path, link_path, target_is_directory=True)


def ensure_assets(exp_root):
    project = Path("/data/zengqiang/project_6_2")
    ensure_link(exp_root / "data", project / "shared_assets" / "data")
    ensure_link(
        exp_root / "checkpoints" / "pretrain",
        project / "shared_assets" / "checkpoints" / "pretrain",
    )
    ensure_link(
        exp_root / "checkpoints" / "pretrain",
        project
        / "experiments"
        / "stage2_blood_T512_lpv2_encoder_ablation_seed0_br4096_20260602"
        / "checkpoints"
        / "pretrain",
    )


def query_gpu(gpu):
    cmd = [
        "nvidia-smi",
        "-i",
        str(gpu),
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        util, mem, total = [int(part.strip()) for part in out.split(",")]
        return util, mem, total
    except Exception:
        return None


def parse_it_s(stderr_path):
    text = stderr_path.read_text(encoding="utf-8", errors="replace")
    vals = []
    for match in re.finditer(r"([0-9]+(?:\.[0-9]+)?)(it/s|s/it)", text):
        value = float(match.group(1))
        unit = match.group(2)
        vals.append(value if unit == "it/s" else 1.0 / value)
    if not vals:
        return None, None, 0
    stable = vals[10:] if len(vals) > 20 else vals
    tail = stable[-50:]
    return statistics.median(tail), tail[-1], len(vals)


def run_one(args, pipeline, repo_dir, batch_real):
    group_name = f"LPv2_bench_br{batch_real}"
    pipeline.GROUPS[group_name] = local_patch_group()

    run_dir = args.exp_root / "runs" / DATASET / f"ipc{IPC}" / group_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.exp_root / "configs" / DATASET / f"ipc{IPC}_{group_name}.yaml"
    args.batch_real = batch_real
    config = pipeline.make_config(
        args, DATASET, ipc=IPC, group_name=group_name, niter=args.niter
    )
    pipeline.write_config(config_path, config)

    command = pipeline.torchrun_command(
        "condense/condense_script.py",
        config_path,
        args.gpu,
        ipc=IPC,
        extra=["--run_mode", "Condense", "--init", "mix"],
    )
    stdout_path = run_dir / "condense_stdout.log"
    stderr_path = run_dir / "condense_stderr.log"
    monitor_path = run_dir / "gpu_monitor.csv"
    (run_dir / "condense_command.txt").write_text(
        " ".join(command) + "\n", encoding="utf-8"
    )

    samples = []
    start = time.time()
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open(
        "w", encoding="utf-8"
    ) as err, monitor_path.open("w", newline="", encoding="utf-8") as mon:
        writer = csv.DictWriter(
            mon, fieldnames=["elapsed_s", "util_pct", "memory_mib", "total_mib"]
        )
        writer.writeheader()
        proc = subprocess.Popen(command, cwd=repo_dir, stdout=out, stderr=err, text=True)
        while proc.poll() is None:
            gpu_state = query_gpu(args.gpu)
            elapsed = time.time() - start
            if gpu_state:
                util, mem, total = gpu_state
                samples.append((elapsed, util, mem, total))
                writer.writerow(
                    {
                        "elapsed_s": f"{elapsed:.3f}",
                        "util_pct": util,
                        "memory_mib": mem,
                        "total_mib": total,
                    }
                )
                mon.flush()
            time.sleep(args.monitor_interval)
        rc = proc.returncode
        gpu_state = query_gpu(args.gpu)
        if gpu_state:
            elapsed = time.time() - start
            util, mem, total = gpu_state
            samples.append((elapsed, util, mem, total))
            writer.writerow(
                {
                    "elapsed_s": f"{elapsed:.3f}",
                    "util_pct": util,
                    "memory_mib": mem,
                    "total_mib": total,
                }
            )

    duration = time.time() - start
    median_it_s, last_it_s, it_s_points = parse_it_s(stderr_path)
    peak_mem = max((sample[2] for sample in samples), default=None)
    avg_util = statistics.mean(sample[1] for sample in samples) if samples else None
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    oom = "out of memory" in stderr_text.lower()
    return {
        "group": group_name,
        "gpu": args.gpu,
        "batch_real": batch_real,
        "niter": args.niter,
        "returncode": rc,
        "oom": oom,
        "duration_s": duration,
        "median_it_s": median_it_s,
        "last_it_s": last_it_s,
        "it_s_points": it_s_points,
        "peak_memory_mib": peak_mem,
        "avg_util_pct": avg_util,
        "run_dir": str(run_dir),
    }


def write_summary(exp_root, rows, suffix):
    report_dir = exp_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / f"bloodmnist_lpv2_batch_real_benchmark_{suffix}.csv"
    fields = [
        "group",
        "gpu",
        "batch_real",
        "niter",
        "returncode",
        "oom",
        "duration_s",
        "median_it_s",
        "last_it_s",
        "it_s_points",
        "peak_memory_mib",
        "avg_util_pct",
        "run_dir",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    md_path = csv_path.with_suffix(".md")
    lines = [
        "| batch_real | GPU | median it/s | last it/s | peak MiB | avg util % | rc | OOM | run_dir |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {batch_real} | {gpu} | {median_it_s:.4f} | {last_it_s:.4f} | {peak_memory_mib} | {avg_util_pct:.1f} | {returncode} | {oom} | `{run_dir}` |".format(
                **{
                    **row,
                    "median_it_s": row["median_it_s"] or 0.0,
                    "last_it_s": row["last_it_s"] or 0.0,
                    "peak_memory_mib": row["peak_memory_mib"] or 0,
                    "avg_util_pct": row["avg_util_pct"] or 0.0,
                }
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Benchmark BloodMNIST LPv2 batch_real.")
    parser.add_argument("--exp_root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--batch_reals", required=True)
    parser.add_argument("--niter", type=int, default=500)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model_num", type=int, default=20)
    parser.add_argument("--pretrain_epochs", type=int, default=60)
    parser.add_argument("--eval_epochs", type=int, default=1)
    parser.add_argument("--epoch_eval_interval", type=int, default=100)
    parser.add_argument("--monitor_interval", type=float, default=1.0)
    args = parser.parse_args()

    args.exp_root.mkdir(parents=True, exist_ok=True)
    ensure_assets(args.exp_root)

    pipeline = load_pipeline()
    patch_make_config(pipeline)
    repo_dir = Path(__file__).resolve().parents[1]
    rows = []
    for item in args.batch_reals.split(","):
        batch_real = int(item.strip())
        if not item.strip():
            continue
        row = run_one(args, pipeline, repo_dir, batch_real)
        rows.append(row)
        write_summary(args.exp_root, rows, f"gpu{args.gpu}")
        if row["oom"]:
            break
    write_summary(args.exp_root, rows, f"gpu{args.gpu}")


if __name__ == "__main__":
    main()

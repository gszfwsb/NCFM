import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


GROUPS_GPU0 = ",".join(
    [
        "LPv2_T1024_lam03_g4_b2_p0",
        "LPv2_T1024_lam08_g4_b2_p0",
        "LPv2_T1024_lam03_g7_b1_p0",
    ]
)
GROUPS_GPU1 = ",".join(
    [
        "LPv2_T1024_lam08_g7_b1_p0",
        "LPv2_T1024_lam03_g4_b1_ens0123",
        "LPv2_T1024_lam08_g4_b1_ens0123",
    ]
)


def now():
    return datetime.now().isoformat(timespec="seconds")


def query_gpus():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True, encoding="utf-8", errors="replace")
    stats = []
    for line in out.splitlines():
        if not line.strip():
            continue
        idx, util, mem_used, mem_total = [part.strip() for part in line.split(",")]
        stats.append(
            {
                "index": int(idx),
                "util": int(util),
                "mem_used": int(mem_used),
                "mem_total": int(mem_total),
            }
        )
    return stats


def gpus_are_free(stats, max_util, max_mem_mb):
    wanted = {0, 1}
    seen = {item["index"] for item in stats}
    if wanted - seen:
        return False
    for item in stats:
        if item["index"] in wanted:
            if item["util"] > max_util or item["mem_used"] > max_mem_mb:
                return False
    return True


def write_status(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def launch_worker(args, gpu, groups, name):
    log_dir = args.exp_root / "launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve().parent / "run_bloodmnist_lpv2_effective_20260603.py"),
        "--exp_root",
        str(args.exp_root),
        "--data_source",
        str(args.data_source),
        "--pretrain_source",
        str(args.pretrain_source),
        "--seed",
        str(args.seed),
        "--gpu",
        str(gpu),
        "--groups",
        groups,
        "--niter",
        str(args.niter),
        "--eval_epochs",
        str(args.eval_epochs),
        "--epoch_eval_interval",
        str(args.epoch_eval_interval),
        "--batch_real",
        str(args.batch_real),
        "--batch_size",
        str(args.batch_size),
    ]
    (log_dir / f"{name}_command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    out = (log_dir / f"{name}_stdout.log").open("w", encoding="utf-8")
    err = (log_dir / f"{name}_stderr.log").open("w", encoding="utf-8")
    return subprocess.Popen(cmd, stdout=out, stderr=err)


def main():
    parser = argparse.ArgumentParser(description="Wait for free L20 GPUs and launch LPv2.")
    parser.add_argument("--exp_root", type=Path, required=True)
    parser.add_argument("--data_source", type=Path, required=True)
    parser.add_argument("--pretrain_source", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--niter", type=int, default=20000)
    parser.add_argument("--eval_epochs", type=int, default=2000)
    parser.add_argument("--epoch_eval_interval", type=int, default=100)
    parser.add_argument("--batch_real", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_util", type=int, default=30)
    parser.add_argument("--max_mem_mb", type=int, default=2500)
    parser.add_argument("--poll_seconds", type=int, default=300)
    args = parser.parse_args()

    status_path = args.exp_root / "RUN_STATUS_LPV2_WATCHER.json"
    args.exp_root.mkdir(parents=True, exist_ok=True)

    while True:
        stats = query_gpus()
        payload = {
            "time": now(),
            "status": "waiting_for_free_gpus",
            "max_util": args.max_util,
            "max_mem_mb": args.max_mem_mb,
            "gpu_stats": stats,
        }
        write_status(status_path, payload)
        if gpus_are_free(stats, args.max_util, args.max_mem_mb):
            break
        time.sleep(args.poll_seconds)

    procs = [
        launch_worker(args, 0, GROUPS_GPU0, "lpv2_seed0_gpu0"),
        launch_worker(args, 1, GROUPS_GPU1, "lpv2_seed0_gpu1"),
    ]
    write_status(
        status_path,
        {
            "time": now(),
            "status": "launched",
            "workers": [
                {"gpu": 0, "groups": GROUPS_GPU0, "pid": procs[0].pid},
                {"gpu": 1, "groups": GROUPS_GPU1, "pid": procs[1].pid},
            ],
        },
    )
    return_codes = [proc.wait() for proc in procs]
    write_status(
        status_path,
        {
            "time": now(),
            "status": "finished",
            "return_codes": return_codes,
        },
    )
    if any(code != 0 for code in return_codes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

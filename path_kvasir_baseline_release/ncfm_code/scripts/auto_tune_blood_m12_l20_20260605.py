import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PYTHON = r"C:\Users\Administrator\anaconda3\envs\pyprc\python.exe"
PROJECT_ROOT = Path(r"C:\xxyProject\NCFMproject_0603")
CODE_DIR = PROJECT_ROOT / "active_code" / "M12_local_patch_rand20_step"
RUNNER = CODE_DIR / "scripts" / "run_blood_m12_rand20_single_20260603.py"

AUTO_ROOT = PROJECT_ROOT / "e" / "btune0605"
LOG_DIR = AUTO_ROOT / "launcher_logs"
STATUS_PATH = AUTO_ROOT / "AUTO_TUNE_STATUS.json"
LEADERBOARD_JSON = AUTO_ROOT / "reports" / "blood_auto_tune_m12_seed0_20260605.json"
LEADERBOARD_MD = AUTO_ROOT / "reports" / "blood_auto_tune_m12_seed0_20260605.md"

DATA_SOURCE = (
    PROJECT_ROOT
    / "archive_legacy_experiments"
    / "experiments_0528"
    / "bloodmnist_method_sweep_seed0_20260530"
    / "data"
    / "medmnist"
)
PRETRAIN_SOURCE = (
    PROJECT_ROOT
    / "archive_legacy_experiments"
    / "experiments_0528"
    / "bloodmnist_method_sweep_seed0_20260530"
    / "checkpoints"
    / "pretrain"
    / "bloodmnist"
)

DATASET = "bloodmnist"
IPC = 10
BASELINE_ACC = 90.12
BASELINE_BACC = 0.9019
SLOTS_PER_GPU = 2
GPU_IDS = [0, 1]


def candidate(name, t, lam, grid, blocks, local_t=512):
    return {
        "name": name,
        "num_freqs": t,
        "lambda_local_patch_ncfd": lam,
        "local_patch_grid": grid,
        "local_patch_encoder_blocks": blocks,
        "local_patch_num_freqs": local_t,
    }


CANDIDATES = [
    candidate("m12_t1024_l06_g4_b2", 1024, 0.6, 4, 2),
    candidate("m12_t1024_l08_g2_b2", 1024, 0.8, 2, 2),
    candidate("m12_t1024_l07_g4_b2", 1024, 0.7, 4, 2),
    candidate("m12_t1024_l05_g2_b2", 1024, 0.5, 2, 2),
    candidate("m12_t1024_l08_g4_b1", 1024, 0.8, 4, 1),
    candidate("m12_t1024_l05_g4_b1", 1024, 0.5, 4, 1),
    candidate("m12_t1024_l08_g4_b2_lf1024", 1024, 0.8, 4, 2, 1024),
    candidate("m12_t512_l08_g4_b2", 512, 0.8, 4, 2),
    candidate("m12_t512_l08_g2_b2", 512, 0.8, 2, 2),
    candidate("m12_t1024_l10_g4_b2", 1024, 1.0, 4, 2),
]


def now():
    return datetime.now().isoformat(timespec="seconds")


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def load_json(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def candidate_exp_root(cand):
    return AUTO_ROOT / cand["name"]


def candidate_metrics_path(cand):
    return (
        candidate_exp_root(cand)
        / "runs"
        / DATASET
        / f"ipc{IPC}"
        / cand["name"]
        / "metrics.json"
    )


def read_metric(cand):
    path = candidate_metrics_path(cand)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    data["auto_tune_name"] = cand["name"]
    data["metric_path"] = str(path)
    data["delta_acc_vs_B_T1024"] = data.get("acc_percent", 0) - BASELINE_ACC
    data["delta_bacc_vs_B_T1024"] = data.get("balanced_acc", 0) - BASELINE_BACC
    return data


def gpu_memory():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True, encoding="utf-8", errors="replace")
    result = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            idx = int(parts[0])
            result[idx] = {
                "memory_used_mb": int(parts[1]),
                "memory_total_mb": int(parts[2]),
                "utilization_gpu": int(parts[3]),
            }
    return result


def external_root_jobs_by_gpu():
    ps = r"""
Get-CimInstance Win32_Process |
  Where-Object {
    $_.Name -match 'python' -and
    $_.CommandLine -match 'NCFMproject_0603' -and
    $_.CommandLine -match 'scripts\\run_'
  } |
  Select-Object ProcessId,CommandLine |
  ConvertTo-Json -Depth 3
"""
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", ps],
            text=True,
            encoding="utf-8",
            errors="replace",
        ).strip()
    except Exception:
        return {gpu: [] for gpu in GPU_IDS}
    if not out:
        return {gpu: [] for gpu in GPU_IDS}
    try:
        records = json.loads(out)
    except Exception:
        return {gpu: [] for gpu in GPU_IDS}
    if isinstance(records, dict):
        records = [records]
    by_gpu = {gpu: [] for gpu in GPU_IDS}
    for rec in records:
        cmd = rec.get("CommandLine") or ""
        for gpu in GPU_IDS:
            if f"--gpu {gpu}" in cmd or f"--gpu={gpu}" in cmd:
                by_gpu[gpu].append(rec)
    return by_gpu


def save_status(queue, running, completed, failed, note=""):
    metrics = [m for cand in CANDIDATES if (m := read_metric(cand))]
    metrics.sort(key=lambda item: item.get("acc_percent", -1), reverse=True)
    payload = {
        "updated_at": now(),
        "note": note,
        "auto_root": str(AUTO_ROOT),
        "slots_per_gpu": SLOTS_PER_GPU,
        "baseline": {"group": "B_T1024", "acc_percent": BASELINE_ACC, "balanced_acc": BASELINE_BACC},
        "gpu": gpu_memory(),
        "queue": [c["name"] for c in queue],
        "running": {
            str(gpu): [
                {
                    "name": job["cand"]["name"],
                    "pid": job["proc"].pid,
                    "log": str(job["stdout"]),
                }
                for job in jobs
            ]
            for gpu, jobs in running.items()
        },
        "completed": completed,
        "failed": failed,
        "leaderboard": metrics,
    }
    write_text(STATUS_PATH, json.dumps(payload, indent=2))
    save_leaderboard(metrics)


def save_leaderboard(metrics):
    AUTO_ROOT.joinpath("reports").mkdir(parents=True, exist_ok=True)
    write_text(LEADERBOARD_JSON, json.dumps(metrics, indent=2))
    lines = [
        "# BloodMNIST M12 Auto Tune Seed0",
        "",
        f"Baseline: B_T1024 ACC={BASELINE_ACC:.2f}, BACC={BASELINE_BACC:.4f}",
        "",
        "| Rank | Group | ACC | Delta ACC | AUC | Macro-F1 | BACC | Delta BACC |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for i, item in enumerate(metrics, 1):
        lines.append(
            "| {rank} | {group} | {acc:.4f} | {dacc:+.4f} | {auc:.4f} | {f1:.4f} | {bacc:.4f} | {dbacc:+.4f} |".format(
                rank=i,
                group=item.get("group", item.get("auto_tune_name")),
                acc=float(item.get("acc_percent", 0)),
                dacc=float(item.get("delta_acc_vs_B_T1024", 0)),
                auc=float(item.get("auc_macro_ovr", 0)),
                f1=float(item.get("macro_f1", 0)),
                bacc=float(item.get("balanced_acc", 0)),
                dbacc=float(item.get("delta_bacc_vs_B_T1024", 0)),
            )
        )
    write_text(LEADERBOARD_MD, "\n".join(lines) + "\n")


def launch(cand, gpu):
    exp_root = candidate_exp_root(cand)
    stdout_path = LOG_DIR / f"{cand['name']}_gpu{gpu}_stdout.log"
    stderr_path = LOG_DIR / f"{cand['name']}_gpu{gpu}_stderr.log"
    exp_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON,
        "-u",
        str(RUNNER),
        "--exp_root",
        str(exp_root),
        "--data_source",
        str(DATA_SOURCE),
        "--pretrain_source",
        str(PRETRAIN_SOURCE),
        "--seed",
        "0",
        "--gpu",
        str(gpu),
        "--workers",
        "8",
        "--batch_real",
        "4096",
        "--batch_size",
        "1024",
        "--model_num",
        "20",
        "--eval_epochs",
        "2000",
        "--epoch_eval_interval",
        "100",
        "--niter",
        "20000",
        "--num_freqs",
        str(cand["num_freqs"]),
        "--local_patch_num_freqs",
        str(cand["local_patch_num_freqs"]),
        "--local_patch_loss_scale",
        "300.0",
        "--lambda_local_patch_ncfd",
        str(cand["lambda_local_patch_ncfd"]),
        "--local_patch_grid",
        str(cand["local_patch_grid"]),
        "--local_patch_encoder_blocks",
        str(cand["local_patch_encoder_blocks"]),
        "--group_name",
        cand["name"],
        "--force",
    ]
    write_text(LOG_DIR / f"{cand['name']}_gpu{gpu}_command.txt", " ".join(cmd) + "\n")
    out = stdout_path.open("w", encoding="utf-8")
    err = stderr_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(CODE_DIR), stdout=out, stderr=err, text=True)
    return {"cand": cand, "proc": proc, "stdout": stdout_path, "stderr": stderr_path, "out_handle": out, "err_handle": err}


def close_job_handles(job):
    for key in ("out_handle", "err_handle"):
        try:
            job[key].close()
        except Exception:
            pass


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    AUTO_ROOT.joinpath("reports").mkdir(parents=True, exist_ok=True)
    append_text(LOG_DIR / "auto_tune_events.log", f"[{now()}] auto tuner started\n")
    if not DATA_SOURCE.exists():
        raise FileNotFoundError(DATA_SOURCE)
    if not PRETRAIN_SOURCE.exists():
        raise FileNotFoundError(PRETRAIN_SOURCE)
    if not RUNNER.exists():
        raise FileNotFoundError(RUNNER)

    completed = []
    failed = []
    queue = []
    for cand in CANDIDATES:
        if read_metric(cand):
            completed.append(cand["name"])
        else:
            queue.append(cand)

    running = {gpu: [] for gpu in GPU_IDS}
    while queue or any(running.values()):
        external = external_root_jobs_by_gpu()
        for gpu in GPU_IDS:
            alive = []
            for job in running[gpu]:
                rc = job["proc"].poll()
                if rc is None:
                    alive.append(job)
                    continue
                close_job_handles(job)
                name = job["cand"]["name"]
                if rc == 0 and read_metric(job["cand"]):
                    completed.append(name)
                    append_text(LOG_DIR / "auto_tune_events.log", f"[{now()}] completed {name} rc={rc}\n")
                else:
                    failed.append({"name": name, "returncode": rc, "stdout": str(job["stdout"]), "stderr": str(job["stderr"])})
                    append_text(LOG_DIR / "auto_tune_events.log", f"[{now()}] failed {name} rc={rc}\n")
            running[gpu] = alive

        save_status(queue, running, completed, failed, note="loop")

        for gpu in GPU_IDS:
            external_count = len(external.get(gpu, []))
            used_slots = external_count + len(running[gpu])
            while queue and used_slots < SLOTS_PER_GPU:
                cand = queue.pop(0)
                job = launch(cand, gpu)
                running[gpu].append(job)
                used_slots += 1
                append_text(LOG_DIR / "auto_tune_events.log", f"[{now()}] launched {cand['name']} gpu={gpu} pid={job['proc'].pid}\n")
                save_status(queue, running, completed, failed, note=f"launched {cand['name']} on gpu {gpu}")

        time.sleep(60)

    save_status(queue, running, completed, failed, note="all done")
    append_text(LOG_DIR / "auto_tune_events.log", f"[{now()}] auto tuner finished\n")


if __name__ == "__main__":
    main()

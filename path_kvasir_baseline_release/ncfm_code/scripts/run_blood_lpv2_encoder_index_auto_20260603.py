import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path

import yaml


DATASET = "bloodmnist"
IPC = 10
NCLASS = 8
NCH = 3


FIRST_STAGE_GROUPS = {
    "LPv2_T512_lam03_g4_b2_p0": {
        "lambda_local_patch_ncfd": 0.3,
        "local_patch_grid": 4,
        "local_patch_encoder_blocks": 2,
        "local_patch_encoder_source": "premodel_trained",
        "local_patch_premodel_index": 0,
    },
    "LPv2_T512_lam03_g4_b2_p1": {
        "lambda_local_patch_ncfd": 0.3,
        "local_patch_grid": 4,
        "local_patch_encoder_blocks": 2,
        "local_patch_encoder_source": "premodel_trained",
        "local_patch_premodel_index": 1,
    },
    "LPv2_T512_lam03_g4_b2_p2": {
        "lambda_local_patch_ncfd": 0.3,
        "local_patch_grid": 4,
        "local_patch_encoder_blocks": 2,
        "local_patch_encoder_source": "premodel_trained",
        "local_patch_premodel_index": 2,
    },
    "LPv2_T512_lam03_g4_b2_p3": {
        "lambda_local_patch_ncfd": 0.3,
        "local_patch_grid": 4,
        "local_patch_encoder_blocks": 2,
        "local_patch_encoder_source": "premodel_trained",
        "local_patch_premodel_index": 3,
    },
}


def ensemble_group(lambda_value):
    return {
        "lambda_local_patch_ncfd": lambda_value,
        "local_patch_grid": 4,
        "local_patch_encoder_blocks": 2,
        "local_patch_encoder_source": "ensemble_trained",
        "local_patch_premodel_indices": [0, 1, 2, 3],
        "local_patch_ensemble_aggregate": "mean",
    }


def run_command(command, cwd, stdout_path, stderr_path, env=None):
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open(
        "w", encoding="utf-8"
    ) as err:
        proc = subprocess.run(
            command, cwd=cwd, stdout=out, stderr=err, text=True, env=env
        )
    return proc.returncode


def torchrun_command(script, config_path, gpu, ipc=None, extra=None):
    command = [
        sys.executable,
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


def ddp_env(master_port):
    env = os.environ.copy()
    env.update(
        {
            "RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(master_port),
            "USE_LIBUV": "0",
        }
    )
    return env


def stable_port(base, group_name, stage_offset=0):
    return int(base) + int(stage_offset) + (sum(ord(ch) for ch in group_name) % 1000)


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_config(path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def latest_distilled_data(search_root, dataset, ipc, start_time=None):
    root = Path(search_root)
    candidates = []
    if root.exists():
        for path in root.glob(f"**/{dataset}/ipc{ipc}/**/distilled_data/data_*.pt"):
            if path.name == "data_init.pt":
                continue
            if start_time is None or path.stat().st_mtime >= start_time:
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


def metric_acc(metrics):
    if metrics.get("acc_percent") is not None:
        return float(metrics["acc_percent"])
    if metrics.get("acc") is not None:
        value = float(metrics["acc"])
        return value * 100.0 if value <= 1.0 else value
    return -1.0


def copy_assets(args):
    data_dst = args.exp_root / "data" / "medmnist"
    pretrain_dst = args.exp_root / "checkpoints" / "pretrain" / DATASET
    data_dst.mkdir(parents=True, exist_ok=True)
    pretrain_dst.mkdir(parents=True, exist_ok=True)

    data_src = Path(args.data_source)
    pretrain_src = Path(args.pretrain_source)
    data_file = data_dst / "bloodmnist.npz"
    if not data_file.exists():
        shutil.copy2(data_src / "bloodmnist.npz", data_file)

    for path in pretrain_src.glob("premodel*_*.*"):
        dst = pretrain_dst / path.name
        if not dst.exists():
            shutil.copy2(path, dst)


def make_config(args, group_name, group):
    rdzv_dir = args.exp_root / "rdzv"
    rdzv_dir.mkdir(parents=True, exist_ok=True)
    store_path = (rdzv_dir / f"{DATASET}_{group_name}_{int(time.time() * 1000000)}.store")
    init_method = "file:///" + str(store_path).replace("\\", "/") + "?rank=0&world_size=1"
    return {
        "distibution_train": {
            "backend": "gloo",
            "init_method": init_method,
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
            "save_dir": str(args.exp_root / "results" / "condense" / group_name),
            "pretrain_dir": str(args.exp_root / "checkpoints" / "pretrain"),
        },
        "condense": {
            "ipc": IPC,
            "num_premodel": args.model_num,
            "niter": args.niter,
            "iter_calib": 0,
            "calib_weight": 1,
            "sampling_net": True,
            "num_freqs": 512,
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
            "use_local_patch_sampling_net": False,
            **group,
        },
    }


def status_path(args):
    return args.exp_root / "RUN_STATUS_M09_ENCODER_INDEX.json"


def write_status(args, payload):
    payload = {"updated_at": datetime.now().isoformat(timespec="seconds"), **payload}
    write_text(status_path(args), json.dumps(payload, indent=2))


def save_summary(args, metrics):
    report_dir = args.exp_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(metrics, key=lambda x: x.get("group", ""))
    write_text(
        report_dir / "bloodmnist_lpv2_encoder_index_auto_seed0.json",
        json.dumps(ordered, indent=2),
    )
    fields = [
        "dataset",
        "seed",
        "group",
        "acc_percent",
        "auc_macro_ovr",
        "macro_f1",
        "balanced_acc",
        "global_num_freqs",
        "lambda_local_patch_ncfd",
        "local_patch_grid",
        "local_patch_encoder_blocks",
        "local_patch_encoder_source",
        "local_patch_premodel_index",
        "local_patch_premodel_indices",
        "local_patch_ensemble_aggregate",
        "condensed_path",
        "checkpoint_path",
    ]
    with (report_dir / "bloodmnist_lpv2_encoder_index_auto_seed0.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)
    rows = [
        "| Group | ACC | AUC | Macro-F1 | BACC | lambda | source | index/indices |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in ordered:
        index_value = item.get("local_patch_premodel_index")
        if index_value is None:
            index_value = item.get("local_patch_premodel_indices")
        rows.append(
            f"| {item.get('group')} | {item.get('acc_percent')} | "
            f"{item.get('auc_macro_ovr')} | {item.get('macro_f1')} | "
            f"{item.get('balanced_acc')} | {item.get('lambda_local_patch_ncfd')} | "
            f"{item.get('local_patch_encoder_source')} | {index_value} |"
        )
    write_text(report_dir / "bloodmnist_lpv2_encoder_index_auto_seed0.md", "\n".join(rows) + "\n")


def run_one(args, repo_dir, group_name, group, gpu):
    config_path = args.exp_root / "configs" / DATASET / f"ipc{IPC}_{group_name}.yaml"
    run_dir = args.exp_root / "runs" / DATASET / f"ipc{IPC}" / group_name
    group_save_root = args.exp_root / "results" / "condense" / group_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_config(config_path, make_config(args, group_name, group))

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
        start_time = time.time()
        condense_cmd = torchrun_command(
            "condense/condense_script.py",
            config_path,
            gpu,
            ipc=IPC,
            extra=["--run_mode", "Condense", "--init", "mix"],
        )
        write_text(run_dir / "condense_command.txt", " ".join(map(str, condense_cmd)) + "\n")
        rc = run_command(
            condense_cmd,
            repo_dir,
            run_dir / "condense_stdout.log",
            run_dir / "condense_stderr.log",
            env=ddp_env(stable_port(args.master_port_base, group_name, 0)),
        )
        if rc != 0:
            raise RuntimeError(f"Condense failed for {group_name}; see {run_dir}")
        condensed_path = latest_distilled_data(group_save_root, DATASET, IPC, start_time)
        if condensed_path is None:
            condensed_path = latest_distilled_data(group_save_root, DATASET, IPC)

    if condensed_path is None:
        raise RuntimeError(f"No distilled data found for {group_name}")
    write_text(condensed_record, str(condensed_path) + "\n")

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
        gpu,
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
    write_text(run_dir / "eval_command.txt", " ".join(map(str, eval_cmd)) + "\n")
    rc = run_command(
        eval_cmd,
        repo_dir,
        run_dir / "eval_stdout.log",
        run_dir / "eval_stderr.log",
        env=ddp_env(stable_port(args.master_port_base, group_name, 2000)),
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
        "global_num_freqs": 512,
        "local_patch_num_freqs": 512,
        "batch_real": args.batch_real,
        "batch_size": args.batch_size,
        "condensed_path": str(condensed_path),
        "checkpoint_path": str(checkpoint_path),
        **group,
        **best,
    }
    write_text(metrics_path, json.dumps(metrics, indent=2))
    return metrics


def run_parallel(args, repo_dir, groups):
    queue = list(groups.items())
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    results = []
    failures = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        running = {}
        while queue or running:
            while queue and len(running) < len(gpus):
                group_name, group = queue.pop(0)
                gpu = gpus[len(running) % len(gpus)]
                write_status(
                    args,
                    {
                        "stage": "running",
                        "launching": group_name,
                        "gpu": gpu,
                        "completed": [x.get("group") for x in results],
                        "failures": failures,
                    },
                )
                future = executor.submit(run_one, args, repo_dir, group_name, group, gpu)
                running[future] = (group_name, gpu)
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                group_name, gpu = running.pop(future)
                try:
                    metrics = future.result()
                    results.append(metrics)
                    save_summary(args, results)
                    write_status(
                        args,
                        {
                            "stage": "completed_group",
                            "group": group_name,
                            "gpu": gpu,
                            "completed": [x.get("group") for x in results],
                            "failures": failures,
                        },
                    )
                except Exception as exc:
                    failures.append({"group": group_name, "gpu": gpu, "error": str(exc)})
                    save_summary(args, results)
                    write_status(
                        args,
                        {
                            "stage": "failed_group",
                            "group": group_name,
                            "gpu": gpu,
                            "completed": [x.get("group") for x in results],
                            "failures": failures,
                        },
                    )
                    if not args.continue_on_error:
                        raise
    return results, failures


def choose_winner(metrics):
    if not metrics:
        raise RuntimeError("No completed metrics are available for winner selection.")
    return max(metrics, key=lambda item: (metric_acc(item), float(item.get("macro_f1") or -1.0)))


def main():
    parser = argparse.ArgumentParser(description="BloodMNIST M09 LPv2 encoder-index automation.")
    parser.add_argument("--exp_root", type=Path, required=True)
    parser.add_argument("--data_source", required=True)
    parser.add_argument("--pretrain_source", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--batch_real", type=int, default=8192)
    parser.add_argument("--model_num", type=int, default=20)
    parser.add_argument("--pretrain_epochs", type=int, default=60)
    parser.add_argument("--eval_epochs", type=int, default=2000)
    parser.add_argument("--epoch_eval_interval", type=int, default=100)
    parser.add_argument("--niter", type=int, default=20000)
    parser.add_argument("--master_port_base", type=int, default=29500)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument(
        "--config_only",
        action="store_true",
        help="Only materialize the first-stage configs and exit without launching condense/eval.",
    )
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parents[1]
    args.exp_root.mkdir(parents=True, exist_ok=True)
    copy_assets(args)

    if args.config_only:
        config_paths = []
        for group_name, group in FIRST_STAGE_GROUPS.items():
            config_path = args.exp_root / "configs" / DATASET / f"ipc{IPC}_{group_name}.yaml"
            write_config(config_path, make_config(args, group_name, group))
            config_paths.append(str(config_path))
        write_status(
            args,
            {
                "stage": "config_only_done",
                "groups": list(FIRST_STAGE_GROUPS),
                "configs": config_paths,
            },
        )
        return

    write_status(args, {"stage": "first_stage_start", "groups": list(FIRST_STAGE_GROUPS)})
    first_metrics, failures = run_parallel(args, repo_dir, FIRST_STAGE_GROUPS)
    if failures:
        write_status(args, {"stage": "first_stage_failed", "failures": failures})
        if not args.continue_on_error:
            raise RuntimeError(f"First stage has failures: {failures}")

    winner = choose_winner(first_metrics)
    winner_idx = winner.get("local_patch_premodel_index")
    second_groups = {
        f"LPv2_T512_lam08_g4_b2_p{winner_idx}": {
            "lambda_local_patch_ncfd": 0.8,
            "local_patch_grid": 4,
            "local_patch_encoder_blocks": 2,
            "local_patch_encoder_source": "premodel_trained",
            "local_patch_premodel_index": winner_idx,
        },
        "LPv2_T512_lam03_g4_b2_ens0123_mean": ensemble_group(0.3),
    }
    write_status(
        args,
        {
            "stage": "second_stage_start",
            "winner_first_stage": winner.get("group"),
            "winner_acc": metric_acc(winner),
            "groups": list(second_groups),
        },
    )
    second_metrics, second_failures = run_parallel(args, repo_dir, second_groups)
    all_metrics = first_metrics + second_metrics
    all_failures = failures + second_failures
    save_summary(args, all_metrics)

    lam08 = next(
        (m for m in second_metrics if m.get("group") == f"LPv2_T512_lam08_g4_b2_p{winner_idx}"),
        None,
    )
    if lam08 is not None and metric_acc(lam08) > metric_acc(winner):
        third_groups = {"LPv2_T512_lam08_g4_b2_ens0123_mean": ensemble_group(0.8)}
        write_status(
            args,
            {
                "stage": "third_stage_start",
                "reason": "lam08_winner_beats_lam03_index_winner",
                "groups": list(third_groups),
            },
        )
        third_metrics, third_failures = run_parallel(args, repo_dir, third_groups)
        all_metrics += third_metrics
        all_failures += third_failures
        save_summary(args, all_metrics)

    final_winner = choose_winner(all_metrics)
    write_status(
        args,
        {
            "stage": "done",
            "winner": final_winner.get("group"),
            "winner_acc": metric_acc(final_winner),
            "completed": [m.get("group") for m in all_metrics],
            "failures": all_failures,
        },
    )


if __name__ == "__main__":
    main()

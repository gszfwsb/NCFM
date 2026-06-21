import argparse
import csv
import importlib.util
import json
from datetime import datetime
from pathlib import Path


DATASET = "bloodmnist"
IPC = 10


def format_lambda_tag(value):
    return f"{int(round(value * 10)):02d}"


def default_group_name(args):
    lam_tag = format_lambda_tag(args.lambda_local_patch_ncfd)
    source_tag = (
        "interval"
        if args.local_patch_encoder_source == "model_interval_step"
        else "rand20"
    )
    return (
        f"M12_T{args.num_freqs}_lam{lam_tag}_g{args.local_patch_grid}_"
        f"b{args.local_patch_encoder_blocks}_{source_tag}"
    )


def load_base_runner():
    here = Path(__file__).resolve().parent
    path = here / "run_blood_lpv2_encoder_index_auto_20260603.py"
    spec = importlib.util.spec_from_file_location("m09_base_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_status(exp_root, payload):
    payload = {"updated_at": datetime.now().isoformat(timespec="seconds"), **payload}
    write_text(exp_root / "RUN_STATUS_M12_RAND20_SINGLE.json", json.dumps(payload, indent=2))


def save_plan(exp_root, group_name, group, args):
    report_dir = exp_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": DATASET,
        "seed": args.seed,
        "group": group_name,
        "global_num_freqs": args.num_freqs,
        "local_patch_num_freqs": args.local_patch_num_freqs,
        "local_patch_loss_scale": args.local_patch_loss_scale,
        "batch_real": args.batch_real,
        "batch_size": args.batch_size,
        "niter": args.niter,
        **group,
    }
    write_text(
        report_dir / "bloodmnist_m12_rand20_single_seed0_group_plan.json",
        json.dumps(payload, indent=2),
    )
    rows = [
        "| Group | T | lambda | grid | blocks | encoder source | model_num | batch_real | batch_size |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|",
        (
            f"| {group_name} | {args.num_freqs} | {group['lambda_local_patch_ncfd']} | "
            f"{group['local_patch_grid']} | {group['local_patch_encoder_blocks']} | "
            f"{group['local_patch_encoder_source']} | {group['local_patch_model_num']} | "
            f"{args.batch_real} | {args.batch_size} |"
        ),
    ]
    write_text(report_dir / "bloodmnist_m12_rand20_single_seed0_group_plan.md", "\n".join(rows) + "\n")


def save_summary(exp_root, metrics):
    report_dir = exp_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "bloodmnist_m12_rand20_single_seed0.json"
    csv_path = report_dir / "bloodmnist_m12_rand20_single_seed0.csv"
    md_path = report_dir / "bloodmnist_m12_rand20_single_seed0.md"
    write_text(json_path, json.dumps([metrics], indent=2))
    fields = [
        "dataset",
        "seed",
        "group",
        "acc_percent",
        "auc_macro_ovr",
        "macro_f1",
        "balanced_acc",
        "global_num_freqs",
        "local_patch_num_freqs",
        "lambda_local_patch_ncfd",
        "local_patch_grid",
        "local_patch_encoder_blocks",
        "local_patch_encoder_source",
        "local_patch_model_num",
        "local_patch_loss_scale",
        "batch_real",
        "batch_size",
        "condensed_path",
        "checkpoint_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(metrics)
    rows = [
        "| Group | ACC | AUC | Macro-F1 | BACC | T | lambda | grid | blocks | source |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        (
            f"| {metrics.get('group')} | {metrics.get('acc_percent')} | "
            f"{metrics.get('auc_macro_ovr')} | {metrics.get('macro_f1')} | "
            f"{metrics.get('balanced_acc')} | {metrics.get('global_num_freqs')} | "
            f"{metrics.get('lambda_local_patch_ncfd')} | {metrics.get('local_patch_grid')} | "
            f"{metrics.get('local_patch_encoder_blocks')} | {metrics.get('local_patch_encoder_source')} |"
        ),
    ]
    write_text(md_path, "\n".join(rows) + "\n")


def patch_base_runner(base):
    old_make_config = base.make_config

    def make_config(args, group_name, group):
        cfg = old_make_config(args, group_name, group)
        cfg["condense"]["num_freqs"] = args.num_freqs
        cfg["condense"]["local_patch_num_freqs"] = args.local_patch_num_freqs
        cfg["condense"]["local_patch_loss_scale"] = args.local_patch_loss_scale
        cfg["condense"]["local_patch_feature_dim"] = 0
        cfg["condense"]["local_patch_model_num"] = args.model_num
        cfg["condense"]["local_patch_encoder_seed"] = args.seed
        return cfg

    base.make_config = make_config


def main():
    parser = argparse.ArgumentParser(
        description="Run one BloodMNIST M12 random-20-step local patch configuration."
    )
    parser.add_argument("--exp_root", type=Path, required=True)
    parser.add_argument("--data_source", type=Path, required=True)
    parser.add_argument("--pretrain_source", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch_real", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--model_num", type=int, default=20)
    parser.add_argument("--pretrain_epochs", type=int, default=60)
    parser.add_argument("--eval_epochs", type=int, default=2000)
    parser.add_argument("--epoch_eval_interval", type=int, default=100)
    parser.add_argument("--niter", type=int, default=20000)
    parser.add_argument("--master_port_base", type=int, default=38200)
    parser.add_argument("--num_freqs", type=int, default=1024)
    parser.add_argument("--local_patch_num_freqs", type=int, default=512)
    parser.add_argument("--local_patch_loss_scale", type=float, default=300.0)
    parser.add_argument("--lambda_local_patch_ncfd", type=float, default=0.3)
    parser.add_argument("--local_patch_grid", type=int, default=4)
    parser.add_argument("--local_patch_encoder_blocks", type=int, default=2)
    parser.add_argument(
        "--local_patch_encoder_source",
        default="random_trained_step",
        choices=["random_trained_step", "model_interval_step"],
    )
    parser.add_argument("--group_name", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.exp_root.mkdir(parents=True, exist_ok=True)
    base = load_base_runner()
    patch_base_runner(base)
    repo_dir = Path(__file__).resolve().parents[1]
    group_name = args.group_name or default_group_name(args)

    group = {
        "lambda_local_patch_ncfd": args.lambda_local_patch_ncfd,
        "local_patch_grid": args.local_patch_grid,
        "local_patch_encoder_blocks": args.local_patch_encoder_blocks,
        "local_patch_encoder_source": args.local_patch_encoder_source,
        "local_patch_model_num": args.model_num,
        "local_patch_encoder_seed": args.seed,
        "local_patch_encoder_frozen": True,
        "use_local_patch_sampling_net": False,
    }
    save_plan(args.exp_root, group_name, group, args)
    write_status(
        args.exp_root,
        {
            "stage": "copy_assets",
            "group": group_name,
            "gpu": args.gpu,
            "exp_root": str(args.exp_root),
        },
    )
    base.copy_assets(args)
    write_status(
        args.exp_root,
        {
            "stage": "running_condense_eval",
            "group": group_name,
            "gpu": args.gpu,
            "exp_root": str(args.exp_root),
        },
    )
    metrics = base.run_one(args, repo_dir, group_name, group, args.gpu)
    metrics.update(
        {
            "dataset": DATASET,
            "seed": args.seed,
            "group": group_name,
            "global_num_freqs": args.num_freqs,
            "local_patch_num_freqs": args.local_patch_num_freqs,
            "local_patch_loss_scale": args.local_patch_loss_scale,
            "batch_real": args.batch_real,
            "batch_size": args.batch_size,
            **group,
        }
    )
    metrics_path = (
        args.exp_root
        / "runs"
        / DATASET
        / f"ipc{IPC}"
        / group_name
        / "metrics.json"
    )
    write_text(metrics_path, json.dumps(metrics, indent=2))
    save_summary(args.exp_root, metrics)
    write_status(
        args.exp_root,
        {
            "stage": "completed",
            "group": group_name,
            "gpu": args.gpu,
            "metrics": str(metrics_path),
        },
    )


if __name__ == "__main__":
    main()

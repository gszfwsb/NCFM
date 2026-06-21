import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path


IPC = 10
DATASET = "bloodmnist"


def local_patch_group(lam, local_t):
    return {
        "sampling_net": True,
        "num_freqs": 512,
        "iter_calib": 0,
        "calib_weight": 1,
        "use_local_patch_feature_ncfd": True,
        "local_patch_grid": 4,
        "lambda_local_patch_ncfd": lam,
        "local_patch_feature_dim": 128,
        "local_patch_encoder_blocks": 1,
        "local_patch_num_freqs": local_t,
        "local_patch_encoder_source": "premodel_trained",
        "local_patch_premodel_index": 0,
        "local_patch_encoder_frozen": True,
        "use_local_patch_sampling_net": False,
        "dam_enabled": False,
        "use_ssim_regularization": False,
    }


GROUPS = {
    "LPv2_T512_lam02_g4_lf256_p0_b1": local_patch_group(0.2, 256),
    "LPv2_T512_lam02_g4_lf512_p0_b1": local_patch_group(0.2, 512),
    "LPv2_T512_lam03_g4_lf256_p0_b1": local_patch_group(0.3, 256),
    "LPv2_T512_lam03_g4_lf512_p0_b1": local_patch_group(0.3, 512),
    "LPv2_T512_lam05_g4_lf256_p0_b1": local_patch_group(0.5, 256),
    "LPv2_T512_lam05_g4_lf512_p0_b1": local_patch_group(0.5, 512),
}

FAMILY = {name: "local_patch_v2_current_improvement" for name in GROUPS}

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


def load_pipeline():
    here = Path(__file__).resolve().parent
    path = here / "run_medmnist_formal_pipeline.py"
    spec = importlib.util.spec_from_file_location("formal_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    try:
        os.symlink(target_path, link_path, target_is_directory=True)
    except FileExistsError:
        return


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


def save_group_plan(exp_root):
    report_dir = exp_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "bloodmnist_lpv2_current_improvement_seed0_group_plan.csv"
    md_path = report_dir / "bloodmnist_lpv2_current_improvement_seed0_group_plan.md"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "family", "config"])
        writer.writeheader()
        for group, config in GROUPS.items():
            writer.writerow(
                {
                    "group": group,
                    "family": FAMILY[group],
                    "config": json.dumps(config, sort_keys=True),
                }
            )
    rows = ["| Group | Family | Key Config |", "|---|---|---|"]
    for group, config in GROUPS.items():
        rows.append(
            f"| {group} | {FAMILY[group]} | `{json.dumps(config, sort_keys=True)}` |"
        )
    md_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def save_status(exp_root, selected, completed, failed):
    lines = [
        "# BloodMNIST LPv2 Current Baseline Improvement Seed0 Status",
        "",
        f"- Planned in this worker: {len(selected)}",
        f"- Completed: {len(completed)}",
        f"- Failed: {len(failed)}",
        "",
        "## Completed",
        "",
    ]
    for group in completed:
        lines.append(f"- {group}")
    lines.extend(["", "## Failed", ""])
    for group, error in failed:
        lines.append(f"- {group}: `{error}`")
    (exp_root / "RUN_STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Run focused BloodMNIST LPv2 improvements over current T512 baseline."
    )
    parser.add_argument("--exp_root", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--groups", default="all")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--batch_real", type=int, default=2330)
    parser.add_argument("--model_num", type=int, default=20)
    parser.add_argument("--pretrain_epochs", type=int, default=60)
    parser.add_argument("--eval_epochs", type=int, default=2000)
    parser.add_argument("--epoch_eval_interval", type=int, default=100)
    parser.add_argument("--niter", type=int, default=20000)
    parser.add_argument("--ipc", type=int, default=IPC)
    parser.add_argument("--cam_samples", type=int, default=100)
    parser.add_argument(
        "--summary_prefix", default="bloodmnist_lpv2_current_improvement_seed0"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.groups == "all":
        selected = list(GROUPS)
    else:
        selected = [item.strip() for item in args.groups.split(",") if item.strip()]
        unknown = sorted(set(selected) - set(GROUPS))
        if unknown:
            raise ValueError(f"Unknown groups: {unknown}")

    args.exp_root.mkdir(parents=True, exist_ok=True)
    ensure_assets(args.exp_root)
    pipeline = load_pipeline()
    patch_make_config(pipeline)
    pipeline.GROUPS = {
        "B_NCFM_T512": {
            "sampling_net": True,
            "num_freqs": 512,
            "iter_calib": 0,
            "calib_weight": 1,
            "use_local_patch_feature_ncfd": False,
            "dam_enabled": False,
            "use_ssim_regularization": False,
        },
        **GROUPS,
    }
    pipeline.ensure_headers(args.exp_root)
    save_group_plan(args.exp_root)
    repo_dir = Path(__file__).resolve().parents[1]

    all_metrics = []
    completed = []
    failed = []
    for group_name in selected:
        run_dir = args.exp_root / "runs" / DATASET / f"ipc{IPC}" / group_name
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists() and not args.force:
            print(f"SKIP existing {group_name}: {metrics_path}", flush=True)
            completed.append(group_name)
            continue
        print(f"RUN dataset={DATASET} group={group_name} gpu={args.gpu}", flush=True)
        try:
            run_fn = getattr(
                pipeline,
                "run_condense_eval",
                getattr(pipeline, "run_condense_eval_cam"),
            )
            metrics = run_fn(args, repo_dir, DATASET, IPC, group_name)
            metrics["family"] = FAMILY[group_name]
            metrics["seed"] = 0
            all_metrics.append(metrics)
            completed.append(group_name)
            pipeline.save_summary(
                args.exp_root, all_metrics, f"{args.summary_prefix}_gpu{args.gpu}"
            )
        except Exception as exc:
            failed.append((group_name, repr(exc)))
            save_status(args.exp_root, selected, completed, failed)
            raise
        save_status(args.exp_root, selected, completed, failed)
    if all_metrics:
        pipeline.save_summary(
            args.exp_root, all_metrics, f"{args.summary_prefix}_gpu{args.gpu}"
        )


if __name__ == "__main__":
    main()

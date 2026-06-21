import argparse
import csv
import importlib.util
import json
from pathlib import Path


IPC = 10
DATASET = "bloodmnist"


def local_patch_group(lam, grid):
    return {
        "sampling_net": True,
        "num_freqs": 512,
        "iter_calib": 0,
        "calib_weight": 1,
        "use_local_patch_feature_ncfd": True,
        "local_patch_grid": grid,
        "lambda_local_patch_ncfd": lam,
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


GROUPS = {
    "LPv2_lam50_g4_lf512_p0_b1": local_patch_group(0.5, 4),
    "LPv2_lam80_g4_lf512_p0_b1": local_patch_group(0.80, 4),
    "LPv2_lam50_g2_lf512_p0_b1": local_patch_group(0.50, 2),
    "LPv2_lam50_g7_lf512_p0_b1": local_patch_group(0.50, 7),
}

FAMILY = {name: "local_patch_v2_followup" for name in GROUPS}

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


def save_group_plan(exp_root):
    report_dir = exp_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "bloodmnist_lpv2_lam_grid_followup_seed0_group_plan.csv"
    md_path = report_dir / "bloodmnist_lpv2_lam_grid_followup_seed0_group_plan.md"
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
        "# BloodMNIST LPv2 Lambda/Grid Follow-up Seed0 Status",
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
        description="Run BloodMNIST LPv2 lambda/grid follow-up seed0."
    )
    parser.add_argument("--exp_root", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--groups", default="all")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--batch_real", type=int, default=4096)
    parser.add_argument("--model_num", type=int, default=20)
    parser.add_argument("--pretrain_epochs", type=int, default=60)
    parser.add_argument("--eval_epochs", type=int, default=2000)
    parser.add_argument("--epoch_eval_interval", type=int, default=100)
    parser.add_argument("--niter", type=int, default=20000)
    parser.add_argument("--ipc", type=int, default=IPC)
    parser.add_argument("--cam_samples", type=int, default=100)
    parser.add_argument(
        "--summary_prefix", default="bloodmnist_lpv2_lam_grid_followup_seed0"
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
            metrics = pipeline.run_condense_eval_cam(
                args, repo_dir, DATASET, IPC, group_name
            )
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

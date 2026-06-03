#!/usr/bin/env python3
"""Run evaluation repeats as parallel single-GPU jobs."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_gpus(raw: str) -> list[str]:
    gpus = [item.strip() for item in raw.split(",") if item.strip()]
    if not gpus:
        raise ValueError("At least one GPU id is required")
    return gpus


def sanitize_tag(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)


def run_one(
    args: argparse.Namespace,
    repeat_idx: int,
    gpu: str,
    port: int,
) -> subprocess.Popen:
    tag = f"{sanitize_tag(args.tag)}_r{repeat_idx:02d}"
    cmd = [
        str(ROOT / ".venv" / "bin" / "python"),
        str(ROOT / "tools" / "ncfm_eval_tuner.py"),
        "--tag",
        tag,
        "--config",
        args.config,
        "--load-path",
        args.load_path,
        "--ipc",
        str(args.ipc),
        "--val-repeat",
        "1",
        "--gpu",
        gpu,
        "--nproc",
        "1",
        "--port",
        str(port),
        "--set",
        f"cuda_graph={str(args.cuda_graph).lower()}",
    ]
    for item in args.set:
        cmd.extend(["--set", item])
    cmd.extend(["--set", f"train.seed={args.seed_base + repeat_idx}"])
    return subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--load-path", required=True)
    parser.add_argument("--ipc", required=True, type=int)
    parser.add_argument("--val-repeat", default=10, type=int)
    parser.add_argument("--gpu", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--port-base", default=47000, type=int)
    parser.add_argument(
        "--seed-base",
        default=0,
        type=int,
        help="Base seed for per-repeat train.seed overrides.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override YAML value passed through to ncfm_eval_tuner.py",
    )
    parser.add_argument(
        "--cuda-graph",
        dest="cuda_graph",
        action="store_true",
        default=True,
        help="Enable CUDA graph for evaluation training forwards.",
    )
    parser.add_argument(
        "--no-cuda-graph",
        dest="cuda_graph",
        action="store_false",
        help="Disable CUDA graph for paired timing comparisons.",
    )
    args = parser.parse_args()

    gpus = parse_gpus(args.gpu)
    pending = list(range(args.val_repeat))
    active: list[tuple[int, str, subprocess.Popen]] = []
    summaries = []
    failed = False

    while pending or active:
        while pending and len(active) < len(gpus):
            repeat_idx = pending.pop(0)
            gpu = gpus[repeat_idx % len(gpus)]
            proc = run_one(args, repeat_idx, gpu, args.port_base + repeat_idx)
            active.append((repeat_idx, gpu, proc))
            print(f"Started repeat {repeat_idx + 1}/{args.val_repeat} on GPU {gpu}", flush=True)

        still_active = []
        for repeat_idx, gpu, proc in active:
            if proc.poll() is None:
                still_active.append((repeat_idx, gpu, proc))
                continue

            output = proc.stdout.read() if proc.stdout is not None else ""
            print(output, end="" if output.endswith("\n") else "\n", flush=True)
            try:
                summary = json.loads(output.strip().splitlines()[-1])
            except Exception:
                summary = {
                    "tag": f"{sanitize_tag(args.tag)}_r{repeat_idx:02d}",
                    "returncode": proc.returncode,
                    "stdout": output,
                }
            summary["repeat_index"] = repeat_idx
            summary["gpu"] = gpu
            summaries.append(summary)
            if proc.returncode != 0 or summary.get("returncode") not in (0, None):
                failed = True
        active = still_active

    summaries.sort(key=lambda item: item["repeat_index"])
    means = [item.get("mean") for item in summaries]
    valid_means = [float(value) for value in means if value is not None]
    aggregate = {
        "tag": sanitize_tag(args.tag),
        "returncode": 1 if failed or len(valid_means) != args.val_repeat else 0,
        "val_repeat": args.val_repeat,
        "mean": sum(valid_means) / len(valid_means) if valid_means else None,
        "std": (
            math.sqrt(
                sum((value - sum(valid_means) / len(valid_means)) ** 2 for value in valid_means)
                / len(valid_means)
            )
            if valid_means
            else None
        ),
        "all_result": [f"{value:.3f}" for value in valid_means],
        "runs": summaries,
    }
    print(json.dumps(aggregate, ensure_ascii=False))
    return int(aggregate["returncode"])


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Run NCFM condense variants with isolated configs and logs."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "results" / "tuning_configs"
STDOUT_DIR = ROOT / "results" / "tuning_stdout"


def parse_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None
    try:
        if any(marker in raw for marker in [".", "e", "E"]):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    node = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            raise KeyError(f"Missing nested config section: {dotted_key}")
        node = node[part]
    node[parts[-1]] = value


def latest_print_log(save_parent: Path) -> Path | None:
    logs = list(save_parent.rglob("print.log"))
    if not logs:
        return None
    return max(logs, key=lambda path: path.stat().st_mtime)


def latest_distilled_data(save_parent: Path) -> Path | None:
    data_files = list(save_parent.rglob("distilled_data/data_*.pt"))
    if not data_files:
        return None

    def key(path: Path) -> tuple[int, float]:
        match = re.search(r"data_(\d+|init)\.pt$", path.name)
        if not match:
            return (-1, path.stat().st_mtime)
        value = match.group(1)
        return ((-1 if value == "init" else int(value)), path.stat().st_mtime)

    return max(data_files, key=key)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ipc", required=True, type=int)
    parser.add_argument("--gpu", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--nproc", default=8, type=int)
    parser.add_argument("--port", default=None, type=int)
    parser.add_argument("--init", default="mix", choices=["random", "noise", "mix", "load"])
    parser.add_argument("--load-path", default=None)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override YAML value, for example condense.sampling_net=true",
    )
    args = parser.parse_args()

    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.tag)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    STDOUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(ROOT / args.config, "r") as handle:
        config = yaml.safe_load(handle)

    for assignment in args.set:
        if "=" not in assignment:
            raise ValueError(f"Expected key=value assignment: {assignment}")
        key, raw_value = assignment.split("=", 1)
        set_nested(config, key, parse_value(raw_value))

    config["distibution_train"]["workers"] = min(config["distibution_train"].get("workers", 2), 2)
    save_parent = ROOT / "results" / "tuning_condense" / tag
    config["save_path"]["save_dir"] = os.path.relpath(save_parent, ROOT / "condense")

    config_path = CONFIG_DIR / f"{tag}.yaml"
    with open(config_path, "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    stdout_path = STDOUT_DIR / f"{tag}.out"
    port = args.port if args.port is not None else 38000 + (os.getpid() % 1000)
    cmd = [
        str(ROOT / ".venv" / "bin" / "torchrun"),
        f"--nproc_per_node={args.nproc}",
        "--nnodes=1",
        f"--master_port={port}",
        "condense_script.py",
        f"--gpu={args.gpu}",
        f"--ipc={args.ipc}",
        f"--config_path={config_path}",
        f"--init={args.init}",
    ]
    if args.load_path:
        cmd.append(f"--load_path={ROOT / args.load_path}")

    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )

    started = time.time()
    with open(stdout_path, "w") as stdout:
        proc = subprocess.run(
            cmd,
            cwd=ROOT / "condense",
            env=env,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            text=True,
        )

    log_path = latest_print_log(save_parent)
    data_path = latest_distilled_data(save_parent)
    summary = {
        "tag": tag,
        "returncode": proc.returncode,
        "seconds": round(time.time() - started, 1),
        "config": str(config_path),
        "stdout": str(stdout_path),
        "log": str(log_path) if log_path else None,
        "latest_data": str(data_path) if data_path else None,
    }
    print(json.dumps(summary, ensure_ascii=False))
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())

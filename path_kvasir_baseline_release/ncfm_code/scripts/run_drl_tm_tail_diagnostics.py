import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from NCFM.NCFM import CFLossFunc
from NCFM.dr_ltm import _parse_layers
from utils.ddp import load_state_dict
from utils.utils import define_model


def load_real_npz(path, max_per_class, seed):
    data = np.load(path)
    images = data["train_images"]
    labels = data["train_labels"].reshape(-1)
    rng = np.random.default_rng(seed)
    by_class = {}
    for cls in sorted(np.unique(labels).tolist()):
        idx = np.where(labels == cls)[0]
        if max_per_class > 0 and len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        img = images[idx]
        if img.ndim == 3:
            img = img[..., None]
        img = torch.tensor(img, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        by_class[int(cls)] = img
    return by_class


def load_syn_pt(path):
    data, targets = torch.load(path, map_location="cpu")
    by_class = {}
    for cls in sorted(targets.unique().tolist()):
        by_class[int(cls)] = data[targets == cls].float().cpu()
    return by_class


def parse_synthetic(values):
    parsed = []
    for value in values:
        if "=" not in value:
            raise ValueError(
                "--synthetic must use METHOD=/path/to/data_20000.pt format"
            )
        name, path = value.split("=", 1)
        parsed.append((name.strip(), Path(path).expanduser()))
    return parsed


def extract_spatial(model, images, layers):
    _, features = model.get_feature_from_layer(images, return_features=True)
    spatial = [feat for feat in features if torch.is_tensor(feat) and feat.dim() == 4]
    selected = _parse_layers(layers, len(spatial))
    return [spatial[idx] for idx in selected], selected


def token_effective_rank(tokens):
    if tokens.shape[0] < 2:
        return 1.0
    centered = tokens - tokens.mean(dim=0, keepdim=True)
    _, s, _ = torch.linalg.svd(centered.float(), full_matrices=False)
    p = s / s.sum().clamp_min(1e-8)
    entropy = -(p * p.clamp_min(1e-8).log()).sum()
    return float(torch.exp(entropy).item())


def subsample_tokens(tokens, max_tokens, seed):
    if max_tokens <= 0 or tokens.shape[0] <= max_tokens:
        return tokens
    gen = torch.Generator(device=tokens.device)
    gen.manual_seed(seed)
    idx = torch.randperm(tokens.shape[0], generator=gen, device=tokens.device)[:max_tokens]
    return tokens[idx]


def coverage_stats(real_tokens, syn_tokens, max_tokens, seed):
    real_tokens = F.normalize(real_tokens.float(), dim=1)
    syn_tokens = F.normalize(syn_tokens.float(), dim=1)
    real_sub = subsample_tokens(real_tokens, max_tokens, seed)
    syn_sub = subsample_tokens(syn_tokens, max_tokens, seed + 17)
    sim = real_sub @ syn_sub.t()
    real_to_syn = sim.max(dim=1).values.mean()
    syn_to_real = sim.max(dim=0).values.mean()
    return {
        "coverage_real_to_syn": float(real_to_syn.item()),
        "coverage_syn_to_real": float(syn_to_real.item()),
        "effective_rank_real": token_effective_rank(real_sub),
        "effective_rank_syn": token_effective_rank(syn_sub),
    }


def local_ncfd_stats(real_feat, syn_feat, cf_loss_func, args):
    real = real_feat.permute(0, 2, 3, 1).reshape(
        real_feat.shape[0], real_feat.shape[2] * real_feat.shape[3], real_feat.shape[1]
    )
    syn = syn_feat.permute(0, 2, 3, 1).reshape(
        syn_feat.shape[0], syn_feat.shape[2] * syn_feat.shape[3], syn_feat.shape[1]
    )
    real = F.normalize(real, dim=2)
    syn = F.normalize(syn, dim=2)
    t = torch.randn((args.num_freqs, syn.shape[2]), device=syn.device)
    scores = []
    for pos in range(real.shape[1]):
        scores.append(cf_loss_func(real[:, pos, :], syn[:, pos, :], t, args))
    scores = torch.stack(scores).detach().float()
    k = max(1, int(np.ceil(args.alpha * scores.numel())))
    top = torch.topk(scores, k=k, largest=True).values
    mean = scores.mean()
    max_value = scores.max()
    p90 = torch.quantile(scores, 0.90)
    std = scores.std(unbiased=False)
    return {
        "mean_d": float(mean.item()),
        "max_d": float(max_value.item()),
        "p90_d": float(p90.item()),
        "cvar_d": float(top.mean().item()),
        "tail_ratio": float((max_value / mean.clamp_min(1e-8)).item()),
        "cv": float((std / mean.clamp_min(1e-8)).item()),
        "tail_k": int(k),
        "num_positions": int(scores.numel()),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_npz", type=Path, required=True)
    parser.add_argument("--probe_checkpoint", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--synthetic", action="append", required=True)
    parser.add_argument("--dataset", default="pathmnist")
    parser.add_argument("--nclass", type=int, default=9)
    parser.add_argument("--nch", type=int, default=3)
    parser.add_argument("--size", type=int, default=28)
    parser.add_argument("--layers", default="[1]")
    parser.add_argument("--num_freqs", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--max_real_per_class", type=int, default=256)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    real_by_class = load_real_npz(args.real_npz, args.max_real_per_class, args.seed)
    synthetic_sets = [(name, load_syn_pt(path)) for name, path in parse_synthetic(args.synthetic)]

    model = define_model(
        args.dataset,
        "instance",
        "convnet",
        args.nch,
        3,
        1.0,
        args.nclass,
        logger=None,
        size=args.size,
    ).to(device)
    load_state_dict(args.probe_checkpoint, model)
    model.eval()
    cf_loss = CFLossFunc(alpha_for_loss=0.5, beta_for_loss=0.5)
    cf_args = SimpleNamespace(num_freqs=args.num_freqs)

    discrepancy_rows = []
    coverage_rows = []
    with torch.no_grad():
        for method, syn_by_class in synthetic_sets:
            for cls, real_images_cpu in real_by_class.items():
                if cls not in syn_by_class:
                    continue
                real_images = real_images_cpu.to(device)
                syn_images = syn_by_class[cls].to(device)
                real_features, selected_layers = extract_spatial(model, real_images, args.layers)
                syn_features, _ = extract_spatial(model, syn_images, args.layers)
                for layer_id, real_feat, syn_feat in zip(selected_layers, real_features, syn_features):
                    disc = local_ncfd_stats(real_feat, syn_feat, cf_loss, cf_args)
                    discrepancy_rows.append(
                        {
                            "method": method,
                            "class_id": cls,
                            "layer": layer_id,
                            **disc,
                        }
                    )
                    real_tokens = real_feat.permute(0, 2, 3, 1).reshape(-1, real_feat.shape[1])
                    syn_tokens = syn_feat.permute(0, 2, 3, 1).reshape(-1, syn_feat.shape[1])
                    cov = coverage_stats(real_tokens, syn_tokens, args.max_tokens, args.seed + cls)
                    cov["effective_rank_gap"] = abs(
                        cov["effective_rank_real"] - cov["effective_rank_syn"]
                    )
                    coverage_rows.append(
                        {
                            "method": method,
                            "class_id": cls,
                            "layer": layer_id,
                            **cov,
                        }
                    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "local_discrepancy_stats.csv", discrepancy_rows)
    write_csv(args.out_dir / "support_coverage_rank.csv", coverage_rows)
    (args.out_dir / "diagnostics_summary.json").write_text(
        json.dumps(
            {
                "local_discrepancy_rows": len(discrepancy_rows),
                "support_coverage_rank_rows": len(coverage_rows),
                "methods": [name for name, _ in synthetic_sets],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved diagnostics to {args.out_dir}")


if __name__ == "__main__":
    main()

import math

import torch
import torch.nn.functional as F


def _parse_layers(value, max_layers):
    if value is None:
        return [max_layers - 1]
    if isinstance(value, int):
        layers = [value]
    elif isinstance(value, (list, tuple)):
        layers = [int(v) for v in value]
    elif isinstance(value, str):
        cleaned = value.strip().strip("[]()")
        if not cleaned:
            return [max_layers - 1]
        layers = [int(v.strip()) for v in cleaned.split(",") if v.strip()]
    else:
        layers = [int(value)]
    return [idx for idx in layers if 0 <= idx < max_layers]


def extract_spatial_features(model, images, args):
    """Return selected 4D feature maps from the current NCFM feature extractor."""
    if hasattr(model, "get_feature_from_layer"):
        _, features = model.get_feature_from_layer(images, return_features=True)
    elif hasattr(model, "get_feature"):
        features = model.get_feature(images, 0, getattr(args, "depth", 1) - 1)
    else:
        raise TypeError(
            "DR-LTM requires a model exposing get_feature_from_layer(...) "
            "or get_feature(...)."
        )

    spatial = [feat for feat in features if torch.is_tensor(feat) and feat.dim() == 4]
    if not spatial:
        raise ValueError("No spatial feature maps were produced by the model.")

    layers = _parse_layers(getattr(args, "dr_ltm_layers", None), len(spatial))
    if not layers:
        raise ValueError(
            "dr_ltm_layers selected no valid layers from "
            f"{len(spatial)} spatial maps"
        )
    return [spatial[idx] for idx in layers]


def _spatial_cf_discrepancies(real_feat, syn_feat, cf_loss_func, args):
    """Compute one local NCFD value per spatial position for a single layer."""
    if real_feat.shape[1] != syn_feat.shape[1]:
        raise ValueError(
            f"Channel mismatch: real={real_feat.shape[1]} syn={syn_feat.shape[1]}"
        )
    if real_feat.shape[2:] != syn_feat.shape[2:]:
        raise ValueError(
            f"Spatial mismatch: real={tuple(real_feat.shape[2:])} "
            f"syn={tuple(syn_feat.shape[2:])}"
        )

    bsz_r, channels, height, width = real_feat.shape
    bsz_s = syn_feat.shape[0]
    local_num_freqs = int(
        getattr(args, "dr_ltm_num_freqs", min(int(args.num_freqs), 256))
    )

    real = real_feat.permute(0, 2, 3, 1).reshape(bsz_r, height * width, channels)
    syn = syn_feat.permute(0, 2, 3, 1).reshape(bsz_s, height * width, channels)
    real = F.normalize(real, dim=2)
    syn = F.normalize(syn, dim=2)

    t = torch.randn((local_num_freqs, channels), device=syn_feat.device)
    scores = []
    for pos in range(height * width):
        scores.append(cf_loss_func(real[:, pos, :], syn[:, pos, :], t, args))
    return torch.stack(scores)


def _cvar_top_tail(scores, alpha):
    alpha = float(alpha)
    if alpha <= 0 or alpha > 1:
        raise ValueError(f"dr_ltm_alpha must be in (0, 1], got {alpha}")
    k = max(1, int(math.ceil(alpha * scores.numel())))
    top_values = torch.topk(scores, k=k, largest=True).values
    return top_values.mean(), k


def _cvar_hinge(scores, alpha, eta_mode):
    alpha = float(alpha)
    if alpha <= 0 or alpha > 1:
        raise ValueError(f"dr_ltm_alpha must be in (0, 1], got {alpha}")
    detached = scores.detach()
    quantile = max(0.0, min(1.0, 1.0 - alpha))
    if eta_mode == "mean":
        eta = detached.mean()
    elif eta_mode == "median":
        eta = detached.median()
    else:
        eta = torch.quantile(detached.float(), quantile).to(scores.dtype)
    # Stop-gradient on eta keeps the selected tail threshold from becoming a
    # second optimization target; gradients still flow through scores above eta.
    eta = eta.detach()
    return eta + torch.relu(scores - eta).sum() / (alpha * scores.numel())


def _aggregate_scores(scores, args):
    mode = str(getattr(args, "dr_ltm_mode", "topk")).lower()
    alpha = float(getattr(args, "dr_ltm_alpha", 0.25))
    if mode in {"topk", "top_tail", "tail"}:
        return _cvar_top_tail(scores, alpha)[0]
    if mode in {"hinge", "cvar", "smooth"}:
        eta_mode = str(getattr(args, "dr_ltm_eta_mode", "quantile")).lower()
        return _cvar_hinge(scores, alpha, eta_mode)
    if mode in {"mean", "average", "uniform"}:
        return scores.mean()
    if mode in {"max", "worst"}:
        return scores.max()
    raise ValueError(
        "Unsupported dr_ltm_mode="
        f"{mode}. Expected topk, hinge/cvar, mean, or max."
    )


def _score_stats(scores, args):
    alpha = float(getattr(args, "dr_ltm_alpha", 0.25))
    k = max(1, int(math.ceil(alpha * scores.numel())))
    detached = scores.detach().float()
    mean = detached.mean()
    max_value = detached.max()
    std = detached.std(unbiased=False)
    p90 = torch.quantile(detached, 0.90)
    cvar = torch.topk(detached, k=k, largest=True).values.mean()
    eps = 1e-8
    return {
        "mean": float(mean.item()),
        "max": float(max_value.item()),
        "p90": float(p90.item()),
        "cvar": float(cvar.item()),
        "tail_ratio": float((max_value / mean.clamp_min(eps)).item()),
        "cv": float((std / mean.clamp_min(eps)).item()),
        "k": float(k),
        "count": float(scores.numel()),
    }


def dr_ltm_ncfd_loss(img_real, img_syn, model, cf_loss_func, args):
    """CVaR/top-tail local token NCFD over spatial feature-map positions."""
    detach_real = bool(getattr(args, "dr_ltm_detach_real", True))
    if detach_real:
        with torch.no_grad():
            real_features = extract_spatial_features(model, img_real, args)
    else:
        real_features = extract_spatial_features(model, img_real, args)
    syn_features = extract_spatial_features(model, img_syn, args)

    if detach_real:
        real_features = [feat.detach() for feat in real_features]

    loss_scale = float(getattr(args, "dr_ltm_loss_scale", 300.0))
    layer_losses = []
    stats = []

    for real_feat, syn_feat in zip(real_features, syn_features):
        scores = _spatial_cf_discrepancies(real_feat, syn_feat, cf_loss_func, args)
        layer_losses.append(_aggregate_scores(scores, args))
        stats.append(_score_stats(scores, args))

    raw_loss = torch.stack(layer_losses).mean()
    scaled_loss = loss_scale * raw_loss

    if bool(getattr(args, "dr_ltm_log_components", True)):
        denom = max(1, len(stats))
        args._dr_ltm_last_raw_loss = float(raw_loss.detach().item())
        args._dr_ltm_last_scaled_loss = float(scaled_loss.detach().item())
        args._dr_ltm_last_num_layers = len(layer_losses)
        for key in ["mean", "max", "p90", "cvar", "tail_ratio", "cv", "k", "count"]:
            setattr(
                args,
                f"_dr_ltm_last_{key}",
                sum(item[key] for item in stats) / denom,
            )

    return scaled_loss

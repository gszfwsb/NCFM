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
        cleaned = value.strip()
        if not cleaned:
            return [max_layers - 1]
        cleaned = cleaned.strip("[]()")
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
            "Discrepancy attention requires a model exposing "
            "get_feature_from_layer(...) or get_feature(...)."
        )

    spatial = [feat for feat in features if torch.is_tensor(feat) and feat.dim() == 4]
    if not spatial:
        raise ValueError("No spatial feature maps were produced by the model.")

    layers = _parse_layers(getattr(args, "discrepancy_attention_layers", None), len(spatial))
    if not layers:
        raise ValueError(
            "discrepancy_attention_layers selected no valid layers from "
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
        getattr(args, "discrepancy_attention_num_freqs", min(int(args.num_freqs), 256))
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


def _attention_from_scores(scores, args):
    mode = str(getattr(args, "discrepancy_attention_mode", "softmax")).lower()
    tau = float(getattr(args, "discrepancy_attention_tau", 1.0))
    eps = float(getattr(args, "discrepancy_attention_eps", 1e-8))
    topk = int(getattr(args, "discrepancy_attention_topk", 0))

    if mode == "softmax":
        detached = scores.detach()
        if topk > 0 and topk < detached.numel():
            keep = torch.topk(detached, k=topk, largest=True).indices
            masked = torch.full_like(detached, -float("inf"))
            masked[keep] = detached[keep]
            return torch.softmax(masked / max(tau, eps), dim=0)
        return torch.softmax(detached / max(tau, eps), dim=0)
    if mode == "uniform":
        return torch.full_like(scores, 1.0 / max(1, scores.numel()))
    if mode == "random":
        random_scores = torch.rand_like(scores)
        return torch.softmax(random_scores / max(tau, eps), dim=0)
    if mode in {"normalize", "l1"}:
        clipped = scores.detach().clamp_min(0)
        denom = clipped.sum().clamp_min(eps)
        return clipped / denom
    raise ValueError(
        "Unsupported discrepancy_attention_mode="
        f"{mode}. Expected softmax, uniform, random, normalize."
    )


def discrepancy_attention_ncfd_loss(img_real, img_syn, model, cf_loss_func, args):
    """NCFD over spatial positions, weighted by real/synthetic mismatch."""
    detach_real = bool(getattr(args, "discrepancy_attention_detach_real", True))
    if detach_real:
        with torch.no_grad():
            real_features = extract_spatial_features(model, img_real, args)
    else:
        real_features = extract_spatial_features(model, img_real, args)
    syn_features = extract_spatial_features(model, img_syn, args)

    if detach_real:
        real_features = [feat.detach() for feat in real_features]

    loss_scale = float(getattr(args, "discrepancy_attention_loss_scale", 300.0))
    layer_losses = []
    entropies = []
    max_weights = []
    mean_scores = []

    for real_feat, syn_feat in zip(real_features, syn_features):
        scores = _spatial_cf_discrepancies(real_feat, syn_feat, cf_loss_func, args)
        weights = _attention_from_scores(scores, args)
        layer_losses.append(torch.sum(weights * scores))
        entropy = -(weights * (weights.clamp_min(1e-8)).log()).sum()
        entropies.append(entropy.detach())
        max_weights.append(weights.max().detach())
        mean_scores.append(scores.detach().mean())

    raw_loss = torch.stack(layer_losses).mean()
    scaled_loss = loss_scale * raw_loss

    if bool(getattr(args, "discrepancy_attention_log_components", True)):
        args._dgsa_last_raw_loss = float(raw_loss.detach().item())
        args._dgsa_last_scaled_loss = float(scaled_loss.detach().item())
        args._dgsa_last_entropy = float(torch.stack(entropies).mean().item())
        args._dgsa_last_max_weight = float(torch.stack(max_weights).mean().item())
        args._dgsa_last_mean_score = float(torch.stack(mean_scores).mean().item())
        args._dgsa_last_num_layers = len(layer_losses)

    return scaled_loss

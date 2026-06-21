import torch
import torch.nn.functional as F

from .NCFM import match_loss


def _parse_layers(value, max_layers):
    if value is None:
        return list(range(max_layers))
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        layers = [int(item) for item in items]
    elif isinstance(value, (list, tuple)):
        layers = [int(item) for item in value]
    else:
        layers = [int(value)]
    return [idx for idx in layers if 0 <= idx < max_layers]


def extract_spatial_features(model, x, args):
    """Return spatial feature maps used for DataDAM-style attention matching."""
    if hasattr(model, "get_feature_from_layer"):
        _, features = model.get_feature_from_layer(x, return_features=True)
    elif hasattr(model, "get_feature"):
        features = model.get_feature(x, 0, getattr(args, "depth", 1) - 1)
    else:
        raise TypeError(
            "DataDAM attention matching requires a model with "
            "get_feature_from_layer(...) or get_feature(...)."
        )

    spatial_features = [feat for feat in features if torch.is_tensor(feat) and feat.dim() == 4]
    if not spatial_features:
        raise ValueError("No 4D spatial feature maps found for DataDAM attention matching")

    layers = _parse_layers(getattr(args, "dam_attention_layers", None), len(spatial_features))
    if not layers:
        raise ValueError(
            f"dam_attention_layers selected no valid layers from {len(spatial_features)} spatial maps"
        )
    return [spatial_features[idx] for idx in layers]


def spatial_attention_map(feature, p=2.0, norm="l2", eps=1e-6):
    """Convert [B, C, H, W] features to normalized spatial attention vectors."""
    attention = torch.sum(torch.abs(feature).pow(float(p)), dim=1)
    attention = attention.flatten(start_dim=1)

    if norm == "l2":
        attention = F.normalize(attention, p=2, dim=1, eps=eps)
    elif norm == "l1":
        denom = attention.abs().sum(dim=1, keepdim=True).clamp_min(eps)
        attention = attention / denom
    elif norm in {"none", None}:
        pass
    else:
        raise ValueError(f"Unsupported dam_attention_norm={norm}")
    return attention


def attention_prototype_loss(real_features, syn_features, args):
    """Match class-level mean spatial-attention prototypes layer by layer."""
    if len(real_features) != len(syn_features):
        raise ValueError(
            f"Mismatched attention feature counts: real={len(real_features)} syn={len(syn_features)}"
        )

    p = float(getattr(args, "dam_attention_p", 2.0))
    norm = getattr(args, "dam_attention_norm", "l2")
    eps = float(getattr(args, "dam_attention_eps", 1e-6))
    losses = []

    for real_feat, syn_feat in zip(real_features, syn_features):
        real_attention = spatial_attention_map(real_feat, p=p, norm=norm, eps=eps)
        syn_attention = spatial_attention_map(syn_feat, p=p, norm=norm, eps=eps)

        real_proto = real_attention.mean(dim=0)
        syn_proto = syn_attention.mean(dim=0)
        losses.append(torch.sum((real_proto - syn_proto) ** 2))

    return torch.stack(losses).mean()


def datadam_attention_loss(img_real, img_syn, model, sampling_net, args):
    """Compute DataDAM-style spatial attention matching loss for one class."""
    detach_real = bool(getattr(args, "dam_detach_real", True))
    if detach_real:
        with torch.no_grad():
            real_features = extract_spatial_features(model, img_real, args)
    else:
        real_features = extract_spatial_features(model, img_real, args)

    syn_features = extract_spatial_features(model, img_syn, args)
    if detach_real:
        real_features = [feat.detach() for feat in real_features]

    return attention_prototype_loss(real_features, syn_features, args)


def datadam_guided_match_loss(img_real, img_syn, model, sampling_net, args):
    """Hybrid NCFM + DataDAM attention loss used as compute_match_loss inner_loss_fn."""
    objective = getattr(args, "dam_objective", "ncfm_attention")
    feature_weight = float(getattr(args, "dam_feature_weight", 1.0))
    attention_weight = float(getattr(args, "dam_attention_weight", 10.0))

    feature_loss = img_syn.new_tensor(0.0)
    if objective != "attention_only" and feature_weight != 0.0:
        feature_loss = match_loss(img_real, img_syn, model, sampling_net, args)

    attention_loss = img_syn.new_tensor(0.0)
    if objective != "ncfm" and attention_weight != 0.0:
        attention_loss = datadam_attention_loss(img_real, img_syn, model, sampling_net, args)

    total_loss = feature_weight * feature_loss + attention_weight * attention_loss

    if bool(getattr(args, "dam_log_components", True)):
        args._dam_last_feature_loss = float(feature_loss.detach().item())
        args._dam_last_attention_loss = float(attention_loss.detach().item())
        args._dam_last_total_loss = float(total_loss.detach().item())

    return total_loss

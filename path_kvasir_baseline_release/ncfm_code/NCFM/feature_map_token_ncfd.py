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


def _extract_spatial_features(model, images, args):
    if hasattr(model, "get_feature_from_layer"):
        _, features = model.get_feature_from_layer(images, return_features=True)
    elif hasattr(model, "get_feature"):
        features = model.get_feature(images, 0, getattr(args, "depth", 1) - 1)
    else:
        raise TypeError(
            "Feature-map token NCFD requires get_feature_from_layer(...) "
            "or get_feature(...)."
        )

    spatial = [feat for feat in features if torch.is_tensor(feat) and feat.dim() == 4]
    if not spatial:
        raise ValueError("No 4D feature maps were produced by the model.")
    layers = _parse_layers(getattr(args, "feature_map_token_layers", None), len(spatial))
    if not layers:
        raise ValueError(
            "feature_map_token_layers selected no valid layers from "
            f"{len(spatial)} spatial maps"
        )
    return [spatial[idx] for idx in layers]


def _append_position_channels(tokens, height, width, batch_size, args):
    pos_weight = float(getattr(args, "feature_map_token_pos_weight", 0.0))
    if pos_weight <= 0:
        return tokens
    device = tokens.device
    dtype = tokens.dtype
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
        indexing="ij",
    )
    pos = torch.stack([yy, xx], dim=-1).reshape(1, height * width, 2)
    pos = pos.repeat(batch_size, 1, 1).reshape(batch_size * height * width, 2)
    return torch.cat([tokens, pos_weight * pos], dim=1)


def _tokens_from_feature(feat, args):
    if bool(getattr(args, "feature_map_token_pool", False)):
        grid = int(getattr(args, "feature_map_token_pool_grid", 0))
        if grid > 0:
            feat = F.adaptive_avg_pool2d(feat, (grid, grid))
    batch_size, channels, height, width = feat.shape
    tokens = feat.permute(0, 2, 3, 1).reshape(batch_size * height * width, channels)
    tokens = F.normalize(tokens, dim=1)
    tokens = _append_position_channels(tokens, height, width, batch_size, args)
    return tokens


def feature_map_token_ncfd_loss(img_real, img_syn, model, cf_loss_func, args):
    """NCFD over intermediate feature-map tokens from the current NCFM model."""
    detach_real = bool(getattr(args, "feature_map_token_detach_real", True))
    if detach_real:
        with torch.no_grad():
            real_features = _extract_spatial_features(model, img_real, args)
    else:
        real_features = _extract_spatial_features(model, img_real, args)
    syn_features = _extract_spatial_features(model, img_syn, args)

    if detach_real:
        real_features = [feat.detach() for feat in real_features]

    num_freqs = int(
        getattr(args, "feature_map_token_num_freqs", min(int(args.num_freqs), 256))
    )
    loss_scale = float(getattr(args, "feature_map_token_loss_scale", 300.0))
    layer_losses = []
    token_counts = []
    dims = []
    for real_feat, syn_feat in zip(real_features, syn_features):
        real_tokens = _tokens_from_feature(real_feat, args)
        syn_tokens = _tokens_from_feature(syn_feat, args)
        if real_tokens.shape[1] != syn_tokens.shape[1]:
            raise ValueError(
                "Feature token dimension mismatch: "
                f"real={real_tokens.shape[1]} syn={syn_tokens.shape[1]}"
            )
        t = torch.randn((num_freqs, syn_tokens.shape[1]), device=syn_tokens.device)
        layer_losses.append(cf_loss_func(real_tokens, syn_tokens, t, args))
        token_counts.append(float(syn_tokens.shape[0]))
        dims.append(float(syn_tokens.shape[1]))

    raw_loss = torch.stack(layer_losses).mean()
    scaled_loss = loss_scale * raw_loss

    if bool(getattr(args, "feature_map_token_log_components", True)):
        args._fmt_last_raw_loss = float(raw_loss.detach().item())
        args._fmt_last_scaled_loss = float(scaled_loss.detach().item())
        args._fmt_last_num_layers = len(layer_losses)
        args._fmt_last_token_count = sum(token_counts) / max(1, len(token_counts))
        args._fmt_last_dim = sum(dims) / max(1, len(dims))

    return scaled_loss

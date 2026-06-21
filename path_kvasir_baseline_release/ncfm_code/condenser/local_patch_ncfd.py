import copy
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def patchify_images(images, grid):
    """Split images into a regular grid of non-overlapping patches."""
    if grid <= 0:
        raise ValueError(f"local_patch_grid must be positive, got {grid}")
    bsz, channels, height, width = images.shape
    if height % grid != 0 or width % grid != 0:
        raise ValueError(
            f"Image size {(height, width)} must be divisible by local_patch_grid={grid}"
        )
    patch_h = height // grid
    patch_w = width // grid
    patches = images.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    return patches.view(bsz, grid * grid, channels, patch_h, patch_w)


class FrozenConvPatchEncoder(nn.Module):
    """Frozen shallow ConvNet encoder that maps each patch to a feature vector."""

    def __init__(self, convnet, num_blocks=2):
        super().__init__()
        if not hasattr(convnet, "layers"):
            raise TypeError("FrozenConvPatchEncoder expects a ConvNet with a `layers` dict")

        max_blocks = len(convnet.layers["conv"])
        if num_blocks < 1 or num_blocks > max_blocks:
            raise ValueError(
                f"local_patch_encoder_blocks must be in [1, {max_blocks}], got {num_blocks}"
            )

        blocks = []
        has_norm = len(convnet.layers["norm"]) > 0
        has_pool = len(convnet.layers["pool"]) > 0
        for idx in range(num_blocks):
            modules = [copy.deepcopy(convnet.layers["conv"][idx])]
            if has_norm:
                modules.append(copy.deepcopy(convnet.layers["norm"][idx]))
            modules.append(copy.deepcopy(convnet.layers["act"][idx]))
            if has_pool:
                modules.append(copy.deepcopy(convnet.layers["pool"][idx]))
            blocks.append(nn.Sequential(*modules))

        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()

    def forward(self, x):
        x = self.blocks(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


class LocalPatchEncoderEnsemble(nn.Module):
    """Frozen ensemble wrapper for patch encoders.

    aggregate="mean" keeps the original feature dimension and is the default for
    fair comparison with single-teacher v1. aggregate="concat" is available for
    explicit feature-dimension ablations.
    """

    def __init__(self, encoders, aggregate="mean"):
        super().__init__()
        if not encoders:
            raise ValueError("LocalPatchEncoderEnsemble requires at least one encoder")
        if aggregate not in {"mean", "concat"}:
            raise ValueError(f"Unsupported local_patch_ensemble_aggregate={aggregate}")
        self.encoders = nn.ModuleList(encoders)
        self.aggregate = aggregate
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()

    def forward(self, x):
        feats = [encoder(x) for encoder in self.encoders]
        if self.aggregate == "concat":
            return torch.cat(feats, dim=1)
        return torch.stack(feats, dim=0).mean(dim=0)


class LocalPatchEncoderBank(nn.Module):
    """Frozen patch encoder bank with one sampled encoder per iteration."""

    def __init__(self, encoders, indices):
        super().__init__()
        if not encoders:
            raise ValueError("LocalPatchEncoderBank requires at least one encoder")
        if len(encoders) != len(indices):
            raise ValueError("Encoder bank and index list must have the same length")
        self.encoders = nn.ModuleList(encoders)
        self.indices = [int(index) for index in indices]
        self.active_pos = 0
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()

    @property
    def active_index(self):
        return self.indices[self.active_pos]

    def sample(self, rng):
        self.active_pos = rng.randrange(len(self.encoders))
        return self.active_index

    def forward(self, x):
        return self.encoders[self.active_pos](x)


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _parse_int_list(value):
    if value is None:
        return None
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        cleaned = cleaned.strip("[]()")
        return [int(v.strip()) for v in cleaned.split(",") if v.strip()]
    raise TypeError(f"Cannot parse integer list from {value!r}")


def _default_model_num(args):
    for name in ("local_patch_model_num", "num_premodel", "model_num"):
        if hasattr(args, name):
            return int(getattr(args, name))
    return 20


def local_patch_uses_model_interval(args):
    source = str(getattr(args, "local_patch_encoder_source", "premodel0_trained"))
    return source.lower() in {
        "model_interval",
        "model_interval_step",
        "ncfm_interval",
        "ncfm_interval_step",
    }


def _resolve_encoder_plan(args):
    source = str(getattr(args, "local_patch_encoder_source", "premodel0_trained"))
    source = source.lower()
    stage = str(getattr(args, "local_patch_checkpoint_stage", "")).lower()

    if source.endswith("_trained"):
        stage = "trained"
    elif source.endswith("_init"):
        stage = "init"
    elif not stage:
        stage = "trained"

    if source in {"premodel0_trained", "premodel0_init"}:
        return [0], stage, "single"

    if source in {"premodel_trained", "premodel_init", "single_trained", "single_init"}:
        index = int(getattr(args, "local_patch_premodel_index", 0))
        return [index], stage, "single"

    model_num = _default_model_num(args)
    explicit = _parse_int_list(getattr(args, "local_patch_premodel_indices", None))
    rng_seed = int(getattr(args, "local_patch_encoder_seed", getattr(args, "seed", 0)))
    rng = random.Random(rng_seed)

    if source in {"random_trained", "random_init", "random"}:
        if explicit:
            index = rng.choice(explicit)
        else:
            index = rng.randrange(model_num)
        return [index], stage, "random"

    if source in {
        "random_trained_step",
        "random_step_trained",
        "step_random_trained",
        "random_step",
    }:
        if explicit:
            indices = explicit
        else:
            indices = list(range(model_num))
        return indices, "trained", "random_step"

    if source in {"ensemble_trained", "ensemble_init", "ensemble"}:
        if explicit:
            indices = explicit
        else:
            size = int(getattr(args, "local_patch_ensemble_size", min(4, model_num)))
            if size < 1 or size > model_num:
                raise ValueError(
                    f"local_patch_ensemble_size must be in [1, {model_num}], got {size}"
                )
            indices = list(range(model_num))
            if _as_bool(getattr(args, "local_patch_ensemble_random", False)):
                indices = sorted(rng.sample(indices, size))
            else:
                indices = indices[:size]
        return indices, stage, "ensemble"

    raise ValueError(
        "Unsupported local_patch_encoder_source="
        f"{source}. Supported: premodel0_trained, premodel0_init, "
        "premodel_trained/init, random_trained/init, random_trained_step, "
        "ensemble_trained/init, model_interval_step."
    )


def _build_one_encoder(args, checkpoint_path, num_blocks):
    from utils.ddp import load_state_dict
    from utils.utils import define_model

    model = define_model(
        args.dataset,
        args.norm_type,
        args.net_type,
        args.nch,
        args.depth,
        args.width,
        args.nclass,
        args.logger,
        args.size,
    ).to(args.device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Patch encoder checkpoint not found: {checkpoint_path}")
    load_state_dict(checkpoint_path, model)
    model.eval()
    return FrozenConvPatchEncoder(model, num_blocks=num_blocks).to(args.device)


def build_frozen_patch_encoder(args):
    """Build frozen patch encoder(s) from premodel checkpoints.

    v2 keeps v1 behavior by default:
      local_patch_encoder_source: premodel0_trained
      local_patch_encoder_blocks: 2

    New ablation fields:
      local_patch_encoder_blocks: 1/2/3
      local_patch_encoder_source: premodel_trained | random_trained |
                                  ensemble_trained | model_interval_step
      local_patch_premodel_index: int
      local_patch_premodel_indices: [0,1,2] or "0,1,2"
      local_patch_ensemble_size: int
      local_patch_ensemble_aggregate: mean | concat
    """

    if local_patch_uses_model_interval(args):
        num_blocks = int(getattr(args, "local_patch_encoder_blocks", 2))
        args.local_patch_encoder_indices = []
        args.local_patch_encoder_stage = "model_interval"
        args.local_patch_encoder_mode = "model_interval_step"
        args.local_patch_encoder_paths = []
        args._local_patch_last_encoder_index = "model_interval"
        if getattr(args, "rank", 0) == 0:
            args.logger(
                "Local patch-feature NCFD encoder: "
                f"source={getattr(args, 'local_patch_encoder_source')}, "
                "mode=model_interval_step, stage=interpolated, "
                f"indices=[], blocks={num_blocks}, aggregate=none, checkpoints=[]"
            )
        return None

    indices, stage, mode = _resolve_encoder_plan(args)
    num_blocks = int(getattr(args, "local_patch_encoder_blocks", 2))
    aggregate = str(getattr(args, "local_patch_ensemble_aggregate", "mean")).lower()

    encoders = []
    checkpoint_paths = []
    for index in indices:
        checkpoint_path = os.path.join(
            args.pretrain_dir, f"premodel{int(index)}_{stage}.pth.tar"
        )
        encoders.append(_build_one_encoder(args, checkpoint_path, num_blocks))
        checkpoint_paths.append(checkpoint_path)

    if mode == "random_step":
        patch_encoder = LocalPatchEncoderBank(encoders, indices).to(args.device)
    elif len(encoders) == 1:
        patch_encoder = encoders[0]
    else:
        patch_encoder = LocalPatchEncoderEnsemble(encoders, aggregate=aggregate).to(
            args.device
        )

    patch_encoder.eval()
    args.local_patch_encoder_indices = indices
    args.local_patch_encoder_stage = stage
    args.local_patch_encoder_mode = mode
    args.local_patch_encoder_paths = checkpoint_paths
    if mode == "random_step":
        args.local_patch_encoder_rng = random.Random(
            int(getattr(args, "local_patch_encoder_seed", getattr(args, "seed", 0)))
        )
        args._local_patch_last_encoder_index = patch_encoder.sample(
            args.local_patch_encoder_rng
        )

    if getattr(args, "rank", 0) == 0:
        args.logger(
            "Local patch-feature NCFD encoder: "
            f"source={getattr(args, 'local_patch_encoder_source', 'premodel0_trained')}, "
            f"mode={mode}, stage={stage}, indices={indices}, "
            f"blocks={num_blocks}, aggregate={aggregate}, "
            f"checkpoints={checkpoint_paths}"
        )
    return patch_encoder


def resample_local_patch_encoder(args):
    patch_encoder = getattr(args, "local_patch_encoder", None)
    if patch_encoder is None or not hasattr(patch_encoder, "sample"):
        return getattr(args, "_local_patch_last_encoder_index", None)

    rng = getattr(args, "local_patch_encoder_rng", None)
    if rng is None:
        rng = random.Random(
            int(getattr(args, "local_patch_encoder_seed", getattr(args, "seed", 0)))
        )
        args.local_patch_encoder_rng = rng
    selected = patch_encoder.sample(rng)
    args._local_patch_last_encoder_index = selected
    return selected


def _extract_patch_features(images, patch_encoder, grid):
    patches = patchify_images(images, grid)
    bsz, num_patches, channels, patch_h, patch_w = patches.shape
    patches = patches.view(bsz * num_patches, channels, patch_h, patch_w)
    feats = patch_encoder(patches)
    return feats.view(bsz, num_patches, -1)


def _extract_patch_features_from_model_interval(images, model_interval, grid, num_blocks):
    if not hasattr(model_interval, "layers"):
        raise TypeError(
            "model_interval_step local patch source expects a ConvNet-style "
            "model with a `layers` ModuleDict"
        )

    layers = model_interval.layers
    max_blocks = len(layers["conv"])
    if num_blocks < 1 or num_blocks > max_blocks:
        raise ValueError(
            f"local_patch_encoder_blocks must be in [1, {max_blocks}], got {num_blocks}"
        )

    patches = patchify_images(images, grid)
    bsz, num_patches, channels, patch_h, patch_w = patches.shape
    x = patches.view(bsz * num_patches, channels, patch_h, patch_w)

    has_norm = len(layers["norm"]) > 0
    has_pool = len(layers["pool"]) > 0
    for idx in range(num_blocks):
        x = layers["conv"][idx](x)
        if has_norm:
            x = layers["norm"][idx](x)
        x = layers["act"][idx](x)
        if has_pool:
            x = layers["pool"][idx](x)

    x = F.adaptive_avg_pool2d(x, (1, 1))
    feats = torch.flatten(x, 1)
    return feats.view(bsz, num_patches, -1)


def _local_patch_cf_loss(feat_real, feat_syn, cf_loss_func, args):
    local_num_freqs = int(
        getattr(args, "local_patch_num_freqs", min(int(args.num_freqs), 256))
    )
    loss_scale = float(getattr(args, "local_patch_loss_scale", 300.0))

    num_patches = feat_syn.shape[1]
    feature_dim = feat_syn.shape[2]
    expected_dim = getattr(args, "local_patch_feature_dim", None)
    if expected_dim is not None and int(expected_dim) > 0 and feature_dim != int(expected_dim):
        raise ValueError(
            f"Expected local patch feature dim {expected_dim}, got {feature_dim}. "
            "Set local_patch_feature_dim to the actual value, or 0 to skip this check."
        )

    t = torch.randn((local_num_freqs, feature_dim), device=feat_syn.device)
    loss = feat_syn.new_tensor(0.0)
    for patch_idx in range(num_patches):
        loss = loss + cf_loss_func(
            feat_real[:, patch_idx, :],
            feat_syn[:, patch_idx, :],
            t,
            args,
        )

    return loss_scale * loss / num_patches


def local_patch_feature_ncfd_loss(img_real, img_syn, patch_encoder, cf_loss_func, args):
    """Compute local NCFD over original-image patches encoded as frozen features."""
    if patch_encoder is None:
        raise ValueError("patch_encoder is required when local patch-feature NCFD is enabled")

    grid = int(getattr(args, "local_patch_grid", 4))

    with torch.no_grad():
        feat_real = _extract_patch_features(img_real, patch_encoder, grid)
        feat_real = F.normalize(feat_real, dim=2)
    feat_syn = _extract_patch_features(img_syn, patch_encoder, grid)
    feat_syn = F.normalize(feat_syn, dim=2)

    return _local_patch_cf_loss(feat_real, feat_syn, cf_loss_func, args)


def local_patch_model_interval_ncfd_loss(
    img_real, img_syn, model_interval, cf_loss_func, args
):
    """Compute local NCFD with the current NCFM interpolated model as encoder."""
    grid = int(getattr(args, "local_patch_grid", 4))
    num_blocks = int(getattr(args, "local_patch_encoder_blocks", 2))

    with torch.no_grad():
        feat_real = _extract_patch_features_from_model_interval(
            img_real, model_interval, grid, num_blocks
        )
        feat_real = F.normalize(feat_real, dim=2)
    feat_syn = _extract_patch_features_from_model_interval(
        img_syn, model_interval, grid, num_blocks
    )
    feat_syn = F.normalize(feat_syn, dim=2)

    args._local_patch_last_encoder_index = "model_interval"
    return _local_patch_cf_loss(feat_real, feat_syn, cf_loss_func, args)

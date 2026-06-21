import torch

from .local_patch_ncfd import (
    local_patch_feature_ncfd_loss,
    local_patch_model_interval_ncfd_loss,
    local_patch_uses_model_interval,
    resample_local_patch_encoder,
)
from NCFM.discrepancy_attention import discrepancy_attention_ncfd_loss
from NCFM.dr_ltm import dr_ltm_ncfd_loss
from NCFM.feature_map_token_ncfd import feature_map_token_ncfd_loss


def compute_match_loss(
    args,
    loader_real,
    sample_fn,
    aug_fn,
    inner_loss_fn,
    optim_img,
    class_list,
    timing_tracker,
    model_interval,
    data_grad,
    optim_sampling_net = None,
    sampling_net =None
):

    loss_total = 0
    match_grad_mean = 0
    lp_global_total = 0.0
    lp_local_total = 0.0
    lp_weighted_local_total = 0.0
    lp_total_loss_total = 0.0
    dgsa_global_total = 0.0
    dgsa_att_total = 0.0
    dgsa_weighted_total = 0.0
    dgsa_total_loss_total = 0.0
    dr_ltm_global_total = 0.0
    dr_ltm_loss_total = 0.0
    dr_ltm_weighted_total = 0.0
    dr_ltm_total_loss_total = 0.0
    fmt_global_total = 0.0
    fmt_loss_total = 0.0
    fmt_weighted_total = 0.0
    fmt_total_loss_total = 0.0

    use_local_patch = getattr(args, "use_local_patch_feature_ncfd", False)
    use_discrepancy_attention = getattr(
        args, "use_discrepancy_attention_ncfd", False
    )
    use_dr_ltm = getattr(args, "use_dr_ltm_ncfd", False)
    use_feature_map_token = getattr(args, "use_feature_map_token_ncfd", False)
    use_model_interval_patch = use_local_patch and local_patch_uses_model_interval(args)
    if use_model_interval_patch:
        args._local_patch_last_encoder_index = "model_interval"
    elif use_local_patch:
        resample_local_patch_encoder(args)

    for c in class_list:
        timing_tracker.start_step()

        img, _ = loader_real.class_sample(c)
        timing_tracker.record("data")
        img_syn, _ = sample_fn(c)

        img_aug = aug_fn(torch.cat([img, img_syn]))
        timing_tracker.record("aug")
        n = img.shape[0]

        loss_global = inner_loss_fn(
            img_aug[:n], img_aug[n:], model_interval, sampling_net, args
        )
        loss = loss_global
        if use_local_patch:
            if use_model_interval_patch:
                loss_local = local_patch_model_interval_ncfd_loss(
                    img_aug[:n],
                    img_aug[n:],
                    model_interval,
                    args.cf_loss_func,
                    args,
                )
            else:
                loss_local = local_patch_feature_ncfd_loss(
                    img_aug[:n],
                    img_aug[n:],
                    args.local_patch_encoder,
                    args.cf_loss_func,
                    args,
                )
            weighted_local = args.lambda_local_patch_ncfd * loss_local
            loss = loss_global + weighted_local
            lp_global_total += float(loss_global.detach().item())
            lp_local_total += float(loss_local.detach().item())
            lp_weighted_local_total += float(weighted_local.detach().item())
            lp_total_loss_total += float(loss.detach().item())
        if use_discrepancy_attention:
            loss_att = discrepancy_attention_ncfd_loss(
                img_aug[:n],
                img_aug[n:],
                model_interval,
                args.cf_loss_func,
                args,
            )
            weighted_att = float(
                getattr(args, "lambda_discrepancy_attention_ncfd", 0.1)
            ) * loss_att
            loss = loss + weighted_att
            dgsa_global_total += float(loss_global.detach().item())
            dgsa_att_total += float(loss_att.detach().item())
            dgsa_weighted_total += float(weighted_att.detach().item())
            dgsa_total_loss_total += float(loss.detach().item())
        if use_dr_ltm:
            loss_dr = dr_ltm_ncfd_loss(
                img_aug[:n],
                img_aug[n:],
                model_interval,
                args.cf_loss_func,
                args,
            )
            weighted_dr = float(getattr(args, "lambda_dr_ltm_ncfd", 0.1)) * loss_dr
            loss = loss + weighted_dr
            dr_ltm_global_total += float(loss_global.detach().item())
            dr_ltm_loss_total += float(loss_dr.detach().item())
            dr_ltm_weighted_total += float(weighted_dr.detach().item())
            dr_ltm_total_loss_total += float(loss.detach().item())
        if use_feature_map_token:
            loss_fmt = feature_map_token_ncfd_loss(
                img_aug[:n],
                img_aug[n:],
                model_interval,
                args.cf_loss_func,
                args,
            )
            weighted_fmt = float(
                getattr(args, "lambda_feature_map_token_ncfd", 0.1)
            ) * loss_fmt
            loss = loss + weighted_fmt
            fmt_global_total += float(loss_global.detach().item())
            fmt_loss_total += float(loss_fmt.detach().item())
            fmt_weighted_total += float(weighted_fmt.detach().item())
            fmt_total_loss_total += float(loss.detach().item())
        loss_total += loss.item()
        timing_tracker.record("loss")

        optim_img.zero_grad()
        if optim_sampling_net is not None:
            optim_sampling_net.zero_grad()
            loss.backward(retain_graph=True)
            optim_img.step()
            optim_img.zero_grad()
            (-loss_global).backward()
            optim_sampling_net.step()
            optim_sampling_net.zero_grad()
        else:
            loss.backward()
            optim_img.step()
        if data_grad is not None:
            match_grad_mean += torch.norm(data_grad).item()
        timing_tracker.record("backward")

    if use_local_patch:
        denom = max(1, len(class_list))
        args._lp_last_global_loss = lp_global_total / denom
        args._lp_last_local_loss = lp_local_total / denom
        args._lp_last_weighted_local_loss = lp_weighted_local_total / denom
        args._lp_last_total_loss = lp_total_loss_total / denom
    if use_discrepancy_attention:
        denom = max(1, len(class_list))
        args._dgsa_last_global_loss = dgsa_global_total / denom
        args._dgsa_last_att_loss = dgsa_att_total / denom
        args._dgsa_last_weighted_att_loss = dgsa_weighted_total / denom
        args._dgsa_last_total_loss = dgsa_total_loss_total / denom
    if use_dr_ltm:
        denom = max(1, len(class_list))
        args._dr_ltm_last_global_loss = dr_ltm_global_total / denom
        args._dr_ltm_last_loss = dr_ltm_loss_total / denom
        args._dr_ltm_last_weighted_loss = dr_ltm_weighted_total / denom
        args._dr_ltm_last_total_loss = dr_ltm_total_loss_total / denom
    if use_feature_map_token:
        denom = max(1, len(class_list))
        args._fmt_last_global_loss = fmt_global_total / denom
        args._fmt_last_token_loss = fmt_loss_total / denom
        args._fmt_last_weighted_token_loss = fmt_weighted_total / denom
        args._fmt_last_total_loss = fmt_total_loss_total / denom

    return loss_total, match_grad_mean


def compute_calib_loss(
    sample_fn,
    aug_fn,
    inter_loss_fn,
    optim_img,
    iter_calib,
    class_list,
    timing_tracker,
    model_final,
    calib_weight,
    data_grad,
):

    calib_loss_total = 0
    calib_grad_norm = 0
    for i in range(0, iter_calib):
        for c in class_list:
            timing_tracker.start_step()

            img_syn, label_syn = sample_fn(c)
            timing_tracker.record("data")

            img_aug = aug_fn(torch.cat([img_syn]))
            timing_tracker.record("aug")

            loss = calib_weight * inter_loss_fn(img_aug, label_syn, model_final)
            calib_loss_total += loss.item()
            timing_tracker.record("loss")

            optim_img.zero_grad()
            loss.backward()
            if data_grad is not None:
                calib_grad_norm = torch.norm(data_grad).item()
            optim_img.step()
            timing_tracker.record("backward")

    return calib_loss_total, calib_grad_norm

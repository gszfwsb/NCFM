import torch


def _optimizer_parameters(optimizer):
    return [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    ]


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
    sampling_net =None,
    outer_iter=None,
):

    loss_total = 0
    match_grad_mean = 0

    for c in class_list:
        timing_tracker.start_step()

        img, _ = loader_real.class_sample(c)
        timing_tracker.record("data")
        img_syn, _ = sample_fn(c)

        img_aug = aug_fn(torch.cat([img, img_syn]))
        timing_tracker.record("aug")
        n = img.shape[0]

        loss = inner_loss_fn(img_aug[:n], img_aug[n:], model_interval,sampling_net,args)
        loss_total += loss.item()
        timing_tracker.record("loss")

        optim_img.zero_grad()
        if optim_sampling_net is not None:
            sampling_net_ascent_scale = getattr(args, "sampling_net_ascent_scale", None)
            if sampling_net_ascent_scale is None:
                optim_sampling_net.zero_grad()
                loss.backward(retain_graph=True)
                optim_img.step()
                optim_img.zero_grad()
                (-loss).backward()
                optim_sampling_net.step()
                optim_sampling_net.zero_grad()
            else:
                sampling_net_ascent_scale = float(sampling_net_ascent_scale)
                sampling_net_update_interval = max(
                    1, int(getattr(args, "sampling_net_update_interval", 1))
                )
                update_sampling_net = (
                    sampling_net_ascent_scale != 0
                    and (
                        outer_iter is None
                        or outer_iter % sampling_net_update_interval == 0
                    )
                )
                sampling_net_params = _optimizer_parameters(optim_sampling_net)
                sampling_net_grads = None
                optim_sampling_net.zero_grad()
                if update_sampling_net and sampling_net_params:
                    sampling_net_grads = torch.autograd.grad(
                        loss,
                        sampling_net_params,
                        retain_graph=True,
                        allow_unused=True,
                    )
                loss.backward()
                optim_img.step()
                optim_img.zero_grad()
                optim_sampling_net.zero_grad()
                if sampling_net_grads is not None:
                    for parameter, grad in zip(sampling_net_params, sampling_net_grads):
                        if grad is not None:
                            parameter.grad = (-sampling_net_ascent_scale * grad).detach()
                    optim_sampling_net.step()
                    optim_sampling_net.zero_grad()
        else:
            loss.backward()
            optim_img.step()
        if data_grad is not None:
            match_grad_mean += torch.norm(data_grad).item()
        timing_tracker.record("backward")

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

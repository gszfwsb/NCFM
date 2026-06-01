import time
import torch
import numpy as np
from utils.experiment_tracker import AverageMeter, accuracy
from utils.mix_cut_up import random_indices, rand_bbox
from utils.ddp import sync_distributed_metric
import torch.nn.functional as F


class _GraphForwardModule(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input):
        return self.model(input)


class CudaGraphModelRunner:
    def __init__(self, model, enabled=False, logger=None, rank=0):
        self.model = model
        self.enabled = enabled and torch.cuda.is_available()
        self.logger = logger
        self.rank = rank
        self.graph_key = None
        self.graph_model = None
        self.wrapper = None
        self.disabled_reason = None
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            self.disabled_reason = "DistributedDataParallel is not supported"

    def _log(self, message):
        if self.logger is not None and self.rank == 0:
            self.logger(message)

    def __call__(self, input):
        if not self.enabled:
            return self.model(input)
        if self.disabled_reason is not None:
            if self.disabled_reason:
                self._log(
                    f"CUDA graph disabled for evaluation training: {self.disabled_reason}"
                )
                self.disabled_reason = ""
            return self.model(input)

        key = (tuple(input.shape), input.dtype, input.device)
        if self.graph_key is not None and key != self.graph_key:
            return self.model(input)

        if self.graph_model is None:
            try:
                sample = input.detach().clone()
                self.wrapper = _GraphForwardModule(self.model)
                self.graph_model = torch.cuda.make_graphed_callables(
                    self.wrapper, (sample,)
                )
                self.graph_key = key
                self._log(f"Captured CUDA graph for evaluation train batch shape {tuple(input.shape)}")
            except Exception as exc:
                self.disabled_reason = str(exc)
                self._log(f"CUDA graph disabled for evaluation training: {exc}")
                return self.model(input)

        return self.graph_model(input)


def train_epoch(
    args,
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    aug=None,
    mixup="cut",
    model_runner=None,
    sync_metrics=True,
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        data_time.update(time.time() - end)

        if aug is not None:
            with torch.no_grad():
                input = aug(input)
        r = np.random.rand(1)
        if r < args.mix_p and mixup == "cut":
            lam = np.random.beta(args.beta, args.beta)
            rand_index = random_indices(target, nclass=args.nclass)
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]
            ratio = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2])
            )
            output = model_runner(input) if model_runner is not None else model(input)
            loss = criterion(output, target) * ratio + criterion(output, target_b) * (
                1.0 - ratio
            )
        else:
            output = model_runner(input) if model_runner is not None else model(input)
            loss = criterion(output, target)
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

    metrics = [top1.avg, top5.avg, losses.avg]
    if sync_metrics:
        return sync_distributed_metric(metrics)
    return metrics


def get_softlabel(img, teacher_model, target=None):
    # Get the soft labels
    softlabel = teacher_model(img).detach()  # [n, class]

    # If target is None, directly return the soft labels
    if target is None:
        return softlabel

    # Get the predicted class for each sample in the soft labels
    predicted = torch.argmax(softlabel, dim=1)  # [n]

    # Find the indices of misclassified samples
    incorrect_indices = predicted != target  # [n], True indicates misclassified samples

    # Replace the misclassified parts with the correct labels
    # Initialize the soft labels to all zeros
    corrected_softlabel = softlabel.clone()
    corrected_softlabel[incorrect_indices] = (
        0  # Set all class probabilities to 0 for misclassified samples
    )
    corrected_softlabel[incorrect_indices, target[incorrect_indices]] = (
        1  # Set the correct class probability to 1
    )

    return corrected_softlabel


def train_epoch_softlabel(
    args,
    train_loader,
    model,
    teacher_model,
    criterion,
    optimizer,
    epoch,
    aug=None,
    mixup="cut",
    sync_metrics=True,
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    end = time.time()
    teacher_model.eval()
    model.train()
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            # soft_label = get_softlabel(input,teacher_model,target).detach()
            soft_label = teacher_model(input).detach()
            soft_label = F.softmax(soft_label / args.temperature, dim=1)
        data_time.update(time.time() - end)

        if aug is not None:
            with torch.no_grad():
                input = aug(input)
        r = np.random.rand(1)
        if r < args.mix_p and mixup == "cut":
            lam = np.random.beta(args.beta, args.beta)
            rand_index = random_indices(target, nclass=args.nclass)
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]
            ratio = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2])
            )
            output = model(input)
            output = F.log_softmax(output / args.temperature, dim=1)
            loss = criterion(output, soft_label, args.temperature) * ratio + criterion(
                output, soft_label[rand_index, :], args.temperature
            ) * (1.0 - ratio)
        else:
            output = model(input)
            loss = criterion(output, soft_label, args.temperature)
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

    metrics = [top1.avg, top5.avg, losses.avg]
    if sync_metrics:
        return sync_distributed_metric(metrics)
    return metrics


def train_epoch_softlabel(
    args,
    train_loader,
    model,
    teacher_model,
    criterion,
    optimizer,
    epoch,
    aug=None,
    mixup="cut",
    sync_metrics=True,
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    end = time.time()
    teacher_model.eval()
    model.train()
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            soft_label = get_softlabel(input, teacher_model, target).detach()
        data_time.update(time.time() - end)

        if aug is not None:
            with torch.no_grad():
                input = aug(input)
        r = np.random.rand(1)
        if r < args.mix_p and mixup == "cut":
            lam = np.random.beta(args.beta, args.beta)
            rand_index = random_indices(target, nclass=args.nclass)
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]
            ratio = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2])
            )
            output = model(input)
            loss = criterion(output, soft_label) * ratio + criterion(
                output, soft_label[rand_index, :]
            ) * (1.0 - ratio)
        else:
            output = model(input)
            loss = criterion(output, soft_label)
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

    metrics = [top1.avg, top5.avg, losses.avg]
    if sync_metrics:
        return sync_distributed_metric(metrics)
    return metrics


def validate(val_loader, model, criterion, sync_metrics=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    metrics = [top1.avg, top5.avg, losses.avg]
    if sync_metrics:
        return sync_distributed_metric(metrics)
    return metrics

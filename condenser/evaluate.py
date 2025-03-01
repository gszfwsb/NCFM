import torch
import torch.nn as nn
import torch.optim as optim
from utils.experiment_tracker import get_time
from utils.diffaug import DiffAug
from utils.utils import define_model
from utils.ddp import load_state_dict
import warnings
from utils.train_val import train_epoch, validate, train_epoch_softlabel

warnings.filterwarnings("ignore")
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
import os


def SoftCrossEntropy(inputs, target, temperature=1.0, reduction="average"):
    input_log_likelihood = -F.log_softmax(inputs / temperature, dim=1)
    target_log_likelihood = F.softmax(target / temperature, dim=1)
    batch = inputs.shape[0]
    loss = torch.sum(torch.mul(input_log_likelihood, target_log_likelihood)) / batch
    return loss


# loss_function_kl = nn.KLDivLoss(reduction="batchmean")
def evaluate_syn_data(args, model, train_loader, val_loader, logger=None):
    if args.softlabel:
        teacher_model = define_model(
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
        teacher_path = os.path.join(args.pretrain_dir, f"premodel0_trained.pth.tar")
        load_state_dict(teacher_path, teacher_model)
        train_criterion_sl = SoftCrossEntropy
    train_criterion = nn.CrossEntropyLoss().cuda()
    val_criterion = nn.CrossEntropyLoss().cuda()
    if args.eval_optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.adamw_lr)
        if logger and dist.get_rank() == 0:
            logger(f"Using AdamW optimizer with learning rate: {args.adamw_lr}")
    elif args.eval_optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum
        )
        if logger and dist.get_rank() == 0:
            logger(f"Using SGD optimizer with learning rate: {args.lr}")
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            args.evaluation_epochs // 5,
            2 * args.evaluation_epochs // 5,
            3 * args.evaluation_epochs // 5,
            4 * args.evaluation_epochs // 5,
        ],
        gamma=0.5,
    )
    # scheduler = optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[args.evaluation_epochs//2], gamma=0.1)

    best_acc1, best_acc5 = 0, 0
    acc1, acc5 = 0, 0
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.rank], output_device=args.rank
    )

    if args.dsa:
        aug = DiffAug(strategy=args.dsa_strategy, batch=False)
        if args.rank == 0:
            logger(f"Start training with DSA and {args.mixup} mixup")
    else:
        aug = None
        if args.rank == 0:
            logger(f"Start training with base augmentation and {args.mixup} mixup")
    pbar = tqdm(range(1, args.evaluation_epochs + 1))
    for epoch in range(1, args.evaluation_epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        if args.softlabel and epoch < (
            args.evaluation_epochs - args.epoch_eval_interval
        ):
            acc1_tr, acc5_tr, loss_tr = train_epoch_softlabel(
                args,
                train_loader,
                model,
                teacher_model,
                train_criterion_sl,
                optimizer,
                epoch,
                aug,
                mixup=args.mixup,
            )
        else:
            acc1_tr, acc5_tr, loss_tr = train_epoch(
                args,
                train_loader,
                model,
                train_criterion,
                optimizer,
                epoch,
                aug,
                mixup=args.mixup,
            )
        if args.rank == 0:
            pbar.set_description(
                f"[Epoch {epoch}/{args.evaluation_epochs}] (Train) Top1 {acc1_tr:.1f}  Top5 {acc5_tr:.1f} Lr {optimizer.param_groups[0]['lr']} Loss {loss_tr:.3f}"
            )
            pbar.update(1)
            if (epoch % args.epoch_print_freq == 0) and (logger is not None) == 0:
                logger(
                    "(Train) [Epoch {0}/{1}] {2} Top1 {top1:.1f}  Top5 {top5:.1f}  Loss {loss:.3f}".format(
                        epoch,
                        args.evaluation_epochs,
                        get_time(),
                        top1=acc1_tr,
                        top5=acc5_tr,
                        loss=loss_tr,
                    )
                )

        if (
            epoch % args.epoch_eval_interval == 0
            or epoch == args.evaluation_epochs
            or (epoch % (args.epoch_eval_interval / 50) == 0 and args.ipc >= 50)
        ):
            acc1, acc5, loss_val = validate(val_loader, model, val_criterion)
            is_best = acc1 > best_acc1
            if is_best:
                best_acc1 = acc1
                best_acc5 = acc5
            if logger is not None and args.rank == 0:
                logger(
                    "-------Eval Training Epoch [{} / {}] INFO--------".format(
                        epoch, args.evaluation_epochs
                    )
                )
                logger(
                    f"Current accuracy (top-1 and 5): {acc1:.1f} {acc5:.1f}, loss: {loss_val:.3f}"
                )
                logger(
                    f"Best    accuracy (top-1 and 5): {best_acc1:.1f} {best_acc5:.1f}"
                )

        scheduler.step()

    return best_acc1, acc1

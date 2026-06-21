import os
import torch
import random
import torch.distributed as dist
import torch.functional as F
import numpy as np
import contextlib
from utils.ddp import load_state_dict
from utils.experiment_tracker import LossPlotter
from data.dataloader import (
    ClassDataLoader,
    ClassMemDataLoader,
    AsyncLoader,
    ImageNetMemoryDataLoader,
)
from torch.utils.data import DataLoader, DistributedSampler
import models.resnet as RN
import models.resnet_ap as RNAP
import models.convnet as CN
import models.densenet_cifar as DN
from efficientnet_pytorch import EfficientNet
from torchvision import datasets, transforms
from data.transform import transform_imagenet, transform_medmnist, transform_kvasirv2
from data.dataset import ImageFolder, ImageFolder_mtt
from data.dataset_statistics import MEANS, STDS
from data.medmnist import (
    build_medmnist_dataset,
    get_medmnist_root,
    is_supported_medmnist,
    register_medmnist_stats,
)


class BlurPoolConv2d(torch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        default_filter = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer("blur_filter", filt)

    def forward(self, x):
        blurred = F.conv2d(
            x,
            self.blur_filter,
            stride=1,
            padding=(1, 1),
            groups=self.conv.in_channels,
            bias=None,
        )
        return self.conv.forward(blurred)


def apply_blurpool(mod: torch.nn.Module):
    for name, child in mod.named_children():
        if isinstance(child, torch.nn.Conv2d) and (
            np.max(child.stride) > 1 and child.in_channels >= 16
        ):
            setattr(mod, name, BlurPoolConv2d(child))
        else:
            apply_blurpool(child)


def define_model(dataset, norm_type, net_type, nch, depth, width, nclass, logger, size):

    if net_type == "resnet":
        model = RN.ResNet(
            dataset, depth, nclass, norm_type=norm_type, size=size, nch=nch
        )
    elif net_type == "resnet_ap":
        model = RNAP.ResNetAP(
            dataset, depth, nclass, width=width, norm_type=norm_type, size=size, nch=nch
        )
        apply_blurpool(model)
    elif net_type == "efficient":
        model = EfficientNet.from_name("efficientnet-b0", num_classes=nclass)
    elif net_type == "densenet":
        model = DN.densenet_cifar(nclass)
    elif net_type == "convnet":
        width = int(128 * width)
        model = CN.ConvNet(
            nclass,
            net_norm=norm_type,
            net_depth=depth,
            net_width=width,
            channel=nch,
            im_size=(size, size),
        )
    else:
        raise Exception("unknown network architecture: {}".format(net_type))

    # if logger is not None:
    #     if dist.get_rank() == 0:
    #         logger(f"=> creating model {net_type}-{depth}, norm: {norm_type}")
    #         logger('# model parameters: {:.1f}M'.format(sum([p.data.nelement() for p in model.parameters()]) / 10**6))
    return model


def load_resized_data(
    dataset, data_dir, size=None, nclass=None, load_memory=False, seed=0
):

    if is_supported_medmnist(dataset):
        medmnist_root = get_medmnist_root(data_dir)
        register_medmnist_stats(dataset, medmnist_root, size=size)
    normalize = transforms.Normalize(mean=MEANS[dataset], std=STDS[dataset])
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        if dataset == "cifar10":
            train_dataset = datasets.CIFAR10(
                data_dir, download=True, train=True, transform=transforms.ToTensor()
            )
            transform_test = (
                transforms.Compose([transforms.ToTensor(), normalize])
                if normalize
                else transforms.ToTensor()
            )
            val_dataset = datasets.CIFAR10(
                data_dir, train=False, transform=transform_test
            )
            train_dataset.nclass = 10

        elif dataset == "cifar100":
            train_dataset = datasets.CIFAR100(
                data_dir, download=True, train=True, transform=transforms.ToTensor()
            )
            transform_test = (
                transforms.Compose([transforms.ToTensor(), normalize])
                if normalize
                else transforms.ToTensor()
            )
            val_dataset = datasets.CIFAR100(
                data_dir, train=False, transform=transform_test
            )
            train_dataset.nclass = 100

        elif dataset == "svhn":
            train_dataset = datasets.SVHN(
                os.path.join(data_dir, "SVHN"),
                download=True,
                split="train",
                transform=transforms.ToTensor(),
            )
            train_dataset.targets = train_dataset.labels
            transform_test = (
                transforms.Compose([transforms.ToTensor(), normalize])
                if normalize
                else transforms.ToTensor()
            )
            val_dataset = datasets.SVHN(
                os.path.join(data_dir, "SVHN"), split="test", transform=transform_test
            )
            train_dataset.nclass = 10

        elif dataset == "mnist":
            train_dataset = datasets.MNIST(
                data_dir, download=True, train=True, transform=transforms.ToTensor()
            )
            transform_test = (
                transforms.Compose([transforms.ToTensor(), normalize])
                if normalize
                else transforms.ToTensor()
            )
            val_dataset = datasets.MNIST(
                data_dir, train=False, transform=transform_test
            )
            train_dataset.nclass = 10

        elif dataset == "fashion":
            train_dataset = datasets.FashionMNIST(
                data_dir, download=True, train=True, transform=transforms.ToTensor()
            )
            transform_test = (
                transforms.Compose([transforms.ToTensor(), normalize])
                if normalize
                else transforms.ToTensor()
            )
            val_dataset = datasets.FashionMNIST(
                data_dir, train=False, transform=transform_test
            )
            train_dataset.nclass = 10

        elif is_supported_medmnist(dataset):
            medmnist_root = get_medmnist_root(data_dir)
            train_dataset = build_medmnist_dataset(
                dataset,
                medmnist_root,
                split="train",
                transform=transforms.ToTensor(),
                size=size,
                download=True,
            )
            _, transform_test = transform_medmnist(
                dataset,
                size=28 if size is None else size,
                augment=False,
                from_tensor=False,
                normalize=True,
            )
            val_dataset = build_medmnist_dataset(
                dataset,
                medmnist_root,
                split="test",
                transform=transform_test,
                size=size,
                download=True,
            )

        elif dataset == "tinyimagenet":
            data_path = os.path.join(data_dir, "tinyimagenet")
            transform_test = (
                transforms.Compose([transforms.ToTensor(), normalize])
                if normalize
                else transforms.ToTensor()
            )
            train_dataset = datasets.ImageFolder(
                os.path.join(data_path, "train"), transform=transforms.ToTensor()
            )
            val_dataset = datasets.ImageFolder(
                os.path.join(data_path, "val"), transform=transform_test
            )
            train_dataset.nclass = 200

        elif dataset == "kvasirv2":
            data_path = os.path.join(data_dir, "kvasirv2")
            train_dir = os.path.join(data_path, "train")
            test_dir = os.path.join(data_path, "test")
            resize = transforms.Compose(
                [
                    transforms.Resize(size),
                    transforms.CenterCrop(size),
                    transforms.PILToTensor(),
                ]
            )
            if load_memory:
                train_transform = resize
            else:
                train_transform = transforms.Compose(
                    [
                        transforms.Resize(size),
                        transforms.CenterCrop(size),
                        transforms.ToTensor(),
                    ]
                )
            _, test_transform = transform_kvasirv2(
                size=size,
                augment=False,
                from_tensor=False,
                normalize=True,
            )
            train_dataset = datasets.ImageFolder(
                train_dir,
                transform=train_transform,
            )
            val_dataset = datasets.ImageFolder(
                test_dir,
                transform=test_transform,
            )
            train_dataset.nclass = 8
            val_dataset.nclass = 8

        elif dataset in [
            "imagenette",
            "imagewoof",
            "imagemeow",
            "imagesquawk",
            "imagefruit",
            "imageyellow",
        ]:
            traindir = os.path.join(data_dir, "train")
            valdir = os.path.join(data_dir, "val")
            resize = transforms.Compose(
                [
                    transforms.Resize(size),
                    transforms.CenterCrop(size),
                    transforms.PILToTensor(),
                ]
            )
            if load_memory:
                transform = None
                load_transform = resize
            else:
                transform = transforms.Compose(
                    [resize, transforms.ConvertImageDtype(torch.float)]
                )
                load_transform = None

            _, test_transform = transform_imagenet(size=size)
            train_dataset = ImageFolder_mtt(
                traindir,
                transform=transform,
                type=dataset,
                load_memory=load_memory,
                load_transform=load_transform,
            )
            val_dataset = ImageFolder_mtt(
                valdir, test_transform, type=dataset, load_memory=False
            )

        elif dataset == "imagenet":
            traindir = os.path.join(data_dir, "train")
            valdir = os.path.join(data_dir, "val")
            resize = transforms.Compose(
                [
                    transforms.Resize(size),
                    transforms.CenterCrop(size),
                    transforms.PILToTensor(),
                ]
            )
            if load_memory:
                transform = None
                load_transform = resize
            else:
                transform = transforms.Compose(
                    [resize, transforms.ConvertImageDtype(torch.float)]
                )
                load_transform = None

            _, test_transform = transform_imagenet(size=size)
            train_dataset = ImageFolder(
                traindir,
                transform=transform,
                nclass=nclass,
                seed=seed,
                load_memory=load_memory,
                load_transform=load_transform,
            )
            val_dataset = ImageFolder(
                valdir, test_transform, nclass=nclass, seed=seed, load_memory=False
            )

        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        assert (
            train_dataset[0][0].shape[-1] == val_dataset[0][0].shape[-1]
        ), "Train and Val dataset sizes do not match"

    return train_dataset, val_dataset


def get_plotter(args):
    base_filename = f"{args.dataset}_ipc{args.ipc}_factor{args.factor}_{args.optimizer}_alpha{args.alpha_for_loss}_beta{args.beta_for_loss}_dis{args.dis_metrics}_freqs{args.num_freqs}_calib{args.iter_calib}"
    if getattr(args, "dam_enabled", False):
        dam_layers = getattr(args, "dam_attention_layers", "all")
        if isinstance(dam_layers, (list, tuple)):
            dam_layers = "".join(str(layer) for layer in dam_layers)
        base_filename += (
            f"_DAM_obj{getattr(args, 'dam_objective', 'ncfm_attention')}"
            f"_attnW{getattr(args, 'dam_attention_weight', 10.0)}"
            f"_featW{getattr(args, 'dam_feature_weight', 1.0)}"
            f"_layers{dam_layers}"
        )
    if getattr(args, "use_local_patch_feature_ncfd", False):
        lp_indices = getattr(args, "local_patch_premodel_indices", None)
        if lp_indices is None:
            lp_indices = getattr(args, "local_patch_premodel_index", "")
        if isinstance(lp_indices, (list, tuple)):
            lp_indices = "-".join(str(idx) for idx in lp_indices)
        base_filename += (
            f"_LP_grid{getattr(args, 'local_patch_grid', 4)}"
            f"_lam{getattr(args, 'lambda_local_patch_ncfd', 0.0)}"
            f"_lf{getattr(args, 'local_patch_num_freqs', 'auto')}"
            f"_src{getattr(args, 'local_patch_encoder_source', 'premodel0_trained')}"
            f"_idx{lp_indices}"
            f"_blk{getattr(args, 'local_patch_encoder_blocks', 2)}"
        )
    if getattr(args, "use_dr_ltm_ncfd", False):
        dr_layers = getattr(args, "dr_ltm_layers", "last")
        if isinstance(dr_layers, (list, tuple)):
            dr_layers = "".join(str(layer) for layer in dr_layers)
        base_filename += (
            f"_DRLTM_lam{getattr(args, 'lambda_dr_ltm_ncfd', 0.0)}"
            f"_alpha{getattr(args, 'dr_ltm_alpha', 0.25)}"
            f"_layers{dr_layers}"
            f"_nf{getattr(args, 'dr_ltm_num_freqs', 'auto')}"
            f"_mode{getattr(args, 'dr_ltm_mode', 'topk')}"
        )
    optimizer_info = {
        "type": args.optimizer,
        "lr": (
            args.lr_img * args.lr_scale_adam
            if args.optimizer.lower() in ["adam", "adamw"]
            else args.lr_img
        ),
        "weight_decay": args.weight_decay if args.optimizer.lower() == "adamw" else 0.0,
    }

    plotter = LossPlotter(
        save_path=args.save_dir,
        filename_pattern=base_filename,
        dataset=args.dataset,
        ipc=args.ipc,
        dis_metrics=args.dis_metrics,
        optimizer_info=optimizer_info,
    )
    return plotter


def get_optimizer(optimizer: str= "sgd", parameters=None,lr=0.01, mom_img=0.5,weight_decay=5e-4,logger=None):
    if optimizer.lower() == "sgd":
        optim_img = torch.optim.SGD(parameters, lr=lr, momentum=mom_img)
        if logger and dist.get_rank() == 0:
            logger(f"Using SGD optimizer with learning rate: {lr}")
    elif optimizer.lower() == "adam":
        optim_img = torch.optim.Adam(parameters, lr=lr)
        if logger and dist.get_rank() == 0:
            logger(f"Using Adam optimizer with learning rate: {lr}")
    elif optimizer.lower() == "adamw":
        optim_img = torch.optim.AdamW(
            parameters, lr=lr, weight_decay=weight_decay
        )
        if logger and dist.get_rank() == 0:
            logger(f"Using AdamW optimizer with learning rate: {lr}")
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer.lower()}")
    return optim_img


def get_loader(args):
    if args.run_mode == "Condense":
        if args.dataset == "imagenet":
            # For example,args.imagenet_prepath : "/data/imagenet/imagenet_prepare"
            # ls ./categorized_classes ==> class_0.pt class_1.pt ..
            for local_rank in range(args.local_world_size):
                if local_rank == args.local_rank:
                    loader_real = ImageNetMemoryDataLoader(
                        args.imagenet_prepath, class_list=args.class_list
                    )
                    print(
                        f"============RNAK:{dist.get_rank()}====LOCAL_RANK {local_rank} Loaded Categorized Data=========================="
                    )
                dist.barrier()
            _ = None
        else:
            train_set, _ = load_resized_data(
                args.dataset,
                args.data_dir,
                size=args.size,
                nclass=args.nclass,
                load_memory=args.load_memory,
            )
            if args.load_memory:
                loader_real = ClassMemDataLoader(train_set, batch_size=args.batch_real)
            else:
                loader_real = ClassDataLoader(
                    train_set,
                    batch_size=args.batch_real,
                    num_workers=args.workers,
                    shuffle=True,
                    pin_memory=True,
                    drop_last=True,
                )

        return loader_real, _
    elif args.run_mode == "Evaluation":
        _, val_dataset = load_resized_data(
            args.dataset,
            args.data_dir,
            size=args.size,
            nclass=args.nclass,
            load_memory=args.load_memory,
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=args.world_size, rank=args.rank
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(args.batch_size / args.world_size),
            sampler=val_sampler,
            num_workers=args.workers,
        )
        return _, val_loader

    elif args.run_mode == "Pretrain":
        train_set, val_dataset = load_resized_data(
            args.dataset,
            args.data_dir,
            size=args.size,
            nclass=args.nclass,
            load_memory=args.load_memory,
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=args.world_size, rank=args.rank
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(args.batch_size / args.world_size),
            sampler=val_sampler,
            num_workers=args.workers,
        )
        train_sampler = DistributedSampler(
            train_set, num_replicas=args.world_size, rank=args.rank
        )
        train_loader = DataLoader(
            train_set,
            batch_size=int(args.batch_size / args.world_size),
            sampler=train_sampler,
            num_workers=args.workers,
        )
        return train_loader, val_loader, train_sampler


def get_feature_extractor(args):
    model_init = define_model(
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
    model_final = define_model(
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
    model_interval = define_model(
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
    return model_init, model_interval, model_final


def update_feature_extractor(args, model_init, model_final, model_interval, a=0, b=1):
    if args.num_premodel > 0:
        # Select pre-trained model ID
        slkt_model_id = random.randint(0, args.num_premodel - 1)

        # Construct the paths
        init_path = os.path.join(
            args.pretrain_dir, f"premodel{slkt_model_id}_init.pth.tar"
        )
        final_path = os.path.join(
            args.pretrain_dir, f"premodel{slkt_model_id}_trained.pth.tar"
        )
        # Load the pre-trained models
        load_state_dict(init_path, model_init)
        load_state_dict(final_path, model_final)
        l = (b - a) * torch.rand(1).to(args.device) + a
        # Interpolate to initialize `model_interval`
        for model_interval_param, model_init_param, model_final_param in zip(
            model_interval.parameters(),
            model_init.parameters(),
            model_final.parameters(),
        ):
            model_interval_param.data.copy_(
                l * model_init_param.data + (1 - l) * model_final_param.data
            )

    else:
        if args.iter_calib > 0:
            slkt_model_id = random.randint(0, 9)
            final_path = os.path.join(
                args.pretrain_dir, f"premodel{slkt_model_id}_trained.pth.tar"
            )
            load_state_dict(final_path, model_final)
        # model_interval = define_model(args.dataset, args.norm_type, args.net_type, args.nch, args.depth, args.width, args.nclass, args.logger, args.size).to(args.device)
        slkt_model_id = random.randint(0, 9)
        interval_path = os.path.join(
            args.pretrain_dir, f"premodel{slkt_model_id}_trained.pth.tar"
        )
        load_state_dict(interval_path, model_interval)

    return model_init, model_final, model_interval

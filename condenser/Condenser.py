import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from utils.utils import update_feature_extractor
from utils.ddp import gather_save_visualize, sync_distributed_metric
from NCFM.NCFM import innerloss, interloss, mutil_layer_innerloss, CFLossFunc
from NCFM.SampleNet import SampleNet
from utils.experiment_tracker import TimingTracker, get_time
from data.dataset import TensorDataset
from utils.utils import define_model
from .evaluate import evaluate_syn_data
from torch.utils.data import DistributedSampler, DataLoader
from .decode import decode
from .subsample import subsample
from .condense_transfom import get_train_transform
from .compute_loss import compute_match_loss, compute_calib_loss
from data.dataloader import MultiEpochsDataLoader
import torch.optim as optim
from data.dataloader import AsyncLoader
from tqdm import tqdm
import random


class Condenser:
    def __init__(self, args, nclass_list, nchannel, hs, ws, device="cuda"):
        self.timing_tracker = TimingTracker(args.logger)
        self.args = args
        self.logger = args.logger
        self.ipc = args.ipc
        self.nclass_list = nclass_list
        self.nchannel = nchannel
        self.size = (hs, ws)
        self.device = device
        self.nclass = len(nclass_list)
        self.data = torch.randn(
            size=(self.nclass * self.ipc, self.nchannel, hs, ws),
            dtype=torch.float,
            requires_grad=True,
            device=self.device,
        )
        self.data.data = torch.clamp(self.data.data / 4 + 0.5, min=0.0, max=1.0)
        self.targets = torch.tensor(
            [np.ones(self.ipc) * c for c in self.nclass_list],
            dtype=torch.long,
            requires_grad=False,
            device=self.device,
        ).view(-1)
        self.cls_idx = [[] for _ in range(self.nclass)]
        for i in range(self.data.shape[0]):
            self.cls_idx[self.nclass_list.index(self.targets[i].item())].append(i)
        self.factor = max(1, args.factor)
        self.decode_type = args.decode_type
        self.resize = nn.Upsample(size=self.size, mode="bilinear")
        if dist.get_rank() == 0:
            self.logger(f"Factor: {self.factor} ({self.decode_type})")

    def load_condensed_data(self, loader, init_type="noise", load_path=None):
        if init_type == "random":
            if dist.get_rank() == 0:
                self.logger(
                    "===================Random initialize condensed==================="
                )
            for c in self.nclass_list:
                img, _ = loader.class_sample(c, self.ipc)
                self.data.data[
                    self.ipc
                    * self.nclass_list.index(c) : self.ipc
                    * (self.nclass_list.index(c) + 1)
                ] = img.data.to(self.device)
        elif init_type == "mix":
            if dist.get_rank() == 0:
                self.logger(
                    "===================Mixed initialize condensed==================="
                )
            for c in self.nclass_list:
                img, _ = loader.class_sample(c, self.ipc * self.factor**2)
                img = img.data.to(self.device)
                s = self.size[0] // self.factor
                remained = self.size[0] % self.factor
                k = 0
                n = self.ipc
                h_loc = 0
                for i in range(self.factor):
                    h_r = s + 1 if i < remained else s
                    w_loc = 0
                    for j in range(self.factor):
                        w_r = s + 1 if j < remained else s
                        img_part = F.interpolate(
                            img[k * n : (k + 1) * n], size=(h_r, w_r)
                        )
                        self.data.data[
                            n
                            * self.nclass_list.index(c) : n
                            * (self.nclass_list.index(c) + 1),
                            :,
                            h_loc : h_loc + h_r,
                            w_loc : w_loc + w_r,
                        ] = img_part
                        w_loc += w_r
                        k += 1
                    h_loc += h_r

        elif init_type == "noise":
            if dist.get_rank() == 0:
                self.logger(
                    "===================Noise initialize condensed dataset==================="
                )
            pass
        elif init_type == "load":
            if load_path is None:
                raise ValueError(
                    "===================Please provide the path of the initialization data==================="
                )
            if dist.get_rank() == 0:
                self.logger(
                    "==================designed path initialize condense dataset ==================="
                )
            data, target = torch.load(load_path)
            data_selected = []
            target_selected = []
            for c in self.nclass_list:
                indices = torch.where(target == c)[
                    0
                ]  # Get the indices for the current class
                data_selected.append(data[indices])
                target_selected.append(target[indices])
            # Concatenate all selected data and targets
            self.data.data = torch.cat(data_selected, dim=0).to(self.device)
            self.targets = torch.cat(target_selected, dim=0).to(self.device)

    def parameters(self):
        parameter_list = [self.data]
        return parameter_list

    def class_sample(self, c, max_size=10000):
        target_mask = self.targets == c
        data = self.data[target_mask]
        target = self.targets[target_mask]
        data, target = decode(
            self.decode_type, self.size, data, target, self.factor, bound=max_size
        )
        data, target = subsample(data, target, max_size=max_size)
        return data, target

    def get_syndataLoader(self, args, augment=True):
        train_transform, _ = get_train_transform(
            args.dataset,
            augment=augment,
            rrc=args.rrc,
            rrc_size=self.size[0],
            device=args.device,
        )
        data_dec = []
        target_dec = []
        for c in self.nclass_list:
            target_mask = self.targets == c
            data = self.data[target_mask].detach()
            target = self.targets[target_mask].detach()
            # data, target = self.decode(data, target)
            data, target = decode(
                self.decode_type, self.size, data, target, self.factor, bound=10000
            )

            data_dec.append(data)
            target_dec.append(target)

        data_dec = torch.cat(data_dec)
        target_dec = torch.cat(target_dec)
        if args.rank == 0:
            print("Decode condensed data: ", data_dec.shape)
        train_dataset = TensorDataset(data_dec, target_dec, train_transform)
        nw = 0 if not augment else args.workers
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True
        )
        # train_loader = DataLoader(train_dataset,batch_size=int(args.batch_size/args.world_size),sampler=train_sampler,num_workers=nw)
        train_loader = MultiEpochsDataLoader(
            train_dataset,
            batch_size=int(args.batch_size / args.world_size),
            sampler=train_sampler,
            num_workers=nw,
        )
        return train_loader

    def condense(
        self,
        args,
        plotter,
        loader_real,
        aug,
        optim_img,
        model_init,
        model_interval,
        model_final,
    ):
        loader_real = AsyncLoader(
            loader_real, args.class_list, args.batch_real, args.device
        )
        loader_syn = AsyncLoader(self, args.class_list, 100000, args.device)
        args.cf_loss_func = CFLossFunc(
            alpha_for_loss=args.alpha_for_loss, beta_for_loss=args.beta_for_loss
        )
        # if args.sampling_net:
        #     sampling_net = SampleNet(feature_dim=2048)
        #     optim_sampling_net =
        # scheduler = optim.lr_scheduler.MultiStepLR(
        #     optim_img, milestones=[1 * args.niter // 4,2 * args.niter // 4,  3* args.niter // 4], gamma=0.8
        # )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim_img, mode="min", factor=0.5, patience=500, verbose=False
        )
        gather_save_visualize(self, args)
        if args.local_rank == 0:
            pbar = tqdm(range(1, args.niter))
        for it in range(args.niter):
            model_init, model_final, model_interval = update_feature_extractor(
                args, model_init, model_final, model_interval, a=0, b=1
            )

            self.data.data = torch.clamp(self.data.data, min=0.0, max=1.0)
            match_loss_total, match_grad_mean, calib_loss_total, calib_grad_mean = (
                0,
                0,
                0,
                0,
            )
            match_loss_total, match_grad_mean = compute_match_loss(
                args,
                loader_real=loader_real,
                sample_fn=loader_syn.class_sample,
                aug_fn=aug,
                inner_loss_fn=innerloss if args.depth <= 5 else mutil_layer_innerloss,
                # inner_loss_fn=innerloss,
                optim_img=optim_img,
                class_list=self.args.class_list,
                timing_tracker=self.timing_tracker,
                model_interval=model_interval,
                data_grad=self.data.grad,
            )
            if args.iter_calib > 0:
                calib_loss_total, calib_grad_mean = compute_calib_loss(
                    sample_fn=loader_syn.class_sample,
                    aug_fn=aug,
                    inter_loss_fn=interloss,
                    optim_img=optim_img,
                    iter_calib=args.iter_calib,
                    class_list=self.args.class_list,
                    timing_tracker=self.timing_tracker,
                    model_final=model_final,
                    calib_weight=args.calib_weight,
                    data_grad=self.data.grad,
                )
            calib_loss_total, match_loss_total, match_grad_mean, calib_grad_mean = (
                sync_distributed_metric(
                    [
                        calib_loss_total,
                        match_loss_total,
                        match_grad_mean,
                        calib_grad_mean,
                    ]
                )
            )
            total_grad_mean = (
                match_grad_mean + calib_grad_mean
                if args.iter_calib > 0
                else match_grad_mean
            )
            current_loss = (
                (match_loss_total + calib_loss_total) / args.nclass
                if args.iter_calib > 0
                else (match_loss_total) / args.nclass
            )
            plotter.update_match_loss(match_loss_total / args.nclass)
            if args.iter_calib > 0:
                plotter.update_calib_loss(calib_loss_total / args.nclass)
            if it % args.it_log == 0:
                dist.barrier()
            if args.local_rank == 0:
                pbar.set_description(f"[Niter {it+1}/{args.niter+1}]")
                pbar.update(1)
            if it % args.it_log == 0 and args.rank == 0:
                timing_stats = self.timing_tracker.report(reset=True)
                current_lr = optim_img.param_groups[0]["lr"]
                plotter.plot_and_save_loss_curve()
                if args.iter_calib > 0:
                    args.logger(
                        f"\n{get_time()} (Iter {it:3d}) "
                        f"LR: {current_lr:.6f} "
                        f"inter-loss: {calib_loss_total / args.nclass / args.iter_calib:.2f} "
                        f"inner-loss: {match_loss_total / args.nclass:.2f} "
                        f"grad-norm: {total_grad_mean / args.nclass:.7f} "
                        f"Timing Stats: {timing_stats}"
                    )
                else:
                    args.logger(
                        f"\n{get_time()} (Iter {it:3d}) "
                        f"LR: {current_lr:.6f} "
                        f"inner-loss: {match_loss_total / args.nclass:.2f} "
                        f"grad-norm: {total_grad_mean / args.nclass:.7f} "
                        f"Timing Stats: {timing_stats}"
                    )
            if (it + 1) in args.it_save:
                gather_save_visualize(self, args, iteration=it)
            scheduler.step(current_loss)

    def evaluate(self, args, syndataloader, val_loader):
        if args.rank == 0:
            args.logger("======================Start Evaluation ======================")
        results = []
        all_best_acc = 0
        for i in range(args.val_repeat):
            if args.rank == 0:
                args.logger(
                    f"======================Repeat {i+1}/{args.val_repeat} Starting =================================================================="
                )
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
            best_acc, acc = evaluate_syn_data(
                args, model, syndataloader, val_loader, logger=args.logger
            )
            if all_best_acc < best_acc:
                all_best_acc = best_acc
            results.append(best_acc)
            if args.rank == 0:
                args.logger(
                    f"Repeat {i+1}/{args.val_repeat} => The Best Evaluation Acc: {all_best_acc:.1f} The Last Evaluation Acc :{acc:.1f} \n"
                )
        mean_result = np.mean(results)
        std_result = np.std(results)
        if args.rank == 0:
            args.logger("=" * 50)
            args.logger(f"Evaluation Stop:")
            args.logger(
                f"Mean Accuracy: {mean_result:.3f}", f"Std Deviation: {std_result:.3f}"
            )
            args.logger(f"All result: {[f'{x:.3f}' for x in results]}")
            args.logger("=" * 50)

    def continue_learning(self, args, syndataloader, val_loader):
        if args.rank == 0:
            args.logger("Start Continue Learning ......... :D ")
        mean_result_list = []
        std_result_list = []
        results = []
        all_best_acc = 0
        step_classes = len(self.nclass_list) // args.steps

        all_classes = list(range(self.nclass))
        for current_step in range(1, args.step + 1):
            classes_seen = random.sample(all_classes, current_step * step_classes)
            val_loader_step = get_loader_step(classes_seen, val_loader)
            syndataloader = get_loader_step(classes_seen, syndataloader)
            for i in range(args.val_repeat):
                args.logger(
                    f"======================Repeat {i+1}/{args.val_repeat} Starting =================================================================="
                )
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
                best_acc, acc = evaluate_syn_data(
                    args, model, syndataloader, val_loader_step, logger=args.logger
                )
                if all_best_acc < best_acc:
                    all_best_acc = best_acc
                results.append(best_acc)
                if args.rank == 0:
                    args.logger(
                        f"Step {current_step},Repeat {i+1}/{args.val_repeat} => The Best Evaluation Acc: {all_best_acc:.1f} The Last Evaluation Acc :{acc:.1f} \n"
                    )
            mean_result = np.mean(results)
            std_result = np.std(results)
            mean_result_list.append(mean_result)
            std_result_list.append(std_result)
        if args.rank == 0:
            args.logger("=" * 50)
            args.logger(
                f"All result: {[f'Step {i} Acc: {x:.3f}' for i, x in enumerate(mean_result_list)]}"
            )
            args.logger("=" * 50)


def get_loader_step(classes_seen, val_loader):
    val_data, val_targets = [], []

    for data, target in val_loader:
        mask = torch.tensor(
            [t.item() in classes_seen for t in target], device=target.device
        )
        val_data.append(data[mask])
        val_targets.append(target[mask])

    val_data = torch.cat(val_data)
    val_targets = torch.cat(val_targets)

    val_dataset_step = TensorDataset(val_data, val_targets)
    val_loader_step = DataLoader(val_dataset_step, batch_size=128, shuffle=False)
    return val_loader_step

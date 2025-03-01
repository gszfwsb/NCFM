import torch.distributed as dist
import os
import torch
from datetime import timedelta
from data.save_img import save_img
from collections import OrderedDict


def initialize_distribution_training(backend="nccl", init_method="env://"):
    dist.init_process_group(
        backend=backend, init_method=init_method, timeout=timedelta(seconds=3000)
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # Get local rank from environment variable
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["WORLD_SIZE"])
    # Set the current GPU for this process
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, local_rank, local_world_size, device


def distribute_class(nclass, debug=False):
    if debug:
        nclass = max(nclass // 100, 10)  # Reduce the number of classes for debugging
    classes_per_process = nclass // dist.get_world_size()  # Distribute classes evenly
    remainder = (
        nclass % dist.get_world_size()
    )  # Handle remainder for unequal distribution
    start_class = (
        dist.get_rank() * classes_per_process
    )  # Start class index for this rank
    end_class = start_class + classes_per_process  # End class index for this rank
    if dist.get_rank() == dist.get_world_size() - 1:
        end_class += remainder  # Add remainder to the last rank's class range
    class_list = list(range(start_class, end_class))  # List of classes for this rank
    for rank in range(dist.get_world_size()):
        if dist.get_rank() == rank:
            print(
                f"==========================Rank {dist.get_rank()} has classes {class_list}=========================="
            )
        else:
            dist.barrier()
    return class_list


def load_state_dict(state_dict_path, model):
    state_dict = torch.load(state_dict_path, map_location="cpu")
    # Remove `module.` prefix from keys if it exists
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")  # Remove 'module.' prefix
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)


def gather_save_visualize(synset, args, iteration=None):
    temp_save_dir = os.path.join(
        args.save_dir, "temp_rank_data"
    )  # Temporary directory to save rank data
    os.makedirs(temp_save_dir, exist_ok=True)
    save_iteration = (
        (iteration + 1) if iteration is not None else "init"
    )  # Set iteration name
    temp_file_path = os.path.join(
        temp_save_dir, f"temp_rank_{args.rank}_{save_iteration}.pt"
    )
    torch.save(
        [synset.data.detach().cpu(), synset.targets.cpu()], temp_file_path
    )  # Save data and targets for this rank
    dist.barrier()  # Synchronize all processes
    if args.rank == 0:
        all_data = []
        all_targets = []
        for r in range(args.world_size):
            temp_file_path = os.path.join(
                temp_save_dir, f"temp_rank_{r}_{save_iteration}.pt"
            )
            data, targets = torch.load(temp_file_path)  # Load data from all ranks
            all_data.append(data)
            all_targets.append(targets)
        all_data = torch.cat(all_data, dim=0)  # Concatenate data from all ranks
        all_targets = torch.cat(
            all_targets, dim=0
        )  # Concatenate targets from all ranks
        args.logger(f"the shape of saved data {all_data.shape}")
        args.logger(f"the shape of saved target {all_targets.shape}")
        os.makedirs(args.save_dir, exist_ok=True)
        save_img(
            os.path.join(args.save_dir, "images", f"img_{save_iteration}.png"),
            all_data,
            unnormalize=False,
            dataname=args.dataset,
        )  # Save images
        data_save_path = os.path.join(
            args.save_dir, "distilled_data", f"data_{save_iteration}.pt"
        )
        torch.save(
            [all_data, all_targets], data_save_path
        )  # Save concatenated data and targets
        args.logger(f"All data saved at iteration {save_iteration}.")
        # Clean up temporary directory
        for r in range(args.world_size):
            temp_file_path = os.path.join(
                temp_save_dir, f"temp_rank_{r}_{save_iteration}.pt"
            )
            os.remove(temp_file_path)  # Remove temporary files
        os.rmdir(temp_save_dir)  # Remove the temporary directory
    else:
        pass


def sync_distributed_metric(metric):
    device = torch.device(
        f"cuda:{dist.get_rank()}" if torch.cuda.is_available() else "cpu"
    )
    if isinstance(metric, list):
        # Convert metric to tensor if it isn't already
        metric_tensors = [
            torch.tensor(m, device=device) if not isinstance(m, torch.Tensor) else m
            for m in metric
        ]
        # Use all_reduce to synchronize each tensor across ranks
        for m in metric_tensors:
            dist.all_reduce(m, op=dist.ReduceOp.SUM)
        # Return average for each metric
        return [m.item() / dist.get_world_size() for m in metric_tensors]
    else:
        # Single metric
        if not isinstance(metric, torch.Tensor):
            metric = torch.tensor(metric, device=device)
        # Use all_reduce to synchronize the metric
        dist.all_reduce(metric, op=dist.ReduceOp.SUM)
        # Return the average value
        return metric.item() / dist.get_world_size()

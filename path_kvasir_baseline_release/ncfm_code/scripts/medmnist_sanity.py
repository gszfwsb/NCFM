import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.medmnist import get_medmnist_nch, get_medmnist_nclass, get_medmnist_root, register_medmnist_stats
from utils.utils import load_resized_data


def main():
    parser = argparse.ArgumentParser(description="Check MedMNIST data pipeline.")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--size", type=int, default=28)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    for dataset_name in args.datasets:
        root = get_medmnist_root(args.data_dir)
        mean, std = register_medmnist_stats(dataset_name, root, size=args.size)
        train_set, test_set = load_resized_data(
            dataset_name,
            args.data_dir,
            size=args.size,
            nclass=get_medmnist_nclass(dataset_name),
            load_memory=False,
        )
        loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
        data, target = next(iter(loader))
        print(f"DATASET={dataset_name}")
        print(f"  nclass={get_medmnist_nclass(dataset_name)} nch={get_medmnist_nch(dataset_name)}")
        print(f"  train_len={len(train_set)} test_len={len(test_set)}")
        print(f"  batch_shape={tuple(data.shape)} target_shape={tuple(target.shape)}")
        print(f"  target_dtype={target.dtype} target_values={target[:min(8, len(target))].tolist()}")
        print(f"  mean={mean} std={std}")
        print(f"  tensor_min={float(data.min()):.6f} tensor_max={float(data.max()):.6f}")


if __name__ == "__main__":
    main()

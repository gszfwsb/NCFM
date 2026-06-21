import argparse
import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.convnet import ConvNet


def check_case(name, channel, size, nclass, depth, width, device):
    model = ConvNet(
        num_classes=nclass,
        net_norm="instance",
        net_depth=depth,
        net_width=width,
        channel=channel,
        im_size=(size, size),
    ).to(device)
    x = torch.randn(2, channel, size, size, device=device)
    logits, features = model(x, return_features=True)
    print(f"{name}: ok logits={tuple(logits.shape)} features={tuple(features.shape)}")


def main():
    parser = argparse.ArgumentParser(description="Check ConvNet input-size compatibility.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--width", type=int, default=128)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    check_case("gray28", 1, 28, 2, args.depth, args.width, device)
    check_case("rgb28", 3, 28, 8, args.depth, args.width, device)
    check_case("rgb32", 3, 32, 8, args.depth, args.width, device)


if __name__ == "__main__":
    main()

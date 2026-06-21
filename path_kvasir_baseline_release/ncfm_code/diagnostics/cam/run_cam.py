import argparse
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.dirname(__file__))

from data.dataset_statistics import MEANS, STDS
from data.medmnist import get_medmnist_root, is_supported_medmnist, register_medmnist_stats
from cam_utils import (
    GradCAM,
    cam_entropy,
    load_plain_state_dict,
    resolve_target_layer,
    save_cam_triplet,
    tensor_to_uint8_image,
    topk_activation_ratio,
    write_summary_csv,
)
from utils.utils import define_model, load_resized_data


def build_args():
    parser = argparse.ArgumentParser(description="Run Grad-CAM on a trained evaluator.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--split", default="test", choices=["test"])
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--target_layer", default="auto")
    parser.add_argument("--net_type", default="convnet")
    parser.add_argument("--norm_type", default="instance")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--width", type=float, default=1.0)
    parser.add_argument("--nclass", type=int, required=True)
    parser.add_argument("--nch", type=int, required=True)
    parser.add_argument("--size", type=int, default=28)
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def main():
    args = build_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_supported_medmnist(args.dataset):
        register_medmnist_stats(args.dataset, get_medmnist_root(args.data_dir), size=args.size)

    _, dataset = load_resized_data(
        args.dataset,
        args.data_dir,
        size=args.size,
        nclass=args.nclass,
        load_memory=False,
    )
    model = define_model(
        args.dataset,
        args.norm_type,
        args.net_type,
        args.nch,
        args.depth,
        args.width,
        args.nclass,
        logger=None,
        size=args.size,
    ).to(device)
    model.load_state_dict(load_plain_state_dict(args.checkpoint))
    model.eval()

    target_layer = resolve_target_layer(model, args.target_layer)
    grad_cam = GradCAM(model, target_layer)
    rows = []
    mean, std = MEANS[args.dataset], STDS[args.dataset]
    count = min(args.num_samples, len(dataset))

    try:
        for index in range(count):
            image, target = dataset[index]
            y_true = int(target)
            image_batch = image.unsqueeze(0).to(device)
            cam, logits, probs = grad_cam(image_batch)
            y_pred = int(probs.argmax(dim=1).item())
            confidence = float(probs[0, y_pred].item())
            image_uint8 = tensor_to_uint8_image(image, mean, std)
            original_path, cam_path, overlay_path = save_cam_triplet(
                args.save_dir, index, image_uint8, cam
            )
            rows.append(
                {
                    "index": index,
                    "split": args.split,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "confidence": confidence,
                    "correct": int(y_true == y_pred),
                    "cam_entropy": cam_entropy(cam),
                    "topk_activation_ratio": topk_activation_ratio(cam),
                    "original_path": original_path,
                    "cam_path": cam_path,
                    "overlay_path": overlay_path,
                }
            )
    finally:
        grad_cam.close()

    summary_path = os.path.join(args.save_dir, "summary.csv")
    write_summary_csv(summary_path, rows)
    accuracy = sum(row["correct"] for row in rows) / max(len(rows), 1)
    print(f"Saved CAM summary: {summary_path}")
    print(f"CAM sample accuracy: {accuracy:.4f} ({sum(row['correct'] for row in rows)}/{len(rows)})")


if __name__ == "__main__":
    main()

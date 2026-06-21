import csv
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def resolve_target_layer(model, target_layer="auto"):
    if target_layer == "auto":
        if hasattr(model, "layers") and "conv" in model.layers:
            return model.layers["conv"][-1]
        if hasattr(model, "layer4"):
            return model.layer4[-1]
        raise ValueError("Cannot resolve target layer automatically for this model.")

    module = model
    for part in target_layer.split("."):
        if part == "":
            continue
        if isinstance(module, (torch.nn.ModuleList, list, tuple)):
            module = module[int(part)]
        elif isinstance(module, torch.nn.ModuleDict):
            module = module[part]
        elif part.lstrip("-").isdigit() and hasattr(module, "__getitem__"):
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def load_plain_state_dict(path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    clean = OrderedDict()
    for key, value in state.items():
        clean[key.replace("module.", "")] = value
    return clean


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handles = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def close(self):
        for handle in self.handles:
            handle.remove()

    def __call__(self, image, target_class=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        probs = torch.softmax(logits, dim=1)
        if target_class is None:
            target_class = int(probs.argmax(dim=1).item())
        score = logits[:, target_class].sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max().clamp_min(1e-8)
        return cam.detach().cpu().numpy(), logits.detach(), probs.detach()


def tensor_to_uint8_image(tensor, mean, std):
    x = tensor.detach().cpu().float().clone()
    mean = torch.tensor(mean, dtype=x.dtype).view(-1, 1, 1)
    std = torch.tensor(std, dtype=x.dtype).view(-1, 1, 1)
    x = x * std + mean
    x = x.clamp(0, 1)
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    x = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return x


def make_heatmap(cam):
    cam = np.clip(cam, 0, 1)
    heatmap = np.zeros((*cam.shape, 3), dtype=np.float32)
    heatmap[..., 0] = cam
    heatmap[..., 1] = 0.35 * cam
    heatmap[..., 2] = 0.05 * cam
    return (heatmap * 255.0).round().astype(np.uint8)


def overlay_cam(image_uint8, heatmap_uint8, alpha=0.45):
    out = (1 - alpha) * image_uint8.astype(np.float32) + alpha * heatmap_uint8.astype(np.float32)
    return np.clip(out, 0, 255).round().astype(np.uint8)


def cam_entropy(cam):
    values = cam.astype(np.float64).reshape(-1)
    values = values / max(values.sum(), 1e-12)
    entropy = -(values * np.log(values + 1e-12)).sum()
    return float(entropy / np.log(values.size))


def topk_activation_ratio(cam, top_fraction=0.1):
    values = np.sort(cam.reshape(-1))[::-1]
    k = max(1, int(round(values.size * top_fraction)))
    return float(values[:k].sum() / max(values.sum(), 1e-12))


def save_cam_triplet(save_root, index, image_uint8, cam):
    original_dir = os.path.join(save_root, "images")
    heatmap_dir = os.path.join(save_root, "heatmaps")
    overlay_dir = os.path.join(save_root, "overlays")
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(heatmap_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    heatmap = make_heatmap(cam)
    overlay = overlay_cam(image_uint8, heatmap)
    original_path = os.path.join(original_dir, f"original_{index:05d}.png")
    cam_path = os.path.join(heatmap_dir, f"cam_{index:05d}.png")
    overlay_path = os.path.join(overlay_dir, f"overlay_{index:05d}.png")
    Image.fromarray(image_uint8).save(original_path)
    Image.fromarray(heatmap).save(cam_path)
    Image.fromarray(overlay).save(overlay_path)
    return original_path, cam_path, overlay_path


def write_summary_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "index",
        "split",
        "y_true",
        "y_pred",
        "confidence",
        "correct",
        "cam_entropy",
        "topk_activation_ratio",
        "original_path",
        "cam_path",
        "overlay_path",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

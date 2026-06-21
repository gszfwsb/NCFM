import json
from pathlib import Path


NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip() + "\n"}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.rstrip() + "\n",
    }


COMMON_SETUP = r'''
from pathlib import Path
import json
import math
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 160)

EXP_ROOT = Path("/data/zengqiang/experiments/ncfm_medmnist_ablation_20260519")

def require_exp_root():
    if not EXP_ROOT.exists():
        raise FileNotFoundError(
            f"EXP_ROOT not found: {EXP_ROOT}. "
            "Edit EXP_ROOT in the first code cell to your experiment directory."
        )

def ensure_report_dir(*parts):
    path = EXP_ROOT / "reports" / "cam" / Path(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_group(name):
    if name == "real_train":
        return "real_train"
    if name.startswith("ipc10_"):
        return name[len("ipc10_"):]
    return name

def resolve_result_path(value):
    p = Path(str(value))
    if p.exists():
        return p
    if str(value).startswith("/"):
        return p
    q = EXP_ROOT / value
    return q

def load_eval_metrics():
    require_exp_root()
    rows = []
    for path in sorted((EXP_ROOT / "runs").glob("*/ipc10/*/eval_metrics_best.json")):
        item = read_json(path)
        item["dataset"] = path.parents[2].name
        item["group"] = path.parent.name
        item["metrics_path"] = str(path)
        rows.append(item)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    order = {"A_pure_ncfd_wopsi": 0, "B_minmax_ncfm_psi": 1, "C_code_default_enhanced": 2}
    df["_order"] = df["group"].map(order).fillna(99)
    return df.sort_values(["dataset", "_order"]).drop(columns=["_order"])

def load_cam_summaries():
    require_exp_root()
    rows = []
    for path in sorted((EXP_ROOT / "results" / "cam").glob("*/*/summary.csv")):
        dataset = path.parents[1].name
        group = normalize_group(path.parent.name)
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["dataset"] = dataset
        df["group"] = group
        df["summary_path"] = str(path)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    for col in ["index", "y_true", "y_pred", "confidence", "correct", "cam_entropy", "topk_activation_ratio"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out

def cam_group_summary(cam_df):
    if cam_df.empty:
        return cam_df
    grouped = (
        cam_df.groupby(["dataset", "group"], as_index=False)
        .agg(
            n=("index", "count"),
            cam_acc=("correct", "mean"),
            mean_confidence=("confidence", "mean"),
            mean_entropy=("cam_entropy", "mean"),
            mean_top10_mass=("topk_activation_ratio", "mean"),
            correct_entropy=("cam_entropy", lambda s: s[cam_df.loc[s.index, "correct"] == 1].mean()),
            wrong_entropy=("cam_entropy", lambda s: s[cam_df.loc[s.index, "correct"] == 0].mean()),
            correct_top10_mass=("topk_activation_ratio", lambda s: s[cam_df.loc[s.index, "correct"] == 1].mean()),
            wrong_top10_mass=("topk_activation_ratio", lambda s: s[cam_df.loc[s.index, "correct"] == 0].mean()),
        )
    )
    order = {"real_train": 0, "A_pure_ncfd_wopsi": 1, "B_minmax_ncfm_psi": 2, "C_code_default_enhanced": 3}
    grouped["_order"] = grouped["group"].map(order).fillna(99)
    return grouped.sort_values(["dataset", "_order"]).drop(columns=["_order"])
'''


NOTEBOOKS = {
    "00_formal_metrics_overview.ipynb": [
        md(
            """
            # Formal Metrics Overview

            读取 formal A/B/C 的 `eval_metrics_best.json` 和 pretrain metrics，回答：

            - 强 pretrain 的 ceiling 是否正常？
            - A/B/C 在每个 MedMNIST 数据集上的 ACC/AUC/Macro-F1/Balanced ACC 如何？
            - 是否存在 ACC 高但 Macro-F1 或 Balanced ACC 低的类别不均衡风险？
            """
        ),
        code(COMMON_SETUP),
        code(
            r'''
eval_df = load_eval_metrics()
display(eval_df[[
    "dataset", "group", "acc_percent", "auc_macro_ovr", "macro_f1",
    "balanced_acc", "sensitivity", "specificity", "auprc", "epoch", "checkpoint_path"
]] if not eval_df.empty else eval_df)

report_dir = ensure_report_dir()
if not eval_df.empty:
    eval_df.to_csv(report_dir / "formal_eval_metrics_all.csv", index=False)
'''
        ),
        code(
            r'''
metrics = ["acc_percent", "auc_macro_ovr", "macro_f1", "balanced_acc"]
if not eval_df.empty:
    for metric in metrics:
        pivot = eval_df.pivot_table(index="dataset", columns="group", values=metric, aggfunc="first")
        display(pivot)
        ax = pivot.plot(kind="bar", figsize=(9, 4), rot=0)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.show()
'''
        ),
        code(
            r'''
rows = []
for path in sorted((EXP_ROOT / "reports" / "pretrain").glob("*/metrics.jsonl")):
    dataset = path.parent.name
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            item["dataset"] = dataset
            rows.append(item)

pre_df = pd.DataFrame(rows)
if pre_df.empty:
    print("No pretrain metrics found.")
else:
    final_pre = pre_df.sort_values("epoch").groupby(["dataset", "model_id"], as_index=False).tail(1)
    pre_summary = final_pre.groupby("dataset").agg(
        models=("model_id", "nunique"),
        acc_mean=("acc_percent", "mean"),
        acc_std=("acc_percent", "std"),
        auc_mean=("auc_macro_ovr", "mean"),
        macro_f1_mean=("macro_f1", "mean"),
        balanced_acc_mean=("balanced_acc", "mean"),
    ).reset_index()
    display(pre_summary)
    pre_summary.to_csv(report_dir / "pretrain_final_epoch_summary.csv", index=False)
'''
        ),
        md(
            """
            ## 解读提示

            - `pretrain_final_epoch_summary.csv` 用来判断 feature extractor 是否可靠。
            - `formal_eval_metrics_all.csv` 是正式蒸馏指标总表。
            - 若某组 ACC 高但 Macro-F1 / Balanced ACC 低，应优先怀疑类别不均衡或只保留了容易类别的判别结构。
            """
        ),
    ],
    "01_cam_summary_spatial_bias.ipynb": [
        md(
            """
            # CAM Summary & Spatial Bias

            聚合所有 `results/cam/**/summary.csv`，并从 heatmap 图中估计空间偏置：

            - `edge_mass`: CAM 激活是否偏边缘
            - `center_mass`: 是否偏中心
            - `corner_mass`: 是否偏四角
            - `cam_cx/cam_cy`: 热点质心
            """
        ),
        code(COMMON_SETUP),
        code(
            r'''
cam_df = load_cam_summaries()
summary = cam_group_summary(cam_df)
display(summary)

report_dir = ensure_report_dir()
if not summary.empty:
    summary.to_csv(report_dir / "cam_summary_grouped.csv", index=False)
    cam_df.to_csv(report_dir / "cam_summary_all_rows.csv", index=False)
'''
        ),
        code(
            r'''
def read_cam_array(path_value):
    path = resolve_result_path(path_value)
    if not path.exists():
        return None
    img = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    # heatmap is saved as reddish RGB, red channel preserves the CAM intensity
    cam = img[..., 0]
    total = cam.sum()
    if total <= 1e-12:
        return cam
    return cam

def spatial_stats(cam):
    h, w = cam.shape
    total = float(cam.sum()) + 1e-12
    edge = max(1, round(min(h, w) * 0.15))
    edge_mask = np.zeros((h, w), dtype=bool)
    edge_mask[:edge, :] = True
    edge_mask[-edge:, :] = True
    edge_mask[:, :edge] = True
    edge_mask[:, -edge:] = True
    center_mask = np.zeros((h, w), dtype=bool)
    y0, y1 = h // 4, h - h // 4
    x0, x1 = w // 4, w - w // 4
    center_mask[y0:y1, x0:x1] = True
    corner_mask = np.zeros((h, w), dtype=bool)
    corner = max(1, round(min(h, w) * 0.2))
    corner_mask[:corner, :corner] = True
    corner_mask[:corner, -corner:] = True
    corner_mask[-corner:, :corner] = True
    corner_mask[-corner:, -corner:] = True
    yy, xx = np.mgrid[0:h, 0:w]
    cx = float((cam * xx).sum() / total) / max(w - 1, 1)
    cy = float((cam * yy).sum() / total) / max(h - 1, 1)
    left = float(cam[:, : w // 2].sum()) / total
    top = float(cam[: h // 2, :].sum()) / total
    return {
        "edge_mass": float(cam[edge_mask].sum() / total),
        "center_mass": float(cam[center_mask].sum() / total),
        "corner_mass": float(cam[corner_mask].sum() / total),
        "cam_cx": cx,
        "cam_cy": cy,
        "left_mass": left,
        "right_mass": 1.0 - left,
        "top_mass": top,
        "bottom_mass": 1.0 - top,
    }

rows = []
for _, row in cam_df.iterrows():
    cam = read_cam_array(row["cam_path"])
    if cam is None:
        continue
    rows.append({**row.to_dict(), **spatial_stats(cam)})

spatial_df = pd.DataFrame(rows)
if spatial_df.empty:
    print("No CAM heatmaps found. Run this notebook on the lab server or set EXP_ROOT correctly.")
else:
    spatial_grouped = spatial_df.groupby(["dataset", "group"], as_index=False).agg(
        n=("index", "count"),
        edge_mass=("edge_mass", "mean"),
        center_mass=("center_mass", "mean"),
        corner_mass=("corner_mass", "mean"),
        cam_cx=("cam_cx", "mean"),
        cam_cy=("cam_cy", "mean"),
        left_mass=("left_mass", "mean"),
        top_mass=("top_mass", "mean"),
    )
    display(spatial_grouped)
    spatial_df.to_csv(report_dir / "cam_spatial_bias_all_rows.csv", index=False)
    spatial_grouped.to_csv(report_dir / "cam_spatial_bias_grouped.csv", index=False)
'''
        ),
        code(
            r'''
if "spatial_grouped" in globals() and not spatial_grouped.empty:
    for metric in ["edge_mass", "center_mass", "corner_mass"]:
        pivot = spatial_grouped.pivot_table(index="dataset", columns="group", values=metric, aggfunc="first")
        display(pivot)
        ax = pivot.plot(kind="bar", figsize=(9, 4), rot=0)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.show()
'''
        ),
        md(
            """
            ## 解读提示

            - `edge_mass` 高：模型可能依赖边框/背景伪特征。
            - `corner_mass` 高：可能存在固定角落或采样位置偏置。
            - `center_mass` 高不一定坏，医学主体常位于中心；需和 real-trained CAM 对比。
            """
        ),
    ],
    "02_cam_similarity_to_real.ipynb": [
        md(
            """
            # Real vs Synthetic CAM Similarity

            对同一 test sample，比对 `real_train` 和 A/B/C synthetic-trained evaluator 的 CAM 是否关注同一区域。

            指标：

            - Pearson correlation
            - Cosine similarity
            - Top-10% hot region IoU
            - Top-10% hot region Dice
            """
        ),
        code(COMMON_SETUP),
        code(
            r'''
cam_df = load_cam_summaries()
if cam_df.empty:
    raise RuntimeError("No CAM summaries found.")

def read_cam(path_value):
    path = resolve_result_path(path_value)
    if not path.exists():
        return None
    img = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    cam = img[..., 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-12)
    return cam

def sim_metrics(a, b, top_fraction=0.10):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    pearson = np.corrcoef(a, b)[0, 1] if np.std(a) > 1e-12 and np.std(b) > 1e-12 else np.nan
    cosine = float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))
    k = max(1, int(round(a.size * top_fraction)))
    a_hot = np.zeros_like(a, dtype=bool)
    b_hot = np.zeros_like(b, dtype=bool)
    a_hot[np.argsort(a)[-k:]] = True
    b_hot[np.argsort(b)[-k:]] = True
    inter = np.logical_and(a_hot, b_hot).sum()
    union = np.logical_or(a_hot, b_hot).sum()
    iou = float(inter / max(union, 1))
    dice = float(2 * inter / max(a_hot.sum() + b_hot.sum(), 1))
    return {"pearson": pearson, "cosine": cosine, "top10_iou": iou, "top10_dice": dice}

rows = []
for dataset, ddf in cam_df.groupby("dataset"):
    real = ddf[ddf["group"] == "real_train"].set_index("index")
    if real.empty:
        continue
    for group, gdf in ddf[ddf["group"] != "real_train"].groupby("group"):
        syn = gdf.set_index("index")
        for idx in sorted(set(real.index) & set(syn.index)):
            real_cam = read_cam(real.loc[idx, "cam_path"])
            syn_cam = read_cam(syn.loc[idx, "cam_path"])
            if real_cam is None or syn_cam is None:
                continue
            item = {
                "dataset": dataset,
                "group": group,
                "index": int(idx),
                "real_correct": int(real.loc[idx, "correct"]),
                "syn_correct": int(syn.loc[idx, "correct"]),
                "y_true": int(real.loc[idx, "y_true"]),
                "real_overlay": real.loc[idx, "overlay_path"],
                "syn_overlay": syn.loc[idx, "overlay_path"],
            }
            item.update(sim_metrics(real_cam, syn_cam))
            rows.append(item)

sim_df = pd.DataFrame(rows)
report_dir = ensure_report_dir()
if sim_df.empty:
    print("No comparable CAM pairs found.")
else:
    sim_grouped = sim_df.groupby(["dataset", "group"], as_index=False).agg(
        n=("index", "count"),
        pearson=("pearson", "mean"),
        cosine=("cosine", "mean"),
        top10_iou=("top10_iou", "mean"),
        top10_dice=("top10_dice", "mean"),
        both_correct=("syn_correct", "mean"),
    )
    display(sim_grouped)
    sim_df.to_csv(report_dir / "cam_similarity_to_real_all_rows.csv", index=False)
    sim_grouped.to_csv(report_dir / "cam_similarity_to_real_grouped.csv", index=False)
'''
        ),
        code(
            r'''
if "sim_grouped" in globals() and not sim_grouped.empty:
    for metric in ["pearson", "cosine", "top10_iou", "top10_dice"]:
        pivot = sim_grouped.pivot_table(index="dataset", columns="group", values=metric, aggfunc="first")
        display(pivot)
        ax = pivot.plot(kind="bar", figsize=(9, 4), rot=0)
        ax.set_title(f"Real-vs-synthetic CAM {metric}")
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.show()
'''
        ),
        code(
            r'''
def show_pair_gallery(dataset, group, n=6, sort_by="top10_iou", ascending=True):
    if sim_df.empty:
        print("No similarity rows.")
        return
    sub = sim_df[(sim_df.dataset == dataset) & (sim_df.group == group)].copy()
    if sub.empty:
        print("No rows for", dataset, group)
        return
    sub = sub.sort_values(sort_by, ascending=ascending).head(n)
    fig, axes = plt.subplots(len(sub), 2, figsize=(6, 3 * len(sub)))
    if len(sub) == 1:
        axes = np.array([axes])
    for axrow, (_, row) in zip(axes, sub.iterrows()):
        for ax, key, title in [(axrow[0], "real_overlay", "real"), (axrow[1], "syn_overlay", group)]:
            path = resolve_result_path(row[key])
            if path.exists():
                ax.imshow(Image.open(path))
            ax.set_title(f"{title} idx={row['index']} IoU={row['top10_iou']:.2f}")
            ax.axis("off")
    plt.tight_layout()
    plt.show()

# Example:
# show_pair_gallery("pneumoniamnist", "C_code_default_enhanced", n=6)
'''
        ),
        md(
            """
            ## 解读提示

            - 如果某组性能高但 similarity 低，说明它可能走了和 real-trained 不同的捷径。
            - 如果 similarity 高且 Balanced ACC/Macro-F1 也高，说明 synthetic data 更可能保住了真实判别结构。
            """
        ),
    ],
    "03_cam_gallery_and_class_prototypes.ipynb": [
        md(
            """
            # CAM Gallery & Class Prototypes

            这个 notebook 用于人工审阅：

            - 每组正确/错误样本 overlay 图册
            - 每个类别的平均 CAM prototype
            - correct-only 与 wrong-only 的平均 CAM 对照
            """
        ),
        code(COMMON_SETUP),
        code(
            r'''
cam_df = load_cam_summaries()
if cam_df.empty:
    raise RuntimeError("No CAM summaries found.")
display(cam_group_summary(cam_df))
'''
        ),
        code(
            r'''
def show_overlay_grid(dataset, group, correct=None, n=12, seed=0):
    sub = cam_df[(cam_df.dataset == dataset) & (cam_df.group == group)].copy()
    if correct is not None:
        sub = sub[sub.correct == int(correct)]
    if sub.empty:
        print("No rows for", dataset, group, correct)
        return
    sub = sub.sample(min(n, len(sub)), random_state=seed)
    cols = 4
    rows = math.ceil(len(sub) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, (_, row) in zip(axes, sub.iterrows()):
        path = resolve_result_path(row["overlay_path"])
        if path.exists():
            ax.imshow(Image.open(path))
        ax.set_title(f"idx={int(row['index'])} y={int(row['y_true'])} pred={int(row['y_pred'])} c={row['confidence']:.2f}")
        ax.axis("off")
    for ax in axes[len(sub):]:
        ax.axis("off")
    plt.suptitle(f"{dataset} / {group} / correct={correct}")
    plt.tight_layout()
    plt.show()

# Examples:
# show_overlay_grid("bloodmnist", "real_train", correct=1)
# show_overlay_grid("bloodmnist", "A_pure_ncfd_wopsi", correct=0)
'''
        ),
        code(
            r'''
def read_cam(path_value):
    path = resolve_result_path(path_value)
    if not path.exists():
        return None
    img = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    cam = img[..., 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-12)
    return cam

def build_prototype_table(dataset, group, correct=None):
    sub = cam_df[(cam_df.dataset == dataset) & (cam_df.group == group)].copy()
    if correct is not None:
        sub = sub[sub.correct == int(correct)]
    rows = []
    for cls, cdf in sub.groupby("y_true"):
        cams = []
        for path in cdf["cam_path"]:
            cam = read_cam(path)
            if cam is not None:
                cams.append(cam)
        if cams:
            rows.append((int(cls), np.mean(cams, axis=0), len(cams)))
    return rows

def show_class_prototypes(dataset, group, correct=None):
    protos = build_prototype_table(dataset, group, correct=correct)
    if not protos:
        print("No prototypes for", dataset, group)
        return
    cols = min(5, len(protos))
    rows = math.ceil(len(protos) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, (cls, cam, n) in zip(axes, protos):
        im = ax.imshow(cam, cmap="inferno", vmin=0, vmax=1)
        ax.set_title(f"class {cls}, n={n}")
        ax.axis("off")
    for ax in axes[len(protos):]:
        ax.axis("off")
    plt.suptitle(f"Prototype CAM: {dataset} / {group} / correct={correct}")
    plt.tight_layout()
    plt.show()

# Examples:
# show_class_prototypes("pathmnist", "real_train", correct=1)
# show_class_prototypes("pathmnist", "A_pure_ncfd_wopsi", correct=1)
'''
        ),
        md(
            """
            ## 审阅建议

            - 先看 `real_train` 的 class prototype，建立“真实模型长期看哪里”的参考。
            - 再看 A/B/C 的 prototype 是否偏背景、偏边缘、偏固定角落。
            - 错误样本单独看，通常最容易暴露伪特征。
            """
        ),
    ],
    "04_occlusion_sensitivity.ipynb": [
        md(
            """
            # Occlusion Sensitivity

            用遮挡验证 CAM 热区是否真的影响预测。

            比较：

            - 遮挡 CAM top-k hot region
            - 遮挡 random region
            - 遮挡 cold region

            如果 hot region 遮挡带来的 confidence/probability drop 明显更大，说明 CAM 热区具有更强判别贡献。
            """
        ),
        code(
            COMMON_SETUP
            + r'''
import sys
REPO_ROOT = EXP_ROOT / "src" / "NCFM_medmnist_clean"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader, Subset
from utils.utils import define_model
from utils.ddp import load_state_dict
from data.medmnist import get_medmnist_root, build_medmnist_dataset, get_medmnist_nclass, get_medmnist_nch, register_medmnist_stats
from data.transform import transform_medmnist
from data.dataset_statistics import MEANS, STDS
'''
        ),
        code(
            r'''
def get_checkpoint(dataset, group):
    if group == "real_train":
        return EXP_ROOT / "checkpoints" / "pretrain" / dataset / "premodel0_trained.pth.tar"
    return EXP_ROOT / "checkpoints" / "synthetic_train" / dataset / f"ipc10_{group}_best.pth.tar"

def load_model(dataset, group, device="cuda"):
    nch = get_medmnist_nch(dataset)
    nclass = get_medmnist_nclass(dataset)
    model = define_model(dataset, "instance", "convnet", nch, 3, 1.0, nclass, print, 28).to(device)
    load_state_dict(str(get_checkpoint(dataset, group)), model)
    model.eval()
    return model

def load_test_dataset(dataset):
    root = get_medmnist_root(EXP_ROOT / "data")
    register_medmnist_stats(dataset, root, 28)
    _, test_transform = transform_medmnist(dataset, size=28, augment=False, normalize=True)
    return build_medmnist_dataset(dataset, root, split="test", transform=test_transform, size=28, download=True)

def read_cam_mask(path_value, mode="hot", fraction=0.10):
    path = resolve_result_path(path_value)
    img = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    cam = img[..., 0]
    values = cam.reshape(-1)
    k = max(1, int(round(values.size * fraction)))
    if mode == "hot":
        idx = np.argsort(values)[-k:]
    elif mode == "cold":
        idx = np.argsort(values)[:k]
    else:
        rng = np.random.default_rng(0)
        idx = rng.choice(values.size, size=k, replace=False)
    mask = np.zeros(values.size, dtype=bool)
    mask[idx] = True
    return mask.reshape(cam.shape)

def apply_mask_mean_fill(x, mask):
    # Inputs are normalized, so zero corresponds approximately to channel mean.
    out = x.clone()
    m = torch.tensor(mask, dtype=torch.bool, device=x.device)
    out[:, m] = 0.0
    return out

@torch.no_grad()
def occlusion_rows(dataset, group, max_samples=100, fraction=0.10, device="cuda"):
    cam_df = load_cam_summaries()
    sub = cam_df[(cam_df.dataset == dataset) & (cam_df.group == group)].copy().head(max_samples)
    ds = load_test_dataset(dataset)
    model = load_model(dataset, group, device=device)
    rows = []
    for _, row in sub.iterrows():
        idx = int(row["index"])
        x, y = ds[idx]
        x = x.to(device)
        y = int(y)
        logits = model(x.unsqueeze(0))
        probs = torch.softmax(logits, dim=1)[0]
        pred = int(probs.argmax().item())
        base_prob = float(probs[pred].item())
        item = {"dataset": dataset, "group": group, "index": idx, "y_true": y, "pred": pred, "base_prob": base_prob}
        for mode in ["hot", "random", "cold"]:
            mask = read_cam_mask(row["cam_path"], mode=mode, fraction=fraction)
            x_occ = apply_mask_mean_fill(x, mask)
            p_occ = torch.softmax(model(x_occ.unsqueeze(0)), dim=1)[0]
            item[f"{mode}_prob"] = float(p_occ[pred].item())
            item[f"{mode}_drop"] = base_prob - item[f"{mode}_prob"]
        rows.append(item)
    return pd.DataFrame(rows)

# Example:
# occ = occlusion_rows("pneumoniamnist", "C_code_default_enhanced", max_samples=50)
# display(occ.describe())
'''
        ),
        code(
            r'''
def summarize_occlusion(occ):
    cols = ["hot_drop", "random_drop", "cold_drop"]
    out = occ.groupby(["dataset", "group"])[cols].agg(["mean", "std", "count"])
    display(out)
    ax = occ[cols].mean().plot(kind="bar", figsize=(6, 4))
    ax.set_title("Mean probability drop after occlusion")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()

# Example:
# summarize_occlusion(occ)
'''
        ),
        md(
            """
            ## 解读提示

            - `hot_drop >> random_drop/cold_drop`：CAM 热区确实更关键。
            - `hot_drop` 不明显：CAM 可能只是相关而非因果，或者模型依赖更分散的结构。
            - 建议先在每个 dataset 的 best group 跑 50-100 张样本做 sanity check。
            """
        ),
    ],
}


def main():
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        path = NOTEBOOK_DIR / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb(cells), f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(path)


if __name__ == "__main__":
    main()

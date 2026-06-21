import json
import os
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)


def _safe_float(value):
    if value is None:
        return None
    value = float(value)
    if np.isnan(value) or np.isinf(value):
        return None
    return value


def _all_gather_numpy(array: np.ndarray) -> np.ndarray:
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return array
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, array)
    return np.concatenate(gathered, axis=0)


def classification_metrics_from_logits(
    logits: np.ndarray,
    targets: np.ndarray,
    nclass: Optional[int] = None,
) -> Dict[str, Any]:
    targets = np.asarray(targets).reshape(-1).astype(int)
    logits = np.asarray(logits)
    if nclass is None:
        nclass = int(logits.shape[1])

    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    preds = probs.argmax(axis=1)
    labels = np.arange(nclass)

    metrics: Dict[str, Any] = {
        "n": int(targets.shape[0]),
        "nclass": int(nclass),
        "acc": _safe_float(accuracy_score(targets, preds)),
        "acc_percent": _safe_float(100.0 * accuracy_score(targets, preds)),
        "macro_f1": _safe_float(f1_score(targets, preds, average="macro", zero_division=0)),
        "balanced_acc": _safe_float(balanced_accuracy_score(targets, preds)),
        "per_class_recall": [
            _safe_float(v)
            for v in recall_score(
                targets, preds, labels=labels, average=None, zero_division=0
            )
        ],
        "confusion_matrix": confusion_matrix(targets, preds, labels=labels).tolist(),
    }

    try:
        if nclass == 2:
            metrics["auc_macro_ovr"] = _safe_float(roc_auc_score(targets, probs[:, 1]))
        else:
            metrics["auc_macro_ovr"] = _safe_float(
                roc_auc_score(
                    targets,
                    probs,
                    labels=labels,
                    multi_class="ovr",
                    average="macro",
                )
            )
    except ValueError:
        metrics["auc_macro_ovr"] = None

    if nclass == 2:
        cm = confusion_matrix(targets, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        metrics["sensitivity"] = _safe_float(tp / (tp + fn)) if (tp + fn) else None
        metrics["specificity"] = _safe_float(tn / (tn + fp)) if (tn + fp) else None
        try:
            metrics["auprc"] = _safe_float(average_precision_score(targets, probs[:, 1]))
        except ValueError:
            metrics["auprc"] = None

    return metrics


@torch.no_grad()
def evaluate_loader_metrics(
    model,
    loader,
    criterion=None,
    nclass: Optional[int] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    model.eval()
    logits_list = []
    targets_list = []
    losses = []
    weights = []

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        if criterion is not None:
            loss = criterion(outputs, targets)
            losses.append(float(loss.item()))
            weights.append(int(inputs.size(0)))
        logits_list.append(outputs.detach().cpu())
        targets_list.append(targets.detach().cpu())

    logits = torch.cat(logits_list, dim=0).numpy()
    targets = torch.cat(targets_list, dim=0).numpy()
    logits = _all_gather_numpy(logits)
    targets = _all_gather_numpy(targets)

    metrics = classification_metrics_from_logits(logits, targets, nclass=nclass)
    if losses:
        total_weight = float(np.sum(weights))
        loss_sum = float(np.sum(np.asarray(losses) * np.asarray(weights)))
        local_loss = np.array([loss_sum, total_weight], dtype=np.float64)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            tensor = torch.tensor(local_loss, device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            total_weight = float(tensor[1].item())
            metrics["loss"] = _safe_float(tensor[0].item() / total_weight)
            metrics["loss_weight"] = total_weight
        else:
            metrics["loss"] = _safe_float(local_loss[0] / local_loss[1])
            metrics["loss_weight"] = float(local_loss[1])
    return metrics


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: str, record: Dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def format_metrics(metrics: Dict[str, Any]) -> str:
    parts = [
        f"ACC={metrics.get('acc_percent', 0.0):.2f}",
        f"AUC={metrics.get('auc_macro_ovr'):.4f}" if metrics.get("auc_macro_ovr") is not None else "AUC=NA",
        f"MacroF1={metrics.get('macro_f1'):.4f}" if metrics.get("macro_f1") is not None else "MacroF1=NA",
        f"BalAcc={metrics.get('balanced_acc'):.4f}" if metrics.get("balanced_acc") is not None else "BalAcc=NA",
    ]
    if "sensitivity" in metrics:
        parts.append(
            f"Sens={metrics.get('sensitivity'):.4f}"
            if metrics.get("sensitivity") is not None
            else "Sens=NA"
        )
        parts.append(
            f"Spec={metrics.get('specificity'):.4f}"
            if metrics.get("specificity") is not None
            else "Spec=NA"
        )
        parts.append(
            f"AUPRC={metrics.get('auprc'):.4f}" if metrics.get("auprc") is not None else "AUPRC=NA"
        )
    return " ".join(parts)

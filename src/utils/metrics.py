"""
Evaluation metrics for IDS performance.
"""

import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)


def compute_binary_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute standard binary classification metrics."""
    # Binarise: 0 = benign, 1 = attack
    y_true_bin = (y_true != 0).astype(int)
    y_pred_bin = (y_pred != 0).astype(int)

    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        # Edge case: only one class present in predictions or labels
        # This can happen in extreme cases (model collapse to single class)
        tn, fp, fn, tp = 0, 0, 0, 0
        if cm.shape == (1, 1):
            # Only negatives present in both
            tn = cm[0, 0]
        elif cm.shape == (1, 2):
            tn, fp = cm[0, 0], cm[0, 1]
        elif cm.shape == (2, 1):
            fn, tp = cm[0, 0], cm[1, 0]

    total = tn + fp + fn + tp
    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    fpr = fp / max(fp + tn, 1) if (fp + tn) > 0 else (1.0 if fp > 0 else 0.0)

    # Sanity-check diagnostics: detect logically impossible combinations
    # These warn but do not crash — the metrics are still returned
    import warnings as _warnings
    n_benign = tn + fp
    n_attack = fn + tp
    if fp > 0 and tn == 0:
        # Every benign is misclassified — FPR = 1.0 is correct but extreme
        _warnings.warn(
            f"[compute_binary_metrics] FPR=1.0 detected: all {n_benign} benign "
            f"samples misclassified as attacks. Verify this is intentional "
            f"(e.g., model predicting only attacks)."
        )
    if abs(precision - 1.0) < 1e-6 and fp > 0:
        _warnings.warn(
            f"[compute_binary_metrics] precision≈1.0 with fp={fp} — "
            f"this is mathematically inconsistent. Check confusion matrix: "
            f"cm={cm.tolist()}"
        )
    if n_benign == 0:
        _warnings.warn(
            f"[compute_binary_metrics] No benign (class 0) samples in test set. "
            f"Binary FPR/precision are undefined. Treat with caution."
        )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "fpr": fpr,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """Compute multi-class metrics with full per-class breakdown."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    # Task 6: Add macro-averaged F1 (equal weight per class — key for imbalanced IDS)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    # Task 6: Add per-class recall (attack-class recall is the primary IDS metric)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)

    result = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "f1_macro": f1_macro,           # Task 6: macro F1 — equal weight per class
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
    }

    # Per-class breakdown (Task 6)
    if class_names:
        report = classification_report(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        result["per_class"] = report

    # Task 6: Per-class recall array (index = class index)
    result["recall_per_class"] = {
        c: float(recall_per_class[c]) for c in range(len(recall_per_class))
    }

    return result


def compute_auc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    num_classes: int,
) -> float:
    """Compute AUC-ROC (binary or one-vs-rest)."""
    try:
        if num_classes == 2:
            return roc_auc_score(y_true, y_scores)
        else:
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(y_true, classes=list(range(num_classes)))
            if y_scores.ndim == 1:
                return 0.0
            return roc_auc_score(y_bin, y_scores, multi_class="ovr", average="weighted")
    except Exception:
        return 0.0


def print_metrics(metrics: Dict, prefix: str = ""):
    """Pretty print metrics."""
    print(f"\n{'='*60}")
    if prefix:
        print(f"  {prefix}")
        print(f"{'='*60}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")
        elif isinstance(v, int):
            print(f"  {k:25s}: {v}")
        elif isinstance(v, dict):
            pass  # skip nested dicts in summary
    print(f"{'='*60}\n")

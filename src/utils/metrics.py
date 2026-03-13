"""
Evaluation Metrics Module.
============================================================================
Paper Reference: Section 6, Page 8
============================================================================
"Standard machine learning metrics are used to evaluate the model performance
 of our proposed system, such as accuracy, precision, F1-Score, recall,
 Area under ROC curve (AUC) and false positive rate (FPR)." (Page 8)

Metrics definitions (Page 8):
- Accuracy = (TP + TN) / (TP + FP + TN + FN)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- FPR = FP / (FP + TN)
- F1-Score = 2 * Precision * Recall / (Precision + Recall)
- AUC-ROC: Area under the ROC curve
============================================================================
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)


def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Compute all evaluation metrics from the paper.

    Paper Reference: Section 6, Page 8
    All metrics are computed as described in the paper.

    Args:
        y_true: Ground truth labels (0=Normal, 1=Attack)
        y_pred: Predicted labels
        y_prob: Predicted probabilities for AUC computation (optional)
    Returns:
        dict with all metrics
    """
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Accuracy (Page 8): (TP + TN) / (TP + FP + TN + FN)
    accuracy = accuracy_score(y_true, y_pred)

    # Precision (Page 8): TP / (TP + FP)
    precision = precision_score(y_true, y_pred, zero_division=0)

    # Recall (Page 8): TP / (TP + FN)
    recall = recall_score(y_true, y_pred, zero_division=0)

    # FPR (Page 8): FP / (FP + TN)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # F1-Score (Page 8): 2 * Precision * Recall / (Precision + Recall)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # AUC-ROC (Page 8): Area under ROC curve
    auc = 0.0
    if y_prob is not None and len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'fpr': fpr,
        'f1_score': f1,
        'auc_roc': auc,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
    }

    return metrics


def print_metrics(metrics, agent_name=""):
    """Pretty print metrics."""
    prefix = f"[{agent_name}] " if agent_name else ""
    print(f"{prefix}Accuracy:  {metrics['accuracy']:.4f}")
    print(f"{prefix}Precision: {metrics['precision']:.4f}")
    print(f"{prefix}Recall:    {metrics['recall']:.4f}")
    print(f"{prefix}FPR:       {metrics['fpr']:.4f}")
    print(f"{prefix}F1-Score:  {metrics['f1_score']:.4f}")
    print(f"{prefix}AUC-ROC:   {metrics['auc_roc']:.4f}")


def compute_roc_data(y_true, y_prob):
    """
    Compute ROC curve data points.

    Paper Reference: Section 6, Page 8; Figure 7, Page 11
    "ROC Curve: It is a curve that is drawn with false positive rate (FPR)
     on x-axis and true positive rate (TPR) on the y-axis."

    Returns:
        fpr_arr, tpr_arr, thresholds
    """
    if y_prob is not None and len(np.unique(y_true)) > 1:
        return roc_curve(y_true, y_prob)
    return np.array([0, 1]), np.array([0, 1]), np.array([1, 0])

"""
Evaluation script: Load a trained model and evaluate on test data.
Produces detailed per-class metrics, confusion matrix, ROC curves.
"""

import os
import json
import torch
import numpy as np
import argparse
from collections import OrderedDict

from src.config import Config
from src.data.preprocessor import load_dataset
from src.agents.ppo_agent import PPOAgent
from src.utils.metrics import (
    compute_binary_metrics,
    compute_multiclass_metrics,
    compute_auc,
    print_metrics,
)


def evaluate_model(
    model_path: str,
    cfg: Config,
):
    """Load and evaluate a trained model."""
    device = torch.device(
        cfg.training.device if torch.cuda.is_available() else "cpu"
    )

    # Load data
    X_train, X_test, y_train, y_test, le = load_dataset(cfg)
    num_classes = len(le.classes_)
    state_dim = X_test.shape[1]

    # Load model
    model_state = torch.load(model_path, map_location=device, weights_only=True)

    # Create agent for evaluation
    eval_agent = PPOAgent(
        state_dim=state_dim,
        action_dim=num_classes,
        cfg=cfg.ppo,
        device=device,
    )
    eval_agent.set_model_state(model_state)

    # Run predictions
    y_pred = []
    y_scores = []

    for i in range(len(X_test)):
        state = X_test[i].astype(np.float32)
        action_idx, _, _ = eval_agent.select_action(state, deterministic=True)
        # select_action returns an int class index (not one-hot), so use directly
        predicted_class = int(action_idx)
        y_pred.append(predicted_class)
        y_scores.append(action_idx)

    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    # Binary metrics
    binary_metrics = compute_binary_metrics(y_test, y_pred)
    print_metrics(binary_metrics, prefix="Binary Classification Metrics")

    # Multi-class metrics
    multi_metrics = compute_multiclass_metrics(
        y_test, y_pred, class_names=list(le.classes_)
    )
    print_metrics(multi_metrics, prefix="Multi-class Metrics")

    # AUC
    auc = compute_auc(y_test, y_scores, num_classes)
    print(f"  AUC-ROC: {auc:.4f}")

    # Per-class details
    if "per_class" in multi_metrics:
        print("\nPer-class Performance:")
        print("-" * 60)
        for cls_name, cls_metrics in multi_metrics["per_class"].items():
            if isinstance(cls_metrics, dict) and "precision" in cls_metrics:
                print(
                    f"  {cls_name:25s} | "
                    f"P: {cls_metrics['precision']:.3f} | "
                    f"R: {cls_metrics['recall']:.3f} | "
                    f"F1: {cls_metrics['f1-score']:.3f} | "
                    f"Support: {cls_metrics['support']}"
                )

    # Save results
    results = {
        **binary_metrics,
        "auc_roc": auc,
        "multi_class": {
            k: v for k, v in multi_metrics.items() if k != "per_class"
        },
    }
    results_path = os.path.join(cfg.training.output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Federated RL-IDS")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--dataset", type=str, default="edge_iiot",
                        choices=["edge_iiot", "nsl_kdd", "iomt_2024", "unsw_nb15", "unified"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = Config()
    cfg.training.dataset = args.dataset
    cfg.training.device = args.device

    evaluate_model(args.model, cfg)

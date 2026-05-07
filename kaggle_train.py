"""
Kaggle Training Script for Federated RL-based IDS.

Features:
    - Checkpoint/Resume: Saves checkpoint every round. If Kaggle times out (12h),
      re-run the notebook — training resumes from the last completed round.
    - Metric Plotting: Automatically generates accuracy, F1, FPR, trust,
      attention, and client accuracy plots after each dataset finishes.
    - Configurable Clients: NUM_CLIENTS can be set to any value (4, 6, 8, ...).

Setup:
    1. Upload folder src/ + requirements.txt as Kaggle Dataset (name: nt549-code)
    2. Upload folder Dataset/ as Kaggle Dataset (name: NT549_full_datasets)
    3. Create Notebook, add both datasets, enable GPU
    4. Run: %run /kaggle/input/datasets/phungvannamanh/nt549-code/NT549_2/kaggle_train.py
"""

import subprocess
import sys
import os
import shutil

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION — Adjust these as needed
# ══════════════════════════════════════════════════════════════

NUM_ROUNDS = 100                    # Total FL communication rounds
NUM_CLIENTS = 10                    # Number of federated clients
LOCAL_EPISODES = 8                  # Local RL episodes per round per client
SAMPLE_LIMIT = 50000                # Max samples per CSV file
SEED = 42
DATASETS_TO_TRAIN = ["edge_iiot"]

# ── BASELINE MODE ──────────────────────────────────────────────
# Set RUN_MODE = "baseline" to run non-federated single-agent baseline
# Set RUN_MODE = "federated" to run federated training (original)
# Set RUN_MODE = "compare" to run both baseline + federated and compare
RUN_MODE = "compare"            # <-- CHANGE THIS to switch modes
BASELINE_ROUNDS = 20             # rounds for baseline (with supervised pretrain)
# k_sel: controlled by clients_per_round below
# k_min: hardcoded to 5 in train.py (max(5, num_clients // 2))

# ══════════════════════════════════════════════════════════════
#  STEP 1: Setup code & data
# ══════════════════════════════════════════════════════════════

CODE_DATASET_SLUG = "nt549-code"
DATA_DATASET_SLUG = "NT549_full_datasets"
WORK_DIR = "/kaggle/working/NT549"

if os.path.exists("/kaggle"):
    # ── Copy code from Kaggle input into working dir ──
    code_input = f"/kaggle/input/datasets/phungvannamanh/{CODE_DATASET_SLUG}/NT549/"
    if os.path.exists(code_input):
        if not os.path.exists(WORK_DIR):
            shutil.copytree(code_input, WORK_DIR)
            print(f"✓ Copied code from {code_input}")
        os.chdir(WORK_DIR)
    else:
        input_dir = "/kaggle/input/datasets/phungvannamanh"
        found = False
        if os.path.isdir(input_dir):
            for d in os.listdir(input_dir):
                src_path = os.path.join(input_dir, d, "src")
                if os.path.isdir(src_path):
                    if not os.path.exists(WORK_DIR):
                        shutil.copytree(os.path.join(input_dir, d), WORK_DIR)
                    os.chdir(WORK_DIR)
                    print(f"✓ Auto-found code in {d}")
                    found = True
                    break
        # Fallback: try flat input
        if not found:
            for d in os.listdir("/kaggle/input"):
                src_path = os.path.join("/kaggle/input", d, "src")
                if os.path.isdir(src_path):
                    if not os.path.exists(WORK_DIR):
                        shutil.copytree(os.path.join("/kaggle/input", d), WORK_DIR)
                    os.chdir(WORK_DIR)
                    print(f"✓ Auto-found code in /kaggle/input/{d}")
                    found = True
                    break
        if not found:
            print("⚠ Could not find code dataset! Please upload src/ to Kaggle Dataset.")
            sys.exit(1)

    # Install requirements
    req_file = os.path.join(WORK_DIR, "requirements.txt")
    if os.path.exists(req_file):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "-r", req_file], check=True)

    sys.path.insert(0, WORK_DIR)

    # ── Link dataset folders ──
    DATASET_FOLDERS = [
        "CIC-BCCC-NRC-Edge-IIoTSet-2022",
        "CIC-BCCC-NRC-IoMT-2024",
        "NSL-KDD",
        "UNSW-NB15",
    ]

    dataset_dir = os.path.join(WORK_DIR, "Dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    def find_data_folder(root, name):
        """Recursively find a folder with data files."""
        for dirpath, dirnames, filenames in os.walk(root):
            if os.path.basename(dirpath) == name:
                if any(f.endswith(('.csv', '.txt', '.arff')) for f in filenames):
                    return dirpath
        return None

    print("\n── Scanning /kaggle/input/ for datasets... ──")
    for folder_name in DATASET_FOLDERS:
        target = os.path.join(dataset_dir, folder_name)
        if os.path.exists(target):
            print(f"  ✓ {folder_name} already linked")
            continue
        found = find_data_folder("/kaggle/input", folder_name)
        if found:
            os.symlink(found, target)
            print(f"  ✓ Linked {folder_name} → {found}")
        else:
            print(f"  ⚠ {folder_name} not found (skipping)")

    print(f"\nWorking dir: {os.getcwd()}")
    print(f"Dataset dir: {os.listdir(dataset_dir)}")

    # ── Auto-find and load checkpoints from any dataset ──
    found_ckpt = True
    # Check 3 possible locations:
    # 1. Inside NT549_2/ (if outputs/ was copied with code)
    # 2. Sibling to NT549_2/ (outputs/ at same level as NT549_2/ in dataset)
    # 3. Inside Kaggle working dir
    search_roots = [
        os.path.join(WORK_DIR, "outputs"),                  # NT549_2/outputs/
        os.path.join(WORK_DIR, "..", "outputs"),           # sibling to NT549_2/ in dataset
        "/kaggle/working/NT549/outputs",                  # direct path
    ]
    for search_root in search_roots:
        if os.path.exists(search_root):
            print(f"\n── Auto-found checkpoints at {search_root} ──")
            for ds in DATASETS_TO_TRAIN:
                src = os.path.join(search_root, f"outputs_{ds}")
                dst = os.path.join(WORK_DIR, "..", f"outputs_{ds}")
                dst = os.path.abspath(dst)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copytree(src, dst)
                    print(f"  ✓ {ds}: checkpoint loaded")
                elif os.path.exists(dst):
                    print(f"  ✓ {ds}: checkpoint already exists")
                else:
                    print(f"  ⚠ {ds}: no checkpoint found")
            found_ckpt = True
            break

    if not found_ckpt:
        print("\n── No checkpoint found (will start fresh) ──")
else:
    print("Not running on Kaggle — using local paths")


# ══════════════════════════════════════════════════════════════
#  STEP 2: V2/V3 Config Fixes (apply BEFORE training)
# ══════════════════════════════════════════════════════════════

from src.config import Config

def apply_v3_config(cfg: Config) -> Config:
    """
    Apply V3 config to Config for federated training — MATCHES baseline_train.py V3.

    V3 was verified working on baseline (edge_iiot: Acc=0.84, F1=0.80).
    Applying the SAME parameters to federated for fair comparison.

    Key changes from default (config.py):
      Reward:
        tn_reward 5.0→1.0    (ngăn FPR=1.0 collapse)
        fn_penalty 3.0→4.0  (tăng attack detection signal)
        collapse_thr 0.65→0.70, collapse_pen 20→15, macro_f1_coef 5→3, ...
      PPO:
        lr_actor 3e-4→1e-4, lr_critic 1e-3→5e-4
        clip_epsilon 0.2→0.1, ppo_epochs 4→8, mini_batch 64→128
        lr_warmup_rounds=5 (warmup 5 rounds + cosine decay)
      Selector:
        selector_eval_interval 5→3 (33 updates vs 20 for 100 rounds)
    Note: return_norm in GAE + per-mb advantage norm are already in ppo_agent.py (always active).
    """
    # Reward
    cfg.reward.tn_reward = 1.0
    cfg.reward.fn_penalty = 4.0
    cfg.reward.balance_coef = 1.0
    cfg.reward.entropy_coef = 1.0
    cfg.reward.hhi_coef = 1.0
    cfg.reward.collapse_thr = 0.70
    cfg.reward.collapse_pen = 15.0
    cfg.reward.macro_f1_coef = 3.0

    # PPO
    cfg.ppo.lr_actor = 1e-4
    cfg.ppo.lr_critic = 5e-4
    cfg.ppo.clip_epsilon = 0.1
    cfg.ppo.ppo_epochs = 8
    cfg.ppo.mini_batch_size = 128
    cfg.ppo.lr_min_factor = 0.05
    cfg.ppo.lr_warmup_rounds = 5

    # Selector: more frequent updates (5→3) for ~33 updates in 100 rounds
    cfg.training.selector_eval_interval = 3

    return cfg


# ══════════════════════════════════════════════════════════════
#  STEP 3: Import & Check Environment
# ══════════════════════════════════════════════════════════════

import torch
import json
import numpy as np

from src.config import Config
from src.train import run_training, load_checkpoint
from baseline_train import run_baseline, compare_with_federated

print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    mem = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
    print(f"GPU Memory: {mem / 1e9:.1f} GB")


# ══════════════════════════════════════════════════════════════
#  STEP 4: Plotting Utilities
# ══════════════════════════════════════════════════════════════

def plot_training_metrics(history, dataset_name, output_dir):
    """Generate and save all metric plots from training history."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not available, skipping plots")
        return

    rounds = history.get("rounds", [])
    if len(rounds) < 2:
        print("  [INFO] Not enough data points for plots")
        return

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ── 1. Main Metrics (Accuracy, F1, Precision, Recall) ──
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rounds, history["accuracy"], "o-", label="Accuracy", linewidth=2)
    ax.plot(rounds, history["f1"], "s-", label="F1-Score", linewidth=2)
    ax.plot(rounds, history["precision"], "^-", label="Precision", linewidth=2)
    ax.plot(rounds, history["recall"], "D-", label="Recall", linewidth=2)
    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"{dataset_name.upper()} — Detection Metrics", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "metrics.png"), dpi=150)
    plt.close(fig)

    # ── 2. False Positive Rate ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rounds, [f * 100 for f in history["fpr"]], "r-o", linewidth=2)
    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel("FPR (%)", fontsize=12)
    ax.set_title(f"{dataset_name.upper()} — False Positive Rate", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(max(history["fpr"]) * 110, 5))
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "fpr.png"), dpi=150)
    plt.close(fig)

    # ── 3. Trust Scores over rounds ──
    if history.get("trust_scores") and len(history["trust_scores"]) > 0:
        num_clients = len(history["trust_scores"][0])
        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(num_clients):
            scores = [ts[i] if i < len(ts) else 0 for ts in history["trust_scores"]]
            ax.plot(rounds[:len(scores)], scores, "o-", label=f"Client {i+1}", linewidth=1.5)
        ax.set_xlabel("Communication Round", fontsize=12)
        ax.set_ylabel("Trust Score", fontsize=12)
        ax.set_title(f"{dataset_name.upper()} — Trust Score Evolution", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "trust_scores.png"), dpi=150)
        plt.close(fig)

    # ── 4. Dynamic Attention Weights (stacked area) ──
    if history.get("attention_values") and len(history["attention_values"]) > 0:
        num_clients = len(history["attention_values"][0])
        fig, ax = plt.subplots(figsize=(10, 5))
        attn_data = np.array(history["attention_values"])
        # Normalize to sum=1 per round
        attn_sums = attn_data.sum(axis=1, keepdims=True)
        attn_sums[attn_sums == 0] = 1
        attn_norm = attn_data / attn_sums
        ax.stackplot(rounds[:len(attn_norm)],
                      *[attn_norm[:, i] for i in range(num_clients)],
                      labels=[f"Client {i+1}" for i in range(num_clients)],
                      alpha=0.7)
        ax.set_xlabel("Communication Round", fontsize=12)
        ax.set_ylabel("Normalized Attention Weight", fontsize=12)
        ax.set_title(f"{dataset_name.upper()} — Attention Weight Distribution", fontsize=14)
        ax.legend(loc="upper right", fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "attention_weights.png"), dpi=150)
        plt.close(fig)

    # ── 5. Per-Client Accuracy ──
    if history.get("client_accuracies") and len(history["client_accuracies"]) > 0:
        num_clients = len(history["client_accuracies"][0])
        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(num_clients):
            acc = [aa[i] if i < len(aa) else 0 for aa in history["client_accuracies"]]
            ax.plot(rounds[:len(acc)], acc, "o-", label=f"Client {i+1}", linewidth=1.5)
        ax.set_xlabel("Communication Round", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(f"{dataset_name.upper()} — Per-Client Accuracy", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "agent_accuracy.png"), dpi=150)
        plt.close(fig)

    # ── 6. Combined Summary (2x2 subplot) ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Accuracy + F1
    axes[0, 0].plot(rounds, history["accuracy"], "b-o", label="Accuracy", linewidth=2)
    axes[0, 0].plot(rounds, history["f1"], "g-s", label="F1", linewidth=2)
    axes[0, 0].set_title("Accuracy & F1-Score")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1.05)

    # Top-right: FPR
    axes[0, 1].plot(rounds, [f * 100 for f in history["fpr"]], "r-o", linewidth=2)
    axes[0, 1].set_title("False Positive Rate (%)")
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: Trust Scores
    if history.get("trust_scores") and len(history["trust_scores"]) > 0:
        n = len(history["trust_scores"][0])
        for i in range(n):
            scores = [ts[i] if i < len(ts) else 0 for ts in history["trust_scores"]]
            axes[1, 0].plot(rounds[:len(scores)], scores, "-", label=f"Client {i+1}")
    axes[1, 0].set_title("Trust Scores")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: Client Accuracies
    if history.get("client_accuracies") and len(history["client_accuracies"]) > 0:
        n = len(history["client_accuracies"][0])
        for i in range(n):
            acc = [aa[i] if i < len(aa) else 0 for aa in history["client_accuracies"]]
            axes[1, 1].plot(rounds[:len(acc)], acc, "-", label=f"Client {i+1}")
    axes[1, 1].set_title("Client Accuracies")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1.05)

    for ax in axes.flat:
        ax.set_xlabel("Round")

    fig.suptitle(f"FedRL-IDS Training Summary — {dataset_name.upper()}", fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  ✓ Plots saved to {plots_dir}/")


def print_final_summary(all_results):
    """Print a summary table of all dataset results."""
    print("\n" + "=" * 90)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Dataset':<15} {'Acc%':>8} {'Prec%':>8} {'Rec%':>8} {'F1%':>8} {'FPR%':>8} {'Rounds':>8}")
    print("-" * 90)
    for name, metrics in all_results.items():
        if metrics:
            print(
                f"{name.upper():<15} "
                f"{metrics.get('accuracy', 0)*100:>7.2f} "
                f"{metrics.get('precision', 0)*100:>7.2f} "
                f"{metrics.get('recall', 0)*100:>7.2f} "
                f"{metrics.get('f1_score', 0)*100:>7.2f} "
                f"{metrics.get('fpr', 0)*100:>7.2f} "
                f"{'done':>8}"
            )
    print("=" * 90)


def compare_federated_vs_baseline(fed_history, baseline_history, dataset_name):
    """Print side-by-side comparison of federated vs baseline results."""
    print("\n" + "=" * 90)
    print(f"  COMPARISON: Federated vs Baseline — {dataset_name.upper()}")
    print("=" * 90)

    metrics = ["accuracy", "precision", "recall", "f1_score", "f1_macro"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "F1-Macro"]

    # Get final values
    print(f"\n  {'Metric':<15} {'Federated':>12} {'Baseline':>12} {'Delta':>12} {'Winner':>10}")
    print(f"  {'-'*55}")

    fed_wins = 0
    for m, label in zip(metrics, metric_labels):
        f_val = fed_history.get(m, [0])[-1] if fed_history.get(m) else 0.0
        b_val = baseline_history.get(m, [0])[-1] if baseline_history.get(m) else 0.0
        delta = f_val - b_val
        sign = "+" if delta >= 0 else ""
        winner = "Federated" if delta > 0 else "Baseline" if delta < 0 else "Tie"
        if delta > 0:
            fed_wins += 1
        print(f"  {label:<15} {f_val:>12.4f} {b_val:>12.4f} {sign}{delta:>11.4f} {winner:>10}")

    # FPR comparison (lower is better)
    f_fpr = fed_history.get("fpr", [1.0])[-1] if fed_history.get("fpr") else 1.0
    b_fpr = baseline_history.get("fpr", [1.0])[-1] if baseline_history.get("fpr") else 1.0
    fpr_delta = f_fpr - b_fpr
    fpr_sign = "+" if fpr_delta >= 0 else ""
    fpr_winner = "Federated" if f_fpr < b_fpr else "Baseline" if f_fpr > b_fpr else "Tie"
    print(f"  {'FPR':<15} {f_fpr:>12.4f} {b_fpr:>12.4f} {fpr_sign}{fpr_delta:>11.4f} {fpr_winner:>10}")

    # Training curves — peak and convergence speed
    if fed_history.get("accuracy") and baseline_history.get("accuracy"):
        f_peak = max(fed_history["accuracy"])
        b_peak = max(baseline_history["accuracy"])
        f_best_round = fed_history["accuracy"].index(f_peak) + 1
        b_best_round = baseline_history["accuracy"].index(b_peak) + 1
        print(f"\n  {'Metric':<15} {'Federated':>12} {'Baseline':>12} {'Winner':>10}")
        print(f"  {'-'*45}")
        print(f"  {'Peak Accuracy':<15} {f_peak:>12.4f} {b_peak:>12.4f} "
              f"{'Federated' if f_peak > b_peak else 'Baseline':>10}")
        print(f"  {'Best Round':<15} {f_best_round:>12} {b_best_round:>12} "
              f"{'Federated' if f_best_round < b_best_round else 'Baseline':>10}")
        print(f"  {'Total Rounds':<15} {len(fed_history['accuracy']):>12} {len(baseline_history['accuracy']):>12}")

    print(f"\n  Interpretation:")
    if fed_wins >= 3:
        print(f"  → Federated outperforms Baseline in {fed_wins}/5 metrics.")
        print(f"    FLTrust + RL Client Selection is beneficial.")
    else:
        print(f"  → Baseline outperforms Federated in {5-fed_wins}/5 metrics.")
        print(f"    Possible causes:")
        print(f"    1. Non-IID data partitions hurt FL aggregation")
        print(f"    2. RL selector hasn't converged yet")
        print(f"    3. FLTrust threshold too strict, filtering good updates")
        print(f"    4. Need more rounds for FL to converge")

    print("=" * 90)


# ══════════════════════════════════════════════════════════════
#  STEP 5a: BASELINE MODE — Single PPO Agent (No Federation)
# ══════════════════════════════════════════════════════════════

if RUN_MODE in ("baseline", "compare"):
    print(f"\n{'='*70}")
    print(f"  BASELINE MODE: Single PPO Agent (No Federation)")
    print(f"{'='*70}")
    print(f"  Dataset:    {DATASETS_TO_TRAIN}")
    print(f"  Rounds:     {BASELINE_ROUNDS}")
    print(f"  Episodes:   {LOCAL_EPISODES}")
    print(f"  Max steps:  2000")
    print(f"  V3 fixes:   TN_REWARD=1.0, lr=1e-4, clip=0.1, epochs=8, batch=128")
    print(f"  Purpose:    Isolate PPO vs Federated performance")
    print(f"{'='*70}")

    baseline_results = {}
    for dataset_name in DATASETS_TO_TRAIN:
        print(f"\n{'='*70}")
        print(f"  BASELINE: {dataset_name.upper()}")
        print(f"{'='*70}")

        cfg = Config()
        cfg = apply_v3_config(cfg)   # V3 config (same as baseline_train.py)
        cfg.training.dataset = dataset_name
        cfg.training.local_episodes = LOCAL_EPISODES
        cfg.training.device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.training.seed = SEED
        cfg.training.sample_limit_per_file = SAMPLE_LIMIT

        if os.path.exists("/kaggle"):
            output_dir = "/kaggle/working/NT549/outputs"
        else:
            output_dir = os.path.join(WORK_DIR, "outputs")
        cfg.training.output_dir = os.path.join(output_dir, "baseline_cen")
        os.makedirs(cfg.training.output_dir, exist_ok=True)

        try:
            history, final_metrics = run_baseline(cfg, num_rounds=BASELINE_ROUNDS)
            baseline_results[dataset_name] = final_metrics
            print(f"\n  ✓ {dataset_name} baseline done — Acc: {final_metrics.get('accuracy', 0):.4f}")

            # Save plots for baseline
            from pathlib import Path
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plots_dir = os.path.join(cfg.training.output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            rounds = history.get("rounds", [])

            if len(rounds) >= 2:
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                axes[0, 0].plot(rounds, history["accuracy"], "b-o", label="Acc", linewidth=2)
                axes[0, 0].plot(rounds, history["f1_score"], "g-s", label="F1", linewidth=2)
                axes[0, 0].plot(rounds, history["f1_macro"], "c-^", label="F1m", linewidth=2)
                axes[0, 0].set_title("Accuracy & F1")
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].set_ylim(0, 1.05)

                axes[0, 1].plot(rounds, [f * 100 for f in history["fpr"]], "r-o", linewidth=2)
                axes[0, 1].set_title("False Positive Rate (%)")
                axes[0, 1].grid(True, alpha=0.3)

                axes[1, 0].plot(rounds, history["episode_rewards"], "m-o", linewidth=1.5)
                axes[1, 0].set_title("Episode Reward")
                axes[1, 0].grid(True, alpha=0.3)

                axes[1, 1].plot(rounds, history["entropies"], "k-o", linewidth=1.5)
                axes[1, 1].set_title("Policy Entropy")
                axes[1, 1].grid(True, alpha=0.3)

                for ax in axes.flat:
                    ax.set_xlabel("Round")
                fig.suptitle(f"Baseline (No Federation) — {dataset_name.upper()}", fontsize=14)
                fig.tight_layout()
                fig.savefig(os.path.join(plots_dir, "baseline_summary.png"), dpi=150)
                plt.close(fig)
                print(f"  ✓ Plots saved to {plots_dir}/")

            # Compare with federated results
            from pathlib import Path
            if os.path.exists("/kaggle"):
                fed_path = Path("/kaggle/working/NT549/outputs") / f"outputs_{dataset_name}" / "training_history.json"
            else:
                fed_path = Path(WORK_DIR) / "outputs" / f"outputs_{dataset_name}" / "training_history.json"

            if fed_path.exists():
                compare_with_federated(history, str(fed_path))

        except Exception as e:
            print(f"\n  ✗ {dataset_name} baseline failed: {e}")
            import traceback
            traceback.print_exc()
            baseline_results[dataset_name] = None

        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Baseline summary
    print("\n" + "=" * 90)
    print("  BASELINE RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Dataset':<15} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'F1m':>8} {'FPR':>8}")
    print("-" * 90)
    for name, metrics in baseline_results.items():
        if metrics:
            print(
                f"{name.upper():<15} "
                f"{metrics.get('accuracy', 0):>8.4f} "
                f"{metrics.get('precision', 0):>8.4f} "
                f"{metrics.get('recall', 0):>8.4f} "
                f"{metrics.get('f1_score', 0):>8.4f} "
                f"{metrics.get('f1_macro', 0):>8.4f} "
                f"{metrics.get('fpr', 0):>8.4f}"
            )
    print("=" * 90)

    if RUN_MODE == "compare":
        print("\n\n" + "=" * 70)
        print("  BASELINE COMPLETE — Switching to FEDERATED mode...")
        print("=" * 70)
    else:
        print("\n  ALL BASELINE TRAINING COMPLETE")
        print("=" * 70)
        sys.exit(0)  # Exit after baseline, don't run federated


# ══════════════════════════════════════════════════════════════
#  STEP 5b: FEDERATED MODE — PPO + FLTrust + RL Client Selector
# ══════════════════════════════════════════════════════════════

all_results = {}

for dataset_name in DATASETS_TO_TRAIN:
    print(f"\n{'='*70}")
    print(f"  TRAINING: {dataset_name.upper()} ({NUM_ROUNDS} rounds, {NUM_CLIENTS} clients)")
    print(f"{'='*70}")

    if os.path.exists("/kaggle"):
        output_dir = "/kaggle/working/NT549/outputs"
    else:
        output_dir = os.path.join(WORK_DIR, "outputs")

    os.makedirs(output_dir, exist_ok=True)

    # train.py appends outputs_{dataset} to output_dir, so checkpoint is at:
    #   {output_dir}/outputs_{dataset}/checkpoint_latest.pt
    checkpoint_dir = os.path.join(output_dir, f"outputs_{dataset_name}")
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
    resume_from = None

    if os.path.exists(checkpoint_path):
        try:
            ckpt = load_checkpoint(checkpoint_path)
            completed_round = ckpt["round_idx"]
            if completed_round + 1 >= NUM_ROUNDS:
                print(f"  ✓ {dataset_name} already completed ({completed_round+1}/{NUM_ROUNDS} rounds). Skipping.")
                # Load final metrics from history
                hist = ckpt.get("history", {})
                if hist.get("accuracy"):
                    all_results[dataset_name] = {
                        "accuracy": hist["accuracy"][-1],
                        "precision": hist["precision"][-1],
                        "recall": hist["recall"][-1],
                        "f1_score": hist["f1"][-1],
                        "fpr": hist["fpr"][-1],
                    }
                    # Still plot
                    plot_training_metrics(hist, dataset_name, checkpoint_dir)
                continue
            else:
                print(f"  ↻ Found checkpoint at round {completed_round+1}/{NUM_ROUNDS}. Resuming...")
                resume_from = checkpoint_path
        except Exception as e:
            print(f"  ⚠ Checkpoint corrupted, starting fresh: {e}")
            resume_from = None

    # ── Configure ──
    cfg = Config()
    cfg = apply_v3_config(cfg)   # V3 config (same as baseline_train.py)
    cfg.training.dataset = dataset_name
    cfg.training.num_clients = NUM_CLIENTS
    cfg.training.num_rounds = NUM_ROUNDS
    cfg.training.local_episodes = LOCAL_EPISODES
    cfg.training.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.training.seed = SEED
    cfg.training.sample_limit_per_file = SAMPLE_LIMIT
    cfg.training.output_dir = output_dir
    cfg.training.client_selection_enabled = True     # RL Client Selector (Tier-2)
    cfg.training.clients_per_round = NUM_CLIENTS - 2  # K_sel = 8 (decays to min 5)

    print(f"\n  V3 Config applied (matches baseline_train.py):")
    print(f"    TN_REWARD={cfg.reward.tn_reward}, FN_PENALTY={cfg.reward.fn_penalty}")
    print(f"    lr_actor={cfg.ppo.lr_actor}, lr_critic={cfg.ppo.lr_critic}")
    print(f"    clip_epsilon={cfg.ppo.clip_epsilon}, epochs={cfg.ppo.ppo_epochs}, batch={cfg.ppo.mini_batch_size}")

    # ── Train ──
    try:
        history, final_metrics = run_training(cfg, resume_checkpoint=resume_from)
        all_results[dataset_name] = final_metrics
        print(f"\n  ✓ {dataset_name} done — Accuracy: {final_metrics.get('accuracy', 0):.4f}")

        # ── Plot metrics ──
        print(f"\n  Generating plots for {dataset_name}...")
        plot_training_metrics(history, dataset_name, checkpoint_dir)

    except Exception as e:
        print(f"\n  ✗ {dataset_name} failed: {e}")
        import traceback
        traceback.print_exc()
        all_results[dataset_name] = None
        continue

    # ── Memory cleanup between datasets ──
    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print(f"\n  [MEM] {dataset_name} cleanup: torch.cuda.empty_cache() + gc.collect() done")

# ══════════════════════════════════════════════════════════════
#  STEP 6: Summary
# ══════════════════════════════════════════════════════════════

print_final_summary(all_results)

# Compare federated vs baseline in "compare" mode
if RUN_MODE == "compare" and baseline_results:
    for dataset_name in DATASETS_TO_TRAIN:
        if dataset_name not in baseline_results or baseline_results[dataset_name] is None:
            continue

        # Load baseline history
        if os.path.exists("/kaggle"):
            baseline_hist_path = Path("/kaggle/working/NT549/outputs/baseline_cen/baseline_v3_history.json")
        else:
            baseline_hist_path = Path(WORK_DIR) / "outputs" / "baseline_cen" / "baseline_v3_history.json"

        if not baseline_hist_path.exists():
            # Try finding any baseline history file
            possible_paths = list(Path("/kaggle/working/NT549/outputs").glob("**/baseline*history*.json")) if os.path.exists("/kaggle") else list(Path(WORK_DIR).glob("outputs/**/baseline*history*.json"))
            for bp in possible_paths:
                if dataset_name in str(bp) or "cen" in str(bp):
                    baseline_hist_path = bp
                    break

        if baseline_hist_path.exists():
            try:
                with open(baseline_hist_path) as f:
                    baseline_history = json.load(f)
                # Load federated history
                if os.path.exists("/kaggle"):
                    fed_hist_path = Path("/kaggle/working/NT549/outputs") / f"outputs_{dataset_name}" / "training_history.json"
                else:
                    fed_hist_path = Path(WORK_DIR) / "outputs" / f"outputs_{dataset_name}" / "training_history.json"
                with open(fed_hist_path) as f:
                    fed_history = json.load(f)
                compare_federated_vs_baseline(fed_history, baseline_history, dataset_name)
            except Exception as e:
                print(f"\n  ⚠ Could not compare: {e}")

print("\n" + "=" * 70)
print("  ALL TRAINING COMPLETE")
print("=" * 70)

# List all output files
if os.path.exists("/kaggle"):
    for ds in DATASETS_TO_TRAIN:
        out = os.path.join("/kaggle/working/NT549/outputs", f"outputs_{ds}")
        if os.path.exists(out):
            files = os.listdir(out)
            print(f"\n  {ds}: {files}")


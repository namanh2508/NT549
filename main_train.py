"""
Main Training Script for FDRL-IDS.
============================================================================
Paper: "Federated reinforcement learning based intrusion detection system
        using dynamic attention mechanism"
Journal: Journal of Information Security and Applications 78 (2023) 103608
============================================================================
This script runs the complete training pipeline:
  1. Load and preprocess dataset (NSL-KDD, CIC-IoMT-2024, or CIC-Edge-IIoTSet-2022)
  2. Train Denoising Autoencoder (Section 4.1, Page 5)
  3. Transform features through DAE
  4. Distribute data among agents (Section 6, Page 8)
  5. Run federated training with dynamic attention (Algorithms 1-3, Page 6)
  6. Evaluate and visualize results (Section 6, Pages 8-13)

Usage:
  # Scenario A: NSL-KDD (original paper)
  python main_train.py --dataset nsl-kdd --experiment random

  # Scenario B: Same-domain IoT
  python main_train.py --dataset iomt --experiment random --subsample_per_file 10000

  # Scenario C: Cross-domain IoT
  python main_train.py --dataset iomt --test_dataset iiot --experiment random --subsample_per_file 10000
============================================================================
"""

import argparse
import os
import sys
import time
import json
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.nsl_kdd import (
    load_nsl_kdd, preprocess_nsl_kdd, distribute_data_random,
    distribute_data_customized
)
from src.data.cic_common import (
    load_cic_dataset, preprocess_cic, resolve_cic_dataset_path,
    distribute_data_customized_iomt, distribute_data_customized_iiot
)
from src.models.denoising_autoencoder import (
    DenoisingAutoencoder, train_dae, transform_with_dae
)
from src.federated_learning.orchestrator import FederatedOrchestrator
from src.utils.metrics import compute_metrics, print_metrics, compute_roc_data
from src.utils.visualization import (
    plot_all_training_curves, plot_roc_curve, plot_accuracy_vs_num_agents
)
from src.utils.config import *


def parse_args():
    parser = argparse.ArgumentParser(description='FDRL-IDS Training')
    parser.add_argument('--dataset', type=str, default='nsl-kdd',
                        choices=['nsl-kdd', 'iomt', 'iiot'],
                        help='Training dataset: nsl-kdd, iomt (CIC-IoMT-2024), '
                             'iiot (CIC-Edge-IIoTSet-2022)')
    parser.add_argument('--test_dataset', type=str, default=None,
                        choices=['nsl-kdd', 'iomt', 'iiot'],
                        help='Test dataset (if different from --dataset, '
                             'enables cross-dataset evaluation)')
    parser.add_argument('--data_dir', type=str, default='./Dataset',
                        help='Directory containing dataset folders')
    parser.add_argument('--experiment', type=str, default='random',
                        choices=['random', 'customized', 'scalability'],
                        help='Experiment type (Section 6, Page 8)')
    parser.add_argument('--num_agents', type=int, default=8,
                        help='Number of agents (default 8 for random, 2 for customized)')
    parser.add_argument('--num_rounds', type=int, default=30,
                        help='Number of federated rounds')
    parser.add_argument('--episodes_per_round', type=int, default=3,
                        help='Episodes per round per agent')
    parser.add_argument('--attention_k', type=float, default=30,
                        help='Attention param k (Section 6.1, Page 8)')
    parser.add_argument('--attention_a', type=float, default=50,
                        help='Attention param a (Section 6.1, Page 8)')
    parser.add_argument('--use_dae', action='store_true', default=True,
                        help='Use Denoising Autoencoder (default: enabled)')
    parser.add_argument('--no_dae', action='store_false', dest='use_dae',
                        help='Disable Denoising Autoencoder')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='Optional cap on train samples for quick smoke runs')
    parser.add_argument('--max_test_samples', type=int, default=None,
                        help='Optional cap on test samples for quick smoke runs')
    parser.add_argument('--subsample_per_file', type=int, default=None,
                        help='Max rows per CSV file for CIC datasets '
                             '(recommended: 10000 for ~150K total rows)')
    return parser.parse_args()


def set_seeds(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dataset_paths(data_dir):
    """Resolve NSL-KDD train/test file paths from common local/Kaggle layouts.

    NOTE: Kaggle strips '+' from filenames on dataset upload, so
      KDDTrain+.txt  ->  KDDTrain.txt
      KDDTest+.txt   ->  KDDTest.txt
    Both variants are tried automatically.
    """
    candidates = [
        data_dir,
        os.path.join(data_dir, 'NSL-KDD'),
        '/kaggle/input/nsl-kdd',
        '/kaggle/input/nsl-kdd/NSL-KDD',
        '/kaggle/input/nslkdd',
        '/kaggle/input/nslkdd/NSL-KDD',
    ]
    # Try original names first, then Kaggle-stripped names ('+' removed)
    name_pairs = [
        ('KDDTrain+.txt', 'KDDTest+.txt'),  # original local filenames
        ('KDDTrain.txt',  'KDDTest.txt'),   # Kaggle upload strips '+'
    ]

    for base_dir in candidates:
        for train_name, test_name in name_pairs:
            train_path = os.path.join(base_dir, train_name)
            test_path  = os.path.join(base_dir, test_name)
            if os.path.exists(train_path) and os.path.exists(test_path):
                return train_path, test_path

    raise FileNotFoundError(
        "Could not find NSL-KDD dataset files.\n"
        "  Expected: KDDTrain+.txt / KDDTrain.txt  and  KDDTest+.txt / KDDTest.txt\n"
        f"  Searched directories: {candidates}\n"
        "  Tip: On Kaggle, '+' is stripped from filenames on upload."
    )


def maybe_subsample(X, y, max_samples, seed):
    """Optionally subsample dataset for smoke tests."""
    if max_samples is None or max_samples >= len(X):
        return X, y

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X), size=max_samples, replace=False)
    return X[indices], y[indices]


def run_experiment(args):
    """
    Run a single experiment.

    Paper Reference: Section 6, Pages 8-13
    Two types of experiments:
    1. Random/Uniform split: 8 agents, random equal-size data splits (Page 8)
    2. Customized split: 2 agents, Agent0=Normal+DoS, Agent1=Normal+Probe+U2R+R2L (Page 9)
    """
    # Build descriptive labels for output
    dataset_label = args.dataset.upper().replace('-', '_')
    test_dataset_name = args.test_dataset or args.dataset
    test_label = test_dataset_name.upper().replace('-', '_')
    is_cross_dataset = (args.test_dataset is not None
                        and args.test_dataset != args.dataset)

    print("=" * 70)
    print("  FDRL-IDS: Federated Deep Reinforcement Learning based IDS")
    print("  Paper: JISA 78 (2023) 103608")
    print(f"  Experiment: {args.experiment}")
    print(f"  Train dataset: {dataset_label}")
    print(f"  Test dataset:  {test_label}"
          + (" (cross-dataset)" if is_cross_dataset else ""))
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)
    set_seeds(args.seed)

    # ==================================================================
    # STEP 1: Load and Preprocess Dataset
    # ==================================================================
    train_attack_names = None  # Used for customized CIC distribution
    raw_train_df = None  # Used for customized NSL-KDD distribution

    if args.dataset == 'nsl-kdd':
        # --- NSL-KDD path (Paper Section 5.1, Page 7) ---
        print("\n[Step 1] Loading NSL-KDD dataset...")
        train_path, test_path = resolve_dataset_paths(args.data_dir)
        print(f"  Train file: {train_path}")
        print(f"  Test file:  {test_path}")

        train_df, test_df = load_nsl_kdd(train_path, test_path)
        import pandas as pd
        raw_train_df = train_df.copy()

        X_train, y_train, X_test, y_test, scaler, feature_dim = preprocess_nsl_kdd(
            train_df, test_df
        )

    elif args.dataset in ('iomt', 'iiot'):
        # --- CIC dataset path ---
        print(f"\n[Step 1] Loading CIC {dataset_label} dataset...")
        train_dir = resolve_cic_dataset_path(args.data_dir, args.dataset)
        print(f"  Train dir: {train_dir}")

        train_df = load_cic_dataset(
            train_dir, subsample_per_file=args.subsample_per_file, seed=args.seed
        )

        if is_cross_dataset:
            # Cross-dataset: load test dataset separately
            # Bug 1 fix: use different seed for test to ensure independent subsampling
            print(f"\n  Loading test dataset: {test_label}...")
            test_dir = resolve_cic_dataset_path(args.data_dir, test_dataset_name)
            print(f"  Test dir: {test_dir}")
            test_df = load_cic_dataset(
                test_dir, subsample_per_file=args.subsample_per_file, seed=args.seed + 1
            )
        else:
            # Same-domain: test_df=None triggers 80/20 stratified split
            test_df = None

        result = preprocess_cic(train_df, test_df, seed=args.seed)
        X_train, y_train, X_test, y_test = result[0], result[1], result[2], result[3]
        scaler, feature_dim = result[4], result[5]
        train_attack_names = result[6]
        test_attack_names = result[7]  # Bug 2 fix: capture test_attack_names

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Optional subsampling for smoke tests
    original_train_len = len(X_train)
    X_train, y_train = maybe_subsample(
        X_train, y_train, args.max_train_samples, args.seed
    )
    X_test, y_test = maybe_subsample(
        X_test, y_test, args.max_test_samples, args.seed
    )
    if args.max_train_samples is not None or args.max_test_samples is not None:
        print(f"  [Subsample] Train={len(X_train)} | Test={len(X_test)}")

    # Keep train_attack_names aligned with X_train.
    # If maybe_subsample reduced X_train, re-apply the same indices to attack_names
    # so that customized CIC distribution matches the correct rows.
    if train_attack_names is not None and len(train_attack_names) != len(X_train):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(original_train_len, size=len(X_train), replace=False)
        train_attack_names = train_attack_names[idx]

    # ==================================================================
    # STEP 2: Train Denoising Autoencoder (Optional)
    # Paper Reference: Section 4.1, Page 5
    # "Features representations will be passed through a denoising
    #  autoencoder (DAE) to protect the model from adversarial attacks"
    # ==================================================================
    if args.use_dae:
        print("\n[Step 2] Training Denoising Autoencoder...")
        dae = DenoisingAutoencoder(
            input_dim=feature_dim,
            hidden_dim=DAE_HIDDEN_DIM,
            noise_factor=DAE_NOISE_FACTOR
        )
        dae, dae_losses = train_dae(
            dae, X_train, DEVICE,
            epochs=DAE_EPOCHS,
            batch_size=DAE_BATCH_SIZE,
            lr=DAE_LEARNING_RATE
        )

        # Transform features through DAE
        print("  Transforming features through DAE...")
        X_train_dae = transform_with_dae(dae, X_train, DEVICE)
        X_test_dae = transform_with_dae(dae, X_test, DEVICE)
        input_dim = X_train_dae.shape[1]
        print(f"  Feature dim after DAE: {input_dim}")

        # Re-normalize the DAE output
        from sklearn.preprocessing import MinMaxScaler
        dae_scaler = MinMaxScaler()
        X_train_final = dae_scaler.fit_transform(X_train_dae)
        X_test_final = dae_scaler.transform(X_test_dae)
        # Warning 3 fix: clip test data after DAE scaler (cross-domain values may exceed [0,1])
        X_test_final = np.clip(X_test_final, 0.0, 1.0)
    else:
        print("\n[Step 2] Skipping DAE, using preprocessed features directly...")
        X_train_final = X_train
        X_test_final = X_test
        input_dim = feature_dim

    # ==================================================================
    # STEP 3: Distribute Data Among Agents
    # Paper Reference: Section 6, Page 8-9
    # ==================================================================
    print(f"\n[Step 3] Distributing data among agents ({args.experiment} split)...")

    if args.experiment == 'random':
        # Section 6, Page 8: "all the available data is randomly split
        # into approximately equal parts and shared among the agents"
        num_agents = args.num_agents
        agent_data = distribute_data_random(X_train_final, y_train, num_agents, args.seed)
        # Random split params (Section 6.1, Page 8): k=30, a=50
        attn_k = args.attention_k
        attn_a = args.attention_a

    elif args.experiment == 'customized':
        # Section 6.1, Page 9: 2 agents with customized data
        num_agents = 2
        if args.dataset == 'nsl-kdd':
            agent_data = distribute_data_customized(
                raw_train_df, y_train, X_train_final, num_agents=2, seed=args.seed
            )
        elif args.dataset == 'iomt':
            agent_data = distribute_data_customized_iomt(
                train_attack_names, y_train, X_train_final, seed=args.seed
            )
        elif args.dataset == 'iiot':
            agent_data = distribute_data_customized_iiot(
                train_attack_names, y_train, X_train_final, seed=args.seed
            )
        # Customized split params (Section 6.1, Page 10): k=50000, a=200
        attn_k = 50000
        attn_a = 200

    else:
        raise ValueError(f"Unknown experiment type: {args.experiment}")

    # ==================================================================
    # STEP 4: Create Federated Orchestrator and Train
    # Paper Reference: Algorithms 1-3, Section 4, Pages 4-6
    # ==================================================================
    print(f"\n[Step 4] Creating federated system with {num_agents} agents...")

    orchestrator = FederatedOrchestrator(
        num_agents=num_agents,
        input_dim=input_dim,
        hidden_layers=DQN_HIDDEN_LAYERS,
        num_actions=NUM_ACTIONS,
        lr=DQN_LEARNING_RATE,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        memory_capacity=MEMORY_CAPACITY,
        batch_size_replay=BATCH_SIZE_REPLAY,
        per_alpha=PER_ALPHA,
        per_beta_start=PER_BETA_START,
        per_beta_end=PER_BETA_END,
        omega=OMEGA,
        dropout=DQN_DROPOUT,
        attention_k=attn_k,
        attention_a=attn_a,
        episodes_per_round=args.episodes_per_round,
        device=str(DEVICE)
    )

    # Assign data to agents
    orchestrator.assign_data(agent_data, test_split_ratio=TEST_SPLIT_RATIO)

    # ==================================================================
    # STEP 5: Run Federated Training
    # Paper Reference: Section 4, Pages 4-6
    # ==================================================================
    print(f"\n[Step 5] Starting federated training for {args.num_rounds} rounds...")
    start_time = time.time()

    history = orchestrator.train(
        num_rounds=args.num_rounds,
        global_test_X=X_test_final,
        global_test_y=y_test,
        verbose=True
    )

    total_time = time.time() - start_time
    print(f"\n[Training Complete] Total time: {total_time:.1f}s")

    # ==================================================================
    # STEP 6: Final Evaluation
    # Paper Reference: Table 3, Page 11
    # ==================================================================
    print("\n[Step 6] Final Evaluation on Test Set...")
    final_metrics = orchestrator.evaluate_global(X_test_final, y_test)
    print("\n" + "=" * 50)
    if is_cross_dataset:
        print(f"  FINAL RESULTS (Train: {dataset_label}, Test: {test_label})")
    else:
        print(f"  FINAL RESULTS ({dataset_label}, {args.experiment} split)")
    print("=" * 50)
    print_metrics(final_metrics, "Global Model")

    if args.dataset == 'nsl-kdd':
        # Paper Reference: Table 3 expected results for NSL-KDD random:
        # Accuracy=0.9669, FPR=0.0195, Recall=0.9514,
        # Precision=0.9769, F1=0.964, AUC=0.994
        print("\n[Paper Reference] Table 3 (Page 11) Expected for NSL-KDD Random:")
        print("  Accuracy=0.9669, FPR=0.0195, Recall=0.9514")
        print("  Precision=0.9769, F1=0.964, AUC=0.994")

    # ==================================================================
    # STEP 7: Plot Results
    # Paper Reference: Figures 3-7, Pages 9-13
    # ==================================================================
    print(f"\n[Step 7] Generating plots...")
    if is_cross_dataset:
        plot_suffix = f"{args.dataset}_to_{test_dataset_name}_{args.experiment}"
        title_suffix = f"(Train:{dataset_label}, Test:{test_label}, {args.experiment})"
    else:
        plot_suffix = f"{args.dataset}_{args.experiment}"
        title_suffix = f"({dataset_label}, {args.experiment} split)"

    plot_dir = os.path.join(args.output_dir, f'plots_{plot_suffix}')

    plot_all_training_curves(
        history, num_agents,
        title_suffix=title_suffix,
        save_dir=plot_dir
    )

    # ROC Curve (Figure 7, Page 13)
    # Get predictions with probabilities for ROC
    # Warning 5 fix: batch inference instead of item-by-item loop
    agent = orchestrator.agents[0]
    agent.dqn.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test_final).to(DEVICE)
        q_vals = agent.dqn(X_tensor)
        probs = torch.softmax(q_vals, dim=1)
        probabilities = probs[:, 1].cpu().numpy()
        predictions = q_vals.argmax(dim=1).cpu().numpy()

    fpr_arr, tpr_arr, _ = compute_roc_data(y_test, probabilities)
    roc_path = os.path.join(plot_dir, 'roc_curve.png') if plot_dir else None
    plot_roc_curve(
        fpr_arr, tpr_arr, final_metrics['auc_roc'],
        title=f"ROC Curve - {title_suffix}",
        save_path=roc_path
    )

    # ==================================================================
    # STEP 8: Save Results
    # ==================================================================
    results = {
        'experiment': args.experiment,
        'dataset': args.dataset,
        'test_dataset': test_dataset_name,
        'cross_dataset': is_cross_dataset,
        'subsample_per_file': args.subsample_per_file,
        'num_agents': num_agents,
        'num_rounds': args.num_rounds,
        'attention_k': attn_k,
        'attention_a': attn_a,
        'final_metrics': {k: float(v) for k, v in final_metrics.items()},
        'training_time': total_time,
    }

    results_path = os.path.join(args.output_dir, f'results_{plot_suffix}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[Results] Saved to {results_path}")

    return final_metrics, history


def run_scalability_experiment(args):
    """
    Scalability experiment: accuracy vs number of agents.

    Paper Reference: Section 6.3, Page 11; Figure 8, Page 13
    "we performed an experiment in which we simulated the system multiple
     times with varying number of agents in an increasing fashion"
    "the average accuracy of the system remains constant, when the number
     of agents increases" (Page 11)
    """
    print("=" * 70)
    print("  Scalability Experiment (Section 6.3, Page 11)")
    print(f"  Dataset: {args.dataset.upper()}")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)
    set_seeds(args.seed)

    # Load data based on dataset choice
    if args.dataset == 'nsl-kdd':
        train_path, test_path = resolve_dataset_paths(args.data_dir)
        train_df, test_df = load_nsl_kdd(train_path, test_path)
        X_train, y_train, X_test, y_test, scaler, feature_dim = preprocess_nsl_kdd(
            train_df, test_df
        )
    elif args.dataset in ('iomt', 'iiot'):
        train_dir = resolve_cic_dataset_path(args.data_dir, args.dataset)
        train_df = load_cic_dataset(
            train_dir, subsample_per_file=args.subsample_per_file, seed=args.seed
        )
        result = preprocess_cic(train_df, seed=args.seed)
        X_train, y_train, X_test, y_test = result[0], result[1], result[2], result[3]
        scaler, feature_dim = result[4], result[5]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    X_train, y_train = maybe_subsample(
        X_train, y_train, args.max_train_samples, args.seed
    )
    X_test, y_test = maybe_subsample(
        X_test, y_test, args.max_test_samples, args.seed
    )

    # Optional DAE
    if args.use_dae:
        dae = DenoisingAutoencoder(feature_dim, DAE_HIDDEN_DIM, DAE_NOISE_FACTOR)
        dae, _ = train_dae(dae, X_train, DEVICE, DAE_EPOCHS, DAE_BATCH_SIZE, DAE_LEARNING_RATE)
        X_train_final = transform_with_dae(dae, X_train, DEVICE)
        X_test_final = transform_with_dae(dae, X_test, DEVICE)
        from sklearn.preprocessing import MinMaxScaler
        dae_scaler = MinMaxScaler()
        X_train_final = dae_scaler.fit_transform(X_train_final)
        X_test_final = dae_scaler.transform(X_test_final)
        X_test_final = np.clip(X_test_final, 0.0, 1.0)
        input_dim = X_train_final.shape[1]
    else:
        X_train_final, X_test_final = X_train, X_test
        input_dim = feature_dim

    # Test with different numbers of agents
    agent_counts = [2, 4, 6, 8]
    avg_accuracies = []

    for n_agents in agent_counts:
        print(f"\n--- Testing with {n_agents} agents ---")
        agent_data = distribute_data_random(X_train_final, y_train, n_agents, args.seed)

        orchestrator = FederatedOrchestrator(
            num_agents=n_agents, input_dim=input_dim,
            hidden_layers=DQN_HIDDEN_LAYERS, num_actions=NUM_ACTIONS,
            lr=DQN_LEARNING_RATE, gamma=GAMMA,
            epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, memory_capacity=MEMORY_CAPACITY,
            batch_size_replay=BATCH_SIZE_REPLAY,
            per_alpha=PER_ALPHA, per_beta_start=PER_BETA_START,
            per_beta_end=PER_BETA_END, omega=OMEGA, dropout=DQN_DROPOUT,
            attention_k=args.attention_k, attention_a=args.attention_a,
            episodes_per_round=args.episodes_per_round, device=str(DEVICE)
        )
        orchestrator.assign_data(agent_data, test_split_ratio=TEST_SPLIT_RATIO)

        history = orchestrator.train(
            num_rounds=args.num_rounds,
            global_test_X=X_test_final, global_test_y=y_test,
            verbose=False
        )

        final = orchestrator.evaluate_global(X_test_final, y_test)
        avg_accuracies.append(final['accuracy'])
        print(f"  {n_agents} agents -> Accuracy: {final['accuracy']:.4f}")

    # Plot (Figure 8, Page 13)
    plot_dir = os.path.join(args.output_dir, 'plots_scalability')
    os.makedirs(plot_dir, exist_ok=True)
    plot_accuracy_vs_num_agents(
        agent_counts, avg_accuracies,
        title=f"({args.dataset.upper()})",
        save_path=os.path.join(plot_dir, 'scalability.png')
    )

    return agent_counts, avg_accuracies


if __name__ == '__main__':
    args = parse_args()

    if args.experiment == 'scalability':
        run_scalability_experiment(args)
    else:
        run_experiment(args)

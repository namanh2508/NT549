"""
Main training pipeline for Federated RL-based IDS.

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                    Central Server                        │
    │  ┌─────────┐  ┌──────────┐  ┌──────────────────────┐   │
    │  │ FLTrust │  │  Fed+    │  │ Dynamic Attention     │   │
    │  │ (trust  │  │ (person- │  │ (performance-aware    │   │
    │  │  scores)│  │  alise)  │  │  weighting)           │   │
    │  └────┬────┘  └────┬─────┘  └──────────┬───────────┘   │
    │       └────────────┴───────────────────┘                │
    │                    │                                     │
    │              Aggregated Model                           │
    │            ┌───────┴────────┐                           │
    │            │  Meta-Agent    │ ← Tier 2                  │
    │            │  (coordinator) │                           │
    │            └───────┬────────┘                           │
    └────────────────────┼────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────┴────┐    ┌────┴────┐    ┌────┴────┐
    │Client 0 │    │Client 1 │    │Client N │  ← Tier 1
    │  (PPO)  │    │  (PPO)  │    │  (PPO)  │
    │ Local   │    │ Local   │    │ Local   │
    │ Data    │    │ Data    │    │ Data    │
    └─────────┘    └─────────┘    └─────────┘
"""

import os
import json
import time
import torch
import numpy as np
from collections import OrderedDict
from typing import List, Dict, Optional, Tuple

from src.config import Config
from src.data.preprocessor import (
    load_dataset,
    partition_data_non_iid,
    create_root_dataset,
)
from src.agents.local_client import LocalClient
from src.agents.meta_agent import MetaAgent
from src.agents.ppo_agent import PPOAgent
from src.federated.aggregator import FederatedAggregator
from src.federated.client_selector import RLClientSelector, compute_model_divergence, compute_gradient_alignment
from src.environment.ids_env import MultiClassIDSEnvironment
from src.models.networks import NoveltyDetector
from src.utils.metrics import (
    compute_binary_metrics,
    compute_multiclass_metrics,
    print_metrics,
)


def retrain_novelty_detector(
    novelty_detector: torch.nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    aggregated_model: OrderedDict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    cfg: Config,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Retrain the novelty detector autoencoder using high-confidence samples
    from the current global model.

    FIX (F): The novelty detector was trained once at round 0 and never updated,
    making it static and unable to adapt to evolving attack patterns. This function
    retrains the autoencoder periodically using samples the current model predicts
    with high confidence as "known" (normal) traffic.

    The retraining uses confident predictions as pseudo-labels to select only
    samples the model has already learned well, ensuring the autoencoder
    learns an up-to-date boundary for "known" traffic.

    Returns (novelty_threshold, confidence_threshold) for adaptive thresholding.
    """
    eval_agent = PPOAgent(
        state_dim=X_train.shape[1],
        action_dim=num_classes,
        cfg=cfg.ppo,
        device=device,
        dataset=cfg.training.dataset,
    )
    eval_agent.set_model_state(aggregated_model)

    # Get prediction confidences on training data
    all_confs = []
    batch_size = 2048
    with torch.no_grad():
        for start in range(0, len(X_train), batch_size):
            batch = torch.FloatTensor(X_train[start:start + batch_size]).to(device)
            logits = eval_agent.actor(batch)  # [batch, num_classes]
            probs = torch.softmax(logits, dim=-1)
            conf, _ = probs.max(dim=-1)  # max probability per sample
            all_confs.append(conf.cpu().numpy())
    all_confs = np.concatenate(all_confs)

    # Select high-confidence samples (model is confident these are "known")
    CONF_THRESHOLD = 0.9
    confident_mask = all_confs >= CONF_THRESHOLD
    n_confident = confident_mask.sum()
    if n_confident < 100:
        # Not enough confident samples; keep existing threshold
        return None, None

    X_confident = X_train[confident_mask]
    print(f"    Novelty retrain: {n_confident}/{len(X_train)} high-confidence samples (conf>={CONF_THRESHOLD})")

    # Retrain autoencoder on high-confidence subset
    novelty_detector.train()
    nd_optim = torch.optim.Adam(novelty_detector.parameters(), lr=1e-3)
    batch_size_nd = 1024
    for nd_epoch in range(10):  # quick retrain (10 epochs vs original 20)
        perm = np.random.permutation(len(X_confident))
        for start in range(0, len(X_confident), batch_size_nd):
            idx = perm[start:start + batch_size_nd]
            batch = torch.FloatTensor(X_confident[idx]).to(device)
            recon = novelty_detector(batch)
            loss = ((batch - recon) ** 2).mean()
            nd_optim.zero_grad()
            loss.backward()
            nd_optim.step()

    # Recompute threshold from high-confidence training errors
    novelty_detector.eval()
    all_errors = []
    with torch.no_grad():
        for start in range(0, len(X_confident), batch_size):
            batch = torch.FloatTensor(X_confident[start:start + batch_size]).to(device)
            errs = novelty_detector.reconstruction_error(batch).cpu().numpy()
            all_errors.append(errs)
    all_errors = np.concatenate(all_errors)
    new_thresh = float(np.percentile(all_errors, cfg.training.novelty_threshold * 100))
    return new_thresh, CONF_THRESHOLD


def train_server_model(
    server_agent: PPOAgent,
    X_root: np.ndarray,
    y_root: np.ndarray,
    reward_cfg,
    num_classes: int,
    num_episodes: int = 3,
    max_steps: int = 2000,
) -> OrderedDict:
    """Train server model on root dataset and return its state."""
    env = MultiClassIDSEnvironment(
        X=X_root, y=y_root,
        reward_cfg=reward_cfg,
        num_classes=num_classes,
    )

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        step = 0
        while not done and step < max_steps:
            action, log_prob, value = server_agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            server_agent.store_transition(state, action, log_prob, reward, value, done)
            state = next_state
            step += 1

    server_agent.update()
    return server_agent.get_model_state()


def evaluate_global_model(
    model_state: OrderedDict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    cfg: Config,
    device: torch.device,
) -> Dict:
    """Evaluate a model state on test data using the PPO agent."""
    eval_agent = PPOAgent(
        state_dim=X_test.shape[1],
        action_dim=num_classes,
        cfg=cfg.ppo,
        device=device,
        dataset=cfg.training.dataset,
    )
    eval_agent.set_model_state(model_state)

    y_pred = []
    for i in range(len(X_test)):
        state = X_test[i].astype(np.float32)
        action_idx, _, _ = eval_agent.select_action(state, deterministic=True)
        # action_idx is already the predicted class index (int)
        y_pred.append(int(action_idx))

    y_pred = np.array(y_pred)
    binary_metrics = compute_binary_metrics(y_test, y_pred)
    multi_metrics = compute_multiclass_metrics(y_test, y_pred)

    return {**binary_metrics, **multi_metrics}


def evaluate_with_meta_agent(
    aggregated_model: OrderedDict,
    local_clients: list,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    cfg: Config,
    device: torch.device,
    meta_agent,
) -> Dict:
    """
    Evaluate using the meta-agent to coordinate local agent decisions.

    The meta-agent (Tier-2) observes the actions from all Tier-1 local agents
    and produces a refined final decision. This provides a comparison between
    the standard aggregated model and the meta-agent-coordinated approach.

    FIX (B): Properly integrates meta-agent output into evaluation so its
    contribution can be measured and validated.
    """
    eval_agent = PPOAgent(
        state_dim=X_test.shape[1],
        action_dim=num_classes,
        cfg=cfg.ppo,
        device=device,
        dataset=cfg.training.dataset,
    )
    eval_agent.set_model_state(aggregated_model)

    y_pred_meta = []
    y_pred_base = []

    for i in range(len(X_test)):
        state = X_test[i].astype(np.float32)

        # Base evaluation: use aggregated model directly (no meta-agent)
        action_idx, _, _ = eval_agent.select_action(state, deterministic=True)
        y_pred_base.append(int(action_idx))

        # Meta-agent evaluation: collect all local agent actions, then meta-agent decides
        all_action_vectors = []
        for client in local_clients:
            action_idx, _, _ = client.ppo.select_action(state, deterministic=True)
            # action_idx is a class index (int). Convert to one-hot for meta-agent input.
            one_hot = np.zeros(num_classes, dtype=np.float32)
            one_hot[int(action_idx)] = 1.0
            all_action_vectors.append(one_hot)

        agent_actions = np.stack(all_action_vectors)  # [num_clients, num_classes]
        meta_action_idx = meta_agent.predict_class(
            agent_actions, state, deterministic=True
        )
        y_pred_meta.append(int(meta_action_idx))

    y_pred_meta = np.array(y_pred_meta)
    y_pred_base = np.array(y_pred_base)

    meta_binary = compute_binary_metrics(y_test, y_pred_meta)
    meta_multi = compute_multiclass_metrics(y_test, y_pred_meta)
    base_binary = compute_binary_metrics(y_test, y_pred_base)
    base_multi = compute_multiclass_metrics(y_test, y_pred_base)

    def _safe_metric(metrics: Dict, key: str, default: float = 0.0) -> float:
        """Safely extract a metric with fallback, logging a warning if missing."""
        # Try both unprefixed and _weighted suffixed keys
        val = metrics.get(key)
        if val is not None:
            return val
        val_weighted = metrics.get(f"{key}_weighted")
        if val_weighted is not None:
            return val_weighted
        import warnings
        warnings.warn(f"[evaluate_with_meta_agent] Metric '{key}' not found in "
                      f"metrics dict. Keys present: {list(metrics.keys())}. "
                      f"Returning default={default}. This may indicate a metrics "
                      f"API mismatch.")
        return default

    return {
        "meta_accuracy": meta_binary["accuracy"],
        "meta_precision": _safe_metric(meta_multi, "precision"),
        "meta_recall": _safe_metric(meta_multi, "recall"),
        "meta_f1": _safe_metric(meta_multi, "f1_score"),
        "meta_fpr": meta_binary["fpr"],
        "base_accuracy": base_binary["accuracy"],
        "base_precision": _safe_metric(base_multi, "precision"),
        "base_recall": _safe_metric(base_multi, "recall"),
        "base_f1": _safe_metric(base_multi, "f1_score"),
        "base_fpr": base_binary["fpr"],
    }


def save_checkpoint(
    filepath: str,
    round_idx: int,
    aggregated_model: OrderedDict,
    server_state: OrderedDict,
    local_client_states: List[OrderedDict],
    reputations: List[float],
    history: Dict,
    best_accuracy: float,
    lr_states: Optional[Dict] = None,
    meta_agent_state: Optional[OrderedDict] = None,
    selector_state: Optional[OrderedDict] = None,
):
    """Save a full training checkpoint so training can be resumed."""
    ckpt = {
        "round_idx": round_idx,
        "aggregated_model": aggregated_model,
        "server_state": server_state,
        "local_client_states": local_client_states,
        "reputations": reputations,
        "history": history,
        "best_accuracy": best_accuracy,
    }
    if lr_states is not None:
        ckpt["lr_states"] = lr_states
    if meta_agent_state is not None:
        ckpt["meta_agent_state"] = meta_agent_state
    if selector_state is not None:
        ckpt["selector_state"] = selector_state
    torch.save(ckpt, filepath)


def load_checkpoint(filepath: str) -> Dict:
    """Load a training checkpoint."""
    return torch.load(filepath, map_location="cpu", weights_only=False)


def run_training(cfg: Config, resume_checkpoint: Optional[str] = None):
    """Main federated RL training loop."""
    print("=" * 70)
    print("  Federated RL-IDS: PPO + FedTrust + Fed+ + Dynamic Attention")
    print("=" * 70)

    # Setup
    device = torch.device(
        cfg.training.device if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    os.makedirs(cfg.training.output_dir, exist_ok=True)
    # Append dataset name to output_dir so results are stored per-dataset
    cfg.training.output_dir = os.path.join(cfg.training.output_dir, f"outputs_{cfg.training.dataset}")
    os.makedirs(cfg.training.output_dir, exist_ok=True)

    # ── Load data ──
    print("\n[1/5] Loading dataset...")
    X_train, X_test, y_train, y_test, le = load_dataset(cfg)
    num_classes = len(le.classes_)
    state_dim = X_train.shape[1]
    action_dim = num_classes

    # Update PPO config
    cfg.ppo.action_dim = action_dim

    # Adaptive entropy coefficient: scale with num_classes to prevent action collapse
    base_entropy = cfg.ppo.entropy_coef
    adaptive_entropy = base_entropy * max(1.0, num_classes / 3.0)
    cfg.ppo.entropy_coef = min(adaptive_entropy, 0.1)  # cap at 0.1
    print(f"  Adaptive entropy coef: {cfg.ppo.entropy_coef:.4f} (num_classes={num_classes})")

    # Periodic novelty retraining interval (FIX F: make it non-static)
    novelty_retrain_interval = getattr(cfg.training, "novelty_retrain_interval", 50)
    if novelty_retrain_interval > 0:
        print(f"  Novelty detector retraining every {novelty_retrain_interval} rounds")

    # ── Create root dataset for FLTrust ──
    print("[2/5] Creating root dataset & partitioning data...")
    X_root, y_root = create_root_dataset(
        X_train, y_train,
        size=cfg.fed_trust.root_dataset_size,
        balanced=cfg.fed_trust.root_dataset_per_class,
        seed=cfg.training.seed,
    )
    print(f"  Root dataset: {len(X_root)} samples")

    # ── Partition data (non-IID for realistic healthcare setting) ──
    partitions = partition_data_non_iid(
        X_train, y_train,
        num_clients=cfg.training.num_clients,
        seed=cfg.training.seed,
    )
    for i, (xp, yp) in enumerate(partitions):
        unique, counts = np.unique(yp, return_counts=True)
        print(f"  Client {i}: {len(xp)} samples, classes: {dict(zip(unique, counts))}")

    # ── Create local clients (Tier 1) ──
    print("[3/5] Initialising clients & federated components...")
    local_clients: List[LocalClient] = []
    for i, (xp, yp) in enumerate(partitions):
        # Split client data into train/test for attention computation
        split = int(len(xp) * 0.8)
        client = LocalClient(
            client_id=i,
            X_train=xp[:split],
            y_train=yp[:split],
            X_test=xp[split:],
            y_test=yp[split:],
            num_classes=num_classes,
            cfg=cfg,
            device=device,
        )
        local_clients.append(client)

    # Server PPO agent (for FLTrust)
    server_agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        cfg=cfg.ppo,
        device=device,
        agent_id=-1,
        dataset=cfg.training.dataset,
    )

    # Federated aggregator
    aggregator = FederatedAggregator(cfg, device)

    # Meta-agent (Tier 2)
    meta_agent = None
    if cfg.training.meta_agent_enabled:
        meta_agent = MetaAgent(
            num_agents=cfg.training.num_clients,
            action_dim=action_dim,
            state_dim=state_dim,
            cfg=cfg,
            device=device,
        )

    # RL-based Client Selector (Tier 3)
    client_selector = None
    if cfg.training.client_selection_enabled:
        client_selector = RLClientSelector(
            num_clients=cfg.training.num_clients,
            state_dim_per_client=8,     # +gradient_alignment + minority_class_fraction
            hidden_dim=cfg.training.selector_hidden_dim,
            cfg=cfg.ppo,
            device=device,
            total_rounds=cfg.training.num_rounds,
        )
        k_init = cfg.training.clients_per_round
        k_min = max(3, k_init // 2)
        print(f"  RL Client Selector enabled: curriculum K_sel {k_init}->{k_min}")

    # ── Initialise global model ──
    init_state = local_clients[0].get_model_state()
    aggregator.set_global_model(init_state)
    for client in local_clients:
        client.set_model_state(init_state)
    server_agent.set_model_state(init_state)

    # ── Setup LR schedulers for all clients (prevents catastrophic forgetting) ──
    if cfg.ppo.lr_scheduler_enabled:
        total_rounds = cfg.training.num_rounds
        min_lr_actor = cfg.ppo.lr_actor * cfg.ppo.lr_min_factor
        min_lr_critic = cfg.ppo.lr_critic * cfg.ppo.lr_min_factor
        for client in local_clients:
            client.ppo.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                client.ppo.actor_optim, T_max=total_rounds, eta_min=min_lr_actor
            )
            client.ppo.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                client.ppo.critic_optim, T_max=total_rounds, eta_min=min_lr_critic
            )
        server_agent.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            server_agent.actor_optim, T_max=total_rounds, eta_min=min_lr_actor
        )
        server_agent.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            server_agent.critic_optim, T_max=total_rounds, eta_min=min_lr_critic
        )
        print(f"  LR schedulers: CosineAnnealing over {total_rounds} rounds")

    # ── Resume from checkpoint if provided ──
    start_round = 0
    best_accuracy = 0.0
    history = {
        "rounds": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "f1_macro": [],          # Task 6: macro F1 (equal-weight per class)
        "fpr": [],
        "trust_scores": [],
        "attention_values": [],
        "client_accuracies": [],
        "selected_clients": [],
        "meta_agent_accuracy": [],
        "meta_agent_f1": [],
    }

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"\n  Resuming from checkpoint: {resume_checkpoint}")
        ckpt = load_checkpoint(resume_checkpoint)
        start_round = ckpt["round_idx"] + 1
        best_accuracy = ckpt["best_accuracy"]
        history = ckpt["history"]

        # Move checkpoint tensors to target device (checkpoint was saved on GPU, loaded to CPU)
        def to_device(state_dict, device):
            return {k: v.to(device) for k, v in state_dict.items()}

        # Restore model states
        aggregated_model = to_device(ckpt["aggregated_model"], device)
        aggregator.set_global_model(aggregated_model)
        server_agent.set_model_state(to_device(ckpt["server_state"], device))
        for i, client in enumerate(local_clients):
            if i < len(ckpt["local_client_states"]):
                client.set_model_state(to_device(ckpt["local_client_states"][i], device))

        # Restore reputations
        aggregator.fl_trust.reputations = ckpt["reputations"]

        # Restore meta-agent state if present in checkpoint
        if "meta_agent_state" in ckpt and meta_agent is not None:
            meta_agent.set_state(ckpt["meta_agent_state"])

        # Restore client selector state if present in checkpoint
        if "selector_state" in ckpt and client_selector is not None:
            client_selector.set_state(to_device(ckpt["selector_state"], device))

        # Restore LR scheduler states
        if "lr_states" in ckpt and cfg.ppo.lr_scheduler_enabled:
            lr_states = ckpt["lr_states"]
            for i, client in enumerate(local_clients):
                if client.ppo.actor_scheduler and f"client_{i}_actor" in lr_states:
                    client.ppo.actor_scheduler.load_state_dict(lr_states[f"client_{i}_actor"])
                    client.ppo.critic_scheduler.load_state_dict(lr_states[f"client_{i}_critic"])
            if server_agent.actor_scheduler and "server_actor" in lr_states:
                server_agent.actor_scheduler.load_state_dict(lr_states["server_actor"])
                server_agent.critic_scheduler.load_state_dict(lr_states["server_critic"])

        print(f"  Resumed at round {start_round}/{cfg.training.num_rounds}, best_acc={best_accuracy:.4f}")

    # ── Train novelty detector (autoencoder on known traffic) ──
    print("  Training novelty detector (autoencoder)...")
    novelty_detector = NoveltyDetector(input_dim=state_dim, latent_dim=32).to(device)
    nd_optim = torch.optim.Adam(novelty_detector.parameters(), lr=1e-3)
    nd_batch_size = 2048
    for nd_epoch in range(20):
        perm = np.random.permutation(len(X_train))
        for nd_start in range(0, min(len(X_train), nd_batch_size * 4), nd_batch_size):
            nd_idx = perm[nd_start:nd_start + nd_batch_size]
            batch = torch.FloatTensor(X_train[nd_idx]).to(device)
            recon = novelty_detector(batch)
            nd_loss = ((batch - recon) ** 2).mean()
            nd_optim.zero_grad()
            nd_loss.backward()
            nd_optim.step()

    # Compute threshold at configured percentile of training errors (batched)
    novelty_detector.eval()
    all_errors_list = []
    with torch.no_grad():
        for nd_start in range(0, len(X_train), nd_batch_size):
            batch = torch.FloatTensor(X_train[nd_start:nd_start + nd_batch_size]).to(device)
            errs = novelty_detector.reconstruction_error(batch).cpu().numpy()
            all_errors_list.append(errs)
    all_errors = np.concatenate(all_errors_list)
    novelty_thresh = float(np.percentile(all_errors, cfg.training.novelty_threshold * 100))
    print(f"  Novelty threshold (p{cfg.training.novelty_threshold*100:.0f}): {novelty_thresh:.6f}")

    # Inject novelty detector into agent environments (on CPU for env inference)
    novelty_detector_cpu = novelty_detector.cpu()
    for client in local_clients:
        client.env._novelty_detector = novelty_detector_cpu
        client.env._novelty_threshold = novelty_thresh
        client.test_env._novelty_detector = novelty_detector_cpu
        client.test_env._novelty_threshold = novelty_thresh

    # ── Training loop ──
    # Auto-adjust eval interval for short runs
    eval_interval = cfg.training.eval_interval
    if cfg.training.num_rounds <= 20:
        eval_interval = 1
    print(f"\n[4/5] Starting federated training ({cfg.training.num_rounds} rounds, eval every {eval_interval})...")
    if start_round > 0:
        print(f"  Resuming from round {start_round + 1}")

    # Carry-over evaluation metrics across rounds so the selector reward always
    # uses the most recent real evaluation (FIX 4 — selector eval timing bug).
    eval_metrics: Dict[str, float] = {
        "accuracy": 0.0, "precision": 0.0, "recall": 0.0,
        "f1_score": 0.0, "f1_macro": 0.0, "fpr": 0.0,
    }
    meta_eval = None
    # Bug 4 fix: persist previous-round losses so selector always has real losses
    # (not 0.0 placeholders) for the state features at Step 0.
    prev_round_losses: List[float] = [0.0] * cfg.training.num_clients

    for round_idx in range(start_round, cfg.training.num_rounds):
        round_start = time.time()

        # ── Step 0: RL Client Selection (Tier 3) ──────────────────────────────
        if client_selector is not None:
            reputations = aggregator.fl_trust.reputations[:cfg.training.num_clients]

            # Curriculum K_sel: decays from K_init → K_min over rounds
            k_init = cfg.training.clients_per_round
            k_min = max(3, k_init // 2)
            k_sel = RLClientSelector.k_sel_schedule(
                round_idx, k_init=k_init, k_min=k_min,
                total_rounds=cfg.training.num_rounds,
                num_clients=cfg.training.num_clients,
            )

            # Build per-client state features for selector (7 features per client)
            data_shares = [
                client.num_train_samples / sum(c.num_train_samples for c in local_clients)
                for client in local_clients
            ]

            # Use previous-round divergences and gradient alignments (stale, as designed)
            # These are available from the previous round's end state
            prev_divergences = getattr(client_selector, '_prev_divergences', None)
            prev_gradient_aligns = getattr(client_selector, '_prev_gradient_alignments', None)
            if prev_divergences is None:
                prev_divergences = [0.0] * cfg.training.num_clients
            if prev_gradient_aligns is None:
                prev_gradient_aligns = [0.0] * cfg.training.num_clients

            # Attention from previous round (proxy for trust × attention signal)
            prev_attention = history["attention_values"][-1] if history["attention_values"] else \
                             [1.0 / cfg.training.num_clients] * cfg.training.num_clients

            selected_indices = client_selector.select_clients(
                reputations=reputations,
                attention_weights=prev_attention,
                client_losses=prev_round_losses,  # Bug 4 fix: use previous round's real losses
                model_divergences=prev_divergences,
                gradient_alignments=prev_gradient_aligns,
                data_shares=data_shares,
                deterministic=(round_idx % cfg.training.selector_eval_interval != 0),
                k_sel=k_sel,
                minority_fractions=[c.minority_class_fraction for c in local_clients],  # Task 3 Option A
            )
        else:
            selected_indices = list(range(cfg.training.num_clients))

        selected_clients = [local_clients[i] for i in selected_indices]
        selected_local_models = [client.get_model_state() for client in selected_clients]

        # ── Step A: Local training (Tier 1) — selected clients only ──────────
        # FIX (Global Start): Save pre-train snapshots BEFORE local training.
        # This ensures delta = post_train - pre_train is a pure training update,
        # not contaminated by personalisation offsets from previous rounds.
        pre_train_models = [client.get_model_state() for client in selected_clients]

        client_metrics = []
        for client in selected_clients:
            metrics = client.train_local(num_episodes=cfg.training.local_episodes)
            client_metrics.append(metrics)

        # ── Step B: Train server model on root dataset ──
        server_state = train_server_model(
            server_agent, X_root, y_root,
            cfg.reward, num_classes,
            num_episodes=5,
            max_steps=min(len(X_root), cfg.training.max_steps_per_episode),
        )

        # ── Step C: Evaluate all clients for selector state (not just selected) ─
        #   Evaluates ALL clients so selector has complete per-client F1/loss state
        all_client_accuracies = []
        all_client_f1s = []
        all_client_losses = []
        for client in local_clients:
            acc, loss = client.evaluate_on_test()
            all_client_accuracies.append(acc)
            all_client_f1s.append(client.env.get_accuracy())  # per-round F1 proxy
            all_client_losses.append(loss)

        # Compute gradient alignments for ALL clients (for selector state)
        # g_k = cos(Δ_k, Δ_glob): measures alignment with global gradient direction
        # Bug 5 fix: compute explicit deltas = post_train - pre_train so cosine
        # similarity measures gradient alignment, not raw model magnitude.
        # For unselected clients (no training), use zero delta.
        all_gradient_alignments = []
        # Build post/pre model dicts for selected clients
        selected_posts = {i: client.get_model_state() for i, client in zip(selected_indices, selected_clients)}
        selected_pres = {i: pre for i, pre in zip(selected_indices, pre_train_models)}
        # Server delta
        server_delta = aggregator.compute_update(server_state, aggregator.global_model)
        for i in range(cfg.training.num_clients):
            if i in selected_indices:
                delta = aggregator.compute_update(selected_posts[i], selected_pres[i])
                all_gradient_alignments.append(delta)
            else:
                # Unselected: zero delta (no training update)
                all_gradient_alignments.append(OrderedDict(
                    (k, torch.zeros_like(v)) for k, v in aggregator.global_model.items()
                ))
        if server_delta:
            raw_alignments = compute_gradient_alignment(
                local_updates=all_gradient_alignments,
                global_update=server_delta,
                device=device,
            )
            all_gradient_alignments = list(raw_alignments)  # aligned with 0..K-1 order

        # ── Step D: Federated aggregation — selected clients only ──────────────
        #   Only selected clients contribute to aggregation (RL selection policy)
        selected_client_infos = [
            {
                "num_samples": client.num_train_samples,
                "accuracy": acc,
                "loss": client.current_loss,
            }
            for client, acc in zip(selected_clients,
                                    [all_client_accuracies[i] for i in selected_indices])
        ]

        # Collect POST-training model states for aggregation
        post_train_models = [client.get_model_state() for client in selected_clients]

        aggregated_model, trust_scores, attention_values = aggregator.aggregate_round(
            local_models=post_train_models,
            server_model=server_state,
            client_infos=selected_client_infos,
            selected_indices=selected_indices,
            minority_fractions=[
                local_clients[i].minority_class_fraction for i in selected_indices
            ],
            pre_train_models=pre_train_models,  # FIX: Global Start — clean deltas
        )

        # ── Step E: Apply Fed+ personalisation — selected clients only ──────────
        # FIX (Global Start): First set client to the NEW global model,
        # then apply personalisation on top. This ensures the personalisation
        # offset theta_k is computed relative to the fresh global model,
        # not the old personalised state.
        for client in selected_clients:
            # Step 1: Give client the clean global model
            client.set_model_state(aggregated_model)
            # Step 2: Apply personalisation on top of global model
            personalised = aggregator.personalise_for_agent(
                agent_id=client.client_id,
                local_model=client.get_model_state(),
                eta=cfg.ppo.lr_actor,
            )
            client.set_model_state(personalised)

        # Update server model with aggregated
        server_agent.set_model_state(aggregated_model)

        # ── Step E.1: Step LR schedulers (prevents catastrophic forgetting) ──
        if cfg.ppo.lr_scheduler_enabled:
            for client in selected_clients:
                if client.ppo.actor_scheduler is not None:
                    client.ppo.actor_scheduler.step()
                    client.ppo.critic_scheduler.step()
            if server_agent.actor_scheduler is not None:
                server_agent.actor_scheduler.step()
                server_agent.critic_scheduler.step()

        # ── Step F: Periodic novelty detector retraining (FIX F) ──
        # Retrain autoencoder on high-confidence samples from current global model
        # to keep the novelty boundary adaptive to evolving patterns.
        if novelty_retrain_interval > 0 and (round_idx + 1) % novelty_retrain_interval == 0:
            new_thresh, conf_thresh = retrain_novelty_detector(
                novelty_detector=novelty_detector,
                X_train=X_train,
                y_train=y_train,
                aggregated_model=aggregated_model,
                X_test=X_test,
                y_test=y_test,
                num_classes=num_classes,
                cfg=cfg,
                device=device,
            )
            if new_thresh is not None:
                novelty_thresh = new_thresh
                novelty_detector_cpu = novelty_detector.cpu()
                for client in local_clients:
                    client.env._novelty_detector = novelty_detector_cpu
                    client.env._novelty_threshold = novelty_thresh
                    client.test_env._novelty_detector = novelty_detector_cpu
                    client.test_env._novelty_threshold = novelty_thresh
                novelty_detector.to(device)
                print(f"    Novelty detector retrained: thresh={novelty_thresh:.6f} (conf>={conf_thresh})")

        # FIX 4: drop placeholder eval_metrics/meta_eval defaults — these used to
        # be overwritten with zeros before Step G.2 ran, polluting the selector reward.
        # We now keep the previous round's real metrics in `eval_metrics` / `meta_eval`
        # (initialised outside the loop) and refresh them BEFORE Step G.2.
        selector_loss_info = ""  # initialise before G.2 block which may skip

        # ── Step G: Meta-agent RL training (Tier 2) ──
        # FIX (Meta-Agent Illusion): Train Meta-Agent on the Root Dataset (X_root),
        # NOT the local testing set. This grounds the Meta-Agent in a globally
        # representative distribution, preventing it from just learning to ensemble
        # locally overfitted client predictions.
        if meta_agent is not None and round_idx % cfg.training.meta_agent_eval_interval == 0:
            sample_count = min(200, len(X_root))
            for idx in range(sample_count):
                state = X_root[idx].astype(np.float32)
                true_label = y_root[idx]

                # Collect all local agent actions as one-hot vectors
                # FIX (B): select_action returns class index (int), not one-hot.
                # Convert to one-hot [num_classes] per agent for meta-agent input.
                all_action_vectors = []
                for client in local_clients:
                    action_idx, _, _ = client.ppo.select_action(state, deterministic=True)
                    one_hot = np.zeros(num_classes, dtype=np.float32)
                    one_hot[int(action_idx)] = 1.0
                    all_action_vectors.append(one_hot)
                agent_actions = np.stack(all_action_vectors)  # [num_agents, num_classes]

                # Meta-agent selects action via its PPO policy
                meta_action, meta_lp, meta_val = meta_agent.select_action(agent_actions, state)

                # Reward: correct classification with class-balanced weighting (Fix 6)
                # Uses inverse-frequency weighting so minority-class correct predictions
                # get higher reward, preventing meta-agent from collapsing to majority class
                predicted_class = int(np.argmax(meta_action))
                if predicted_class == true_label:
                    # Correct: reward by inverse class frequency (rarer class = higher reward)
                    cls_freq = y_root[:sample_count].tolist().count(true_label) / sample_count
                    freq_weight = 1.0 / (cls_freq + 1e-8)
                    freq_weight = min(freq_weight, 3.0)  # cap at 3x
                    meta_reward = 1.0 * freq_weight
                else:
                    # Wrong: penalty proportional to how common the true class is
                    cls_freq = y_root[:sample_count].tolist().count(true_label) / sample_count
                    penalty_scale = max(0.1, cls_freq)  # lower penalty for rare-class errors
                    meta_reward = -0.5 * penalty_scale
                done = (idx == sample_count - 1)

                meta_agent.store_transition(
                    state, agent_actions, meta_action, meta_lp, meta_reward, meta_val, done
                )

            meta_agent.update()

        # ── Evaluation (FIX 4: BEFORE record_selection so selector reward uses real metrics) ──
        meta_info = ""
        meta_eval_refreshed_this_round = False
        if (round_idx + 1) % eval_interval == 0 or round_idx == 0:
            eval_metrics = evaluate_global_model(
                aggregated_model, X_test, y_test,
                num_classes, cfg, device,
            )

            # Meta-agent evaluation (periodic, to measure Tier-2 contribution)
            if (meta_agent is not None
                    and cfg.training.meta_agent_enabled
                    and (round_idx + 1) % cfg.training.meta_agent_eval_interval == 0):
                meta_eval = evaluate_with_meta_agent(
                    aggregated_model, local_clients,
                    X_test, y_test, num_classes, cfg, device, meta_agent,
                )
                meta_info = (f"Meta-Acc: {meta_eval['meta_accuracy']:.4f} | "
                             f"Meta-F1: {meta_eval['meta_f1']:.4f}")
            else:
                meta_eval = None
            meta_eval_refreshed_this_round = True

        # ── Step G.2: Selector PPO update (Tier 3) ───────────────────────────────
        if (client_selector is not None
                and (round_idx + 1) % cfg.training.selector_eval_interval == 0):

            # Use meta-agent accuracy in reward if available; else fall back to global
            meta_acc_for_reward = (
                meta_eval["meta_accuracy"]
                if meta_eval is not None else eval_metrics["accuracy"]
            )

            # Get Bernoulli probabilities and hybrid scores for reward / record
            reputations = aggregator.fl_trust.reputations[:cfg.training.num_clients]
            prev_attn = history["attention_values"][-1] if history["attention_values"] else \
                        [1.0 / cfg.training.num_clients] * cfg.training.num_clients
            data_shares = [
                client.num_train_samples / sum(c.num_train_samples for c in local_clients)
                for client in local_clients
            ]
            bernoulli_probs, hybrid_scores = client_selector.get_hybrid_scores(
                reputations=reputations,
                attention_weights=prev_attn,
                client_losses=all_client_losses,
                model_divergences=[compute_model_divergence(
                    client.get_model_state(), aggregator.global_model
                ) for client in local_clients],
                gradient_alignments=all_gradient_alignments,
                data_shares=data_shares,
                minority_fractions=[c.minority_class_fraction for c in local_clients],  # Task 3 Option A
            )

            # Record this round's selection for selector training
            client_selector.record_selection(
                selected_indices=selected_indices,
                global_accuracy=eval_metrics["accuracy"],
                meta_accuracy=meta_acc_for_reward,
                trust_scores=trust_scores,
                client_divergences=[
                    compute_model_divergence(client.get_model_state(), aggregator.global_model)
                    for client in local_clients
                ],
                current_probs=bernoulli_probs,
            )

            # Update F1 EMAs from all clients (selector learns from all clients' performance)
            client_selector.update_f1_from_round(
                selected_indices=selected_indices,
                client_accuracies=all_client_accuracies,
                client_f1s=all_client_f1s,
            )

            # Store divergences and gradient alignments for next round's state
            client_selector._prev_divergences = [
                compute_model_divergence(client.get_model_state(), aggregator.global_model)
                for client in local_clients
            ]
            client_selector._prev_gradient_alignments = all_gradient_alignments

            # PPO update on collected selector transitions (with entropy decay via round_idx)
            selector_info = client_selector.update(round_idx=round_idx)
            if selector_info:
                selector_loss_info = (
                    f"Sel-Actor: {selector_info.get('selector_actor_loss', 0):.4f} | "
                    f"Sel-Critic: {selector_info.get('selector_critic_loss', 0):.4f} | "
                    f"H: {selector_info.get('selector_entropy', 0):.3f}"
                )
            else:
                selector_loss_info = ""
            # ── Evaluation ──

        round_time = time.time() - round_start

        if meta_eval_refreshed_this_round:
            # History append for meta-agent (eval already ran above before Step G.2)
            if meta_eval is not None:
                history["meta_agent_accuracy"].append(meta_eval["meta_accuracy"])
                history["meta_agent_f1"].append(meta_eval["meta_f1"])
            else:
                history["meta_agent_accuracy"].append(None)
                history["meta_agent_f1"].append(None)

            history["rounds"].append(round_idx + 1)
            history["meta_agent_enabled"] = cfg.training.meta_agent_enabled  # Task 7: ablation record
            history["accuracy"].append(eval_metrics["accuracy"])
            history["precision"].append(eval_metrics["precision"])
            history["recall"].append(eval_metrics["recall"])
            history["f1"].append(eval_metrics["f1_score"])
            history["f1_macro"].append(eval_metrics.get("f1_macro", 0.0))   # Task 6
            history["fpr"].append(eval_metrics["fpr"])
            history["trust_scores"].append(trust_scores)
            history["attention_values"].append(
                [float(a) for a in attention_values]
            )
            history["client_accuracies"].append(all_client_accuracies)
            history["selected_clients"].append(selected_indices)

            # Get temporal reputations for logging
            reputations = aggregator.fl_trust.reputations[:cfg.training.num_clients]

            # Curriculum K_sel for logging
            k_init = cfg.training.clients_per_round
            k_min = max(3, k_init // 2)
            k_sel_this_round = RLClientSelector.k_sel_schedule(
                round_idx, k_init=k_init, k_min=k_min,
                total_rounds=cfg.training.num_rounds,
                num_clients=cfg.training.num_clients,
            )

            # Task 6: log f1_macro and minority-class recall
            f1_macro_val = eval_metrics.get("f1_macro", 0.0)
            # Minority class recall (first non-zero class or class 1)
            recall_per_class = eval_metrics.get("recall_per_class", {})
            minority_recall = recall_per_class.get(1, recall_per_class.get(min(recall_per_class.keys()), 0.0)) if recall_per_class else 0.0

            print(
                f"Round {round_idx+1:4d}/{cfg.training.num_rounds} | "
                f"Acc: {eval_metrics['accuracy']:.4f} | "
                f"Prec: {eval_metrics['precision']:.4f} | "
                f"Rec: {eval_metrics['recall']:.4f} | "
                f"F1: {eval_metrics['f1_score']:.4f} | "
                f"F1m: {f1_macro_val:.4f} | "   # Task 6: macro F1
                f"FPR: {eval_metrics['fpr']:.4f} | "
                f"MinRec: {minority_recall:.4f} | "   # Task 6: minority class recall
                f"K_sel: {k_sel_this_round} | "
                f"Sel: {selected_indices} | "
                f"Trust: [{', '.join(f'{s:.2f}' for s in trust_scores)}] | "
                f"Rep: [{', '.join(f'{r:.2f}' for r in reputations)}]"
                + (f" | {meta_info}" if meta_info else "")
                + (f" | {selector_loss_info}" if selector_loss_info else "")
                + f" | Time: {round_time:.1f}s"
            )

            # Save best model
            if eval_metrics["accuracy"] > best_accuracy:
                best_accuracy = eval_metrics["accuracy"]
                torch.save(
                    aggregated_model,
                    os.path.join(cfg.training.output_dir, "best_model.pt"),
                )
                if meta_agent is not None:
                    torch.save(
                        meta_agent.get_state(),
                        os.path.join(cfg.training.output_dir, "best_meta_agent.pt"),
                    )
                if client_selector is not None:
                    torch.save(
                        client_selector.get_state(),
                        os.path.join(cfg.training.output_dir, "best_selector.pt"),
                    )

        # ── Periodic save ──
        if (round_idx + 1) % cfg.training.save_interval == 0:
            torch.save(
                aggregated_model,
                os.path.join(
                    cfg.training.output_dir,
                    f"model_round_{round_idx+1}.pt",
                ),
            )

        # ── Checkpoint every round for resume capability ──
        lr_states = {}
        if cfg.ppo.lr_scheduler_enabled:
            for i, client in enumerate(local_clients):
                if client.ppo.actor_scheduler:
                    lr_states[f"client_{i}_actor"] = client.ppo.actor_scheduler.state_dict()
                    lr_states[f"client_{i}_critic"] = client.ppo.critic_scheduler.state_dict()
            if server_agent.actor_scheduler:
                lr_states["server_actor"] = server_agent.actor_scheduler.state_dict()
                lr_states["server_critic"] = server_agent.critic_scheduler.state_dict()

        save_checkpoint(
            filepath=os.path.join(cfg.training.output_dir, "checkpoint_latest.pt"),
            round_idx=round_idx,
            aggregated_model=aggregated_model,
            server_state=server_agent.get_model_state(),
            local_client_states=[a.get_model_state() for a in local_clients],
            reputations=list(aggregator.fl_trust.reputations[:cfg.training.num_clients]),
            history=history,
            best_accuracy=best_accuracy,
            lr_states=lr_states if lr_states else None,
            meta_agent_state=meta_agent.get_state() if meta_agent is not None else None,
            selector_state=client_selector.get_state() if client_selector is not None else None,
        )

        # Bug 4 fix: update previous-round losses for next round's selector state
        prev_round_losses = list(all_client_losses)

        # Also save history incrementally (JSON, lightweight)
        history_path = os.path.join(cfg.training.output_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2, default=str)

    # ── Final evaluation ──
    print("\n[5/5] Final evaluation...")
    final_metrics = evaluate_global_model(
        aggregated_model, X_test, y_test,
        num_classes, cfg, device,
    )
    print_metrics(final_metrics, prefix=f"FINAL — {cfg.training.dataset.upper()}")

    # Save history
    history_path = os.path.join(cfg.training.output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)

    # Save final model
    torch.save(
        aggregated_model,
        os.path.join(cfg.training.output_dir, "final_model.pt"),
    )

    print(f"\nBest accuracy: {best_accuracy:.4f}")
    print(f"Results saved to: {cfg.training.output_dir}")

    return history, final_metrics


def run_multi_seed(cfg: Config) -> Dict[str, Dict[str, float]]:
    """
    Run federated training with multiple random seeds for statistical rigor.

    Each seed runs an independent full training run. Results are aggregated
    to report mean ± standard deviation across seeds, providing confidence
    intervals for all metrics rather than single-run point estimates.

    Returns a dict of per-seed result dicts and a summary dict with
    mean ± std for each metric.
    """
    seeds = cfg.training.seeds
    print("=" * 70)
    print(f"  Multi-Seed Sweep: {len(seeds)} seeds {seeds}")
    print("=" * 70)

    all_runs = []
    for i, seed in enumerate(seeds):
        print(f"\n{'─' * 70}")
        print(f"  Seed {i + 1}/{len(seeds)}: seed={seed}")
        print(f"{'─' * 70}")
        cfg_run = Config()
        # Copy all training settings from input cfg
        cfg_run.training = cfg.training
        cfg_run.ppo = cfg.ppo
        cfg_run.fed_trust = cfg.fed_trust
        cfg_run.fed_plus = cfg.fed_plus
        cfg_run.attention = cfg.attention
        cfg_run.reward = cfg.reward
        # Override seed
        cfg_run.training.seed = seed
        # Override output dir to include seed
        base_output = cfg_run.training.output_dir
        cfg_run.training.output_dir = os.path.join(base_output, f"seed_{seed}")
        os.makedirs(cfg_run.training.output_dir, exist_ok=True)

        history, final_metrics = run_training(cfg_run, resume_checkpoint=None)
        all_runs.append({
            "seed": seed,
            "history": history,
            "final_metrics": final_metrics,
        })

    # Aggregate statistics across seeds
    metric_names = ["accuracy", "precision", "recall", "f1_score", "fpr"]
    summary = {}
    for metric in metric_names:
        values = [run["final_metrics"].get(metric, 0.0) for run in all_runs]
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        summary[metric] = {"mean": mean_val, "std": std_val}
        print(f"  {metric:20s}: {mean_val:.4f} ± {std_val:.4f}")

    # Save aggregated summary
    summary_path = os.path.join(cfg.training.output_dir, "multi_seed_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"runs": all_runs, "summary": summary}, f, indent=2, default=str)
    print(f"\n  Multi-seed summary saved to: {summary_path}")

    return all_runs, summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Federated RL-IDS Training")
    parser.add_argument("--dataset", type=str, default="edge_iiot",
                        choices=["edge_iiot", "nsl_kdd", "iomt_2024", "unsw_nb15"])
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--num_rounds", type=int, default=10)
    parser.add_argument("--local_episodes", type=int, default=5)
    parser.add_argument(
        "--max_steps_per_episode",
        type=int,
        default=None,
        help="Cap RL env steps per episode (default: use TrainingConfig, usually 2000). "
        "Lower for smoke tests, e.g. 128.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint_latest.pt to resume training")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Multiple seeds for multi-run statistical rigor (e.g., --seeds 42 123 777)")
    parser.add_argument("--novelty_retrain_interval", type=int, default=50,
                        help="Retrain novelty autoencoder every N rounds (0=disabled)")
    parser.add_argument("--meta_agent", action="store_true", default=False,
                        help="Enable meta-agent (Tier-2). Use --no-meta_agent to disable.")
    args = parser.parse_args()

    cfg = Config()
    cfg.training.dataset = args.dataset
    cfg.training.num_clients = args.num_clients
    cfg.training.num_rounds = args.num_rounds
    cfg.training.local_episodes = args.local_episodes
    if args.max_steps_per_episode is not None:
        cfg.training.max_steps_per_episode = args.max_steps_per_episode
    cfg.training.device = args.device
    cfg.training.seed = args.seed
    cfg.training.novelty_retrain_interval = args.novelty_retrain_interval
    cfg.training.meta_agent_enabled = args.meta_agent  # Task 7: ablation flag

    if args.seeds is not None:
        cfg.training.seeds = args.seeds
        run_multi_seed(cfg)
    else:
        run_training(cfg, resume_checkpoint=args.resume)

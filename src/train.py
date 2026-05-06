
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
from src.agents.ppo_agent import PPOAgent
from src.federated.aggregator import FederatedAggregator
from src.federated.client_selector import RLClientSelector, compute_gradient_alignment, compute_model_divergence
from src.environment.ids_env import MultiClassIDSEnvironment
from src.utils.metrics import (
    compute_binary_metrics,
    compute_multiclass_metrics,
    print_metrics,
)


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

    #  Use ONLY multi-class metrics for multi-class IDS.
    # Previously this returned {**binary_metrics, **multi_metrics} which caused
    # "recall == accuracy" because binary recall = TP/(TP+FN) = accuracy only when
    # num_classes=1 or specific class distributions. For multi-class, binary recall
    # is fundamentally different from multiclass weighted recall.
    # We now return only multi-class metrics and explicitly compute FPR separately
    # using a binary binarisation (Benign=0, Attack=1).
    multi_metrics = compute_multiclass_metrics(y_test, y_pred)

    # Compute binary FPR separately: benign samples (class 0) flagged as attack
    y_test_bin = (y_test != 0).astype(int)
    y_pred_bin = (y_pred != 0).astype(int)
    binary_fpr = compute_binary_metrics(y_test_bin, y_pred_bin)["fpr"]

    result = {**multi_metrics, "fpr": binary_fpr}
    return result


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
    if selector_state is not None:
        ckpt["selector_state"] = selector_state
    torch.save(ckpt, filepath)


def load_checkpoint(filepath: str) -> Dict:
    """Load a training checkpoint."""
    return torch.load(filepath, map_location="cpu", weights_only=False)


def run_training(cfg: Config, resume_checkpoint: Optional[str] = None):
    """Main federated RL training loop.

    Architecture: CNN-GRU-CBAM + PPO (Tier-1) + RL Selector (Tier-2) + FLTrust.
    Removed: Meta-Agent, Novelty Detector, Dynamic Attention, Fed+.
    """
    print("=" * 70)
    print("  FedRL-IDS: PPO + FLTrust + RL Client Selector (Resource Efficiency)")
    print("=" * 70)

    # Setup
    device = torch.device(
        cfg.training.device if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

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

    # RL-based Client Selector (Tier 2)
    client_selector = None
    if cfg.training.client_selection_enabled:
        client_selector = RLClientSelector(
            num_clients=cfg.training.num_clients,
            state_dim_per_client=7,   # 7 features: trust, loss, divergence, grad alignment, F1, data share, minority
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
        "f1_macro": [],
        "fpr": [],
        "trust_scores": [],
        "client_accuracies": [],
        "selected_clients": [],
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
    # Persist previous-round losses for selector state features
    prev_round_losses: List[float] = [0.0] * cfg.training.num_clients

    for round_idx in range(start_round, cfg.training.num_rounds):
        round_start = time.time()

        # ── Step 0: RL Client Selection (Tier 2) ──────────────────────────────
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
            prev_divergences = getattr(client_selector, '_prev_divergences', None)
            prev_gradient_aligns = getattr(client_selector, '_prev_gradient_alignments', None)
            if prev_divergences is None:
                prev_divergences = [0.0] * cfg.training.num_clients
            if prev_gradient_aligns is None:
                prev_gradient_aligns = [0.0] * cfg.training.num_clients

            selected_indices, bernoulli_probs = client_selector.select_clients(
                reputations=reputations,
                client_losses=prev_round_losses,
                model_divergences=prev_divergences,
                gradient_alignments=prev_gradient_aligns,
                data_shares=data_shares,
                k_sel=k_sel,
                minority_fractions=[c.minority_class_fraction for c in local_clients],
            )
        else:
            selected_indices = list(range(cfg.training.num_clients))

        selected_clients = [local_clients[i] for i in selected_indices]
        selected_local_models = [client.get_model_state() for client in selected_clients]

        # ── Step A: Local training (Tier 1) — selected clients only ──────────
        #  Save pre-train snapshots BEFORE local training.
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
        # For unselected clients: 0.0 (neutral alignment)
        server_delta = aggregator.compute_update(server_state, aggregator.global_model)
        if server_delta and len(server_delta) > 0:
            # Build local updates for selected clients: Δ_k = post - pre
            selected_posts = {
                i: client.get_model_state()
                for i, client in zip(selected_indices, selected_clients)
            }
            selected_pres = {
                i: pre for i, pre in zip(selected_indices, pre_train_models)
            }
            selected_updates = [
                aggregator.compute_update(selected_posts[i], selected_pres[i])
                for i in selected_indices
            ]
            raw_alignments = compute_gradient_alignment(
                local_updates=selected_updates,
                global_update=server_delta,
                device=device,
            )
            # Map back to all K clients: selected get alignment, unselected get 0.0
            alignment_map = dict(zip(selected_indices, raw_alignments))
            all_gradient_alignments = [
                alignment_map.get(k, 0.0) for k in range(cfg.training.num_clients)
            ]
        else:
            # Server model unchanged (e.g., cold start) — use uniform alignments
            all_gradient_alignments = [0.0] * cfg.training.num_clients

        # ── Step D: Federated aggregation — selected clients only ──────────────
        #   Only selected clients contribute to aggregation (RL selection policy)
        post_train_models = [client.get_model_state() for client in selected_clients]

        aggregated_model, trust_scores = aggregator.aggregate_round(
            local_models=post_train_models,
            server_model=server_state,
            selected_indices=selected_indices,
            pre_train_models=pre_train_models,  # clean deltas via Global Start Principle
        )

        # ── Step E: Apply aggregated global model — NO personalisation during FL training ──
        #  Fed+ personalisation (kappa mixing) was causing client-local
        # overfitting: 90-99% local accuracy but only 38-47% global accuracy.
        # Root cause: kappa=0.01 → sigma=0.01 → κ=1/(1+η·σ) ≈ 0.997
        # This means 99.7% of the client's personalisation is retained, and only
        # 0.3% is contributed to the global model — clients essentially stay private.
        # Solution: give ALL clients the clean aggregated global model (no mixing).
        # Personalisation can be re-enabled post-FL if per-client inference is needed.
        for client in selected_clients:
            client.set_model_state(aggregated_model)

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

        # ── Evaluation ──
        selector_loss_info = ""
        if (round_idx + 1) % eval_interval == 0 or round_idx == 0:
            eval_metrics = evaluate_global_model(
                aggregated_model, X_test, y_test,
                num_classes, cfg, device,
            )

        # ── Selector PPO update (Tier 2) ───────────────────────────────
        if (client_selector is not None
                and round_idx >= cfg.training.selector_eval_interval - 1
                and (round_idx - cfg.training.selector_eval_interval + 1) % cfg.training.selector_eval_interval == 0):

            # Selector reward uses FLTrust temporal reputations (K-length list).
            # NOTE: trust_scores from aggregate_round is SELECTED-only (length = len(selected_clients)).
            # We MUST use fl_trust.reputations (K-length) for correct indexing in compute_reward:
            #   selected_trusts = [reputations[k] for k in selected_indices]
            # Using the returned trust_scores would cause IndexError or wrong trust values.
            reputations = aggregator.fl_trust.reputations[:cfg.training.num_clients]

            client_selector.record_selection(
                selected_indices=selected_indices,
                global_accuracy=eval_metrics["accuracy"],
                trust_scores=reputations,   # K-length FLTrust reputations, not selected-only
                bernoulli_probs=bernoulli_probs,
            )

            # Update F1 EMAs from all clients
            client_selector.update_f1_from_round(
                selected_indices=selected_indices,
                client_f1s=all_client_f1s,
            )

            # Store divergences and gradient alignments for next round's state
            client_selector._prev_divergences = [
                compute_model_divergence(client.get_model_state(), aggregator.global_model)
                for client in local_clients
            ]
            client_selector._prev_gradient_alignments = all_gradient_alignments

            # PPO update with entropy decay
            selector_info = client_selector.update(round_idx=round_idx)
            if selector_info:
                selector_loss_info = (
                    f"Sel-Actor: {selector_info.get('selector_actor_loss', 0):.4f} | "
                    f"Sel-Critic: {selector_info.get('selector_critic_loss', 0):.4f} | "
                    f"H: {selector_info.get('selector_entropy', 0):.3f}"
                )

        round_time = time.time() - round_start

        if (round_idx + 1) % eval_interval == 0 or round_idx == 0:
            history["rounds"].append(round_idx + 1)
            history["accuracy"].append(eval_metrics["accuracy"])
            history["precision"].append(eval_metrics["precision"])
            history["recall"].append(eval_metrics["recall"])
            history["f1"].append(eval_metrics["f1_score"])
            history["f1_macro"].append(eval_metrics.get("f1_macro", 0.0))
            history["fpr"].append(eval_metrics["fpr"])
            history["trust_scores"].append(trust_scores)
            history["client_accuracies"].append(all_client_accuracies)
            history["selected_clients"].append(selected_indices)

            # Get temporal reputations for logging
            reputations = aggregator.fl_trust.reputations[:cfg.training.num_clients]

            f1_macro_val = eval_metrics.get("f1_macro", 0.0)
            recall_per_class = eval_metrics.get("recall_per_class", {})
            minority_recall = recall_per_class.get(1, recall_per_class.get(min(recall_per_class.keys()), 0.0)) if recall_per_class else 0.0

            print(
                f"Round {round_idx+1:4d}/{cfg.training.num_rounds} | "
                f"Acc: {eval_metrics['accuracy']:.4f} | "
                f"Prec: {eval_metrics['precision']:.4f} | "
                f"Rec: {eval_metrics['recall']:.4f} | "
                f"F1: {eval_metrics['f1_score']:.4f} | "
                f"F1m: {f1_macro_val:.4f} | "
                f"FPR: {eval_metrics['fpr']:.4f} | "
                f"MinRec: {minority_recall:.4f} | "
                f"K_sel: {k_sel} | " if client_selector is not None else "",
                f"Sel: {selected_indices} | "
                f"Trust: [{', '.join(f'{s:.2f}' for s in trust_scores)}] | "
                f"Rep: [{', '.join(f'{r:.2f}' for r in reputations)}]"
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
                        choices=["edge_iiot", "nsl_kdd", "iomt_2024", "unsw_nb15", "unified"],
                        help="'unified' trains on all 4 datasets combined with universal 3-class taxonomy")
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

    if args.seeds is not None:
        cfg.training.seeds = args.seeds
        run_multi_seed(cfg)
    else:
        run_training(cfg, resume_checkpoint=args.resume)

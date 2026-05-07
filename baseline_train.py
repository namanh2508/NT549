"""
Non-Federated Baseline: Single PPO Agent on Full Dataset.

Purpose: Isolate whether the poor federated results come from:
  1. The federated architecture (aggregation, client selection, etc.)
  2. The PPO + RL approach itself
  3. The environment / reward function

This script uses IDENTICAL components as the federated version:
  - Same CNNGRU+CBAM actor-critic architecture
  - Same MultiClassIDSEnvironment
  - Same reward function (MCC-based)
  - Same data preprocessing and splits

The ONLY difference: no federation — single agent trains on full data.

V2 FIXES (compared to V1):
  1. TN_REWARD lowered 5.0 → 1.0 to prevent over-predicting Benign (was causing FPR=1.0)
  2. Per-episode return normalization to prevent critic loss explosion (59→140)
  3. More episodes per round (8 vs 5) to reduce gradient variance
  4. Per-round exponential moving average for smoother eval metrics
  5. Learning rate warmup over first 3 rounds
  6. Gradient clipping for both actor and critic
  7. Early stopping if no improvement for 10 rounds

V3 FIXES (compared to V2):
  1. Warmup LR start: 1e-5 → 5e-5 (was too low, R1 accuracy=0.14)
  2. clip_epsilon: 0.15 → 0.1 (prevent large policy shifts causing drops)
  3. Per-minibatch advantage normalization (stabilize PPO updates)
  4. Return normalization in GAE compute (keeps critic loss in check)
  5. Output dir renamed to baseline_v3
"""

import os
import json
import time
import copy
import torch
import numpy as np
from typing import Dict, List

from src.config import Config
from src.data.preprocessor import load_dataset
from src.agents.ppo_agent import PPOAgent
from src.environment.ids_env import MultiClassIDSEnvironment
from src.utils.metrics import compute_multiclass_metrics, compute_binary_metrics


def evaluate_model(
    agent: PPOAgent,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
) -> Dict:
    """Evaluate the agent on test data using deterministic policy."""
    y_pred = []
    for i in range(len(X_test)):
        state = X_test[i].astype(np.float32)
        action_idx, _, _ = agent.select_action(state, deterministic=True)
        y_pred.append(int(action_idx))

    y_pred = np.array(y_pred)
    multi_metrics = compute_multiclass_metrics(y_test, y_pred)

    # Binary FPR: benign (0) flagged as attack
    y_test_bin = (y_test != 0).astype(int)
    y_pred_bin = (y_pred != 0).astype(int)
    binary_fpr = compute_binary_metrics(y_test_bin, y_pred_bin)["fpr"]

    return {**multi_metrics, "fpr": binary_fpr}


def run_baseline(cfg: Config, output_suffix: str = "", num_rounds: int = 20):
    """
    Train a single PPO agent on the full (non-partitioned) dataset.
    V3: With supervised pretraining + reward/PPO stability fixes.
    """
    print("=" * 70)
    print("  BASELINE V3: Single PPO Agent (No Federation)")
    print("  Key fixes: Supervised pretrain, TN_REWARD=1.0, lower LR, tighter clip")
    print("=" * 70)

    device = torch.device(
        cfg.training.device if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[1/5] Loading dataset...")
    X_train, X_test, y_train, y_test, le = load_dataset(cfg)
    num_classes = len(le.classes_)
    state_dim = X_train.shape[1]
    action_dim = num_classes
    print(f"  Full dataset: {len(X_train)} train, {len(X_test)} test")
    print(f"  Classes: {num_classes}, Features: {state_dim}")

    # Class distribution
    class_counts = np.bincount(y_train, minlength=num_classes)
    for c in range(num_classes):
        print(f"    Class {c}: {class_counts[c]} ({100*class_counts[c]/len(y_train):.1f}%)")

    # ── Use cfg.ppo and cfg.reward (set by apply_v3_config in kaggle_train.py) ──
    print("\n[2/5] Using reward and PPO configs from cfg...")
    print(f"  TN_REWARD: {cfg.reward.tn_reward}")
    print(f"  FN_PENALTY: {cfg.reward.fn_penalty}")
    print(f"  lr_actor: {cfg.ppo.lr_actor}")
    print(f"  clip_epsilon: {cfg.ppo.clip_epsilon}")
    print(f"  ppo_epochs: {cfg.ppo.ppo_epochs}")
    print(f"  mini_batch: {cfg.ppo.mini_batch_size}")
    print(f"  warmup_rounds: {cfg.ppo.lr_warmup_rounds}")
    print(f"  lr_min_factor: {cfg.ppo.lr_min_factor}")

    # ── Create environment and agent ─────────────────────────────────────────
    print("\n[3/5] Initialising environment and PPO agent...")
    env = MultiClassIDSEnvironment(
        X=X_train,
        y=y_train,
        reward_cfg=cfg.reward,
        num_classes=num_classes,
    )

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        cfg=cfg.ppo,
        device=device,
        dataset=cfg.training.dataset,
    )

    # ── Supervised Pretraining (warm up policy with CE loss) ─────────────────────
    print("\n[4/5] Supervised Pretraining (CE loss, 3 epochs)...")
    pretrain_info = agent.supervised_pretrain(
        X=X_train,
        y=y_train,
        class_weights=env._class_weights,
        num_epochs=3,
        batch_size=256,
        lr=1e-3,
    )
    print(f"  Pretrain CE Loss: {pretrain_info['pretrain_ce_loss']:.4f}")
    print(f"  Pretrain Accuracy: {pretrain_info['pretrain_accuracy']:.4f}")

    # ── LR warmup + CosineAnnealing ─────────────────────────────────────────
    warmup_rounds = cfg.ppo.lr_warmup_rounds
    warmup_lrs_actor = np.linspace(5e-5, cfg.ppo.lr_actor, warmup_rounds)
    warmup_lrs_critic = np.linspace(5e-5, cfg.ppo.lr_critic, warmup_rounds)

    min_lr_actor = cfg.ppo.lr_actor * cfg.ppo.lr_min_factor
    min_lr_critic = cfg.ppo.lr_critic * cfg.ppo.lr_min_factor
    agent.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        agent.actor_optim,
        T_max=num_rounds - warmup_rounds,
        eta_min=min_lr_actor,
    )
    agent.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        agent.critic_optim,
        T_max=num_rounds - warmup_rounds,
        eta_min=min_lr_critic,
    )
    print(f"  LR schedule: warmup {warmup_rounds} rounds, then CosineAnnealing")

    # ── Training loop ────────────────────────────────────────────────────────
    num_episodes = cfg.training.local_episodes
    max_steps = min(len(X_train), cfg.training.max_steps_per_episode)

    print(f"\n[5/5] Training: {num_rounds} rounds, {num_episodes} eps/round, max_steps={max_steps}")
    print("-" * 80)

    history = {
        "rounds": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "f1_macro": [],
        "fpr": [],
        "episode_rewards": [],
        "actor_losses": [],
        "critic_losses": [],
        "entropies": [],
        "lr_actor": [],
        "ema_accuracy": [],
        "pretrain_ce_loss": pretrain_info["pretrain_ce_loss"],
        "pretrain_accuracy": pretrain_info["pretrain_accuracy"],
    }

    best_accuracy = 0.0
    best_model_state = None
    patience = 10
    rounds_no_improve = 0

    # EMA for evaluation (smoother metrics)
    ema_alpha = 0.3
    ema_accuracy = None

    for round_idx in range(num_rounds):
        round_start = time.time()

        # ── Warmup LR (overrides scheduler) ─────────────────────────────────
        if round_idx < warmup_rounds:
            for pg in agent.actor_optim.param_groups:
                pg["lr"] = warmup_lrs_actor[round_idx]
            for pg in agent.critic_optim.param_groups:
                pg["lr"] = warmup_lrs_critic[round_idx]
        else:
            agent.actor_scheduler.step()
            agent.critic_scheduler.step()

        current_lr = agent.actor_optim.param_groups[0]["lr"]

        # ── Collect rollouts ────────────────────────────────────────────────
        episode_rewards = []
        for ep in range(num_episodes):
            state = env.reset()
            done = False
            step = 0
            ep_reward = 0.0

            while not done and step < max_steps:
                action, log_prob, value = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                agent.store_transition(state, action, log_prob, reward, value, done)
                state = next_state
                ep_reward += reward
                step += 1

            episode_rewards.append(ep_reward)

        # ── PPO update with class weights ───────────────────────────────────
        update_info = agent.update(
            class_weights=env._class_weights,
            focal_gamma=env._focal_gamma,
        )
        avg_ep_reward = np.mean(episode_rewards)

        # ── Evaluation (deterministic) ───────────────────────────────────────
        eval_metrics = evaluate_model(agent, X_test, y_test, num_classes)

        # EMA smoothing
        if ema_accuracy is None:
            ema_accuracy = eval_metrics["accuracy"]
        ema_accuracy = ema_alpha * eval_metrics["accuracy"] + (1 - ema_alpha) * ema_accuracy

        round_time = time.time() - round_start

        # ── Log ─────────────────────────────────────────────────────────────
        history["rounds"].append(round_idx + 1)
        history["accuracy"].append(eval_metrics["accuracy"])
        history["precision"].append(eval_metrics["precision"])
        history["recall"].append(eval_metrics["recall"])
        history["f1_score"].append(eval_metrics["f1_score"])
        history["f1_macro"].append(eval_metrics.get("f1_macro", 0.0))
        history["fpr"].append(eval_metrics["fpr"])
        history["episode_rewards"].append(float(avg_ep_reward))
        history["actor_losses"].append(float(update_info.get("actor_loss", 0)))
        history["critic_losses"].append(float(update_info.get("critic_loss", 0)))
        history["entropies"].append(float(update_info.get("entropy", 0)))
        history["lr_actor"].append(float(current_lr))
        history["ema_accuracy"].append(float(ema_accuracy))

        print(
            f"R{round_idx+1:02d} | "
            f"Acc:{eval_metrics['accuracy']:.4f}(EMA:{ema_accuracy:.4f}) | "
            f"Prec:{eval_metrics['precision']:.4f} | "
            f"Rec:{eval_metrics['recall']:.4f} | "
            f"F1:{eval_metrics['f1_score']:.4f} | "
            f"FPR:{eval_metrics['fpr']:.4f} | "
            f"EpR:{avg_ep_reward:8.0f} | "
            f"A:{update_info.get('actor_loss', 0):.4f} "
            f"C:{update_info.get('critic_loss', 0):.2f} "
            f"H:{update_info.get('entropy', 0):.3f} "
            f"LR:{current_lr:.2e} "
            f"{round_time:.0f}s"
        )

        # ── Save best model ─────────────────────────────────────────────────
        if eval_metrics["accuracy"] > best_accuracy:
            best_accuracy = eval_metrics["accuracy"]
            best_model_state = copy.deepcopy(agent.get_model_state())
            rounds_no_improve = 0
        else:
            rounds_no_improve += 1

        # Checkpoint every 10 rounds
        if (round_idx + 1) % 10 == 0:
            output_dir = cfg.training.output_dir
            os.makedirs(output_dir, exist_ok=True)
            torch.save(
                agent.get_model_state(),
                os.path.join(output_dir, f"baseline_v3_round_{round_idx+1}.pt"),
            )

        # Early stopping
        if rounds_no_improve >= patience:
            print(f"\n  [EARLY STOP] No improvement for {patience} rounds. Stopping.")
            break

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  BASELINE V3 TRAINING COMPLETE (with supervised pretrain)")
    print("=" * 80)
    print(f"  Pretrain: CE={pretrain_info['pretrain_ce_loss']:.4f}, Acc={pretrain_info['pretrain_accuracy']:.4f}")
    print(f"  Best Accuracy: {best_accuracy:.4f}")
    print(f"  Final Round Metrics:")
    print(f"    Accuracy:   {history['accuracy'][-1]:.4f}")
    print(f"    Precision:  {history['precision'][-1]:.4f}")
    print(f"    Recall:     {history['recall'][-1]:.4f}")
    print(f"    F1 Score:   {history['f1_score'][-1]:.4f}")
    print(f"    F1 Macro:   {history['f1_macro'][-1]:.4f}")
    print(f"    FPR:        {history['fpr'][-1]:.4f}")

    # Save best model
    if best_model_state is not None:
        output_dir = cfg.training.output_dir
        os.makedirs(output_dir, exist_ok=True)
        torch.save(best_model_state, os.path.join(output_dir, "baseline_v3_best_model.pt"))
        print(f"\n  Best model saved (Acc={best_accuracy:.4f})")

    # Save history
    history_path = os.path.join(output_dir, f"baseline_v3_history{output_suffix}.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n  History saved to: {history_path}")

    return history, {
        "accuracy": history["accuracy"][-1],
        "precision": history["precision"][-1],
        "recall": history["recall"][-1],
        "f1_score": history["f1_score"][-1],
        "f1_macro": history["f1_macro"][-1],
        "fpr": history["fpr"][-1],
        "best_accuracy": best_accuracy,
    }


def compare_with_federated(baseline_history: Dict, federated_path: str):
    """Compare baseline results with federated training history."""
    print("\n" + "=" * 80)
    print("  COMPARISON: Baseline V3 vs Federated")
    print("=" * 80)

    try:
        with open(federated_path) as f:
            fed = json.load(f)
    except FileNotFoundError:
        print(f"  Federated history not found: {federated_path}")
        return

    print(f"\n  {'Metric':<15} {'Baseline':>12} {'Federated':>12} {'Delta':>12}")
    print(f"  {'-'*51}")

    metrics = ["accuracy", "precision", "recall", "f1_score", "f1_macro"]
    for m in metrics:
        b_final = baseline_history[m][-1] if baseline_history.get(m) else 0.0
        fed_key = m if m != "f1_score" else "f1"
        fed_vals = fed.get(fed_key, [])
        f_final = float(fed_vals[-1]) if fed_vals is not None and len(fed_vals) > 0 else 0.0
        delta = b_final - f_final
        sign = "+" if delta >= 0 else ""
        print(f"  {m:<15} {b_final:>12.4f} {f_final:>12.4f} {sign}{delta:>11.4f}")

    print(f"\n  Baseline = Single PPO V2 (no federation)")
    print(f"  Federated = PPO + FLTrust + RL Client Selector")
    print(f"\n  If baseline >> federated: federated architecture is the bottleneck")
    print(f"  If baseline ≈ federated: PPO approach is the bottleneck")


if __name__ == "__main__":
    from src.config import Config

    cfg = Config()
    cfg.training.dataset = "edge_iiot"
    cfg.training.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.training.output_dir = os.path.join(cfg.training.output_dir, "baseline_v3")
    cfg.training.sample_limit_per_file = 50000
    os.makedirs(cfg.training.output_dir, exist_ok=True)

    NUM_ROUNDS = 20  # Slightly more since LR is lower

    print(f"Dataset: {cfg.training.dataset}")
    print(f"Rounds: {NUM_ROUNDS}, Episodes/round: {cfg.training.local_episodes}, Max steps: {cfg.training.max_steps_per_episode}")
    print(f"Output: {cfg.training.output_dir}")

    baseline_history, final_metrics = run_baseline(cfg, num_rounds=NUM_ROUNDS)

    # Compare with federated results
    from pathlib import Path
    fed_path = Path(__file__).parent / "outputs" / "outputs_edge_iiot" / "training_history.json"
    if fed_path.exists():
        compare_with_federated(baseline_history, str(fed_path))

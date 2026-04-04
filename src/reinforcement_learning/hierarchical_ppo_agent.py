"""
Hierarchical Multi-Agent PPO for IDS — Manager + Worker Architecture
============================================================================
Unified Architecture: combines hierarchical MARL + continuous action space +
online learning for scalable novel attack handling.

  - Manager Agent: Phân loại attack CATEGORY (11 categories, MITRE ATT&CK)
  - Worker Agent: Chọn CONTINUOUS ACTION (4D Gaussian policy)
  - Online Learning: Q-values per attack category from feedback

Công thức toán:
  Manager Policy: π_θ^m(c|s) = P(category = c | s)
  Worker Policy: π_φ^w(a|s, c) = N(μ(s,c), σ(s,c))  (Gaussian)
  Joint Reward: R_joint = R_detection(c, y) + α · R_action(a, c, y)

Research Gap:
  - Hierarchical agent giúp tách biệt "what to do" (category) vs "how to do" (action)
  - Continuous action space: PPO học được response magnitude tối ưu
  - Online learning: adapt được novel attacks mà không cần retrain

Reference:
  - PPO: Schulman et al., "Proximal Policy Optimization Algorithms", 2017
  - Hierarchical RL: "Hierarchical Deep Reinforcement Learning" (Kulkarni et al., 2016)
============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
from collections import defaultdict

# Import attack taxonomy and online learning reward from scalable_decision_reward
from src.reinforcement_learning.scalable_decision_reward import (
    AttackTaxonomy,
    OnlineLearningReward,
    ActionSpace,
)


# ============================================================================
# ATTACK CATEGORIES & CONTINUOUS ACTION SPACE
# ============================================================================
# Manager uses MITRE ATT&CK-inspired taxonomy (11 categories)
NUM_CATEGORIES = 11  # BENIGN, RECON, DOS, EXPLOITATION, PRIV_ESC,
                     # LATERAL, EXFILTRATION, PERSISTENCE, EVASION, MALWARE, UNKNOWN

# Worker outputs continuous 4D action: [block_duration, throttle_rate, alert_severity, log_verbosity]
CONTINUOUS_ACTION_DIM = 4


# ============================================================================
# CONTINUOUS ACTION SPACE
# ============================================================================
class ContinuousActionBounds:
    """Bounds for continuous action dimensions."""

    BLOCK_DURATION = (0.0, 300.0)   # seconds
    THROTTLE_RATE = (0.0, 100.0)    # percent
    ALERT_SEVERITY = (0.0, 1.0)     # 0-1
    LOG_VERBOSITY = (0.0, 10.0)     # 0-10

    ACTION_BOUNDS = [BLOCK_DURATION, THROTTLE_RATE, ALERT_SEVERITY, LOG_VERBOSITY]
    ACTION_NAMES = ['block_duration', 'throttle_rate', 'alert_severity', 'log_verbosity']

    @staticmethod
    def clip_action(action):
        """Clip continuous action to valid bounds."""
        clipped = []
        for a, (lo, hi) in zip(action, ContinuousActionBounds.ACTION_BOUNDS):
            clipped.append(max(lo, min(hi, a)))
        return np.array(clipped)

    @staticmethod
    def action_to_dict(action):
        return {
            'block_duration': f"{action[0]:.1f}s",
            'throttle_rate': f"{action[1]:.1f}%",
            'alert_severity': f"{action[2]:.2f}",
            'log_verbosity': f"{action[3]:.0f}",
        }


# ============================================================================
# MANAGER NETWORK — 11-Category Classification
# ============================================================================
class ManagerNetwork(nn.Module):
    """
    Manager network outputs category distribution over 11 MITRE ATT&CK categories.

    Policy: π_θ^m(c|s) = P(category = c | s)
    Output: softmax over 11 categories
    """

    def __init__(self, input_dim, hidden_layers, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_categories = NUM_CATEGORIES

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.category_head = nn.Linear(prev_dim, NUM_CATEGORIES)

    def forward(self, x):
        features = self.backbone(x)
        category_logits = self.category_head(features)
        return category_logits

    def get_category_probs(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def sample_category(self, x):
        probs = self.get_category_probs(x)
        dist = torch.distributions.Categorical(probs)
        category = dist.sample()
        log_prob = dist.log_prob(category)
        return category, log_prob, probs


# ============================================================================
# CONTINUOUS WORKER NETWORK — Gaussian Action Policy
# ============================================================================
class ContinuousWorkerNetwork(nn.Module):
    """
    Worker network outputs continuous action distribution given category.

    Policy: π_φ^w(a|s, c) = N(μ(s,c), σ(s,c))
    Input: state + category embedding
    Output: mean and log_std for each of 4 action dimensions
    """

    def __init__(self, input_dim, hidden_layers, action_dim=CONTINUOUS_ACTION_DIM,
                 num_categories=NUM_CATEGORIES, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.num_categories = num_categories

        embed_dim = 32
        self.category_embedding = nn.Embedding(num_categories, embed_dim)
        self.input_with_embed = input_dim + embed_dim

        layers = []
        prev_dim = self.input_with_embed
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # Actor head: mean and log_std for each action dimension
        self.actor_mean = nn.Linear(prev_dim, action_dim)
        self.actor_log_std = nn.Linear(prev_dim, action_dim)

    def forward(self, x, category):
        """Forward pass returning Gaussian parameters."""
        cat_embed = self.category_embedding(category)
        combined = torch.cat([x, cat_embed], dim=-1)
        features = self.backbone(combined)

        mean = self.actor_mean(features)
        log_std = self.actor_log_std(features)
        log_std = torch.clamp(log_std, -20, 2)  # Stability
        std = torch.exp(log_std)

        return mean, std

    def sample_action(self, x, category):
        """
        Sample continuous action from Gaussian policy.

        Returns:
            action: numpy array [action_dim] (clipped)
            log_prob: float
            mean: numpy array [action_dim]
            std: numpy array [action_dim]
        """
        mean, std = self.forward(x, category)

        # Sample from Gaussian
        dist = torch.distributions.Normal(mean, std)
        action_raw = dist.sample()
        log_prob = dist.log_prob(action_raw).sum(dim=-1)  # Sum over dims

        # Clip to bounds
        action_clipped = torch.clamp(
            action_raw,
            torch.tensor([b[0] for b in ContinuousActionBounds.ACTION_BOUNDS]).to(action_raw.device),
            torch.tensor([b[1] for b in ContinuousActionBounds.ACTION_BOUNDS]).to(action_raw.device),
        )

        return (
            action_clipped.detach().cpu().numpy(),
            log_prob.detach().cpu().numpy(),
            mean.detach().cpu().numpy(),
            std.detach().cpu().numpy(),
        )

    def get_action(self, x, category):
        """Get deterministic mean action (for inference)."""
        mean, std = self.forward(x, category)
        action_clipped = torch.clamp(
            mean,
            torch.tensor([b[0] for b in ContinuousActionBounds.ACTION_BOUNDS]).to(mean.device),
            torch.tensor([b[1] for b in ContinuousActionBounds.ACTION_BOUNDS]).to(mean.device),
        )
        return action_clipped.detach().cpu().numpy(), mean.detach().cpu().numpy(), std.detach().cpu().numpy()


# ============================================================================
# SHARED CRITIC — Value Function
# ============================================================================
class SharedCritic(nn.Module):
    """Shared critic estimates state value V(s)."""

    def __init__(self, input_dim, hidden_layers, dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ============================================================================
# UNIFIED HIERARCHICAL PPO AGENT — Manager + Continuous Worker + Online Learning
# ============================================================================
class HierarchicalPPOAgent:
    """
    Unified Hierarchical PPO Agent.

    Architecture:
      - Manager: Classifies attack category (11 MITRE ATT&CK categories)
      - Worker: Outputs continuous 4D action (Gaussian policy)
      - Shared Critic: Estimates V(s)
      - Online Learning: Q-values per category updated from feedback

    This unifies:
      - Hierarchical MARL (Manager→category, Worker→action)
      - Continuous action space (PPO Gaussian policy)
      - Scalable novel attack handling (UNKNOWN category + online Q-learning)
    """

    def __init__(
        self,
        agent_id,
        input_dim,
        hidden_layers=None,
        num_categories=NUM_CATEGORIES,
        action_dim=CONTINUOUS_ACTION_DIM,
        lr_manager=3e-4,
        lr_worker=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        clip_epsilon=0.2,
        ppo_epochs=4,
        mini_batch_size=64,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        dropout=0.1,
        use_online_learning=True,
        device='cpu',
    ):
        self.agent_id = agent_id
        self.device = torch.device(device)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_categories = num_categories
        self.action_dim = action_dim
        self.use_online_learning = use_online_learning

        hidden_layers = hidden_layers or [128, 64, 32]

        # Networks
        self.manager = ManagerNetwork(input_dim, hidden_layers, dropout).to(self.device)
        self.worker = ContinuousWorkerNetwork(input_dim, hidden_layers, action_dim, dropout=dropout).to(self.device)
        self.critic = SharedCritic(input_dim, hidden_layers, dropout).to(self.device)

        # Optimizers
        self.manager_optimizer = optim.Adam(self.manager.parameters(), lr=lr_manager)
        self.worker_optimizer = optim.Adam(self.worker.parameters(), lr=lr_worker)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Online learning reward system
        self.reward_system = OnlineLearningReward(
            use_online_learning=use_online_learning,
            use_base_reward=True
        )

        # Rollout buffer
        self.buffer = UnifiedRolloutBuffer()

        # Stats
        self.num_samples_trained = 0
        self.episode_losses = []

    # -------------------------------------------------------------------------
    # Action Selection
    # -------------------------------------------------------------------------

    def select_action(self, state, training=True):
        """
        Hierarchical action selection: Manager→category, Worker→continuous action.

        Args:
            state: numpy array [input_dim]
            training: If True, sample; if False, use mean

        Returns:
            category: int (0-10)
            action: numpy array [4] (continuous)
            info: dict with manager_log_prob, worker_log_prob, value
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Manager: classify category
        if training:
            category, m_log_prob, _ = self.manager.sample_category(state_tensor)
            category = category.item()
        else:
            category_probs = self.manager.get_category_probs(state_tensor)
            category = torch.argmax(category_probs, dim=-1).item()
            m_log_prob = torch.log(category_probs[0, category] + 1e-8)

        # Worker: sample continuous action given category
        category_tensor = torch.LongTensor([category]).to(self.device)
        if training:
            action, w_log_prob, mean, std = self.worker.sample_action(state_tensor, category_tensor)
            action = action[0]
            w_log_prob = w_log_prob[0]
            mean = mean[0]
            std = std[0]
        else:
            mean, std = self.worker.get_action(state_tensor, category_tensor)
            action = mean
            w_log_prob = 0.0  # deterministic

        # Critic: estimate value
        with torch.no_grad():
            value = self.critic(state_tensor).item()

        info = {
            'manager_log_prob': m_log_prob.item() if training else 0.0,
            'worker_log_prob': w_log_prob,
            'value': value,
            'category': category,
            'mean': mean,
            'std': std,
        }

        return category, action, info

    # -------------------------------------------------------------------------
    # Reward Computation
    # -------------------------------------------------------------------------

    def compute_reward(self, action, category, is_attack):
        """
        Compute reward using continuous action + online learning.

        Args:
            action: numpy array [4]
            category: int (0-10, MITRE category index)
            is_attack: bool

        Returns:
            reward: float
        """
        # Base reward from continuous action appropriateness
        base_reward = self.reward_system._compute_base_reward_continuous(
            action, category, is_attack
        )

        # Online learning component
        learned_reward = 0.0
        if self.use_online_learning:
            # Exploration bonus for novel/unseen situations
            cat_count = self.reward_system.action_counts[category, 0]
            if cat_count < 10:
                learned_reward += 0.3 * (10 - cat_count) / 10
            # Q-value estimate
            learned_reward += self.reward_system.Q[category, 0] * 0.1

        return base_reward + learned_reward

    def update_q_from_feedback(self, category, reward):
        """Update Q-values from reward feedback (online learning)."""
        if not self.use_online_learning:
            return
        self.reward_system.update_from_feedback(
            attack_signature=f'cat_{category}',
            action_id=0,  # Use dim 0 as representative
            feedback_reward=reward
        )

    # -------------------------------------------------------------------------
    # Storage
    # -------------------------------------------------------------------------

    def store_transition(self, state, category, action, info, reward, value):
        self.buffer.store(state, category, action, info, reward, value)

    # -------------------------------------------------------------------------
    # Federated Learning Interface
    # -------------------------------------------------------------------------

    def get_weights(self):
        return {
            'manager': copy.deepcopy(self.manager.state_dict()),
            'worker': copy.deepcopy(self.worker.state_dict()),
            'critic': copy.deepcopy(self.critic.state_dict()),
        }

    def set_weights(self, weights):
        if 'manager' in weights:
            self.manager.load_state_dict(weights['manager'])
        if 'worker' in weights:
            self.worker.load_state_dict(weights['worker'])
        if 'critic' in weights:
            self.critic.load_state_dict(weights['critic'])

    def get_num_samples_trained(self):
        return self.num_samples_trained

    def reset_round_counter(self):
        self.num_samples_trained = 0
        self.episode_losses = []

    def get_network(self):
        """Return manager for FLTrust proxy initialization."""
        return self.manager

    # -------------------------------------------------------------------------
    # Federated Bridge — binary y (0/1) to internal category
    # -------------------------------------------------------------------------

    def train_episode(self, X, y):
        """
        Federated-compatible train_episode.

        Interprets y as binary (0=normal, 1=attack).
        Maps to MITRE taxonomy categories internally.
        """
        y = np.array(y)
        y_binary = y.astype(bool)
        # Map: normal→BENIGN(0), attack→DOS(2) as fallback
        y_categories = np.where(y_binary, 2, 0)  # 2=DOS category

        stats = self._train_episode_internal(X, y_categories, y_binary)
        return stats['total_loss'], stats['detection_accuracy']

    def evaluate(self, X, y):
        """Federated-compatible evaluate."""
        y = np.array(y)
        y_binary = y.astype(bool)
        y_categories = np.where(y_binary, 2, 0)
        return self._evaluate_internal(X, y_categories, y_binary)

    # -------------------------------------------------------------------------
    # Internal Training
    # -------------------------------------------------------------------------

    def _train_episode_internal(self, X, y_categories, y_attacks):
        """Internal training loop."""
        N = len(X)
        y_categories = np.array(y_categories)
        y_attacks = np.array(y_attacks)

        self.buffer.clear()
        correct = 0

        # Phase 1: Collect trajectories
        for i in range(N):
            state = X[i]
            true_cat = y_categories[i]
            true_attack = bool(y_attacks[i])

            category, action, info = self.select_action(state, training=True)
            reward = self.compute_reward(action, category, true_attack)

            self.store_transition(state, category, action, info, reward, info['value'])

            # Track accuracy: correct if manager classifies category well
            if category == true_cat:
                correct += 1

            # Update Q-values from reward feedback
            self.update_q_from_feedback(category, reward)

        # Phase 2: Compute advantages
        self.buffer.compute_advantages()

        # Phase 3: PPO updates
        total_losses = []
        for _ in range(self.ppo_epochs):
            for batch in self.buffer.get_batches(self.mini_batch_size):
                loss = self._ppo_update_batch(batch)
                total_losses.append(loss)

        self.buffer.clear()

        self.num_samples_trained += N
        self.episode_losses.append(np.mean(total_losses))

        return {
            'total_loss': np.mean(total_losses),
            'detection_accuracy': correct / N,
        }

    def _ppo_update_batch(self, batch):
        """PPO update for one batch."""
        (states, categories, actions, info_batch, rewards, values, advantages) = batch

        states_t = torch.FloatTensor(states).to(self.device)
        categories_t = torch.LongTensor(categories).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        values_t = torch.FloatTensor(values).to(self.device)

        # Old log probs
        m_old = torch.FloatTensor([i['manager_log_prob'] for i in info_batch]).to(self.device)
        w_old = torch.FloatTensor([i['worker_log_prob'] for i in info_batch]).to(self.device)

        # === Update Manager ===
        new_cat_probs = self.manager.get_category_probs(states_t)
        new_m_log_probs = torch.log(new_cat_probs.gather(1, categories_t.unsqueeze(1)) + 1e-8).squeeze()

        ratio_m = torch.exp(new_m_log_probs - m_old)
        clip_m = torch.clamp(ratio_m, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        manager_loss = -torch.min(ratio_m * advantages_t, clip_m * advantages_t).mean()
        entropy_m = -(new_cat_probs * torch.log(new_cat_probs + 1e-8)).sum(dim=-1).mean()

        # === Update Worker (continuous) ===
        mean, std = self.worker(states_t, categories_t)

        # Gaussian log prob
        dist = torch.distributions.Normal(mean, std)
        w_log_probs = dist.log_prob(actions_t).sum(dim=-1)

        ratio_w = torch.exp(w_log_probs - w_old)
        clip_w = torch.clamp(ratio_w, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        worker_loss = -torch.min(ratio_w * advantages_t, clip_w * advantages_t).mean()

        # Gaussian entropy
        entropy_w = 0.5 * (1 + torch.log(2 * np.pi * std ** 2 + 1e-8)).sum(dim=-1).mean()

        # === Update Critic ===
        values_pred = self.critic(states_t).squeeze()
        critic_loss = F.mse_loss(values_pred, values_t + advantages_t)

        # Total loss
        total_loss = (
            manager_loss
            + worker_loss
            + self.value_coef * critic_loss
            - self.entropy_coef * (entropy_m + entropy_w)
        )

        self.manager_optimizer.zero_grad()
        self.worker_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        total_loss.backward()
        nn.utils.clip_grad_norm_(self.manager.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.worker.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

        self.manager_optimizer.step()
        self.worker_optimizer.step()
        self.critic_optimizer.step()

        return total_loss.item()

    def _evaluate_internal(self, X, y_categories, y_attacks):
        """Internal evaluation."""
        N = len(X)
        tp = tn = fp = fn = 0

        for i in range(N):
            state = X[i]
            true_cat = y_categories[i]
            true_attack = bool(y_attacks[i])

            category, action, _ = self.select_action(state, training=False)

            # Detection: is predicted category an attack?
            pred_attack = (category != 0)  # 0=BENIGN

            if pred_attack and true_attack:
                tp += 1
            elif not pred_attack and not true_attack:
                tn += 1
            elif pred_attack and not true_attack:
                fp += 1
            else:
                fn += 1

        accuracy = (tp + tn) / (tp + tn + fp + fn) if N > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fpr': fpr,
            'auc_roc': accuracy,  # proxy
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        }


# ============================================================================
# UNIFIED ROLLOUT BUFFER
# ============================================================================
class UnifiedRolloutBuffer:
    """Buffer for unified hierarchical + continuous agent transitions."""

    def __init__(self):
        self.clear()

    def store(self, state, category, action, info, reward, value):
        self.states.append(state)
        self.categories.append(category)
        self.actions.append(action)
        self.info_batch.append(info)
        self.rewards.append(reward)
        self.values.append(value)

    def clear(self):
        self.states = []
        self.categories = []
        self.actions = []
        self.info_batch = []
        self.rewards = []
        self.values = []
        self.advantages = []

    def compute_advantages(self, gamma=0.99):
        self.advantages = []
        for r, v in zip(self.rewards, self.values):
            self.advantages.append(r - v)
        adv = np.array(self.advantages)
        mean = adv.mean()
        std = adv.std() + 1e-8
        self.advantages = ((adv - mean) / std).tolist()

    def get_batches(self, batch_size):
        n = len(self.states)
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]
            yield (
                np.array([self.states[i] for i in batch_idx]),
                np.array([self.categories[i] for i in batch_idx]),
                np.array([self.actions[i] for i in batch_idx]),
                [self.info_batch[i] for i in batch_idx],
                np.array([self.rewards[i] for i in batch_idx]),
                np.array([self.values[i] for i in batch_idx]),
                np.array([self.advantages[i] for i in batch_idx]),
            )

    def __len__(self):
        return len(self.states)

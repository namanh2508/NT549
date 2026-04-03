"""
Scalable Decision-Making PPO Agent with CONTINUOUS Action Space.
============================================================================
CONTINUOUS ACTION SPACE - PPO hoạt động tốt trên continuous action space

Thay vì discrete actions (ALLOW, DROP, ...), ta dùng continuous values:
  - action[0]: block_duration (0-300 seconds)
  - action[1]: throttle_rate (0-100 %)
  - action[2]: alert_severity (0.0-1.0)
  - action[3]: log_level (0-10)

Actor outputs: mean và std cho mỗi action dimension
π_θ(a|s) = Normal(μ(s), σ(s))

Scalable features:
  - Attack taxonomy generalization
  - Online learning từ feedback
  - Novel attack handling với exploration

Reference:
  - Schulman et al., "Proximal Policy Optimization Algorithms", 2017
  - Continuous control with PPO
============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
from collections import defaultdict

from src.reinforcement_learning.scalable_decision_reward import (
    OnlineLearningReward,
    AttackTaxonomy,
)


# ============================================================================
# CONTINUOUS ACTION SPACE DEFINITION
# ============================================================================
class ContinuousActionSpace:
    """
    Continuous Action Space cho IDS.

    Thay vì discrete choices, mỗi action dimension là continuous:
      - block_duration: 0-300 seconds (0=allow, >0=block)
      - throttle_rate: 0-100% (0=no throttle, 100=full block)
      - alert_severity: 0.0-1.0 (0=no alert, 1=critical alert)
      - log_verbosity: 0-10 (0=silent, 10=verbose logging)

    Agent outputs mean và std for each dimension.
    During inference: sample from N(μ, σ)
    """

    # Action bounds [min, max]
    ACTION_BOUNDS = [
        (0.0, 300.0),     # block_duration (seconds)
        (0.0, 100.0),     # throttle_rate (percent)
        (0.0, 1.0),       # alert_severity (0-1)
        (0.0, 10.0),       # log_verbosity (0-10)
    ]

    NUM_ACTION_DIMS = len(ACTION_BOUNDS)

    # Action names for logging
    ACTION_NAMES = [
        'block_duration',
        'throttle_rate',
        'alert_severity',
        'log_verbosity',
    ]

    @staticmethod
    def clip_action(action):
        """Clip continuous action to valid bounds."""
        clipped = []
        for i, (a, (low, high)) in enumerate(zip(action, ContinuousActionSpace.ACTION_BOUNDS)):
            clipped.append(max(low, min(high, a)))
        return np.array(clipped)

    @staticmethod
    def action_to_dict(action):
        """Convert action array to readable dict."""
        return {
            'block_duration': f"{action[0]:.1f}s",
            'throttle_rate': f"{action[1]:.1f}%",
            'alert_severity': f"{action[2]:.2f}",
            'log_verbosity': f"{action[3]:.0f}",
        }


# ============================================================================
# CONTINUOUS ACTOR-CRITIC NETWORK
# ============================================================================
class ContinuousActorCriticNetwork(nn.Module):
    """
    Actor-Critic Network cho CONTINUOUS action space.

    Actor outputs: mean và log_std for each action dimension
    π_θ(a|s) = N(μ(s), σ(s)) where σ = exp(log_std)

   Critic outputs: state value V(s)
    """

    def __init__(self, input_dim, hidden_layers=None, action_dim=4, dropout=0.1):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [128, 64, 32]

        self.action_dim = action_dim

        # Shared backbone
        backbone_layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            backbone_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        self.backbone = nn.Sequential(*backbone_layers)

        # Actor head: outputs mean và log_std for each action dim
        self.actor_mean = nn.Linear(prev_dim, action_dim)
        self.actor_log_std = nn.Linear(prev_dim, action_dim)

        # Critic head: outputs V(s)
        self.critic = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        """
        Forward pass.

        Args:
            state: [batch, input_dim]

        Returns:
            action_mean: [batch, action_dim]
            action_std: [batch, action_dim]
            value: [batch, 1]
        """
        features = self.backbone(state)

        # Actor: mean và std
        action_mean = self.actor_mean(features)
        action_log_std = self.actor_log_std(features)
        action_std = torch.exp(torch.clamp(action_log_std, -20, 2))  # Clamp for stability

        # Critic
        value = self.critic(features)

        return action_mean, action_std, value

    def sample_action(self, state):
        """
        Sample action from policy.

        Returns:
            action: numpy array (continuous)
            log_prob: float
            value: float
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            mean, std, value = self.forward(state_t)
            mean = mean.squeeze(0).numpy()
            std = std.squeeze(0).numpy()

            # Sample from N(mean, std)
            action = np.random.normal(mean, std)
            action = ContinuousActionSpace.clip_action(action)

        return action, mean, std, value.item()


# ============================================================================
# SCALABLE CONTINUOUS PPO AGENT
# ============================================================================
class ScalableContinuousPPOAgent:
    """
    Scalable PPO Agent với CONTINUOUS action space.

    Đặc điểm:
    1. Continuous action space (PPO-friendly)
    2. Attack taxonomy generalization
    3. Online learning từ feedback
    4. Novel attack exploration

    Interface tương thích với FederatedOrchestrator.
    """

    def __init__(self, agent_id, input_dim, hidden_layers=None,
                 action_dim=ContinuousActionSpace.NUM_ACTION_DIMS,
                 lr=3e-4, gamma=0.99, clip_epsilon=0.2, ppo_epochs=4,
                 mini_batch_size=64, value_coef=0.5, entropy_coef=0.01,
                 max_grad_norm=0.5, dropout=0.1,
                 use_online_learning=True,
                 device='cpu'):
        """
        Initialize Scalable Continuous PPO Agent.

        Args:
            agent_id: Unique agent ID
            input_dim: Feature dimension
            action_dim: = 4 (continuous action dimensions)
            lr: Learning rate
            use_online_learning: Nếu True, học từ feedback
        """
        self.agent_id = agent_id
        self.device = torch.device(device)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.action_dim = action_dim
        self.use_online_learning = use_online_learning

        # Network
        self.network = ContinuousActorCriticNetwork(
            input_dim=input_dim,
            hidden_layers=hidden_layers or [128, 64, 32],
            action_dim=action_dim,
            dropout=dropout
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # Reward system với online learning
        self.reward_system = OnlineLearningReward(
            use_online_learning=use_online_learning,
            use_base_reward=True
        )

        # Rollout buffer
        self.buffer = ContinuousRolloutBuffer()

        # Stats
        self.episode_losses = []
        self.num_samples_trained = 0

    def select_action(self, state, training=True):
        """
        Select continuous action.

        Args:
            state: numpy array [input_dim]
            training: If True, sample; if False, use mean

        Returns:
            action: numpy array [action_dim]
            info: dict với mean, std, value
        """
        if training:
            action, mean, std, value = self.network.sample_action(state)
            return action, {'mean': mean, 'std': std, 'value': value}
        else:
            # Use deterministic action (mean)
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                mean, std, value = self.network(state_t)
                action = ContinuousActionSpace.clip_action(mean.squeeze(0).numpy())
            return action, {'mean': mean.squeeze(0).numpy(), 'std': std.squeeze(0).numpy(), 'value': value.item()}

    def compute_reward(self, action, attack_sig, is_attack, attack_cat):
        """
        Compute reward cho continuous action.

        Reward based on:
        1. Action appropriateness for attack category
        2. Action magnitude (severity matching)
        3. Session cumulative impact

        Args:
            action: numpy array [action_dim]
            attack_sig: Attack signature string
            is_attack: bool
            attack_cat: category index

        Returns:
            reward: float
        """
        base_reward = self.reward_system._compute_base_reward_continuous(
            action, attack_cat, is_attack
        )

        # Add online learning component
        learned_reward = 0.0
        if self.use_online_learning:
            # Q-value lookup by category
            learned_reward += self.reward_system.Q[attack_cat, 0] * 0.1

        return base_reward + learned_reward

    def train_episode(self, X, y):
        """
        Train one episode.

        Args:
            X: Feature matrix [N, input_dim]
            y: Labels [N] (0=normal, 1=attack)

        Returns:
            loss: float
            accuracy: float
        """
        self.network.train()
        self.buffer.clear()

        correct = 0
        n = len(X)

        for i in range(n):
            state = X[i]
            label = int(y[i])
            is_attack = label == 1

            # Get attack taxonomy
            attack_sig = 'normal' if not is_attack else 'generic_attack'
            taxonomy = AttackTaxonomy.get_default_severity(attack_sig)
            attack_cat = taxonomy['category_idx']

            # Select action
            action, info = self.select_action(state, training=True)

            # Compute reward
            reward = self.compute_reward(action, attack_sig, is_attack, attack_cat)

            # Track accuracy (simplified: based on whether action was appropriate)
            base_r = self.reward_system._compute_base_reward_continuous(
                action, attack_cat, is_attack
            )
            if (is_attack and base_r > 0) or (not is_attack and base_r > 0):
                correct += 1

            # Store transition
            self.buffer.store(
                state=state,
                action=action,
                reward=reward,
                value=info['value'],
                mean=info['mean'],
                std=info['std'],
                attack_cat=attack_cat,
                is_attack=is_attack
            )

        # PPO update
        loss = self._ppo_update()

        # Update Q-values from experience
        if self.use_online_learning:
            self._update_q_values()

        accuracy = correct / n
        self.num_samples_trained += n
        self.episode_losses.append(loss)

        return loss, accuracy

    def _ppo_update(self):
        """PPO update cho continuous actions."""
        if len(self.buffer) < self.mini_batch_size:
            return 0.0

        total_loss = 0.0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            for batch in self.buffer.get_batches(self.mini_batch_size):
                states, actions, rewards, values, means, stds, advantages = batch

                states_t = torch.FloatTensor(states).to(self.device)
                actions_t = torch.FloatTensor(actions).to(self.device)
                advantages_t = torch.FloatTensor(advantages).to(self.device)
                values_t = torch.FloatTensor(values).to(self.device)

                # Get current policy distribution
                mean, std, value_pred = self.network(states_t)

                # Compute log probability of sampled actions
                # log π(a|s) = sum over dims of log N(a|μ, σ)
                log_probs = -0.5 * ((actions_t - mean) / (std + 1e-8)) ** 2 - torch.log(std + 1e-8) - 0.5 * np.log(2 * np.pi)
                log_probs = log_probs.sum(dim=-1)

                # Old log probs from buffer (stored means/stds)
                old_means_t = torch.FloatTensor(means).to(self.device)
                old_stds_t = torch.FloatTensor(stds).to(self.device)
                old_log_probs = -0.5 * ((actions_t - old_means_t) / (old_stds_t + 1e-8)) ** 2 - torch.log(old_stds_t + 1e-8) - 0.5 * np.log(2 * np.pi)
                old_log_probs = old_log_probs.sum(dim=-1)

                # PPO ratio
                ratio = torch.exp(log_probs - old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * advantages_t
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_t
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(value_pred.squeeze(), values_t + advantages_t)

                # Entropy bonus (encourage exploration)
                entropy = 0.5 * (1 + torch.log(2 * np.pi * std ** 2 + 1e-8)).sum(dim=-1).mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                n_updates += 1

        return total_loss / max(n_updates, 1)

    def _update_q_values(self):
        """Update Q-values từ experience."""
        cat_action_rewards = defaultdict(list)

        for i in range(len(self.buffer.states)):
            cat = self.buffer.attack_categories[i]
            # Use average reward as Q-value estimate
            cat_action_rewards[cat].append(self.buffer.rewards[i])

        for cat, rewards in cat_action_rewards.items():
            if rewards:
                avg_reward = np.mean(rewards)
                # Update Q[cat, 0] (using action dim 0 as representative)
                alpha = 0.1
                self.reward_system.Q[cat, 0] = (
                    self.reward_system.Q[cat, 0] * (1 - alpha) + alpha * avg_reward
                )

    def evaluate(self, X, y):
        """Evaluate agent."""
        self.network.eval()

        tp = tn = fp = fn = 0

        with torch.no_grad():
            for i in range(len(X)):
                state = X[i]
                label = int(y[i])
                is_attack = label == 1

                action, _ = self.select_action(state, training=False)

                taxonomy = AttackTaxonomy.get_default_severity('normal' if not is_attack else 'generic_attack')
                attack_cat = taxonomy['category_idx']

                base_r = self.reward_system._compute_base_reward_continuous(
                    action, attack_cat, is_attack
                )

                pred_attack = base_r < 0  # Simplified: negative reward = attack

                if pred_attack and is_attack:
                    tp += 1
                elif not pred_attack and not is_attack:
                    tn += 1
                elif pred_attack and not is_attack:
                    fp += 1
                else:
                    fn += 1

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }

    def get_weights(self):
        """Get network weights for FL."""
        return copy.deepcopy(self.network.state_dict())

    def set_weights(self, weights):
        """Set network weights from FL."""
        self.network.load_state_dict(weights)

    def get_num_samples_trained(self):
        return self.num_samples_trained

    def reset_round_counter(self):
        self.num_samples_trained = 0


# ============================================================================
# CONTINUOUS ROLLOUT BUFFER
# ============================================================================
class ContinuousRolloutBuffer:
    """Buffer for continuous action transitions."""

    def __init__(self):
        self.clear()

    def store(self, state, action, reward, value, mean, std, attack_cat, is_attack):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.means.append(mean)
        self.stds.append(std)
        self.attack_categories.append(attack_cat)
        self.is_attacks.append(is_attack)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.means = []
        self.stds = []
        self.attack_categories = []
        self.is_attacks = []
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
                np.array([self.actions[i] for i in batch_idx]),
                np.array([self.rewards[i] for i in batch_idx]),
                np.array([self.values[i] for i in batch_idx]),
                np.array([self.means[i] for i in batch_idx]),
                np.array([self.stds[i] for i in batch_idx]),
                np.array([self.advantages[i] for i in batch_idx]),
            )

    def __len__(self):
        return len(self.states)

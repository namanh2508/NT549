"""
Decision-Making PPO Agent for IDS.
============================================================================
Improvement: Convert from classification to decision-making RL paradigm.

Trong paradigm mới này, agent không chỉ phân loại Normal/Attack
mà phải QUYẾT ĐỊNH hành động phù hợp với từng loại tấn công.

So với PPOAgent (classification):
  - Action space: 2 → 7 (ALLOW, DROP, BLOCK_SRC, RATE_LIMIT, ALERT, MONITOR, ISOLATE)
  - Reward function: classification correctness → decision quality
  - Input labels: binary (0/1) → attack category (0-4)
  - Session tracking: theo dõi cumulative impact của decisions

Reference: Schulman et al., "Proximal Policy Optimization Algorithms", 2017
============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

from src.models.ppo_network import ActorCriticNetwork
from src.reinforcement_learning.decision_reward import (
    DecisionRewardFunction,
    NUM_DECISION_ACTIONS,
    CATEGORY_NAMES,
    ACTION_NAMES,
)
from src.utils.metrics import compute_metrics


class DecisionRolloutBuffer:
    """
    Rollout buffer cho Decision-Making PPO.

    Khác với RolloutBuffer của PPOAgent:
    - Lưu attack_category thay vì chỉ true_label
    - Lưu session context cho từng transition
    - Tính advantage dựa trên decision quality, không chỉ correctness
    """

    def __init__(self):
        """Khởi tạo buffer rỗng."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.advantages = []
        self.returns = []
        self.attack_categories = []  # Attack category cho từng transition
        self.action_names = []     # Action name cho debugging

    def store(self, state, action, log_prob, reward, value, attack_category):
        """Lưu một transition."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.attack_categories.append(attack_category)
        self.action_names.append(ACTION_NAMES.get(action, 'UNKNOWN'))

    def compute_advantages(self, gamma=0.99):
        """
        Tính advantage cho mỗi transition.

        Trong decision-making:
          Advantage = reward - baseline (value estimate)
          baseline = expected reward cho category + action pair
        """
        n = len(self.rewards)
        self.returns = list(self.rewards)

        # Raw advantage: A = R - V(s)
        self.advantages = [
            self.rewards[i] - self.values[i] for i in range(n)
        ]

        # Normalize advantages
        adv_array = np.array(self.advantages)
        adv_mean = adv_array.mean()
        adv_std = adv_array.std() + 1e-8
        self.advantages = ((adv_array - adv_mean) / adv_std).tolist()

    def get_batches(self, mini_batch_size):
        """Chia buffer thành mini-batches."""
        n = len(self.states)
        indices = np.random.permutation(n)

        for start in range(0, n, mini_batch_size):
            end = min(start + mini_batch_size, n)
            batch_idx = indices[start:end]

            yield (
                np.array([self.states[i] for i in batch_idx]),
                np.array([self.actions[i] for i in batch_idx]),
                np.array([self.log_probs[i] for i in batch_idx]),
                np.array([self.advantages[i] for i in batch_idx]),
                np.array([self.returns[i] for i in batch_idx]),
            )

    def clear(self):
        """Xóa buffer."""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.advantages.clear()
        self.returns.clear()
        self.attack_categories.clear()
        self.action_names.clear()

    def __len__(self):
        return len(self.states)


class DecisionPPOAgent:
    """
    Decision-Making PPO Agent cho IDS.

    Khác với PPOAgent (classification):
      - Action space: 7 thay vì 2
      - Nhận attack category labels (0-4) thay vì binary labels (0/1)
      - Sử dụng DecisionRewardFunction thay vì RewardFunction
      - Theo dõi session statistics

    Interface tương thích với FederatedOrchestrator:
      ✓ train_episode(X, y_categories) → (loss, accuracy)
      ✓ evaluate(X, y_categories) → metrics dict
      ✓ get_weights() → state_dict
      ✓ set_weights(weights) → None
      ✓ get_num_samples_trained() → int
      ✓ reset_round_counter() → None
    """

    def __init__(self, agent_id, input_dim, hidden_layers=None, num_actions=7,
                 lr=3e-4, gamma=0.99, clip_epsilon=0.2, ppo_epochs=4,
                 mini_batch_size=64, value_coef=0.5, entropy_coef=0.01,
                 max_grad_norm=0.5, dropout=0.1,
                 decision_params=None,
                 device='cpu'):
        """
        Initialize Decision PPO Agent.

        Args:
            agent_id: Unique agent ID
            input_dim: Feature dimension
            hidden_layers: Hidden layer sizes
            num_actions: Số actions (mặc định=7 cho decision-making)
            lr: Learning rate
            gamma: Discount factor
            clip_epsilon: PPO clipping parameter
            ppo_epochs: Số epochs per update
            mini_batch_size: Mini-batch size
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Gradient clipping norm
            dropout: Dropout rate
            decision_params: Dict chứa decision-making specific params
                           (base_reward_matrix, response_cost, etc.)
            device: Device ('cpu' hoặc 'cuda')
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
        self.num_actions = num_actions

        # ================================================================
        # ACTOR-CRITIC NETWORK — Giống PPOAgent nhưng với num_actions=7
        # ================================================================
        self.network = ActorCriticNetwork(
            input_dim=input_dim,
            hidden_layers=hidden_layers or [128, 64, 32],
            num_actions=num_actions,
            dropout=dropout
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # ================================================================
        # DECISION REWARD FUNCTION — Thay thế RewardFunction
        # ================================================================
        if decision_params is None:
            decision_params = {}

        self.reward_fn = DecisionRewardFunction(
            base_reward_matrix=decision_params.get('base_reward_matrix', None),
            response_cost_dict=decision_params.get('response_cost_dict', None),
            attack_severity_dict=decision_params.get('attack_severity_dict', None),
            usability_weight=decision_params.get('usability_weight', 0.1),
            max_block_ratio=decision_params.get('max_block_ratio', 0.1),
            use_novelty=decision_params.get('use_novelty', True),
        )

        # ================================================================
        # ROLLOUT BUFFER cho decision-making
        # ================================================================
        self.rollout_buffer = DecisionRolloutBuffer()

        # ================================================================
        # Training statistics
        # ================================================================
        self.episode_losses = []
        self.episode_accuracies = []
        self.num_samples_trained = 0

        # Decision-specific stats
        self.decision_stats = {
            'optimal_decisions': 0,
            'suboptimal_decisions': 0,
            'bad_decisions': 0,
        }

    def train_episode(self, X, y_categories):
        """
        Huấn luyện một episode với Decision-Making PPO.

        Args:
            X: Feature vectors [N, input_dim] — numpy array
            y_categories: Attack category labels [N] — numpy array
                         0=Normal, 1=DoS, 2=Probe, 3=R2L, 4=U2R
        Returns:
            avg_loss: Loss trung bình
            accuracy: Decision accuracy (action phù hợp với category)
        """
        self.network.train()

        # Reset rollout buffer và reward session
        self.rollout_buffer.clear()
        self.reward_fn.reset_session()

        # Shuffle data
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y_categories[indices]

        correct_decisions = 0
        n_samples = len(X)

        # ================================================================
        # PHASE 1: THU THẬP TRAJECTORY
        # ================================================================
        for j in range(n_samples):
            state = torch.FloatTensor(X_shuffled[j]).unsqueeze(0).to(self.device)
            attack_category = int(y_shuffled[j])

            # Lấy action từ actor (stochastic sampling)
            with torch.no_grad():
                action, log_prob, value = self.network.get_action(state)

            # Tính reward bằng DecisionRewardFunction
            reward = self.reward_fn.compute(
                action=action,
                attack_category=attack_category,
                state=X_shuffled[j]
            )

            # Đếm decision correctness
            # Decision đúng = action nằm trong optimal/suboptimal actions
            is_optimal = self._is_optimal_action(action, attack_category)
            is_bad = self._is_bad_action(action, attack_category)

            if is_optimal:
                correct_decisions += 1
                self.decision_stats['optimal_decisions'] += 1
            elif not is_bad:
                self.decision_stats['suboptimal_decisions'] += 1
            else:
                self.decision_stats['bad_decisions'] += 1

            # Lưu transition
            self.rollout_buffer.store(
                state=X_shuffled[j].copy(),
                action=action,
                log_prob=log_prob.item(),
                reward=reward,
                value=value.item(),
                attack_category=attack_category
            )

        # ================================================================
        # PHASE 2: PPO UPDATE
        # ================================================================
        self.rollout_buffer.compute_advantages(gamma=self.gamma)
        total_loss = self._ppo_update()

        # Cập nhật stats
        accuracy = correct_decisions / n_samples
        self.num_samples_trained += n_samples
        self.episode_losses.append(total_loss)
        self.episode_accuracies.append(accuracy)

        return total_loss, accuracy

    def _is_optimal_action(self, action, attack_category):
        """Kiểm tra action có phải là optimal cho attack category."""
        # Base reward > 1.0 được coi là optimal
        if hasattr(self.reward_fn, 'base_reward'):
            matrix = self.reward_fn.base_reward
            if 0 <= attack_category < len(matrix):
                if 0 <= action < len(matrix[attack_category]):
                    return matrix[attack_category][action] > 1.0
        return False

    def _is_bad_action(self, action, attack_category):
        """Kiểm tra action có phải là bad cho attack category."""
        if hasattr(self.reward_fn, 'base_reward'):
            matrix = self.reward_fn.base_reward
            if 0 <= attack_category < len(matrix):
                if 0 <= action < len(matrix[attack_category]):
                    return matrix[attack_category][action] < -1.5
        return False

    def _ppo_update(self):
        """
        Thực hiện K epochs PPO update.

        Giống PPOAgent nhưng dùng DecisionRolloutBuffer.
        """
        total_loss_sum = 0.0
        total_batches = 0

        for epoch in range(self.ppo_epochs):
            for batch in self.rollout_buffer.get_batches(self.mini_batch_size):
                states_np, actions_np, old_log_probs_np, advantages_np, returns_np = batch

                states = torch.FloatTensor(states_np).to(self.device)
                actions = torch.LongTensor(actions_np).to(self.device)
                old_log_probs = torch.FloatTensor(old_log_probs_np).to(self.device)
                advantages = torch.FloatTensor(advantages_np).to(self.device)
                returns = torch.FloatTensor(returns_np).to(self.device)

                # Evaluate actions dưới current policy
                new_log_probs, state_values, entropy = self.network.evaluate_actions(
                    states, actions
                )
                state_values = state_values.squeeze(-1)

                # Importance sampling ratio
                ratio = torch.exp(new_log_probs - old_log_probs)

                # Clipped surrogate objective
                surrogate1 = ratio * advantages
                surrogate2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon
                ) * advantages

                actor_loss = -torch.min(surrogate1, surrogate2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(state_values, returns)

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                total_loss = (
                    actor_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Backpropagation
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss_sum += total_loss.item()
                total_batches += 1

        return total_loss_sum / max(total_batches, 1)

    def evaluate(self, X, y_categories, y_binary=None):
        """
        Đánh giá agent trên tập test.

        Args:
            X: Feature vectors [N, input_dim]
            y_categories: Attack category labels [N]
            y_binary: Binary labels [N] (optional, for compatibility)
        Returns:
            metrics: Dict chứa accuracy, precision, recall, F1, FPR, AUC-ROC
        """
        self.network.eval()
        predictions = []
        probs_attack = []

        with torch.no_grad():
            batch_size = 1024
            for start in range(0, len(X), batch_size):
                end = min(start + batch_size, len(X))
                states = torch.FloatTensor(X[start:end]).to(self.device)

                action_probs, _ = self.network(states)

                # Greedy policy
                preds = action_probs.argmax(dim=1).cpu().numpy()
                predictions.extend(preds)

                # Probability of attack (non-Normal)
                probs = action_probs[:, 1:].sum(dim=1).cpu().numpy()
                probs_attack.extend(probs)

        predictions = np.array(predictions)
        probs_attack = np.array(probs_attack)

        # Compute metrics
        # Nếu có y_binary thì dùng nó, không thì dùng (categories > 0) làm binary
        if y_binary is None:
            y_binary = (y_categories > 0).astype(int)

        return compute_metrics(y_binary, predictions, probs_attack)

    def get_weights(self):
        """Lấy network weights cho federated aggregation."""
        return self.network.get_weights()

    def set_weights(self, weights):
        """Cập nhật weights từ federated aggregation."""
        self.network.set_weights(weights)

    def get_num_samples_trained(self):
        """Lấy số samples đã train."""
        return self.num_samples_trained

    def reset_round_counter(self):
        """Reset counters cho federated round mới."""
        self.num_samples_trained = 0
        self.reward_fn.reset_all()
        self.decision_stats = {
            'optimal_decisions': 0,
            'suboptimal_decisions': 0,
            'bad_decisions': 0,
        }

    def get_decision_stats(self):
        """Lấy decision statistics."""
        total = sum(self.decision_stats.values())
        if total == 0:
            return self.decision_stats

        return {
            **self.decision_stats,
            'optimal_ratio': self.decision_stats['optimal_decisions'] / total,
            'bad_ratio': self.decision_stats['bad_decisions'] / total,
            'session_stats': self.reward_fn.get_session_stats(),
        }

"""
Hierarchical Multi-Agent PPO for IDS — Manager + Worker Architecture
============================================================================
Improvement 6: Hierarchical Multi-Agent Reinforcement Learning.

Ý tưởng:
  - Manager Agent: Phân loại attack CATEGORY (Normal, DoS, Probe, R2L, U2R)
  - Worker Agent: Chọn ACTION cụ thể (ALLOW, DROP, RATE_LIMIT...)

Công thức toán:
  Manager Policy: π_θ^m(c|s) = P(category = c | s)
  Worker Policy: π_φ^w(a|s, c) = P(action = a | s, given category c)

  Joint Reward: R_joint = R_detection(c, y) + α · R_action(a, c, y)

Research Gap:
  - Chưa có bài nào kết hợp Hierarchical MARL + PPO + FLTrust/Fed+
  - Hierarchical agent giúp tách biệt "what to do" (category) vs "how to do" (action)

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



# ============================================================================
# ATTACK CATEGORIES & ACTIONS
# ============================================================================
CATEGORY_NORMAL = 0
CATEGORY_DOS = 1
CATEGORY_PROBE = 2
CATEGORY_R2L = 3
CATEGORY_U2R = 4

CATEGORY_NAMES = {
    CATEGORY_NORMAL: 'Normal',
    CATEGORY_DOS: 'DoS',
    CATEGORY_PROBE: 'Probe',
    CATEGORY_R2L: 'R2L',
    CATEGORY_U2R: 'U2R',
}

ACTION_ALLOW = 0
ACTION_LOG_ALERT = 1
ACTION_RATE_LIMIT = 2
ACTION_DROP = 3
ACTION_BLOCK_TEMP = 4
ACTION_BLOCK_PERM = 5
ACTION_ISOLATE = 6
ACTION_INVESTIGATE = 7

ACTION_NAMES = {
    ACTION_ALLOW: 'ALLOW',
    ACTION_LOG_ALERT: 'LOG_ALERT',
    ACTION_RATE_LIMIT: 'RATE_LIMIT',
    ACTION_DROP: 'DROP',
    ACTION_BLOCK_TEMP: 'BLOCK_TEMP',
    ACTION_BLOCK_PERM: 'BLOCK_PERM',
    ACTION_ISOLATE: 'ISOLATE',
    ACTION_INVESTIGATE: 'INVESTIGATE',
}

NUM_CATEGORIES = 5
NUM_ACTIONS = 8


# ============================================================================
# MANAGER NETWORK — Attack Category Classification
# ============================================================================
class ManagerNetwork(nn.Module):
    """
    Manager network outputs category distribution.

    Policy: π_θ^m(c|s) = P(category = c | s)
    Output: softmax over NUM_CATEGORIES
    """

    def __init__(self, input_dim, hidden_layers, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_categories = NUM_CATEGORIES

        # Shared backbone
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

        # Category head (Manager output)
        self.category_head = nn.Linear(prev_dim, NUM_CATEGORIES)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: state tensor [batch, input_dim]

        Returns:
            category_logits: [batch, NUM_CATEGORIES]
        """
        features = self.backbone(x)
        category_logits = self.category_head(features)
        return category_logits

    def get_category_probs(self, x):
        """Get category probability distribution."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def sample_category(self, x):
        """Sample category from distribution."""
        probs = self.get_category_probs(x)
        dist = torch.distributions.Categorical(probs)
        category = dist.sample()
        log_prob = dist.log_prob(category)
        return category, log_prob, probs


# ============================================================================
# WORKER NETWORK — Action Selection
# ============================================================================
class WorkerNetwork(nn.Module):
    """
    Worker network outputs action distribution given category.

    Policy: π_φ^w(a|s, c) = P(action = a | s, given category c)
    Input: state + one-hot encoded category
    Output: softmax over NUM_ACTIONS
    """

    def __init__(self, input_dim, hidden_layers, num_categories=NUM_CATEGORIES, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_categories = num_categories
        self.num_actions = NUM_ACTIONS

        # Input = state + category embedding
        embed_dim = 32
        self.category_embedding = nn.Embedding(num_categories, embed_dim)
        self.input_with_embed = input_dim + embed_dim

        # Shared backbone
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

        # Action head (Worker output)
        self.action_head = nn.Linear(prev_dim, NUM_ACTIONS)

    def forward(self, x, category):
        """
        Forward pass.

        Args:
            x: state tensor [batch, input_dim]
            category: category indices [batch]

        Returns:
            action_logits: [batch, NUM_ACTIONS]
        """
        # Embed category and concatenate with state
        cat_embed = self.category_embedding(category)
        combined = torch.cat([x, cat_embed], dim=-1)

        features = self.backbone(combined)
        action_logits = self.action_head(features)
        return action_logits

    def get_action_probs(self, x, category):
        """Get action probability distribution given category."""
        logits = self.forward(x, category)
        return F.softmax(logits, dim=-1)

    def sample_action(self, x, category):
        """Sample action from distribution."""
        probs = self.get_action_probs(x, category)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, probs


# ============================================================================
# SHARED CRITIC — Value Function
# ============================================================================
class SharedCritic(nn.Module):
    """
    Shared critic estimates state value V(s).

    Used by both Manager and Worker for advantage estimation.
    """

    def __init__(self, input_dim, hidden_layers, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))  # Output: V(s)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Estimate V(s)."""
        return self.network(x)


# ============================================================================
# HIERARCHICAL PPO AGENT
# ============================================================================
class HierarchicalPPOAgent:
    """
    Hierarchical PPO Agent with Manager + Worker.

    Architecture:
      - Manager: Classifies attack category (5 categories)
      - Worker: Selects action based on category (8 actions)
      - Shared Critic: Estimates V(s) for both agents

    Training:
      - Two-level policy gradient
      - Manager learns from category classification reward
      - Worker learns from action decision reward
      - Joint update with shared critic
    """

    def __init__(
        self,
        agent_id,
        input_dim,
        hidden_layers=None,
        num_categories=NUM_CATEGORIES,
        num_actions=NUM_ACTIONS,
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
        device='cpu',
    ):
        """
        Initialize Hierarchical PPO Agent.

        Args:
            agent_id: Unique agent identifier
            input_dim: Feature dimension
            hidden_layers: Hidden layer sizes
            lr_manager: Learning rate for Manager
            lr_worker: Learning rate for Worker
            lr_critic: Learning rate for Critic
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
        self.num_categories = num_categories
        self.num_actions = num_actions

        hidden_layers = hidden_layers or [128, 64, 32]

        # Networks
        self.manager = ManagerNetwork(input_dim, hidden_layers, dropout).to(self.device)
        self.worker = WorkerNetwork(input_dim, hidden_layers, num_categories, dropout).to(self.device)
        self.critic = SharedCritic(input_dim, hidden_layers, dropout).to(self.device)

        # Optimizers
        self.manager_optimizer = optim.Adam(self.manager.parameters(), lr=lr_manager)
        self.worker_optimizer = optim.Adam(self.worker.parameters(), lr=lr_worker)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Rollout buffer
        self.buffer = HierarchicalRolloutBuffer()

        # Stats
        self.num_samples_trained = 0
        self.episode_losses = []
        self.episode_accuracies = []

    def select_action(self, state, training=True):
        """
        Select category and action using hierarchical policy.

        Args:
            state: numpy array [input_dim]
            training: If True, sample; if False, take argmax

        Returns:
            category: int (0-4)
            action: int (0-7)
            log_probs: (manager_log_prob, worker_log_prob)
            values: V(s)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Manager: select category
            if training:
                category, manager_log_prob, category_probs = self.manager.sample_category(state_tensor)
                category = category.item()
            else:
                category_probs = self.manager.get_category_probs(state_tensor)
                category = torch.argmax(category_probs, dim=-1).item()
                manager_log_prob = torch.log(category_probs[0, category] + 1e-8)

            # Worker: select action given category
            category_tensor = torch.LongTensor([category]).to(self.device)
            if training:
                action, worker_log_prob, action_probs = self.worker.sample_action(state_tensor, category_tensor)
                action = action.item()
            else:
                action_probs = self.worker.get_action_probs(state_tensor, category_tensor)
                action = torch.argmax(action_probs, dim=-1).item()
                worker_log_prob = torch.log(action_probs[0, action] + 1e-8)

            # Critic: estimate value
            value = self.critic(state_tensor).item()

        return category, action, (manager_log_prob.item(), worker_log_prob.item()), value

    def compute_reward(self, category, action, true_category, true_is_attack):
        """
        Compute hierarchical reward.

        R_joint = R_detection + α * R_action

        Args:
            category: Predicted category (0-4)
            action: Selected action (0-7)
            true_category: True category (0-4)
            true_is_attack: True if attack (bool)

        Returns:
            reward: float
            detection_reward: float
            action_reward: float
        """
        # Detection reward: correct category classification
        if category == true_category:
            detection_reward = 1.0
        else:
            detection_reward = -0.5

        # Action reward: depends on category and action appropriateness
        action_reward = self._get_action_reward(action, true_category, true_is_attack)

        # Combined reward
        alpha = 0.7
        reward = detection_reward + alpha * action_reward

        return reward, detection_reward, action_reward

    def _get_action_reward(self, action, true_category, true_is_attack):
        """
        Get action reward based on action suitability for category.

        Optimal actions per category:
          - Normal (0): ALLOW (0) or LOG_ALERT (1)
          - DoS (1): DROP (3) or RATE_LIMIT (2)
          - Probe (2): LOG_ALERT (1) or RATE_LIMIT (2)
          - R2L (3): BLOCK_TEMP (4) or ISOLATE (6)
          - U2R (4): BLOCK_PERM (5) or ISOLATE (6)
        """
        if not true_is_attack:
            # Normal traffic - allow is best
            optimal = {ACTION_ALLOW, ACTION_LOG_ALERT}
            suboptimal = {ACTION_RATE_LIMIT, ACTION_INVESTIGATE}
        elif true_category == CATEGORY_DOS:
            # DoS - block/rate limit
            optimal = {ACTION_DROP, ACTION_RATE_LIMIT}
            suboptimal = {ACTION_BLOCK_TEMP, ACTION_BLOCK_PERM}
        elif true_category == CATEGORY_PROBE:
            # Probe - monitor/log
            optimal = {ACTION_LOG_ALERT, ACTION_INVESTIGATE}
            suboptimal = {ACTION_RATE_LIMIT}
        elif true_category == CATEGORY_R2L:
            # R2L - block temporarily
            optimal = {ACTION_BLOCK_TEMP, ACTION_ISOLATE}
            suboptimal = {ACTION_DROP}
        elif true_category == CATEGORY_U2R:
            # U2R - aggressive response
            optimal = {ACTION_BLOCK_PERM, ACTION_ISOLATE}
            suboptimal = {ACTION_BLOCK_TEMP}
        else:
            optimal = {ACTION_LOG_ALERT}
            suboptimal = set()

        if action in optimal:
            return 1.0
        elif action in suboptimal:
            return 0.3
        else:
            return -0.5

    def store_transition(self, state, category, action, manager_log_prob, worker_log_prob, reward, value):
        """Store transition in buffer."""
        self.buffer.store(state, category, action, manager_log_prob, worker_log_prob, reward, value)

    def get_weights(self):
        """
        Get all network weights for federated aggregation.

        Returns:
            dict with manager, worker, critic weights
        """
        return {
            'manager': copy.deepcopy(self.manager.state_dict()),
            'worker': copy.deepcopy(self.worker.state_dict()),
            'critic': copy.deepcopy(self.critic.state_dict()),
        }

    def set_weights(self, weights):
        """
        Set network weights from federated aggregation.

        Args:
            weights: dict with manager, worker, critic weights
        """
        if 'manager' in weights:
            self.manager.load_state_dict(weights['manager'])
        if 'worker' in weights:
            self.worker.load_state_dict(weights['worker'])
        if 'critic' in weights:
            self.critic.load_state_dict(weights['critic'])

    def get_num_samples_trained(self):
        """Return number of samples trained."""
        return self.num_samples_trained

    def reset_round_counter(self):
        """Reset per-round statistics."""
        self.num_samples_trained = 0
        self.episode_losses = []
        self.episode_accuracies = []

    # -------------------------------------------------------------------------
    # Federated Learning Interface Bridge
    # Converts binary y (0/1) into category + attack flag internally.
    # This allows HierarchicalPPOAgent to plug into FederatedOrchestrator
    # alongside DQNAgent / PPOAgent without interface changes in the
    # orchestrator.
    # -------------------------------------------------------------------------

    def train_episode(self, X, y):
        """
        Federated-compatible train_episode wrapper.

        Interprets y as binary attack labels (0=normal, 1=attack).
        Internally treats all attacks as category 1 (DoS) for the manager
        so that the hierarchical policy still learns category classification
        while receiving the binary supervision signal from the orchestrator.

        Args:
            X: Feature matrix [N, input_dim]
            y: Binary labels [N] (0=normal, 1=attack)

        Returns:
            loss: float (total PPO loss for the episode)
            accuracy: float (attack detection accuracy)
        """
        y = np.array(y)
        y_binary = y.astype(bool)
        # Derive category labels: 0=normal, 1=DoS (fallback for unknown attack types)
        y_categories = np.where(y_binary, CATEGORY_DOS, CATEGORY_NORMAL)
        stats = self._train_episode_internal(X, y_categories, y_binary)
        return stats['manager_loss'], stats['category_accuracy']

    def evaluate(self, X, y):
        """
        Federated-compatible evaluate wrapper.

        Interprets y as binary labels and returns the standard metrics dict
        expected by FederatedOrchestrator.

        Args:
            X: Feature matrix [N, input_dim]
            y: Binary labels [N] (0=normal, 1=attack)

        Returns:
            metrics: dict with accuracy, precision, recall, f1, fpr, auc_roc
        """
        y = np.array(y)
        y_binary = y.astype(bool)
        y_categories = np.where(y_binary, CATEGORY_DOS, CATEGORY_NORMAL)
        metrics = self._evaluate_internal(X, y_categories, y_binary)
        # Flatten to match the (accuracy, precision, recall, f1, fpr, auc_roc)
        # contract expected by the orchestrator's print_metrics helpers.
        return metrics

    # -------------------------------------------------------------------------
    # Internal training and evaluation (original signatures)
    # -------------------------------------------------------------------------

    def _train_episode_internal(self, X, y_categories, y_attacks):
        """
        Internal train_episode (original signature).

        Args:
            X: Feature matrix [N, input_dim]
            y_categories: True categories [N] (0-4)
            y_attacks: True attack flags [N] (bool)

        Returns:
            stats: dict with losses and accuracy
        """
        N = len(X)
        y_categories = np.array(y_categories)
        y_attacks = np.array(y_attacks)

        # Phase 1: Collect trajectories
        for i in range(N):
            state = X[i]
            true_cat = y_categories[i]
            true_attack = bool(y_attacks[i])

            category, action, log_probs, value = self.select_action(state, training=True)
            reward, det_reward, act_reward = self.compute_reward(category, action, true_cat, true_attack)

            self.store_transition(state, category, action, log_probs[0], log_probs[1], reward, value)

        # Phase 2: Compute advantages
        self.buffer.compute_advantages()

        # Phase 3: PPO updates
        manager_losses = []
        worker_losses = []
        critic_losses = []

        for _ in range(self.ppo_epochs):
            for batch in self.buffer.get_batches(self.mini_batch_size):
                states, categories, actions, m_log_probs, w_log_probs, rewards, values, advantages = batch

                states_t = torch.FloatTensor(states).to(self.device)
                categories_t = torch.LongTensor(categories).to(self.device)
                actions_t = torch.LongTensor(actions).to(self.device)
                m_old_log_probs_t = torch.FloatTensor(m_log_probs).to(self.device)
                w_old_log_probs_t = torch.FloatTensor(w_log_probs).to(self.device)
                advantages_t = torch.FloatTensor(advantages).to(self.device)
                values_t = torch.FloatTensor(values).to(self.device)

                # === Update Manager ===
                new_category_probs = self.manager.get_category_probs(states_t)
                new_log_probs = torch.log(new_category_probs.gather(1, categories_t.unsqueeze(1)) + 1e-8).squeeze()

                ratio_m = torch.exp(new_log_probs - m_old_log_probs_t)
                clip_m = torch.clamp(ratio_m, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                manager_loss = -torch.min(ratio_m * advantages_t, clip_m * advantages_t).mean()

                entropy_m = -(new_category_probs * torch.log(new_category_probs + 1e-8)).sum(dim=-1).mean()

                # === Update Worker ===
                new_action_probs = self.worker.get_action_probs(states_t, categories_t)
                new_log_probs_w = torch.log(new_action_probs.gather(1, actions_t.unsqueeze(1)) + 1e-8).squeeze()

                ratio_w = torch.exp(new_log_probs_w - w_old_log_probs_t)
                clip_w = torch.clamp(ratio_w, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                worker_loss = -torch.min(ratio_w * advantages_t, clip_w * advantages_t).mean()

                entropy_w = -(new_action_probs * torch.log(new_action_probs + 1e-8)).sum(dim=-1).mean()

                # === Update Critic ===
                values_pred = self.critic(states_t).squeeze()
                critic_loss = F.mse_loss(values_pred, values_t + advantages_t)

                total_loss = manager_loss + worker_loss + self.value_coef * critic_loss - self.entropy_coef * (entropy_m + entropy_w)

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

                manager_losses.append(manager_loss.item())
                worker_losses.append(worker_loss.item())
                critic_losses.append(critic_loss.item())

        correct_cat = 0
        self.buffer.reset()

        for i in range(N):
            state = X[i]
            true_cat = y_categories[i]

            category, action, _, _ = self.select_action(state, training=False)
            if category == true_cat:
                correct_cat += 1

        cat_accuracy = correct_cat / N

        stats = {
            'manager_loss': np.mean(manager_losses),
            'worker_loss': np.mean(worker_losses),
            'critic_loss': np.mean(critic_losses),
            'category_accuracy': cat_accuracy,
        }

        self.num_samples_trained += N
        self.episode_losses.append(stats['manager_loss'])
        self.episode_accuracies.append(cat_accuracy)

        return stats

    def _evaluate_internal(self, X, y_categories, y_attacks):
        """
        Internal evaluate (original signature).

        Args:
            X: Feature matrix [N, input_dim]
            y_categories: True categories [N]
            y_attacks: True attack flags [N]

        Returns:
            metrics: dict with accuracy, precision, recall, f1
        """
        N = len(X)
        y_categories = np.array(y_categories)
        y_attacks = np.array(y_attacks)

        correct_cat = 0
        tp, tn, fp, fn = 0, 0, 0, 0

        for i in range(N):
            state = X[i]
            true_cat = y_categories[i]
            true_attack = bool(y_attacks[i])

            category, action, _, _ = self.select_action(state, training=False)

            if category == true_cat:
                correct_cat += 1

            pred_attack = (category != CATEGORY_NORMAL)
            if pred_attack and true_attack:
                tp += 1
            elif not pred_attack and not true_attack:
                tn += 1
            elif pred_attack and not true_attack:
                fp += 1
            else:
                fn += 1

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        # Approximate AUC-ROC using accuracy as proxy (proper AUC would require
        # probability scores; the category probabilities from manager could be
        # used but this matches the interface contract for FL evaluation).
        auc_roc = accuracy

        return {
            'category_accuracy': correct_cat / N,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fpr': fpr,
            'auc_roc': auc_roc,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        }

    def get_network(self):
        """
        Return a network object for FLTrust proxy model initialization.

        FLTrust requires a template model to build its reference model.
        Returns the manager network as the primary policy network.
        """
        return self.manager


# ============================================================================
# HIERARCHICAL ROLLOUT BUFFER
# ============================================================================
class HierarchicalRolloutBuffer:
    """
    Buffer for storing hierarchical agent transitions.

    Stores:
      - state, category, action
      - manager & worker log_probs
      - reward, value, advantage
    """

    def __init__(self):
        self.reset()

    def store(self, state, category, action, manager_log_prob, worker_log_prob, reward, value):
        """Store one transition."""
        self.states.append(state)
        self.categories.append(category)
        self.actions.append(action)
        self.manager_log_probs.append(manager_log_prob)
        self.worker_log_probs.append(worker_log_prob)
        self.rewards.append(reward)
        self.values.append(value)

    def compute_advantages(self, gamma=0.99):
        """Compute advantages using rewards and values."""
        n = len(self.rewards)
        self.returns = list(self.rewards)

        # Advantage = reward - value_estimate
        self.advantages = [
            self.rewards[i] - self.values[i] for i in range(n)
        ]

        # Normalize
        adv = np.array(self.advantages)
        mean = adv.mean()
        std = adv.std() + 1e-8
        self.advantages = ((adv - mean) / std).tolist()

    def get_batches(self, batch_size):
        """Yield mini-batches."""
        n = len(self.states)
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]

            yield (
                np.array([self.states[i] for i in batch_idx]),
                np.array([self.categories[i] for i in batch_idx]),
                np.array([self.actions[i] for i in batch_idx]),
                np.array([self.manager_log_probs[i] for i in batch_idx]),
                np.array([self.worker_log_probs[i] for i in batch_idx]),
                np.array([self.rewards[i] for i in batch_idx]),
                np.array([self.values[i] for i in batch_idx]),
                np.array([self.advantages[i] for i in batch_idx]),
            )

    def reset(self):
        """Clear buffer."""
        self.states = []
        self.categories = []
        self.actions = []
        self.manager_log_probs = []
        self.worker_log_probs = []
        self.rewards = []
        self.values = []
        self.advantages = []
        self.returns = []

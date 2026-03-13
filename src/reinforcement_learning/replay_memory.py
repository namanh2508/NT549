"""
Prioritized Experience Replay (PER) Module.
============================================================================
Paper Reference: Section 2.2.3, Page 3; Algorithm 1 Lines 15-18, Page 6
============================================================================
"Many RL models stores the previous samples in replay buffer and retrain
 the model with those samples to improve the agents in remembering and
 reusing previous experiences." (Section 2.2.3, Page 3)

"In prioritized experience replay, samples are prioritized based on metrics
 similar to TD." (Section 2.2.3, Page 3)

Priority for i-th sample (Eq. from Section 2.2.3, Page 3):
  P(i) = p_i^alpha / sum(p_k^alpha)
  where p_i = |delta_i| + epsilon

Importance Sampling weight (Section 2.2.3, Page 3):
  w_i = (1/N * 1/P(i))^beta

Algorithm 1 Lines 13-18 (Page 6):
  Line 13: Calculate error_vector = |Q_value_vector - target_Q_value_vector|
  Line 14: Compute current_state_loss_weight = (sum(error_vector))^omega
  Line 15: Store <state, action, reward, loss_weight> in memory M
  Line 17: Select batch from M where P_s proportional to state_loss_weight
  Line 18: Replay training on sampled batch
============================================================================
"""

import numpy as np


class SumTree:
    """
    Sum Tree data structure for efficient prioritized sampling.
    Each leaf stores a priority value, and internal nodes store the sum
    of their children's priorities. This allows O(log n) sampling.
    """

    def __init__(self, capacity):
        self.capacity = capacity  # Max number of leaf nodes
        self.tree = np.zeros(2 * capacity - 1)  # Binary tree array
        self.data = [None] * capacity  # Data storage at leaves
        self.write_pos = 0  # Current write position
        self.num_entries = 0  # Number of entries currently stored

    def _propagate(self, idx, change):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, value):
        """Find leaf node for a given cumulative value."""
        left = 2 * idx + 1
        right = 2 * idx + 2

        if left >= len(self.tree):
            return idx

        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])

    def total(self):
        """Return sum of all priorities."""
        return self.tree[0]

    def add(self, priority, data):
        """Add data with given priority."""
        tree_idx = self.write_pos + self.capacity - 1
        self.data[self.write_pos] = data
        self.update(tree_idx, priority)

        self.write_pos = (self.write_pos + 1) % self.capacity
        self.num_entries = min(self.num_entries + 1, self.capacity)

    def update(self, tree_idx, priority):
        """Update priority at a specific tree index."""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get(self, value):
        """Get data corresponding to a cumulative priority value."""
        idx = self._retrieve(0, value)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayMemory:
    """
    Prioritized Experience Replay Memory.

    Paper Reference: Section 2.2.3, Page 3; Algorithm 1, Page 6

    Stores experience tuples (state, action, reward, loss_weight) and
    samples them with probability proportional to their priority
    (loss_weight).

    Algorithm 1, Line 15 (Page 6):
    "Store the tuple <current_state, action_taken, reward,
     current_state_loss_weight> in memory M"

    Algorithm 1, Line 17 (Page 6):
    "Select a sample batch of size b from the memory M, where the
     probability of selecting each sample, i.e., P_s is proportional
     to the state_loss_weight obtained during its training phase"
    """

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_end=1.0,
                 beta_steps=10000, epsilon=1e-6):
        """
        Args:
            capacity: Maximum number of experiences to store
            alpha: Prioritization exponent (Section 2.2.3, Page 3)
                   Controls how much prioritization is used
            beta_start: Initial importance sampling exponent
            beta_end: Final importance sampling exponent
            beta_steps: Number of steps to anneal beta from start to end
            epsilon: Small constant added to priorities (Section 2.2.3, Page 3)
                     "p_i = |delta_i| + epsilon"
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.epsilon = epsilon
        self.step_count = 0
        self.max_priority = 1.0

    def push(self, state, action, reward, loss_weight):
        """
        Store experience with priority.

        Paper Reference: Algorithm 1, Lines 14-15, Page 6
        "Compute current_state_loss_weight = (sum(error_vector))^omega"
        "Store tuple <current_state, action_taken, reward, loss_weight> in memory M"

        The priority is based on the loss_weight (TD error based).
        """
        # Priority = (loss_weight + epsilon)^alpha
        # Paper: P(i) = p_i^alpha / sum(p_k^alpha) where p_i = |delta_i| + epsilon
        priority = (abs(loss_weight) + self.epsilon) ** self.alpha
        self.max_priority = max(self.max_priority, priority)

        experience = (state, action, reward, loss_weight)
        self.tree.add(priority, experience)

    def sample(self, batch_size):
        """
        Sample a batch of experiences with prioritized sampling.

        Paper Reference: Algorithm 1, Line 17, Page 6
        "Select a sample batch of size b from the memory M, where the
         probability of selecting each sample, i.e., P_s is proportional
         to the state_loss_weight"

        Also applies Importance Sampling (Section 2.2.3, Page 3):
        "w_i = (1/N * 1/P(i))^beta"

        Returns:
            batch: List of (state, action, reward, loss_weight) tuples
            indices: Tree indices for updating priorities
            is_weights: Importance sampling weights
        """
        batch = []
        indices = []
        priorities = []

        # Anneal beta
        self.beta = min(
            self.beta_end,
            self.beta_start + (self.beta_end - self.beta_start)
            * self.step_count / max(self.beta_steps, 1)
        )
        self.step_count += 1

        # Divide total priority into equal segments and sample one from each
        total = self.tree.total()
        if total == 0:
            return [], [], []

        segment = total / batch_size

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = np.random.uniform(low, high)
            idx, priority, data = self.tree.get(value)

            if data is not None:
                batch.append(data)
                indices.append(idx)
                priorities.append(priority)

        if len(batch) == 0:
            return [], [], []

        # Compute Importance Sampling weights (Section 2.2.3, Page 3)
        # w_i = (1/N * 1/P(i))^beta
        priorities = np.array(priorities)
        sampling_probs = priorities / total
        n = self.tree.num_entries

        is_weights = (n * sampling_probs) ** (-self.beta)
        is_weights = is_weights / is_weights.max()  # Normalize

        return batch, indices, is_weights

    def update_priorities(self, indices, loss_weights):
        """
        Update priorities for sampled experiences after replay training.

        Args:
            indices: Tree indices of sampled experiences
            loss_weights: Updated loss weights after replay
        """
        for idx, lw in zip(indices, loss_weights):
            priority = (abs(lw) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.num_entries

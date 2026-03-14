"""
DQN Agent Module - Core Agent Algorithm.
============================================================================
Paper Reference: Algorithm 1, Page 6 (Section 4.1, Page 5)
============================================================================
This implements the complete agent algorithm as described in Algorithm 1.

Algorithm 1 Steps (Page 6):
  1. Feature vectors X1, X2, ..., XN are provided as input
  2. Set epsilon for epsilon-greedy approach
  3. Allocate memory M for prioritized experience replay
  4. Assign total episodes E
  5-19. Training loop:
    - For each episode:
      - Shuffle feature vectors (Line 6)
      - For each feature vector:
        - Set state_vector = Xj (Line 8)
        - Get Q_value_vector from DQN (Line 9)
        - Select action with epsilon-greedy (Line 10)
        - Obtain reward r (Line 11)
        - Update DQN weights using Bellman's equation (Line 12)
        - Calculate error_vector (Line 13)
        - Compute current_state_loss_weight (Line 14)
        - Store experience in memory M (Line 15)
      - Sample batch from M (Line 17)
      - Replay training on sampled batch (Line 18)
============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

from src.models.dqn import DeepQNetwork
from src.reinforcement_learning.replay_memory import PrioritizedReplayMemory
from src.utils.metrics import compute_metrics


class DQNAgent:
    """
    Deep Q-Network Agent for Intrusion Detection.

    Paper Reference: Algorithm 1, Page 6; Section 4.1, Page 5
    "All the agents are equipped with a Deep-Q network for employing the
     reinforcement learning approach to detect the network intrusions."

    Each agent:
    - Maintains its own DQN model
    - Uses epsilon-greedy exploration
    - Stores experiences in Prioritized Experience Replay memory
    - Trains on both live data and replayed experiences
    """

    def __init__(self, agent_id, input_dim, hidden_layers=None, num_actions=2,
                 lr=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=0.995, memory_capacity=10000, batch_size_replay=64,
                 per_alpha=0.6, per_beta_start=0.4, per_beta_end=1.0,
                 omega=0.5, dropout=0.1, device='cpu'):
        """
        Initialize DQN Agent.

        Paper Reference: Algorithm 1 Lines 1-4, Page 6
        "Set the value of epsilon - the hyperparameter of epsilon-greedy approach"
        "Allocate the memory M required for prioritized experience replay"

        Args:
            agent_id: Unique identifier for this agent
            input_dim: Dimension of state vector (feature dimension)
            hidden_layers: DQN hidden layer sizes
            num_actions: Number of actions (2: normal/attack)
            lr: Learning rate
            gamma: Discount factor (Eq. 2-4, Page 3)
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay rate per episode
            memory_capacity: Replay memory size
            batch_size_replay: Batch size for experience replay
            per_alpha: PER prioritization exponent
            per_beta_start: PER importance sampling start
            per_beta_end: PER importance sampling end
            omega: Power factor for loss_weight (Algorithm 1, Line 14)
            dropout: Dropout rate
            device: torch device
        """
        self.agent_id = agent_id
        self.device = torch.device(device)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.omega = omega
        self.batch_size_replay = batch_size_replay
        self.num_actions = num_actions

        # === Algorithm 1, Line 9: Agent's DQN ===
        self.dqn = DeepQNetwork(
            input_dim=input_dim,
            hidden_layers=hidden_layers or [128, 64, 32],
            num_actions=num_actions,
            dropout=dropout
        ).to(self.device)

        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()  # Huber Loss (Section 2.2.3, Page 3)

        # === Algorithm 1, Line 3: Allocate memory M ===
        # "Allocate the memory M required for prioritized experience replay"
        self.memory = PrioritizedReplayMemory(
            capacity=memory_capacity,
            alpha=per_alpha,
            beta_start=per_beta_start,
            beta_end=per_beta_end,
        )

        # Training statistics
        self.episode_losses = []
        self.episode_accuracies = []
        self.num_samples_trained = 0

    def train_episode(self, X, y):
        """
        Train one episode on given data.

        Paper Reference: Algorithm 1, Lines 5-19, Page 6
        This implements the full training loop for one episode.

        Args:
            X: Feature vectors numpy array [N, input_dim]
            y: Labels numpy array [N]
        Returns:
            avg_loss: Average loss for this episode
            accuracy: Accuracy for this episode
        """
        self.dqn.train()

        # === Algorithm 1, Line 6: Shuffle feature vectors ===
        # "Shuffle all the available feature vectors to randomize the sequence of states"
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        total_loss = 0.0
        correct_predictions = 0
        n_samples = len(X)

        # === Algorithm 1, Lines 7-16: Iterate over feature vectors ===
        for j in range(n_samples):
            # === Line 8: Set state_vector = Xj ===
            state = torch.FloatTensor(X_shuffled[j]).unsqueeze(0).to(self.device)
            true_label = int(y_shuffled[j])

            # === Line 9: Get Q_value_vector from DQN ===
            # "Send the current state_vector as input to agent's DQN
            #  and collect the output values as Q_value_vector"
            q_values = self.dqn(state)

            # === Line 10: Select action with epsilon-greedy ===
            # "Select action based on Q_value_vector,
            #  action = argmax(Q values) with epsilon-greedy methodology"
            action, _ = self.dqn.get_action(state, self.epsilon)

            # === Line 11: Obtain reward r ===
            # "Based on the actual output from the dataset and the predicted
            #  action, obtain the reward r"
            if action == true_label:
                reward = 1.0  # Correct prediction
                correct_predictions += 1
            else:
                reward = -1.0  # Incorrect prediction

            # === Line 12: Update DQN weights using Bellman's equation ===
            # "Q_new(s, a, r) = r + gamma * Q_old(s, a, r)"
            # 'a' is the action TAKEN (from epsilon-greedy, Line 10),
            # NOT the true label. This ensures:
            #   - Correct action  -> Q[action] pushed UP   (reward=+1)
            #   - Wrong action    -> Q[action] pushed DOWN (reward=-1)
            target_q_values = q_values.clone().detach()
            target_q_values[0, action] = reward + self.gamma * q_values[0, action].item()

            # Compute loss and update
            loss = self.criterion(q_values, target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # === Line 13: Calculate error_vector ===
            # "error_vector = |Q_value_vector - target_Q_value_vector|"
            # "where the target_value_vector contains 0s at all indices
            #  except at the actual action's index (contains 1)"
            with torch.no_grad():
                target_vector = torch.zeros(self.num_actions).to(self.device)
                target_vector[true_label] = 1.0
                error_vector = torch.abs(
                    torch.softmax(q_values[0], dim=0) - target_vector
                )

                # === Line 14: Compute current_state_loss_weight ===
                # "current_state_loss_weight = (sum(error_vector[i]))^omega"
                state_loss_weight = (error_vector.sum().item()) ** self.omega

            # === Line 15: Store experience in memory M ===
            # "Store tuple <current_state, action_taken, reward,
            #  current_state_loss_weight> in memory M"
            self.memory.push(
                X_shuffled[j].copy(),
                action,
                reward,
                state_loss_weight
            )

        # === Algorithm 1, Lines 17-18: Prioritized Experience Replay ===
        # "Select a sample batch of size b from memory M, where probability
        #  of selecting each sample P_s is proportional to state_loss_weight"
        replay_loss = self._replay_training()

        # Update epsilon (decay after each episode)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        avg_loss = total_loss / n_samples
        accuracy = correct_predictions / n_samples
        self.num_samples_trained += n_samples
        self.episode_losses.append(avg_loss)
        self.episode_accuracies.append(accuracy)

        return avg_loss, accuracy

    def _replay_training(self):
        """
        Perform prioritized experience replay training.

        Paper Reference: Algorithm 1, Lines 17-18, Page 6
        "Select a sample batch of size b from the memory M, where the
         probability of selecting each sample, i.e., P_s is proportional
         to the state_loss_weight obtained during its training phase"
        "Replay the training process on the sampled batch"
        """
        if len(self.memory) < self.batch_size_replay:
            return 0.0

        # Sample batch from memory
        batch, indices, is_weights = self.memory.sample(self.batch_size_replay)
        if len(batch) == 0:
            return 0.0

        is_weights = torch.FloatTensor(is_weights).to(self.device)

        total_replay_loss = 0.0
        new_loss_weights = []

        for i, (state, action, reward, _) in enumerate(batch):
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # Get current Q values
            q_values = self.dqn(state_t)

            # Build target: r + gamma * Q_old(s, a) — same Bellman as live training
            target_q = q_values.clone().detach()
            target_q[0, action] = reward + self.gamma * q_values[0, action].item()

            # Apply importance sampling weight
            loss = self.criterion(q_values, target_q) * is_weights[i]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_replay_loss += loss.item()

            # Update priority
            with torch.no_grad():
                new_q = self.dqn(state_t)
                td_error = abs(reward + self.gamma * new_q[0, action].item() - new_q[0, action].item())
                new_loss_weights.append(td_error ** self.omega)

        # Update priorities in memory
        self.memory.update_priorities(indices[:len(new_loss_weights)], new_loss_weights)

        return total_replay_loss / len(batch) if len(batch) > 0 else 0.0

    def evaluate(self, X, y):
        """
        Evaluate the agent on given data.

        Paper Reference: Algorithm 3, Line 3, Page 6
        "Compute the accuracy of the aggregated model on the available test dataset"

        Args:
            X: Feature vectors [N, input_dim]
            y: True labels [N]
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        self.dqn.eval()
        predictions = []
        q_probs = []

        with torch.no_grad():
            for i in range(len(X)):
                state = torch.FloatTensor(X[i]).unsqueeze(0).to(self.device)
                q_values = self.dqn(state)
                pred = q_values.argmax(dim=1).item()
                predictions.append(pred)

                # Probability for AUC calculation
                probs = torch.softmax(q_values[0], dim=0)
                q_probs.append(probs[1].item())  # Probability of attack class

        predictions = np.array(predictions)
        q_probs = np.array(q_probs)

        return compute_metrics(y, predictions, q_probs)

    def get_weights(self):
        """
        Get DQN model weights for federated aggregation.

        Paper Reference: Algorithm 2, Page 6
        "Input: Deep Q-Network weights for each of the available agents: W1, W2, ...WN"
        """
        return self.dqn.get_weights()

    def set_weights(self, weights):
        """
        Set DQN model weights from aggregated model.

        Paper Reference: Algorithm 2, Line 8, Page 6
        "each agent will then update its own Q-network with this aggregated
         network weight matrix and continue its training process"
        """
        self.dqn.set_weights(weights)

    def get_num_samples_trained(self):
        """
        Get number of samples used in current round.

        Paper Reference: Algorithm 3, Line 5, Page 6
        "obtain the number of samples = num_samples used during the training process"
        """
        return self.num_samples_trained

    def reset_round_counter(self):
        """Reset the sample counter for a new federated round."""
        self.num_samples_trained = 0

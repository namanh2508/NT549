"""
Deep Q-Network (DQN) Module.
============================================================================
Paper Reference: Section 2.2.2 (Page 3), Section 4.1 (Page 5), Algorithm 1 (Page 6)
============================================================================
"In Deep Q-Learning, the Q values for all possible action space for a state
 is estimated with the help of Deep Neural Networks." (Section 2.2.2, Page 3)

"All the agents are equipped with a Deep-Q network for employing the
 reinforcement learning approach to detect the network intrusions." (Section 4.1, Page 5)

DQN Loss Function (Eq. 4, Page 3):
  loss = (r + gamma * max_a' Q_hat(s, a') - Q(s, a))^2

Architecture:
  Input: State vector (feature_dim or DAE hidden_dim)
  Hidden layers: Fully connected with ReLU
  Output: Q-values for 2 actions [normal, attack]
============================================================================
"""

import torch
import torch.nn as nn
import copy


class DeepQNetwork(nn.Module):
    """
    Deep Q-Network for intrusion detection.

    Paper Reference: Section 4.1, Page 5; Algorithm 1, Page 6
    "The preprocessed state vector will then be provided as the input to the
     agent's Deep Q-Network and the Q values get generated."

    Input: State vector (network traffic features)
    Output: Q-values for 2 actions:
      - Action 0: Non-malicious (Normal)
      - Action 1: Malicious (Attack)

    "If action 0 has maximum Q value, it indicates that the agent predicts
     the given state or the current sample belongs to the non malicious
     category. Similarly if action 1 has maximum Q values indicates that
     the agent predicts the given state belongs to an attack/malicious
     category." (Section 4.1, Page 5)
    """

    def __init__(self, input_dim, hidden_layers=None, num_actions=2, dropout=0.1):
        """
        Args:
            input_dim: Dimension of input state vector
            hidden_layers: List of hidden layer sizes, e.g. [128, 64, 32]
            num_actions: Number of actions (2 for binary classification)
            dropout: Dropout rate for regularization
        """
        super(DeepQNetwork, self).__init__()

        if hidden_layers is None:
            hidden_layers = [128, 64, 32]

        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # Output layer: Q-values for each action
        layers.append(nn.Linear(prev_dim, num_actions))

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        """
        Forward pass: compute Q-values for all actions.

        Paper Reference: Algorithm 1, Line 9, Page 6
        "Send the current state_vector as the input to agent's DQN and
         collect the output values as Q_value_vector"

        Args:
            state: State tensor [batch_size, input_dim]
        Returns:
            q_values: Q-value tensor [batch_size, num_actions]
        """
        return self.network(state)

    def get_action(self, state, epsilon=0.0):
        """
        Select action using epsilon-greedy policy.

        Paper Reference: Algorithm 1, Line 10, Page 6
        "Select action based on the Q_value_vector,
         action = argmax(Q values of the two actions)
         with the epsilon-greedy methodology"

        Args:
            state: State tensor [1, input_dim]
            epsilon: Exploration rate for epsilon-greedy
        Returns:
            action: Selected action (0 or 1)
            q_values: Q-values for all actions
        """
        if torch.rand(1).item() < epsilon:
            # Explore: random action
            action = torch.randint(0, 2, (1,)).item()
            with torch.no_grad():
                q_values = self.forward(state)
        else:
            # Exploit: best action
            with torch.no_grad():
                q_values = self.forward(state)
                action = q_values.argmax(dim=1).item()

        return action, q_values

    def get_weights(self):
        """
        Get model weights as a state dict.
        Used for federated aggregation (Algorithm 2, Page 6).
        """
        return copy.deepcopy(self.state_dict())

    def set_weights(self, weights):
        """
        Set model weights from a state dict.

        Paper Reference: Algorithm 2, Line 8, Page 6
        "each agent will then update its own Q-network with this
         aggregated network weight matrix and continue its training process"
        """
        self.load_state_dict(weights)

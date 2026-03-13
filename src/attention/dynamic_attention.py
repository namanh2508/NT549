"""
Dynamic Attention Value Computation Module.
============================================================================
Paper Reference: Algorithm 3, Page 6; Section 4.2-4.4, Pages 6-7; Eq. 6, 8, 9, 14-15
============================================================================
Algorithm 3: Dynamic Attention value computation (Page 6)
  Input: Test dataset representing data distribution at the current agent
         and the aggregated model weights received from the central server
  1. Initialize attention_multiplier = 1
  2. Compute accuracy of the aggregated model on the test dataset
  3. Calculate attention_multiplier = 1 + k * (1 - accuracy) * a^(-accuracy)
     (Eq. 6/15, Page 6-7)
  4. Compute attention_value = num_samples * attention_multiplier
     (Eq. 8-9, Page 7)

Attention Multiplier Derivation (Section 4.4, Pages 7):
  - Passes through points (0, 1+k) and (1, 1)
  - k controls maximum attention multiplier value
  - a controls speed of attention drop as accuracy increases
  - Final form: attention_multiplier = 1 + k * (1 - accuracy) * a^(-accuracy)
    (Eq. 14-15, Page 7)

Purpose (Section 4.2-4.3, Pages 6-7):
  - Agents with more data get higher base attention (num_samples)
  - Agents with lower accuracy get higher multiplier (dynamic component)
  - Ensures fair aggregation even with uneven data distributions

Parameters (Section 6.1, Page 8-10):
  - Random split: k=30, a=50
  - Customized split NSL-KDD: k=50000, a=200
============================================================================
"""

import numpy as np


def compute_attention_multiplier(accuracy, k, a):
    """
    Compute the dynamic attention multiplier for an agent.

    Paper Reference: Eq. 6 (Page 6), Eq. 14-15 (Page 7), Algorithm 3 Line 4 (Page 6)
    "attention_multiplier = 1 + k * (1 - accuracy) * a^(-accuracy)"

    Derivation (Section 4.4, Page 7):
    - This equation passes through (accuracy=0, multiplier=1+k) and (accuracy=1, multiplier=1)
    - When accuracy is low, multiplier is high -> more weight in aggregation
    - When accuracy is high, multiplier approaches 1 -> standard weighting
    - k: controls maximum attention value (when accuracy=0, multiplier=1+k)
    - a: controls the decay speed of the multiplier as accuracy increases

    "The value of k determines how high the attention value should be and
     the value of a determines how fast the attention multiplier drops as
     the accuracy moves from 0 to 1." (Section 4.3, Page 7)

    Args:
        accuracy: Model accuracy on agent's test set (0 to 1)
        k: Maximum attention multiplier parameter
        a: Decay speed parameter
    Returns:
        attention_multiplier: float >= 1
    """
    # Clamp accuracy to [0, 1]
    accuracy = np.clip(accuracy, 0.0, 1.0)

    # Eq. 6/15 (Page 6-7):
    # attention_multiplier = 1 + k * (1 - accuracy) * a^(-accuracy)
    attention_multiplier = 1.0 + k * (1.0 - accuracy) * (a ** (-accuracy))

    return attention_multiplier


def compute_static_attention_value(num_samples):
    """
    Compute static attention value (without dynamic component).

    Paper Reference: Eq. 7, Section 4.2, Page 6
    "A simple formulation for the attention value is to use the number of
     samples used in the current round of training process at the agent"
    "static_attention_value = number_of_training_samples"

    Args:
        num_samples: Number of training samples at the agent
    Returns:
        static_attention_value: int
    """
    return num_samples


def compute_dynamic_attention_value(num_samples, accuracy, k, a):
    """
    Compute dynamic attention value using the attention multiplier.

    Paper Reference: Eq. 8-9, Section 4.3, Page 7; Algorithm 3 Line 5, Page 6
    "dynamic_attention_value = num_training_samples * attention_multiplier" (Eq. 8)

    Substituting the attention_multiplier from Eq. 6 gives Eq. 9 (Page 7):
    "dynamic_attention_value = num_training_samples * (1 + k * (1 - accuracy) * a^(-accuracy))"

    "Compute the attention value using the equation
     attention_value = num_samples * attention_multiplier, and send it to
     the central server along with the local network weights for model
     aggregation." (Algorithm 3 Line 5, Page 6)

    Args:
        num_samples: Number of training samples at the agent
        accuracy: Model accuracy on agent's test data
        k: Attention parameter k
        a: Attention parameter a
    Returns:
        dynamic_attention_value: float
    """
    multiplier = compute_attention_multiplier(accuracy, k, a)
    dynamic_attention_value = num_samples * multiplier
    return dynamic_attention_value


class AttentionManager:
    """
    Manages dynamic attention values for all agents.

    Paper Reference: Algorithm 3, Page 6; Section 4.2-4.4, Pages 6-7

    This class tracks attention values across federated rounds
    and provides methods to compute and update them.
    """

    def __init__(self, num_agents, k=30, a=50):
        """
        Args:
            num_agents: Number of agents in the system
            k: Attention parameter k (Section 6.1, Page 8)
               "For the first experiment type, i.e., random data split,
                the parameters k and a have been tuned to the values 30
                and 50 respectively, resulting in the best performance"
            a: Attention parameter a
        """
        self.num_agents = num_agents
        self.k = k
        self.a = a

        # Track attention values per round for each agent
        # Paper Reference: Figures 3.c, 4.c, 5.c, 6.c (Pages 9-12)
        # These figures plot attention_value vs round_number for each agent
        self.attention_history = {i: [] for i in range(num_agents)}

        # Initialize attention multipliers to 1 (Algorithm 3, Line 2)
        # "Initialize attention_multiplier = 1"
        self.attention_multipliers = {i: 1.0 for i in range(num_agents)}

    def compute_attention(self, agent_id, num_samples, accuracy):
        """
        Compute dynamic attention value for a specific agent.

        Paper Reference: Algorithm 3, Lines 3-5, Page 6
        1. "Compute the accuracy of the aggregated model on the available test dataset"
        2. "Calculate attention_multiplier = 1 + k * (1 - accuracy) * a^(-accuracy)"
        3. "attention_value = num_samples * attention_multiplier"

        Args:
            agent_id: Agent identifier
            num_samples: Number of training samples used by this agent
            accuracy: Accuracy of aggregated model on agent's test dataset
        Returns:
            attention_value: Dynamic attention value
        """
        # Compute attention multiplier (Algorithm 3 Line 4)
        multiplier = compute_attention_multiplier(accuracy, self.k, self.a)
        self.attention_multipliers[agent_id] = multiplier

        # Compute attention value (Algorithm 3 Line 5)
        attention_value = compute_dynamic_attention_value(
            num_samples, accuracy, self.k, self.a
        )

        # Store history for plotting (Figures 3.c, 4.c, 5.c, 6.c)
        self.attention_history[agent_id].append(attention_value)

        return attention_value

    def get_attention_history(self):
        """Get attention value history for all agents (for plotting)."""
        return self.attention_history

    def get_multipliers(self):
        """Get current attention multipliers for all agents."""
        return self.attention_multipliers

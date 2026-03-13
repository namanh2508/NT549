"""
Central Server Module - Federated Aggregation.
============================================================================
Paper Reference: Algorithm 2, Page 6; Section 4, Pages 4-5
============================================================================
Algorithm 2: Central server algorithm (Page 6)
  Input: Deep Q-Network weights for each agent: W1, W2, ...WN
         Attention values for each agent: A1, A2, ...AN
  1. Calculate total attention sum: AT = sum(Ai) for i=1 to N
  2. Initialize weight matrix W_sum with zeros
  3. For each agent k:
       W_sum = W_sum + Ak * Wk
  4. Compute aggregated weights: W_agg = W_sum / AT
  5. Send W_agg to all participating agents

This is an attention-weighted variant of FedAvg where the weights in
the averaging are determined by the dynamic attention mechanism instead
of just the number of samples.
============================================================================
"""

import torch
import copy
from collections import OrderedDict


class CentralServer:
    """
    Central Server for Federated Model Aggregation.

    Paper Reference: Algorithm 2, Page 6
    "The central server will be provided with the network weights of all the
     available agents along with their attention values. After obtaining the
     input, it will compute W_sum, sum of network weight matrices which are
     multiplied by their respective attention values."

    The server:
    1. Collects DQN weights from all agents
    2. Collects attention values from all agents
    3. Computes attention-weighted average of all agent models
    4. Distributes the aggregated model back to all agents
    """

    def __init__(self):
        """Initialize the central server."""
        self.aggregated_weights = None
        self.round_number = 0
        self.aggregation_history = []  # Track aggregation stats per round

    def aggregate(self, agent_weights, attention_values):
        """
        Perform attention-weighted model aggregation.

        Paper Reference: Algorithm 2, Lines 1-7, Page 6

        Step-by-step (matching Algorithm 2):
        Line 2: "Calculate the sum of attention values of all available agents.
                  Total attention sum AT = sum(Ai) for i=1 to N"
        Line 3: "Initialize a weight matrix W_sum with zeros in the shape of
                  Q-Network weights"
        Lines 4-6: "For each agent k: W_sum = W_sum + Ak * Wk"
                    "where Ak and Wk are the attention value and the weight
                     of the kth agent respectively"
        Line 7: "Divide the calculated weighted sum by the total attention
                  sum to obtain the aggregated network weights,
                  i.e., W_agg = W_sum / AT"

        Args:
            agent_weights: List of state_dicts, one per agent [W1, W2, ..., WN]
            attention_values: List of attention values [A1, A2, ..., AN]
        Returns:
            aggregated: OrderedDict of aggregated model weights (W_agg)
        """
        assert len(agent_weights) == len(attention_values), \
            "Number of agent weights must match number of attention values"

        num_agents = len(agent_weights)

        # === Algorithm 2, Line 2: Compute total attention sum ===
        # "Total attention sum AT = sum(Ai) for i=1 to N"
        total_attention_sum = sum(attention_values)

        if total_attention_sum == 0:
            total_attention_sum = 1.0  # Avoid division by zero

        # === Algorithm 2, Line 3: Initialize W_sum with zeros ===
        # "Initialize a weight matrix W_sum with zeros in the shape of Q-Network weights"
        reference_weights = agent_weights[0]
        w_sum = OrderedDict()
        for key in reference_weights:
            w_sum[key] = torch.zeros_like(reference_weights[key], dtype=torch.float32)

        # === Algorithm 2, Lines 4-6: Accumulate weighted sum ===
        # "For each agent k: W_sum = W_sum + Ak * Wk"
        for k in range(num_agents):
            ak = attention_values[k]  # Attention value of agent k
            wk = agent_weights[k]     # Weights of agent k's DQN
            for key in w_sum:
                w_sum[key] += ak * wk[key].float()

        # === Algorithm 2, Line 7: Divide by total attention sum ===
        # "W_agg = W_sum / AT"
        aggregated = OrderedDict()
        for key in w_sum:
            aggregated[key] = w_sum[key] / total_attention_sum

        self.aggregated_weights = aggregated
        self.round_number += 1

        # Log aggregation statistics
        self.aggregation_history.append({
            'round': self.round_number,
            'num_agents': num_agents,
            'attention_values': list(attention_values),
            'total_attention_sum': total_attention_sum,
        })

        return aggregated

    def get_aggregated_weights(self):
        """
        Get the aggregated model weights.

        Paper Reference: Algorithm 2, Line 8, Page 6
        "Send the computed W_agg to all the participating agents and each
         agent will then update its own Q-network with this aggregated
         network weight matrix and continue its training process"
        """
        return copy.deepcopy(self.aggregated_weights)

    def get_aggregation_history(self):
        """Get all aggregation statistics for analysis."""
        return self.aggregation_history

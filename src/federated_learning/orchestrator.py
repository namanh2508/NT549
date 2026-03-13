"""
Federated Learning Orchestrator Module.
============================================================================
Paper Reference: Section 4, Pages 4-6; Algorithms 1-3, Page 6
============================================================================
This module ties together all components of the FDRL-IDS:
  - Multiple DQN Agents (Algorithm 1)
  - Central Server for aggregation (Algorithm 2)
  - Dynamic Attention Mechanism (Algorithm 3)

Overall Training Pipeline (Section 4, Pages 4-5):
  1. Data is distributed among agents (random or customized split)
  2. Each agent has a DAE for feature denoising (Section 4.1, Page 5)
  3. For each federated round:
     a. Each agent trains locally for E episodes (Algorithm 1)
     b. Each agent computes attention value (Algorithm 3)
     c. Central server aggregates models using attention-weighted averaging (Algorithm 2)
     d. Aggregated weights are sent back to all agents
  4. Evaluate final model on test data

"In our system, the data at each agent node is not shared with any other
 nodes. At the same time, however, all the agents in the system benefit,
 via the attention weighted model aggregation process, from the distribution
 and pattern of the data available at all the other agents." (Page 1, Abstract)
============================================================================
"""

import numpy as np
import torch
import copy
import time

from src.reinforcement_learning.agent import DQNAgent
from src.federated_learning.server import CentralServer
from src.attention.dynamic_attention import AttentionManager
from src.utils.metrics import compute_metrics, print_metrics


class FederatedOrchestrator:
    """
    Orchestrator for the Federated Deep Reinforcement Learning IDS.

    Paper Reference: Section 4, Pages 4-6
    "We present a Federated Deep Reinforcement Learning-based IDS in which
     multiple agents are deployed on the network in a distributed fashion,
     and each of these agents runs a Deep Q-Network logic."

    This class coordinates:
    - Agent creation and data assignment
    - Local training at each agent
    - Dynamic attention computation
    - Central server aggregation
    - Evaluation after each round
    """

    def __init__(self, num_agents, input_dim, hidden_layers=None, num_actions=2,
                 lr=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=0.995, memory_capacity=10000, batch_size_replay=64,
                 per_alpha=0.6, per_beta_start=0.4, per_beta_end=1.0,
                 omega=0.5, dropout=0.1, attention_k=30, attention_a=50,
                 episodes_per_round=3, device='cpu'):
        """
        Initialize the federated learning system.

        Paper Reference: Section 6, Page 8
        "we simulated an IDS containing eight agents and one central server"

        Args:
            num_agents: Number of distributed agents (default 8 for random split)
            input_dim: Feature dimension (after DAE encoding)
            hidden_layers: DQN hidden layers
            num_actions: 2 (binary: normal/attack)
            lr: DQN learning rate
            gamma: RL discount factor (Eq. 2-4, Page 3)
            epsilon_start/end/decay: Epsilon-greedy params (Alg. 1, Line 2)
            memory_capacity: PER memory size (Alg. 1, Line 3)
            batch_size_replay: PER batch size (Alg. 1, Line 17)
            per_alpha/per_beta_start/per_beta_end: PER params (Section 2.2.3, Page 3)
            omega: Loss weight power (Alg. 1, Line 14)
            dropout: DQN dropout
            attention_k: Attention param k (Section 6.1, Page 8)
            attention_a: Attention param a (Section 6.1, Page 8)
            episodes_per_round: Local episodes per federated round
            device: torch device string
        """
        self.num_agents = num_agents
        self.input_dim = input_dim
        self.episodes_per_round = episodes_per_round
        self.device = device

        # === Create distributed agents (Section 4.1, Page 5) ===
        # "multiple agents are deployed on the network in a distributed fashion"
        self.agents = []
        for i in range(num_agents):
            agent = DQNAgent(
                agent_id=i,
                input_dim=input_dim,
                hidden_layers=hidden_layers,
                num_actions=num_actions,
                lr=lr,
                gamma=gamma,
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                epsilon_decay=epsilon_decay,
                memory_capacity=memory_capacity,
                batch_size_replay=batch_size_replay,
                per_alpha=per_alpha,
                per_beta_start=per_beta_start,
                per_beta_end=per_beta_end,
                omega=omega,
                dropout=dropout,
                device=device
            )
            self.agents.append(agent)

        # === Central server (Algorithm 2, Page 6) ===
        self.server = CentralServer()

        # === Dynamic attention manager (Algorithm 3, Page 6) ===
        # Parameters k and a from Section 6.1, Page 8-10
        self.attention_manager = AttentionManager(
            num_agents=num_agents,
            k=attention_k,
            a=attention_a
        )

        # Training history for all agents across rounds
        # Used for plotting (Figures 3-6, Pages 9-12)
        self.round_accuracies = {i: [] for i in range(num_agents)}
        self.round_losses = {i: [] for i in range(num_agents)}
        self.round_attention_values = {i: [] for i in range(num_agents)}

    def assign_data(self, agent_data_list, test_split_ratio=0.2):
        """
        Assign data to each agent and split into train/test.

        Paper Reference: Section 6, Page 8
        Random: "all the available data is randomly split into approximately
                 equal parts and shared among the agents"
        Customized: "one agent is supplied with data samples containing only
                     the normal and DOS attack samples, while the other agent
                     is given normal as well as Probe, U2R, and R2L attack types"
                     (Section 6.1, Page 9)

        The test set at each agent is used for attention computation:
        Algorithm 3, Page 6:
        "test dataset is a constant subset sampled out from the available
         training data to represent its data distribution at the agent"

        Args:
            agent_data_list: List of (X, y) tuples, one per agent
            test_split_ratio: Fraction used as test set for attention
        """
        assert len(agent_data_list) == self.num_agents

        self.agent_train_data = []
        self.agent_test_data = []

        for i, (X, y) in enumerate(agent_data_list):
            # Split into train and test for attention computation
            n = len(X)
            n_test = int(n * test_split_ratio)
            indices = np.arange(n)
            np.random.shuffle(indices)

            test_idx = indices[:n_test]
            train_idx = indices[n_test:]

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            self.agent_train_data.append((X_train, y_train))
            self.agent_test_data.append((X_test, y_test))

            print(f"[Orchestrator] Agent {i}: Train={len(X_train)}, "
                  f"Test(attention)={len(X_test)}")

    def train(self, num_rounds, global_test_X=None, global_test_y=None, verbose=True):
        """
        Execute the full federated training process.

        Paper Reference: Algorithms 1-3 combined, Section 4, Pages 4-6

        For each federated round:
        1. Each agent trains locally for E episodes (Algorithm 1)
        2. Each agent evaluates aggregated model on its test set (Algorithm 3, Line 3)
        3. Each agent computes dynamic attention value (Algorithm 3, Lines 4-5)
        4. Central server aggregates with attention weights (Algorithm 2)
        5. Aggregated weights sent back to all agents (Algorithm 2, Line 8)

        "Simultaneously, during the execution of the above process at each
         of the distributed agents, the central server runs Algorithm 2 and
         the attention values for each of the agents will be computed dynamically
         using Algorithm 3." (Section 4.1, Page 5)

        Args:
            num_rounds: Number of federated rounds
            global_test_X: Optional global test features for evaluation
            global_test_y: Optional global test labels for evaluation
            verbose: Whether to print progress
        Returns:
            history: Dict with training history for all agents
        """
        history = {
            'round_accuracies': {i: [] for i in range(self.num_agents)},
            'round_losses': {i: [] for i in range(self.num_agents)},
            'round_attention_values': {i: [] for i in range(self.num_agents)},
            'global_metrics': [],
        }

        for round_num in range(1, num_rounds + 1):
            round_start = time.time()

            if verbose:
                print(f"\n{'='*60}")
                print(f"  Federated Round {round_num}/{num_rounds}")
                print(f"{'='*60}")

            # ============================================================
            # STEP 1: Local training at each agent (Algorithm 1, Page 6)
            # ============================================================
            # "For each round of training process, the agent is given a
            #  batch of training samples" (Section 4.1, Page 5)
            agent_weights = []
            attention_values = []

            for i in range(self.num_agents):
                agent = self.agents[i]
                X_train, y_train = self.agent_train_data[i]
                X_test, y_test = self.agent_test_data[i]

                # Reset round counter for attention computation
                agent.reset_round_counter()

                # Train for E episodes (Algorithm 1, Line 5: "for i in range 1 to E")
                round_loss = 0.0
                round_acc = 0.0
                for ep in range(self.episodes_per_round):
                    loss, acc = agent.train_episode(X_train, y_train)
                    round_loss += loss
                    round_acc += acc

                avg_loss = round_loss / self.episodes_per_round
                avg_acc = round_acc / self.episodes_per_round

                # ============================================================
                # STEP 2: Evaluate aggregated model on agent's test data
                #         (Algorithm 3, Line 3, Page 6)
                # ============================================================
                # "Compute the accuracy of the aggregated model on the
                #  available test dataset"
                test_metrics = agent.evaluate(X_test, y_test)
                test_accuracy = test_metrics['accuracy']

                # ============================================================
                # STEP 3: Compute dynamic attention value (Algorithm 3, Lines 4-5)
                # ============================================================
                # "attention_value = num_samples * attention_multiplier"
                num_samples = agent.get_num_samples_trained()
                attn_value = self.attention_manager.compute_attention(
                    agent_id=i,
                    num_samples=num_samples,
                    accuracy=test_accuracy
                )

                # Collect weights and attention for aggregation
                agent_weights.append(agent.get_weights())
                attention_values.append(attn_value)

                # Store round history for plotting
                history['round_accuracies'][i].append(avg_acc)
                history['round_losses'][i].append(avg_loss)
                history['round_attention_values'][i].append(attn_value)

                if verbose:
                    multiplier = self.attention_manager.get_multipliers()[i]
                    print(f"  Agent {i}: Loss={avg_loss:.4f}, "
                          f"TrainAcc={avg_acc:.4f}, TestAcc={test_accuracy:.4f}, "
                          f"Attn={attn_value:.1f}, Mult={multiplier:.3f}")

            # ============================================================
            # STEP 4: Central server aggregation (Algorithm 2, Page 6)
            # ============================================================
            # "central server will compute W_sum, sum of network weight
            #  matrices multiplied by their respective attention values"
            aggregated_weights = self.server.aggregate(
                agent_weights, attention_values
            )

            # ============================================================
            # STEP 5: Distribute aggregated weights to all agents
            #         (Algorithm 2, Line 8, Page 6)
            # ============================================================
            # "Send the computed W_agg to all the participating agents
            #  and each agent will then update its own Q-network with this
            #  aggregated network weight matrix"
            for i in range(self.num_agents):
                self.agents[i].set_weights(
                    copy.deepcopy(aggregated_weights)
                )

            # ============================================================
            # STEP 6 (Optional): Evaluate on global test set
            # ============================================================
            if global_test_X is not None and global_test_y is not None:
                # Use agent 0 (has the aggregated weights) for global eval
                global_metrics = self.agents[0].evaluate(global_test_X, global_test_y)
                history['global_metrics'].append(global_metrics)

                if verbose:
                    elapsed = time.time() - round_start
                    print(f"  >> Global Acc={global_metrics['accuracy']:.4f}, "
                          f"FPR={global_metrics['fpr']:.4f}, "
                          f"AUC={global_metrics['auc_roc']:.4f} "
                          f"[{elapsed:.1f}s]")

        return history

    def evaluate_global(self, X_test, y_test):
        """
        Evaluate the aggregated model on global test data.

        Paper Reference: Table 3, Page 11 (Section 6.1)
        Reports accuracy, FPR, recall, precision, F1-Score, AUC-ROC.

        Args:
            X_test: Global test features
            y_test: Global test labels
        Returns:
            metrics: Dict of all evaluation metrics
        """
        # All agents have the same aggregated weights after each round
        return self.agents[0].evaluate(X_test, y_test)

    def get_training_history(self):
        """Get attention value history for all agents (for plotting)."""
        return self.attention_manager.get_attention_history()

    def get_aggregation_history(self):
        """Get server aggregation history."""
        return self.server.get_aggregation_history()

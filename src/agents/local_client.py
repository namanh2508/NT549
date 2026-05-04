"""
Tier-1 Local Client: wraps PPO agent + local IDS environment.
Each local client trains on its partition of data and contributes
to federated aggregation.
"""

import numpy as np
from typing import Dict, Tuple
from collections import OrderedDict

from src.config import Config
from src.agents.ppo_agent import PPOAgent
from src.environment.ids_env import MultiClassIDSEnvironment

import torch


class LocalClient:
    """
    Tier-1 local client (FL client) in the federated RL architecture.
    Runs PPO on a local data partition.
    """

    def __init__(
        self,
        client_id: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        num_classes: int,
        cfg: Config,
        device: torch.device,
    ):
        self.client_id = client_id
        self.num_classes = num_classes
        self.cfg = cfg
        self.device = device

        # Local environment
        self.env = MultiClassIDSEnvironment(
            X=X_train, y=y_train,
            reward_cfg=cfg.reward,
            num_classes=num_classes,
            seed=cfg.training.seed + client_id,
        )

        # Test environment for attention computation
        self.test_env = MultiClassIDSEnvironment(
            X=X_test, y=y_test,
            reward_cfg=cfg.reward,
            num_classes=num_classes,
            shuffle=False,
            seed=cfg.training.seed,
        )

        # PPO agent — dataset-aware backbone (CNN/LSTM/GRU)
        self.ppo = PPOAgent(
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim,
            cfg=cfg.ppo,
            device=device,
            agent_id=client_id,
            dataset=cfg.training.dataset,
        )

        self.num_train_samples = len(X_train)
        self._local_accuracy = 0.0
        self._train_metrics: Dict[str, float] = {}

        # Task 3 Option A: track minority class fraction for minority-aware selection
        # Minority class = the least frequent class in this client's data
        class_counts = {}
        for label in y_train:
            class_counts[label] = class_counts.get(label, 0) + 1
        self._minority_class_id = min(class_counts, key=class_counts.get)
        self._minority_class_fraction = class_counts[self._minority_class_id] / max(len(y_train), 1)

    def train_local(self, num_episodes: int) -> Dict[str, float]:
        """Run local PPO training for num_episodes."""
        # BUG-C fix: reset novelty tracking at start of each FL round
        self.env.reset_novelty_tracking()
        total_reward = 0.0
        total_steps = 0
        ep_accuracies = []

        for ep in range(num_episodes):
            self.ppo.reset_hidden()  # reset recurrent hidden state per episode
            state = self.env.reset()
            done = False
            ep_reward = 0.0
            step = 0

            while not done and step < self.cfg.training.max_steps_per_episode:
                action, log_prob, value = self.ppo.select_action(state)
                next_state, reward, done, info = self.env.step(action)

                # Per-step sample weight: inverse class frequency (rarer class = higher weight)
                # Task 2: WeightedRandomSampler equivalent for RL
                true_label = info.get("true_label", 0)
                sample_weight = float(self.env._class_weights[true_label])
                self.ppo.store_transition(state, action, log_prob, reward, value, done, sample_weight)

                state = next_state
                ep_reward += reward
                step += 1

            total_reward += ep_reward
            total_steps += step
            ep_accuracies.append(self.env.get_accuracy())

        # PPO update after collection
        # Pass class-imbalance parameters from environment (Fix 2)
        update_info = self.ppo.update(
            class_weights=self.env._class_weights,
            focal_gamma=self.env._focal_gamma,
        )

        self._local_accuracy = np.mean(ep_accuracies) if ep_accuracies else 0.0
        self._train_metrics = {
            "client_id": self.client_id,
            "avg_reward": total_reward / max(num_episodes, 1),
            "avg_accuracy": self._local_accuracy,
            "total_steps": total_steps,
            **update_info,
        }
        return self._train_metrics

    def evaluate_on_test(self) -> Tuple[float, float]:
        """Evaluate current model on local test data (for attention computation).

        Returns:
            tuple: (accuracy, mean_episode_loss) where loss is computed over
            the evaluation episode steps using negative reward as proxy loss.
            Higher loss = worse client performance = more attention needed.
        """
        state = self.test_env.reset()
        done = False
        step = 0
        total_loss = 0.0
        n_steps = 0

        while not done and step < len(self.test_env):
            action, _, _ = self.ppo.select_action(state, deterministic=True)
            next_state, reward, done, info = self.test_env.step(action)
            state = next_state
            # Use negative reward as proxy loss: more negative reward = worse performance
            # This correctly captures that lower reward → higher "loss" → more attention needed
            total_loss += -reward
            step += 1
            n_steps += 1

        accuracy = self.test_env.get_accuracy()
        mean_loss = total_loss / max(n_steps, 1)
        self._local_accuracy = accuracy
        # Store for retrieval via current_loss property
        self._current_eval_loss = mean_loss
        return accuracy, mean_loss

    @property
    def current_loss(self) -> float:
        """Return the most recent evaluation loss for attention weighting.

        Falls back to PPO actor_loss if evaluate_on_test() has not been called yet.
        """
        return getattr(self, "_current_eval_loss", None) or self._train_metrics.get("actor_loss", 0.0)

    def get_model_state(self) -> OrderedDict:
        return self.ppo.get_model_state()

    def set_model_state(self, state: OrderedDict):
        self.ppo.set_model_state(state)

    @property
    def local_accuracy(self) -> float:
        return self._local_accuracy

    @property
    def train_metrics(self) -> Dict[str, float]:
        return self._train_metrics

    @property
    def minority_class_fraction(self) -> float:
        """Fraction of this client's data belonging to its minority class."""
        return self._minority_class_fraction

    @property
    def minority_class_id(self) -> int:
        """Class index of this client's minority class."""
        return self._minority_class_id

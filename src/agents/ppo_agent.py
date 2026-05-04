"""
Proximal Policy Optimisation (PPO) agent.

Supports two policy types:
  - Categorical (softmax) policy: for discrete multi-class classification.
    Each action dimension corresponds to one class; output is logits + Categorical.
  - Gaussian policy: for continuous control (legacy; use Categorical for IDS).

The policy type is automatically selected based on the action_dim:
  - action_dim == 1  → binary classification → use Categorical on sigmoid(logit)
  - action_dim >= 2  → multi-class          → use Categorical on softmax(logits)
  - To force Gaussian (continuous control), pass policy_type='gaussian'.

FIX (C): Replaced Gaussian-only policy with Categorical (softmax) policy for
multi-class IDS. Gaussian policy is wrong for classification: it outputs a
distribution over continuous action vectors, but IDS requires selecting one
discrete class from K possibilities. Categorical policy correctly models
this as a softmax over K class logits, which is the standard approach for
any classification task (RL or supervised).

References:
    Schulman et al., "Proximal Policy Optimization Algorithms", arXiv 2017.
    Schulman et al., "High-Dimensional Continuous Control Using GAE", ICLR 2016.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

from src.config import PPOConfig
from src.models.networks import build_actor, CriticNetwork


class RolloutBuffer:
    """Stores transitions for a PPO rollout (categorical actions)."""

    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []   # stored as class indices (int)
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []
        self.sample_weights: List[float] = []  # per-step importance weights (for weighted sampling)

    def add(self, state, action, log_prob, reward, value, done, sample_weight: float = 1.0):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.sample_weights.append(sample_weight)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.states)

    def compute_gae(self, last_value: float, gamma: float, lam: float):
        """Compute Generalised Advantage Estimation."""
        advantages = np.zeros(len(self.rewards), dtype=np.float32)
        returns = np.zeros(len(self.rewards), dtype=np.float32)

        gae = 0.0
        next_value = last_value
        for t in reversed(range(len(self.rewards))):
            delta = (
                self.rewards[t]
                + gamma * next_value * (1 - self.dones[t])
                - self.values[t]
            )
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            advantages[t] = gae
            returns[t] = gae + self.values[t]
            next_value = self.values[t]

        return advantages, returns

    def to_tensors(self, device: torch.device):
        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions = torch.LongTensor(np.array(self.actions)).to(device)
        log_probs = torch.FloatTensor(np.array(self.log_probs)).to(device)
        sample_weights = torch.FloatTensor(np.array(self.sample_weights)).to(device)
        return states, actions, log_probs, sample_weights


class PPOAgent:
    """
    PPO agent for IDS with Categorical (discrete multi-class) policy.

    The actor outputs logits for each class; the distribution is Categorical.
    This is correct for classification tasks where the agent must pick ONE class.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        cfg: PPOConfig,
        device: torch.device,
        agent_id: int = 0,
        dataset: str = "edge_iiot",
    ):
        self.cfg = cfg
        self.device = device
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dataset = dataset

        # Actor: dataset-aware backbone (MLP/CNN/LSTM/GRU) via build_actor factory
        self.actor = build_actor(dataset, state_dim, action_dim, hidden_dim=cfg.hidden_dim, override_backbone="cnn_gru").to(device)
        self.critic = CriticNetwork(state_dim, cfg.hidden_dim).to(device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)

        # LR schedulers to prevent catastrophic forgetting
        self.actor_scheduler = None
        self.critic_scheduler = None

        self.buffer = RolloutBuffer()

        # Hidden state for recurrent models (LSTM/GRU)
        self._h_state = None
        self._c_state = None

    def reset_hidden(self):
        """Reset hidden state for recurrent models. Call at episode/round start."""
        self._h_state = None
        self._c_state = None

    @torch.no_grad()
    def select_action(self, state: np.ndarray, deterministic: bool = False):
        """
        Select action using Categorical policy.

        Returns:
            action: class index as int numpy scalar (for buffer storage)
            log_prob: scalar log probability of the selected action
            value: state value estimate
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits = self.actor(state_t)                        # [1, action_dim]
        dist = torch.distributions.Categorical(logits=logits)

        if deterministic:
            action_idx = dist.probs.argmax(dim=-1)         # greedy: pick best class
        else:
            action_idx = dist.sample()                       # stochastic: sample

        log_prob = dist.log_prob(action_idx).item()
        value = self.critic(state_t).item()

        return action_idx.item(), log_prob, value

    def store_transition(self, state, action, log_prob, reward, value, done, sample_weight: float = 1.0):
        """Store a transition. action should be a class index (int)."""
        self.buffer.add(state, action, log_prob, reward, value, done, sample_weight)

    def update(
        self,
        class_weights: Optional[np.ndarray] = None,
        focal_gamma: float = 2.0,
    ) -> Dict[str, float]:
        """
        Run PPO update on collected rollout.

        Args:
            class_weights: [num_classes] array of per-class weights for focal loss.
                If None, uses uniform weights (no reweighting).
            focal_gamma: focal loss gamma. Higher = more focus on hard (minority) examples.
                gamma=0 disables focal weighting.
        """
        if len(self.buffer) == 0:
            return {}

        # Last value for GAE (bootstrap from last state)
        last_state = torch.FloatTensor(self.buffer.states[-1]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            last_value = self.critic(last_state).item()

        advantages, returns = self.buffer.compute_gae(
            last_value, self.cfg.gamma, self.cfg.gae_lambda
        )

        states, action_indices, old_log_probs, buffer_weights = self.buffer.to_tensors(self.device)
        # action_indices: [batch] LongTensor of class indices
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        # Normalise advantages
        if len(advantages_t) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # Compute class weights for focal loss (Fix 2)
        if class_weights is not None:
            cw_t = torch.FloatTensor(class_weights).to(self.device)
        else:
            cw_t = None

        total_loss_actor = 0.0
        total_loss_critic = 0.0
        total_entropy = 0.0

        dataset_size = len(states)
        for _ in range(self.cfg.ppo_epochs):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, self.cfg.mini_batch_size):
                end = min(start + self.cfg.mini_batch_size, dataset_size)
                idx = indices[start:end]

                mb_states = states[idx]                     # [mb, state_dim]
                mb_actions = action_indices[idx]            # [mb] LongTensor
                mb_old_log_probs = old_log_probs[idx]       # [mb]
                mb_advantages = advantages_t[idx]          # [mb]
                mb_returns = returns_t[idx]                # [mb]
                mb_weights = buffer_weights[idx]            # [mb] — per-step sample weights

                # New log probs under current policy
                logits = self.actor(mb_states)              # [mb, action_dim]
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)   # [mb]
                entropy = dist.entropy()                     # [mb]
                probs = torch.softmax(logits, dim=-1)       # [mb, num_classes]

                # PPO ratio: exp(new_log_prob - old_log_prob)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio, 1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon
                ) * mb_advantages

                # Focal loss: down-weight easy (majority-class) samples
                # focal_weight = (1 - p_true_class)^gamma
                if focal_gamma > 0 and cw_t is not None:
                    # Get probability of the taken action per sample
                    p_taken = probs.gather(1, mb_actions.unsqueeze(1)).squeeze(1)  # [mb]
                    focal_weight = (1.0 - p_taken.detach()).pow(focal_gamma)        # [mb]
                    # Class weight for each sample's true class
                    cls_w = cw_t[mb_actions]                                         # [mb]
                    # Combine: focal * class_weight * per-step sample weight
                    combined_weight = focal_weight * cls_w * mb_weights
                    actor_loss = -(torch.min(surr1, surr2) * combined_weight).mean()
                else:
                    weighted_adv = torch.min(surr1, surr2) * mb_weights
                    actor_loss = -weighted_adv.mean()

                entropy_loss = -entropy.mean()              # maximise entropy

                # Critic loss
                values = self.critic(mb_states)             # [mb]
                critic_loss = nn.MSELoss()(values, mb_returns)

                loss = (
                    actor_loss
                    + self.cfg.value_coef * critic_loss
                    + self.cfg.entropy_coef * entropy_loss
                )

                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                self.actor_optim.step()
                self.critic_optim.step()

                total_loss_actor += actor_loss.item()
                total_loss_critic += critic_loss.item()
                total_entropy += entropy.mean().item()

        self.buffer.clear()

        num_updates = max(1, self.cfg.ppo_epochs * (dataset_size // self.cfg.mini_batch_size))
        return {
            "actor_loss": total_loss_actor / num_updates,
            "critic_loss": total_loss_critic / num_updates,
            "entropy": total_entropy / num_updates,
        }

    # ─── Model state management ────────────

    def get_model_state(self) -> OrderedDict:
        """Return combined actor+critic state dict for federated aggregation."""
        state = OrderedDict()
        for k, v in self.actor.state_dict().items():
            state[f"actor.{k}"] = v.clone()
        for k, v in self.critic.state_dict().items():
            state[f"critic.{k}"] = v.clone()
        return state

    def set_model_state(self, state: OrderedDict):
        """Load combined state dict."""
        actor_state = OrderedDict()
        critic_state = OrderedDict()
        for k, v in state.items():
            if k.startswith("actor."):
                actor_state[k[6:]] = v
            elif k.startswith("critic."):
                critic_state[k[7:]] = v
        self.actor.load_state_dict(actor_state)
        self.critic.load_state_dict(critic_state)

    def get_actor_state(self) -> OrderedDict:
        return OrderedDict(
            (k, v.clone()) for k, v in self.actor.state_dict().items()
        )

    def set_actor_state(self, state: OrderedDict):
        self.actor.load_state_dict(state)

"""
Tier-2 Meta-Agent: Coordinates decisions from multiple local agents.

The meta-agent uses PPO with continuous actions to learn an optimal
policy for combining local agent outputs into refined final decisions.
This is a true RL-based coordinator, not a supervised ensemble.

References:
    Schulman et al., "Proximal Policy Optimization Algorithms", arXiv 2017.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from collections import OrderedDict

from src.config import Config
from src.models.networks import CNNGRUActor, init_weights


class MetaRolloutBuffer:
    """Stores transitions for meta-agent PPO updates."""

    def __init__(self):
        self.states: List[np.ndarray] = []
        self.agent_actions: List[np.ndarray] = []
        self.meta_actions: List[np.ndarray] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def add(self, state, agent_actions, meta_action, log_prob, reward, value, done):
        self.states.append(state)
        self.agent_actions.append(agent_actions)
        self.meta_actions.append(meta_action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)

    def compute_gae(self, last_value: float, gamma: float, lam: float):
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


class MetaCritic(nn.Module):
    """
    Value network for meta-agent.

    Processes [batch, num_agents, action_dim] through Conv1D over the agent
    dimension (treating each client as a timestep), then concatenates with
    state and produces a scalar value.
    """

    def __init__(self, num_agents: int, action_dim: int, state_dim: int,
                 hidden_dim: int = 128):
        super().__init__()
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.state_dim = state_dim

        # Conv1D over agent dimension: [batch, num_agents, action_dim] → [batch, action_dim, num_agents]
        # GroupNorm(1, C): normalize per-sample — safe for any batch/seq size
        self.conv1 = nn.Conv1d(action_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.GroupNorm(1, hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.GroupNorm(1, hidden_dim)

        # After conv: [batch, hidden_dim, num_agents] → pool over num_agents → [batch, hidden_dim]
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Concatenate pooled temporal representation + state → value
        critic_in = hidden_dim + state_dim
        self.net = nn.Sequential(
            nn.Linear(critic_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, agent_actions: torch.Tensor, state: torch.Tensor):
        """
        Args:
            agent_actions: [batch, num_agents, action_dim]
            state: [batch, state_dim]
        Returns:
            value: [batch]
        """
        # Conv1D: [batch, num_agents, action_dim] → [batch, action_dim, num_agents]
        x = agent_actions.permute(0, 2, 1)
        x = torch.relu(self.bn1(self.conv1(x)))    # [batch, hidden, num_agents]
        x = torch.relu(self.bn2(self.conv2(x)))     # [batch, hidden, num_agents]
        x = self.pool(x).squeeze(-1)                # [batch, hidden]

        # Concatenate temporal pooled + state
        x = torch.cat([x, state], dim=-1)           # [batch, hidden + state_dim]
        return self.net(x).squeeze(-1)              # [batch]


class MetaAgent:
    """
    Tier-2 meta-agent using PPO for RL-based coordination.
    Takes local agent actions + state and produces final decision
    via CNNGRUActor (flattened concatenation input), trained with environment reward.
    """

    def __init__(
        self,
        num_agents: int,
        action_dim: int,
        state_dim: int,
        cfg: Config,
        device: torch.device,
    ):
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.device = device
        self.cfg = cfg

        # CNNGRUActor input: per-agent features [action_dim + state_dim]
        # Each agent = 1 timestep for Conv1D temporal learning.
        # Input to Conv1D: [batch, seq_len=num_agents, feature_dim=action_dim+state_dim]
        actor_input_dim = action_dim + state_dim
        self.actor = CNNGRUActor(
            input_dim=actor_input_dim,
            action_dim=action_dim,
            hidden_dim=cfg.training.meta_hidden_dim,
            num_layers=2,
            seq_len=num_agents,   # each client = 1 timestep for Conv1D
            dropout=0.1,
        ).to(device)

        # Critic: Conv1D over agent dimension + state concatenation
        self.critic = MetaCritic(
            num_agents=num_agents,
            action_dim=action_dim,
            state_dim=state_dim,
            hidden_dim=cfg.training.meta_hidden_dim,
        ).to(device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.buffer = MetaRolloutBuffer()

        # PPO hyperparameters (reuse from main PPO config)
        self.clip_epsilon = cfg.ppo.clip_epsilon
        self.ppo_epochs = cfg.ppo.ppo_epochs
        self.entropy_coef = cfg.ppo.entropy_coef
        self.value_coef = cfg.ppo.value_coef
        self.gamma = cfg.ppo.gamma
        self.gae_lambda = cfg.ppo.gae_lambda
        self.max_grad_norm = cfg.ppo.max_grad_norm

    def _build_input(self, agent_actions: np.ndarray, state: np.ndarray) -> torch.Tensor:
        """
        Build reshaped input for CNNGRUActor.

        Reshape: each client = 1 timestep for Conv1D temporal learning.
        agent_actions: [num_agents, action_dim]  → view → [1, num_agents, action_dim]
        state is passed separately (shape [1, state_dim]).

        Returns [1, num_agents, action_dim].
        """
        aa = torch.FloatTensor(agent_actions).to(self.device)          # [num_agents, action_dim]
        st = torch.FloatTensor(state).to(self.device)                  # [state_dim]

        # Tile state to match num_agents for per-agent concatenation
        st_tiled = st.unsqueeze(0).expand(self.num_agents, -1)        # [num_agents, state_dim]

        # Concatenate along feature dim: [num_agents, action_dim + state_dim]
        combined = torch.cat([aa, st_tiled], dim=1)                   # [num_agents, action_dim + state_dim]

        # Reshape to [1, num_agents, action_dim + state_dim] for Conv1D
        return combined.unsqueeze(0)                                   # [1, num_agents, action_dim + state_dim]

    def _build_critic_input(
        self, agent_actions: np.ndarray, state: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build shaped inputs for MetaCritic.

        Args:
            agent_actions: [num_agents, action_dim]
            state: [state_dim]

        Returns:
            (agent_actions_tensor [1, num_agents, action_dim], state_tensor [1, state_dim])
        """
        aa = torch.FloatTensor(agent_actions).to(self.device)    # [num_agents, action_dim]
        st = torch.FloatTensor(state).to(self.device)            # [state_dim]
        return aa.unsqueeze(0), st.unsqueeze(0)                 # [1, num_agents, action_dim], [1, state_dim]

    @torch.no_grad()
    def predict(
        self,
        agent_actions: np.ndarray,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Combine local agent actions into a final decision via learned policy.
        Returns raw logits (for logging/debugging).
        """
        x = self._build_input(agent_actions, state)
        action, _ = self.actor.act(x, deterministic=deterministic)
        return action.cpu().numpy().flatten()

    @torch.no_grad()
    def predict_class(
        self,
        agent_actions: np.ndarray,
        state: np.ndarray,
        deterministic: bool = True,
    ) -> int:
        """
        Combine local agent actions and return the predicted class index.
        Uses CNNGRUActor's Categorical distribution.
        """
        x = self._build_input(agent_actions, state)
        dist = self.actor.get_distribution(x)
        if deterministic:
            return dist.probs.argmax(dim=-1).item()
        return dist.sample().item()

    @torch.no_grad()
    def select_action(
        self,
        agent_actions: np.ndarray,
        state: np.ndarray,
    ):
        """Select action and return (action, log_prob, value) for buffer storage."""
        x = self._build_input(agent_actions, state)                # [1, num_agents, action_dim]
        action, log_prob = self.actor.act(x)
        aa_t, st_t = self._build_critic_input(agent_actions, state)  # [1, num_agents, action_dim], [1, state_dim]
        value = self.critic(aa_t, st_t)
        # Return np.ndarray so buffer stores numpy arrays (not torch.Tensors)
        return (
            action.cpu().numpy().flatten().astype(np.float32),
            log_prob.item(),
            value.item(),
        )

    def store_transition(self, state, agent_actions, meta_action, log_prob, reward, value, done):
        self.buffer.add(state, agent_actions, meta_action, log_prob, reward, value, done)

    def update(self) -> Dict[str, float]:
        """Run PPO update on collected meta-agent rollout."""
        if len(self.buffer) < 2:
            self.buffer.clear()
            return {}

        # Last value for GAE
        last_state = torch.FloatTensor(self.buffer.states[-1]).unsqueeze(0).to(self.device)
        last_aa = torch.FloatTensor(self.buffer.agent_actions[-1]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            last_value = self.critic(last_aa, last_state).item()

        advantages, returns = self.buffer.compute_gae(
            last_value, self.gamma, self.gae_lambda
        )

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        aa = torch.FloatTensor(np.array(self.buffer.agent_actions)).to(self.device)
        meta_acts = torch.FloatTensor(np.array(self.buffer.meta_actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.buffer.log_probs)).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8) if len(advantages_t) > 1 else advantages_t

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        dataset_size = len(states)
        mini_batch = max(32, dataset_size // 4)

        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, mini_batch):
                end = min(start + mini_batch, dataset_size)
                idx = indices[start:end]

                mb_states = states[idx]
                mb_aa = aa[idx]
                mb_meta_acts = meta_acts[idx]
                mb_old_lp = old_log_probs[idx]
                mb_adv = advantages_t[idx]
                mb_ret = returns_t[idx]

                # Actor: [MB, num_agents, action_dim] + state tiled -> [MB, num_agents, action_dim+state_dim]
                mb_st_expanded = mb_states.unsqueeze(1).expand(-1, self.num_agents, -1)
                batch_inputs = torch.cat([mb_aa, mb_st_expanded], dim=2)  # [MB, num_agents, action_dim+state_dim]

                new_lp, entropy, _ = self.actor.evaluate(batch_inputs, mb_meta_acts)
                ratio = torch.exp(new_lp - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

                values = self.critic(mb_aa, mb_states)
                critic_loss = nn.MSELoss()(values, mb_ret)

                loss = actor_loss + self.value_coef * critic_loss

                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optim.step()
                self.critic_optim.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()

        self.buffer.clear()
        num_updates = max(1, self.ppo_epochs * (dataset_size // mini_batch))
        return {
            "meta_actor_loss": total_actor_loss / num_updates,
            "meta_critic_loss": total_critic_loss / num_updates,
        }

    def get_state(self) -> OrderedDict:
        state = OrderedDict()
        for k, v in self.actor.state_dict().items():
            state[f"meta_actor.{k}"] = v.clone()
        for k, v in self.critic.state_dict().items():
            state[f"meta_critic.{k}"] = v.clone()
        return state

    def set_state(self, state: OrderedDict):
        actor_state = OrderedDict()
        critic_state = OrderedDict()
        for k, v in state.items():
            if k.startswith("meta_actor."):
                actor_state[k[11:]] = v
            elif k.startswith("meta_critic."):
                critic_state[k[12:]] = v
        if actor_state:
            self.actor.load_state_dict(actor_state)
        if critic_state:
            self.critic.load_state_dict(critic_state)

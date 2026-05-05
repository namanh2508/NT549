"""
RL-Based Client Selection for FedRL-IDS.

This module replaces the original Tier-2 client selector. After architectural
review, the previous design suffered from two critical flaws:

  [A] Authority overlap: The Selector tried to "filter bad clients" — the same
      job that FLTrust already does in the aggregator. This created a circular
      dependency: Selector picks client X → FLTrust gives X weight≈0 →
      Global model still improves → Selector gets positive reward even though
      it chose WRONG.  This is False Credit Assignment.

  [B] Hybrid score masking: Score_k = π_k × Trust_k × Attention_k with
      top-K deterministic selection effectively disabled the RL policy's
      gradient.  The neural network's learned probabilities were overridden
      by pre-computed trust signals at selection time, so PPO never learned
      which clients are actually good.

NEW DESIGN — Resource Efficiency (Refactoring Phase 2):

  Objective: Minimize communication cost (number of clients selected) while
  maintaining global model quality.  FLTrust handles Byzantine robustness
  (filtering bad gradients); the RL Selector learns to participate with the
  FEWEST clients possible.

  State (7 features per client, REMOVED attention):
    R_k  : FLTrust temporal reputation               ∈ [0, 1]
    l_k  : Evaluation loss (−reward proxy)           ∈ [0, ∞)
    Δ_k  : Model divergence ‖w_k−w_glob‖/‖w_glob‖   ∈ [0, ∞)
    g_k  : Gradient alignment cos(Δ_k, Δ_glob)      ∈ [−1, 1]
    f1_k : Historical F1 EMA                         ∈ [0, 1]
    s_k  : Normalized data share n_k/Σn             ∈ [0, 1]
    m_k  : Minority class fraction                   ∈ [0, 1]

  Action: Bernoulli per client, pure RL policy.
    π_θ(a_k=1|s)_k = sigmoid(logit_k) ∈ (0, 1)
    log_prob = Σ_k [a_k·log p_k + (1−a_k)·log(1−p_k)]
    Selection: stochastic Bernoulli sample; fallback to argmax(p_k) if 0 clients.

  Reward:
    R_t = ΔAcc_global
          − λ · (|S_t| / K)              ← penalize selecting many clients
          − γ · mean_{k∈S_t}(1 − R_k)   ← penalize including untrusted clients
    where λ = 0.5, γ = 1.0.

References:
  PPO: Schulman et al., arXiv 2017.
  FLTrust: Cao et al., NDSS 2021.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

from src.config import PPOConfig


# ─── Helpers ───────────────────────────────────────────────────────────────────

def flatten_state_dict(state: OrderedDict) -> torch.Tensor:
    """Flatten an OrderedDict of tensors into a single 1D tensor."""
    return torch.cat([v.flatten().float() for v in state.values()])


def entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Shannon entropy H(P) = −Σ p · log(p)."""
    p = probs.clamp(min=eps, max=1 - eps)
    return -(p * torch.log(p)).sum(dim=-1)


def compute_model_divergence(client_state: OrderedDict, global_model: OrderedDict) -> float:
    """
    Compute relative model divergence: ||w_client - w_global|| / ||w_global||.

    Measures how different a client's model is from the global model.
    Higher divergence may indicate:
    - Client-specific learning
    - Data distribution shift
    - Potential attack (malicious updates)

    Args:
        client_state: OrderedDict of client's model parameters
        global_model: OrderedDict of global model's parameters

    Returns:
        Relative L2 divergence (float), or 0.0 if global model is zero/near-zero
    """
    w_client = flatten_state_dict(client_state)
    w_global = flatten_state_dict(global_model)

    norm_global = torch.norm(w_global).item()
    if norm_global < 1e-12:
        return 0.0

    diff = w_client - w_global
    divergence = torch.norm(diff).item() / norm_global
    return float(divergence)


def compute_gradient_alignment(
    local_updates: List[OrderedDict],
    global_update: OrderedDict,
    device: torch.device,
) -> List[float]:
    """
    Compute cosine alignment between local gradient updates and global update direction.

    Args:
        local_updates: List of local update OrderedDicts (already computed as post - pre)
        global_update: Server/global update direction
        device: torch device

    Returns:
        List of cosine similarities (floats in [-1, 1]) for each local update
    """
    g_global = flatten_state_dict(global_update).to(device)
    norm_global = torch.norm(g_global).item()

    if norm_global < 1e-12 or len(local_updates) == 0:
        return [0.0] * len(local_updates)

    alignments = []
    for local_update in local_updates:
        g_local = flatten_state_dict(local_update).to(device)
        dot = torch.dot(g_local, g_global).item()
        norm_local = torch.norm(g_local).item()

        if norm_local < 1e-12:
            alignments.append(0.0)
        else:
            cosine = dot / (norm_local * norm_global)
            alignments.append(float(cosine))

    return alignments


# ─── Rollout Buffer ────────────────────────────────────────────────────────────

class SelectorRolloutBuffer:
    """Stores (state, binary-action-vector, log_prob, reward, done) for selector PPO."""

    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []   # [K] binary vectors
        self.log_probs: List[float] = []     # scalar = sum_k Bernoulli log_prob
        self.rewards: List[float] = []
        self.dones: List[bool] = []

    def add(self, state, action: np.ndarray, log_prob: float, reward: float, done: bool):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.states)

    def to_tensors(self, device: torch.device):
        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions = torch.FloatTensor(np.array(self.actions)).to(device)  # [T, K]
        log_probs = torch.FloatTensor(np.array(self.log_probs)).to(device)
        return states, actions, log_probs


# ─── Selector Policy Network ───────────────────────────────────────────────────

class SelectorActor(nn.Module):
    """
    Bernoulli policy network: maps state → K independent sigmoid probabilities.

    Input:  state vector [7K] = [R_0, l_0, Δ_0, g_0, f1_0, s_0, m_0, ...]
    Output: K sigmoid probabilities p_k ∈ (0, 1)
    """

    def __init__(self, input_dim: int, num_clients: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.logits_head = nn.Linear(hidden_dim, num_clients)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.logits_head.weight, gain=0.01)
        nn.init.zeros_(self.logits_head.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return raw logits [batch, num_clients]."""
        h = self.net(state)
        return self.logits_head(h)

    def get_probs(self, state: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities p_k."""
        h = self.net(state)
        return torch.sigmoid(self.logits_head(h))


class SelectorCritic(nn.Module):
    """Value network baseline V(s) for advantage estimation."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


# ─── Main Client Selector ──────────────────────────────────────────────────────

class RLClientSelector:
    """
    Resource-Efficient RL Client Selector for FedRL-IDS.

    Key design decisions:

    1. OBJECTIVE — Resource Efficiency (not client quality filtering):
       The Selector's job is to MINIMIZE communication overhead (number of
       clients selected) while maintaining global model accuracy.  FLTrust
       already handles Byzantine robustness in the aggregator; the Selector
       should NOT try to replicate that.

    2. ACTION SPACE — Pure Bernoulli, no hybrid override:
       The PPO policy outputs p_k = sigmoid(logit_k) per client.
       Actions are sampled directly from Bernoulli(p_k).
       NO hybrid_score = p_k × Trust × Attention override.
       Trust appears in the REWARD (penalizing untrusted selections), not
       in the action computation.

    3. REWARD — Resource Efficiency + Untrusted Penalty:
       R_t = ΔAcc_global − λ·(|S|/K) − γ·mean(1−Trust_k)
       This cleanly separates concerns:
         - ΔAcc: global model quality (what matters)
         - λ term: cost of selecting many clients
         - γ term: penalty for including FLTrust-trusted-low clients

    4. SEPARATION OF CONCERNS:
       FLTrust  → Byzantine robustness (weights bad clients down to 0)
       Selector → Communication efficiency (learns to select fewer clients)
       Neither interferes with the other's job.
    """

    # Reward coefficients — tuned for the new objective
    DEFAULT_LAMBDA = 0.5   # communication cost weight
    DEFAULT_GAMMA = 1.0    # untrusted-penalty weight

    def __init__(
        self,
        num_clients: int,
        state_dim_per_client: int = 7,   # 7 features (attention removed)
        hidden_dim: int = 128,
        cfg: PPOConfig = None,
        device: torch.device = None,
        total_rounds: int = 100,
    ):
        self.num_clients = num_clients
        self.state_dim_per_client = state_dim_per_client
        self.total_state_dim = num_clients * state_dim_per_client
        self.device = device or torch.device("cpu")
        self.cfg = cfg or self._default_ppo_config()
        self.hidden_dim = hidden_dim
        self.total_rounds = total_rounds

        self.actor = SelectorActor(self.total_state_dim, num_clients, hidden_dim).to(self.device)
        self.critic = SelectorCritic(self.total_state_dim, hidden_dim).to(self.device)

        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=cfg.lr_actor if cfg else 3e-4,
        )
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.lr_critic if cfg else 1e-3,
        )

        self.buffer = SelectorRolloutBuffer()

        # Historical F1 EMA per client
        self.f1_ema: List[float] = [0.5] * num_clients
        self.f1_alpha = 0.3

        # Previous round tracking
        self._prev_global_accuracy: float = 0.0
        self._prev_trust_scores: List[float] = [1.0 / num_clients] * num_clients

        # Reward coefficients
        self.lambda_comm = self.DEFAULT_LAMBDA
        self.gamma_untrusted = self.DEFAULT_GAMMA

        # Training hyperparameters
        self.gamma = self.cfg.gamma
        self.gae_lambda = self.cfg.gae_lambda
        self.clip_epsilon = self.cfg.clip_epsilon
        self.ppo_epochs = self.cfg.ppo_epochs
        self.entropy_coef_init = self.cfg.entropy_coef
        self.entropy_coef_min = 0.02
        self.value_coef = self.cfg.value_coef
        self.max_grad_norm = self.cfg.max_grad_norm

        # Selection HHI tracking for diversity penalty
        self._selection_history: List[int] = []

        # Over-selection tracking
        self._selection_counts: np.ndarray = np.zeros(num_clients, dtype=np.int32)
        self._total_rounds_seen: int = 0

        # Cache real state for buffer consistency (Markov property fix)
        self._last_selection_state: Optional[np.ndarray] = None
        self._last_probs: Optional[np.ndarray] = None

    @staticmethod
    def _default_ppo_config() -> PPOConfig:
        from src.config import PPOConfig
        return PPOConfig(
            lr_actor=3e-4,
            lr_critic=1e-3,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            entropy_coef=0.05,
            value_coef=0.5,
            max_grad_norm=0.5,
            ppo_epochs=4,
            mini_batch_size=32,
            hidden_dim=128,
        )

    # ─── Entropy Decay Schedule ─────────────────────────────────────────────

    def entropy_coef_at_round(self, round_idx: int) -> float:
        """
        Linear decay: high entropy early (exploration) → low entropy late (exploitation).
        """
        progress = round_idx / max(self.total_rounds - 1, 1)
        return max(
            self.entropy_coef_min,
            self.entropy_coef_init - progress * (self.entropy_coef_init - self.entropy_coef_min),
        )

    # ─── Curriculum K_sel ────────────────────────────────────────────────────

    @staticmethod
    def k_sel_schedule(
        round_idx: int,
        k_init: int,
        k_min: int,
        total_rounds: int,
        num_clients: int = None,
    ) -> int:
        """
        Linear decay of K_sel from k_init → k_min over training.
        The curriculum overrides Bernoulli sampling: we force top-k_sel
        by p_k to guarantee at least k_sel clients participate.
        """
        if total_rounds <= 1:
            k_sel = k_init
        else:
            decay_rate = (k_init - k_min) / (total_rounds - 1)
            k_sel = int(k_init - round_idx * decay_rate)
        k_sel = max(k_min, k_sel)
        if num_clients is not None:
            k_sel = min(k_sel, num_clients)
        return k_sel

    # ─── State Construction ────────────────────────────────────────────────────

    def build_state(
        self,
        reputations: List[float],
        client_losses: List[float],
        model_divergences: List[float],
        gradient_alignments: List[float],
        f1_scores: List[float],
        data_shares: List[float],
        minority_fractions: List[float] = None,
    ) -> np.ndarray:
        """
        Build the full selector state vector (7 features per client).

        Features (attention removed):
          R_k  : FLTrust temporal reputation       ∈ [0, 1]
          l_k  : Evaluation loss (−reward proxy)  ∈ [0, ∞)
          Δ_k  : Model divergence                  ∈ [0, ∞)
          g_k  : Gradient alignment                ∈ [−1, 1]
          f1_k : Historical F1 EMA                 ∈ [0, 1]
          s_k  : Normalized data share             ∈ [0, 1]
          m_k  : Minority class fraction           ∈ [0, 1]

        Returns:
            state: [K * 7] flattened numpy array
        """
        if minority_fractions is None:
            minority_fractions = [0.0] * self.num_clients
        features = []
        for k in range(self.num_clients):
            features.append([
                reputations[k],                        # R_k ∈ [0, 1]
                client_losses[k],                       # l_k ∈ [0, ∞)
                model_divergences[k],                   # Δ_k ∈ [0, ∞)
                gradient_alignments[k],                 # g_k ∈ [−1, 1]
                f1_scores[k],                          # f1_k ∈ [0, 1]
                data_shares[k],                        # s_k ∈ [0, 1]
                minority_fractions[k],                # m_k ∈ [0, 1]
            ])
        return np.array(features, dtype=np.float32).flatten()

    def update_f1_ema(self, client_f1s: List[float]):
        """Update rolling F1 exponential moving averages."""
        for k in range(self.num_clients):
            self.f1_ema[k] = self.f1_alpha * client_f1s[k] + (1 - self.f1_alpha) * self.f1_ema[k]

    # ─── Bernoulli Selection ───────────────────────────────────────────────────

    @torch.no_grad()
    def select_clients(
        self,
        reputations: List[float],
        client_losses: List[float],
        model_divergences: List[float],
        gradient_alignments: List[float],
        data_shares: List[float],
        minority_fractions: List[float] = None,
        k_sel: int = None,
    ) -> Tuple[List[int], np.ndarray]:
        """
        Pure Bernoulli client selection via RL policy.

        NO hybrid score override. The RL policy's sigmoid probabilities
        directly determine selection. Trust/attention are features in the
        state, not gates on the output.

        Args:
            reputations: FLTrust reputations [K]
            client_losses: evaluation losses [K]
            model_divergences: per-client model divergences [K]
            gradient_alignments: cosine(Δ_k, Δ_glob) [K]
            data_shares: normalized data shares [K]
            minority_fractions: minority class fractions [K]
            k_sel: if provided, force top-k_sel by probability (curriculum override)

        Returns:
            selected_indices: list of selected client IDs
            bernoulli_probs: [K] numpy array of p_k values
        """
        state = self.build_state(
            reputations=reputations,
            client_losses=client_losses,
            model_divergences=model_divergences,
            gradient_alignments=gradient_alignments,
            f1_scores=self.f1_ema,
            data_shares=data_shares,
            minority_fractions=minority_fractions,
        )
        self._last_selection_state = state
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Pure RL policy probabilities — NO hybrid score multiplication
        probs = self.actor.get_probs(state_t).squeeze(0)  # [K] sigmoid outputs
        probs_np = probs.cpu().numpy()
        self._last_probs = probs_np

        if k_sel is not None:
            # Curriculum override: ensure at least k_sel clients participate.
            # Sort by probability and take top-k_sel.
            _, top_indices = torch.topk(probs, min(k_sel, self.num_clients))
            selected = top_indices.tolist()
        else:
            # Stochastic Bernoulli sampling
            bernoulli_dist = torch.distributions.Bernoulli(probs)
            samples = bernoulli_dist.sample()  # [K] binary
            selected = torch.where(samples > 0.5)[0].tolist()

            # Fallback: if zero clients selected, pick the highest-probability client
            if not selected:
                selected = [torch.argmax(probs).item()]

        return selected, probs_np

    # ─── Reward Computation ────────────────────────────────────────────────────

    def compute_reward(
        self,
        global_accuracy: float,
        trust_scores: List[float],
        selected_indices: List[int],
    ) -> float:
        """
        Resource-Efficiency Reward Function.

        R_t = ΔAcc_global − λ · (|S_t| / K) − γ · mean_{k∈S_t}(1 − R_k)

        where:
          ΔAcc_global : improvement in global model accuracy vs previous round
          |S_t| / K   : fraction of clients selected (communication cost proxy)
          (1 − R_k)   : how untrusted client k is according to FLTrust

        This cleanly separates concerns:
          - FLTrust decides "how much to trust each client" → appears in R_k
          - RL Selector learns "how few clients can I select and still be accurate"
        """
        eps = 1e-8

        # 1. Accuracy improvement (primary objective)
        delta_global = global_accuracy - self._prev_global_accuracy

        # 2. Communication cost penalty — reward FEWER selections
        num_selected = len(selected_indices)
        comm_penalty = self.lambda_comm * (num_selected / self.num_clients)

        # 3. Untrusted-client penalty — penalize selecting FLTrust-low-reputation clients
        if num_selected > 0:
            selected_trusts = [trust_scores[k] for k in selected_indices]
            mean_untrusted = np.mean([1.0 - t for t in selected_trusts])
        else:
            mean_untrusted = 0.0
        untrusted_penalty = self.gamma_untrusted * mean_untrusted

        reward = delta_global - comm_penalty - untrusted_penalty

        return reward

    # ─── PPO Buffer Storage ───────────────────────────────────────────────────

    def record_selection(
        self,
        selected_indices: List[int],
        global_accuracy: float,
        trust_scores: List[float],
        bernoulli_probs: np.ndarray,
    ):
        """
        Store the (state, action, reward) transition for the selector's PPO buffer.

        Called at END of each FL round. Computes the resource-efficiency reward
        and appends to the rollout buffer for batched PPO updates.
        """
        reward = self.compute_reward(
            global_accuracy=global_accuracy,
            trust_scores=trust_scores,
            selected_indices=selected_indices,
        )

        # Use cached real state (Markov property fix)
        if self._last_selection_state is not None:
            state = self._last_selection_state
        else:
            state = self.build_state(
                reputations=self._prev_trust_scores,
                client_losses=[0.0] * self.num_clients,
                model_divergences=[0.0] * self.num_clients,
                gradient_alignments=[0.0] * self.num_clients,
                f1_scores=self.f1_ema,
                data_shares=[1.0 / self.num_clients] * self.num_clients,
            )

        # Binary action vector [K]
        action_vec = np.zeros(self.num_clients, dtype=np.float32)
        for k in selected_indices:
            action_vec[k] = 1.0

        # Bernoulli log_prob: sum over clients of [a_k·log p_k + (1−a_k)·log(1−p_k)]
        p = np.clip(bernoulli_probs, 1e-6, 1 - 1e-6)
        log_prob = float(np.sum(
            action_vec * np.log(p) + (1 - action_vec) * np.log(1 - p)
        ))

        self.buffer.add(state, action_vec, log_prob, reward, done=False)

        # Update tracking
        self._prev_global_accuracy = global_accuracy
        self._prev_trust_scores = trust_scores[:]
        self._selection_history.extend(selected_indices)
        self._total_rounds_seen += 1
        for k in selected_indices:
            self._selection_counts[k] += 1

    def update_f1_from_round(
        self,
        selected_indices: List[int],
        client_f1s: List[float],
    ):
        """Update F1 EMAs at END of round for next round's state."""
        self.update_f1_ema(client_f1s)

    # ─── PPO Update ───────────────────────────────────────────────────────────

    def update(self, round_idx: int = 0) -> Dict[str, float]:
        """
        Run PPO update on the selector's rollout buffer.

        Uses GAE(λ) for advantage estimation with clipped surrogate objective.
        Entropy coefficient decays over rounds to transition from exploration
        to exploitation.
        """
        if len(self.buffer) < 2:
            self.buffer.clear()
            return {}

        current_entropy_coef = self.entropy_coef_at_round(round_idx)

        # Precompute all state values for GAE bootstrap
        with torch.no_grad():
            all_values: List[float] = []
            for s in self.buffer.states:
                v = self.critic(
                    torch.FloatTensor(s).unsqueeze(0).to(self.device)
                ).item()
                all_values.append(v)

            last_state = torch.FloatTensor(self.buffer.states[-1]).unsqueeze(0).to(self.device)
            last_value = self.critic(last_state).item()

        # GAE computation
        rewards = np.array(self.buffer.rewards, dtype=np.float32)
        dones = np.array(self.buffer.dones, dtype=np.float32)

        advantages = np.zeros(len(rewards), dtype=np.float32)
        returns = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0
        next_value = last_value
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * (1.0 - dones[t]) - all_values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = gae + all_values[t]
            next_value = all_values[t]

        states_t, actions_t, old_log_probs_t = self.buffer.to_tensors(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        if len(advantages_t) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        dataset_size = len(states_t)

        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, self.cfg.mini_batch_size):
                end = min(start + self.cfg.mini_batch_size, dataset_size)
                idx = indices[start:end]

                mb_states = states_t[idx]
                mb_actions = actions_t[idx]      # [MB, K] binary
                mb_old_lp = old_log_probs_t[idx]  # [MB]
                mb_adv = advantages_t[idx]
                mb_ret = returns_t[idx]

                # Bernoulli log_prob from current policy
                probs = self.actor.get_probs(mb_states)  # [MB, K]
                p_clipped = probs.clamp(min=1e-6, max=1 - 1e-6)
                new_log_probs = (
                    mb_actions * torch.log(p_clipped)
                    + (1 - mb_actions) * torch.log(1 - p_clipped)
                ).sum(dim=-1)  # [MB]

                # PPO clipped surrogate
                ratio = torch.exp(new_log_probs - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus with decay schedule
                ent = entropy(probs).mean()
                entropy_loss = -ent  # maximise entropy

                # Critic: MSE
                values = self.critic(mb_states)
                critic_loss = nn.MSELoss()(values, mb_ret)

                loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    + current_entropy_coef * entropy_loss
                )

                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optim.step()
                self.critic_optim.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += ent.item()

        self.buffer.clear()

        num_updates = max(1, self.ppo_epochs * (dataset_size // self.cfg.mini_batch_size))
        return {
            "selector_actor_loss": total_actor_loss / num_updates,
            "selector_critic_loss": total_critic_loss / num_updates,
            "selector_entropy": total_entropy / num_updates,
            "selector_entropy_coef": current_entropy_coef,
        }

    # ─── Model State ──────────────────────────────────────────────────────────

    def get_state(self) -> OrderedDict:
        """Return selector model state for checkpointing."""
        state = OrderedDict()
        for k, v in self.actor.state_dict().items():
            state[f"selector_actor.{k}"] = v.clone()
        for k, v in self.critic.state_dict().items():
            state[f"selector_critic.{k}"] = v.clone()
        return state

    def set_state(self, state: OrderedDict):
        """Restore selector model state from checkpoint."""
        actor_state = OrderedDict()
        critic_state = OrderedDict()
        for k, v in state.items():
            if k.startswith("selector_actor."):
                actor_state[k[15:]] = v
            elif k.startswith("selector_critic."):
                critic_state[k[16:]] = v
        if actor_state:
            self.actor.load_state_dict(actor_state)
        if critic_state:
            self.critic.load_state_dict(critic_state)

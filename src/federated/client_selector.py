"""
RL-Based Client Selection Module for Federated RL-IDS (Publication-Ready Version).

Implements a PPO-based Tier-3 client selection agent with theoretical improvements
over the RL-AUDPS module of RL-UDHFL, purpose-built for the FedRL-IDS pipeline.

Seven key improvements over the original AUDPS-inspired design:

  [1] ACTION SPACE — Bernoulli (independent binary) per-client selection
      Replaces non-differentiable Top-K softmax. Each client k has independent
      probability p_k; action a_k ∈ {0,1}. PPO-compatible because log_prob
      decomposes as Σ Bernoulli.log_prob(a_k | p_k).

  [2] STATE SPACE — Adds gradient alignment feature
      g_k = cosine_similarity(Δ_k, Δ_global). Critical for detecting malicious
      or low-quality clients whose gradients diverge from the global direction.

  [3] REWARD — Meta-Agent-Informed
      Uses meta_agent_accuracy_gain, not just raw global accuracy. The meta-agent
      is the Tier-2 coordinator — its improvement signals genuine decision quality.

  [4] EXPLORATION — Entropy Regularization with Decay Schedule
      Entropy bonus decays linearly from initial_entropy → min_entropy over
      total_rounds rounds. High exploration early; focused exploitation late.

  [5] SELECTION BIAS — Entropy + KL-divergence vs uniform
      Replaces variance-based penalty. H(π) penalises low diversity in the
      selection distribution; D_KL(π||uniform) penalises deviation from uniform.

  [6] HYBRID SELECTION SCORE — Learned × Trust × Attention
      Final Score_k = π_select(k|s) × Trust_k × Attention_k.
      Combines the RL policy's learned value with the system's established
      trust-and-attention signals for robustness.

  [7] CURRICULUM SELECTION — K_sel decays over rounds
      K_sel(t) = max(K_min, K_init − floor(t × rate)).
      Starts with more clients for stable aggregation; converges to fewer,
      high-quality clients as the policy matures.

Mathematical Formulation
─────────────────────────────────────────────────────────────────────────────

STATE (per client k, 7 features):
  f_k = [ R_k, a_k, l_k, Δ_k, g_k, f1_k, s_k ]

  R_k      : FLTrust temporal reputation         ∈ [0, 1]
  a_k      : Normalized attention weight         ∈ [0, 1], Σa = 1
  l_k      : Evaluation loss (−reward proxy)     ∈ [0, ∞)
  Δ_k      : Model divergence ‖w_k−w_glob‖/‖w_glob‖  ∈ [0, ∞)
  g_k      : Gradient alignment cos(Δ_k, Δ_glob)  ∈ [−1, 1]
  f1_k     : Historical F1 EMA                   ∈ [0, 1]
  s_k      : Normalized data share n_k/Σn        ∈ [0, 1], Σs = 1

  Full state: s_t = [f_0, f_1, …, f_{K-1}]  →  dim = 7K

ACTION (Bernoulli per client):
  π_θ(a_k=1 | s_t)_k = p_k = sigmoid(logit_k)    ∈ (0, 1)
  π_θ(a | s_t) = Π_k Bernoulli(a_k | p_k)
  log_prob = Σ_k [a_k · log p_k + (1−a_k) · log(1−p_k)]

SELECTION SCORE (hybrid):
  Score_k = π_θ(k|s_t) · Trust_k · Attention_k
  Selected = {k | Score_k ranks in top-K_sel(t)}

REWARD:
  R_t = ΔAcc_t
         − λ · K_sel(t) / K
         − μ · |H(selected_trust) − H_target|
         − ν · mean(|Δ_k|)
         − ρ · [β · H(π) + (1−β) · D_KL(π || uniform)]

  where:
    ΔAcc_t        = α₁·Δglobal_acc + α₂·Δmeta_acc   (meta-agent-weighted gain)
    H(π)          = −Σ_k p_k · log(p_k + ε)        (entropy of selection dist)
    D_KL(π‖unif)  = Σ_k p_k · log(K · p_k)          (KL vs uniform)
    H_target       = log(K_sel)                       (target entropy for selected set)

References:
  RL-UDHFL: Mohammadpour et al., IEEE IoT Journal 2026.
  FedRL-IDS: Original FedRL-IDS system with PPO + FLTrust + Fed+ + Dynamic Attention.
  PPO: Schulman et al., "Proximal Policy Optimization Algorithms", arXiv 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

from src.config import PPOConfig


# ─── Helpers ───────────────────────────────────────────────────────────────────

def flatten_state_dict(state: OrderedDict) -> torch.Tensor:
    """Flatten an OrderedDict of tensors into a single 1D tensor."""
    return torch.cat([v.flatten().float() for v in state.values()])


def compute_model_divergence(local_model: OrderedDict, global_model: OrderedDict) -> float:
    """Compute normalised L2 divergence: ||w_local − w_glob|| / ||w_glob||."""
    local_flat = flatten_state_dict(local_model)
    global_flat = flatten_state_dict(global_model)
    global_norm = torch.norm(global_flat)
    if global_norm < 1e-12:
        return 0.0
    return (torch.norm(local_flat - global_flat) / global_norm).item()


def compute_gradient_alignment(
    local_updates: List[OrderedDict],
    global_update: OrderedDict,
    device: torch.device,
) -> List[float]:
    """
    Compute cosine similarity between each client's update Δ_k and the
    average global update direction Δ_glob.

    g_k = cos(Δ_k, Δ_glob) = ⟨Δ_k, Δ_glob⟩ / (‖Δ_k‖ · ‖Δ_glob‖)

    This measures whether client k's gradient direction aligns with the
    global gradient direction — low alignment indicates a divergent or
    potentially malicious local update.

    Args:
        local_updates: list of K OrderedDict client model deltas
        global_update: OrderedDict server model delta (from root dataset)
        device: torch device

    Returns:
        alignments: list of K cosine similarities ∈ [−1, 1]
    """
    glob_flat = flatten_state_dict(global_update)
    glob_norm = torch.norm(glob_flat)
    if glob_norm < 1e-12:
        return [0.0] * len(local_updates)

    alignments = []
    for lu in local_updates:
        local_flat = flatten_state_dict(lu)
        local_norm = torch.norm(local_flat)
        if local_norm < 1e-12:
            alignments.append(0.0)
            continue
        cos_sim = torch.dot(local_flat, glob_flat) / (local_norm * glob_norm)
        alignments.append(cos_sim.item())
    return alignments


def entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Shannon entropy H(P) = −Σ p · log(p)."""
    p = probs.clamp(min=eps, max=1 - eps)
    return -(p * torch.log(p)).sum(dim=-1)


def kl_divergence_uniform(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    KL divergence D_KL(P ‖ U) where U is the uniform distribution.
    D_KL(P ‖ U) = Σ_k p_k · log(K · p_k)
    """
    K = probs.shape[-1]
    p = probs.clamp(min=eps, max=1 - eps)
    return (p * torch.log(K * p)).sum(dim=-1)


# ─── Rollout Buffer (Bernoulli / K-length binary action) ───────────────────────

class SelectorRolloutBuffer:
    """
    Stores (state, binary-action-vector, log_prob, reward, done) for selector PPO.

    Action is a K-dimensional binary vector [a_0, …, a_{K-1}], a_k ∈ {0,1}.
    log_prob is the sum of per-client Bernoulli log-probs.
    """

    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []   # binary vectors [K]
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
        actions = torch.FloatTensor(np.array(self.actions)).to(device)  # [T, K] float for BCE
        log_probs = torch.FloatTensor(np.array(self.log_probs)).to(device)
        return states, actions, log_probs


# ─── Selector Policy Network ───────────────────────────────────────────────────

class SelectorActor(nn.Module):
    """
    Actor network for the client selector (Bernoulli policy).

    Maps full state vector [R_0, a_0, l_0, Δ_0, g_0, f1_0, s_0, …]
    to K independent sigmoid probabilities (one per client).

    Output: K logits (one per client) passed through sigmoid to get p_k ∈ (0,1).
    During training, we use the straight-through estimator: we keep logits for
    the PPO objective but use sigmoid probabilities for the Bernoulli distribution.
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
        # K independent logits (one per client Bernoulli probability)
        self.logits_head = nn.Linear(hidden_dim, num_clients)

        # Orthogonal init
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.logits_head.weight, gain=0.01)
        nn.init.zeros_(self.logits_head.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Returns raw logits [batch, num_clients].
        Caller applies sigmoid to get probabilities p_k.
        """
        h = self.net(state)
        return h, self.logits_head(h)

    def get_logits(self, state: torch.Tensor) -> torch.Tensor:
        """Return raw logits for PPO ratio computation."""
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
    Publication-ready PPO-based RL client selector for FedRL-IDS.

    Key design decisions (all 7 improvements applied):

    1. ACTION SPACE — Bernoulli per-client (not Top-K softmax)
       Each client k has independent p_k = sigmoid(logit_k).
       log_prob = Σ_k [a_k·log p_k + (1−a_k)·log(1−p_k)].
       PPO objective uses the full K-dimensional action vector.

    2. STATE SPACE — 7 features per client including gradient alignment g_k.
       g_k = cos(Δ_k, Δ_glob) is the primary Byzantine/malicious-client signal.

    3. REWARD — Meta-agent informed, not just global accuracy.
       ΔAcc = α₁·Δglobal_acc + α₂·Δmeta_acc, α₂ > 0 rewards genuine
       decision refinement from Tier-2 coordination.

    4. EXPLORATION — Entropy decay schedule.
       entropy_coef(t) = max(entropy_min, entropy_init − t/T × (entropy_init − entropy_min))

    5. SELECTION BIAS — Entropy H(π) + KL vs uniform.
       Penalises both low diversity (H → 0) and non-uniform deviation.

    6. HYBRID SCORE — RL × Trust × Attention.
       Score_k = π_select(k|s) · Trust_k · Attention_k.
       Robust to RL policy errors by gating with established trust signals.

    7. CURRICULUM — K_sel(t) = max(K_min, K_init − ⌊t·decay⌋).
       Starts inclusive, converges to selective.
    """

    # Reward coefficient defaults
    DEFAULT_LAMBDA = 0.02    # communication cost weight
    DEFAULT_MU = 0.10        # trust entropy penalty weight
    DEFAULT_NU = 0.05        # divergence penalty weight
    DEFAULT_RHO = 0.05       # selection bias penalty weight
    DEFAULT_ALPHA_META = 0.4 # meta-agent weight in accuracy gain
    DEFAULT_BETA_KL = 0.5    # KL component of bias penalty

    def __init__(
        self,
        num_clients: int,
        state_dim_per_client: int = 8,   # 7 base features + minority_class_fraction
        hidden_dim: int = 128,
        cfg: PPOConfig = None,
        device: torch.device = None,
        total_rounds: int = 100,
    ):
        """
        Args:
            num_clients: total number of FL clients K
            state_dim_per_client: number of features per client (8: 7 base + minority_fraction)
            hidden_dim: hidden layer size for actor/critic
            cfg: PPOConfig for selector training hyperparameters
            device: torch device
            total_rounds: total training rounds (for entropy decay schedule)
        """
        self.num_clients = num_clients
        self.state_dim_per_client = state_dim_per_client
        self.total_state_dim = num_clients * state_dim_per_client  # 8K (was 7K, +minority_class_fraction)
        self.device = device or torch.device("cpu")
        self.cfg = cfg or self._default_ppo_config()
        self.hidden_dim = hidden_dim
        self.total_rounds = total_rounds

        # Selector PPO networks
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
        self._prev_meta_accuracy: float = 0.0
        self._prev_trust_scores: List[float] = [1.0 / num_clients] * num_clients
        self._prev_selected_mask: List[bool] = [False] * num_clients

        # Reward coefficients (can be overridden via cfg or constructor)
        self.lambda_comm = self.DEFAULT_LAMBDA
        self.mu_trust = self.DEFAULT_MU
        self.nu_div = self.DEFAULT_NU
        self.rho_bias = self.DEFAULT_RHO
        self.alpha_meta = self.DEFAULT_ALPHA_META
        self.beta_kl = self.DEFAULT_BETA_KL

        # Training hyperparameters
        self.gamma = self.cfg.gamma
        self.gae_lambda = self.cfg.gae_lambda
        self.clip_epsilon = self.cfg.clip_epsilon
        self.ppo_epochs = self.cfg.ppo_epochs
        self.entropy_coef_init = self.cfg.entropy_coef
        self.entropy_coef_min = 0.02  # Bug 7 fix: was 0.002 — for short runs (5-10 rounds),
        # 0.002 decays to 0 almost immediately, causing entropy collapse (all clients
        # get 0 or 1 selected) in early rounds. 0.02 maintains diversity for longer,
        # and the HHI/over-selection penalties prevent a single client from dominating.
        self.value_coef = self.cfg.value_coef
        self.max_grad_norm = self.cfg.max_grad_norm

        # Entropy tracking for logging
        self._current_entropy: float = 0.0
        self._current_kl_uniform: float = 0.0

        # Fix 3: Selection HHI tracking for diversity penalty
        self._selection_history: List[int] = []  # flat list of all selected client IDs across rounds

        # Task 4: Over-selection tracking — per-client selection count for over-selection penalty
        self._selection_counts: np.ndarray = np.zeros(num_clients, dtype=np.int32)
        self._total_rounds_seen: int = 0

        # Fix: Cache the actual state built in select_clients() so that record_selection()
        # can use the REAL state (not placeholder zeros) when storing (state, action, reward)
        # into the PPO buffer. Without this, the Markov property is violated because
        # the stored state does not correspond to the state at which the action was taken.
        self._last_selection_state: Optional[np.ndarray] = None

    @staticmethod
    def _default_ppo_config() -> PPOConfig:
        from src.config import PPOConfig
        return PPOConfig(
            lr_actor=3e-4,
            lr_critic=1e-3,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            entropy_coef=0.05,     # higher initial for strong early exploration
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

        coef(t) = max(entropy_min, entropy_init − (t/T) × (entropy_init − entropy_min))

        This prevents the policy from collapsing to a deterministic selection
        too early, while still allowing convergence in late rounds.
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

        Starts more inclusive (stable early aggregation), converges to
        selective (fewer, higher-quality clients) as the policy matures.
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

    # ─── State Construction ──────────────────────────────────────────────────

    def build_state(
        self,
        reputations: List[float],
        attention_weights: List[float],
        client_losses: List[float],
        model_divergences: List[float],
        gradient_alignments: List[float],
        f1_scores: List[float],
        data_shares: List[float],
        minority_fractions: List[float] = None,
    ) -> np.ndarray:
        """
        Build the full selector state vector (8 features per client).

        Task 3 Option A: Added minority_class_fraction as 8th feature.
        This biases selection toward clients with minority class data, ensuring
        rare attack patterns are represented in every round's aggregation.

        Args:
            reputations: FLTrust reputation per client [K]
            attention_weights: normalized attention weight per client [K]
            client_losses: evaluation loss per client [K]
            model_divergences: ||w_k - w_glob|| / ||w_glob|| per client [K]
            gradient_alignments: cosine(Δ_k, Δ_glob) per client [K]
            f1_scores: historical F1 EMA per client [K]
            data_shares: n_k / Σn per client [K]
            minority_fractions: fraction of client's data that is minority class [K]

        Returns:
            state: [K * 8] flattened numpy array
        """
        if minority_fractions is None:
            minority_fractions = [0.0] * self.num_clients
        features = []
        for k in range(self.num_clients):
            features.append([
                reputations[k],                        # R_k ∈ [0, 1]
                attention_weights[k],                   # a_k ∈ [0, 1], Σa = 1
                client_losses[k],                       # l_k ∈ [0, ∞)
                model_divergences[k],                   # Δ_k ∈ [0, ∞)
                gradient_alignments[k],                 # g_k ∈ [−1, 1]
                f1_scores[k],                         # f1_k ∈ [0, 1]
                data_shares[k],                        # s_k ∈ [0, 1], Σs = 1
                minority_fractions[k],                # m_k ∈ [0, 1]  ← Task 3 Option A
            ])

        state = np.array(features, dtype=np.float32).flatten()
        return state

    def update_f1_ema(self, client_f1s: List[float]):
        """Update rolling F1 exponential moving averages."""
        for k in range(self.num_clients):
            self.f1_ema[k] = self.f1_alpha * client_f1s[k] + (1 - self.f1_alpha) * self.f1_ema[k]

    # ─── Bernoulli Selection ─────────────────────────────────────────────────

    @torch.no_grad()
    def select_clients(
        self,
        reputations: List[float],
        attention_weights: List[float],
        client_losses: List[float],
        model_divergences: List[float],
        gradient_alignments: List[float],
        data_shares: List[float],
        deterministic: bool = False,
        k_sel: int = None,
        minority_fractions: List[float] = None,
    ) -> List[int]:
        """
        Bernoulli-based client selection with hybrid scoring.

        IMPROVEMENT [1]: Uses independent Bernoulli draws per client, not Top-K softmax.
        Each client k has probability p_k = sigmoid(logit_k) from the actor network.
        Final score = p_k × Trust_k × Attention_k; top-k_sel by score are selected.

        Why Bernoulli + hybrid score is better than pure RL Top-K for PPO:
          (a) log_prob decomposes: Σ_k log P(a_k|s) — exact, no approximation needed
          (b) Hybrid gating with Trust × Attention prevents the RL policy from
              selecting clients that FLTrust would penalise anyway
          (c) Continuous relaxation: p_k ∈ (0,1) is differentiable even though
              the discrete selection is not; PPO handles the discrete via sampling

        IMPROVEMENT [6]: Hybrid Score_k = p_k × Trust_k × Attention_k.
        This combines the RL policy's learned value with the system's established
        trust-and-attention signals for robustness — a wrong RL selection is
        penalised by low Trust or Attention.

        Task 3 Option A: minority_fractions passed to build_state as 8th feature,
        biasing selection toward clients with minority class data.

        Args:
            reputations: FLTrust reputations [K]
            attention_weights: attention weights [K]
            client_losses: evaluation losses [K]
            model_divergences: per-client model divergences [K]
            gradient_alignments: cosine(Δ_k, Δ_glob) per client [K]
            data_shares: normalized data shares [K]
            deterministic: if True, select top-K by hybrid score (no randomness)
            k_sel: number of clients to select (from curriculum or cfg)
            minority_fractions: fraction of client's data that is minority class [K]

        Returns:
            selected_indices: list of selected client IDs
        """
        state = self.build_state(
            reputations, attention_weights, client_losses,
            model_divergences, gradient_alignments, self.f1_ema, data_shares,
            minority_fractions=minority_fractions,
        )
        # Cache the REAL state so record_selection() can use it later in the same round.
        # This ensures the (state, action, reward) tuple stored in the buffer is consistent.
        self._last_selection_state = state
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Bernoulli probabilities from actor
        probs = self.actor.get_probs(state_t).squeeze(0)  # [K], sigmoid outputs

        # Hybrid score: RL_prob × Trust × Attention
        trust_t = torch.FloatTensor(reputations).to(self.device)
        hybrid_scores = probs * trust_t * torch.FloatTensor(attention_weights).to(self.device)  # [K]

        if deterministic or k_sel is not None:
            k = k_sel if k_sel is not None else self.num_clients
            _, top_indices = torch.topk(hybrid_scores, k)
            selected = top_indices.tolist()
        else:
            # Stochastic: sample Bernoulli actions
            bernoulli_dist = torch.distributions.Bernoulli(probs)
            samples = bernoulli_dist.sample()  # [K] binary
            selected = torch.where(samples > 0.5)[0].tolist()
            # Fallback: if no clients selected, pick top-1
            if not selected:
                selected = [torch.argmax(probs).item()]

        return selected

    def get_selection_distribution(
        self,
        reputations: List[float],
        attention_weights: List[float],
        client_losses: List[float],
        model_divergences: List[float],
        gradient_alignments: List[float],
        data_shares: List[float],
        minority_fractions: List[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the full Bernoulli probabilities and hybrid scores for all clients.
        Used for logging, reward computation, and the hybrid score analysis.
        """
        state = self.build_state(
            reputations, attention_weights, client_losses,
            model_divergences, gradient_alignments, self.f1_ema, data_shares,
            minority_fractions=minority_fractions,
        )
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        probs = self.actor.get_probs(state_t).squeeze(0).detach().cpu().numpy()   # [K] sigmoid p
        trust_t = np.array(reputations)
        hybrid = probs * trust_t * np.array(attention_weights)  # [K] hybrid score

        return probs, hybrid, trust_t

    # ─── Reward Computation ─────────────────────────────────────────────────

    def compute_reward(
        self,
        global_accuracy: float,
        meta_accuracy: float,
        trust_scores: List[float],
        prev_selected_mask: List[bool],
        client_divergences: List[float],
        current_probs: np.ndarray,
    ) -> float:
        """
        Compute the selector's reward for the current round.

        IMPROVEMENT [3]: Uses meta_agent_accuracy_gain alongside global accuracy.
        The meta-agent (Tier-2) refines decisions across all clients; its
        improvement signals genuine decision quality beyond raw model accuracy.

        IMPROVEMENT [5]: Replaces variance-based bias penalty with entropy H(π)
        and KL-divergence D_KL(π || uniform).

        Full formula:
          R_t = α₁·Δglobal_acc + α₂·Δmeta_acc
                − λ · (K_sel / K)
                − μ · |H(selected_trust) − H_target|
                − ν · mean(|Δ_k|)
                − ρ · [β · H(π) + (1−β) · D_KL(π || uniform)]

        where:
          Δglobal_acc = global_accuracy − prev_global_accuracy
          Δmeta_acc   = meta_accuracy   − prev_meta_accuracy
          H(π)        = −Σ_k p_k · log(p_k + ε)   (entropy of selection dist)
          D_KL(π‖U)   = Σ_k p_k · log(K · p_k)    (KL vs uniform)
          H_target    = log(K_sel)                 (target entropy for selected set)

        Args:
            global_accuracy: current round global model accuracy
            meta_accuracy: current round meta-agent accuracy
            trust_scores: FLTrust trust scores for all clients [K]
            prev_selected_mask: which clients were selected last round [K]
            client_divergences: ||w_k − w_glob|| / ||w_glob|| for all clients [K]
            current_probs: current Bernoulli probabilities [K]

        Returns:
            reward: scalar float
        """
        eps = 1e-8

        # 1. Accuracy gains (primary objectives)
        delta_global = global_accuracy - self._prev_global_accuracy
        delta_meta = meta_accuracy - self._prev_meta_accuracy
        alpha_global = 1.0 - self.alpha_meta
        acc_gain = alpha_global * delta_global + self.alpha_meta * delta_meta

        # 2. Communication cost penalty — use CURRENT round's selected count
        # prev_selected_mask holds this round's actual selection (built from selected_indices
        # before record_selection stores it in the buffer). This is the correct signal
        # because the reward should reflect the cost of this round's choice.
        num_selected = sum(prev_selected_mask)
        comm_penalty = self.lambda_comm * (num_selected / self.num_clients)

        # 3. Trust entropy penalty
        selected_trust = np.array([
            ts for ts, sel in zip(trust_scores, prev_selected_mask) if sel
        ])
        if len(selected_trust) >= 2:
            # Normalise to distribution for entropy
            p_trust = selected_trust / (selected_trust.sum() + eps)
            h_trust = -np.sum(p_trust * np.log(p_trust + eps))
            k_sel_actual = len(selected_trust)
            h_target = np.log(k_sel_actual + eps)
            trust_entropy_penalty = self.mu_trust * abs(h_trust - h_target)
        else:
            trust_entropy_penalty = 0.0

        # 4. Divergence penalty
        if len(prev_selected_mask) > 0:
            div_penalty = self.nu_div * np.mean(client_divergences)
        else:
            div_penalty = 0.0

        # 5. Selection bias: entropy bonus + KL penalty vs uniform (IMPROVEMENT [5])
        # Higher H(π) means more diverse selections → BONUS.
        # Higher KL(π‖U) means deviation from uniform → PENALTY.
        probs = current_probs.clip(min=eps, max=1 - eps)
        h_policy = -np.sum(probs * np.log(probs))
        h_max = np.log(self.num_clients)  # maximum possible entropy
        kl_uniform = np.sum(probs * np.log(self.num_clients * probs))
        # Normalise both to [0, 1] before combining
        h_norm = h_policy / (h_max + eps)
        kl_norm = kl_uniform / (h_max + eps)

        diversity_bonus = self.rho_bias * self.beta_kl * h_norm
        kl_penalty = self.rho_bias * (1.0 - self.beta_kl) * kl_norm
        # net adjustment: penalises non-diverse / non-uniform selection
        bias_adjustment = kl_penalty - diversity_bonus

        reward = (
            acc_gain
            - comm_penalty
            - trust_entropy_penalty
            - div_penalty
            - bias_adjustment
            - self._compute_selection_diversity_penalty()
            - self._compute_over_selection_penalty()   # Task 4: over-selection penalty
        )

        # Track for logging
        self._current_entropy = h_norm
        self._current_kl_uniform = kl_norm

        return reward

    # ─── Storage ──────────────────────────────────────────────────────────────

    def store_selection(
        self,
        state: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        done: bool = False,
    ):
        """Store a (state, binary-action-vector, log_prob, reward, done)."""
        self.buffer.add(state, action.astype(np.float32), log_prob, reward, done)

    def update_f1_from_round(
        self,
        selected_indices: List[int],
        client_accuracies: List[float],
        client_f1s: List[float],
    ):
        """
        Update F1 EMAs at END of round for next round's state.
        Uses all clients' F1 scores, not just selected, for complete state.
        """
        self.update_f1_ema(client_f1s)

    def _compute_selection_diversity_penalty(self) -> float:
        """
        Fix 3: Penalize selection HHI — if the same clients are always selected,
        the Herfindahl index approaches 1.0 and this penalty is positive.

        Uses cumulative selection history across all rounds so far.
        Only activates when we have enough history (>= 2 rounds of data).
        """
        if len(self._selection_history) < self.num_clients:
            return 0.0  # not enough data for meaningful HHI

        # Distribution of selections across all rounds
        counts = np.bincount(self._selection_history, minlength=self.num_clients)
        total = len(self._selection_history)
        probs = counts / total

        # HHI = sum(p_i^2); ranges from 1/num_clients (uniform) to 1.0 (single client)
        hhi = float(np.sum(probs ** 2))
        min_hhi = 1.0 / self.num_clients
        max_hhi = 1.0
        normalized_hhi = (hhi - min_hhi) / (max_hhi - min_hhi + 1e-8)

        # Penalty coefficient — stronger than the bias penalty alone
        DIV_PENALTY_COEF = 0.1
        return DIV_PENALTY_COEF * normalized_hhi

    def _compute_over_selection_penalty(self) -> float:
        """
        Task 4: Over-selection penalty — prevents the same high-trust clients
        from being selected in every round, which would starve minority-data
        clients and cause their local models to degrade.

        Penalises clients that have been selected in >= 80% of seen rounds.
        Uses exponential decay: more rounds at max selection rate = higher penalty.
        """
        if self._total_rounds_seen == 0:
            return 0.0

        # Selection rate per client
        rates = self._selection_counts / max(self._total_rounds_seen, 1)

        # Penalise clients with selection rate >= 0.8 (over-selected threshold)
        over_selected = rates >= 0.8
        if not over_selected.any():
            return 0.0

        # Excess above threshold, scaled exponentially
        excess = rates[over_selected] - 0.8
        penalty = float(np.mean(excess)) * 0.15
        return penalty

    def record_selection(
        self,
        selected_indices: List[int],
        global_accuracy: float,
        meta_accuracy: float,
        trust_scores: List[float],
        client_divergences: List[float],
        current_probs: np.ndarray,
    ):
        """
        Call at END of round to:
          1. Build binary action vector for buffer
          2. Compute and store reward
          3. Update previous-round tracking variables
        """
        prev_mask = [k in selected_indices for k in range(self.num_clients)]

        reward = self.compute_reward(
            global_accuracy=global_accuracy,
            meta_accuracy=meta_accuracy,
            trust_scores=trust_scores,
            prev_selected_mask=prev_mask,
            client_divergences=client_divergences,
            current_probs=current_probs,
        )

        # Use the REAL state that was cached when select_clients() was called this round.
        # Previously this rebuilt state with placeholder zeros for most features,
        # violating the Markov property (stored state ≠ state at time of action).
        if self._last_selection_state is not None:
            state = self._last_selection_state
        else:
            # Safe fallback for round 0 / unexpected call order
            state = self.build_state(
                reputations=self._prev_trust_scores,
                attention_weights=[1.0 / self.num_clients] * self.num_clients,
                client_losses=[0.0] * self.num_clients,
                model_divergences=[0.0] * self.num_clients,
                gradient_alignments=[0.0] * self.num_clients,
                f1_scores=self.f1_ema,
                data_shares=[1.0 / self.num_clients] * self.num_clients,
            )

        # Binary action vector
        action_vec = np.zeros(self.num_clients, dtype=np.float32)
        for k in selected_indices:
            action_vec[k] = 1.0

        # Bernoulli log_prob (sum of independent per-client log_probs)
        # We approximate using the sigmoid probabilities stored in current_probs
        p = np.clip(current_probs, 1e-6, 1 - 1e-6)
        log_prob = np.sum(
            action_vec * np.log(p) + (1 - action_vec) * np.log(1 - p)
        )

        self.buffer.add(state, action_vec, float(log_prob), reward, done=False)

        # Update tracking
        self._prev_global_accuracy = global_accuracy
        self._prev_meta_accuracy = meta_accuracy
        self._prev_trust_scores = trust_scores[:]
        # Fix 3: Track selection history for HHI diversity penalty
        self._selection_history.extend(selected_indices)
        # Task 4: Update over-selection counts
        self._total_rounds_seen += 1
        for k in selected_indices:
            self._selection_counts[k] += 1

    # ─── PPO Update ─────────────────────────────────────────────────────────

    def update(self, round_idx: int = 0) -> Dict[str, float]:
        """
        Run PPO update on the selector's rollout buffer.

        IMPROVEMENT [4]: Entropy coefficient decays over rounds.
        Early rounds: high entropy → strong exploration.
        Late rounds: low entropy → focused exploitation.

        The update uses the full K-dimensional binary action vector with
        Bernoulli log_prob = Σ_k [a_k·log p_k + (1−a_k)·log(1−p_k)].
        """
        if len(self.buffer) < 2:
            self.buffer.clear()
            return {}

        current_entropy_coef = self.entropy_coef_at_round(round_idx)

        # Precompute ALL state values once so GAE bootstrapping uses V(s_t)
        # instead of incorrectly re-using the previous step's advantage.
        with torch.no_grad():
            all_values: List[float] = []
            for s in self.buffer.states:
                v = self.critic(
                    torch.FloatTensor(s).unsqueeze(0).to(self.device)
                ).item()
                all_values.append(v)

            last_state = torch.FloatTensor(self.buffer.states[-1]).unsqueeze(0).to(self.device)
            last_value = self.critic(last_state).item()

        # GAE
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
            next_value = all_values[t]  # bootstrap next iteration with V(s_t)

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
                mb_actions = actions_t[idx]      # [MB, K] binary (0/1)
                mb_old_lp = old_log_probs_t[idx]  # [MB] scalar sum per sample
                mb_adv = advantages_t[idx]
                mb_ret = returns_t[idx]

                # Get current Bernoulli probabilities
                probs = self.actor.get_probs(mb_states)  # [MB, K], sigmoid p_k

                # Bernoulli log_prob: Σ_k [a_k·log p_k + (1−a_k)·log(1−p_k)]
                p_clipped = probs.clamp(min=1e-6, max=1 - 1e-6)
                new_log_probs = (
                    mb_actions * torch.log(p_clipped)
                    + (1 - mb_actions) * torch.log(1 - p_clipped)
                ).sum(dim=-1)  # [MB]

                # PPO clipped objective (scalar advantage)
                ratio = torch.exp(new_log_probs - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus (with DECAYED coefficient — IMPROVEMENT [4])
                ent = entropy(probs).mean()
                entropy_loss = -ent  # maximise entropy → minimise −entropy

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

    def get_hybrid_scores(
        self,
        reputations: List[float],
        attention_weights: List[float],
        client_losses: List[float],
        model_divergences: List[float],
        gradient_alignments: List[float],
        data_shares: List[float],
        minority_fractions: List[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (bernoulli_probs, hybrid_scores) for all clients.
        hybrid_score_k = p_k × Trust_k × Attention_k.
        Used for logging and for the hybrid selection decision.
        """
        probs, hybrid, _ = self.get_selection_distribution(
            reputations, attention_weights, client_losses,
            model_divergences, gradient_alignments, data_shares,
            minority_fractions=minority_fractions,
        )
        return probs, hybrid

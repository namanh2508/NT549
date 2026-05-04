"""
FLTrust implementation with Temporal Reputation (RL-UDHFL inspired).

The server maintains a small clean root dataset and computes a
server model update.  Trust scores are based on ReLU-clipped cosine
similarity between local updates and the server update, enhanced with
temporal reputation growth/decay to prevent trust collapse.

References:
    Cao et al., "FLTrust: Byzantine-robust Federated Learning via Trust
        Bootstrapping", NDSS 2021.
    Mohammadpour et al., "RL-UDHFL: RL-Enhanced Utility-Driven
        Hierarchical Federated Learning for IoT", IEEE IoT Journal 2026.
"""

import torch
import numpy as np
from typing import List, Dict
from collections import OrderedDict


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two flat tensors."""
    dot = torch.dot(a, b)
    norm_a = torch.norm(a)
    norm_b = torch.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return (dot / (norm_a * norm_b)).item()


def flatten_state_dict(state: OrderedDict) -> torch.Tensor:
    """Flatten an OrderedDict of tensors into a single 1D tensor."""
    return torch.cat([v.flatten().float() for v in state.values()])


def unflatten_state_dict(
    flat: torch.Tensor, reference: OrderedDict
) -> OrderedDict:
    """Unflatten a 1D tensor back into an OrderedDict matching reference shapes."""
    result = OrderedDict()
    offset = 0
    for k, v in reference.items():
        numel = v.numel()
        result[k] = flat[offset: offset + numel].reshape(v.shape).clone()
        offset += numel
    return result


class FLTrust:
    """
    FLTrust aggregation with trust bootstrapping + temporal reputation.

    Enhanced with RL-UDHFL inspired reputation growth/decay to prevent
    trust collapse on datasets where server and local models diverge.

    Steps per round:
    1. Server trains on root dataset -> server model update g_0
    2. For each client i, compute cosine trust TS_i = max(0, cos(g_i, g_0))
    3. Update temporal reputation: R_i based on normalised cosine contribution
    4. Final trust = R_i * TS_i  (no artificial floor — trust mechanism is
       disabled by setting reputation to 1.0 if you want plain cosine weighting)
    5. Normalise each g_i to have same magnitude as g_0
    6. Weighted average with combined trust scores

    FIX v2: Removed trust_floor clamping. The floor caused all trust scores
    to collapse to 0.1 regardless of client quality, effectively disabling
    the trust mechanism entirely. Now the floor is only applied as a
    minimum multiplier on top of reputation, not as an absolute override.

    FIX v3: Fixed reputation dynamics. Previous config had decay(0.1) > growth(0.05),
    structurally biasing reputation toward zero. Now uses adaptive sigmoid-based
    dynamics where the sign of (cs - threshold) determines direction and
    magnitude is proportional to the cosine score itself.
    """

    # Cosine similarity threshold for positive/negative classification.
    # Clients with cos >= threshold are growing in reputation.
    COSINE_POSITIVE_THRESHOLD = 0.0  # 0 = neutral; positive cosine = agree with server

    def __init__(self, device: torch.device, num_agents: int = 4,
                 trust_floor: float = 0.0,
                 reputation_growth: float = 0.1,
                 reputation_decay: float = 0.05,
                 initial_reputation: float = 0.5):
        """
        Args:
            trust_floor: Minimum trust weight (default 0 = no floor, trust
                         can go arbitrarily low). Set >0 only if you want to
                         prevent total exclusion of a client.
            reputation_growth: Rate at which reputation increases for positive
                               contributions (cosine > threshold).
            reputation_decay: Rate at which reputation decreases for negative
                             contributions (cosine <= threshold).
                             NOTE: growth > decay ensures good clients
                             accumulate reputation faster than bad clients lose it.
            initial_reputation: Starting reputation for new/reset clients.
        """
        self.device = device
        self.trust_floor = trust_floor
        self.reputation_growth = reputation_growth   # FIX: was 0.05, now 0.1
        self.reputation_decay = reputation_decay      # FIX: was 0.1, now 0.05
        # Temporal reputation scores (persist across rounds)
        self.reputations = [initial_reputation] * num_agents

    def update_reputations(self, cosine_scores: List[float]) -> None:
        """
        Update temporal reputations using RL-UDHFL-inspired dynamics.

        For positive contribution (cs > COSINE_POSITIVE_THRESHOLD):
            R_i^{t+1} = R_i^t + gamma_r * (cs - threshold) * (1 - R_i^t)
            The better the client agrees with server (higher cs), the faster
            reputation grows, asymptotically toward 1.0.

        For negative contribution (cs <= threshold):
            R_i^{t+1} = R_i^t - delta_r * (threshold - cs) * R_i^t
            The worse the disagreement (lower cs), the faster reputation decays,
            asymptotically toward 0.0.

        With growth=0.1, decay=0.05 (growth > decay), good clients accumulate
        reputation faster than bad clients lose it — preventing the collapse
        seen in the original implementation.
        """
        for i, cs in enumerate(cosine_scores):
            if i >= len(self.reputations):
                self.reputations.append(0.5)

            delta = cs - self.COSINE_POSITIVE_THRESHOLD  # positive = good, negative = bad

            if delta > 0:
                # Proportional growth: scale by both cosine quality and headroom (1-R)
                # R_i += growth_rate * delta * (1 - R_i)
                self.reputations[i] += self.reputation_growth * delta * (1.0 - self.reputations[i])
            else:
                # Proportional decay: scale by how negative and current R
                # R_i -= decay_rate * |delta| * R_i
                self.reputations[i] -= self.reputation_decay * abs(delta) * self.reputations[i]

            # Clamp to [0, 1]
            self.reputations[i] = max(0.0, min(1.0, self.reputations[i]))

    def compute_trust_scores(
        self,
        server_update: OrderedDict,
        client_updates: List[OrderedDict],
    ) -> List[float]:
        """
        Compute trust scores using FLTrust cosine similarity with
        temporal reputation as a bonus (not a multiplier).

        Trust = cosine_score + reputation_bonus

        where cosine_score is the max(0, cosine) normalized across agents,
        and reputation_bonus = beta * (reputation - 0.5) shifts good clients
        above average and penalizes bad clients below average.

        This prevents trust collapse: even if all cosine similarities are
        small (~0.1), the normalized cosine distribution + reputation bonus
        still differentiates clients based on their relative alignment with
        the server update.

        The reputation dynamics remain unchanged (growth > decay for stability),
        but reputation no longer directly multiplies the trust score.
        """
        g0 = flatten_state_dict(server_update).to(self.device)
        cosine_scores = []
        for cu in client_updates:
            gi = flatten_state_dict(cu).to(self.device)
            cs = cosine_similarity(gi, g0)
            cosine_scores.append(cs)  # keep sign; ReLU is applied via reputation

        # Update temporal reputations
        self.update_reputations(cosine_scores)

        # Step 1: ReLU on cosine to get non-negative alignment scores
        alignment = [max(0.0, cs) for cs in cosine_scores]

        # Step 2: Temperature-scaled softmax over alignment scores (Fix 4 / Bug 8)
        # Temperature τ controls sharpness: low τ = sharper distribution, high τ = uniform
        # FIX: τ=0.1 was too sharp — when all cosine similarities are small (~0.1),
        # exp(0.1/0.1) = exp(1) = 2.7 while exp(0.05/0.1) = exp(0.5) = 1.65,
        # giving only 1.6x differentiation. τ=1.0 (no sharpening) preserves the relative
        # ordering while still applying softmax. The reputation bonus then provides
        # additional differentiation on top of the softmax-distributed cosine scores.
        temperature = 1.0
        align_t = torch.FloatTensor(alignment)
        exp_scores = torch.exp(align_t / temperature)
        cos_weighted = exp_scores / (exp_scores.sum() + 1e-8)
        cos_weighted = cos_weighted.tolist()

        # Step 3: Compute reputation bonus
        # Shift reputation so 0.5 → 0, >0.5 → positive, <0.5 → negative
        beta = 0.2  # reduced from 0.3 — reputation bonus should not dominate cosine distribution
        rep_bonus = [beta * (rep - 0.5) for rep in self.reputations[:len(cos_weighted)]]

        # Step 4: Final raw trust = temperature-scaled cosine + reputation bonus
        raw_scores = []
        for i in range(len(cos_weighted)):
            trust = cos_weighted[i] + rep_bonus[i]
            if self.trust_floor > 0:
                trust = max(trust, self.trust_floor)
            raw_scores.append(trust)

        # Step 5: Normalise trust scores to sum to 1.0
        total_raw = sum(raw_scores)
        if total_raw > 1e-12:
            scores = [s / total_raw for s in raw_scores]
        else:
            n = len(raw_scores)
            scores = [1.0 / n] * n

        # Step 6: Anti-concentration cap — apply AFTER normalisation to truly enforce 50% max
        # Applying before normalisation distorts relative weights; applying after preserves distribution
        max_trust = 0.5  # no single client gets more than 50% of total trust
        excess = sum(max(0.0, s - max_trust) for s in scores)
        if excess > 1e-9:
            for i in range(len(scores)):
                if scores[i] > max_trust:
                    scores[i] = max_trust
            # Redistribute excess proportionally to clients below cap
            below_cap = [i for i, s in enumerate(scores) if s < max_trust]
            if below_cap:
                surplus_per_client = excess / len(below_cap)
                for i in below_cap:
                    scores[i] = min(max_trust, scores[i] + surplus_per_client)
            # Re-normalise to sum exactly to 1.0
            total = sum(scores)
            scores = [s / total for s in scores]

        return scores

    def clip_updates(
        self,
        client_updates: List[OrderedDict],
        max_norm: float = 10.0,
    ) -> List[OrderedDict]:
        """
        Clip each client update to a maximum L2 norm.

        FIX: Replaces normalise_updates which scaled client updates to match
        server magnitude — this destroyed PPO network weights because:
          - V(s) and π(a|s) depend on absolute weight values
          - Multiplying all weights by an arbitrary scale factor corrupts
            the Softmax/Sigmoid outputs, causing action collapse

        Instead, we use simple norm clipping: if ||Δ_k|| > max_norm,
        scale down to max_norm. This bounds the update magnitude without
        distorting the relative structure of Actor/Critic parameters.
        """
        clipped = []
        for cu in client_updates:
            gi = flatten_state_dict(cu).to(self.device)
            gi_norm = torch.norm(gi).item()
            if gi_norm > max_norm and gi_norm > 1e-12:
                scale = max_norm / gi_norm
                gi_clipped = gi * scale
                clipped.append(unflatten_state_dict(gi_clipped, cu))
            else:
                clipped.append(cu)
        return clipped

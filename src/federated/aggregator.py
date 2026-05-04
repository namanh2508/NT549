"""
Combined Federated Aggregator: FedTrust + Fed+ + Dynamic Attention.

This is the core innovation — combining three complementary FL techniques:

1. FedTrust: Trust bootstrapping via server root dataset
   - Assigns trust scores (ReLU-clipped cosine similarity)
   - Normalises update magnitudes
   → Provides Byzantine robustness

2. Fed+: Personalisation via regularisation
   - Maintains per-agent personalisation θ_k
   - Regularised local updates
   → Handles non-IID data

3. Dynamic Attention: Performance-aware weighting
   - Agents with lower accuracy get higher attention
   → Improves fairness across heterogeneous data distributions

Pipeline per round:
    a) Each agent trains locally (PPO) → local model w_k
    b) Compute model updates: Δ_k = w_k − w̃ (global model)
    c) Server computes its own update Δ_0 from root dataset
    d) FLTrust: trust scores + normalisation of Δ_k
    e) Dynamic Attention: attention weights from accuracy + sample count
    f) Combined weights = trust_score * attention_weight
    g) Fed+ aggregation with combined weights
    h) Apply Fed+ personalisation for each agent
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict

from src.config import Config
from src.federated.fed_trust import FLTrust
from src.federated.fed_plus import FedPlus
from src.federated.dynamic_attention import DynamicAttention


class FederatedAggregator:
    """
    Combines FLTrust + Fed+ + Dynamic Attention for federated RL-IDS.
    """

    def __init__(self, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.device = device

        self.fl_trust = FLTrust(
            device,
            num_agents=cfg.training.num_clients,
            trust_floor=cfg.fed_trust.trust_floor,
            reputation_growth=cfg.fed_trust.reputation_growth,
            reputation_decay=cfg.fed_trust.reputation_decay,
            initial_reputation=cfg.fed_trust.initial_reputation,
        )
        self.fed_plus = FedPlus(cfg.fed_plus, device)
        self.dyn_attn = DynamicAttention(cfg.attention)

        self._global_model: Optional[OrderedDict] = None
        self._personalisations: Dict[int, OrderedDict] = {}

    @property
    def global_model(self) -> Optional[OrderedDict]:
        return self._global_model

    def set_global_model(self, model: OrderedDict):
        self._global_model = OrderedDict(
            (k, v.clone()) for k, v in model.items()
        )

    def compute_update(
        self, current_model: OrderedDict, global_model: OrderedDict
    ) -> OrderedDict:
        """Compute model update (delta) = current - global."""
        delta = OrderedDict()
        for key in current_model.keys():
            delta[key] = current_model[key].float() - global_model[key].float()
        return delta

    def aggregate_round(
        self,
        local_models: List[OrderedDict],
        server_model: OrderedDict,
        client_infos: List[Dict],
        selected_indices: List[int],
        minority_fractions: List[float] = None,
        pre_train_models: List[OrderedDict] = None,
    ) -> Tuple[OrderedDict, List[float], List[float]]:
        """
        Full aggregation pipeline for one federated round.

        FIX (Global Start Principle):
        If pre_train_models is provided, client_updates are computed as:
            Δ_k = post_train_model - pre_train_model
        This isolates the pure training contribution and prevents
        personalisation bias (θ_k from previous rounds) from leaking
        into the global model.

        If pre_train_models is None, falls back to:
            Δ_k = local_model - global_model  (legacy behaviour)

        Args:
            local_models: post-training model state dicts from clients
            server_model: server model trained on root dataset
            client_infos: list of dicts with 'num_samples' and 'accuracy'
            selected_indices: indices of selected clients
            minority_fractions: fraction of each client's data that is minority class
            pre_train_models: model states BEFORE local training (for clean delta)

        Returns:
            aggregated_model, trust_scores, attention_weights
        """
        if self._global_model is None:
            self._global_model = OrderedDict(
                (k, v.clone()) for k, v in server_model.items()
            )

        # --- Step 1: Compute model updates (clean deltas) ---
        # FIX: Use pre_train_models if available to compute pure training delta
        # This prevents personalisation θ_k from leaking into the update
        if pre_train_models is not None:
            client_updates = [
                self.compute_update(post, pre)
                for post, pre in zip(local_models, pre_train_models)
            ]
        else:
            client_updates = [
                self.compute_update(lm, self._global_model)
                for lm in local_models
            ]
        server_update = self.compute_update(server_model, self._global_model)

        # --- Step 2: FLTrust — trust scores from deltas ---
        trust_scores = self.fl_trust.compute_trust_scores(
            server_update, client_updates
        )

        # --- Step 3: Clip update magnitudes (replaces dangerous magnitude normalisation) ---
        clipped_updates = self.fl_trust.clip_updates(client_updates, max_norm=10.0)

        # --- Step 4: Reconstruct models (global + clipped delta) ---
        reconstructed_models = []
        for cu in clipped_updates:
            model = OrderedDict()
            for key in self._global_model.keys():
                model[key] = self._global_model[key].float() + cu[key].float()
            reconstructed_models.append(model)

        # --- Step 5: Dynamic Attention weights ---
        attention_values = self.dyn_attn.compute_all_attentions(client_infos)

        # --- Step 5b: Minority class trust boost ---
        if minority_fractions is not None:
            beta_minority = 0.3
            attention_values = [
                att * (1.0 + beta_minority * mf)
                for att, mf in zip(attention_values, minority_fractions)
            ]

        # --- Step 6: Combine trust × attention for aggregation weights ---
        combined_weights = [
            ts * att for ts, att in zip(trust_scores, attention_values)
        ]
        total_weight = sum(combined_weights)

        if total_weight < 1e-12:
            aggregated = OrderedDict(
                (k, v.clone()) for k, v in server_model.items()
            )
        else:
            norm_weights = [w / total_weight for w in combined_weights]
            aggregated = self.fed_plus.aggregate(
                reconstructed_models, weights=norm_weights
            )

        # Update global model
        self._global_model = OrderedDict(
            (k, v.clone()) for k, v in aggregated.items()
        )

        # Expand trust_scores and attention_values to all clients
        full_trust = [0.0] * self.cfg.training.num_clients
        full_attention = [0.0] * self.cfg.training.num_clients
        for idx, ts, att in zip(selected_indices, trust_scores, attention_values):
            full_trust[idx] = ts
            full_attention[idx] = att

        return aggregated, full_trust, full_attention

    def personalise_for_agent(
        self,
        agent_id: int,
        local_model: OrderedDict,
        eta: float = 3e-4,
    ) -> OrderedDict:
        """
        Apply Fed+ personalisation for a specific agent.
        Computes θ_k and returns personalised model.
        """
        if self._global_model is None:
            return local_model

        # Compute personalisation component
        theta = self.fed_plus.compute_personalisation(
            local_model, self._global_model
        )
        self._personalisations[agent_id] = theta

        # Apply Fed+ local update mixing
        personalised = self.fed_plus.local_update_step(
            local_model, self._global_model, theta, eta
        )
        return personalised

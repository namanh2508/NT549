"""
Simplified Federated Aggregator for FedRL-IDS.

After architectural review, the tri-technique pipeline (FLTrust + Fed+ + Dynamic Attention)
was causing two systemic problems:

  [A] Authority overlap with RL Selector: The Selector's reward was contaminated by
      the aggregator's trust/attention weights. Selecting a bad client → FLTrust
      gave it weight≈0 → Global model still improved → Selector got positive reward
      even though it chose WRONG.  False Credit Assignment.

  [B] Complexity compounding: Three aggregation techniques multiplied together
      (Trust × Attention × Fed+) made it impossible to debug which component
      was responsible for any observed behavior.

NEW DESIGN — Clean 3-Step Aggregation (Refactoring Phase 3):

  Step 1: FLTrust — Byzantine robustness
    - Compute cosine similarity between each client's update Δ_k and the
      server's update Δ_0 from the root dataset.
    - Update temporal reputation based on alignment quality.
    - Raw trust = minmax-normalized cosine + reputation bonus.

  Step 2: Normalize — Probabilistic weighting
    - Normalize raw trust scores to sum to 1.0.
    - Apply anti-concentration cap (no client > 50% of total weight).

  Step 3: Weighted Average — FedAvg-style aggregation
    - w_global = Σ_k Trust_k · Δ_k_clip + w_global_prev
    - Simple, interpretable, stable.

  Separation of Concerns:
    FLTrust   → Byzantine robustness (what we trust)
    Selector  → Resource efficiency (how many clients)
    Aggregator → Just computes the weighted average

References:
  Cao et al., "FLTrust: Byzantine-robust Federated Learning via Trust
      Bootstrapping", NDSS 2021.
"""

import torch
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict

from src.config import Config
from src.federated.fed_trust import FLTrust


class FederatedAggregator:
    """
    Clean federated aggregation: FLTrust + Weighted Average only.

    No Dynamic Attention (removed — caused authority overlap with RL Selector).
    No Fed+ personalisation inside the loop (can be applied post-FL if needed).
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

        self._global_model: Optional[OrderedDict] = None

    @property
    def global_model(self) -> Optional[OrderedDict]:
        return self._global_model

    def set_global_model(self, model: OrderedDict):
        self._global_model = OrderedDict(
            (k, v.clone()) for k, v in model.items()
        )

    @staticmethod
    def compute_update(
        current_model: OrderedDict, reference_model: OrderedDict
    ) -> OrderedDict:
        """Compute model delta = current − reference."""
        delta = OrderedDict()
        for key in current_model.keys():
            delta[key] = (
                current_model[key].float() - reference_model[key].float()
            )
        return delta

    def aggregate_round(
        self,
        local_models: List[OrderedDict],
        server_model: OrderedDict,
        selected_indices: List[int],
        pre_train_models: List[OrderedDict] = None,
    ) -> Tuple[OrderedDict, List[float]]:
        """
        Three-step aggregation for one federated round.

        Step 1 — FLTrust:
          Compute trust scores from cosine similarity between client updates
          and the server's update. Update temporal reputations.

        Step 2 — Normalise:
          Normalise trust scores to sum to 1.0 with anti-concentration cap.

        Step 3 — Weighted Average:
          Clip each update to max_norm=10.0.
          w_global = Σ_k Trust_k · (w_global_prev + Δ_k_clip)

        Args:
            local_models: post-training model state dicts from selected clients
            server_model: server model trained on root dataset
            selected_indices: indices of selected clients
            pre_train_models: model states BEFORE local training (for clean deltas)

        Returns:
            aggregated_model, trust_scores (one per selected client)
        """
        if self._global_model is None:
            self._global_model = OrderedDict(
                (k, v.clone()) for k, v in server_model.items()
            )

        # ── Step 1: Compute model updates (clean deltas) ────────────────────────
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

        # ── Step 1 (cont): FLTrust trust scores ────────────────────────────────
        trust_scores = self.fl_trust.compute_trust_scores(
            server_update, client_updates
        )

        # ── Step 2: Clip update magnitudes ───────────────────────────────────────
        clipped_updates = self.fl_trust.clip_updates(client_updates, max_norm=10.0)

        # ── Step 3: Reconstruct models with clipped deltas ─────────────────────
        reconstructed_models = []
        for cu in clipped_updates:
            model = OrderedDict()
            for key in self._global_model.keys():
                model[key] = self._global_model[key].float() + cu[key].float()
            reconstructed_models.append(model)

        # ── Step 3 (cont): Weighted average with trust scores ───────────────────
        total_weight = sum(trust_scores)
        if total_weight < 1e-12:
            aggregated = OrderedDict(
                (k, v.clone()) for k, v in server_model.items()
            )
        else:
            norm_weights = [w / total_weight for w in trust_scores]
            aggregated = self._weighted_average(reconstructed_models, norm_weights)

        # Update global model
        self._global_model = OrderedDict(
            (k, v.clone()) for k, v in aggregated.items()
        )

        return aggregated, trust_scores

    def _weighted_average(
        self,
        models: List[OrderedDict],
        weights: List[float],
    ) -> OrderedDict:
        """Weighted average of model state dicts."""
        result = OrderedDict()
        for key in models[0].keys():
            result[key] = sum(
                w * m[key].float() for w, m in zip(weights, models)
            )
        return result

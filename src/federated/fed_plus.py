"""
Fed+ implementation.

References:
    Kundu et al., "Fed+: A Unified Approach to Federated Personalization
        via Parameterized Regularization", arXiv 2022.

Fed+ unifies federated learning with personalisation via regularisation.
Key idea: each party learns w_k = w̃ + θ_k where θ_k is a personalised
component regulated by Ψ.

We implement FedAvg+ variant (Ψ = (σδ/2)||·||²) with the two-step
local update:
    w_k ← κ[w_k − η∇f_k(w_k)] + (1−κ)[w̃ + θ_k]
where κ = 1/(1 + ηδ).

The personalisation θ_k = [1+δ]⁻¹(w_k − w̃) for FedAvg+.
"""

import torch
from typing import List
from collections import OrderedDict

from src.config import FedPlusConfig


class FedPlus:
    """
    Fed+ federated aggregation with personalisation.
    Supports FedAvg+ (l2 regularisation → mean aggregation + personalisation).
    """

    def __init__(self, cfg: FedPlusConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

    def compute_personalisation(
        self,
        local_model: OrderedDict,
        global_model: OrderedDict,
    ) -> OrderedDict:
        """
        Compute θ_k = [1 + δ]⁻¹ (w_k − w̃) for FedAvg+.
        """
        delta = self.cfg.delta
        theta = OrderedDict()
        for key in local_model.keys():
            diff = local_model[key].float() - global_model[key].float()
            theta[key] = diff / (1.0 + delta)
        return theta

    def local_update_step(
        self,
        local_weights: OrderedDict,
        global_model: OrderedDict,
        theta: OrderedDict,
        eta: float,
    ) -> OrderedDict:
        """
        Apply Fed+ regularised update (post-gradient step mixing):
            w_k ← κ·w_k + (1−κ)·(w̃ + θ_k)
        where κ = 1/(1 + η·σ).

        In practice, the PPO agent already does the gradient step,
        so we apply the mixing afterwards.
        """
        sigma = self.cfg.sigma
        kappa = 1.0 / (1.0 + eta * sigma)

        updated = OrderedDict()
        for key in local_weights.keys():
            w_local = local_weights[key].float()
            w_global = global_model[key].float()
            t_k = theta[key].float()
            updated[key] = kappa * w_local + (1 - kappa) * (w_global + t_k)
        return updated

    def aggregate(
        self,
        local_models: List[OrderedDict],
        weights: List[float] = None,
    ) -> OrderedDict:
        """
        FedAvg+ aggregation: weighted mean of local models.
        For FedAvg+ the aggregation function A is simply the mean.
        """
        if not local_models:
            raise ValueError("No local models to aggregate")

        n = len(local_models)
        if weights is None:
            weights = [1.0 / n] * n
        else:
            total = sum(weights)
            weights = [w / total for w in weights]

        result = OrderedDict()
        for key in local_models[0].keys():
            result[key] = sum(
                w * m[key].float() for w, m in zip(weights, local_models)
            )
        return result

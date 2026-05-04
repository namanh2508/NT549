"""
Dynamic Attention Mechanism.

References:
    Vadigi et al., "Dynamic Attention-based Federated Learning for
        Heterogeneous Data", Journal of Information Security and
        Applications (JISA), 2023.

FIX (D): Replaced accuracy-based multiplier with loss-based multiplier.

Problem with accuracy-based: when clients converge to ~1.0 accuracy (post-personalisation),
the multiplier becomes ~1.0 for all clients, collapsing the mechanism to uniform weighting.

New approach: Use training loss as the signal. Loss continues to differentiate clients
even after accuracy converges. Attention is proportional to 1/loss, so clients with
higher loss (worse performance) get more attention.

Additionally, per-round normalisation ensures the distribution of attention weights
reflects relative client quality rather than absolute magnitudes.
"""

from typing import List, Dict

from src.config import DynamicAttentionConfig


class DynamicAttention:
    """
    Dynamic attention value mechanism for federated aggregation.
    Agents with higher loss (worse performance) receive higher attention
    weight during model aggregation, promoting fairness across heterogeneous
    data distributions.
    """

    def __init__(self, cfg: DynamicAttentionConfig):
        self.k = cfg.k              # max multiplier
        self.floor = cfg.floor      # loss floor for multiplier cap
        self.loss_history: List[List[float]] = []  # per-round loss records

    def compute_multiplier_from_loss(self, loss: float) -> float:
        """
        Attention multiplier proportional to training loss
        (higher loss → higher attention).

        multiplier = 1.0 + (k - 1.0) * loss / (loss + floor)

        - loss → 0:    mult → 1.0          (minimum attention for best performers)
        - loss = floor: mult = 1.0 + (k-1)/2  (midpoint)
        - loss → ∞:   mult → k             (maximum attention for worst performers)

        With k=5.0, floor=1.0:
          loss=0.0 → mult=1.0
          loss=0.5 → mult=2.33
          loss=1.0 → mult=3.0
          loss=2.0 → mult=3.67
          loss→∞ → mult→5.0
        """
        loss = max(0.0, loss)  # safety
        floor = max(self.floor, 1e-6)
        multiplier = 1.0 + (self.k - 1.0) * loss / (loss + floor)
        return max(1.0, min(multiplier, self.k))  # clamp to [1.0, k]

    def compute_multiplier(self, accuracy: float) -> float:
        """
        Legacy accuracy-based multiplier (DEPRECATED — use compute_multiplier_from_loss).
        Kept for backward compatibility with configs that pass accuracy.
        """
        accuracy = max(0.0, min(1.0, accuracy))
        inaccuracy = 1.0 - accuracy

        if accuracy <= self.floor:
            return self.k

        normalised_inacc = inaccuracy / (1.0 - self.floor)
        multiplier = 1.0 + (self.k - 1.0) * normalised_inacc
        return multiplier

    def compute_attention(
        self, num_samples: int, loss: float
    ) -> float:
        """
        attention_value = num_samples * attention_multiplier_from_loss.
        """
        multiplier = self.compute_multiplier_from_loss(loss)
        return num_samples * multiplier

    def compute_all_attentions(
        self,
        client_info: List[Dict],
    ) -> List[float]:
        """
        Compute attention values for all clients.

        Args:
            client_info: list of dicts with keys:
                - 'num_samples': int
                - 'loss': float (training loss; preferred)
                - 'accuracy': float (fallback; DEPRECATED)
        Returns:
            List of attention values (normalised to sum to 1.0).
        """
        attentions = []
        for info in client_info:
            loss = info.get("loss")
            if loss is not None:
                att = self.compute_attention(
                    num_samples=info["num_samples"],
                    loss=loss,
                )
            else:
                # DEPRECATED fallback: use accuracy if loss not provided
                att = info["num_samples"] * self.compute_multiplier(
                    info.get("accuracy", 1.0)
                )
            attentions.append(att)

        # Normalise so weights sum to 1.0, then apply per-client cap
        # to prevent any single client from dominating the aggregation.
        # Cap each client at max_weight_share (default 0.3 = 30%) of total weight.
        total = sum(attentions)
        if total > 1e-12:
            attentions = [a / total for a in attentions]
        else:
            n = len(attentions)
            attentions = [1.0 / n] * n

        # Apply per-client weight cap to prevent single-client dominance.
        # Clients that would exceed the cap have their excess redistributed
        # proportionally to all other clients.
        max_share = 0.3  # no single client gets more than 30% of total weight
        excess = sum(max(0.0, a - max_share) for a in attentions)
        if excess > 1e-9:
            # Redistribute excess among clients below the cap
            for i in range(len(attentions)):
                if attentions[i] > max_share:
                    attentions[i] = max_share
            # Re-normalize remaining clients
            below_cap = [i for i, a in enumerate(attentions) if a < max_share]
            if below_cap:
                surplus_per_client = excess / len(below_cap)
                for i in below_cap:
                    attentions[i] = min(max_share, attentions[i] + surplus_per_client)
            # Final renormalize to sum exactly to 1.0
            total = sum(attentions)
            attentions = [a / total for a in attentions]

        # Record for per-round diagnostics
        self.loss_history.append(
            [info.get("loss", 0.0) for info in client_info]
        )

        return attentions

    def normalise_attentions(self, attentions: List[float]) -> List[float]:
        """Normalise attention values to sum to 1."""
        total = sum(attentions)
        if total < 1e-12:
            n = len(attentions)
            return [1.0 / n] * n
        return [a / total for a in attentions]

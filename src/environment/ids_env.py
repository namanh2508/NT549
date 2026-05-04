"""
IDS Environment for Reinforcement Learning.

The agent observes network traffic features and outputs a continuous
action (detection confidence).  The environment computes rewards using:

    R(t) = α·TP − β·FP − γ·FN + δ·(1 − normalised_latency) + ε·novelty_bonus

Supports per-attack-type decision thresholds and novelty detection via
autoencoder reconstruction error (Chalapathy & Chawla, 2019).
"""

import time
import numpy as np
import torch
from typing import Dict, Optional, Tuple, Set

from src.config import RewardConfig


class IDSEnvironment:
    """
    Gym-like environment wrapping a partition of the IDS dataset.

    State:  feature vector of a network flow  (shape = [state_dim])
    Action: continuous float in [-1, 1] via tanh
            * > 0  → classify as *attack*
            * ≤ 0  → classify as *benign*
            |value| encodes confidence

    Reward: the composite reward function from the paper design.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        reward_cfg: RewardConfig,
        num_classes: int,
        shuffle: bool = True,
        seed: int = 42,
        novelty_detector=None,
        novelty_threshold: float = 0.65,
    ):
        self.X = X
        self.y = y
        self.reward_cfg = reward_cfg
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)

        self.state_dim = X.shape[1]
        self.action_dim = 1  # single continuous confidence score

        # Track novelty: classes seen across ALL episodes (persistent)
        self._seen_classes: Set[int] = set()
        # Autoencoder-based novelty detector (set externally after training)
        self._novelty_detector = novelty_detector
        self._novelty_threshold = novelty_threshold

        # Episode bookkeeping
        self._idx = 0
        self._order: np.ndarray = np.arange(len(X))
        self._step_count = 0
        self._episode_metrics: Dict[str, int] = {}

        # Latency tracking
        self._step_start_time: float = 0.0
        self._max_latency: float = 0.01  # baseline max latency (seconds)

        self.reset()

    # ─── Gym-like API ──────────────────────

    def reset(self) -> np.ndarray:
        """Reset environment for a new episode. _seen_classes persists across episodes."""
        if self.shuffle:
            self.rng.shuffle(self._order)
        self._idx = 0
        self._step_count = 0
        self._episode_metrics = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        self._step_start_time = time.perf_counter()
        return self._get_state()

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step.

        Args:
            action: either:
                - float in [-1, 1] (legacy continuous binary action)
                - int class index (new Categorical policy, action is already the class)
        Returns:
            next_state, reward, done, info
        """
        step_start = time.perf_counter()

        true_label = self.y[self._order[self._idx]]
        is_attack = int(true_label != 0)  # 0 = benign in label encoding

        # Decision — support both legacy float and new int action formats
        if isinstance(action, (int, np.integer)):
            # New Categorical policy: action is already a class index
            predicted_class = int(action)
            predicted_attack = int(predicted_class != 0)
        else:
            # Legacy: continuous action in [-1, 1]; > 0 = attack
            predicted_attack = int(float(action) > 0)
            predicted_class = predicted_attack

        # Confusion matrix update
        if predicted_attack == 1 and is_attack == 1:
            self._episode_metrics["tp"] += 1
            tp, fp, fn = 1, 0, 0
        elif predicted_attack == 1 and is_attack == 0:
            self._episode_metrics["fp"] += 1
            tp, fp, fn = 0, 1, 0
        elif predicted_attack == 0 and is_attack == 1:
            self._episode_metrics["fn"] += 1
            tp, fp, fn = 0, 0, 1
        else:
            self._episode_metrics["tn"] += 1
            tp, fp, fn = 0, 0, 0

        # Latency
        latency = time.perf_counter() - step_start
        self._max_latency = max(self._max_latency, latency + 1e-9)
        norm_latency = min(latency / self._max_latency, 1.0)

        # Novelty bonus: autoencoder reconstruction error OR unseen class
        novelty_bonus = 0.0
        if is_attack == 1 and predicted_attack == 1:
            if true_label not in self._seen_classes:
                # Class never seen in any episode — potentially novel attack
                novelty_bonus = 1.0
                self._seen_classes.add(true_label)
            elif self._novelty_detector is not None:
                # Use autoencoder reconstruction error for anomaly scoring
                state_t = torch.FloatTensor(self.X[self._order[self._idx]]).unsqueeze(0)
                with torch.no_grad():
                    recon_err = self._novelty_detector.reconstruction_error(state_t).item()
                if recon_err > self._novelty_threshold:
                    novelty_bonus = min(recon_err / self._novelty_threshold - 1.0, 1.0)

        # Composite reward
        r = self.reward_cfg
        reward = (
            r.alpha * tp
            - r.beta * fp
            - r.gamma * fn
            + r.delta * (1.0 - norm_latency)
            + r.epsilon * novelty_bonus
        )

        # Advance
        self._idx += 1
        self._step_count += 1
        done = self._idx >= len(self.X)

        next_state = self._get_state() if not done else np.zeros(self.state_dim, dtype=np.float32)

        info = {
            "true_label": true_label,
            "predicted_attack": predicted_attack,
            "is_attack": is_attack,
            "tp": tp, "fp": fp, "fn": fn,
            "novelty": novelty_bonus,
            "latency": latency,
            "episode_metrics": self._episode_metrics.copy(),
        }

        return next_state, reward, done, info

    def _get_state(self) -> np.ndarray:
        idx = self._order[self._idx % len(self._order)]
        return self.X[idx].astype(np.float32)

    def reset_novelty_tracking(self):
        """Reset novelty tracking at the start of each FL round.

        BUG-C fix: _seen_classes persists across ALL episodes/rounds, so after
        FL round 1 all attack classes are "seen" and novelty_bonus=0 forever.
        Calling this at the start of each FL round re-enables novelty detection.
        """
        self._seen_classes.clear()

    @property
    def episode_metrics(self) -> Dict[str, int]:
        return self._episode_metrics

    def get_accuracy(self) -> float:
        m = self._episode_metrics
        total = m["tp"] + m["fp"] + m["fn"] + m["tn"]
        if total == 0:
            return 0.0
        return (m["tp"] + m["tn"]) / total

    def __len__(self):
        return len(self.X)


class MultiClassIDSEnvironment(IDSEnvironment):
    """
    Extended IDS environment that supports per-attack-type decisions.
    The action space has num_classes dimensions — one confidence score per class.
    The agent outputs confidence for each attack category, and we select
    the class with the highest confidence.

    Fix: Class-balanced reward with inverse-frequency weighting, prediction
    entropy bonus, HHI bias penalty, and collapse detection to prevent
    single-class collapse on imbalanced datasets.
    """

    # ── Reward parameters (config-driven with class-level fallbacks) ───────────
    # These were previously hardcoded. They now fall back to reward_cfg fields
    # when available, so MultiClassIDSEnvironment respects the central config.
    #
    # reward_cfg fields used (with class-level fallback defaults):
    #   TP_REWARD    → reward_cfg.tp_reward    (default 3.0)
    #   TN_REWARD    → reward_cfg.tn_reward    (default 5.0)
    #   FP_PENALTY   → reward_cfg.fp_penalty   (default 2.0)
    #   FN_PENALTY   → reward_cfg.fn_penalty   (default 3.0)
    #   BALANCE_COEF → reward_cfg.balance_coef  (default 2.0)
    #   ENTROPY_COEF → reward_cfg.entropy_coef  (default 2.0)
    #   HHI_COEF     → reward_cfg.hhi_coef      (default 2.5)
    #   COLLAPSE_THR → reward_cfg.collapse_thr   (default 0.65)
    #   COLLAPSE_PEN → reward_cfg.collapse_pen  (default 20.0)
    #   MACRO_F1_COEF→ reward_cfg.macro_f1_coef (default 5.0)
    #   CLASS_WEIGHT_CAP → reward_cfg.class_weight_cap (default 3.0)
    #   ADAPTIVE_CLASS_WEIGHT_CAP → reward_cfg.adaptive_cap (default 50.0)
    #   FOCAL_GAMMA  → reward_cfg.focal_gamma   (default 2.0)
    TP_REWARD: float = 3.0
    TN_REWARD: float = 5.0
    FP_PENALTY: float = 2.0
    FN_PENALTY: float = 3.0
    BALANCE_REWARD_COEF: float = 2.0
    ENTROPY_BONUS_COEF: float = 2.0
    HHI_PENALTY_COEF: float = 2.5
    COLLAPSE_THRESHOLD: float = 0.65
    COLLAPSE_PENALTY: float = 20.0
    MACRO_F1_BONUS_COEF: float = 5.0
    CLASS_WEIGHT_CAP: float = 3.0
    ADAPTIVE_CLASS_WEIGHT_CAP: float = 50.0
    FOCAL_GAMMA: float = 2.0

    def __init__(self, X, y, reward_cfg, num_classes, **kwargs):
        super().__init__(X, y, reward_cfg, num_classes, **kwargs)
        self.action_dim = num_classes  # one score per class

        # ── Apply reward_cfg overrides to class-level attributes ─────────────
        # This makes MultiClassIDSEnvironment config-driven while preserving
        # class-default fallbacks for fields not present in reward_cfg.
        self._apply_reward_config(reward_cfg)

        # Track per-class metrics
        self._class_metrics: Dict[int, Dict[str, int]] = {
            c: {"tp": 0, "fp": 0, "fn": 0} for c in range(num_classes)
        }

        # ── Class-imbalance tracking ────────────────────────────────────────
        # Dataset-level class frequencies (for inverse-frequency weighting)
        self._dataset_class_counts = np.bincount(y, minlength=num_classes)
        total_samples = len(y)
        self._dataset_class_freqs = self._dataset_class_counts / total_samples
        # Inverse-frequency weights: rare classes get higher weight
        inv_freqs = 1.0 / (self._dataset_class_freqs + 1e-8)
        # Adaptive cap: sqrt of max imbalance ratio, capped at ADAPTIVE_CLASS_WEIGHT_CAP
        actual_max_ratio = np.max(self._dataset_class_freqs) / (np.min(self._dataset_class_freqs) + 1e-8)
        adaptive_cap = min(self.ADAPTIVE_CLASS_WEIGHT_CAP, max(self.CLASS_WEIGHT_CAP, np.sqrt(actual_max_ratio)))
        self._class_weights = np.minimum(
            inv_freqs / (inv_freqs.sum() / num_classes),
            adaptive_cap,
        )
        self._sum_inv_freqs = inv_freqs.sum()

        # Per-episode predicted class distribution (for HHI bias penalty)
        self._episode_pred_counts: Dict[int, int] = {c: 0 for c in range(num_classes)}
        self._episode_total_preds: int = 0

        # Prediction collapse detection
        self._collapse_countdown: int = 0  # steps until next collapse check
        self._collapse_detected: bool = False

        # Store focal gamma for PPO agent to access
        self._focal_gamma = self.FOCAL_GAMMA

    def _apply_reward_config(self, reward_cfg):
        """
        Override class-level reward constants with fields from reward_cfg.
        Fields not present in reward_cfg retain their class-level defaults.
        """
        self.TP_REWARD              = getattr(reward_cfg, 'tp_reward',        self.TP_REWARD)
        self.TN_REWARD              = getattr(reward_cfg, 'tn_reward',        self.TN_REWARD)
        self.FP_PENALTY             = getattr(reward_cfg, 'fp_penalty',      self.FP_PENALTY)
        self.FN_PENALTY             = getattr(reward_cfg, 'fn_penalty',       self.FN_PENALTY)
        self.BALANCE_REWARD_COEF   = getattr(reward_cfg, 'balance_coef',    self.BALANCE_REWARD_COEF)
        self.ENTROPY_BONUS_COEF     = getattr(reward_cfg, 'entropy_coef',    self.ENTROPY_BONUS_COEF)
        self.HHI_PENALTY_COEF       = getattr(reward_cfg, 'hhi_coef',         self.HHI_PENALTY_COEF)
        self.COLLAPSE_THRESHOLD     = getattr(reward_cfg, 'collapse_thr',     self.COLLAPSE_THRESHOLD)
        self.COLLAPSE_PENALTY      = getattr(reward_cfg, 'collapse_pen',     self.COLLAPSE_PENALTY)
        self.MACRO_F1_BONUS_COEF   = getattr(reward_cfg, 'macro_f1_coef',   self.MACRO_F1_BONUS_COEF)
        self.CLASS_WEIGHT_CAP       = getattr(reward_cfg, 'class_weight_cap', self.CLASS_WEIGHT_CAP)
        self.ADAPTIVE_CLASS_WEIGHT_CAP = getattr(reward_cfg, 'adaptive_cap',  self.ADAPTIVE_CLASS_WEIGHT_CAP)
        self.FOCAL_GAMMA           = getattr(reward_cfg, 'focal_gamma',      self.FOCAL_GAMMA)

    def reset(self) -> np.ndarray:
        state = super().reset()
        self._class_metrics = {
            c: {"tp": 0, "fp": 0, "fn": 0} for c in range(self.num_classes)
        }
        self._episode_pred_counts = {c: 0 for c in range(self.num_classes)}
        self._episode_total_preds = 0
        self._collapse_countdown = 0
        self._collapse_detected = False
        return state

    # ── MCC-based reward computation (simplified from 12+ components) ───────

    def _compute_class_balanced_reward(
        self,
        tp: int, fp: int, fn: int,
        true_label: int,
        predicted_class: int,
        norm_latency: float,
        novelty_bonus: float,
        class_bonus: float,
        tn: int = 0,
    ) -> float:
        """
        Simplified reward using MCC (Matthews Correlation Coefficient).

        FIX: The previous 12+ component reward caused "Reward Design Smells"
        where competing terms (entropy bonus vs HHI penalty, balanced accuracy
        vs macro F1) cancel each other out, confusing PPO optimisation.

        New design principles:
        1. MCC as primary signal — naturally handles class imbalance by
           incorporating all 4 quadrants of the confusion matrix.
        2. Cost-sensitive weighting — FN penalty > FP penalty because
           missing an attack is more dangerous than a false alarm.
        3. Collapse penalty — kept as safety net against single-class collapse.
        4. Novelty bonus — kept for zero-day attack detection.
        5. Removed: HHI penalty, entropy bonus, balanced accuracy bonus,
           macro F1 bonus (all redundant with MCC).
        """
        w = self._class_weights[true_label]

        # 1. Cost-sensitive per-step reward (FN >> FP in IDS)
        #    FN_PENALTY is boosted by fn_weight_boost (default 2.0) because
        #    missing an attack is far more dangerous than a false alarm.
        fn_boost = getattr(self.reward_cfg, 'fn_weight_boost', 2.0)
        step_reward = (
            w * self.TP_REWARD * tp
            + self.TN_REWARD * tn
            - w * self.FP_PENALTY * fp
            - w * self.FN_PENALTY * fn_boost * fn
        )

        # 2. MCC bonus (computed from running episode confusion matrix)
        #    MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
        #    Range: [-1, 1], where 1 = perfect, 0 = random, -1 = inverse
        mcc_bonus = 0.0
        mcc_coef = getattr(self.reward_cfg, 'mcc_coef', 5.0)
        m = self._episode_metrics
        e_tp, e_fp, e_fn, e_tn = m["tp"], m["fp"], m["fn"], m["tn"]
        if self._episode_total_preds >= 10:
            numerator = e_tp * e_tn - e_fp * e_fn
            denominator = np.sqrt(
                max((e_tp + e_fp) * (e_tp + e_fn) * (e_tn + e_fp) * (e_tn + e_fn), 1e-8)
            )
            mcc = numerator / denominator
            mcc_bonus = mcc_coef * mcc  # reward range: [-mcc_coef, +mcc_coef]

        # 3. Correct class identification bonus
        reward = step_reward + mcc_bonus + class_bonus

        # 4. Latency + novelty (preserved from original design)
        reward += self.reward_cfg.delta * (1.0 - norm_latency)
        reward += self.reward_cfg.epsilon * novelty_bonus

        # 5. Collapse detection penalty — safety net (kept, every 20 steps)
        self._collapse_countdown -= 1
        if self._collapse_countdown <= 0:
            self._collapse_countdown = 20
            if self._episode_total_preds >= 10:
                top_prob = max(self._episode_pred_counts.values()) / self._episode_total_preds
                if top_prob > self.COLLAPSE_THRESHOLD:
                    collapse_severity = top_prob - self.COLLAPSE_THRESHOLD
                    reward -= self.COLLAPSE_PENALTY * collapse_severity
                    self._collapse_detected = True
                else:
                    self._collapse_detected = False

        return reward

    def get_episode_hhi(self) -> float:
        """
        Compute Herfindahl-Hirschman Index of predicted class distribution.
        Returns 1.0 when all predictions are one class (max collapse),
        returns ~1/num_classes when perfectly uniform.
        """
        if self._episode_total_preds == 0:
            return 1.0 / self.num_classes
        probs = np.array([
            self._episode_pred_counts[c] / self._episode_total_preds
            for c in range(self.num_classes)
        ])
        return float(np.sum(probs ** 2))

    def get_hhi_penalty(self) -> float:
        """
        Compute per-episode HHI bias penalty.
        hhi = sum((pred_count/total)^2) over classes
        hhi ranges from 1/num_classes (uniform) to 1.0 (single class).
        Penalty = HHI_PENALTY_COEF * (hhi - 1/num_classes) / (1 - 1/num_classes)
        Returns 0 when predictions are perfectly uniform, positive when collapsed.
        """
        hhi = self.get_episode_hhi()
        min_hhi = 1.0 / self.num_classes
        max_hhi = 1.0
        if max_hhi == min_hhi:
            normalized_hhi = 0.0
        else:
            normalized_hhi = (hhi - min_hhi) / (max_hhi - min_hhi)
        return self.HHI_PENALTY_COEF * normalized_hhi

    def get_class_distribution(self) -> Dict[int, float]:
        """Return per-class prediction distribution as a fraction of total predictions."""
        if self._episode_total_preds == 0:
            return {c: 0.0 for c in range(self.num_classes)}
        return {
            c: self._episode_pred_counts[c] / self._episode_total_preds
            for c in range(self.num_classes)
        }

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step with multi-class action.

        Args:
            action: either:
                - np.ndarray of shape [num_classes] with confidence per class
                  (legacy: predicted_class = argmax(action))
                - int class index (new Categorical policy: action is already the class)
        """
        step_start = time.perf_counter()

        true_label = self.y[self._order[self._idx]]

        # Support both legacy array (argmax) and new int (class index) actions
        if isinstance(action, (int, np.integer)):
            predicted_class = int(action)
        else:
            predicted_class = int(np.argmax(action))

        is_attack = int(true_label != 0)
        predicted_attack = int(predicted_class != 0)

        # Confusion for binary
        if predicted_attack == 1 and is_attack == 1:
            self._episode_metrics["tp"] += 1
            tp, fp, fn, tn = 1, 0, 0, 0
        elif predicted_attack == 1 and is_attack == 0:
            self._episode_metrics["fp"] += 1
            tp, fp, fn, tn = 0, 1, 0, 0
        elif predicted_attack == 0 and is_attack == 1:
            self._episode_metrics["fn"] += 1
            tp, fp, fn, tn = 0, 0, 1, 0
        else:
            self._episode_metrics["tn"] += 1
            tp, fp, fn, tn = 0, 0, 0, 1

        # Per-class metrics
        if predicted_class == true_label:
            self._class_metrics[true_label]["tp"] += 1
        else:
            self._class_metrics[predicted_class]["fp"] += 1
            self._class_metrics[true_label]["fn"] += 1

        # Extra reward for correct class identification (class_bonus passed to _compute_class_balanced_reward)
        class_bonus = 0.5 if predicted_class == true_label else -0.5

        # Latency
        latency = time.perf_counter() - step_start
        self._max_latency = max(self._max_latency, latency + 1e-9)
        norm_latency = min(latency / self._max_latency, 1.0)

        # Novelty: autoencoder reconstruction error OR unseen class
        novelty_bonus = 0.0
        if is_attack == 1 and predicted_attack == 1:
            if true_label not in self._seen_classes:
                novelty_bonus = 1.0
                self._seen_classes.add(true_label)
            elif self._novelty_detector is not None:
                state_t = torch.FloatTensor(self.X[self._order[self._idx]]).unsqueeze(0)
                with torch.no_grad():
                    recon_err = self._novelty_detector.reconstruction_error(state_t).item()
                if recon_err > self._novelty_threshold:
                    novelty_bonus = min(recon_err / self._novelty_threshold - 1.0, 1.0)

        # Track predicted class distribution for entropy bonus and HHI penalty
        self._episode_pred_counts[predicted_class] += 1
        self._episode_total_preds += 1

        # Class-balanced composite reward
        reward = self._compute_class_balanced_reward(
            tp=tp, fp=fp, fn=fn,
            true_label=true_label,
            predicted_class=predicted_class,
            norm_latency=norm_latency,
            novelty_bonus=novelty_bonus,
            class_bonus=class_bonus,
            tn=tn,
        )

        self._idx += 1
        self._step_count += 1
        done = self._idx >= len(self.X)
        next_state = self._get_state() if not done else np.zeros(self.state_dim, dtype=np.float32)

        info = {
            "true_label": true_label,
            "predicted_class": predicted_class,
            "predicted_attack": predicted_attack,
            "is_attack": is_attack,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "novelty": novelty_bonus,
            "latency": latency,
            "class_metrics": {k: v.copy() for k, v in self._class_metrics.items()},
            "episode_metrics": self._episode_metrics.copy(),
            "collapse_detected": self._collapse_detected,
            "pred_hhi": self.get_episode_hhi(),
        }

        return next_state, reward, done, info

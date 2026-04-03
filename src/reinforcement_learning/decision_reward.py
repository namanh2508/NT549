"""
Decision-Making Reward Function Module for IDS.
============================================================================
Improvement 2: Convert from classification to decision-making RL.

Trong paradigm decision-making, agent không chỉ phân loại Normal/Attack
mà phải QUYẾT ĐỊNH hành động phù hợp với từng loại tấn công.

Attack Categories: Normal, DoS, Probe, R2L, U2R
Response Actions: ALLOW, DROP, BLOCK_SRC, RATE_LIMIT, ALERT, MONITOR, ISOLATE

Reward Matrix (base_reward[category][action]):
  - Optimal action: reward cao nhất
  - Suboptimal action: reward thấp hoặc penalty nhẹ
  - Bad action: penalty nặng

Delayed Cost: Một số actions (BLOCK_SRC, ISOLATE) có chi phí dài hạn
             vì ảnh hưởng đến usability của mạng.

Cumulative Impact: Nếu agent block/rate-limit quá nhiều kết nối hợp lệ,
                  cumulative penalty được áp dụng.

Reference: Schulman et al., "Proximal Policy Optimization Algorithms", 2017
           (Reward shaping principles)
============================================================================
"""

import numpy as np
from collections import deque
import hashlib


# ============================================================================
# ACTION DEFINITIONS
# ============================================================================
# Action IDs cho decision-making RL
ACTION_ALLOW = 0
ACTION_DROP = 1
ACTION_BLOCK_SRC = 2
ACTION_RATE_LIMIT = 3
ACTION_ALERT = 4
ACTION_MONITOR = 5
ACTION_ISOLATE = 6

ACTION_NAMES = {
    ACTION_ALLOW: 'ALLOW',
    ACTION_DROP: 'DROP',
    ACTION_BLOCK_SRC: 'BLOCK_SRC',
    ACTION_RATE_LIMIT: 'RATE_LIMIT',
    ACTION_ALERT: 'ALERT',
    ACTION_MONITOR: 'MONITOR',
    ACTION_ISOLATE: 'ISOLATE',
}

NUM_DECISION_ACTIONS = 7


# ============================================================================
# ATTACK CATEGORY DEFINITIONS
# ============================================================================
CATEGORY_NORMAL = 0
CATEGORY_DOS = 1
CATEGORY_PROBE = 2
CATEGORY_R2L = 3
CATEGORY_U2R = 4

CATEGORY_NAMES = {
    CATEGORY_NORMAL: 'Normal',
    CATEGORY_DOS: 'DoS',
    CATEGORY_PROBE: 'Probe',
    CATEGORY_R2L: 'R2L',
    CATEGORY_U2R: 'U2R',
}


# ============================================================================
# REWARD MATRIX: base_reward[category][action]
# ============================================================================
# Row = Attack Category (0:Normal, 1:DoS, 2:Probe, 3:R2L, 4:U2R)
# Col = Action (0:ALLOW, 1:DROP, 2:BLOCK_SRC, 3:RATE_LIMIT, 4:ALERT, 5:MONITOR, 6:ISOLATE)
#
# Thiết kế nguyên tắc:
#   - Normal traffic: CHỈ nên ALLOW hoặc MONITOR, BLOCK/DROP = rất bad
#   - DoS attack: DROP/RATE_LIMIT/BLOCK_SRC là optimal, ALLOW = rất bad
#   - Probe attack: ALERT/MONITOR optimal, DROP/BLOCK = acceptable
#   - R2L attack: BLOCK_SRC/ISOLATE optimal, ALLOW = nguy hiểm
#   - U2R attack: BLOCK_SRC/ISOLATE optimal nhất, ALLOW = nguy hiểm nhất
BASE_REWARD_MATRIX = [
    # Normal
    [2.0, -2.0, -3.0, -0.5, -0.5, 1.0, -3.0],
    # DoS
    [-3.0, 2.0, 1.0, 1.5, -0.5, -0.5, -1.0],
    # Probe
    [-2.0, -1.0, -1.0, 1.0, 2.0, 1.5, -2.0],
    # R2L
    [-3.0, -1.0, 2.0, -2.0, 1.0, -1.0, 1.5],
    # U2R
    [-4.0, 1.0, 2.5, -3.0, -0.5, -2.0, 2.0],
]


# ============================================================================
# RESPONSE COST (delayed cost cho usability)
# ============================================================================
# Chi phí dài hạn của mỗi action (ảnh hưởng đến network usability)
RESPONSE_COST = {
    ACTION_ALLOW: 0.0,
    ACTION_DROP: 0.1,
    ACTION_BLOCK_SRC: 0.5,
    ACTION_RATE_LIMIT: 0.2,
    ACTION_ALERT: 0.0,
    ACTION_MONITOR: 0.0,
    ACTION_ISOLATE: 0.8,
}


# ============================================================================
# ATTACK SEVERITY (để tính delayed penalty)
# ============================================================================
# Mức độ nguy hiểm của mỗi loại tấn công (0-4)
ATTACK_SEVERITY = {
    CATEGORY_NORMAL: 0,
    CATEGORY_DOS: 2,
    CATEGORY_PROBE: 1,
    CATEGORY_R2L: 3,
    CATEGORY_U2R: 4,
}


class DecisionRewardFunction:
    """
    Reward Function cho Decision-Making IDS.

    Khác với classification reward (chỉ phân biệt đúng/sai),
    decision reward phản ánh CHẤT LƯỢNG QUYẾT ĐỊNH:
      - Action có phù hợp với attack category không?
      - Có gây unnecessary disruption không?
      - Cumulative impact có balanced không?

    Reward = base_reward(category, action)
           + delayed_cost(action)
           + cumulative_impact(session_stats)
           + novelty_bonus
    """

    def __init__(self,
                 base_reward_matrix=None,
                 response_cost_dict=None,
                 attack_severity_dict=None,
                 usability_weight=0.1,
                 max_block_ratio=0.1,
                 use_novelty=True,
                 novelty_hash_bits=16):
        """
        Initialize Decision Reward Function.

        Args:
            base_reward_matrix: Reward matrix [5 categories x 7 actions].
                               None = use default BASE_REWARD_MATRIX.
            response_cost_dict: Delayed cost for each action.
                               None = use default RESPONSE_COST.
            attack_severity_dict: Severity score for each category.
                                 None = use default ATTACK_SEVERITY.
            usability_weight: Weight for cumulative impact penalty.
                            Higher = stricter on blocking legitimate traffic.
            max_block_ratio: Maximum allowed ratio of blocked/allowed
                            before cumulative penalty kicks in.
            use_novelty: Whether to use novelty bonus for new attack patterns.
            novelty_hash_bits: Bits for state fingerprinting (novelty detection).
        """
        # Reward matrix: base_reward[category][action]
        if base_reward_matrix is None:
            self.base_reward = [row[:] for row in BASE_REWARD_MATRIX]
        else:
            self.base_reward = base_reward_matrix

        # Response costs
        if response_cost_dict is None:
            self.response_cost = RESPONSE_COST.copy()
        else:
            self.response_cost = response_cost_dict

        # Attack severity
        if attack_severity_dict is None:
            self.attack_severity = ATTACK_SEVERITY.copy()
        else:
            self.attack_severity = attack_severity_dict

        self.usability_weight = usability_weight
        self.max_block_ratio = max_block_ratio

        # ================================================================
        # Session tracking for cumulative impact
        # ================================================================
        self.session_blocked = 0   # Số kết nối bị block
        self.session_allowed = 0    # Số kết nối được allow
        self.session_alerts = 0    # Số alerts đã tạo
        self.total_impact_score = 0.0  # Tích lũy impact

        # ================================================================
        # Novelty tracking (khuyến khích generalization)
        # ================================================================
        self.use_novelty = use_novelty
        self.novelty_hash_bits = novelty_hash_bits
        self._seen_attack_hashes = set()
        self._hash_visit_counts = {}

    def compute(self, action, attack_category, state=None):
        """
        Tính reward cho một quyết định.

        Args:
            action: Action ID (0-6)
            attack_category: Attack category index (0-4)
            state: State vector for novelty detection (optional)

        Returns:
            reward: float
        """
        reward = 0.0

        # ================================================================
        # 1. BASE REWARD từ matrix
        # ================================================================
        if 0 <= attack_category < len(self.base_reward):
            if 0 <= action < len(self.base_reward[attack_category]):
                reward += self.base_reward[attack_category][action]

        # ================================================================
        # 2. DELAYED COST (response cost)
        # BLOCK_SRC và ISOLATE có chi phí dài hạn cho usability
        # ================================================================
        if action in self.response_cost:
            delayed_cost = self.response_cost[action]
            # U2R và R2L attack nguy hiểm hơn → cho phép block mạnh hơn
            severity = self.attack_severity.get(attack_category, 0)
            adjusted_cost = delayed_cost * (1.0 - 0.1 * severity)
            reward -= adjusted_cost

        # ================================================================
        # 3. CUMULATIVE IMPACT (penalty cho việc block quá nhiều)
        # Nếu block/allowed ratio > max_block_ratio → penalty
        # ================================================================
        self._update_session_stats(action, attack_category)
        cumulative_penalty = self._compute_cumulative_penalty()
        reward += cumulative_penalty

        # ================================================================
        # 4. NOVELTY BONUS (khuyến khích phát hiện attack patterns mới)
        # Chỉ áp dụng khi agent phản ứng đúng với attack
        # ================================================================
        if self.use_novelty and state is not None:
            is_correct_response = (self.base_reward[attack_category][action] > 0)
            if is_correct_response:
                novelty_bonus = self._compute_novelty_bonus(state)
                reward += novelty_bonus

        return reward

    def _update_session_stats(self, action, attack_category):
        """
        Cập nhật session statistics sau mỗi quyết định.

        Args:
            action: Action ID
            attack_category: Attack category
        """
        # Cập nhật counters
        if action == ACTION_ALLOW:
            self.session_allowed += 1
        elif action in (ACTION_DROP, ACTION_BLOCK_SRC, ACTION_ISOLATE):
            self.session_blocked += 1
        elif action == ACTION_ALERT:
            self.session_alerts += 1

        # Tính impact score tích lũy
        # Attack càng nặng + action càng disruptive → impact càng cao
        severity = self.attack_severity.get(attack_category, 0)
        action_cost = self.response_cost.get(action, 0)
        impact = (severity / 4.0) * action_cost
        self.total_impact_score += impact

    def _compute_cumulative_penalty(self):
        """
        Tính cumulative impact penalty.

        Nếu agent block quá nhiều kết nối (so với allowed),
        áp dụng penalty để tránh over-blocking behavior.

        Returns:
            penalty: float (negative value)
        """
        total_decisions = self.session_blocked + self.session_allowed

        if total_decisions < 10:
            # Chưa đủ samples để đánh giá
            return 0.0

        block_ratio = self.session_blocked / total_decisions

        # Penalty khi block ratio vượt ngưỡng
        if block_ratio > self.max_block_ratio:
            excess = block_ratio - self.max_block_ratio
            # Severity-weighted: usability_weight càng lớn → penalty càng nặng
            penalty = -excess * self.usability_weight * 10.0
            return penalty

        return 0.0

    def _compute_novelty_bonus(self, state):
        """
        Tính novelty bonus cho pattern attack mới.

        Ý tưởng: Nếu agent phát hiện và phản ứng đúng với
        một attack pattern mới → thưởng thêm.

        Args:
            state: numpy array - state vector

        Returns:
            novelty_bonus: float in [0, 0.3]
        """
        if state is None:
            return 0.0

        # Tạo fingerprint
        discretized = np.round(state * (2 ** self.novelty_hash_bits)).astype(np.int32)
        state_hash = hashlib.md5(discretized.tobytes()).hexdigest()

        # Get visit count
        visit_count = self._hash_visit_counts.get(state_hash, 0)

        # Update
        self._hash_visit_counts[state_hash] = visit_count + 1
        self._seen_attack_hashes.add(state_hash)

        # Novelty giảm dần theo số lần thấy
        novelty = 0.3 / (1.0 + visit_count)

        return novelty

    def reset_session(self):
        """
        Reset session statistics.

        Gọi khi bắt đầu episode/round mới để đánh giá
        cumulative impact trên từng episode.
        """
        self.session_blocked = 0
        self.session_allowed = 0
        self.session_alerts = 0
        self.total_impact_score = 0.0

    def reset_novelty(self):
        """Reset novelty tracker."""
        self._seen_attack_hashes.clear()
        self._hash_visit_counts.clear()

    def reset_all(self):
        """Reset cả session và novelty."""
        self.reset_session()
        self.reset_novelty()

    def get_session_stats(self):
        """
        Lấy session statistics để logging.

        Returns:
            dict: Session stats
        """
        total = self.session_blocked + self.session_allowed
        block_ratio = self.session_blocked / total if total > 0 else 0.0

        return {
            'blocked': self.session_blocked,
            'allowed': self.session_allowed,
            'alerts': self.session_alerts,
            'total_decisions': total,
            'block_ratio': block_ratio,
            'total_impact_score': self.total_impact_score,
            'unique_attack_patterns': len(self._seen_attack_hashes),
        }

    @staticmethod
    def get_action_name(action_id):
        """Get action name from ID."""
        return ACTION_NAMES.get(action_id, 'UNKNOWN')

    @staticmethod
    def get_category_name(category_id):
        """Get category name from ID."""
        return CATEGORY_NAMES.get(category_id, 'UNKNOWN')

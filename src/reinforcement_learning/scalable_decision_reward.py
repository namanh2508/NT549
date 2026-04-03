"""
Scalable Decision-Making Reward System for IDS.
============================================================================
THIẾT KẾ SCALABLE - HỌC TỪ FEEDBACK CHO NOVEL ATTACKS

Vấn đề với hardcoded approach:
  - Attack types cố định → không handle được novel attacks
  - Reward matrix cố định → không adapt được khi environment thay đổi
  - Hardcoded mapping → không scale được across datasets

Giải pháp scalable:
  1. ATTACK TAXONOMY: Dùng severity/impact scores thay vì hardcoded categories
  2. ACTION SPACE: Actions defined by cost-benefit characteristics
  3. ONLINE LEARNING: Học từ reward feedback để update action preferences
  4. NOVEL ATTACK DETECTION: Detect unknown attacks, explore responses, learn

Reference:
  - MITRE ATT&CK framework (generalized categories)
  - Multi-Armed Bandit for action selection
  - Online learning with exploration-exploitation tradeoff
============================================================================
"""

import numpy as np
from collections import defaultdict
import hashlib
import copy


# ============================================================================
# PART 1: ATTACK TAXONOMY - Generalized representation
# ============================================================================
# Thay vì hardcoded attack names (neptune, ipsweep, ...),
# dùng FEATURE VECTOR mô tả attack characteristics

class AttackTaxonomy:
    """
    Attack Taxonomy dùng MITRE ATT&CK-inspired categories.

    Mỗi attack được mô tả bởi:
      - category: high-level attack category (DoS, Exfiltration, PrivEsc, etc.)
      - severity: 0-4 (0=benign, 4=critical)
      - network_impact: 0-4 (impact on network availability/integrity)
      - data_impact: 0-4 (impact on data confidentiality/integrity)
      - lateral_movement: boolean (can spread to other systems)

    Dùng scores thay vì discrete labels → generalize được sang novel attacks
    """

    # Generalized attack categories (inspired by MITRE ATT&CK)
    CATEGORIES = [
        'BENIGN',           # 0: Normal traffic
        'RECONNAISSANCE',  # 1: Information gathering (probe, scan)
        'DOS',              # 2: Denial of Service
        'EXPLOITATION',    # 3: Exploitation of vulnerabilities
        'PRIV_ESC',         # 4: Privilege Escalation
        'LATERAL',         # 5: Lateral Movement
        'EXFILTRATION',    # 6: Data Exfiltration
        'PERSISTENCE',     # 7: Maintaining foothold
        'EVASION',         # 8: Evading detection
        'MALWARE',         # 9: Malware/Ransomware
        'UNKNOWN',          # 10: Novel/Unknown attack
    ]

    CATEGORY_INDEX = {cat: i for i, cat in enumerate(CATEGORIES)}

    @staticmethod
    def get_default_severity(attack_signature):
        """
        Lấy default severity score cho attack dựa trên signature.

        Nếu attack không known → trả về UNKNOWN category và medium severity.
        Severity scores:
          0 = Benign (normal traffic)
          1 = Low (reconnaissance, minor probes)
          2 = Medium (DoS, scanning)
          3 = High (exploitation, privilege escalation)
          4 = Critical (data exfiltration, ransomware, APT)

        Args:
            attack_signature: String identifier của attack
        Returns:
            dict với category_idx, severity, network_impact, data_impact, lateral
        """
        sig_lower = attack_signature.lower()

        # ================================================================
        # BENIGN
        # ================================================================
        if sig_lower == 'normal':
            return {
                'category_idx': 0,
                'category': 'BENIGN',
                'severity': 0,
                'network_impact': 0,
                'data_impact': 0,
                'lateral_movement': False,
            }

        # ================================================================
        # RECONNAISSANCE (severity=1)
        # ================================================================
        if sig_lower in ('ipsweep', 'nmap', 'portsweep', 'satan', 'mscan', 'saint',
                         'ipsweep_probe', 'scan', 'recon'):
            return {
                'category_idx': 1,
                'category': 'RECONNAISSANCE',
                'severity': 1,
                'network_impact': 1,
                'data_impact': 1,
                'lateral_movement': False,
            }

        # ================================================================
        # DOS (severity=2)
        # ================================================================
        if sig_lower in ('back', 'land', 'neptune', 'pod', 'smurf', 'teardrop',
                         'mailbomb', 'apache2', 'processtable', 'udpstorm',
                         'syn flood', 'udp flood', 'icmp flood', 'dos_attack',
                         'ddos'):
            return {
                'category_idx': 2,
                'category': 'DOS',
                'severity': 2,
                'network_impact': 4,
                'data_impact': 0,
                'lateral_movement': False,
            }

        # ================================================================
        # EXPLOITATION (severity=3)
        # ================================================================
        if sig_lower in ('buffer_overflow', 'loadmodule', 'perl', 'sqlattack',
                         'xterm', 'httptunnel', 'ps', 'rootkit',
                         'ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf',
                         'sendmail', 'named', 'snmpgetattack', 'snmpguess',
                         'warezclient', 'warezmaster', 'xlock', 'xsnoop', 'worm',
                         'exploit', 'buffer_overflow_attack', 'vulnerability_exploit'):
            return {
                'category_idx': 3,
                'category': 'EXPLOITATION',
                'severity': 3,
                'network_impact': 2,
                'data_impact': 3,
                'lateral_movement': True,
            }

        # ================================================================
        # PRIVILEGE ESCALATION (severity=4)
        # ================================================================
        if sig_lower in ('rootkit', 'loadmodule', 'buffer_overflow', 'perl',
                         'privilege_escalation', 'root_access'):
            return {
                'category_idx': 4,
                'category': 'PRIV_ESC',
                'severity': 4,
                'network_impact': 1,
                'data_impact': 4,
                'lateral_movement': True,
            }

        # ================================================================
        # LATERAL MOVEMENT (severity=3)
        # ================================================================
        if sig_lower in ('ssh', 'telnet', 'rsh', 'rlogin', 'ftp', 'sftp',
                         'lateral_movement', 'pivot'):
            return {
                'category_idx': 5,
                'category': 'LATERAL',
                'severity': 3,
                'network_impact': 2,
                'data_impact': 3,
                'lateral_movement': True,
            }

        # ================================================================
        # DATA EXFILTRATION (severity=4)
        # ================================================================
        if sig_lower in ('spy', 'data_theft', 'exfiltration', 'data_steal',
                         'unauthorized_data_access', ' ftp_write'):
            return {
                'category_idx': 6,
                'category': 'EXFILTRATION',
                'severity': 4,
                'network_impact': 1,
                'data_impact': 4,
                'lateral_movement': True,
            }

        # ================================================================
        # PERSISTENCE (severity=3)
        # ================================================================
        if sig_lower in ('backdoor', 'rootkit', 'persistence', 'bootkit',
                         'trojan', 'trapdoor'):
            return {
                'category_idx': 7,
                'category': 'PERSISTENCE',
                'severity': 3,
                'network_impact': 1,
                'data_impact': 3,
                'lateral_movement': False,
            }

        # ================================================================
        # EVASION (severity=2)
        # ================================================================
        if sig_lower in ('stealth', 'covert', 'tunnel', 'encrypted',
                         'evasion', 'obfuscation'):
            return {
                'category_idx': 8,
                'category': 'EVASION',
                'severity': 2,
                'network_impact': 1,
                'data_impact': 2,
                'lateral_movement': False,
            }

        # ================================================================
        # MALWARE (severity=4)
        # ================================================================
        if sig_lower in ('virus', 'worm', 'ransomware', 'trojan', 'bot',
                         'keylogger', 'rootkit_malware', 'malware'):
            return {
                'category_idx': 9,
                'category': 'MALWARE',
                'severity': 4,
                'network_impact': 3,
                'data_impact': 4,
                'lateral_movement': True,
            }

        # ================================================================
        # UNKNOWN / NOVEL ATTACK (severity=2, category=UNKNOWN)
        # Nếu không match → coi là novel attack
        # ================================================================
        return {
            'category_idx': 10,
            'category': 'UNKNOWN',
            'severity': 2,  # Default medium severity
            'network_impact': 2,
            'data_impact': 2,
            'lateral_movement': False,
            'is_novel': True,  # Flag để track novel attacks
        }


# ============================================================================
# PART 2: GENERALIZED ACTION SPACE
# ============================================================================
# Actions defined by characteristics: cost, speed, disruption_level

class ActionSpace:
    """
    Generalized Action Space cho IDS.

    Mỗi action được define bởi:
      - name: tên action
      - cost: cost cho network/service (0-1)
      - speed: tốc độ xử lý (0=instant, 1=slow)
      - disruption: mức độ gián đoạn service (0=none, 1=high)
      - reversibility: có thể reverse không (0=no, 1=yes)
      - collection_evidence: có thu thập evidence không

    Định nghĩa này cho phép AGENT học khi nào nên dùng action nào
    dựa trên attack characteristics.
    """

    ACTIONS = {
        0: {
            'name': 'ALLOW',
            'description': 'Allow traffic through',
            'cost': 0.0,
            'speed': 0.0,  # instant
            'disruption': 0.0,
            'reversibility': 0,
            'collect_evidence': False,
        },
        1: {
            'name': 'LOG_ALERT',
            'description': 'Log and alert, allow through',
            'cost': 0.0,
            'speed': 0.1,
            'disruption': 0.0,
            'reversibility': 0,
            'collect_evidence': True,
        },
        2: {
            'name': 'RATE_LIMIT',
            'description': 'Throttle connection bandwidth',
            'cost': 0.1,
            'speed': 0.2,
            'disruption': 0.2,
            'reversibility': 1,
            'collect_evidence': True,
        },
        3: {
            'name': 'DROP_CONNECTION',
            'description': 'Drop current connection',
            'cost': 0.1,
            'speed': 0.1,
            'disruption': 0.1,
            'reversibility': 1,
            'collect_evidence': True,
        },
        4: {
            'name': 'BLOCK_SOURCE_TEMPORARY',
            'description': 'Block source IP for limited time',
            'cost': 0.3,
            'speed': 0.2,
            'disruption': 0.4,
            'reversibility': 1,
            'collect_evidence': True,
        },
        5: {
            'name': 'BLOCK_SOURCE_PERMANENT',
            'description': 'Block source IP permanently',
            'cost': 0.5,
            'speed': 0.2,
            'disruption': 0.5,
            'reversibility': 0,
            'collect_evidence': True,
        },
        6: {
            'name': 'ISOLATE',
            'description': 'Isolate connection/session completely',
            'cost': 0.6,
            'speed': 0.3,
            'disruption': 0.7,
            'reversibility': 0,
            'collect_evidence': True,
        },
        7: {
            'name': 'INVESTIGATE',
            'description': 'Defer decision, collect more info',
            'cost': 0.0,
            'speed': 0.5,
            'disruption': 0.0,
            'reversibility': 1,
            'collect_evidence': True,
        },
    }

    NUM_ACTIONS = len(ACTIONS)

    @staticmethod
    def get_action_properties(action_id):
        """Get properties of an action."""
        return ActionSpace.ACTIONS.get(action_id, None)

    @staticmethod
    def get_all_action_ids():
        """Get list of all action IDs."""
        return list(ActionSpace.ACTIONS.keys())


# ============================================================================
# PART 3: ONLINE LEARNING REWARD SYSTEM
# ============================================================================
# Học từ reward feedback để update action preferences

class OnlineLearningReward:
    """
    Reward System có khả năng HỌC TỪ FEEDBACK.

    Core concept: Multi-Armed Bandit for each attack category
    - Với mỗi attack type, duy trì Q-value estimate cho mỗi action
    - Q-values được update dựa trên reward feedback
    - Exploration policy (epsilon-greedy) để handle novel situations

    Reward = base_reward(attack, action) + learned_component(Q-values)

    Base reward được tính từ:
    1. Action appropriateness: action có suitable cho attack type không
    2. Action cost: action gây bao nhiêu disruption
    3. Delayed consequence: action có prevent future attacks không
    """

    # Learning rate cho Q-value updates
    LEARNING_RATE = 0.1

    # Exploration rate (for novel attacks)
    EXPLORATION_RATE = 0.3

    def __init__(self, use_online_learning=True, use_base_reward=True):
        """
        Initialize Online Learning Reward System.

        Args:
            use_online_learning: Nếu True, học từ feedback
                              Nếu False, dùng fixed base reward
            use_base_reward: Nếu True, kết hợp base reward với learned reward
        """
        self.use_online_learning = use_online_learning
        self.use_base_reward = use_base_reward

        # ================================================================
        # Q-values for each (attack_category, action) pair
        # Q[category_idx][action_id] = average reward received
        # Shape: [num_categories x num_actions]
        # ================================================================
        num_categories = len(AttackTaxonomy.CATEGORIES)
        num_actions = ActionSpace.NUM_ACTIONS

        # Initialize Q-values optimistically (encourage exploration)
        self.Q = np.ones((num_categories, num_actions)) * 0.5

        # Count of how many times each (category, action) pair was tried
        self.action_counts = np.zeros((num_categories, num_actions))

        # Running sum of rewards for each (category, action)
        self.reward_sums = np.zeros((num_categories, num_actions))

        # ================================================================
        # Novel attack tracking
        # ================================================================
        self.known_attacks = set()  # Attack signatures that have been seen
        self.novel_attacks = set()  # Novel attack signatures
        self.novel_attack_learned = {}  # Track if we've learned action for novel attack

        # ================================================================
        # Session statistics for cumulative impact
        # ================================================================
        self.session_stats = {
            'total_blocks': 0,
            'total_allows': 0,
            'total_alerts': 0,
            'total_investigates': 0,
        }

        # ================================================================
        # Success history (for delayed reward computation)
        # ================================================================
        self.recent_decisions = []  # List of (attack_sig, action, was_correct)
        self.max_recent = 100

    def compute_reward(self, action_id, attack_signature, attack_properties=None,
                      delayed_outcome=None, session_context=None):
        """
        Compute reward cho một quyết định.

        Reward = base_component + learned_component

        Args:
            action_id: Action được chọn (0-7)
            attack_signature: String identifier của attack (e.g., 'neptune')
            attack_properties: Optional dict với attack characteristics
            delayed_outcome: Optional dict với outcome information
            session_context: Optional dict với session statistics

        Returns:
            reward: float
        """
        # ================================================================
        # Step 1: Get attack taxonomy
        # ================================================================
        if attack_properties is None:
            taxonomy = AttackTaxonomy.get_default_severity(attack_signature)
        else:
            # Use provided properties
            taxonomy = copy.copy(attack_properties)

        category_idx = taxonomy['category_idx']
        severity = taxonomy['severity']
        is_novel = taxonomy.get('is_novel', attack_signature in self.novel_attacks)

        # Track known vs novel attacks
        if is_novel:
            self.novel_attacks.add(attack_signature)
            self.novel_attack_learned.setdefault(attack_signature, False)
        else:
            self.known_attacks.add(attack_signature)

        # ================================================================
        # Step 2: Compute base reward
        # ================================================================
        base_reward = self._compute_base_reward(
            action_id, category_idx, severity, taxonomy
        )

        # ================================================================
        # Step 3: Compute learned reward component (from Q-values)
        # ================================================================
        learned_reward = 0.0
        if self.use_online_learning:
            # Exploration bonus for novel situations
            if is_novel or self.action_counts[category_idx, action_id] < 5:
                # Not enough data → add exploration bonus
                learned_reward += self.EXPLORATION_RATE

            # Use current Q-value as reward estimate
            learned_reward += self.Q[category_idx, action_id]

        # ================================================================
        # Step 4: Add session cumulative impact penalty
        # ================================================================
        session_penalty = self._compute_session_penalty(session_context)

        # ================================================================
        # Step 5: Total reward
        # ================================================================
        total_reward = base_reward + learned_reward + session_penalty

        # ================================================================
        # Step 6: Update statistics
        # ================================================================
        self._update_session_stats(action_id)
        self._add_to_recent_decisions(attack_signature, action_id, base_reward > 0)

        return total_reward

    def _compute_base_reward(self, action_id, category_idx, severity, taxonomy):
        """
        Compute base reward dựa trên action-appropriateness.

        Thiết kế nguyên tắc:
        1. Severity cao + action mạnh (block, isolate) → reward cao
        2. Severity thấp + action nhẹ (allow, log) → reward cao
        3. Severity cao + action nhẹ (allow) → reward âm nặng
        4. Severity thấp + action mạnh (permanent block) → reward âm nhẹ

        Reward matrix động theo severity level.
        """
        action_props = ActionSpace.get_action_properties(action_id)
        if action_props is None:
            return -1.0

        action_cost = action_props['cost']
        action_disruption = action_props['disruption']

        # ================================================================
        # BENIGN traffic (severity=0)
        # ================================================================
        if category_idx == 0:  # BENIGN
            if action_id in (0, 1):  # ALLOW, LOG_ALERT
                return 2.0  # Correct: allow benign traffic
            elif action_id in (3, 4):  # DROP, BLOCK_TEMP
                return -1.5  # Overreaction: blocking normal traffic
            elif action_id in (5, 6):  # BLOCK_PERM, ISOLATE
                return -2.5  # Major overreaction
            elif action_id == 2:  # RATE_LIMIT
                return -0.5  # Minor inconvenience
            else:  # INVESTIGATE
                return 0.5  # OK but unnecessary

        # ================================================================
        # DOS attacks (severity=2, high network_impact)
        # ================================================================
        if category_idx == 2:  # DOS
            if action_id in (3, 4):  # DROP, BLOCK_TEMP
                return 2.0 + (severity / 4.0)  # Good: stop DoS
            elif action_id == 2:  # RATE_LIMIT
                return 1.5  # Good: slow down DoS
            elif action_id == 5:  # BLOCK_PERM
                return 1.0  # OK: permanent block
            elif action_id == 6:  # ISOLATE
                return 0.5  # OK but overkill
            elif action_id == 0:  # ALLOW
                return -2.0  # Bad: let DoS through
            elif action_id == 1:  # LOG_ALERT
                return -0.5  # Bad: should block
            else:  # INVESTIGATE
                return -1.0  # Bad: need immediate action

        # ================================================================
        # RECONNAISSANCE (severity=1, low impact)
        # ================================================================
        if category_idx == 1:  # RECON
            if action_id in (1, 7):  # LOG_ALERT, INVESTIGATE
                return 2.0  # Good: collect info
            elif action_id == 2:  # RATE_LIMIT
                return 1.5  # Good: slow down recon
            elif action_id in (0,):  # ALLOW
                return 0.5  # OK: low threat
            elif action_id in (3, 4):  # DROP, BLOCK
                return -0.5  # Overreaction but understandable
            else:
                return -1.0

        # ================================================================
        # EXPLOITATION, PRIV_ESC, LATERAL, EXFILTRATION, MALWARE
        # (severity >= 3, high data impact, often lateral movement)
        # ================================================================
        if category_idx in (3, 4, 5, 6, 9):  # EXPLOITATION + others
            if action_id == 5:  # BLOCK_PERM
                return 2.5 + (severity / 4.0)  # Best: stop attacker
            elif action_id == 6:  # ISOLATE
                return 2.0 + (severity / 4.0)  # Good: contain
            elif action_id == 4:  # BLOCK_TEMP
                return 1.5  # OK: temporary block
            elif action_id == 3:  # DROP
                return 1.0  # OK: stop this attempt
            elif action_id == 1:  # LOG_ALERT
                return 0.0  # Not enough: need to block
            elif action_id == 0:  # ALLOW
                return -3.0 - severity  # Very bad: let attacker in
            elif action_id == 7:  # INVESTIGATE
                return -1.0  # Risky: need immediate action
            else:  # RATE_LIMIT
                return -1.5

        # ================================================================
        # PERSISTENCE, EVASION (severity=2-3)
        # ================================================================
        if category_idx in (7, 8):  # PERSISTENCE, EVASION
            if action_id in (5, 6):  # BLOCK_PERM, ISOLATE
                return 2.0  # Good: prevent persistence
            elif action_id == 4:  # BLOCK_TEMP
                return 1.5  # OK
            elif action_id == 1:  # LOG_ALERT
                return 1.0  # OK: document behavior
            elif action_id == 0:  # ALLOW
                return -2.5  # Bad: let threat persist
            else:
                return 0.5

        # ================================================================
        # UNKNOWN / NOVEL (severity=2, category=UNKNOWN)
        # Use cautious approach: block moderately
        # ================================================================
        if category_idx == 10:  # UNKNOWN
            # For unknown attacks, prefer blocking moderately
            if action_id in (4, 5):  # BLOCK_TEMP/PERM
                return 1.0  # Cautious: block
            elif action_id in (3, 6):  # DROP, ISOLATE
                return 0.8  # Cautious
            elif action_id == 1:  # LOG_ALERT
                return 0.5  # OK: document
            elif action_id == 7:  # INVESTIGATE
                return 0.3  # OK: learn more
            elif action_id == 0:  # ALLOW
                return -1.0  # Risky for unknown
            else:  # RATE_LIMIT
                return 0.2

        # ================================================================
        # Catch-all fallback
        # ================================================================
        return 0.0

    def _compute_session_penalty(self, session_context):
        """
        Compute cumulative impact penalty từ session statistics.

        Nếu agent block quá nhiều trong session → penalty
        Để tránh over-blocking behavior.
        """
        if session_context is None:
            return 0.0

        blocks = session_context.get('total_blocks', 0)
        allows = session_context.get('total_allows', 1)
        total = blocks + allows

        if total < 10:
            return 0.0

        block_ratio = blocks / total
        max_acceptable = 0.2  # 20% blocks

        if block_ratio > max_acceptable:
            excess = block_ratio - max_acceptable
            penalty = -excess * 2.0  # Scaled penalty
            return penalty

        return 0.0

    def _update_session_stats(self, action_id):
        """Update session statistics."""
        if action_id == 0:  # ALLOW
            self.session_stats['total_allows'] += 1
        elif action_id in (3, 4, 5, 6):  # DROP, BLOCK_TEMP, BLOCK_PERM, ISOLATE
            self.session_stats['total_blocks'] += 1
        elif action_id == 1:  # LOG_ALERT
            self.session_stats['total_alerts'] += 1
        elif action_id == 7:  # INVESTIGATE
            self.session_stats['total_investigates'] += 1

    def _add_to_recent_decisions(self, attack_sig, action_id, was_good):
        """Add to recent decisions history for delayed reward."""
        self.recent_decisions.append((attack_sig, action_id, was_good))
        if len(self.recent_decisions) > self.max_recent:
            self.recent_decisions.pop(0)

    def update_from_feedback(self, attack_signature, action_id, feedback_reward):
        """
        Update Q-values từ feedback.

        Đây là core của online learning:
        - Agent nhận reward feedback từ environment
        - Q-value được update theo công thức:
          Q(c,a) = Q(c,a) + α * (r - Q(c,a))

        Args:
            attack_signature: Attack type identifier
            action_id: Action đã thực hiện
            feedback_reward: Reward từ environment
        """
        if not self.use_online_learning:
            return

        # Get attack category
        taxonomy = AttackTaxonomy.get_default_severity(attack_signature)
        category_idx = taxonomy['category_idx']

        # Update action count
        self.action_counts[category_idx, action_id] += 1

        # Update running reward sum
        self.reward_sums[category_idx, action_id] += feedback_reward

        # Compute new Q-value (running average)
        n = self.action_counts[category_idx, action_id]
        self.Q[category_idx, action_id] = (
            self.reward_sums[category_idx, action_id] / n
        )

        # Mark novel attack as "learned" if we have enough samples
        if taxonomy.get('is_novel', False):
            if n >= 10:  # After 10 samples, consider it learned
                self.novel_attack_learned[attack_signature] = True

    def get_best_action_for_attack(self, attack_signature, exploit_only=False):
        """
        Get best action cho một attack dựa trên learned Q-values.

        Args:
            attack_signature: Attack identifier
            exploit_only: Nếu True, không exploration (chỉ exploit learned)

        Returns:
            best_action_id: Action tốt nhất
            q_value: Q-value của action đó
            is_novel: Attack có phải là novel không
        """
        taxonomy = AttackTaxonomy.get_default_severity(attack_signature)
        category_idx = taxonomy['category_idx']
        is_novel = taxonomy.get('is_novel', False) or attack_signature in self.novel_attacks

        if exploit_only or not is_novel:
            # Exploitation: use learned Q-values
            best_action = int(np.argmax(self.Q[category_idx]))
            return best_action, self.Q[category_idx, best_action], is_novel

        # ================================================================
        # Exploration: add randomness for novel attacks
        # Epsilon-greedy: với xác suất epsilon, chọn random
        # ================================================================
        if np.random.random() < self.EXPLORATION_RATE:
            # Random action for exploration
            best_action = np.random.choice(ActionSpace.get_all_action_ids())
        else:
            # Use best known action
            best_action = int(np.argmax(self.Q[category_idx]))

        return best_action, self.Q[category_idx, best_action], is_novel

    def get_action_confidence(self, attack_signature):
        """
        Get confidence scores cho mỗi action với attack.

        Returns:
            dict: action_id -> (q_value, count, confidence)
        """
        taxonomy = AttackTaxonomy.get_default_severity(attack_signature)
        category_idx = taxonomy['category_idx']

        results = {}
        for action_id in ActionSpace.get_all_action_ids():
            q_val = self.Q[category_idx, action_id]
            count = int(self.action_counts[category_idx, action_id])
            # Confidence = 1 / (1 + count) → high confidence khi count cao
            confidence = count / (count + 10)
            results[action_id] = (q_val, count, confidence)

        return results

    def reset_session(self):
        """Reset session statistics."""
        self.session_stats = {
            'total_blocks': 0,
            'total_allows': 0,
            'total_alerts': 0,
            'total_investigates': 0,
        }
        self.recent_decisions.clear()

    def reset_all(self):
        """Reset everything including learned Q-values."""
        self.reset_session()
        num_categories = len(AttackTaxonomy.CATEGORIES)
        num_actions = ActionSpace.NUM_ACTIONS
        self.Q = np.ones((num_categories, num_actions)) * 0.5
        self.action_counts = np.zeros((num_categories, num_actions))
        self.reward_sums = np.zeros((num_categories, num_actions))
        self.known_attacks.clear()
        self.novel_attacks.clear()
        self.novel_attack_learned.clear()

    def get_stats(self):
        """Get comprehensive statistics."""
        return {
            'known_attacks': len(self.known_attacks),
            'novel_attacks': len(self.novel_attacks),
            'learned_from_novel': sum(1 for v in self.novel_attack_learned.values() if v),
            'total_action_updates': int(np.sum(self.action_counts)),
            'session_stats': self.session_stats.copy(),
        }

    @staticmethod
    def get_category_name(category_idx):
        """Get category name from index."""
        return AttackTaxonomy.CATEGORIES[category_idx] if 0 <= category_idx < len(AttackTaxonomy.CATEGORIES) else 'UNKNOWN'

    @staticmethod
    def get_action_name(action_id):
        """Get action name from ID."""
        props = ActionSpace.get_action_properties(action_id)
        return props['name'] if props else 'UNKNOWN'

    def _compute_base_reward_continuous(self, action, category_idx, is_attack):
        """
        Compute base reward cho continuous action space.

        Maps continuous action dimensions to reward:
          - action[0]: block_duration (0-300s) -> 0=allow, >0=block
          - action[1]: throttle_rate (0-100%) -> 0=no throttle, 100=full block
          - action[2]: alert_severity (0-1) -> 0=no alert, 1=critical
          - action[3]: log_verbosity (0-10) -> 0=silent, 10=verbose

        Args:
            action: numpy array [4] continuous values
            category_idx: attack category index (0-10)
            is_attack: bool, True if this is attack traffic

        Returns:
            reward: float
        """
        block_duration = action[0]  # 0-300 seconds
        throttle_rate = action[1]   # 0-100 percent
        alert_severity = action[2]  # 0-1
        log_level = action[3]       # 0-10

        # Normalize action intensities
        block_intensity = block_duration / 300.0  # 0-1
        throttle_intensity = throttle_rate / 100.0  # 0-1

        # ================================================================
        # BENIGN traffic (severity=0)
        # ================================================================
        if category_idx == 0:  # BENIGN
            # Good: allow traffic (low block_duration, low throttle)
            if block_intensity < 0.1 and throttle_intensity < 0.1:
                return 2.0
            # Medium bad: minor throttling
            elif block_intensity < 0.3 and throttle_intensity < 0.3:
                return 0.5
            # Bad: significant blocking
            elif block_intensity < 0.6:
                return -1.0
            else:
                return -2.5

        # ================================================================
        # DOS attacks (severity=2, high network_impact)
        # ================================================================
        if category_idx == 2:  # DOS
            # Good: block or throttle aggressively
            if block_intensity > 0.5 or throttle_intensity > 0.7:
                return 2.0 + alert_severity
            elif block_intensity > 0.2 or throttle_intensity > 0.4:
                return 1.5
            elif alert_severity > 0.5:
                return 0.5  # Alert is OK but not enough
            else:
                return -2.0  # Allow through is bad

        # ================================================================
        # RECONNAISSANCE (severity=1, low impact)
        # ================================================================
        if category_idx == 1:  # RECON
            # Good: alert and moderate logging
            if alert_severity > 0.5 and log_level > 5:
                return 2.0
            # OK: throttle moderately
            elif throttle_intensity > 0.2:
                return 1.5
            # Allow is OK for low threat
            elif block_intensity < 0.1:
                return 0.5
            else:
                return -0.5

        # ================================================================
        # EXPLOITATION, PRIV_ESC, LATERAL, EXFILTRATION, MALWARE
        # (severity >= 3, high data impact)
        # ================================================================
        if category_idx in (3, 4, 5, 6, 9):  # EXPLOITATION + others
            # Best: strong blocking action
            if block_intensity > 0.7 or throttle_intensity > 0.8:
                return 2.5 + alert_severity
            # Good: moderate block with alert
            elif block_intensity > 0.4 or throttle_intensity > 0.5:
                return 2.0
            # OK: just alert
            elif alert_severity > 0.6:
                return 1.0
            # Allow is very bad
            elif block_intensity < 0.1:
                return -3.0
            else:
                return 0.5

        # ================================================================
        # PERSISTENCE, EVASION (severity=2-3)
        # ================================================================
        if category_idx in (7, 8):  # PERSISTENCE, EVASION
            if block_intensity > 0.5:
                return 2.0
            elif alert_severity > 0.5:
                return 1.5
            elif block_intensity < 0.1:
                return -2.5
            else:
                return 0.5

        # ================================================================
        # UNKNOWN / NOVEL (severity=2, category=UNKNOWN)
        # Cautious approach: moderate blocking
        # ================================================================
        if category_idx == 10:  # UNKNOWN
            if block_intensity > 0.4 and alert_severity > 0.3:
                return 1.0
            elif throttle_intensity > 0.3:
                return 0.8
            elif alert_severity > 0.5:
                return 0.5
            elif block_intensity < 0.1:
                return -1.0
            else:
                return 0.2

        # ================================================================
        # Catch-all fallback
        # ================================================================
        return 0.0

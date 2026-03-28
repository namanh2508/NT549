"""
Enhanced Reward Function Module for IDS.
============================================================================
Improvement 2: Redesign hàm reward từ đơn giản (đúng/sai) sang hàm
phức tạp hơn, có trọng số động và novelty bonus.
============================================================================

Bài gốc chỉ dùng reward đơn giản:
  reward = +1 nếu dự đoán đúng
  reward = -1 nếu dự đoán sai

Hàm reward đề xuất:
  R(t) = α·TP − β·FP − γ·FN + δ·(1 − normalized_latency) + ε·novelty_bonus

Trong đó:
  - α (alpha): Trọng số cho True Positive
      → Thưởng khi agent phát hiện đúng tấn công
  - β (beta):  Trọng số phạt cho False Positive
      → Phạt khi agent cảnh báo sai (normal bị gán nhãn attack)
      → Có thể điều chỉnh: môi trường cho phép FP cao thì giảm β
  - γ (gamma_fn): Trọng số phạt cho False Negative
      → Phạt khi agent bỏ sót tấn công (attack bị gán nhãn normal)
      → Môi trường nhạy cảm (bệnh viện, ngân hàng) nên TĂNG γ
        để ưu tiên không bỏ sót tấn công
  - δ (delta): Trọng số cho latency penalty (tùy chọn)
      → Khuyến khích agent ra quyết định nhanh (hiện đặt = 0)
  - ε (epsilon_nov): Trọng số cho novelty bonus
      → Thưởng thêm khi agent phát hiện đúng một pattern tấn công
        chưa thấy trước đó → khuyến khích generalization thay vì memorization

Ý nghĩa thiết kế:
  - TP/TN phụ thuộc vào cả action VÀ true_label:
      * TP: action=1 (attack) AND true_label=1 (attack)  → agent phát hiện đúng attack
      * TN: action=0 (normal) AND true_label=0 (normal)  → agent phân loại đúng normal
      * FP: action=1 (attack) AND true_label=0 (normal)  → agent cảnh báo sai
      * FN: action=0 (normal) AND true_label=1 (attack)  → agent bỏ sót attack
  - Mỗi sample rơi vào đúng 1 trong 4 trường hợp trên
  - Novelty bonus chỉ áp dụng khi TP (phát hiện đúng attack mới)
============================================================================
"""

import numpy as np
from collections import defaultdict
import hashlib


class RewardFunction:
    """
    Enhanced Reward Function for IDS agent.

    Thay thế hàm reward đơn giản (±1) bằng hàm reward có cấu trúc,
    cho phép điều chỉnh trọng số theo ngữ cảnh bảo mật.

    R(t) = α·TP − β·FP − γ·FN + δ·(1 − latency) + ε·novelty_bonus
    """

    def __init__(self, alpha=1.0, beta=0.5, gamma_fn=2.0, delta=0.0,
                 epsilon_nov=0.3, tn_reward=0.2, use_novelty=True,
                 novelty_hash_bits=16):
        """
        Initialize reward function.

        Args:
            alpha: Trọng số thưởng cho True Positive (phát hiện đúng attack)
                   Mặc định = 1.0 (thưởng tiêu chuẩn)
            beta: Trọng số phạt cho False Positive (cảnh báo sai)
                  Mặc định = 0.5 (phạt vừa phải)
                  Giảm beta nếu môi trường chấp nhận nhiều FP
            gamma_fn: Trọng số phạt cho False Negative (bỏ sót attack)
                      Mặc định = 2.0 (phạt nặng hơn FP vì bỏ sót attack
                      nguy hiểm hơn cảnh báo sai)
                      TĂNG giá trị này cho môi trường nhạy cảm
                      (bệnh viện, ngân hàng, hạ tầng quan trọng)
            delta: Trọng số cho latency bonus
                   Mặc định = 0.0 (tắt, vì IDS batch mode không có latency)
                   Bật (>0) nếu chạy real-time và muốn thưởng quyết định nhanh
            epsilon_nov: Trọng số cho novelty bonus
                         Mặc định = 0.3
                         Thưởng thêm khi phát hiện đúng pattern attack mới
                         → khuyến khích generalization thay vì memorization
            tn_reward: Trọng số thưởng cho True Negative (phân loại đúng normal)
                       Mặc định = 0.2
                       Thưởng nhẹ khi agent phân loại đúng traffic bình thường.
                       Giá trị nhỏ hơn alpha vì phát hiện attack khó hơn.
                       Đặt = 0 nếu muốn dùng công thức gốc không có TN reward.
            use_novelty: Có sử dụng novelty bonus hay không
            novelty_hash_bits: Số bit cho hash fingerprint của state
                               (dùng để xác định pattern đã thấy chưa)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma_fn = gamma_fn
        self.delta = delta
        self.epsilon_nov = epsilon_nov
        self.tn_reward = tn_reward
        self.use_novelty = use_novelty

        # ================================================================
        # Novelty tracking: theo dõi các pattern đã thấy
        # Sử dụng hashing để nhận diện pattern tương tự
        # Khi agent phát hiện đúng một pattern attack chưa thấy trước đó,
        # nó nhận thêm novelty bonus → khuyến khích khám phá
        # ================================================================
        self.novelty_hash_bits = novelty_hash_bits
        # Tập hợp các hash fingerprint đã thấy (cho mỗi lớp)
        self._seen_attack_hashes = set()
        # Đếm số lần mỗi hash xuất hiện để giảm dần novelty bonus
        self._hash_visit_counts = defaultdict(int)

    def compute(self, action, true_label, state=None, latency=None):
        """
        Tính reward cho một quyết định phân loại.

        Args:
            action: Hành động agent chọn (0=normal, 1=attack)
            true_label: Nhãn thực tế (0=normal, 1=attack)
            state: Vector đặc trưng gốc (dùng cho novelty detection)
            latency: Thời gian quyết định chuẩn hóa [0,1] (tùy chọn)
        Returns:
            reward: Giá trị reward tổng hợp (float)
        """
        reward = 0.0

        if action == 1 and true_label == 1:
            # ============================================================
            # TRUE POSITIVE: Agent phát hiện đúng tấn công
            # Đây là kết quả tốt nhất → thưởng cao
            # R += α · 1.0
            # ============================================================
            reward += self.alpha * 1.0

            # Cộng thêm novelty bonus nếu đây là pattern attack mới
            if self.use_novelty and state is not None:
                novelty = self._compute_novelty_bonus(state)
                reward += self.epsilon_nov * novelty

        elif action == 0 and true_label == 0:
            # ============================================================
            # TRUE NEGATIVE: Agent phân loại đúng traffic bình thường
            # Cũng là kết quả tốt, nhưng thưởng ít hơn TP
            # (vì phát hiện attack khó hơn phân loại normal)
            # tn_reward được config riêng, mặc định = 0.2
            # ============================================================
            reward += self.tn_reward

        elif action == 1 and true_label == 0:
            # ============================================================
            # FALSE POSITIVE: Agent cảnh báo sai (normal → attack)
            # Phạt: R -= β · 1.0
            # FP gây phiền nhiễu nhưng ít nguy hiểm hơn FN
            # → β thường < γ
            # ============================================================
            reward -= self.beta * 1.0

        elif action == 0 and true_label == 1:
            # ============================================================
            # FALSE NEGATIVE: Agent bỏ sót tấn công (attack → normal)
            # Phạt NẶNG: R -= γ · 1.0
            # Đây là trường hợp nguy hiểm nhất trong IDS:
            # tấn công lọt qua mà không bị phát hiện
            # → γ thường > β (mặc định γ=2.0 > β=0.5)
            #
            # Trong môi trường nhạy cảm (bệnh viện, ngân hàng):
            # tăng γ lên để agent ưu tiên recall cao
            # ============================================================
            reward -= self.gamma_fn * 1.0

        # ================================================================
        # Latency bonus (tùy chọn, mặc định tắt)
        # R += δ · (1 - normalized_latency)
        # Trong chế độ batch, không có latency nên delta=0
        # Nếu triển khai real-time, latency = thời gian xử lý / max_time
        # ================================================================
        if self.delta > 0 and latency is not None:
            reward += self.delta * (1.0 - np.clip(latency, 0.0, 1.0))

        return reward

    def _compute_novelty_bonus(self, state):
        """
        Tính novelty bonus cho một state (pattern tấn công).

        Ý tưởng: Nếu agent phát hiện đúng một pattern tấn công mà nó
        chưa từng thấy trước đó → thưởng thêm để khuyến khích
        generalization (học khái quát) thay vì memorization (nhớ mẫu).

        Cách hoạt động:
        1. Tạo fingerprint (hash) cho state vector
        2. Kiểm tra hash đã xuất hiện trong tập đã thấy chưa
        3. Nếu chưa thấy: novelty = 1.0 (bonus tối đa)
        4. Nếu đã thấy n lần: novelty = 1/(1+n) (giảm dần)
           → Các pattern quen thuộc vẫn nhận bonus nhỏ

        Args:
            state: numpy array - vector đặc trưng của sample
        Returns:
            novelty: float trong [0, 1], bonus cho pattern mới
        """
        # Tạo fingerprint bằng cách discretize state vector rồi hash
        # Discretize: chia mỗi feature thành 2^bits bucket
        # Điều này nhóm các state "tương tự" vào cùng hash
        discretized = np.round(state * (2 ** self.novelty_hash_bits)).astype(np.int32)
        state_hash = hashlib.md5(discretized.tobytes()).hexdigest()

        # Đếm số lần đã thấy pattern này
        visit_count = self._hash_visit_counts[state_hash]

        # Cập nhật bộ đếm
        self._hash_visit_counts[state_hash] = visit_count + 1
        self._seen_attack_hashes.add(state_hash)

        # Novelty bonus giảm dần theo số lần thấy
        # Lần đầu (visit=0): bonus = 1.0
        # Lần 2 (visit=1): bonus = 0.5
        # Lần 3 (visit=2): bonus = 0.333
        # ...
        novelty = 1.0 / (1.0 + visit_count)

        return novelty

    def reset_novelty(self):
        """
        Reset novelty tracker.
        Gọi khi bắt đầu một federated round mới để đánh giá lại novelty.
        """
        self._seen_attack_hashes.clear()
        self._hash_visit_counts.clear()

    def get_stats(self):
        """
        Trả về thống kê novelty (để logging/debug).

        Returns:
            dict: {'unique_patterns': số pattern unique đã thấy,
                   'total_visits': tổng số lần truy cập}
        """
        return {
            'unique_attack_patterns': len(self._seen_attack_hashes),
            'total_visits': sum(self._hash_visit_counts.values()),
        }

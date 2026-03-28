"""
PPO Agent Module — Proximal Policy Optimization cho IDS.
============================================================================
Reference chính:
  - "Deep Reinforcement Learning for Wireless Communications and Networking"
    Section 3.4.3, Algorithm 3.8, Pages 72-73
  - Schulman et al., "Proximal Policy Optimization Algorithms", arXiv 2017

Tổng quan Algorithm 3.8 (PPO pseudo algorithm):
  1: Khởi tạo actor π_θ và critic V_φ ngẫu nhiên.
  2: while chưa hội tụ do
  3:   for N actor song song do
  4:     Thu thập T transitions bằng old policy π_θ_old.
  5:     Tính Generalized Advantage cho mỗi transition bằng critic.
  6:   end for
  7:   for K epochs do
  8:     Lấy mẫu M transitions từ dữ liệu đã thu thập.
  9:     Huấn luyện actor: maximize clipped surrogate objective.
  10:    Huấn luyện critic: minimize MSE bằng TD learning.
  11:  end for
  12:  θ_old ← θ.
  13: end while

Thích ứng PPO cho bài toán IDS (phân loại nhị phân):
  - Mỗi sample network traffic là một "state" độc lập (single-step episode)
  - Hành động: phân loại (0=Normal, 1=Attack) — discrete action space
  - Không có next state: advantage A(s,a) = R(s,a) - V(s)
  - PPO thu thập batch lớn rồi cập nhật nhiều epochs → ổn định hơn DQN

So sánh với DQN (Algorithm 1, bài báo gốc):
  - DQN: Cập nhật weight sau MỖI sample → noisy gradients
  - PPO: Thu thập TOÀN BỘ episode → tính advantage → cập nhật K epochs
         → gradient ổn định hơn, hội tụ tốt hơn
  - DQN: Epsilon-greedy exploration → phụ thuộc hyperparameter ε
  - PPO: Stochastic policy tự khám phá → exploration tự nhiên
  - DQN: Không có constraint cho policy update → có thể không ổn định
  - PPO: Clipped objective giới hạn mức thay đổi policy → rất ổn định

Tích hợp Federated Learning:
  PPOAgent export cùng interface với DQNAgent (get_weights, set_weights,
  evaluate, train_episode) để tích hợp seamless với FederatedOrchestrator
  và CentralServer (Algorithm 2 của bài báo gốc).
============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

from src.models.ppo_network import ActorCriticNetwork
from src.reinforcement_learning.reward import RewardFunction
from src.utils.metrics import compute_metrics


class RolloutBuffer:
    """
    Buffer lưu trữ trajectory (chuỗi transitions) cho PPO.

    Reference: Algorithm 3.8, Step 4 —
    "Collect T transitions using old policy π_θ_old."

    Trong PPO, ta thu thập TOÀN BỘ trajectory trước khi cập nhật policy.
    Điều này khác với DQN, nơi mỗi sample được xử lý ngay lập tức.

    Buffer lưu cho mỗi transition:
      - state: vector đặc trưng
      - action: hành động đã chọn (0 hoặc 1)
      - log_prob: log π_θ_old(a|s) — xác suất log của hành động dưới OLD policy
                  Cần lưu để tính importance sampling ratio ρ_t sau
      - reward: phần thưởng nhận được
      - value: V(s) ước lượng từ critic tại thời điểm thu thập
      - advantage: A(s,a) = R - V(s), tính SAU khi thu thập xong toàn bộ
    """

    def __init__(self):
        """Khởi tạo buffer rỗng cho một trajectory mới."""
        self.states = []       # Danh sách state vectors
        self.actions = []      # Danh sách actions đã thực hiện
        self.log_probs = []    # Danh sách log probabilities dưới old policy
        self.rewards = []      # Danh sách rewards nhận được
        self.values = []       # Danh sách V(s) từ critic
        self.advantages = []   # Danh sách advantages (tính sau)
        self.returns = []      # Danh sách returns = reward (single step)

    def store(self, state, action, log_prob, reward, value):
        """
        Lưu một transition vào buffer.

        Gọi trong quá trình thu thập trajectory (Algorithm 3.8, Step 4).

        Args:
            state: numpy array — vector đặc trưng
            action: int — hành động (0 hoặc 1)
            log_prob: float — log π_θ_old(a|s)
            reward: float — phần thưởng
            value: float — V(s) từ critic
        """
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)

    def compute_advantages(self, gamma=0.99):
        """
        Tính advantage cho mỗi transition trong buffer.

        Reference: Algorithm 3.8, Step 5 —
        "Compute the generalized advantage of each transition using the critic."

        Trong bài toán IDS (single-step episodes), mỗi sample là INDEPENDENT.
        Không có chuỗi states liên tiếp → GAE đơn giản thành:

          Return    = reward (vì không có next state)
          Advantage = Return - V(s) = reward - V(s)

        Advantage cho biết hành động tốt hơn (+) hay tệ hơn (-) so với
        kỳ vọng trung bình V(s) của critic. Đây là tín hiệu giúp actor
        biết nên tăng hay giảm xác suất của hành động đó.

        Sau khi tính, advantages được NORMALIZE (zero mean, unit std) để:
        - Giảm variance của gradient estimate
        - Tăng tính ổn định khi training
        - Đảm bảo không bị ảnh hưởng bởi scale của reward

        Args:
            gamma: Discount factor (không used trong single-step,
                   giữ lại cho khả năng mở rộng)
        """
        n = len(self.rewards)

        # Tính returns — trong single-step: return = reward
        # (Nếu mở rộng sang multi-step, đây sẽ là discounted cumulative return)
        self.returns = list(self.rewards)

        # Tính raw advantage: A(s,a) = R - V(s)
        # A > 0: hành động tốt hơn kỳ vọng → tăng probability
        # A < 0: hành động tệ hơn kỳ vọng → giảm probability
        self.advantages = [
            self.returns[i] - self.values[i] for i in range(n)
        ]

        # Normalize advantages → zero mean, unit standard deviation
        # Điều này rất quan trọng cho tính ổn định training:
        # - Tránh gradient quá lớn/nhỏ
        # - Đảm bảo ~50% transitions có advantage > 0 và ~50% < 0
        adv_array = np.array(self.advantages)
        adv_mean = adv_array.mean()
        adv_std = adv_array.std() + 1e-8  # +epsilon tránh chia 0

        self.advantages = ((adv_array - adv_mean) / adv_std).tolist()

    def get_batches(self, mini_batch_size):
        """
        Chia buffer thành các mini-batches cho PPO update.

        Reference: Algorithm 3.8, Step 8 —
        "Sample M transitions from the previously collected."

        Thay vì dùng toàn bộ buffer cùng lúc, chia thành mini-batches
        giúp GPU xử lý hiệu quả hơn và tạo ra noise tốt cho SGD.

        Args:
            mini_batch_size: Kích thước mỗi mini-batch
        Yields:
            Tuple (states, actions, log_probs, advantages, returns) cho mỗi batch
        """
        n = len(self.states)

        # Tạo indices ngẫu nhiên và chia thành mini-batches
        indices = np.random.permutation(n)

        for start in range(0, n, mini_batch_size):
            end = min(start + mini_batch_size, n)
            batch_idx = indices[start:end]

            yield (
                np.array([self.states[i] for i in batch_idx]),      # states
                np.array([self.actions[i] for i in batch_idx]),      # actions
                np.array([self.log_probs[i] for i in batch_idx]),    # old log probs
                np.array([self.advantages[i] for i in batch_idx]),   # advantages
                np.array([self.returns[i] for i in batch_idx]),      # returns
            )

    def clear(self):
        """Xóa buffer sau khi đã cập nhật xong — chuẩn bị cho trajectory mới."""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.advantages.clear()
        self.returns.clear()

    def __len__(self):
        """Số transitions hiện có trong buffer."""
        return len(self.states)


class PPOAgent:
    """
    PPO Agent cho Intrusion Detection — thay thế DQN Agent.

    Reference: Algorithm 3.8 (Page 73) — PPO pseudo algorithm

    So sánh interface với DQNAgent:
      ✓ train_episode(X, y) → (loss, accuracy)
      ✓ evaluate(X, y) → metrics dict
      ✓ get_weights() → state_dict
      ✓ set_weights(weights) → None
      ✓ get_num_samples_trained() → int
      ✓ reset_round_counter() → None

    Cải tiến so với DQN:
      1. Actor-Critic architecture thay cho value-based only
      2. Clipped surrogate objective → ổn định hơn
      3. Stochastic policy → exploration tự nhiên
      4. Multiple epochs update trên cùng data → hiệu quả hơn
      5. Advanced reward function (R = α·TP − β·FP − γ·FN + novelty)
    """

    def __init__(self, agent_id, input_dim, hidden_layers=None, num_actions=2,
                 lr=3e-4, gamma=0.99, clip_epsilon=0.2, ppo_epochs=4,
                 mini_batch_size=64, value_coef=0.5, entropy_coef=0.01,
                 max_grad_norm=0.5, dropout=0.1,
                 reward_alpha=1.0, reward_beta=0.5, reward_gamma_fn=2.0,
                 reward_delta=0.0, reward_epsilon_nov=0.3, reward_tn=0.2,
                 device='cpu'):
        """
        Khởi tạo PPO Agent.

        Reference: Algorithm 3.8, Step 1 —
        "Initialize randomly the actor π_θ and the critic V_φ."

        Args:
            agent_id: ID duy nhất cho agent (trong hệ thống federated)
            input_dim: Chiều đặc trưng đầu vào (sau DAE encoding)
            hidden_layers: Kích thước hidden layers [128, 64, 32]
            num_actions: Số hành động rời rạc (2: normal/attack)

            --- PPO Hyperparameters ---
            lr: Learning rate cho cả actor và critic.
                PPO thường dùng LR nhỏ hơn DQN (3e-4 vs 1e-3)
                vì cập nhật trên toàn bộ batch thay vì từng sample.
            gamma: Discount factor. Trong single-step IDS, gamma không
                   ảnh hưởng trực tiếp (giữ lại cho mở rộng).
            clip_epsilon: Tham số clipping ε trong PPO objective.
                         Eq. 3.96 (Page 72): clip(ρ_t, 1-ε, 1+ε)
                         ε=0.2 → cho phép policy thay đổi tối đa ±20%
                         mỗi lần update. Giá trị nhỏ → update thận trọng.
            ppo_epochs: Số epochs K cập nhật trên mỗi batch trajectory.
                       Algorithm 3.8, Step 7: "for K epochs do"
                       K=4 là giá trị phổ biến. Nhiều epochs → sử dụng
                       data hiệu quả hơn, nhưng quá nhiều → overfitting.
            mini_batch_size: Kích thước mini-batch cho mỗi PPO update.
                            Algorithm 3.8, Step 8: "Sample M transitions"
            value_coef: Hệ số c₁ cho value loss trong total loss.
                       Total loss = actor_loss + c₁·value_loss - c₂·entropy
                       c₁=0.5: critic được train chậm hơn actor 50%.
            entropy_coef: Hệ số c₂ cho entropy bonus.
                         Entropy bonus khuyến khích exploration bằng cách
                         thưởng khi policy phân phối đều (uncertain).
                         c₂=0.01: entropy bonus nhỏ, chỉ đủ để tránh
                         policy suy biến thành deterministic quá sớm.
            max_grad_norm: Giới hạn gradient norm để tránh exploding gradients.
                          Gradient clipping giúp training ổn định.

            --- Advanced Reward Parameters ---
            reward_alpha: Trọng số TP thưởng
            reward_beta: Trọng số FP phạt
            reward_gamma_fn: Trọng số FN phạt (NẶNG NHẤT)
            reward_delta: Trọng số latency bonus
            reward_epsilon_nov: Trọng số novelty bonus
            reward_tn: Trọng số thưởng cho TN (phân loại đúng normal), mặc định 0.2

            device: Thiết bị tính toán ('cpu' hoặc 'cuda')
        """
        self.agent_id = agent_id
        self.device = torch.device(device)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_actions = num_actions

        # ================================================================
        # ACTOR-CRITIC NETWORK — Thay thế DQN
        # Reference: Algorithm 3.8, Step 1:
        # "Initialize randomly the actor π_θ and the critic V_φ."
        # Actor (policy) và Critic (value) chia sẻ backbone
        # ================================================================
        self.network = ActorCriticNetwork(
            input_dim=input_dim,
            hidden_layers=hidden_layers or [128, 64, 32],
            num_actions=num_actions,
            dropout=dropout
        ).to(self.device)

        # Optimizer duy nhất cho toàn bộ network (shared backbone + 2 heads)
        # PPO thường dùng Adam optimizer với learning rate nhỏ
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # ================================================================
        # ADVANCED REWARD FUNCTION — Thay thế reward ±1 đơn giản
        # R(t) = α·TP − β·FP − γ·FN + δ·(1-latency) + ε·novelty_bonus
        # ================================================================
        self.reward_fn = RewardFunction(
            alpha=reward_alpha,
            beta=reward_beta,
            gamma_fn=reward_gamma_fn,
            delta=reward_delta,
            epsilon_nov=reward_epsilon_nov,
            tn_reward=reward_tn,
            use_novelty=True
        )

        # ================================================================
        # ROLLOUT BUFFER — Lưu trajectory cho PPO update
        # Khác với DQN sử dụng PER (Prioritized Experience Replay),
        # PPO sử dụng on-policy data: chỉ dùng data từ policy HIỆN TẠI,
        # sau khi update xong thì XÓA và thu thập data mới.
        # ================================================================
        self.rollout_buffer = RolloutBuffer()

        # ================================================================
        # Thống kê training — tương thích interface DQNAgent
        # ================================================================
        self.episode_losses = []
        self.episode_accuracies = []
        self.num_samples_trained = 0

    def train_episode(self, X, y):
        """
        Huấn luyện một episode với PPO.

        Quy trình 2 giai đoạn (theo Algorithm 3.8):
          Phase 1 — Thu thập trajectory (Steps 3-6):
            Duyệt qua toàn bộ training data, với mỗi sample:
            - Lấy action từ actor (sampling từ phân phối)
            - Tính reward bằng Advanced Reward Function
            - Lưu (state, action, log_prob, reward, value) vào buffer
          Phase 2 — PPO Update (Steps 7-12):
            - Tính advantages: A(s,a) = R - V(s)
            - For K epochs:
              - Chia buffer thành mini-batches
              - Tính importance sampling ratio ρ_t
              - Tính clipped surrogate loss
              - Tính value loss + entropy bonus
              - Cập nhật network

        Args:
            X: Feature vectors [N, input_dim] — numpy array
            y: Labels [N] — numpy array (0=Normal, 1=Attack)
        Returns:
            avg_loss: Loss trung bình cho episode
            accuracy: Accuracy cho episode
        """
        self.network.train()

        # ================================================================
        # PHASE 1: THU THẬP TRAJECTORY
        # Reference: Algorithm 3.8, Steps 3-6
        # "for N actors in parallel do
        #    Collect T transitions using old policy π_θ_old.
        #    Compute generalized advantage using the critic."
        #
        # Trong bối cảnh IDS: mỗi "transition" = một sample traffic
        # N=1 actor (single agent), T = toàn bộ dataset
        # ================================================================
        self.rollout_buffer.clear()  # Xóa buffer cũ (on-policy: chỉ dùng data mới)

        # Shuffle dữ liệu để randomize thứ tự training
        # (Tương đương Algorithm 1, Line 6 trong bài gốc)
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        correct_predictions = 0
        n_samples = len(X)

        # --- Thu thập trajectory: duyệt qua mỗi sample ---
        for j in range(n_samples):
            # Chuyển state vector sang tensor PyTorch
            state = torch.FloatTensor(X_shuffled[j]).unsqueeze(0).to(self.device)
            true_label = int(y_shuffled[j])

            # Lấy action từ actor bằng stochastic sampling
            # Khác với DQN dùng epsilon-greedy, PPO SAMPLE từ phân phối
            # π_θ(a|s) → exploration tự nhiên, không cần hyperparameter ε
            with torch.no_grad():
                action, log_prob, value = self.network.get_action(state)

            # Tính reward bằng Advanced Reward Function
            # R(t) = α·TP − β·FP − γ·FN + δ·(1-latency) + ε·novelty
            reward = self.reward_fn.compute(
                action=action,
                true_label=true_label,
                state=X_shuffled[j]  # State gốc cho novelty detection
            )

            # Đếm correct predictions cho accuracy
            if action == true_label:
                correct_predictions += 1

            # Lưu transition vào rollout buffer
            self.rollout_buffer.store(
                state=X_shuffled[j].copy(),
                action=action,
                log_prob=log_prob.item(),      # Scalar log probability
                reward=reward,
                value=value.item()             # Scalar state value
            )

        # ================================================================
        # PHASE 2: PPO UPDATE
        # Reference: Algorithm 3.8, Steps 5-12
        #
        # Step 5: "Compute the generalized advantage"
        # Steps 7-11: K epochs of clipped updates
        # Step 12: "θ_old ← θ" (implicit: weights updated in-place)
        # ================================================================

        # --- Step 5: Tính advantages ---
        # A(s,a) = R - V(s) + normalize
        self.rollout_buffer.compute_advantages(gamma=self.gamma)

        # --- Steps 7-11: K epochs of PPO updates ---
        total_loss = self._ppo_update()

        # Cập nhật thống kê
        accuracy = correct_predictions / n_samples
        self.num_samples_trained += n_samples
        self.episode_losses.append(total_loss)
        self.episode_accuracies.append(accuracy)

        return total_loss, accuracy

    def _ppo_update(self):
        """
        Thực hiện K epochs cập nhật PPO trên rollout buffer.

        Reference: Algorithm 3.8, Steps 7-11:
        "for K epochs do
           Sample M transitions from the previously collected.
           Train the actor to maximize the clipped surrogate objective.
           Train the critic to minimize the MSE using TD learning.
         end for"

        PPO Loss tổng hợp:
          L_total = L_actor + c₁·L_critic - c₂·H(π)
        Trong đó:
          L_actor = -min(ρ·A, clip(ρ, 1-ε, 1+ε)·A)  [clipped surrogate]
          L_critic = MSE(V(s), Return)                [value prediction error]
          H(π) = -Σ π(a|s)·log π(a|s)                [entropy bonus]

        Returns:
            avg_loss: Loss trung bình trên toàn bộ K epochs
        """
        total_loss_sum = 0.0
        total_batches = 0

        # ================================================================
        # Loop qua K epochs (mỗi epoch duyệt toàn bộ buffer)
        # Nhiều epochs cho phép sử dụng data hiệu quả hơn:
        # cùng một trajectory được cập nhật nhiều lần.
        # Clipping đảm bảo policy không thay đổi quá nhiều.
        # ================================================================
        for epoch in range(self.ppo_epochs):

            # Chia buffer thành mini-batches ngẫu nhiên
            for batch in self.rollout_buffer.get_batches(self.mini_batch_size):
                states_np, actions_np, old_log_probs_np, advantages_np, returns_np = batch

                # Chuyển numpy arrays sang PyTorch tensors
                states = torch.FloatTensor(states_np).to(self.device)
                actions = torch.LongTensor(actions_np).to(self.device)
                old_log_probs = torch.FloatTensor(old_log_probs_np).to(self.device)
                advantages = torch.FloatTensor(advantages_np).to(self.device)
                returns = torch.FloatTensor(returns_np).to(self.device)

                # ========================================================
                # ĐÁNH GIÁ LẠI actions dưới policy MỚI (đã cập nhật)
                # Cần tính lại log_prob và value vì network đã thay đổi
                # qua các bước cập nhật trước đó trong epoch này.
                # ========================================================
                new_log_probs, state_values, entropy = self.network.evaluate_actions(
                    states, actions
                )
                # Squeeze state_values từ [batch, 1] → [batch]
                state_values = state_values.squeeze(-1)

                # ========================================================
                # TÍNH IMPORTANCE SAMPLING RATIO ρ_t
                # Eq. 3.95 (Page 72): ρ_t = π_θ(a|s) / π_θ_old(a|s)
                #
                # Trong log space: log(ρ_t) = log π_new - log π_old
                # → ρ_t = exp(log π_new - log π_old)
                #
                # ρ_t cho biết mức độ policy MỚI khác policy CŨ:
                # - ρ ≈ 1: policy gần giống nhau
                # - ρ > 1: hành động này xác suất CAO hơn dưới policy mới
                # - ρ < 1: hành động này xác suất THẤP hơn dưới policy mới
                # ========================================================
                ratio = torch.exp(new_log_probs - old_log_probs)

                # ========================================================
                # CLIPPED SURROGATE OBJECTIVE — Mấu chốt của PPO
                # Eq. 3.96 (Page 72):
                # L_CLIP = E[min(ρ·A, clip(ρ, 1-ε, 1+ε)·A)]
                #
                # Hai thành phần:
                # 1) surrogate1 = ρ·A — objective TRPO không giới hạn
                # 2) surrogate2 = clip(ρ, 1-ε, 1+ε)·A — bị giới hạn
                #
                # Lấy min() của hai thành phần:
                # - Khi A > 0 (hành động tốt): ρ bị giới hạn trên ≤ 1+ε
                #   → Tránh tăng xác suất quá mạnh (Hình 3.8a, Page 72)
                # - Khi A < 0 (hành động xấu): ρ bị giới hạn dưới ≥ 1-ε
                #   → Tránh giảm xác suất quá mạnh (Hình 3.8b, Page 72)
                #
                # Kết quả: policy thay đổi VỪA PHẢI mỗi lần update
                # → training RẤT ỔN ĐỊNH so với vanilla policy gradient hoặc DQN
                # ========================================================
                surrogate1 = ratio * advantages
                surrogate2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_epsilon,     # Giới hạn dưới: 1 - ε
                    1.0 + self.clip_epsilon       # Giới hạn trên: 1 + ε
                ) * advantages

                # Actor loss = negative vì ta muốn MAXIMIZE objective
                # nhưng optimizer MINIMIZE loss → thêm dấu trừ
                actor_loss = -torch.min(surrogate1, surrogate2).mean()

                # ========================================================
                # CRITIC (VALUE) LOSS
                # Algorithm 3.8, Step 10:
                # "Train the critic to minimize the MSE using TD learning."
                #
                # L_critic = MSE(V(s), Return)
                # = (V(s) - R)²
                #
                # Critic học dự đoán return kỳ vọng từ mỗi state.
                # Khi V(s) chính xác → advantage A = R - V(s) ít noise hơn
                # → actor update ổn định hơn.
                # ========================================================
                value_loss = nn.functional.mse_loss(state_values, returns)

                # ========================================================
                # ENTROPY BONUS
                # H(π) = -Σ π(a|s)·log π(a|s)
                #
                # Entropy cao = policy phân phối đều = explorer
                # Entropy thấp = policy tập trung 1 action = exploiter
                #
                # Trừ entropy_loss trong total loss (có dấu trừ) →
                # TĂNG entropy → khuyến khích exploration
                # Tránh policy suy biến thành deterministic quá sớm
                # ========================================================
                entropy_loss = -entropy.mean()

                # ========================================================
                # TOTAL LOSS — Kết hợp 3 thành phần
                # L = L_actor + c₁·L_critic + c₂·L_entropy
                #   = actor_loss + value_coef·value_loss + entropy_coef·(-entropy)
                #
                # Mỗi thành phần đóng vai trò:
                # - actor_loss: Cải thiện policy (chọn hành động tốt hơn)
                # - value_loss: Cải thiện dự đoán value (giảm variance)
                # - entropy_loss: Duy trì exploration (tránh hội tụ sớm)
                # ========================================================
                total_loss = (
                    actor_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # ========================================================
                # BACKPROPAGATION + GRADIENT CLIPPING
                # Gradient clipping giới hạn norm của gradient vector:
                # if ||∇L|| > max_norm: ∇L = ∇L * max_norm / ||∇L||
                # Tránh exploding gradients gây training không ổn định
                # ========================================================
                self.optimizer.zero_grad()
                total_loss.backward()

                # Clip gradient norm để tránh exploding gradients
                nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.max_grad_norm
                )

                # Cập nhật parameters
                self.optimizer.step()

                total_loss_sum += total_loss.item()
                total_batches += 1

        # ================================================================
        # Algorithm 3.8, Step 12: "θ_old ← θ"
        # Trong PPO implementation, old policy được cập nhật NGẦM:
        # lần thu thập trajectory tiếp theo sẽ dùng network đã update.
        # Không cần copy riêng θ_old vì log_probs đã được lưu trong buffer.
        # ================================================================

        return total_loss_sum / max(total_batches, 1)

    def evaluate(self, X, y):
        """
        Đánh giá agent trên tập dữ liệu test.

        Tương thích interface DQNAgent.evaluate() để dùng trong
        Algorithm 3 (Dynamic Attention), bài báo gốc:
        "Compute the accuracy of the aggregated model on the available test dataset"

        Trong evaluation, dùng GREEDY policy (argmax) thay vì sampling
        để đánh giá performance tốt nhất của agent.

        Args:
            X: Feature vectors [N, input_dim]
            y: True labels [N]
        Returns:
            metrics: Dict chứa accuracy, precision, recall, F1, FPR, AUC-ROC
        """
        self.network.eval()
        predictions = []
        probs_attack = []  # Xác suất lớp attack cho ROC curve

        with torch.no_grad():
            # Batch inference để tăng tốc (thay vì xử lý từng sample)
            batch_size = 1024
            for start in range(0, len(X), batch_size):
                end = min(start + batch_size, len(X))
                states = torch.FloatTensor(X[start:end]).to(self.device)

                # Lấy action probabilities từ actor
                action_probs, _ = self.network(states)

                # Greedy: chọn action có xác suất cao nhất
                preds = action_probs.argmax(dim=1).cpu().numpy()
                predictions.extend(preds)

                # Xác suất attack (class 1) cho AUC-ROC
                probs = action_probs[:, 1].cpu().numpy()
                probs_attack.extend(probs)

        predictions = np.array(predictions)
        probs_attack = np.array(probs_attack)

        return compute_metrics(y, predictions, probs_attack)

    def get_weights(self):
        """
        Lấy trọng số network cho federated aggregation.

        Tương thích Algorithm 2 (bài báo gốc):
        "Input: Deep Q-Network weights for each of the available agents"
        Với PPO, thay "DQN weights" bằng "Actor-Critic weights".
        """
        return self.network.get_weights()

    def set_weights(self, weights):
        """
        Cập nhật trọng số network từ mô hình tổng hợp.

        Tương thích Algorithm 2, Line 8:
        "each agent will update its own Q-network with this aggregated
         network weight matrix and continue its training process"
        """
        self.network.set_weights(weights)

    def get_num_samples_trained(self):
        """Lấy số samples đã train trong round hiện tại."""
        return self.num_samples_trained

    def reset_round_counter(self):
        """
        Reset bộ đếm samples cho federated round mới.
        Đồng thời reset novelty tracker của reward function,
        vì mô hình đã được tổng hợp lại → "kiến thức mới".
        """
        self.num_samples_trained = 0
        # Reset novelty sau mỗi round: mô hình thay đổi đáng kể
        # sau aggregation, nên cần đánh giá lại novelty
        self.reward_fn.reset_novelty()

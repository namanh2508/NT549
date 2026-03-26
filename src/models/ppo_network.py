"""
Actor-Critic Network for PPO (Proximal Policy Optimization).
============================================================================
Reference: "Deep Reinforcement Learning for Wireless Communications and
           Networking" — Section 3.4.3, Algorithm 3.8, Pages 72-73
============================================================================
PPO uses an Actor-Critic architecture where:
  - Actor (Policy Network π_θ): Outputs a probability distribution over
    discrete actions. Given a state s, it returns π_θ(a|s) for each action a.
  - Critic (Value Network V_φ): Estimates the state-value function V(s),
    predicting the expected return from state s.

Architecture Design:
  - A shared feature extractor backbone processes the input state vector.
    Sharing layers between actor and critic reduces parameters and allows
    features learned by the critic (about state value) to benefit the actor
    (about which action to take), and vice versa.
  - Two separate heads branch from the shared backbone:
    * Actor head:  Linear -> Softmax -> action probabilities
    * Critic head: Linear -> scalar state value

  Input: State vector (network traffic features, post-DAE encoding)
  Output Actor:  Action probabilities [batch_size, num_actions]
  Output Critic: State values [batch_size, 1]

PPO Clipped Surrogate Objective (Eq. 3.96, Page 72):
  L_CLIP(θ) = E_t[ min( ρ_t * A_t, clip(ρ_t, 1-ε, 1+ε) * A_t ) ]
  where ρ_t = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) is the importance sampling ratio

This network enables both discrete (classification: normal/attack) and future
extension to continuous action spaces (e.g., alert severity levels).
============================================================================
"""

import torch
import torch.nn as nn
import copy


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic Network implementing parameter sharing between policy and value.

    Reference: Algorithm 3.8 (Page 73) — Step 1:
    "Initialize randomly the actor π_θ and the critic V_φ."

    In our implementation, θ and φ share the backbone parameters,
    reducing total parameter count and improving training stability.
    """

    def __init__(self, input_dim, hidden_layers=None, num_actions=2, dropout=0.1):
        """
        Initialize the Actor-Critic network.

        Args:
            input_dim: Dimension of input state vector (feature dimension after DAE)
            hidden_layers: List of hidden layer sizes for the shared backbone,
                           e.g. [128, 64, 32]. Defaults to [128, 64, 32].
            num_actions: Number of discrete actions (2 for IDS: normal/attack)
            dropout: Dropout rate for regularization in hidden layers
        """
        super(ActorCriticNetwork, self).__init__()

        # Sử dụng hidden layers mặc định nếu không được cung cấp
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]

        # ================================================================
        # SHARED BACKBONE — Feature extractor chung cho cả Actor và Critic.
        # Các layer ẩn dùng chung giúp giảm số lượng tham số và cho phép
        # Actor và Critic chia sẻ knowledge về biểu diễn đặc trưng.
        # ================================================================
        backbone_layers = []
        prev_dim = input_dim

        for h_dim in hidden_layers:
            # Linear layer: biến đổi tuyến tính input
            backbone_layers.append(nn.Linear(prev_dim, h_dim))
            # ReLU activation: tạo phi tuyến tính cần thiết cho deep learning
            backbone_layers.append(nn.ReLU())
            # Dropout: ngẫu nhiên tắt một số neuron khi training để chống overfitting
            backbone_layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # Gói các layer thành một nn.Sequential module
        self.shared_backbone = nn.Sequential(*backbone_layers)

        # ================================================================
        # ACTOR HEAD — Policy network π_θ(a|s)
        # Đầu ra là xác suất cho mỗi hành động thông qua Softmax.
        # Softmax đảm bảo tổng xác suất = 1, phù hợp cho phân phối xác suất.
        # Reference: Algorithm 3.8 — Actor quyết định hành động dựa trên
        # phân phối xác suất, cho phép khám phá (exploration) tự nhiên
        # thông qua stochastic sampling thay vì epsilon-greedy.
        # ================================================================
        self.actor_head = nn.Sequential(
            nn.Linear(prev_dim, num_actions),  # Map từ hidden dim đến số actions
            nn.Softmax(dim=-1)                 # Chuyển logits thành xác suất
        )

        # ================================================================
        # CRITIC HEAD — Value network V_φ(s)
        # Đầu ra là một scalar duy nhất ước lượng giá trị của state hiện tại.
        # V(s) = Expected return bắt đầu từ state s theo policy hiện tại.
        # Critic dùng để tính advantage A(s,a) = R - V(s), giúp giảm
        # variance của policy gradient estimate.
        # ================================================================
        self.critic_head = nn.Sequential(
            nn.Linear(prev_dim, 1)  # Map từ hidden dim đến 1 giá trị scalar
        )

        # Lưu số lượng actions để sử dụng sau
        self.num_actions = num_actions

    def forward(self, state):
        """
        Forward pass qua toàn bộ Actor-Critic network.

        Reference: Algorithm 3.8 — Steps 4-5:
        Actor tạo phân phối hành động, Critic ước lượng giá trị state.

        Args:
            state: Tensor trạng thái [batch_size, input_dim]
        Returns:
            action_probs: Xác suất hành động [batch_size, num_actions]
            state_value:  Giá trị trạng thái [batch_size, 1]
        """
        # Truyền state qua shared backbone để trích xuất đặc trưng chung
        shared_features = self.shared_backbone(state)

        # Actor head: tính xác suất hành động π_θ(a|s)
        action_probs = self.actor_head(shared_features)

        # Critic head: ước lượng giá trị V_φ(s)
        state_value = self.critic_head(shared_features)

        return action_probs, state_value

    def get_action(self, state):
        """
        Chọn hành động bằng cách sampling từ phân phối xác suất của Actor.

        Khác với DQN sử dụng epsilon-greedy (chọn hành động tốt nhất với
        xác suất 1-ε, ngẫu nhiên với ε), PPO tự nhiên khám phá thông qua
        stochastic sampling từ π_θ(a|s). Điều này cho phép exploration
        mượt mà hơn mà không cần hyperparameter ε.

        Reference: Algorithm 3.8, Step 4 —
        "Collect T transitions using old policy π_θ_old."

        Args:
            state: Tensor trạng thái [1, input_dim] (single sample)
        Returns:
            action: Hành động được chọn (0=Normal, 1=Attack)
            log_prob: Log probability của hành động đã chọn, log π_θ(a|s).
                      Cần lưu lại để tính importance sampling ratio ρ_t sau này.
            value: Giá trị V(s) từ Critic, dùng để tính advantage.
        """
        # Forward pass để lấy phân phối hành động và giá trị state
        action_probs, state_value = self.forward(state)

        # Tạo phân phối Categorical từ xác suất hành động
        # Categorical distribution cho phép sampling discrete actions
        dist = torch.distributions.Categorical(action_probs)

        # Sample một hành động từ phân phối — đây là cách PPO khám phá
        action = dist.sample()

        # Tính log probability: log π_θ(a|s)
        # Cần lưu log_prob để sau này tính ratio ρ_t = π_new / π_old
        log_prob = dist.log_prob(action)

        return action.item(), log_prob, state_value

    def evaluate_actions(self, states, actions):
        """
        Đánh giá lại các hành động đã thực hiện dưới policy hiện tại.

        Đây là bước quan trọng trong PPO update (Algorithm 3.8, Steps 8-10).
        Khi cập nhật network, ta cần tính lại log_prob và value cho các
        transitions đã thu thập (dưới old policy), nhưng bây giờ dùng
        policy mới (đã cập nhật). Điều này cho phép tính importance
        sampling ratio ρ_t = π_θ_new(a|s) / π_θ_old(a|s).

        Args:
            states: Batch states [batch_size, input_dim]
            actions: Batch actions đã thực hiện [batch_size]
        Returns:
            log_probs: Log π_θ(a|s) theo policy mới [batch_size]
            state_values: V(s) theo critic mới [batch_size, 1]
            entropy: Entropy của phân phối hành động, đo mức độ "ngẫu nhiên"
                     của policy. Entropy cao = khám phá nhiều. Thêm entropy
                     bonus vào loss function khuyến khích policy không suy biến
                     thành deterministic quá sớm (tránh premature convergence).
        """
        # Forward pass để lấy action probabilities và state values mới
        action_probs, state_values = self.forward(states)

        # Tạo phân phối Categorical từ xác suất mới
        dist = torch.distributions.Categorical(action_probs)

        # Tính log probability cho CÁC HÀNH ĐỘNG ĐÃ THỰC HIỆN (từ old policy)
        # nhưng đánh giá bằng policy MỚI
        log_probs = dist.log_prob(actions)

        # Tính entropy: H(π) = -Σ π(a|s) * log π(a|s)
        # Entropy cao → policy phân phối đều → khám phá nhiều
        # Entropy thấp → policy tập trung vào 1 action → khai thác (exploit)
        entropy = dist.entropy()

        return log_probs, state_values, entropy

    def get_weights(self):
        """
        Lấy toàn bộ trọng số của network (backbone + actor + critic).
        Dùng cho federated aggregation (Algorithm 2 của bài báo gốc).
        """
        return copy.deepcopy(self.state_dict())

    def set_weights(self, weights):
        """
        Cập nhật trọng số của network từ mô hình tổng hợp.
        Algorithm 2, Line 8: "each agent will update its own network
        with the aggregated weight matrix"
        """
        self.load_state_dict(weights)

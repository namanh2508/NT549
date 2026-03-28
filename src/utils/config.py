"""
Configuration file for FDRL-IDS.
============================================================================
Paper: "Federated reinforcement learning based intrusion detection system
        using dynamic attention mechanism"
Journal: Journal of Information Security and Applications 78 (2023) 103608
============================================================================
Hyperparameters are extracted from Section 6 (Experimental Results) of the paper.
- k and a parameters for attention: Page 8-9 (Section 6.1)
  * Random split: k=30, a=50
  * Customized split NSL-KDD: k=50000, a=200
- Number of agents: 8 agents for random split (Page 8, Section 6)
  * 2 agents for customized split (Page 9, Section 6.1)
- Deep Q-Network architecture: Section 4.1, Page 5
- Prioritized Experience Replay: Section 2.2.3, Page 3
- Epsilon-greedy: Algorithm 1, Page 6
"""

import torch

# ===========================================================================
# Device Configuration
# ===========================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================================================================
# Dataset Configuration (Section 5.1, Page 7, Table 2)
# NSL-KDD: 148,517 records, 41 features, binary classification
# ===========================================================================
DATASET_NAME = "NSL-KDD"
# Number of input features after preprocessing (41 original features,
# after one-hot encoding of categorical features the dimension increases)
# We will set this dynamically after preprocessing
NUM_FEATURES = None  # Set dynamically
NUM_ACTIONS = 2  # Binary classification: 0=Normal, 1=Attack (Section 4.1, Page 5)

# ===========================================================================
# Deep Q-Network Configuration (Section 4.1, Page 5)
# The paper uses a DQN with fully connected layers.
# Input: state vector (feature dimension), Output: Q-values for 2 actions
# ===========================================================================
DQN_HIDDEN_LAYERS = [128, 64, 32]  # Hidden layer sizes
DQN_LEARNING_RATE = 0.001
DQN_DROPOUT = 0.1

# ===========================================================================
# Denoising Autoencoder Configuration (Section 4.1, Page 5)
# "Features representations will be passed through a denoising autoencoder
#  (DAE) to protect the model from adversarial attacks"
# ===========================================================================
DAE_NOISE_FACTOR = 0.3  # Noise factor for corruption
DAE_HIDDEN_DIM = 64  # Bottleneck dimension
DAE_LEARNING_RATE = 0.001
DAE_EPOCHS = 20
DAE_BATCH_SIZE = 256

# ===========================================================================
# Reinforcement Learning Configuration (Algorithm 1, Page 6)
# ===========================================================================
# Epsilon-greedy parameters (Algorithm 1, line 2, Page 6)
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# Discount factor gamma (Eq. 2-4, Page 3)
GAMMA = 0.99

# Reward values (Section 4.1, Page 5)
# "Based on the original category of the sample from the dataset,
#  the agent obtains a reward r"
REWARD_CORRECT = 1.0
REWARD_INCORRECT = -1.0

# ===========================================================================
# Prioritized Experience Replay Configuration (Section 2.2.3, Page 3)
# "the priority for i-th sample is given by P(i) = p_i^alpha / sum(p_k^alpha)"
# ===========================================================================
MEMORY_CAPACITY = 10000
BATCH_SIZE_REPLAY = 64
PER_ALPHA = 0.6  # Prioritization factor alpha (Section 2.2.3, Page 3)
PER_BETA_START = 0.4  # Importance sampling beta (Section 2.2.3, Page 3)
PER_BETA_END = 1.0
PER_EPSILON = 1e-6  # Small constant epsilon for priority (Section 2.2.3)
OMEGA = 0.5  # Power factor for state_loss_weight (Algorithm 1, line 14, Page 6)

# ===========================================================================
# Federated Learning Configuration (Section 4, Page 4-5)
# If traing locally , uncomment the following line to set NUM_ROUNDS = 30
# ===========================================================================
# NOTE: NUM_AGENTS, NUM_ROUNDS, EPISODES_PER_ROUND are intentionally NOT
# defined here — they are set by the user in the notebook/script and must
# not be overwritten by 'from src.utils.config import *'.
# Default values for reference:
#   NUM_AGENTS = 8  (random split), 2 (customized split)
#   NUM_ROUNDS = 30
#   EPISODES_PER_ROUND = 3

# ===========================================================================
# Dynamic Attention Mechanism (Section 4.3-4.4, Eq.6, Page 6-7)
# attention_multiplier = 1 + k * (1 - accuracy) * a^(-accuracy)
# ===========================================================================
# For random split (Section 6.1, Page 8):
ATTENTION_K = 30  # Controls max attention multiplier value
ATTENTION_A = 50  # Controls speed of attention drop

# For customized split NSL-KDD (Section 6.1, Page 10):
# ATTENTION_K = 50000
# ATTENTION_A = 200

# ===========================================================================
# CIC Dataset Configuration
# ===========================================================================
# CIC-BCCC-NRC datasets generated by CICFlowMeter (79 numeric features)
CIC_DEFAULT_SUBSAMPLE_PER_FILE = 10000  # Caps rows per CSV, ~150K total
CIC_DATASET_DIRS = {
    'iomt': 'CIC-BCCC-NRC-IoMT-2024',
    'iiot': 'CIC-BCCC-NRC-Edge-IIoTSet-2022',
}

# ===========================================================================
# Training Configuration
# ===========================================================================
TRAIN_BATCH_SIZE = 256
TEST_SPLIT_RATIO = 0.2  # Fraction of agent's data used as test set for attention
RANDOM_SEED = 42

# ===========================================================================
# PPO Configuration (Improvement 1: Replace DQN with PPO)
# Reference: "Deep RL for Wireless Communications", Algorithm 3.8, Page 73
# ===========================================================================
# PPO Clipping parameter ε (Eq. 3.96, Page 72)
# Giới hạn mức thay đổi policy mỗi lần update: clip(ρ, 1-ε, 1+ε)
# ε=0.2 là giá trị chuẩn, cho phép policy thay đổi ±20%
PPO_CLIP_EPSILON = 0.2

# Số epochs K cập nhật trên mỗi batch trajectory
# Algorithm 3.8, Step 7: "for K epochs do"
# K=4 cân bằng giữa hiệu quả sử dụng data và nguy cơ overfitting
PPO_EPOCHS = 4

# Kích thước mini-batch cho PPO update
# Algorithm 3.8, Step 8: "Sample M transitions"
PPO_MINI_BATCH_SIZE = 64

# Learning rate cho PPO (thường thấp hơn DQN vì update trên batch lớn)
PPO_LEARNING_RATE = 3e-4

# Hệ số c₁ cho critic loss trong total loss
# L = L_actor + c₁·L_critic - c₂·entropy
PPO_VALUE_COEF = 0.5

# Hệ số c₂ cho entropy bonus (khuyến khích exploration)
PPO_ENTROPY_COEF = 0.01

# Giới hạn gradient norm để tránh exploding gradients
PPO_MAX_GRAD_NORM = 0.5

# ===========================================================================
# Advanced Reward Configuration (Improvement 2: Redesigned Reward Function)
# R(t) = α·TP − β·FP − γ·FN + δ·(1-latency) + ε·novelty_bonus
# ===========================================================================
# Trọng số thưởng True Positive (phát hiện đúng tấn công)
REWARD_ALPHA = 1.0

# Trọng số phạt False Positive (cảnh báo giả)
# β < γ: cảnh báo giả ít nghiêm trọng hơn bỏ lọt tấn công
REWARD_BETA = 0.5

# Trọng số phạt False Negative (bỏ lọt tấn công)
# γ > β: bỏ lọt tấn công là lỗi nghiêm trọng nhất
# Tăng giá trị này cho môi trường nhạy cảm (bệnh viện, ngân hàng)
REWARD_GAMMA_FN = 2.0

# Trọng số latency bonus (mặc định tắt cho offline training)
REWARD_DELTA = 0.0

# Trọng số novelty bonus (thưởng khi phát hiện pattern tấn công mới)
REWARD_EPSILON_NOV = 0.3

# Trọng số thưởng True Negative (phân loại đúng normal)
# Mặc định = 0.2, nhỏ hơn alpha vì phát hiện attack quan trọng hơn
REWARD_TN = 0.2

# ===========================================================================
# Aggregation Method Configuration
# Options: 'attention', 'fltrust', 'attention_fltrust', 'fedplus', 'fltrust_fedplus', 'attention_fltrust_fedplus'
# ===========================================================================
# 'attention': Original attention-weighted FedAvg (Algorithm 2, Paper)
# 'fltrust': Byzantine-resilient via cosine similarity filtering (FLTRUST.pdf)
# 'attention_fltrust': Dynamic Attention + FLTrust combined
#                = attention_values (Algorithm 3) × trust_scores (FLTrust)
# 'fedplus': Server momentum for non-IID data correction (FED+.pdf)
# 'fltrust_fedplus': FLTrust filtering + Fed+ momentum combined
# 'attention_fltrust_fedplus': FULL - Dynamic Attention + FLTrust + Fed+ combined
AGGREGATE_METHOD = 'attention'

# ===========================================================================
# FLTrust Configuration (Byzantine-resilient Aggregation)
# Reference: FLTRUST.pdf - "FLTrust: Byzantine-resilient Federated Learning
# via Trust Server" - Equations 7-8
# ===========================================================================
# Cosine similarity threshold τ (tau)
# Agents with cos(θ_i) < τ are excluded as Byzantine
# Reference: FLTRUST.pdf - default τ = 0.5
FLTRUST_TAU = 0.5

# Dimension suppression factor β (beta)
# Reference: FLTRUST.pdf, Equation 7
FLTRUST_BETA = 0.5

# Ratio of training data used as proxy dataset for trusted direction
# Reference: FLTRUST.pdf - "server has access to a small clean proxy
# dataset D_0 (e.g., 0.1% of total data)"
FLTRUST_PROXY_RATIO = 0.01  # 1% of data

# ===========================================================================
# Fed+ Configuration (Server-side Momentum for Non-IID)
# Reference: FED+.pdf - "FED+: A Unified Framework for Byzantine-Resilient
# Federated Learning" - Algorithm 1: momentum update rule
# ===========================================================================
# Momentum coefficient β
# Reference: FED+.pdf, Algorithm 1: m_{t+1} = β × m_t + Σ(n_i/n) × Δw_i
# Typical value: 0.9
FEDPLUS_BETA = 0.9

# Learning rate η for Fed+ model update
# Reference: FED+.pdf: w_{t+1} = w_t - η × m_{t+1}
FEDPLUS_ETA = 1.0

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
# ===========================================================================
# Number of agents (Section 6, Page 8)
# "we simulated an IDS containing eight agents and one central server"
NUM_AGENTS = 8  # For random split experiment
NUM_ROUNDS = 30  # Number of federated rounds
EPISODES_PER_ROUND = 3  # Episodes each agent trains per round

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

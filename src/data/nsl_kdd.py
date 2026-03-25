"""
NSL-KDD Dataset Preprocessing Module.
============================================================================
Paper Reference: Section 5.1, Page 7, Table 2
============================================================================
"NSL-KDD is a well-known benchmark dataset, used to evaluate the performance
 of any IDS. This dataset was developed in 2009 by Tavallaee et al. based on
 the KDDCUP'99 dataset. There are 148,517 records in the NSL-KDD dataset
 including both the classes i.e., malicious and non-malicious. Each record has
 43 attributes of which 41 features are related to network traffic and the
 last two represents the label (normal/attack) and the severity of input record."

Dataset structure (Table 2, Page 7):
 - Normal:  125,973 training / 22,544 testing  = 148,517 total
 - DoS:     45,927 training  / 7,458 testing   = 53,385 total
 - Probe:   11,656 training  / 2,421 testing   = 14,077 total
 - R2L:     995 training     / 2,887 testing   = 3,882 total
 - U2R:     52 training      / 67 testing       = 119 total

41 features: 4 categorical, 6 binary, 10 continuous, 23 discrete (Page 7)

Binary classification: Normal (0) vs Attack (1)
============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import os

# Column names for NSL-KDD dataset (43 columns total)
# 41 features + label + difficulty_level
COLUMN_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty_level'
]

# Attack type to category mapping (Section 5.1, Page 7)
# "Every label in the dataset is either normal or one of the 38 types of attacks"
# "All these classes can be categorized into 4 groups: DoS, Probe, R2L, U2R"
ATTACK_CATEGORY_MAP = {
    'normal': 'Normal',
    # DoS attacks
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'mailbomb': 'DoS',
    'apache2': 'DoS', 'processtable': 'DoS', 'udpstorm': 'DoS',
    # Probe attacks
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
    'satan': 'Probe', 'mscan': 'Probe', 'saint': 'Probe',
    # R2L attacks
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L',
    'multihop': 'R2L', 'phf': 'R2L', 'spy': 'R2L',
    'warezclient': 'R2L', 'warezmaster': 'R2L', 'sendmail': 'R2L',
    'named': 'R2L', 'snmpgetattack': 'R2L', 'snmpguess': 'R2L',
    'xlock': 'R2L', 'xsnoop': 'R2L', 'worm': 'R2L',
    # U2R attacks
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R',
    'rootkit': 'U2R', 'httptunnel': 'U2R', 'ps': 'U2R',
    'sqlattack': 'U2R', 'xterm': 'U2R',
}


def load_nsl_kdd(train_path, test_path):
    """
    Load NSL-KDD dataset from train and test files.

    Paper Reference: Section 5.1, Page 7
    "There are 148,517 records in the NSL-KDD dataset including both the
     classes i.e., malicious and non-malicious."

    Args:
        train_path: Path to KDDTrain+.txt
        test_path: Path to KDDTest+.txt
    Returns:
        train_df, test_df: Pandas DataFrames
    """
    train_df = pd.read_csv(train_path, names=COLUMN_NAMES, header=None)
    test_df = pd.read_csv(test_path, names=COLUMN_NAMES, header=None)

    print(f"[Data] Loaded training set: {len(train_df)} samples")
    print(f"[Data] Loaded testing set: {len(test_df)} samples")

    return train_df, test_df


def preprocess_nsl_kdd(train_df, test_df):
    """
    Preprocess NSL-KDD dataset for binary classification.

    Paper Reference: Section 5.1, Page 7
    Binary classification: Normal (0) vs Attack (1)

    Steps:
    1. Drop 'difficulty_level' column (not a network feature)
    2. Map attack labels to binary (Normal=0, Attack=1)
    3. One-hot encode categorical features (protocol_type, service, flag)
       - "Out of all the 41 attributes, there are 4 categorical, 6 binary,
          10 continuous and 23 discrete features" (Page 7)
    4. Normalize features using MinMaxScaler

    Args:
        train_df, test_df: Raw DataFrames
    Returns:
        X_train, y_train, X_test, y_test: numpy arrays
        scaler: fitted MinMaxScaler
        feature_dim: int, number of features after preprocessing
    """
    # Step 1: Drop difficulty_level (last column, not a network feature)
    train_df = train_df.drop('difficulty_level', axis=1)
    test_df = test_df.drop('difficulty_level', axis=1)

    # Step 2: Binary label encoding (Page 7)
    # Normal = 0, Any attack = 1
    train_df['label'] = train_df['label'].apply(
        lambda x: 0 if x == 'normal' else 1
    )
    test_df['label'] = test_df['label'].apply(
        lambda x: 0 if x == 'normal' else 1
    )

    # Separate features and labels
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    train_df = train_df.drop('label', axis=1)
    test_df = test_df.drop('label', axis=1)

    # Step 3: One-hot encode categorical features
    # Paper mentions 4 categorical features (Page 7)
    # protocol_type, service, flag are the main categorical columns
    categorical_cols = ['protocol_type', 'service', 'flag']

    # Encode train and test independently to avoid vocabulary leakage,
    # then align test columns to train's feature space.
    train_df = pd.get_dummies(train_df, columns=categorical_cols, dtype=float)
    test_df = pd.get_dummies(test_df, columns=categorical_cols, dtype=float)

    # Align: keep only train columns (unseen test categories → 0)
    test_df = test_df.reindex(columns=train_df.columns, fill_value=0.0)

    X_train = train_df.values.astype(np.float32)
    X_test = test_df.values.astype(np.float32)

    # Step 4: Normalize features using MinMaxScaler
    # Section 4.1, Page 5: "output of the DAE is then passed to the data
    # preprocessor to normalize the learned state representation"
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    feature_dim = X_train.shape[1]
    print(f"[Preprocessing] Feature dimension after encoding: {feature_dim}")
    print(f"[Preprocessing] Training samples: {len(X_train)}")
    print(f"[Preprocessing] Testing samples: {len(X_test)}")
    print(f"[Preprocessing] Train label distribution: Normal={np.sum(y_train==0)}, Attack={np.sum(y_train==1)}")
    print(f"[Preprocessing] Test label distribution: Normal={np.sum(y_test==0)}, Attack={np.sum(y_test==1)}")

    return X_train, y_train, X_test, y_test, scaler, feature_dim


def get_attack_category(label_str):
    """Map specific attack type to its category."""
    return ATTACK_CATEGORY_MAP.get(label_str, 'Unknown')


def distribute_data_random(X, y, num_agents, seed=42):
    """
    Random/Uniform data distribution among agents.

    Paper Reference: Section 6, Page 8
    "In the first experiment, all the available data is randomly split into
     approximately equal parts and shared among the agents."

    Args:
        X: Feature matrix
        y: Labels
        num_agents: Number of agents
        seed: Random seed
    Returns:
        List of (X_agent, y_agent) tuples, one per agent
    """
    np.random.seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    agent_splits = np.array_split(indices, num_agents)
    agent_data = []

    for i, split_indices in enumerate(agent_splits):
        X_agent = X[split_indices]
        y_agent = y[split_indices]
        agent_data.append((X_agent, y_agent))
        print(f"[Data Distribution] Agent {i}: {len(split_indices)} samples "
              f"(Normal: {np.sum(y_agent==0)}, Attack: {np.sum(y_agent==1)})")

    return agent_data


def distribute_data_customized(X_train_df, y_train, X_processed, num_agents=2, seed=42):
    """
    Customized data distribution among agents.

    Paper Reference: Section 6.1, Page 9
    "For the NSL-KDD dataset, one agent is supplied with data samples containing
     only the normal and DOS attack samples, while the other agent is given
     normal as well as Probe, U2R, and R2L attack types."

    NOTE: This requires the original DataFrame with attack-type labels to
    perform the customized split. For simplicity this function requires
    passing the raw train_df with original labels before binary encoding.

    Args:
        X_train_df: Original training DataFrame (before binary encoding)
        y_train: Binary labels
        X_processed: Processed feature matrix
        num_agents: Number of agents (default=2 for customized)
        seed: Random seed
    Returns:
        List of (X_agent, y_agent) tuples
    """
    # Map labels to categories
    categories = X_train_df['label'].apply(get_attack_category).values

    # Agent 0: Normal + DoS samples
    agent0_mask = np.isin(categories, ['Normal', 'DoS'])
    # Agent 1: Normal + Probe + U2R + R2L samples
    agent1_mask = np.isin(categories, ['Normal', 'Probe', 'U2R', 'R2L'])

    agent_data = [
        (X_processed[agent0_mask], y_train[agent0_mask]),
        (X_processed[agent1_mask], y_train[agent1_mask]),
    ]

    for i, (X_a, y_a) in enumerate(agent_data):
        print(f"[Customized Distribution] Agent {i}: {len(X_a)} samples "
              f"(Normal: {np.sum(y_a==0)}, Attack: {np.sum(y_a==1)})")

    return agent_data

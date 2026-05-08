"""
Configuration for Federated RL-based IDS.
Combines PPO + FLTrust (Byzantine-robust) + RL Client Selector (Resource Efficiency).
"""
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")


# ─── Universal Taxonomy (3-class) ───────────────────────────────────────────
# Maps all datasets to a common 3-class taxonomy for unified IDS model.
# This enables a single model to detect attacks across different network environments
# without dataset-specific classification heads.
UNIVERSAL_TAXONOMY = {
    "Benign": 0,
    "Attack": 1,
    "Recon": 2,
}
UNIVERSAL_CLASS_NAMES = ["Benign", "Attack", "Recon"]

# Maps from original attack categories → universal taxonomy
# Format: {dataset_name: {original_label: universal_label}}
UNIVERSAL_ATTACK_MAPS = {
    "edge_iiot": {
        "Benign Traffic": "Benign",
        "DDoS HTTP Flood": "Attack",
        "DDoS TCP SYN Flood": "Attack",
        "Backdoor": "Attack",
        "Ransomware": "Attack",
        "OS Fingerprinting": "Recon",
        "Port Scanning": "Recon",
        "Vulnerability Scanner": "Recon",
        "Password Attack": "Attack",
        "SQL Injection": "Attack",
        "Uploading Attack": "Attack",
        "XSS": "Attack",
        "DDoS UDP Flood": "Attack",
        "DDoS ICMP Flood": "Attack",
        "DoS ICMP Flood": "Attack",
        "DoS TCP Flood": "Attack",
        "DoS UDP Flood": "Attack",
        "MITM ARP Spoofing": "Attack",
        "MQTT DDoS Publish Flood": "Attack",
        "MQTT DoS Connect Flood": "Attack",
        "MQTT DoS Publish Flood": "Attack",
        "MQTT Malformed": "Attack",
        "Recon OS Scan": "Recon",
        "Recon Ping Sweep": "Recon",
        "Recon Port Scan": "Recon",
        "Recon Vulnerability Scan": "Recon",
        "Unknown": "Attack",
    },
    "nsl_kdd": {
        "Benign": "Benign",
        "DoS": "Attack",
        "Probe": "Recon",
        "R2L": "Attack",
        "U2R": "Attack",
        "Unknown": "Attack",
    },
    "iomt_2024": {
        "Benign": "Benign",
        "DDoS ICMP Flood": "Attack",
        "DDoS UDP Flood": "Attack",
        "DoS ICMP Flood": "Attack",
        "DoS TCP Flood": "Attack",
        "DoS UDP Flood": "Attack",
        "MITM": "Attack",
        "MQTT_Attack": "Attack",
        "Recon": "Recon",
        "Unknown": "Attack",
    },
    "unsw_nb15": {
        "Benign": "Benign",
        "Generic": "Attack",
        "Exploits": "Attack",
        "Fuzzers": "Attack",
        "DoS": "Attack",
        "Reconnaissance": "Recon",
        "Analysis": "Attack",
        "Backdoor": "Attack",
        "Shellcode": "Attack",
        "Worms": "Attack",
        "Unknown": "Attack",
    },
}


# ─── NSL-KDD column names ───
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

NSL_KDD_CATEGORICAL = ["protocol_type", "service", "flag"]

# NSL-KDD attack mapping to categories
NSL_KDD_ATTACK_MAP = {
    "normal": "Benign",
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS",
    "apache2": "DoS", "processtable": "DoS", "udpstorm": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L",
    "warezclient": "R2L", "warezmaster": "R2L", "snmpgetattack": "R2L",
    "named": "R2L", "xlock": "R2L", "xsnoop": "R2L",
    "sendmail": "R2L", "httptunnel": "R2L", "worm": "R2L",
    "snmpguess": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "xterm": "U2R", "ps": "U2R",
    "sqlattack": "U2R",
}

# IoMT attack-name to category mapping  
IOMT_ATTACK_MAP = {
    "Benign Traffic": "Benign",
    "DDoS ICMP Flood": "DDoS",
    "DDoS UDP Flood": "DDoS",
    "DoS ICMP Flood": "DoS",
    "DoS TCP Flood": "DoS",
    "DoS UDP Flood": "DoS",
    "MITM ARP Spoofing": "MITM",
    "MQTT DDoS Publish Flood": "MQTT_Attack",
    "MQTT DoS Connect Flood": "MQTT_Attack",
    "MQTT DoS Publish Flood": "MQTT_Attack",
    "MQTT Malformed": "MQTT_Attack",
    "Recon OS Scan": "Recon",
    "Recon Ping Sweep": "Recon",
    "Recon Port Scan": "Recon",
    "Recon Vulnerability Scan": "Recon",
}

# ─── IoMT columns to drop (non-numeric identifiers) ───
IOMT_DROP_COLS = ["Flow ID", "Src IP", "Dst IP", "Timestamp", "Attack Name"]

# ─── Edge-IIoT attack-name to category mapping ───
EDGE_IIOT_ATTACK_MAP = {
    "Benign Traffic": "Benign",
    "DDoS HTTP Flood": "DDoS",
    "DDoS TCP SYN Flood": "DDoS",
    "Backdoor": "Intrusion",
    "Ransomware": "Malware",
    "OS Fingerprinting": "Recon",
    "Port Scanning": "Recon",
    "Vulnerability Scanner": "Recon",
    "Password Attack": "BruteForce",
    "SQL Injection": "Injection",
    "Uploading Attack": "Injection",
    "XSS": "Injection",
}

# Edge-IIoT columns to drop (same CICFlowMeter format as IoMT)
EDGE_IIOT_DROP_COLS = ["Flow ID", "Src IP", "Dst IP", "Timestamp", "Attack Name"]

@dataclass
class RewardConfig:
    """Reward function coefficients for the multi-class IDS environment.

    R(t) = TP_REWARD·TP − FP_PENALTY·FP − FN_PENALTY·FN_BOOST·FN
           + MCC_COEF·MCC + delta·(1 − latency) − collapse_penalty
    """
    alpha: float = 1.0     # TP reward (binary env)
    beta: float = 0.8      # FP penalty (binary env)
    gamma: float = 0.8     # FN penalty (binary env)
    delta: float = 0.3     # latency bonus

    # Multi-class reward parameters (MultiClassIDSEnvironment)
    tp_reward: float = 3.0          # reward for correct attack detection
    tn_reward: float = 1.0          # reward for correct benign — critical for low FPR
    fp_penalty: float = 2.0         # penalty for false alarms
    fn_penalty: float = 4.0         # penalty for missed attacks
    balance_coef: float = 1.0       # balanced accuracy bonus
    entropy_coef: float = 1.0       # prediction diversity bonus
    hhi_coef: float = 1.0          # HHI bias penalty
    collapse_thr: float = 0.70     # trigger when single class > 65%
    collapse_pen: float = 15.0     # penalty when collapse detected
    macro_f1_coef: float = 3.0     # macro F1 reward
    class_weight_cap: float = 3.0   # fallback cap if no class imbalance
    adaptive_cap: float = 50.0      # max cap for highly imbalanced datasets
    focal_gamma: float = 2.0        # focal loss gamma for PPO
    mcc_coef: float = 5.0            # MCC reward scaling coefficient
    fn_weight_boost: float = 2.0     # FN penalty multiplier (missing attacks is worse than false alarms)


@dataclass
class NetworkConfig:
    """Neural network architecture configuration."""
    backbone: str = "cnn_gru"      # cnn_gru only (Transformer removed)
    seq_len: int = 1               # sequence length for GRU
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.15
    dataset: str = "edge_iiot"    # edge_iiot | nsl_kdd | iomt_2024 | unsw_nb15 | unified


@dataclass
class PPOConfig:
    """PPO hyper-parameters."""
    lr_actor: float = 1e-4
    lr_critic: float = 5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.1
    entropy_coef: float = 0.01          # base entropy; auto-scaled by num_classes
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 8
    mini_batch_size: int = 128
    hidden_dim: int = 256
    action_dim: int = 1  # continuous confidence score per sample
    lr_scheduler_enabled: bool = True   # cosine annealing LR to prevent forgetting
    lr_min_factor: float = 0.05         # minimum LR = base_lr * lr_min_factor
    lr_warmup_rounds: int = 5           # warmup rounds before cosine decay


@dataclass
class FedTrustConfig:
    """FLTrust parameters with temporal reputation (inspired by RL-UDHFL).

    Key parameter rationale:
    - reputation_growth (0.1) > reputation_decay (0.05): good clients accumulate
      reputation faster than bad clients lose it — prevents the collapse seen
      when decay exceeded growth.
    - trust_floor=0: no artificial minimum. Trust can reach 0 for malicious
      clients. Set >0 only if guaranteed minimum inclusion is needed.
    """
    root_dataset_size: int = 2000         # must be large enough to produce a reliable
                                          # server update; 500 was too small vs client
                                          # datasets (6000-12000 samples), causing near-zero
                                          # cosine similarities that collapse trust scores
    root_dataset_per_class: bool = True   # balanced sampling
    # Temporal reputation (RL-UDHFL inspired growth/decay)
    # NOTE: growth > decay is critical for stability — prevents trust collapse
    trust_floor: float = 0.0             # no floor — trust can reach 0
    reputation_growth: float = 0.1        # γ_r: growth rate (must exceed decay)
    reputation_decay: float = 0.05        # δ_r: decay rate (must be < growth)
    initial_reputation: float = 0.5      # R_0: initial reputation score


@dataclass
class FedTrustAttentionConfig:
    """FedTrust attention multiplier configuration.

    FIX (D): Replaced accuracy-based multiplier with loss-based multiplier.

    The attention multiplier is computed as: multiplier = k / (1 + loss/floor)
    - loss → 0: mult → k (maximum attention for worst performers)
    - loss = floor: mult = k / 2
    - loss → ∞: mult → 0

    Using loss instead of accuracy prevents the saturation problem where
    all clients converge to ~1.0 accuracy and the multiplier collapses to ~1.0
    for all clients, eliminating differentiation.

    With k=5.0, floor=0.5:
      loss=0.0 → mult=5.0 (worst performers)
      loss=0.5 → mult=2.5
      loss=2.0 → mult=1.25
      loss→∞ → mult→0
    """
    k: float = 5.0       # max attention multiplier
    floor: float = 0.1   # Bug 6 fix: was 0.5 — floor=0.5 means even the BEST-performing
                          # client (loss=0) gets multiplier 3.0 instead of 1.0. This collapses
                          # the dynamic range. floor=0.1 allows best clients to get near 1.0
                          # multiplier while still giving worst clients ~4.8x more attention.


@dataclass
class TrainingConfig:
    """Overall training configuration."""
    num_clients: int = 10
    num_rounds: int = 100          # federated rounds
    local_episodes: int = 8         # local RL episodes per round
    batch_size: int = 256
    max_steps_per_episode: int = 2000
    eval_interval: int = 10
    save_interval: int = 20
    device: str = "cuda"            # or "cpu"
    seed: int = 42
    dataset: str = "edge_iiot"     # "edge_iiot", "nsl_kdd", "iomt_2024", "unsw_nb15", or "unified"
    sample_limit_per_file: int = 50000   # limit large CSV files
    test_ratio: float = 0.2
    output_dir: str = os.path.join(BASE_DIR, "outputs")

    # RL-based client selection
    client_selection_enabled: bool = True
    clients_per_round: int = 8          # K_sel: number of clients selected per round
    selector_hidden_dim: int = 128
    selector_eval_interval: int = 2    # update selector every N rounds (~95 updates for 200 rounds)


@dataclass
class Config:
    seeds: List[int] = field(default_factory=lambda: [42, 123, 777])


@dataclass
class Config:
    reward: RewardConfig = field(default_factory=RewardConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    fed_trust: FedTrustConfig = field(default_factory=FedTrustConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

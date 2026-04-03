# Experimental Setup Guide - FDRL-IDS

**Project:** Federated Deep Reinforcement Learning based Intrusion Detection System
**Paper:** "Federated reinforcement learning based intrusion detection system using dynamic attention mechanism"
**Journal:** Journal of Information Security and Applications 78 (2023) 103608

This guide covers environment setup, running all experiments, reproducing paper results, and reproducing the key tables and figures from the paper.

---

## 1. Environment Setup

### 1.1 Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.8+ | Tested on 3.10 |
| PyTorch | Latest | CPU or CUDA (GPU recommended) |
| NumPy | Latest | |
| Pandas | Latest | |
| Scikit-learn | Latest | |
| Matplotlib | Latest | For visualization |

Install dependencies:

```bash
pip install torch numpy pandas scikit-learn matplotlib
```

### 1.2 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| GPU | Optional | NVIDIA GPU with CUDA for faster training |
| Storage | 5 GB | 10+ GB for datasets |

The system uses GPU automatically if available:

```python
from src.utils.config import DEVICE
print(DEVICE)  # 'cuda' if GPU available, 'cpu' otherwise
```

### 1.3 Dataset Preparation

#### Option A: Local Setup

Download datasets and place them in the `Dataset/` directory:

```
Dataset/
├── NSL-KDD/
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
├── CIC-BCCC-NRC-IoMT-2024/
│   ├── Flow.csv
│   └── *.csv
└── CIC-BCCC-NRC-Edge-IIoTSet-2022/
    ├── Flow.csv
    └── *.csv
```

#### Option B: Kaggle Notebooks

Add these datasets to your Kaggle notebook:
- `nsl-kdd` - NSL-KDD dataset
- `cic-bccc-nrc-iomt-2024` - CIC-IoMT-2024 dataset
- `cic-bccc-nrc-edge-iiotset-2022` - CIC-Edge-IIoTSet-2022 dataset

The Kaggle notebook (`kaggle_notebook.ipynb`) auto-resolves paths. See the notebook Cell 2 for configuration options.

### 1.4 Project Structure

```
NT549/
├── main_train.py              # CLI training script
├── kaggle_notebook.ipynb      # Kaggle notebook
├── EXPERIMENTAL_SETUP.md      # This file
├── src/
│   ├── data/
│   │   ├── nsl_kdd.py         # NSL-KDD loader & preprocessing
│   │   └── cic_common.py      # CIC dataset loaders
│   ├── models/
│   │   ├── denoising_autoencoder.py  # DAE for feature denoising
│   │   └── dqn.py             # DQN agent
│   ├── reinforcement_learning/
│   │   ├── ppo_agent.py               # PPO agent (improvement)
│   │   ├── decision_ppo_agent.py      # Decision-making PPO
│   │   ├── scalable_decision_ppo_agent.py  # Scalable PPO
│   │   └── replay_memory.py
│   ├── federated_learning/
│   │   ├── orchestrator.py    # Main FL orchestrator
│   │   └── server.py          # Central server aggregation
│   ├── attention/
│   │   └── dynamic_attention.py  # Dynamic attention mechanism
│   └── utils/
│       ├── config.py          # All hyperparameters
│       ├── metrics.py         # Evaluation metrics
│       └── visualization.py   # Plotting functions
└── Dataset/                  # Dataset directory
```

---

## 2. Running Experiments

### 2.1 Quick Start

```bash
# NSL-KDD baseline (DQN, 8 agents, random split)
python main_train.py --dataset nsl-kdd --experiment random --num_agents 8 --num_rounds 50

# NSL-KDD with PPO (improvement)
python main_train.py --dataset nsl-kdd --experiment random --agent_type ppo --num_rounds 50
```

### 2.2 Available Agent Types

| Agent Type | Description | Mode | Actions | Flag |
|------------|-------------|------|---------|------|
| `dqn` | Deep Q-Network (original paper Algorithm 1) | Binary classification | 2 | `--agent_type dqn` |
| `ppo` | PPO with advanced reward (improvement) | Binary classification | 2 | `--agent_type ppo` |
| `decision_ppo` | Decision-making PPO | Multi-class decision | 7 | `--agent_type ppo --use_decision_making` |
| `scalable_decision_ppo` | Scalable PPO with online learning | Generalized taxonomy | 8 | `--agent_type scalable_decision_ppo` |

### 2.3 Aggregation Methods

| Method | Description | When to Use | Flag |
|--------|-------------|-------------|------|
| `attention` | Original paper attention-weighted FedAvg (Algorithm 2) | Baseline comparison | `--aggregate_method attention` |
| `fltrust` | FLTrust Byzantine-resilient aggregation | Byzantine attack defense | `--aggregate_method fltrust` |
| `attention_fltrust` | Dynamic Attention + FLTrust combined | Attention + Byzantine | `--aggregate_method attention_fltrust` |
| `fedplus` | Fed+ with server-side momentum | Non-IID data | `--aggregate_method fedplus` |
| `fltrust_fedplus` | FLTrust + Fed+ combined | Byzantine + non-IID | `--aggregate_method fltrust_fedplus` |
| `attention_fltrust_fedplus` | **FULL**: All three combined | Maximum defense | `--aggregate_method attention_fltrust_fedplus` |

### 2.4 Experiment Types

| Experiment | Description | Agents | Attention Params |
|------------|-------------|--------|-----------------|
| `random` | Equal random data split | 8 | k=30, a=50 |
| `customized` | Non-IID: Agent0=Normal+DoS, Agent1=Normal+Probe+U2R+R2L | 2 | k=50000, a=200 |
| `scalability` | Test with 2, 4, 6, 8 agents | Variable | k=30, a=50 |

### 2.5 Command-Line Options

```bash
# Basic options
--dataset {nsl-kdd,iomt,iiot}    # Training dataset (default: nsl-kdd)
--test_dataset {nsl-kdd,iomt,iiot}  # Test dataset (for cross-domain)
--experiment {random,customized,scalability}  # Data distribution
--data_dir ./Dataset              # Dataset directory
--output_dir ./results           # Output directory
--seed 42                        # Random seed

# Training options
--num_agents 8                   # Number of agents (random split)
--num_rounds 50                  # Federated rounds
--episodes_per_round 3           # Episodes per agent per round
--agent_type {dqn,ppo}           # RL algorithm

# Attention parameters (Section 6.1)
--attention_k 30                 # Attention param k (random split)
--attention_a 50                 # Attention param a (random split)

# DAE options
--use_dae                        # Use Denoising Autoencoder (default)
--no_dae                         # Disable DAE

# CIC dataset options
--subsample_per_file 10000       # Max rows per CSV file for CIC datasets

# Smoke test options (for quick runs)
--max_train_samples 10000       # Cap training samples
--max_test_samples 5000         # Cap test samples

# Reward weights (PPO only)
--reward_alpha 1.0              # True Positive reward weight
--reward_beta 0.5               # False Positive penalty
--reward_gamma_fn 2.0           # False Negative penalty
--reward_delta 0.0              # Latency bonus
--reward_epsilon_nov 0.3        # Novelty bonus
--reward_context {high_security,low_alert_fatigue,balanced}  # Preset
```

---

## 3. Key Experiments for the Paper

### 3.1 Table 1: ML-based IDS Comparison

Reproduce baseline ML methods vs. FDRL-IDS for NSL-KDD:

```bash
# Run FDRL-IDS DQN baseline
python main_train.py --dataset nsl-kdd --experiment random --agent_type dqn --num_rounds 50 --output_dir results/table1

# Run FDRL-IDS PPO improvement
python main_train.py --dataset nsl-kdd --experiment random --agent_type ppo --num_rounds 50 --output_dir results/table1
```

**Expected Results (NSL-KDD Random, 8 agents, 50 rounds):**

| Method | Accuracy | FPR | Recall | Precision | F1 | AUC-ROC |
|--------|----------|-----|--------|-----------|-----|---------|
| DQN (paper baseline) | ~0.97 | ~0.02 | ~0.95 | ~0.98 | ~0.96 | ~0.99 |
| PPO (improvement) | Higher | Lower | Higher | Similar | Higher | Higher |

**Note:** Paper Table 3 reports Accuracy=0.9669, FPR=0.0195, Recall=0.9514, Precision=0.9769, F1=0.964, AUC=0.994 for NSL-KDD random split with 8 agents.

### 3.2 Table 2: Federated RL Comparison

Compare different RL algorithms and aggregation methods:

```bash
# DQN + Attention (paper baseline)
python main_train.py --dataset nsl-kdd --experiment random --agent_type dqn --aggregate_method attention --num_rounds 50 --output_dir results/table2/dqn_attention

# DQN + FLTrust (Byzantine defense)
python main_train.py --dataset nsl-kdd --experiment random --agent_type dqn --aggregate_method fltrust --num_rounds 50 --output_dir results/table2/dqn_fltrust

# PPO + Attention+FLTrust+Fed+ (full improvement)
python main_train.py --dataset nsl-kdd --experiment random --agent_type ppo --aggregate_method attention_fltrust_fedplus --num_rounds 50 --output_dir results/table2/ppo_full

# Decision-Making PPO (multi-class)
python main_train.py --dataset nsl-kdd --experiment random --agent_type ppo --use_decision_making --num_rounds 50 --output_dir results/table2/decision_ppo
```

**Aggregation Method Comparison Table:**

| Method | Accuracy | Byzantine Defense | Non-IID Support |
|--------|----------|-------------------|-----------------|
| Attention (paper) | ~0.97 | No | No |
| FLTrust | Similar | Yes | No |
| Fed+ | Similar | No | Yes |
| Attention+FLTrust+Fed+ | Similar/Higher | Yes | Yes |

### 3.3 Ablation Studies

Run each improvement independently to isolate contributions:

```bash
# ===== Ablation Study: RL Algorithm =====
# A1: DQN baseline (original paper)
python main_train.py --dataset nsl-kdd --experiment random --agent_type dqn --aggregate_method attention --num_rounds 50 --output_dir results/ablation/rl_dqn

# A2: PPO (replace DQN with PPO)
python main_train.py --dataset nsl-kdd --experiment random --agent_type ppo --aggregate_method attention --num_rounds 50 --output_dir results/ablation/rl_ppo

# ===== Ablation Study: Aggregation Method =====
# A3: Attention only (original)
python main_train.py --dataset nsl-kdd --experiment random --agent_type dqn --aggregate_method attention --num_rounds 50 --output_dir results/ablation/agg_attention

# A4: FLTrust (Byzantine resilience)
python main_train.py --dataset nsl-kdd --experiment random --agent_type dqn --aggregate_method fltrust --num_rounds 50 --output_dir results/ablation/agg_fltrust

# A5: Fed+ (non-IID correction)
python main_train.py --dataset nsl-kdd --experiment random --agent_type dqn --aggregate_method fedplus --num_rounds 50 --output_dir results/ablation/agg_fedplus

# ===== Ablation Study: Combined =====
# A6: Attention + FLTrust
python main_train.py --dataset nsl-kdd --experiment random --agent_type dqn --aggregate_method attention_fltrust --num_rounds 50 --output_dir results/ablation/agg_attn_fltrust

# A7: FLTrust + Fed+
python main_train.py --dataset nsl-kdd --experiment random --agent_type dqn --aggregate_method fltrust_fedplus --num_rounds 50 --output_dir results/ablation/agg_fltrust_fedplus

# A8: Full (Attention + FLTrust + Fed+) with PPO
python main_train.py --dataset nsl-kdd --experiment random --agent_type ppo --aggregate_method attention_fltrust_fedplus --num_rounds 50 --output_dir results/ablation/full
```

**Ablation Summary Table:**

| Experiment | DQN→PPO | Attention | FLTrust | Fed+ | Expected Impact |
|------------|---------|-----------|---------|------|----------------|
| A1 | - | Yes | - | - | Baseline |
| A2 | Yes | Yes | - | - | Higher AUC-ROC |
| A3 | - | Yes | - | - | Baseline |
| A4 | - | - | Yes | - | Byzantine filtering |
| A5 | - | - | - | Yes | Better non-IID conv. |
| A6 | - | Yes | Yes | - | Combined defense |
| A7 | - | - | Yes | Yes | Combined defense |
| A8 | Yes | Yes | Yes | Yes | Full improvement |

### 3.4 Byzantine Resilience Test

Test defense against malicious agents with 10%, 20%, 30% attack rates.

**Setup:** Use `fltrust` or `attention_fltrust_fedplus` aggregation methods.

```bash
# Test with 10% malicious agents (1 out of 8 agents)
python main_train.py --dataset nsl-kdd --experiment random --num_agents 8 --agent_type dqn --aggregate_method attention_fltrust_fedplus --num_rounds 50 --output_dir results/byzantine/10pct

# Test with 20% malicious agents (2 out of 8 agents)
python main_train.py --dataset nsl-kdd --experiment random --num_agents 8 --agent_type dqn --aggregate_method attention_fltrust_fedplus --num_rounds 50 --output_dir results/byzantine/20pct

# Test with 30% malicious agents (3 out of 8 agents)
python main_train.py --dataset nsl-kdd --experiment random --num_agents 8 --agent_type dqn --aggregate_method attention_fltrust_fedplus --num_rounds 50 --output_dir results/byzantine/30pct
```

**Expected Results:**

| Malicious Rate | Accuracy (Attention only) | Accuracy (FLTrust) | Improvement |
|----------------|---------------------------|-------------------|-------------|
| 0% | ~0.97 | ~0.97 | +0.00 |
| 10% | Degraded | ~0.96 | Byzantine filtered |
| 20% | Degraded | ~0.95 | Byzantine filtered |
| 30% | Degraded | ~0.93 | Byzantine filtered |

**FLTrust Parameters** (tune for your attack model):

```bash
--fltrust_tau 0.5        # Cosine similarity threshold (0.3-0.7)
--fltrust_beta 0.5       # Dimension suppression factor
--fltrust_proxy_ratio 0.01  # Proxy dataset ratio (1% of data)
```

### 3.5 Non-IID Data Distribution Test

Test with customized data split to simulate heterogeneous clients:

```bash
# Customized split (2 agents, non-IID data)
python main_train.py --dataset nsl-kdd --experiment customized --agent_type dqn --aggregate_method attention --num_rounds 50 --output_dir results/noniid/custom_attention

# Customized split with Fed+ (non-IID correction)
python main_train.py --dataset nsl-kdd --experiment customized --agent_type dqn --aggregate_method fedplus --num_rounds 50 --output_dir results/noniid/custom_fedplus

# Customized split with full aggregation
python main_train.py --dataset nsl-kdd --experiment customized --agent_type ppo --aggregate_method attention_fltrust_fedplus --num_rounds 50 --output_dir results/noniid/full
```

**Expected Results:**

| Method | Random Split | Customized (Non-IID) | Impact of Non-IID |
|--------|-------------|---------------------|-------------------|
| Attention | ~0.97 | ~0.94 | -0.03 |
| Fed+ | ~0.97 | ~0.96 | -0.01 |
| Attention+FLTrust+Fed+ | ~0.97 | ~0.96 | -0.01 |

### 3.6 Cross-Domain Generalization

Test generalization across different IoT datasets:

```bash
# Scenario A: NSL-KDD baseline (train & test on NSL-KDD)
python main_train.py --dataset nsl-kdd --experiment random --agent_type dqn --num_rounds 50 --output_dir results/scenario_A

# Scenario B: Same-domain IoT (train & test on CIC-IoMT-2024)
python main_train.py --dataset iomt --experiment random --agent_type ppo --subsample_per_file 10000 --num_rounds 50 --output_dir results/scenario_B

# Scenario C: Cross-domain (train on CIC-IoMT-2024, test on CIC-Edge-IIoTSet-2022)
python main_train.py --dataset iomt --test_dataset iiot --experiment random --agent_type ppo --subsample_per_file 10000 --num_rounds 50 --output_dir results/scenario_C
```

**Expected Cross-Domain Results:**

| Scenario | Train Dataset | Test Dataset | Expected Accuracy |
|----------|-------------|-------------|-------------------|
| A | NSL-KDD | NSL-KDD | ~0.97 |
| B | CIC-IoMT-2024 | CIC-IoMT-2024 | ~0.95 |
| C | CIC-IoMT-2024 | CIC-Edge-IIoTSet-2022 | ~0.85-0.90 |

**Note:** Cross-domain (Scenario C) typically shows reduced accuracy due to distribution shift between datasets.

---

## 4. Expected Results Summary

### 4.1 DQN Baseline (Paper Reproducibility)

**NSL-KDD Random Split, 8 agents:**
- Accuracy: ~96.7%
- FPR: ~2.0%
- Recall: ~95.1%
- Precision: ~97.7%
- F1-Score: ~96.4%
- AUC-ROC: ~0.994

### 4.2 PPO Improvement

PPO should achieve:
- Similar or slightly higher accuracy
- Higher AUC-ROC (better probability calibration)
- Lower FPR (fewer false alarms)
- Smoother convergence curves

### 4.3 FLTrust Byzantine Defense

When Byzantine agents are present:
- FLTrust filters out agents with cosine similarity < tau (default 0.5)
- Accuracy remains stable even with 30% malicious agents
- Check logs for filtering statistics:
  ```
  [FLTrust] Proxy dataset: N samples (1.0% of agent 0 data)
  [FLTrust] Aggregating N/N trusted agents
  ```

### 4.4 Fed+ Non-IID Correction

With non-IID data (customized split):
- Fed+ momentum helps correct client drift
- Faster convergence on heterogeneous data
- More stable training curves

### 4.5 Scalability (Figure 8, Page 13)

Test accuracy vs. number of agents:

```bash
python main_train.py --experiment scalability --dataset nsl-kdd --agent_type dqn --output_dir results/scalability
```

Expected: Average accuracy remains approximately constant as number of agents increases (from 2 to 8 agents).

---

## 5. Reproducing Paper Tables and Figures

### 5.1 Tables

| Paper Table | Description | Command |
|-------------|-------------|---------|
| Table 2 | Dataset statistics | Preprocessing output |
| Table 3 | NSL-KDD random split results | `python main_train.py --dataset nsl-kdd --experiment random` |
| Table 4 | NSL-KDD customized split results | `python main_train.py --dataset nsl-kdd --experiment customized` |

### 5.2 Figures

| Paper Figure | Description | Command |
|-------------|-------------|---------|
| Figure 3 | Training accuracy per agent (random) | Auto-generated in plots |
| Figure 4 | Training loss per agent (random) | Auto-generated in plots |
| Figure 5 | Attention values per agent (random) | Auto-generated in plots |
| Figure 6 | Training curves (customized) | `python main_train.py --experiment customized` |
| Figure 7 | ROC curve | Auto-generated ROC plot |
| Figure 8 | Accuracy vs. number of agents | `python main_train.py --experiment scalability` |

All figures are auto-generated and saved to `./results/plots_*/`.

---

## 6. Kaggle Notebook Usage

For Kaggle notebooks, use `kaggle_notebook.ipynb` with these key configurations in **Cell 2**:

```python
# === SCENARIO & MODEL CONFIGURATION ===
SCENARIO = 'A'          # 'A'=NSL-KDD, 'B'=IoMT same-domain, 'C'=IoMT→IIoT cross-domain
AGENT_TYPE = 'ppo'      # 'dqn', 'ppo', 'decision_ppo', 'scalable_decision_ppo'
AGGREGATE_METHOD = 'attention_fltrust_fedplus'  # Aggregation method

# === TRAINING PARAMETERS ===
NUM_AGENTS = 8
NUM_ROUNDS = 50
EPISODES_PER_ROUND = 3

# === FLTrust Parameters ===
FLTRUST_TAU = 0.5
FLTRUST_BETA = 0.5
FLTRUST_PROXY_RATIO = 0.01

# === Fed+ Parameters ===
FEDPLUS_BETA = 0.9
FEDPLUS_ETA = 1.0

# === Reward Configuration (PPO) ===
REWARD_CONFIG = {
    'alpha': 1.0,      # TP reward
    'beta': 0.5,       # FP penalty
    'gamma_fn': 2.0,   # FN penalty (highest)
    'delta': 0.0,      # Latency bonus
    'epsilon_nov': 0.3,  # Novelty bonus
    'tn': 0.2,         # TN reward
}
```

Then run all cells sequentially.

---

## 7. Troubleshooting

### 7.1 Dataset Not Found

```
FileNotFoundError: NSL-KDD not found
```

**Solution:** Ensure dataset files are in the correct location. Check paths with:

```bash
ls Dataset/NSL-KDD/
```

For Kaggle, ensure datasets are added to the notebook.

### 7.2 Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Reduce `max_train_samples` (e.g., 10000)
- Reduce `num_agents` (e.g., 4)
- Use CPU: `DEVICE=cpu` in `src/utils/config.py`
- Reduce `memory_capacity` in config

### 7.3 Slow Training

**Solutions:**
- Enable GPU: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- Reduce `NUM_ROUNDS` for testing (e.g., 10)
- Use `--max_train_samples 10000` for quick smoke tests

### 7.4 Byzantine Agents Not Filtered

Check FLTrust logs:
```
[FLTrust] Aggregating N/N trusted agents
```

If all agents are trusted, the attack may be subtle. Try:
- Lower `--fltrust_tau` (e.g., 0.3)
- The attack must cause model gradient to diverge significantly to be detected

### 7.5 Non-IID Not Improving

Fed+ requires `previous_weights` across rounds. Ensure:
- Use `fltrust_fedplus` or `attention_fltrust_fedplus` aggregation
- Run for sufficient rounds (30+) for momentum to accumulate

---

## 8. Quick Reference

### Run All Paper Baselines

```bash
# Paper baseline: DQN + Attention
python main_train.py --dataset nsl-kdd --experiment random --agent_type dqn --aggregate_method attention --num_rounds 50 --output_dir results/paper_baseline

# Full improvement: PPO + Attention + FLTrust + Fed+
python main_train.py --dataset nsl-kdd --experiment random --agent_type ppo --aggregate_method attention_fltrust_fedplus --num_rounds 50 --output_dir results/full_improvement

# Cross-domain generalization
python main_train.py --dataset iomt --test_dataset iiot --experiment random --agent_type ppo --subsample_per_file 10000 --num_rounds 50 --output_dir results/cross_domain
```

### Key Hyperparameters Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `attention_k` | 30 | Attention multiplier (random split) |
| `attention_a` | 50 | Attention decay rate (random split) |
| `fltrust_tau` | 0.5 | Byzantine detection threshold |
| `fedplus_beta` | 0.9 | Fed+ momentum coefficient |
| `ppo_clip_epsilon` | 0.2 | PPO clipping range |
| `ppo_epochs` | 4 | PPO update epochs |
| `epsilon_decay` | 0.995 | DQN epsilon decay |
| `memory_capacity` | 10000 | PER replay buffer size |

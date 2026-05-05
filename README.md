# FedRL-IDS: Resource-Efficient Byzantine-Robust Federated Intrusion Detection

**FedRL-IDS** là hệ thống phát hiện xâm nhập mạng (IDS) sử dụng kiến trúc **Federated Reinforcement Learning** hai tầng, kết hợp PPO (Proximal Policy Optimization) với FLTrust (Byzantine-robust aggregation) và RL-based client selection. Thiết kế cho môi trường IoT/IIoT phân tán với dữ liệu Non-IID.

---

## Mục lục

- [Kiến trúc Tổng quan](#1-kiến-trúc-tổng-quan)
- [Thiết kế Separation of Concerns](#2-thiết-kế-separation-of-concerns)
- [Module chính](#3-module-chính)
- [Chi tiết từng thành phần](#4-chi-tiết-từng-thành-phần)
- [Tài liệu tham khảo Paper ↔ Code](#5-tài-liệu-tham-khảo-paper--code)
- [Cài đặt & Chạy](#6-cài-đặt--chạy)
- [Thay đổi kiến trúc (Refactoring)](#7-thay-đổi-kiến-trúc-refactoring)

---

## 1. Kiến trúc Tổng quan

### 1.1 Sơ đồ hai tầng

```
┌──────────────────────────────────────────────────────────────────────┐
│                        CENTRAL SERVER                                  │
│                                                                       │
│   ┌────────────────────────────────────────────────────────────┐    │
│   │        FLTrust Aggregation (Byzantine-Robust)                 │    │
│   │  Cosine Similarity(Δ_k, Δ_0) + Temporal Reputation           │    │
│   │  growth=0.1 > decay=0.05  (prevent trust collapse)           │    │
│   └────────────────────────────────────────────────────────────┘    │
│                              ▲                                        │
│          Aggregated Global Model  │                                   │
└───────────────────────────────┼──────────────────────────────────────┘
              ┌─────────────────┼──────────────────┐
              │                 │                  │
    ┌─────────▼────────┐  ┌────▼────┐    ┌──────▼──────┐
    │  RL Client       │  │ Client  │    │  Client 0   │
    │  Selector        │──│    N   │... │             │
    │  (Tier-2)       │  │CNNGRU  │    │  CNNGRU     │
    │  Bernoulli PPO   │  │-CBAM   │    │  -CBAM      │
    │  7-feature state│  └─────────┘    └─────────────┘
    └─────────────────┘                     (Tier-1: Local PPO)
```

### 1.2 Luồng huấn luyện một round

```
[1] Tier-2: RL Selector     select_clients() → selected_indices [K_sel]
                          Bernoulli PPO, curriculum K_sel decay
                                  │
[2] Tier-1: Selected Clients  train_local() → local model updates
                                  │
[3] Server                    train_server_model() → Δ_0 (root dataset)
                                  │
[4] FLTrust                   cosine(Δ_k, Δ_0) → trust scores
                          reputation update (growth > decay)
                                  │
[5] Aggregator               weighted average (trust scores)
                                  │
[6] Global Model            broadcast to selected clients only
                                  │
[7] Evaluation              accuracy, F1, FPR, MCC
                                  │
[8] Selector Reward         R = ΔAcc − 0.5·|S|/K − 1.0·mean(1−R_k)
```

### 1.3 Cái gì đã bị xóa và tại sao

| Thành phần bị xóa | Lý do |
|---------------------|--------|
| **Meta-Agent (Tier-2 cũ)** | Overfitting trên local test data (Meta-Agent Illusion) |
| **Dynamic Attention** | Authority overlap với FLTrust → gây False Credit Assignment |
| **Autoencoder (Novelty Detector)** | Phức tạp hóa reward function; không cần thiết cho detection |
| **Fed+ Mixing trong Aggregation** | κ ≈ 0.997 khiến 99.7% personalisation được giữ lại → global model không cải thiện |

---

## 2. Thiết kế Separation of Concerns

### 2.1 Hai objective hoàn toàn độc lập

```
┌─────────────────────────────────────────────────────────────┐
│  FLTrust                                  RL Selector        │
│  ─────────                                ──────────        │
│  Objective: Byzantine Robustness          Objective: Communication Efficiency  │
│  Mechanism: cos(Δ_k, Δ_0)               Mechanism: Bernoulli PPO             │
│  Output: R_k ∈ [0, 1]                   Output: which clients to select     │
└─────────────────────────────────────────────────────────────┘
  FLTrust hoàn toàn không biết về RL Selector và ngược lại.
  Hai cơ chế không can thiệp lẫn nhau.
```

**Vấn đề cũ (False Credit Assignment):**
- Thiết kế cũ: `Score_k = π_sel(k|s) × Trust_k × Attention_k`
- Selector chọn client X → FLTrust cho X weight≈0 → Global model vẫn cải thiện → Selector được thưởng dương dù **chọn sai**
- Lý do: FLTrust đã cứu thất bại của Selector → Selector không bao giờ học đúng

**Giải pháp mới:**
- Selector: học chọn **ít clients nhất** mà vẫn giữ quality (qua ΔAcc)
- FLTrust: hoàn toàn độc lập, không ảnh hưởng đến selector reward
- Selector chỉ bị phạt khi chọn client **không tin được** (R_k thấp) — nhưng trust vẫn không block selection

---

## 3. Module chính

### 3.1 Bảng Component

| Module | File | Chức năng |
|--------|------|-----------|
| **PPO Agent** | `src/agents/ppo_agent.py` | Actor-Critic với Categorical policy, GAE(λ=0.95), Focal Loss |
| **Local Client** | `src/agents/local_client.py` | Wrapper: PPO agent + IDS environment, minority class tracking |
| **IDS Environment** | `src/environment/ids_env.py` | Gym-like MDP, MCC-based reward (8 components → MCC), collapse penalty |
| **FLTrust** | `src/federated/fed_trust.py` | Cosine similarity + temporal reputation (growth=0.1 > decay=0.05) |
| **Aggregator** | `src/federated/aggregator.py` | 3-step: FLTrust → Normalize → Weighted Average |
| **RL Selector (Tier-2)** | `src/federated/client_selector.py` | Bernoulli PPO, 7-feature state, resource-efficiency reward |
| **CNNGRUActor** | `src/models/networks.py` | Conv1D → CBAM (channel+spatial) → GRU(2) → Mean Pool → logits |
| **Preprocessor** | `src/data/preprocessor.py` | ADASYN+RENN, RF feature selection, Non-IID partition |
| **Training Loop** | `src/train.py` | Federated orchestration, checkpoint, history JSON |
| **Config** | `src/config.py` | Dataclass configs: PPO, FLTrust, Reward, Training |

### 3.2 Cấu trúc thư mục

```
src/
├── train.py                     # Entry point — federated loop
├── evaluate.py                  # Standalone evaluation
├── config.py                   # Dataclass configs
├── agents/
│   ├── ppo_agent.py          # PPO Actor-Critic
│   └── local_client.py       # PPO + IDS Env wrapper
├── environment/
│   └── ids_env.py            # Gym-like MDP
├── federated/
│   ├── aggregator.py         # FLTrust → Normalize → Weighted Avg
│   ├── fed_trust.py         # Cosine similarity + Temporal Reputation
│   └── client_selector.py   # Bernoulli PPO Selector
├── models/
│   └── networks.py          # CNNGRUActor, CriticNetwork
├── data/
│   └── preprocessor.py       # ADASYN+RENN, RF features, Non-IID partition
└── utils/
    └── metrics.py           # compute_binary/multiclass_metrics
```

---

## 4. Chi tiết từng thành phần

### 4.1 FLTrust — Byzantine-Robust Aggregation (`src/federated/fed_trust.py`)

**Input:** `server_update` (OrderedDict), `client_updates` (List[OrderedDict])
**Output:** `trust_scores` (List[float])

**Step 1 — Cosine Similarity:**
```
cos_k = cos(Δ_k, Δ_0) = ⟨Δ_k, Δ_0⟩ / (||Δ_k|| · ||Δ_0||)
```
Client update Δ_k được so sánh với server update Δ_0 (train trên root dataset sạch). Cosine > 0 nghĩa là cùng hướng gradient.

**Step 2 — Min-Max Normalization:**
```python
cos_weighted[k] = (cos[k] - min(cos)) / (max(cos) - min(cos))
```
Đảm bảo trust scores phân bố trong [0, 1], không bị nén lại như temperature-softmax.

**Step 3 — Temporal Reputation Dynamics:**
```
Growth:   R += 0.1 * (cos - 0.5) * (1 - R)     [cos > 0.5: R tăng]
Decay:    R -= 0.05 * (0.5 - cos) * R          [cos < 0.5: R giảm]
Clamp:    R = clamp(R, 0.0, 1.0)
```
Growth > Decay → client tốt tích lũy reputation nhanh hơn client xấu mất đi. **Đã sửa lỗi:** bản cũ decay > growth khiến trust collapse.

**Trust Score cuối cùng:**
```
trust_k = cos_weighted[k] + 0.2 * (reputation[k] - 0.5)
```
Reputation là bonus, không phải multiplier → tránh trust collapse khi tất cả cosine đều nhỏ.

### 4.2 Aggregator — 3-Step Pipeline (`src/federated/aggregator.py`)

```python
def aggregate_round(self, local_models, server_model, selected_indices, pre_train_models):
    # Step 1: Clean deltas (Global Start Principle)
    client_updates = [compute_update(post, pre)
                     for post, pre in zip(local_models, pre_train_models)]
    server_update = compute_update(server_model, self._global_model)

    # Step 2: FLTrust trust scores
    trust_scores = self.fl_trust.compute_trust_scores(server_update, client_updates)

    # Step 3: Weighted Average (with norm clipping)
    clipped = self.fl_trust.clip_updates(client_updates, max_norm=10.0)
    norm_weights = [w / sum(trust_scores) for w in trust_scores]
    aggregated = weighted_average(reconstructed_models, norm_weights)

    return aggregated, trust_scores
```

**Global Start Principle:** Δ = post_train − pre_train (snapshot trước khi training). Đảm bảo delta là pure gradient update, không chứa personalisation offset từ round trước.

### 4.3 IDS Environment — MCC Reward (`src/environment/ids_env.py`)

```python
def _compute_class_balanced_reward(self, tp, fp, fn, true_label, tn=0, ...):
    w = self._class_weights[true_label]    # inverse class frequency
    fn_boost = 2.0                          # FN penalty × 2.0

    step_reward = (
        w * TP_REWARD * tp
        + TN_REWARD * tn
        - w * FP_PENALTY * fp
        - w * FN_PENALTY * fn_boost * fn
    )

    # MCC: tự cân bằng 4 cells confusion matrix
    mcc = (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) + ε)
    reward = step_reward + 5.0 * mcc + class_bonus + latency_bonus - collapse_penalty
    return reward
```

**Tại sao MCC?** Với NSL-KDD (53% Normal), predict all-Normal đạt 53% accuracy nhưng 0% F1 cho attacks. MCC symmetric trên cả 4 cells, range [-1, 1].

**Collapse Penalty:** Mỗi 20 steps, nếu một class chiếm > 65% predictions → phạt tỷ lệ vượt quá.

### 4.4 PPO Agent — Tier-1 Local Training (`src/agents/ppo_agent.py`)

```python
def update(self, class_weights, focal_gamma):
    advantages, returns = buffer.compute_gae(last_value, gamma=0.99, lam=0.95)

    for epoch in range(ppo_epochs):
        for mb in mini_batches:
            # Focal loss: down-weight easy (majority-class) samples
            focal_weight = (1 - probs[action]) ** focal_gamma
            weighted_ce = focal_weight * F.cross_entropy(logits, actions, reduction='none')

            # PPO ratio: clip to prevent destructive updates
            ratio = exp(new_log_prob - old_log_prob)
            surr1 = ratio * advantages
            surr2 = clamp(ratio, 0.8, 1.2) * advantages
            actor_loss = -min(surr1, surr2).mean()
```

**Entropy Coefficient:** Adaptive theo số classes:
```
c_e = min(c_e_base * max(1, C/3), 0.1)
```
UNSW-NB15 (10 classes): c_e = 0.033; NSL-KDD (5 classes): c_e = 0.017

**LR Cosine Annealing:**
```
η_t = η_min + 0.5 * (η_0 - η_min) * (1 + cos(π * t / T))
```
actor: η_0 = 3e-4, η_min = 3e-5; critic: η_0 = 1e-3, η_min = 1e-4

### 4.5 RL Client Selector — Tier-2 (`src/federated/client_selector.py`)

**7 Features per client:**
```
R_k  : FLTrust temporal reputation         ∈ [0, 1]
l_k  : Evaluation loss (−reward proxy)    ∈ [0, ∞)
Δ_k  : Model divergence ‖w_k−w_glob‖/‖w_glob‖ ∈ [0, ∞)
g_k  : Gradient alignment cos(Δ_k, Δ_glob)    ∈ [−1, 1]
f1_k : Historical F1 EMA                  ∈ [0, 1]
s_k  : Normalized data share n_k/Σn       ∈ [0, 1]
m_k  : Minority class fraction             ∈ [0, 1]
Full state: [K * 7] flattened
```

**Bernoulli Policy:**
```python
probs = sigmoid(actor(state))              # [K] — pure RL policy
bernoulli_dist = Bernoulli(probs)
selected = where(bernoulli_dist.sample() > 0.5)[0]
```

**Curriculum K_sel:**
```python
k_sel(t) = max(k_min, k_init - t * (k_init - k_min) / (T-1))
# Fallback to top-k by probability if K_sel rounds to 0
```

**Resource-Efficiency Reward:**
```python
R_t = ΔAcc_global
      - 0.5 * (|S_t| / K)           # penalize selecting many clients
      - 1.0 * mean(1 - R_k)        # penalize untrusted clients
```

### 4.6 CNN-GRU-CBAM Backbone (`src/models/networks.py`)

```
Input: [batch, seq_len, feature_dim]
  ↓ permute(0, 2, 1)
[batch, feature_dim, seq_len]         # Conv1D expects (channels, length)
  ↓ Conv1D(k=3, c=32) → GroupNorm(1,32) → ReLU
[batch, 32, seq_len]
  ↓ Conv1D(k=5, c=64) → GroupNorm(1,64) → ReLU
[batch, 64, seq_len]
  ↓ CBAM: Channel Attention (MLP shared) → Spatial Attention (Conv1d k=7)
[batch, 64, seq_len]
  ↓ permute(0, 2, 1)
[batch, seq_len, 64]
  ↓ GRU(hidden=128, layers=2, batch_first=True)
[batch, seq_len, 128]
  ↓ LayerNorm → Mean Pool over seq
[batch, 128]
  ↓ Linear → class logits
[batch, action_dim]
```

**CBAM Channel Attention:** `σ(MLP(AvgPool) + MLP(MaxPool))`
**CBAM Spatial Attention:** `σ(Conv1d(k=7, [AvgPool; MaxPool]))`
**GroupNorm(1,C):** Thay BatchNorm vì RL inference thường batch=1.

### 4.7 Preprocessing Pipeline (`src/data/preprocessor.py`)

```
Raw Dataset → Label Encoding → Train/Test Split (stratified) → Fit Scaler on Train Only
  → ADASYN+RENN (train only) → RF Feature Selection (balanced data) → Non-IID Partition
```

**ADASYN:** Adaptive oversampling — tạo nhiều synthetic samples hơn ở minority regions (khác SMOTE đều đặn).
**RENN:** Edited Nearest Neighbors — remove noisy/borderline samples misclassified by their k-NN.
**Non-IID Partition:** Client i nhận 50% từ class (i mod C), 50% từ các class khác.

---

## 5. Tài liệu tham khảo Paper ↔ Code

| Paper | Thành phần | File |
|-------|------------|------|
| Cao et al., "FLTrust", NDSS 2021 | FLTrust: cosine similarity, root dataset | `src/federated/fed_trust.py` |
| RL-UDHFL, Mohammadpour et al., IEEE IoT 2026 | Temporal reputation (growth>decay) | `src/federated/fed_trust.py` |
| Schulman et al., "PPO", arXiv 2017 | PPO, GAE, clipped surrogate objective | `src/agents/ppo_agent.py` |
| Lin et al., "Focal Loss", ICCV 2017 | Focal loss γ=2.0 for class imbalance | `src/agents/ppo_agent.py` |
| Woo et al., "CBAM", ECCV 2018 | Channel + Spatial attention | `src/models/networks.py` |
| Ferrag et al., "SFedRL-IDS", IEEE Access 2022 | Hierarchical FL-RL-IDS | `src/train.py` |
| He et al., "ADASYN", IEEE SMC 2008 | Adaptive oversampling | `src/data/preprocessor.py` |
| CNN-GRU baseline (network flow) | Conv1D temporal patterns | `src/models/networks.py` |

---

## 6. Cài đặt & Chạy

```bash
# Training cơ bản
python -m src.train --dataset nsl_kdd --num_clients 10 --num_rounds 50 --device cuda

# Multi-seed (statistical rigor)
python -m src.train --seeds 42 123 777 --dataset nsl_kdd --num_rounds 50

# Resume từ checkpoint
python -m src.train --dataset edge_iiot --resume outputs/checkpoint_latest.pt

# Evaluation
python -m src.evaluate --model outputs/best_model.pt --dataset nsl_kdd
```

---

## 7. Thay đổi kiến trúc (Refactoring)

### 7.1 Các lỗi đã sửa (trong codebase, không còn trong README)

| Lỗi | File | Mô tả |
|------|------|--------|
| **Trust Score Index** | `train.py` | `trust_scores` (selected-only) → `reputations` (K-length) |
| **Empty server_delta** | `train.py` | Check `len(delta) > 0` thay vì truthiness |
| **Personalization Leakage** | `aggregator.py` | Global Start Principle — Δ = post − pre |
| **Reward Design Smells** | `ids_env.py` | 12+ components → MCC-based reward |
| **Scaler Leakage** | `preprocessor.py` | Fit scaler trên train data ONLY |
| **Feature Selection Bias** | `preprocessor.py` | RF train trên SMOTE-balanced data |
| **Non-IID Overlap** | `preprocessor.py` | Sequential non-overlapping per-class pools |

### 7.2 Dead files đã xóa

```
src/federated/dynamic_attention.py   ← Authority overlap với FLTrust
src/federated/fed_plus.py            ← κ≈0.997 → global model không cải thiện
src/models/networks.py:NoveltyDetector ← Không cần thiết, phức tạp hóa
src/agents/meta_agent.py             ← Overfitting (Meta-Agent Illusion)
```

---

## 8. Triển khai Thực tế (Production Deployment)

Hệ thống hỗ trợ triển khai real-time inference cho môi trường Edge/IoT thông qua kiến trúc **FastAPI + ONNX Runtime + Uvicorn Workers**.

### 8.1 Tại sao chọn ONNX Runtime cho IDS?
- **Low Latency:** Tối ưu hóa kernel (loại bỏ autograd overhead), giảm thời gian inference từ ~20ms (PyTorch) xuống còn **~3-5ms/flow**.
- **High Throughput:** Tận dụng `FastAPI async` kết hợp nhiều `Uvicorn workers` (1 worker = 1 OS process = 1 ONNX Session), đạt thông lượng ~800 requests/s với batch processing.
- **Cross-platform:** Triển khai dễ dàng trên các Edge Gateway (Raspberry Pi, Jetson Nano, ARM) mà không cần cài đặt môi trường PyTorch cồng kềnh.

### 8.2 Luồng xử lý Inference Pipeline
```text
[Network Flow] ──TCP──▶ [FastAPI /predict] ──▶ [JSON payload]
                               │
                        ┌──────▼─────────┐
                        │  Preprocessing │ (MinMaxScaler + Reshape [1, seq, feat])
                        └──────┬─────────┘
                               │ np.float32
                        ┌──────▼─────────┐
                        │  ONNX Runtime  │ (Inference Session / worker)
                        └──────┬─────────┘
                               │ logits [1, C]
                        ┌──────▼─────────┐
                        │  Post-process  │ (Softmax → argmax → class_id)
                        └──────┬─────────┘
                               │ JSON response
                        ┌──────▼─────────┐
                        │  HTTP 200 OK   │ {label, confidence, is_attack}
                        └────────────────┘

8.3 Demo Code Cấu Trúc
src/deploy/export_onnx.py: Script chuyển đổi model .pt sang .onnx (hỗ trợ INT8 Quantization).
src/deploy/api.py: FastAPI server với endpoint /predict và /predict/batch.

## 9. Kết quả kỳ vọng (10 rounds)

| Dataset | Accuracy | F1-Score | FPR |
|---------|----------|----------|-----|
| NSL-KDD | ~89% | ~0.88 | ~6% |
| UNSW-NB15 | ~93% | ~0.93 | ~7% |
| CIC Edge-IIoT | ~99.9% | ~0.9998 | ~0.1% |
| CIC IoMT | ~97% | ~0.968 | ~3% |

---

*FedRL-IDS — Research project cho Network Intrusion Detection trong môi trường IoT/IIoT phân tán với Federated Reinforcement Learning.*

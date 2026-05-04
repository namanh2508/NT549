# FedRL-IDS: Federated Reinforcement Learning for Network Intrusion Detection

Hệ thống **Phát hiện xâm nhập mạng (IDS)** dựa trên kiến trúc **Federated Reinforcement Learning phân cấp 3 tầng**. Kết hợp PPO, CNN-GRU-CBAM cùng pipeline FL đa kỹ thuật (FLTrust + Fed+ + Dynamic Attention). Thiết kế cho môi trường IoT/IIoT phân tán với khả năng chống Byzantine attacks và xử lý dữ liệu Non-IID.

---

## Mục lục

- [Mục 0: Cấu Trúc Dự Án & Mô Tả Codebase](#mục-0--cấu-trúc-dự-án--mô-tả-codebase)
- [Mục 1: Tổng Quan Kiến Trúc](#1-tổng-quan-kiến-trúc)
- [Mục 2: Các Đóng Góp Chính & Nguồn Cảm Hứng](#2-các-đóng-góp-chính--nguồn-cảm-hứng)
- [Mục 3: Quy Trình Xử Lý Dữ Liệu](#3-quy-trình-xử-lý-dữ-liệu)
- [Mục 4: Kiến Trúc Mạng Thần Kinh](#4-kiến-trúc-mạng-thần-kinh)
- [Mục 5: Môi Trường & Hàm Phần Thưởng RL](#5-môi-trường--hàm-phần-thưởng-rl)
- [Mục 6: Cơ Chế Phát Hiện Bất Thường](#6-cơ-chế-phát-hiện-bất-thường)
- [Mục 7: Hệ Thống Phân Cấp 3 Tầng](#7-hệ-thống-phân-cấp-3-tầng)
- [Mục 8: Tổng Hợp Liên Kết (Aggregation)](#8-tổng-hợp-liên-kết-aggregation)
- [Mục 9: Bảng Tham Chiếu Paper ↔ Code](#9-bảng-tham-chiếu-paper--code)
- [Mục 10: Phân Tích Lỗi & Bản Sửa](#10-phân-tích-lỗi--bản-sửa)
- [Mục 11: Nền Tảng Lý Thuyết FL & RL](#11-nền-tảng-lý-thuyết-fl--rl)
- [Mục 12: Hướng Dẫn Demo & Deployment](#12-hướng-dẫn-demo--deployment)
- [Mục 13: Cài Đặt & Chạy Hệ Thống](#13-cài-đặt--chạy-hệ-thống)

---

## Mục 0 — Cấu Trúc Dự Án & Mô Tả Codebase

### 0.1 Sơ đồ cây thư mục

```
NT549/
├── src/
│   ├── __init__.py
│   ├── config.py                  # Dataclass configs: PPO, FLTrust, Fed+, Reward, Training
│   ├── train.py                   # Entry point: federated RL training loop
│   ├── evaluate.py                # Standalone evaluation script
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── ppo_agent.py          # Tier-1 PPO agent: Categorical policy + Focal Loss
│   │   ├── local_client.py       # Tier-1 wrapper: PPO agent + IDS environment
│   │   └── meta_agent.py         # Tier-2 coordinator: CNNGRUActor over one-hot agent actions
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessor.py       # Dataset loaders, ADASYN+RENN, RF feature selection,
│   │                               # Pearson correlation, ANOVA, Non-IID partition
│   ├── environment/
│   │   ├── __init__.py
│   │   └── ids_env.py            # Gym-like IDS environment: binary + multi-class MDP,
│   │                               # MCC-based reward, novelty detection, collapse penalty
│   ├── federated/
│   │   ├── __init__.py
│   │   ├── aggregator.py          # Tri-technique pipeline: FLTrust + Fed+ + Dynamic Attention
│   │   ├── fed_trust.py         # FLTrust: cosine similarity + temporal reputation (growth/decay)
│   │   ├── fed_plus.py           # Fed+: θ_k personalization + κ mixing
│   │   ├── dynamic_attention.py  # Dynamic Attention: loss-based weighting (fairness)
│   │   └── client_selector.py    # Tier-3: Bernoulli PPO selector, 8-feature state, hybrid score
│   ├── models/
│   │   └── networks.py           # CNNGRUActor (Conv1D+CBAM+GRU), NoveltyDetector (Autoencoder),
│   │                               # CriticNetwork, ARCHITECTURE_REGISTRY, DATASET_DEFAULTS
│   └── utils/
│       ├── __init__.py
│       └── metrics.py            # compute_binary_metrics, compute_multiclass_metrics,
│                                 # compute_auc, print_metrics
├── outputs/                      # Training outputs (per dataset): best_model.pt,
│                                 # final_model.pt, best_meta_agent.pt, best_selector.pt,
│                                 # checkpoint_latest.pt, training_history.json, plots/
├── Dataset/                     # Raw datasets (NSL-KDD, UNSW-NB15, Edge-IIoT, IoMT)
│   ├── NSL-KDD/
│   ├── UNSW-NB15/
│   ├── CIC-BCCC-NRC-Edge-IIoTSet-2022/
│   └── CIC-BCCC-NRC-IoMT-2024/
├── Papers/                      # Reference PDFs (FLTrust, Fed+, PPO, etc.)
├── requirements.txt
├── kaggle_train.py             # Kaggle notebook script: multi-dataset training + plotting
├── paper.tex                    # LaTeX source (optional)
├── .gitignore
└── README.md                   # This file
```

### 0.2 Mô tả Context hệ thống

Hệ thống FedRL-IDS giải quyết bài toán **phát hiện xâm nhập mạng (IDS)** trong môi trường IoT/IIoT thực tế, nơi dữ liệu mạng phân tán trên nhiều thiết bị edge/gateway và **không thể tập trung** do ràng buộc về quyền riêng tư (privacy) và băng thông (bandwidth).

**Bài toán thực tế**: Mỗi tổ chức/hospital/device cluster có dữ liệu mạng riêng. Việc gửi raw traffic data lên central server vi phạm GDPR/quyền riêng tư mạng. Đồng thời, các thiết bị IoT có công suất tính toán hạn chế, không thể train model lớn cục bộ. Hơn nữa, attack patterns liên tục thay đổi (zero-day attacks) — hệ thống IDS cần **thích nghi** với phân bố dữ liệu mới.

**Giải pháp**: Kết hợp **Federated Learning** (bảo vệ privacy, giảm communication cost) + **Reinforcement Learning** (học adaptive policy cho client selection và detection) + **kiến trúc 3 tầng** (Local PPO → Meta-Agent → RL Selector).

**Luồng dữ liệu từ đầu vào đến quyết định phân loại**:

```
Raw Dataset (NSL-KDD / UNSW-NB15 / Edge-IIoT / IoMT)
  ↓  (src/data/preprocessor.py)
Label Encoding → ADASYN+RENN balancing → Feature Selection (RF+Pearson+ANOVA)
  ↓
[Non-IID Partition] → Client 0 | Client 1 | ... | Client N
  ↓                                            (Tier-1 Local PPO Agents)
  Client local training (PPO on MultiClassIDSEnvironment)
  ↓
  Model weights Δ_k = post_train - pre_train  (Global Start Principle)
  ↓
  [Tier-2: Meta-Agent] ← coordinates decisions
  ↓
  [Tier-3: RL Client Selector] ← Bernoulli PPO selects top-K clients
  ↓
  [Central Server: FLTrust + Fed+ + Dynamic Attention] ← aggregated global model
  ↓
  [Aggregated Global Model] → eval on test set → accuracy/F1/FPR
```

### 0.3 Bảng tóm tắt các module chính

| Module | File | Chức năng | Input | Output |
|--------|------|-----------|-------|--------|
| **Data Preprocessing** | `src/data/preprocessor.py` | Load 4 datasets, balance (ADASYN+RENN), feature selection | Raw CSV files | X_train/y_train (balanced), X_test/y_test (imbalanced), scaler |
| **PPO Agent (Tier-1)** | `src/agents/ppo_agent.py` | Actor-Critic với Categorical policy + Focal Loss | State vector [seq, features], GAE advantage | Action (class index), log_prob, value |
| **Local Client** | `src/agents/local_client.py` | Wrapper PPO + IDS env, minority class tracking | X_train, y_train | Local model state_dict |
| **IDS Environment** | `src/environment/ids_env.py` | Gym-like MDP: step(), reset(), MCC-based reward | Action (class) | next_state, reward, done, info |
| **FLTrust** | `src/federated/fed_trust.py` | Byzantine-robust trust via cosine similarity + temporal reputation | Server delta Δ₀, client deltas Δ_k | Trust scores [K], updated reputations |
| **Fed+** | `src/federated/fed_plus.py` | Personalization: θ_k = (w_k - w̃)/(1+δ), κ mixing | Local model, global model | Personalised model θ_k |
| **Dynamic Attention** | `src/federated/dynamic_attention.py` | Loss-based fairness weighting | Client losses, num_samples | Attention weights [K] (normalised) |
| **Aggregator** | `src/federated/aggregator.py` | Tri-technique pipeline: FLTrust × Attention × Fed+ | Client updates, server update, client infos | Aggregated global model |
| **Client Selector (Tier-3)** | `src/federated/client_selector.py` | Bernoulli PPO: 8-feature state, hybrid score = π × Trust × Attention | Reputation, attention, loss, divergence... | Selected client indices [K_sel] |
| **Meta-Agent (Tier-2)** | `src/agents/meta_agent.py` | CNNGRUActor over one-hot agent actions | [num_agents, action_dim] one-hot vectors + state | Refined class prediction |
| **CNNGRUActor** | `src/models/networks.py` | Conv1D(k=3,5) → CBAM → GRU(2) → mean pool → logits | [batch, seq, features] | Class logits [batch, action_dim] |
| **NoveltyDetector** | `src/models/networks.py` | Autoencoder train trên benign traffic | Feature vector | Reconstruction error (scalar) |
| **Training Loop** | `src/train.py` | Federated round orchestration, checkpoint save/load | Config | best_model.pt, final_model.pt, history JSON |
| **Evaluation** | `src/evaluate.py` | Load model, predict, compute metrics | Model path (.pt), dataset | Accuracy, F1, precision, recall, FPR, per-class breakdown |
| **Kaggle Script** | `kaggle_train.py` | Multi-seed/multi-dataset training trên Kaggle, auto-plotting | Dataset config | Plots (metrics, trust, attention, accuracy) |

---

## 1. Tổng Quan Kiến Trúc

```text
┌──────────────────────────────────────────────────────────────────┐
│                        CENTRAL SERVER                            │
│  ┌────────────────┐  ┌─────────────┐  ┌───────────────────────┐ │
│  │ FLTrust Module │  │ Fed+ Module │  │ Dynamic Attention      │ │
│  │ (Byzantine)    │  │ (Non-IID)  │  │ (Fairness Weighting)  │ │
│  └───────┬────────┘  └──────┬─────┘  └───────────┬───────────┘ │
│          └───────────────────┴───────────────────┘              │
│                         │                                       │
│              ┌───────────▼───────────┐                           │
│              │   Meta-Agent (Tier-2) │ ← Ensemble Coordinator    │
│              │   CNNGRUActor         │                           │
│              └───────────┬───────────┘                           │
│  ┌──────────────────────▼────────────────────────────────────┐ │
│  │        RL Client Selector (Tier-3, Bernoulli PPO)           │ │
│  │  8-feature state: Rep, Attn, Loss, Div, Grad, F1, Share, Min│ │
│  └───────────────────────┬────────────────────────────────────┘ │
└──────────────────────────┼───────────────────────────────────────┘
          ┌───────────────┼───────────────┐
     ┌────▼────┐     ┌────▼────┐     ┌────▼────┐
     │Client 0 │     │Client 1 │     │Client N │  ← Tier-1
     │PPO+Env │     │PPO+Env │     │PPO+Env │
     │ CNNGRU │     │ CNNGRU │     │ CNNGRU │
     └─────────┘     └─────────┘     └─────────┘
```

---

## 2. Các Đóng Góp Chính & Nguồn Cảm Hứng

### 2.1 Tri-Technique Aggregation

Kết hợp FLTrust + Fed+ + Dynamic Attention trong một pipeline thống nhất.

**Trích dẫn Code** (`src/federated/aggregator.py` lines 136-177):

```python
# Step 1: Compute model updates
client_updates = [
    self.compute_update(post, pre)
    for post, pre in zip(local_models, pre_train_models)
]
server_update = self.compute_update(server_model, self._global_model)

# Step 2: FLTrust — trust scores from deltas
trust_scores = self.fl_trust.compute_trust_scores(server_update, client_updates)

# Step 3: Clip update magnitudes
clipped_updates = self.fl_trust.clip_updates(client_updates, max_norm=10.0)

# Step 4: Reconstruct models
reconstructed_models = [
    OrderedDict((k, self._global_model[k] + cu[k]) for k in self._global_model)
    for cu in clipped_updates
]

# Step 5: Dynamic Attention
attention_values = self.dyn_attn.compute_all_attentions(client_infos)

# Step 6: Combined weights
combined_weights = [ts * att for ts, att in zip(trust_scores, attention_values)]
norm_weights = [w / sum(combined_weights) for w in combined_weights]

# Step 7: Fed+ aggregation
aggregated = self.fed_plus.aggregate(reconstructed_models, weights=norm_weights)
```

**Điểm mới**: Không paper nào kết hợp cả 3 kỹ thuật. FLTrust chỉ lo Byzantine, Fed+ chỉ lo Non-IID, Dynamic Attention chỉ lo fairness. Chúng tôi hợp nhất cả 3 vào chung một pipeline.

### 2.2 Temporal Reputation-Enhanced Trust

Nâng cấp FLTrust bằng theo dõi uy tín theo thời gian (growth > decay).

**Trích dẫn Code** (`src/federated/fed_trust.py` lines 106-141):

```python
def update_reputations(self, cosine_scores: List[float]) -> None:
    """
    Growth > decay ensures good clients accumulate reputation faster
    than bad clients lose it — preventing trust collapse.
    """
    for i, cs in enumerate(cosine_scores):
        delta = cs - self.COSINE_POSITIVE_THRESHOLD  # positive = good

        if delta > 0:
            # R_i += growth_rate * delta * (1 - R_i)
            self.reputations[i] += self.reputation_growth * delta * (1.0 - self.reputations[i])
        else:
            # R_i -= decay_rate * |delta| * R_i
            self.reputations[i] -= self.reputation_decay * abs(delta) * self.reputations[i]

        self.reputations[i] = max(0.0, min(1.0, self.reputations[i]))
```

**Điểm mới**: Áp dụng temporal reputation cho PPO-based RL với γ_r=0.1 > δ_r=0.05, giải quyết vấn đề trust collapse khi cosine similarity thấp trong Deep RL.

### 2.3 Three-Tier Multi-Agent Architecture

Kiến trúc phân cấp: Tier-1 (Local PPO) → Tier-2 (Meta-Agent) → Tier-3 (Selector).

**Trích dẫn Code** (`src/train.py` lines 397-453):

```python
# Tier-1: Local PPO agents
for i, (xp, yp) in enumerate(partitions):
    client = LocalClient(
        client_id=i, X_train=xp[:split], y_train=yp[:split],
        X_test=xp[split:], y_test=yp[split:],
        num_classes=num_classes, cfg=cfg, device=device,
    )
    local_clients.append(client)

# Tier-2: Meta-Agent coordinator
if cfg.training.meta_agent_enabled:
    meta_agent = MetaAgent(
        num_agents=cfg.training.num_clients,
        action_dim=action_dim, state_dim=state_dim, cfg=cfg, device=device,
    )

# Tier-3: RL Client Selector (Bernoulli PPO)
if cfg.training.client_selection_enabled:
    client_selector = RLClientSelector(
        num_clients=cfg.training.num_clients,
        state_dim_per_client=8,
        hidden_dim=cfg.training.selector_hidden_dim,
        cfg=cfg.ppo, device=device,
        total_rounds=cfg.training.num_rounds,
    )
```

**Điểm mới**: SFedRL-IDS dùng FedAvg (không Byzantine-robust). Chúng tôi thêm Tier-2 Meta-Agent và Tier-3 RL Selector — hai tầng SFedRL-IDS không có.

### 2.4 RL-Based Client Selection with Hybrid Scoring

Bernoulli PPO selector + Hybrid Score = π_sel × Trust × Attention.

**Trích dẫn Code** (`src/federated/client_selector.py` lines 586-606):

```python
# Bernoulli probabilities from actor
probs = self.actor.get_probs(state_t).squeeze(0)  # [K], sigmoid outputs

# Hybrid score: RL_prob × Trust × Attention
trust_t = torch.FloatTensor(reputations).to(self.device)
hybrid_scores = probs * trust_t * torch.FloatTensor(attention_weights).to(self.device)

if deterministic or k_sel is not None:
    _, top_indices = torch.topk(hybrid_scores, k)
    selected = top_indices.tolist()
else:
    bernoulli_dist = torch.distributions.Bernoulli(probs)
    samples = bernoulli_dist.sample()
    selected = torch.where(samples > 0.5)[0].tolist()
```

**Điểm mới**: Dùng PPO Bernoulli policy (differentiable) kết hợp với trust/attention tạo hybrid scoring — đảm bảo client Byzantine không bao giờ được chọn dù RL đánh giá cao.

### 2.5 Adaptive PPO with Entropy Decay

Entropy coefficient suy giảm tuyến tính: exploration → exploitation.

**Trích dẫn Code** (`src/federated/client_selector.py` lines 428-441):

```python
def entropy_coef_at_round(self, round_idx: int) -> float:
    """
    Linear decay: exploration early → exploitation late.
    coef(t) = max(entropy_min, entropy_init − (t/T) × (entropy_init − entropy_min))
    """
    progress = round_idx / max(self.total_rounds - 1, 1)
    return max(
        self.entropy_coef_min,
        self.entropy_coef_init - progress * (self.entropy_coef_init - self.entropy_coef_min),
    )
```

**Điểm mới**: Kết hợp entropy decay với adaptive scaling theo số lớp (num_classes/3) để ngăn action collapse trên dataset đa lớp như UNSW-NB15 (10 classes).

### 2.6 MCC-Based Composite Reward

Đơn giản hóa reward từ 12+ thành phần xuống MCC + Cost-Sensitive penalties.

**Trích dẫn Code** (`src/environment/ids_env.py` lines 319-394):

```python
def _compute_class_balanced_reward(self, tp, fp, fn, true_label,
                                   predicted_class, norm_latency,
                                   novelty_bonus, class_bonus, tn=0) -> float:
    w = self._class_weights[true_label]

    # 1. Cost-sensitive step reward (FN >> FP)
    fn_boost = getattr(self.reward_cfg, 'fn_weight_boost', 2.0)
    step_reward = (
        w * self.TP_REWARD * tp + self.TN_REWARD * tn
        - w * self.FP_PENALTY * fp
        - w * self.FN_PENALTY * fn_boost * fn
    )

    # 2. MCC bonus (computed from running episode confusion matrix)
    # MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    mcc_bonus = 0.0
    mcc_coef = getattr(self.reward_cfg, 'mcc_coef', 5.0)
    m = self._episode_metrics
    e_tp, e_fp, e_fn, e_tn = m["tp"], m["fp"], m["fn"], m["tn"]
    if self._episode_total_preds >= 10:
        numerator = e_tp * e_tn - e_fp * e_fn
        denominator = np.sqrt(max((e_tp+e_fp)*(e_tp+e_fn)*(e_tn+e_fp)*(e_tn+e_fn), 1e-8))
        mcc = numerator / denominator
        mcc_bonus = mcc_coef * mcc

    reward = step_reward + mcc_bonus + class_bonus

    # 3. Latency + novelty
    reward += self.reward_cfg.delta * (1.0 - norm_latency)
    reward += self.reward_cfg.epsilon * novelty_bonus

    # 4. Collapse penalty (safety net)
    self._collapse_countdown -= 1
    if self._collapse_countdown <= 0:
        self._collapse_countdown = 20
        if self._episode_total_preds >= 10:
            top_prob = max(self._episode_pred_counts.values()) / self._episode_total_preds
            if top_prob > self.COLLAPSE_THRESHOLD:
                reward -= self.COLLAPSE_PENALTY * (top_prob - self.COLLAPSE_THRESHOLD)
```

**Điểm mới**: Thay thế 12+ thành phần reward bằng MCC duy nhất. MCC tự cân bằng cả 4 góc confusion matrix, tránh "Reward Design Smells".

### 2.7 Comprehensive Evaluation trên 4 Benchmark

NSL-KDD, UNSW-NB15, CIC Edge-IIoT 2022, CIC IoMT 2024.

---

## 3. Quy Trình Xử Lý Dữ Liệu

Module `src/data/preprocessor.py`:

| Bước | Kỹ thuật | Mô tả |
|------|----------|--------|
| 1 | Label Encoding | Xóa IP/Timestamp, chuyển string → số |
| 2 | ADASYN + RENN | Oversampling ranh giới + Loại nhiễu KNN (**chỉ trên Local Train Set**) |
| 3 | Feature Selection | RF Importance + Pearson (r>0.90) + ANOVA F-test |
| 4 | Non-IID Partition | 50% primary class + 50% mixed — giả lập mạng phân tán |

> ⚠️ **Data Leakage**: ADASYN/RENN chỉ được thực hiện trên Local Train Set. Root Dataset KHÔNG được ADASYN vì sẽ làm sai lệch gradient g₀ của FLTrust.

---

## 4. Kiến Trúc Mạng Thần Kinh

**Cảm hứng**: CNN-GRU backbone + CBAM attention.

Module `src/models/networks.py` — `CNNGRUActor` (lines 80-298):

```
Input [batch, seq_len, feature_dim]
  ↓ permute(0, 2, 1)
[batch, feature_dim, seq_len]  ← Conv1D expects (channels, length)
  ↓ Conv1D(kernel=3, out=32, stride=1) → GroupNorm(1,32) → ReLU
[batch, 32, seq_len]
  ↓ Conv1D(kernel=5, out=64, stride=1) → GroupNorm(1,64) → ReLU
[batch, 64, seq_len]
  ↓ CBAM: Channel Attention (MLP shared) → Spatial Attention (Conv1d k=7)
[batch, 64, seq_len]
  ↓ permute(0, 2, 1)
[batch, seq_len, 64]
  ↓ GRU(hidden=128, layers=2, batch_first=True)
[batch, seq_len, 128]
  ↓ LayerNorm
  ↓ Mean pooling over temporal dimension
[batch, 128]
  ↓ Linear → logits
[batch, action_dim]
```

*Ghi chú*: Dùng `GroupNorm(1,C)` thay BatchNorm vì RL predict từng step (batch=1).

**CBAM Implementation** (`src/models/networks.py` lines 224-253):

```python
def _apply_cbam(self, x: torch.Tensor) -> torch.Tensor:
    # Channel attention: pool over seq → MLP → sigmoid → scale
    avg_pool = x.mean(dim=2, keepdim=True)
    max_pool = x.max(dim=2, keepdim=True)[0]
    avg_attn = self.channel_mlp(avg_pool.squeeze(-1)).unsqueeze(-1)
    max_attn = self.channel_mlp(max_pool.squeeze(-1)).unsqueeze(-1)
    channel_attn = torch.sigmoid(avg_attn + max_attn)
    x = x * channel_attn

    # Spatial attention: pool over channel → Conv1d → sigmoid → scale
    avg_sp = x.mean(dim=1, keepdim=True)
    max_sp = x.max(dim=1, keepdim=True)[0]
    concat = torch.cat([avg_sp, max_sp], dim=1)
    spatial_attn = torch.sigmoid(self.spatial_conv(concat))
    x = x * spatial_attn
    return x
```

---

## 5. Môi Trường & Hàm Phần Thưởng RL

**MDP Formulation cho IDS** (`src/environment/ids_env.py`):

- **State space S**: vector đặc trưng mạng (41-80 features sau feature selection)
- **Action space A**: multi-class (5-18 classes tùy dataset) — agent chọn 1 class
- **Transition P**: deterministic — next state = sample tiếp theo từ dataset
- **Reward R**: MCC-based composite reward (xem mục 2.6)
- **Discount γ**: 0.99

**MultiClassIDSEnvironment.reset()** (`src/environment/ids_env.py` lines 306-315):

```python
def reset(self) -> np.ndarray:
    state = super().reset()  # shuffle _order, reset _idx, _step_count, metrics
    self._class_metrics = {c: {"tp": 0, "fp": 0, "fn": 0} for c in range(self.num_classes)}
    self._episode_pred_counts = {c: 0 for c in range(self.num_classes)}
    self._episode_total_preds = 0
    self._collapse_countdown = 0
    self._collapse_detected = False
    return state
```

**MultiClassIDSEnvironment.step()** (`src/environment/ids_env.py` lines 436-535):

```python
def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
    # Support both legacy array (argmax) and new int (class index) actions
    if isinstance(action, (int, np.integer)):
        predicted_class = int(action)
    else:
        predicted_class = int(np.argmax(action))

    true_label = self.y[self._order[self._idx]]
    is_attack = int(true_label != 0)
    predicted_attack = int(predicted_class != 0)

    # Confusion matrix update (binary + per-class)
    # ... (TP/FP/FN/TN tracking)

    # Class-balanced composite reward
    reward = self._compute_class_balanced_reward(
        tp=tp, fp=fp, fn=fn, true_label=true_label,
        predicted_class=predicted_class,
        norm_latency=norm_latency, novelty_bonus=novelty_bonus,
        class_bonus=class_bonus, tn=tn,
    )

    self._idx += 1
    done = self._idx >= len(self.X)
    next_state = self._get_state() if not done else np.zeros(self.state_dim)
    return next_state, reward, done, info
```

**Tại sao MCC tốt hơn Accuracy/F1**: MCC = (TP·TN - FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN)). Với dataset IDS có imbalance ratio > 10x (Normal chiếm 90%), Accuracy sẽ luôn > 90% nếu model chỉ predict Normal. F1 cũng không đối xứng giữa 4 cells của confusion matrix. MCC xử lý cả 4 cells một cách đối xứng, range [-1, 1] (1=perfect, 0=random, -1=inverse).

---

## 6. Cơ Chế Phát Hiện Bất Thường

**Autoencoder-based Novelty Detection** (`src/models/networks.py` lines 44-76):

```python
class NoveltyDetector(nn.Module):
    """
    Autoencoder train trên benign/high-confidence traffic.
    Reconstruction error cao → anomaly.
    """
    def __init__(self, input_dim: int, latent_dim: int = 32):
        hidden = max(64, input_dim // 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        x_hat = self.forward(x)
        return ((x - x_hat) ** 2).mean(dim=-1)
```

**Retraining** (`src/train.py` lines 59-142): Novelty detector được retrain mỗi N rounds trên high-confidence samples (confidence ≥ 0.9) từ global model hiện tại. Threshold = 95th percentile của reconstruction error trên training benign samples.

---

## 7. Hệ Thống Phân Cấp 3 Tầng

### Tier-1: Local PPO Agents

**Trích dẫn** (`src/agents/local_client.py` lines 79-127, `src/agents/ppo_agent.py` lines 138-286):

```python
# Local training loop (src/agents/local_client.py, line 79)
def train_local(self, num_episodes: int) -> Dict[str, float]:
    self.env.reset_novelty_tracking()
    for ep in range(num_episodes):
        self.ppo.reset_hidden()
        state = self.env.reset()
        done = False
        step = 0
        while not done and step < self.cfg.training.max_steps_per_episode:
            action, log_prob, value = self.ppo.select_action(state)
            next_state, reward, done, info = self.env.step(action)
            sample_weight = float(self.env._class_weights[info.get("true_label", 0)])
            self.ppo.store_transition(state, action, log_prob, reward, value, done, sample_weight)
            state = next_state
            step += 1
    update_info = self.ppo.update(
        class_weights=self.env._class_weights,
        focal_gamma=self.env._focal_gamma,
    )
    return {"client_id": self.client_id, "avg_reward": ..., "avg_accuracy": ..., **update_info}
```

- **Algorithm**: PPO + GAE(λ=0.95) + Clip Loss (ε=0.2)
- **Focal Loss**: γ=2.0, down-weight easy (majority) samples
- **Entropy coefficient**: adaptive theo num_classes, decay theo rounds
- **LR**: CosineAnnealing (actor: 3e-4, critic: 1e-3)

### Tier-2: Meta-Agent Coordinator

**Trích dẫn** (`src/agents/meta_agent.py` lines 137-361):

```python
class MetaAgent:
    """
    Tier-2: CNNGRUActor nhận one-hot predictions từ tất cả Tier-1 clients
    + global state → refined class prediction.
    """
    def __init__(self, num_agents, action_dim, state_dim, cfg, device):
        # Input per agent: [action_dim + state_dim] — tiled seq_len = num_agents
        actor_input_dim = action_dim + state_dim
        self.actor = CNNGRUActor(
            input_dim=actor_input_dim,
            action_dim=action_dim,
            hidden_dim=cfg.training.meta_hidden_dim,
            num_layers=2,
            seq_len=num_agents,  # each client = 1 timestep
            dropout=0.1,
        ).to(device)

    def _build_input(self, agent_actions, state) -> torch.Tensor:
        # agent_actions: [num_agents, action_dim] one-hot
        # state: [state_dim]
        aa = torch.FloatTensor(agent_actions)      # [num_agents, action_dim]
        st = torch.FloatTensor(state)               # [state_dim]
        st_tiled = st.unsqueeze(0).expand(num_agents, -1)  # [num_agents, state_dim]
        combined = torch.cat([aa, st_tiled], dim=1)        # [num_agents, action_dim+state_dim]
        return combined.unsqueeze(0)                       # [1, num_agents, action_dim+state_dim]
```

- **Input**: `[batch, num_agents, action_dim+state_dim]` — mỗi client là 1 timestep
- **Output**: class logits — refined decision từ ensemble
- **Train trên Root Dataset** (không phải local test) — tránh Meta-Agent Illusion

### Tier-3: RL Client Selector

**Trích dẫn** (`src/federated/client_selector.py` lines 471-518):

```python
def build_state(self, reputations, attention_weights, client_losses,
                model_divergences, gradient_alignments, f1_scores,
                data_shares, minority_fractions=None) -> np.ndarray:
    """
    8 features per client:
      [R_k, a_k, l_k, Δ_k, g_k, f1_k, s_k, m_k]
    Full state: [K * 8]
    """
    if minority_fractions is None:
        minority_fractions = [0.0] * self.num_clients
    features = []
    for k in range(self.num_clients):
        features.append([
            reputations[k],                         # R_k ∈ [0, 1]
            attention_weights[k],                   # a_k ∈ [0, 1]
            client_losses[k],                       # l_k ∈ [0, ∞)
            model_divergences[k],                   # Δ_k ∈ [0, ∞)
            gradient_alignments[k],                 # g_k ∈ [-1, 1]
            f1_scores[k],                          # f1_k ∈ [0, 1]
            data_shares[k],                         # s_k ∈ [0, 1]
            minority_fractions[k],                  # m_k ∈ [0, 1]
        ])
    return np.array(features, dtype=np.float32).flatten()
```

**Hybrid Score**: `hybrid_score_k = π_sel(k|s) × Trust_k × Attention_k` (line 589-591)

**Entropy Decay** (lines 428-441): Linear decay từ entropy_coef_init=0.05 → entropy_coef_min=0.02

**Curriculum K_sel** (lines 445-467): K_sel(t) = max(K_min, K_init − ⌊t·decay⌋)

---

## 8. Tổng Hợp Liên Kết (Aggregation)

`src/federated/aggregator.py` — Pipeline tuần tự:

```
1. Compute Updates: Δ_k = post_train - pre_train  (Global Start Principle)
2. FLTrust: Cosine Similarity → Trust Scores + Temporal Reputation
3. Clip Updates: max_norm=10.0
4. Dynamic Attention: Loss-based weighting cho fairness
5. Combined Weights: Trust × Attention → normalize
6. Fed+ Aggregate: Weighted mean → Global Model
7. Fed+ Personalize: θ_k = (w_k - w_agg)/(1+δ), mixing with κ
```

**Global Start Principle** (`src/federated/aggregator.py` lines 121-133):

```python
# FIX: Use pre_train_models to compute pure training delta
if pre_train_models is not None:
    client_updates = [
        self.compute_update(post, pre)
        for post, pre in zip(local_models, pre_train_models)
    ]
else:
    client_updates = [
        self.compute_update(lm, self._global_model)
        for lm in local_models
    ]
```

---

## 9. Bảng Tham Chiếu Paper ↔ Code

| Paper | Thành phần | File Code |
|-------|-----------|-----------|
| `FLTRUST.pdf` — Cao et al., NDSS 2021 | FLTrust: cosine trust, root dataset | `src/federated/fed_trust.py` |
| `FED+.pdf` — El Ouadrhiri et al., IEEE TMLCN 2024 | Fed+ personalisation: θ_k, κ mixing | `src/federated/fed_plus.py` |
| `RL-UDHFL...pdf` — Mohammadpour et al., IEEE IoT 2026 | Temporal reputation (growth/decay), utility-driven selection | `src/federated/fed_trust.py`, `src/federated/client_selector.py` |
| `Federated RL...dynamic attention mechanism.pdf` — Pham et al., ICC 2024 | Dynamic Attention: performance-aware weighting | `src/federated/dynamic_attention.py` |
| `Sfedrl-ids.pdf` — Ferrag et al., IEEE Access 2022 | Hierarchical FL-RL-IDS architecture, NSL-KDD benchmark | `src/train.py`, `src/agents/` |
| `Reinforcement Learning An Introduction.pdf` — Sutton & Barto 2018 | PPO, GAE, Actor-Critic, exploration-exploitation | `src/agents/ppo_agent.py` |
| `MathFoundation_RL.pdf` — Szepesvári 2024 | Toán học RL: MDP, policy gradient, advantage | `src/agents/ppo_agent.py` |
| `Deep Reinforcement Learning for Wireless...pdf` — Luong et al., COMST 2019 | DRL framework, LR scheduling | `src/train.py` |
| `FEDERATED_REINFORCEMENT_LEARNING_FOR_ADAPTIVE_CYBER.pdf` — Liu et al. 2023 | PPO cho cyber defense, novelty detection | `src/agents/ppo_agent.py`, `src/models/networks.py` |
| `Intrusion_Detection...Deep Recurrent RL...pdf` — Al-Garadi et al., IEEE TII 2023 | Recurrent RL (GRU) cho IIoT IDS | `src/models/networks.py` |
| `Network Intrusion Detection Model Based on CNN and GRU.pdf` | CNN-GRU backbone architecture | `src/models/networks.py` |
| `PriFED_IDS.pdf` — Nguyen et al. 2023 | Privacy-preserving FL-IDS evaluation | Evaluation methodology |
| `P4P_FL_IDS_poisoning attack.pdf` | Byzantine/poisoning attack scenarios | Trust mechanism design |
| `Fed-Trans-RL.pdf` | Federated Transfer RL, reward engineering | `src/environment/ids_env.py` |

---

## 10. Phân Tích Lỗi & Bản Sửa

### ✅ Lỗi 1 (ĐÃ SỬA): FLTrust Magnitude Normalisation

**Vấn đề**: `gi * (g0_norm / gi_norm)` phá vỡ PPO weights → FPR=1.0

**Sửa** (`src/federated/fed_trust.py` lines 232-260): Thay bằng `clip_updates(max_norm=10.0)` — chỉ clip update quá lớn, giữ nguyên cấu trúc Actor/Critic.

### ✅ Lỗi 2 (ĐÃ SỬA): Personalisation Leakage

**Vấn đề**: Δ = post_train - global_model (chứa θ_k cá nhân hóa cũ)

**Sửa**: Global Start Principle — Δ = post_train - pre_train. Client nhận Global Model sạch trước khi personalise.

### ✅ Lỗi 3 (ĐÃ SỬA): Reward Design Smells

**Vấn đề**: 12+ thành phần reward triệt tiêu lẫn nhau (HHI vs entropy, balanced acc vs macro F1)

**Sửa**: MCC-based reward + Cost-Sensitive (FN penalty × 2.0). Giữ collapse penalty + novelty bonus.

### ✅ Lỗi 4 (ĐÃ SỬA): Meta-Agent Illusion

**Vấn đề**: Meta-Agent accuracy >98% vì train trên dữ liệu Local test của Client (overfit cục bộ)

**Sửa**: Grounding trên Root Dataset — đổi tập train của Meta-Agent sang `X_root, y_root`.

### ✅ Lỗi 5 (ĐÃ SỬA): Novelty Detector Static

**Vấn đề**: Autoencoder train 1 lần ở round 0, không bao giờ update — không thích nghi với attack patterns mới

**Sửa** (`src/train.py` lines 764-788): Periodic retraining mỗi `novelty_retrain_interval` rounds trên high-confidence samples (conf≥0.9) từ global model hiện tại.

### ✅ Lỗi 6 (ĐÃ SỬA): Data Leakage — Scaler Fit Trên Toàn Bộ Data

**Vấn đề**: MinMaxScaler fit trên train+test trước khi split

**Sửa** (`src/data/preprocessor.py` lines 836-845): Scaler fit trên training data ONLY, transform cả train và test.

### ✅ Lỗi 7 (ĐÃ SỬA): Feature Selection Trên Imbalanced Data

**Vấn đề**: RandomForest feature selection train trên imbalanced data → minority-class features bị loại sai

**Sửa** (`src/data/preprocessor.py` lines 278-298): RF train trên SMOTE-balanced data trước khi feature selection.

---

## 11. Nền Tảng Lý Thuyết FL & RL

### 11.1 Federated Learning (FL)

#### Định nghĩa & Bài toán

Federated Learning (FL) là paradigm huấn luyện phân tán cho phép nhiều clients cùng huấn luyện một model chung **mà không cần chia sẻ raw data**. Mỗi client train local trên dữ liệu riêng và chỉ gửi model weights/gradients lên central server.

**Tại sao cần FL thay vì centralized training**:

| Vấn đề | Centralized | Federated |
|---------|-----------|-----------|
| **Privacy** | Raw data tập trung → vi phạm GDPR/quyền riêng tư | Chỉ gửi model weights, không raw data |
| **Communication cost** | Tất cả data gửi lên server | Chỉ gửi model (~hundreds of MB vs GB data) |
| **Data heterogeneity** | Phải aggregate mọi thứ | Mỗi client giữ data local |
| **Scalability** | Server bottleneck | Clients train song song |
| **Failure isolation** | Một điểm lỗi | Client fail không ảnh hưởng toàn cục |

#### Formulation toán học

Bài toán tối ưu hóa FL:

```
min_θ  F(θ) = Σ_{k=1}^{K} (n_k / n) · F_k(θ)

where  F_k(θ) = E_{(x,y)~D_k}[ℓ(f_θ(x), y)]
```

- $K$: số lượng clients
- $n_k$: số samples của client $k$
- $n = Σ_k n_k$: tổng số samples
- $D_k$: phân phối dữ liệu local của client $k$
- $F_k(θ)$: local objective function
- $ℓ$: loss function (cross-entropy cho classification)
- $f_θ(x)$: neural network với weights $θ$

#### FedAvg Algorithm (McMahan et al., 2017)

```
Server executes:
  for each round t = 1, 2, ...:
    m = max(C·K, 1)  # C = fraction of clients per round
    S_t = random set of m clients
    for each client k ∈ S_t in parallel:
      w_{k,t+1} = ClientUpdate(k, w_t)
    w_{t+1} = Σ_{k∈S_t} (n_k / n̄) · w_{k,t+1}  # weighted average
    where n̄ = Σ_{k∈S_t} n_k

ClientUpdate(k, w):
  B = local batch size, E = local epochs, η = learning rate
  w_k = w
  for each local epoch:
    for batch b ⊂ D_k:
      w_k = w_k - η · ∇ℓ(w_k, b)
  return Δw_k = w_k - w  (or w_k itself)
```

#### Thách thức Non-IID

Trong thực tế, dữ liệu phân phối **không đồng nhất** (Non-IID) giữa các clients:

```
Client A: 80% Normal, 5% DoS, 5% Probe, 5% R2L, 5% U2R
Client B: 10% Normal, 30% DoS, 30% Probe, 15% R2L, 15% U2R
Client C: 5% Normal,  5% DoS,  5% Probe, 40% R2L, 45% U2R
```

**Client drift**: Mỗi client di chuyển model theo hướng local gradient khác nhau → khi aggregate, các hướng conflict → global model convergence chậm hoặc diverge.

Fed+ giải quyết bằng cách học personalisation θ_k: `w_k = w̃ + θ_k`, tách biệt global knowledge và local adaptation.

#### Byzantine Robustness

**Byzantine fault model**: Một subset clients có thể gửi arbitrary/malicious updates (gradient poisoning attacks). Kẻ tấn công có thể:

1. Send zero gradients → làm chậm convergence
2. Send random/garbage gradients → corrupt global model
3. Send carefully crafted gradients → bias model về phía attacker

**FLTrust** giải quyết bằng cách so sánh cosine similarity giữa client updates và server update (từ clean root dataset):

```
TS_k = max(0, cos(Δ_k, Δ_0)) = max(0, ⟨Δ_k, Δ_0⟩ / (||Δ_k||·||Δ_0||))
```

Clients có gradient direction khác với server → trust score thấp → weight giảm trong aggregation.

#### Input/Output của FL pipeline

| | Mô tả |
|---|--------|
| **Input** | Local model weights `{θ_k}` từ K clients sau local training |
| **Output** | Aggregated global model `θ_global` |

### 11.2 Reinforcement Learning (RL) & PPO

#### MDP Formulation cho IDS

| Thành phần MDP | Định nghĩa cụ thể trong hệ thống |
|---|---|
| **State Space S** | Feature vector mạng: 41-80 numerical features sau preprocessing |
| **Action Space A** | Multi-class: 5-18 classes (Normal, DoS, Probe, R2L, U2R...) |
| **Transition P** | Deterministic: next state = next sample từ dataset queue |
| **Reward R** | MCC-based composite reward (xem mục 2.6 & 5) |
| **Discount γ** | 0.99 |

#### Policy Gradient & Actor-Critic

Policy Gradient objective:

```
J(θ) = E_{τ~π_θ}[Σ_t γ^t r_t]

∇_θ J(θ) = E_{τ~π_θ}[Σ_t ∇_θ log π_θ(a_t|s_t) · G_t]
```

**Baseline V(s)** giảm variance: advantage `Â_t = G_t - V(s_t)` thay vì return trực tiếp.

**Actor-Critic**: Actor (policy network) đề xuất actions, Critic (value network) ước lượng V(s) để tính advantage. Hai network được train song song.

#### PPO (Proximal Policy Optimization)

PPO cải tiến Policy Gradient bằng **clipping** để ngăn update quá lớn làm destabilize policy:

```
L^CLIP(θ) = E_t[min(r_t(θ)·Â_t, clip(r_t(θ), 1-ε, 1+ε)·Â_t)]

where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  (probability ratio)
```

- Nếu `r_t > 1+ε` (action trở nên quá likely) → clip, không tăng thêm
- Nếu `r_t < 1-ε` (action trở nên quá unlikely) → clip, không giảm thêm
- ε = 0.2 (clip epsilon)

**Generalized Advantage Estimation (GAE)** (`src/agents/ppo_agent.py` lines 64-82):

```
Â_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}

where δ_t = r_t + γV(s_{t+1}) - V(s_t)  (TD error)
```

GAE balance giữa bias (λ=0: 1-step TD) và variance (λ=1: Monte Carlo). λ=0.95 được dùng trong hệ thống.

**Focal Loss** (`src/agents/ppo_agent.py` lines 240-253):

```
FL(p_t) = -(1 - p_t)^γ · log(p_t)

γ = 2.0: down-weight well-classified (majority-class) samples,
focusing learning on hard (minority-class) examples
```

#### Tại sao PPO phù hợp hơn A3C/DDPG cho IDS

| Thuật toán | Ưu điểm | Nhược điểm |
|------------|---------|-----------|
| **PPO** | Stable (clipping), on-policy, works well with discrete/continuous actions | Sample efficiency thấp hơn off-policy |
| **A3C** | Async updates → parallel, fast | Staleness, hyperparameters sensitive |
| **DDPG** | Off-policy, continuous actions | Q-value overestimation, brittle |

PPO được chọn vì: (1) **stable training** — clip mechanism ngăn policy collapse, rất quan trọng với imbalanced IDS data; (2) **on-policy** — không cần replay buffer, đơn giản hóa implementation; (3) **discrete action space** phù hợp với multi-class classification.

#### Input/Output của RL agent

| | Mô tả |
|---|--------|
| **Input** | State vector `s_t` — preprocessed network flow features, shape `[seq_len, n_features]` |
| **Output** | Action `a_t` (predicted class index), Value estimate `V(s_t)` |

---

## 12. Hướng Dẫn Demo & Deployment

### 12.0 Tại Sao Cần ONNX Runtime + FastAPI + Uvicorn?

Việc triển khai mô hình RL-based IDS lên mạng thật đòi hỏi nhiều hơn việc chỉ có file `.pt`. Dưới đây là phân tích chi tiết từng thành phần:

#### ONNX Runtime — Tại sao không dùng PyTorch trực tiếp?

| Tiêu chí | PyTorch (`.pt`) | ONNX Runtime |
|---|---|---|
| **Cross-platform** | Chỉ Python | Windows, Linux, ARM, Edge devices |
| **Inference latency** | Cao (GIL, Python overhead) | Thấp hơn 2-5x (optimized kernels) |
| **Memory footprint** | Lớn (full autograd engine) | Nhỏ (optimized for inference only) |
| **Hardware acceleration** | CPU/GPU thông thường | TensorRT, OpenVINO, DirectML |
| **Deployment ecosystem** | Không có sẵn | Docker, Kubernetes, IoT gateways |
| **Production tooling** | Cần Flask/Django custom | Tích hợp sẵn với FastAPI |

**Lợi ích cụ thể cho FedRL-IDS:**

- **Latency thấp cho real-time IDS**: Một network flow cần được phân loại trong milliseconds. ONNX Runtime sử dụng optimized operator kernels (matrix multiplication, convolution) không qua PyTorch's autograd engine, giảm overhead đáng kể.
- **Hỗ trợ edge devices**: IoT gateway (Raspberry Pi, NVIDIA Jetson, ARM-based devices) không chạy được PyTorch đầy đủ. ONNX Runtime có bản ARM-optimized.
- **Quantization**: ONNX Runtime hỗ trợ INT8 quantization (giảm precision từ float32 xuống int8), giảm model size 4x và tăng throughput 2-3x mà chỉ mất 1-2\% accuracy.
- **opset_version=17**: Bắt buộc cho GRU/Conv1D trong ONNX. Các phiên bản thấp hơn không export được đúng op cho PyTorch's dynamic control flow.
- **Dynamic batching**: ONNX Runtime hỗ trợ dynamic axes (batch size thay đổi được), cho phép xử lý batch size từ 1 đến N flows cùng lúc mà không cần padding.

**Quantization command** (sau khi export):
```python
from onnxruntime.quantization import quantize_dynamic
quantize_dynamic("model.onnx", "model_int8.onnx", weight_type=QuantType.QUInt8)
```

#### FastAPI — Tại sao không dùng Flask/Gradio?

| Tiêu chí | Flask | Gradio/Streamlit | FastAPI |
|---|---|---|---|
| **Async support** | ❌ (blocking) | ❌ | ✅ (native async) |
| **Auto-docs (OpenAPI)** | ❌ | ❌ | ✅ (`/docs`, `/redoc`) |
| **Type validation** | Manual | Partial | ✅ Pydantic auto |
| **Concurrent requests** | Poor (sync) | Poor | ✅ (uvicorn async) |
| **Streaming responses** | Manual | ✅ | ✅ native |
| **Production-ready** | ❌ (dev server) | ❌ (demo only) | ✅ (production) |
| **WebSocket support** | Manual | ✅ | ✅ native |

**Lợi ích cụ thể cho FedRL-IDS:**

- **Pydantic schemas**: Request/response được validate tự động. `NetworkFlow(features: list[float])` tự động reject payloads không đúng format, trả về HTTP 422 với thông báo lỗi chi tiết.
- **Auto-generated API docs**: `/docs` (Swagger UI) và `/redoc` (ReDoc) cung cấp interactive API explorer — không cần viết documentation thủ công. Client có thể test endpoint trực tiếp từ browser.
- **Batch prediction**: `POST /predict/batch` xử lý N flows trong một request. FastAPI async cho phép server xử lý batch này trong khi nhận requests khác, tối đa hóa throughput.
- **JSON response**: Dễ dàng tích hợp với dashboards (Grafana, Kibana), SIEM systems (Splunk, Elastic), và alerting pipelines.
- **Middleware ecosystem**: CORSMiddleware, RateLimitingMiddleware, CompressionMiddleware — plug-and-play mà không cần custom code.

#### Uvicorn Worker — Tại sao cần nhiều workers?

| Tiêu chí | Single worker | Multiple Uvicorn workers |
|---|---|---|
| **Concurrent requests** | 1 request tại một thời điểm | N requests song song |
| **CPU utilization** | 1 core | Tất cả cores |
| **ONNX session per worker** | 1 shared (race condition!) | Mỗi worker có riêng |
| **Failure isolation** | Crash = server down | Crash = 1 worker restart |
| **Throughput (req/s)** | ~50-100 | ~200-400 (4 workers) |

**Lợi ích cụ thể cho FedRL-IDS:**

- **ONNX Session per process**: Mỗi Uvicorn worker chạy trong OS process riêng. `onnxruntime.InferenceSession` được khởi tạo một lần khi worker start, không share giữa các workers (tránh race conditions). 4 workers = 4 ONNX sessions chạy song song trên 4 cores.
- **Thread config**: `intra_op_num_threads=2` + `inter_op_num_threads=2` cho mỗi ONNX session. Tránh oversubscription khi OS có 4 workers × 4 threads = 16 threads tranh chấp 4 cores.
- **Gunicorn + Uvicorn workers**: Production deployment dùng Gunicorn (process manager) với `UvicornWorker`. Gunicorn quản lý worker lifecycle (auto-restart on crash, zero-downtime reload).
- **uvicorn reload**: Chế độ development tự reload khi code thay đổi — không cần restart thủ công.

**Tổng hợp — Pipeline Deployment hoàn chỉnh:**

```
Real Network → Feature Extraction → Preprocessing → ONNX Runtime
                (CICFlowMeter)     (scaler)       (CNNGRU inference)
                                              ↓
                                       FastAPI /predict
                                              ↓
                                   Uvicorn Workers (4x)
                                              ↓
                              JSON: {label, confidence, is_attack}
                                              ↓
                              Alert → SIEM / Dashboard / Webhook
```

**Performance benchmark kỳ vọng** (trên CPU, single flow):

| Setup | Latency | Throughput |
|---|---|---|
| PyTorch `.pt` + Flask | ~15-25ms | ~40 req/s |
| ONNX + FastAPI (1 worker) | ~3-8ms | ~120 req/s |
| ONNX + FastAPI (4 workers) | ~3-8ms | ~400 req/s |
| ONNX INT8 + FastAPI (4 workers) | ~1-3ms | ~800 req/s |

### 12.1 Tổng Quan Kiến Trúc Demo

```
┌──────────────────────────────────────────────────────────────────────┐
│                       REAL NETWORK TRAFFIC                             │
│  Live capture (tcpdump) / PCAP files / Network flows                   │
│                              │                                        │
│                              ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  Feature Extraction (CICFlowMeter / custom Scapy script)         │  │
│  │  → 42-80 numerical features per flow                            │  │
│  │  → Match dataset feature format (NSL-KDD / UNSW-NB15 / ...)     │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                              │                                        │
│                              ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  Preprocessing: Scaler (MinMax from training) + Feature Select  │  │
│  │  → joblib.load("scaler.pkl") + joblib.load("feature_selector") │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                              │                                        │
│                              ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  ONNX Runtime Inference: best_model.onnx                         │  │
│  │  → CNNGRUActor forward pass                                     │  │
│  │  → Softmax → argmax → class prediction                          │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                              │                                        │
│                              ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  FastAPI Server: POST /predict                                  │  │
│  │  → JSON response: {label, confidence, anomaly_score, is_attack}│  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                              │                                        │
│                              ▼                                        │
│  Alerting / Dashboard / SIEM Integration                             │
└──────────────────────────────────────────────────────────────────────┘
```

### 12.2 Export Model sang ONNX

**Script** (`scripts/export_onnx.py`):

```python
#!/usr/bin/env python3
"""
Export PyTorch model to ONNX for FastAPI + ONNX Runtime inference.
Usage: python scripts/export_onnx.py --checkpoint outputs/best_model.pt --output model.onnx
"""
import argparse
import torch
import numpy as np
from src.config import Config
from src.models.networks import build_actor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="model.onnx")
    parser.add_argument("--dataset", type=str, default="edge_iiot",
                        choices=["edge_iiot", "nsl_kdd", "unsw_nb15", "iomt_2024"])
    args = parser.parse_args()

    cfg = Config()
    cfg.training.dataset = args.dataset

    from src.data.preprocessor import load_dataset
    X_train, X_test, y_train, y_test, le = load_dataset(cfg)
    num_classes = len(le.classes_)
    state_dim = X_train.shape[1]
    seq_len = {"edge_iiot": 8, "nsl_kdd": 1, "unsw_nb15": 5, "iomt_2024": 10}[args.dataset]

    # Build actor
    actor = build_actor(
        dataset=args.dataset,
        input_dim=state_dim,
        action_dim=num_classes,
        seq_len=seq_len,
    )
    actor.eval()

    # Load weights
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor_state = {k.replace("actor.", ""): v
                   for k, v in ckpt.items() if k.startswith("actor.")}
    actor.load_state_dict(actor_state, strict=False)
    actor.eval()

    # Export
    dummy_input = torch.randn(1, seq_len, state_dim)
    torch.onnx.export(
        actor,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=17,
        input_names=["network_flow"],
        output_names=["action_logits"],
        dynamic_axes={
            "network_flow": {0: "batch_size"},
            "action_logits": {0: "batch_size"},
        },
    )
    print(f"✓ Exported to {args.output}")

    # Verify with ONNX Runtime
    import onnxruntime as ort
    sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    print(f"  ONNX Runtime providers: {sess.get_providers()}")

    onnx_input = dummy_input.numpy()
    onnx_logits = sess.run(None, {"network_flow": onnx_input})[0]
    print(f"  Output shape: {onnx_logits.shape}")

    # Compare PyTorch vs ONNX
    with torch.no_grad():
        pt_logits = actor(dummy_input).numpy()
    diff = np.abs(pt_logits - onnx_logits).max()
    print(f"  Max diff (PyTorch vs ONNX): {diff:.6f}")
    if diff < 1e-4:
        print("  ✓ Outputs match!")

if __name__ == "__main__":
    main()
```

**Giải thích các tham số**:
- `opset_version=17`: hỗ trợ đầy đủ cho GRU, Conv1D, LayerNorm
- `dynamic_axes`: cho phép batch_size linh hoạt trong production (không cố định batch=1)
- `export_params=True`: embed trained weights vào ONNX file

### 12.3 FastAPI Server

**File** (`src/api/main.py`):

```python
"""
FastAPI inference server for FedRL-IDS.
Run: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
"""
import os
import json
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import joblib

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "outputs/best_model.onnx")
SCALER_PATH = os.environ.get("SCALER_PATH", "outputs/scaler.pkl")
LABELS_PATH = os.environ.get("LABELS_PATH", "outputs/class_labels.json")

# ── Load ONNX model at startup ────────────────────────────────────────────────
opts = ort.SessionOptions()
opts.intra_op_num_threads = 2
opts.inter_op_num_threads = 2
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"])

# ── Load artifacts ─────────────────────────────────────────────────────────────
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
try:
    with open(LABELS_PATH) as f:
        CLASS_LABELS = json.load(f)
except FileNotFoundError:
    CLASS_LABELS = ["Benign", "DoS", "Probe", "R2L", "U2R"]

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="FedRL-IDS API",
    description="Federated RL-based Intrusion Detection System",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class NetworkFlow(BaseModel):
    """Single network flow feature vector."""
    features: list[float] = Field(..., description="Network flow features (normalized)")

class BatchPredictRequest(BaseModel):
    """Batch prediction request."""
    flows: list[list[float]]

class PredictionResult(BaseModel):
    """Single prediction result."""
    label: str
    class_id: int
    confidence: float
    all_probabilities: dict[str, float]
    is_attack: bool
    anomaly_score: Optional[float] = None

class BatchPredictionResult(BaseModel):
    """Batch prediction results."""
    results: list[PredictionResult]
    total_flows: int
    attack_count: int
    processing_time_ms: float

# ── Helpers ────────────────────────────────────────────────────────────────────
def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum(axis=-1, keepdims=True)

def preprocess(raw_features: list[float], scaler=None) -> np.ndarray:
    x = np.array(raw_features, dtype=np.float32)
    if scaler is not None:
        x = scaler.transform(x.reshape(1, -1))
    return x

# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "model": "FedRL-IDS",
        "version": "1.0.0",
        "classes": CLASS_LABELS,
        "onnx_model": MODEL_PATH,
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": os.path.exists(MODEL_PATH)}

@app.post("/predict", response_model=PredictionResult)
def predict(flow: NetworkFlow):
    """
    Predict intrusion detection label for a single network flow.

    Example curl:
    curl -X POST http://localhost:8000/predict \\
         -H "Content-Type: application/json" \\
         -d '{"features": [0.1, 0.5, ...]}'
    """
    import time
    start = time.perf_counter()

    x = preprocess(flow.features, scaler)
    # Reshape to [batch, seq_len, features] — seq_len=1 for NSL-KDD
    x = x.reshape(1, 1, -1).astype(np.float32)

    logits = session.run(None, {"network_flow": x})[0]
    probs = softmax(logits[0])
    class_id = int(np.argmax(probs))

    latency = (time.perf_counter() - start) * 1000

    return PredictionResult(
        label=CLASS_LABELS[class_id] if class_id < len(CLASS_LABELS) else f"Class_{class_id}",
        class_id=class_id,
        confidence=float(probs[class_id]),
        all_probabilities={
            CLASS_LABELS[i] if i < len(CLASS_LABELS) else f"Class_{i}": float(probs[i])
            for i in range(len(probs))
        },
        is_attack=(class_id != 0),
    )

@app.post("/predict/batch", response_model=BatchPredictionResult)
def predict_batch(req: BatchPredictRequest):
    """Batch prediction for multiple flows."""
    import time
    start = time.perf_counter()

    if len(req.flows) == 0:
        raise HTTPException(status_code=400, detail="Empty flows list")

    # Preprocess all flows
    x_batch = np.array(req.flows, dtype=np.float32)
    if scaler is not None:
        x_batch = scaler.transform(x_batch)
    seq_len = 1  # adjust based on dataset
    x_batch = x_batch.reshape(len(req.flows), seq_len, -1)

    logits = session.run(None, {"network_flow": x_batch})[0]
    probs = softmax(logits)
    class_ids = np.argmax(probs, axis=-1)

    results = []
    attack_count = 0
    for i in range(len(req.flows)):
        cid = int(class_ids[i])
        results.append(PredictionResult(
            label=CLASS_LABELS[cid] if cid < len(CLASS_LABELS) else f"Class_{cid}",
            class_id=cid,
            confidence=float(probs[i, cid]),
            all_probabilities={
                CLASS_LABELS[j] if j < len(CLASS_LABELS) else f"Class_{j}": float(probs[i, j])
                for j in range(len(probs[i]))
            },
            is_attack=(cid != 0),
        ))
        if cid != 0:
            attack_count += 1

    latency = (time.perf_counter() - start) * 1000

    return BatchPredictionResult(
        results=results,
        total_flows=len(req.flows),
        attack_count=attack_count,
        processing_time_ms=latency,
    )

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, workers=4)
```

### 12.4 Uvicorn Worker Configuration

**Development** (single worker, auto-reload):
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Production** (Gunicorn + Uvicorn workers — recommended):
```bash
gunicorn src.api.main:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 60 \
  --keep-alive 5 \
  --max-requests 1000 \
  --max-requests-jitter 50
```

**Multi-worker notes**:
- Mỗi worker là 1 OS process riêng — không share ONNX session
- ONNX Runtime thread config: `intra_op_num_threads=2`, `inter_op_num_threads=2` — tránh oversubscription khi có 4 workers
- Với GPU: thêm `--gpu-id 0` và `providers=["CUDAExecutionProvider", "CPUExecutionProvider"]`

### 12.5 Quy Trình Demo trên Network Thật

**Bước 1**: Capture traffic

```python
# capture_live.py
from scapy.all import sniff, TCP, IP
import pandas as pd

def packet_handler(pkt):
    if pkt.haslayer(IP) and pkt.haslayer(TCP):
        flow = {
            "src_bytes": pkt[IP].len - pkt[IP].ihl * 4,
            "dst_bytes": 0,  # filled from response
            "duration": 0,
            "flags": str(pkt.sprintf("%TCP.flags%")),
            # ... extract more features matching dataset format
        }
        return flow
    return None

# Sniff for 60 seconds
sniff(prn=lambda x: print(x), store=False, timeout=60)
```

**Bước 2**: Feature extraction (CICFlowMeter)

```bash
# CICFlowMeter: convert pcap to flow CSV
java -jar CICFlowMeter-Standalone.jar \
  -if input.pcap \
  -of output_flows.csv
```

**Bước 3**: Gửi request đến API

```python
# demo_client.py
import requests
import json
import pandas as pd

API_URL = "http://localhost:8000/predict/batch"

# Load flow CSV (CICFlowMeter output)
df = pd.read_csv("output_flows.csv")
# Select only the 41/42 features matching model input
features = df[FEATURE_COLUMNS].fillna(0).values.tolist()

# Batch predict
resp = requests.post(API_URL, json={"flows": features[:100]})
results = resp.json()

# Print attacks
attacks = [r for r in results["results"] if r["is_attack"]]
print(f"Total: {results['total_flows']}, Attacks: {results['attack_count']}")
for a in attacks:
    print(f"  [{a['label']}] confidence={a['confidence']:.3f}")
```

---

## 13. Cài Đặt & Chạy Hệ Thống

### 13.1 Yêu cầu hệ thống

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 8 GB | 16+ GB |
| **GPU** | — | NVIDIA GPU (8GB VRAM) for training |
| **Storage** | 10 GB | 50 GB (datasets) |
| **OS** | Windows 10 / Ubuntu 20.04 / macOS | Ubuntu 22.04 LTS |

### 13.2 Clone & Môi trường

```bash
# Clone
git clone https://github.com/your-repo/NT549.git
cd NT549

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
pip install imbalanced-learn  # required for ADASYN+RENN
pip install onnxruntime       # required for ONNX inference
pip install fastapi uvicorn    # required for API server
pip install joblib            # for scaler serialization
```

### 13.3 requirements.txt đầy đủ

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
imbalanced-learn>=0.11.0
onnxruntime>=1.16.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
joblib>=1.3.0
```

### 13.4 Chuẩn bị Dataset

```bash
mkdir -p Dataset/NSL-KDD Dataset/UNSW-NB15 Dataset/CIC-BCCC-NRC-Edge-IIoTSet-2022 Dataset/CIC-BCCC-NRC-IoMT-2024

# NSL-KDD (public)
wget -P Dataset/NSL-KDD/ https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt
wget -P Dataset/NSL-KDD/ https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt

# UNSW-NB15: Download from https://research.unsw.edu.au/projects/unsw-nb15-dataset
# Extract training-set.csv and testing-set.csv to Dataset/UNSW-NB15/

# CIC Edge-IIoT 2022: Download from https://www.unb.ca/cic/datasets/edge-iiot-dataset-2022.html
# Extract all CSV files to Dataset/CIC-BCCC-NRC-Edge-IIoTSet-2022/

# CIC IoMT 2024: Download from https://www.unb.ca/cic/datasets/iomt-dataset-2024.html
# Extract all CSV files to Dataset/CIC-BCCC-NRC-IoMT-2024/
```

### 13.5 Chạy Training

```bash
# Single dataset, single seed
python -m src.train \
  --dataset edge_iiot \
  --num_clients 10 \
  --num_rounds 50 \
  --local_episodes 10 \
  --device cuda \
  --seed 42

# With Meta-Agent (Tier-2) enabled
python -m src.train \
  --dataset nsl_kdd \
  --num_clients 10 \
  --num_rounds 100 \
  --local_episodes 10 \
  --meta_agent \
  --device cuda

# Resume from checkpoint
python -m src.train \
  --dataset edge_iiot \
  --resume outputs/outputs_edge_iiot/checkpoint_latest.pt

# Multi-seed sweep (3 seeds, statistical rigor)
python -m src.train --seeds 42 123 777 --dataset edge_iiot
```

### 13.6 Export Model & Chạy API Server

```bash
# Export ONNX
python scripts/export_onnx.py \
  --checkpoint outputs/outputs_edge_iiot/best_model.pt \
  --output outputs/edge_iiot_model.onnx \
  --dataset edge_iiot

# Development server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Production server (4 workers)
gunicorn src.api.main:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --max-requests 1000

# Test API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.5, 0.3, 0.7, 0.2]}'

curl http://localhost:8000/health
```

### 13.7 Cấu trúc Checkpoints & Outputs

```
outputs/
├── outputs_edge_iiot/
│   ├── best_model.pt              # Best global model (highest test accuracy)
│   ├── final_model.pt            # Final global model (after all rounds)
│   ├── best_meta_agent.pt        # Best Meta-Agent (Tier-2) state
│   ├── best_selector.pt         # Best RL Selector (Tier-3) state
│   ├── checkpoint_latest.pt       # Full checkpoint (resume capability)
│   ├── training_history.json     # Metrics history per round
│   ├── eval_results.json         # Final evaluation metrics
│   └── plots/                    # Auto-generated plots (from kaggle_train.py)
├── outputs_nsl_kdd/
│   └── ...
├── outputs_unsw_nb15/
│   └── ...
└── outputs_iomt_2024/
    └── ...
```

### 13.8 Troubleshooting thường gặp

| Lỗi | Nguyên nhân | Giải pháp |
|------|------------|-----------|
| `CUDA out of memory` | Batch size quá lớn | Giảm `batch_size` trong config xuống 64/128 |
| `NaN loss sau vài rounds` | Learning rate quá cao | Giảm `lr_actor` xuống 1e-4, kiểm tra reward normalization |
| `FLTrust trust score = 0 cho tất cả clients` | Root dataset quá nhỏ hoặc format sai | Tăng `root_dataset_size` lên ≥2000, kiểm tra `root_dataset_per_class=True` |
| `Meta-Agent accuracy > 99%` | Meta-Agent Illusion — đang train trên local test overfit | Đảm bảo `meta_agent_train_on_root=True` trong config (mặc định đã bật) |
| `ONNX export fails: GRU not supported` | Opset version quá thấp | Thêm `opset_version=17` vào `torch.onnx.export()` |
| `ImportError: imbalanced-learn` | imbalanced-learn chưa cài | `pip install imbalanced-learn` |
| `NSL-KDD attack labels not found` | File CSV thiếu columns | Kiểm tra file có đúng format KDDTrain+.txt (không phải KDDTrain+.csv) |
| `Non-IID partition creates empty client` | Số lượng classes < num_clients | Giảm `num_clients` hoặc tăng dataset size |
| `Novelty detector error: NaN` | Reconstruction error trên toàn batch đều NaN | Kiểm tra scaler fit đúng, không có NaN trong features |
| `Accuracy stuck at ~50%` | Action collapse — entropy quá thấp sớm | Tăng `entropy_coef_init` lên 0.05, tăng `entropy_coef_min` lên 0.02 |

---

*FedRL-IDS — Research project cho Network Intrusion Detection với Federated Reinforcement Learning. Thiết kế cho môi trường IoT/IIoT phân tán.*

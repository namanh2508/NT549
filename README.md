# FedRL-IDS: Federated Reinforcement Learning for Network Intrusion Detection

Hệ thống **Phát hiện Xâm nhập Mạng (IDS)** dựa trên kiến trúc **Federated Reinforcement Learning phân cấp 3 tầng**. Kết hợp PPO, CNN-GRU-CBAM cùng Pipeline FL đa kỹ thuật (FLTrust + Fed+ + Dynamic Attention).

---

## Mục Lục
1. [Tổng Quan Kiến Trúc](#1-tổng-quan-kiến-trúc)
2. [Các Đóng Góp Chính & Nguồn Cảm Hứng](#2-các-đóng-góp-chính--nguồn-cảm-hứng)
3. [Quy Trình Xử Lý Dữ Liệu](#3-quy-trình-xử-lý-dữ-liệu)
4. [Kiến Trúc Mạng Thần Kinh](#4-kiến-trúc-mạng-thần-kinh)
5. [Môi Trường & Hàm Phần Thưởng RL](#5-môi-trường--hàm-phần-thưởng-rl)
6. [Cơ Chế Phát Hiện Bất Thường](#6-cơ-chế-phát-hiện-bất-thường)
7. [Hệ Thống Phân Cấp 3 Tầng](#7-hệ-thống-phân-cấp-3-tầng)
8. [Tổng Hợp Liên Kết (Aggregation)](#8-tổng-hợp-liên-kết-aggregation)
9. [Bảng Tham Chiếu Paper ↔ Code](#9-bảng-tham-chiếu-paper--code)
10. [Phân Tích Lỗi & Bản Sửa](#10-phân-tích-lỗi--bản-sửa)

---

## 1. Tổng Quan Kiến Trúc

```text
┌──────────────────────────────────────────────────────────────────┐
│                        CENTRAL SERVER                            │
│  ┌────────────────┐  ┌─────────────┐  ┌───────────────────────┐ │
│  │ FLTrust Module │  │ Fed+ Module │  │ Dynamic Attention     │ │
│  │ (Byzantine)    │  │ (Non-IID)   │  │ (Fairness Weighting)  │ │
│  └───────┬────────┘  └──────┬──────┘  └───────────┬───────────┘ │
│          └──────────────────┴─────────────────────┘              │
│                             │                                    │
│                ┌────────────▼────────────┐                       │
│                │   Global Model          │                       │
│                └────────────▲────────────┘                       │
│                ┌────────────▼────────────┐                       │
│                │  Meta-Agent (Tier-2)    │                       │
│                └────────────▲────────────┘                       │
│                ┌────────────▼────────────┐                       │
│                │  RL Client Selector     │                       │
│                │  (Tier-3, Bernoulli)    │                       │
│                └─────────────────────────┘                       │
└───────────────────────┬──────────────────────────────────────────┘
        ┌───────────────┼───────────────┐
   ┌────▼────┐     ┌────▼────┐     ┌────▼────┐
   │Client 0 │     │Client 1 │     │Client N │  ← Tier-1
   │  PPO    │     │  PPO    │     │  PPO    │
   └─────────┘     └─────────┘     └─────────┘
```

---

## 2. Các Đóng Góp Chính & Nguồn Cảm Hứng

### 2.1) Tri-Technique Aggregation
> Kết hợp FLTrust + Fed+ + Dynamic Attention trong một pipeline thống nhất.

**Cảm hứng từ:**
- 📄 `FLTRUST.pdf` — Cao et al., "FLTrust: Byzantine-robust FL via Trust Bootstrapping", NDSS 2021
- 📄 `FED+.pdf` — El Ouadrhiri et al., "FED+: A Unified Approach to FL with Personalization", IEEE TMLCN 2024
- 📄 `Federated reinforcement learning based intrusion detection system using dynamic attention mechanism.pdf` — Pham et al., IEEE ICC 2024

**Trích dẫn Code** (`src/federated/aggregator.py`):
```python
# Pipeline thống nhất: Trust → Attention → Fed+ Aggregation
trust_scores = self.fl_trust.compute_trust_scores(server_update, client_updates)
attention_values = self.dyn_attn.compute_all_attentions(client_infos)
combined_weights = [ts * att for ts, att in zip(trust_scores, attention_values)]
aggregated = self.fed_plus.aggregate(reconstructed_models, weights=norm_weights)
```
*Điểm mới:* Không paper nào kết hợp cả 3 kỹ thuật. FLTrust chỉ lo Byzantine, Fed+ chỉ lo Non-IID, Dynamic Attention chỉ lo fairness. Chúng tôi hợp nhất cả 3 vào chung một pipeline tổng hợp.

---

### 2.2) Temporal Reputation-Enhanced Trust
> Nâng cấp FLTrust bằng theo dõi uy tín theo thời gian (growth > decay).

**Cảm hứng từ:**
- 📄 `FLTRUST.pdf` — Cosine similarity + magnitude normalisation (gốc)
- 📄 `RL-UDHFL_Reinforcement_Learning-Enhanced_Utility-Driven_Hierarchical_Federated_Learning_for_IoT.pdf` — Aledhari et al., IEEE IoT Journal 2024 — Cơ chế temporal reputation growth/decay

**Trích dẫn Code** (`src/federated/fed_trust.py`):
```python
# Lấy cảm hứng từ RL-UDHFL: growth > decay để ổn định
if delta > 0:
    self.reputations[i] += self.reputation_growth * delta * (1.0 - self.reputations[i])
else:
    self.reputations[i] -= self.reputation_decay * abs(delta) * self.reputations[i]
```
*Điểm mới:* RL-UDHFL dùng reputation cho supervised FL. Chúng tôi áp dụng cho PPO-based RL với γ_r=0.1 > δ_r=0.05, giải quyết vấn đề trust collapse khi cosine similarity thấp trong Deep RL.

---

### 2.3) Three-Tier Multi-Agent Architecture
> Kiến trúc phân cấp: Tier-1 (Local PPO) → Tier-2 (Meta-Agent) → Tier-3 (Selector).

**Cảm hứng từ:**
- 📄 `Sfedrl-ids.pdf` — Ferrag et al., "SFedRL-IDS", IEEE Access 2022 — Cấu trúc phân cấp FL + RL
- 📄 `RL-UDHFL...pdf` — Hierarchical FL cho IoT
- 📄 `Reinforcement Learning An Introduction.pdf` — Sutton & Barto, 2018 — Nền tảng lý thuyết PPO, GAE, Actor-Critic

**Trích dẫn Code** (`src/train.py`):
```python
# Tier-1: Local PPO agents
local_clients.append(LocalClient(...))
# Tier-2: Meta-Agent coordinator
meta_agent = MetaAgent(num_agents=cfg.training.num_clients, ...)
# Tier-3: RL Client Selector (Bernoulli PPO)
client_selector = RLClientSelector(num_clients, ...)
```
*Điểm mới:* SFedRL-IDS dùng FedAvg (không Byzantine-robust). Chúng tôi thêm Tier-2 Meta-Agent (ensemble coordinator) và Tier-3 RL Selector (learned selection) — hai tầng mà SFedRL-IDS không có.

---

### 2.4) RL-Based Client Selection with Hybrid Scoring
> Bernoulli PPO selector + Hybrid Score = π_sel × Trust × Attention.

**Cảm hứng từ:**
- 📄 `RL-UDHFL...pdf` — Utility-driven client selection concept (nhưng dùng heuristic, không dùng learned policy)
- 📄 `FLTRUST.pdf` — Trust scores làm rào cản cho RL selection
- 📄 `MathFoundation_RL.pdf` — Szepesvári, 2024 — Lý thuyết toán học RL

**Trích dẫn Code** (`src/federated/client_selector.py`):
```python
probs = self.actor.get_probs(state_t).squeeze(0)  # Bernoulli probability
hybrid_scores = probs * trust_t * attention_t      # Gating mechanism
_, top_indices = torch.topk(hybrid_scores, k)       # Top-K selection
```
*Điểm mới:* RL-UDHFL dùng heuristic scoring. Chúng tôi dùng PPO Bernoulli policy (differentiable) kết hợp với physical trust/attention signals tạo hybrid scoring — đảm bảo client Byzantine KHÔNG BAO GIỜ được chọn dù RL đánh giá cao.

---

### 2.5) Adaptive PPO with Entropy Decay
> Entropy coefficient suy giảm tuyến tính: exploration → exploitation.

**Cảm hứng từ:**
- 📄 `Reinforcement Learning An Introduction.pdf` — Sutton & Barto — Exploration-exploitation trade-off
- 📄 `Deep Reinforcement Learning for Wireless Communications and Networking.pdf` — Luong et al., IEEE COMST 2019 — DRL scheduling techniques
- 📄 `FEDERATED_REINFORCEMENT_LEARNING_FOR_ADAPTIVE_CYBER.pdf` — Liu et al., 2023 — PPO cho cyber defense

**Trích dẫn Code** (`src/federated/client_selector.py`):
```python
def entropy_coef_at_round(self, round_idx):
    progress = round_idx / max(self.total_rounds - 1, 1)
    return max(self.entropy_coef_min,
               self.entropy_coef_init - progress * (self.entropy_coef_init - self.entropy_coef_min))
```
*Điểm mới:* Kết hợp entropy decay với adaptive scaling theo số lớp (num_classes/3) để ngăn action collapse trên dataset đa lớp như UNSW-NB15 (10 classes).

---

### 2.6) MCC-Based Composite Reward (Cải tiến mới)
> Đơn giản hóa reward từ 12+ thành phần xuống MCC + Cost-Sensitive penalties.

**Cảm hứng từ:**
- 📄 `Fed-Trans-RL.pdf` — Reward engineering cho Federated RL
- 📄 `FEDERATED_REINFORCEMENT_LEARNING_FOR_ADAPTIVE_CYBER.pdf` — Multi-objective reward cho IDS
- 📄 `Intrusion_Detection_Approach_for_Industrial_Internet_of_Things_Traffic...pdf` — Al-Garadi et al., IEEE TII 2023 — Cost-sensitive detection cho IIoT

**Trích dẫn Code** (`src/environment/ids_env.py`):
```python
# MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
numerator = e_tp * e_tn - e_fp * e_fn
denominator = np.sqrt(max((...), 1e-8))
mcc_bonus = mcc_coef * (numerator / denominator)

# Cost-sensitive: FN >> FP (bỏ lọt tấn công nguy hiểm hơn cảnh báo nhầm)
step_reward = w * TP_REWARD * tp + TN_REWARD * tn
            - w * FP_PENALTY * fp - w * FN_PENALTY * fn_boost * fn
```
*Điểm mới:* Thay thế 12+ thành phần reward (entropy, HHI, balanced acc, macro F1...) bằng MCC duy nhất. MCC tự cân bằng cả 4 góc confusion matrix, tránh "Reward Design Smells".

---

### 2.7) Comprehensive Evaluation trên 4 Benchmark
> NSL-KDD, UNSW-NB15, CIC Edge-IIoT 2022, CIC IoMT 2024.

**Cảm hứng từ:**
- 📄 `Sfedrl-ids.pdf` — Benchmark NSL-KDD, UNSW-NB15
- 📄 `P4P_FL_IDS_poisioning attack.pdf` — Evaluation methodology cho FL-IDS
- 📄 `PriFED_IDS.pdf` — Privacy-preserving IDS evaluation
- 📄 `Drone-client_FANET.pdf` — IoT network deployment scenarios

---

## 3. Quy Trình Xử Lý Dữ Liệu

Module `src/data/preprocessor.py`:

| Bước | Kỹ thuật | Mô tả |
|------|----------|-------|
| 1 | Label Encoding | Xóa IP/Timestamp, chuyển string → số |
| 2 | ADASYN + RENN | Oversampling ranh giới + Loại nhiễu KNN (**chỉ trên Local Train Set**) |
| 3 | Feature Selection | Random Forest Importance + Pearson (r>0.95) + ANOVA F-test |
| 4 | Non-IID Partition | 50% primary class + 50% mixed — giả lập mạng phân tán thực tế |

> ⚠️ **Lưu ý Data Leakage:** ADASYN/RENN chỉ được thực hiện trên Local Train Set. Server Root Dataset KHÔNG được ADASYN vì sẽ làm sai lệch gradient g₀ của FLTrust.

---

## 4. Kiến Trúc Mạng Thần Kinh

**Cảm hứng từ:** 📄 `Network Intrusion Detection Model Based on CNN and GRU.pdf`

Module `src/models/networks.py` — `CNNGRUActor`:
```text
Input [batch, seq_len, features]
  → Conv1D(k=3) → GroupNorm → ReLU
  → Conv1D(k=5) → GroupNorm → ReLU
  → CBAM Attention (Channel + Spatial)
  → GRU (2 layers, bidirectional=False)
  → Actor Head: Linear → Categorical (π(a|s))
  → Critic Head: Linear → Scalar (V(s))
```
*Ghi chú:* Dùng `GroupNorm(1,C)` thay BatchNorm vì RL predict từng step (batch=1).

---

## 5. Môi Trường & Hàm Phần Thưởng RL

**Cảm hứng từ:**
- 📄 `Reinforcement Learning An Introduction.pdf` — MDP formulation, GAE, PPO clip loss
- 📄 `MathFoundation_RL.pdf` — Toán học nền tảng RL

Module `src/environment/ids_env.py` — `MultiClassIDSEnvironment`:
```text
Reward (MCC-based, đã cải tiến) =
    Cost-Sensitive Step Reward (TP/TN/FP/FN với class weights)
  + MCC Bonus (Matthews Correlation Coefficient từ episode metrics)
  + Class Identification Bonus (+0.5 đúng / -0.5 sai)
  + Latency Bonus + Novelty Bonus
  - Collapse Penalty (nếu 1 class > 65% predictions)
```

PPO Agent (`src/agents/ppo_agent.py`): **Focal Loss** (γ=2.0) + inverse-class frequency weights.

---

## 6. Cơ Chế Phát Hiện Bất Thường

**Cảm hứng từ:** 📄 `FEDERATED_REINFORCEMENT_LEARNING_FOR_ADAPTIVE_CYBER.pdf` — Novelty/anomaly detection concepts

- **Autoencoder** trong `src/models/networks.py` train trên mẫu Benign/high-confidence
- **Threshold:** 95th percentile sai số tái tạo
- **Retraining:** Mỗi N rounds bằng hàm `retrain_novelty_detector()` trong `src/train.py`

---

## 7. Hệ Thống Phân Cấp 3 Tầng

### Tier-1: Local PPO Agents
📄 Cảm hứng: `Reinforcement Learning An Introduction.pdf`, `Sfedrl-ids.pdf`
- PPO + GAE + Clip Loss trên `MultiClassIDSEnvironment`

### Tier-2: Meta-Agent Coordinator
📄 Cảm hứng: `Sfedrl-ids.pdf` (hierarchical concept), `Fed-Trans-RL.pdf`
- Input: One-Hot vectors từ tất cả Client + State gốc **(từ Root Dataset)**
- Output: Dự đoán nhãn cuối (ensemble coordinator)
- **Đặc biệt:** Train/Eval trực tiếp trên Server Root Dataset để tránh "Meta-Agent Illusion" (ảo giác do đánh giá trên tập dữ liệu overfit của Client).

### Tier-3: RL Client Selector (Bernoulli PPO)
📄 Cảm hứng: `RL-UDHFL...pdf` (utility-driven selection concept)
- 8 features state: Reputation, Attention, Loss, Divergence, Gradient Alignment, F1-EMA, Data Share, Minority Fraction
- Hybrid Score = π_sel(k|s) × Trust_k × Attention_k

---

## 8. Tổng Hợp Liên Kết (Aggregation)

`src/federated/aggregator.py` — Pipeline tuần tự:

```text
1. Compute Updates: Δ_k = post_train - pre_train (Global Start Principle)
2. FLTrust: Cosine Similarity → Trust Scores + Temporal Reputation
3. Clip Updates: max_norm=10.0 (thay vì magnitude normalisation)
4. Dynamic Attention: Loss-based weighting cho fairness
5. Combined Weights: Trust × Attention → normalize
6. Fed+ Aggregate: Weighted mean → Global Model
7. Fed+ Personalise: θ_k = (w_k - w_agg)/(1+δ), mixing with κ
```

---

## 9. Bảng Tham Chiếu Paper ↔ Code

| Paper (thư mục `Papers/`) | Thành phần hệ thống | File Code |
|---|---|---|
| `FLTRUST.pdf` — Cao et al., NDSS 2021 | FLTrust: cosine trust, trimmed mean, root dataset | `src/federated/fed_trust.py` |
| `FED+.pdf` — El Ouadrhiri et al., IEEE TMLCN 2024 | Fed+ personalisation: θ_k, κ mixing, δ smoothing | `src/federated/fed_plus.py` |
| `RL-UDHFL...pdf` — Aledhari et al., IEEE IoT 2024 | Temporal reputation (growth/decay), utility-driven selection | `src/federated/fed_trust.py`, `src/federated/client_selector.py` |
| `Federated RL...dynamic attention mechanism.pdf` — Pham et al., ICC 2024 | Dynamic Attention: performance-aware weighting | `src/federated/dynamic_attention.py` |
| `Sfedrl-ids.pdf` — Ferrag et al., IEEE Access 2022 | Hierarchical FL-RL-IDS architecture, NSL-KDD benchmark | `src/train.py`, `src/agents/` |
| `Reinforcement Learning An Introduction.pdf` — Sutton & Barto 2018 | PPO, GAE, Actor-Critic, exploration-exploitation | `src/agents/ppo_agent.py` |
| `MathFoundation_RL.pdf` — Szepesvári 2024 | Toán học RL: MDP, policy gradient, advantage | `src/agents/ppo_agent.py` |
| `Deep Reinforcement Learning for Wireless...pdf` — Luong et al., COMST 2019 | DRL framework, LR scheduling, network RL | `src/train.py` (cosine annealing) |
| `FEDERATED_REINFORCEMENT_LEARNING_FOR_ADAPTIVE_CYBER.pdf` — Liu et al. 2023 | PPO cho cyber defense, novelty detection | `src/agents/ppo_agent.py`, `src/models/networks.py` |
| `Intrusion_Detection...Deep Recurrent RL...pdf` — Al-Garadi et al., TII 2023 | Recurrent RL (GRU) cho IIoT IDS | `src/models/networks.py` (GRU layers) |
| `Network Intrusion Detection Model Based on CNN and GRU.pdf` | CNN-GRU backbone architecture | `src/models/networks.py` (CNNGRUActor) |
| `PriFED_IDS.pdf` — Nguyen et al. 2023 | Privacy-preserving FL-IDS evaluation | Evaluation methodology |
| `P4P_FL_IDS_poisioning attack.pdf` | Byzantine/poisoning attack scenarios | Trust mechanism design |
| `Fed-Trans-RL.pdf` | Federated Transfer RL, reward engineering | `src/environment/ids_env.py` |
| `Drone-client_FANET.pdf` — Bujari et al. 2021 | IoT deployment scenarios, scale projection | System architecture design |

---

## 10. Phân Tích Lỗi & Bản Sửa

### ✅ Lỗi 1 (ĐÃ SỬA): FLTrust Magnitude Normalisation
**Vấn đề:** `gi * (g0_norm / gi_norm)` phá vỡ PPO weights → FPR=1.0
**Sửa:** Thay bằng `clip_updates(max_norm=10.0)` — chỉ clip update quá lớn, giữ nguyên cấu trúc Actor/Critic.

### ✅ Lỗi 2 (ĐÃ SỬA): Personalisation Leakage
**Vấn đề:** Δ = post_train - global_model (chứa θ_k cá nhân hóa cũ)
**Sửa:** Global Start Principle — Δ = post_train - pre_train. Client nhận Global Model sạch trước khi personalise.

### ✅ Lỗi 3 (ĐÃ SỬA): Reward Design Smells
**Vấn đề:** 12+ thành phần reward triệt tiêu lẫn nhau (HHI vs entropy, balanced acc vs macro F1)
**Sửa:** MCC-based reward + Cost-Sensitive (FN penalty × 2.0). Giữ collapse penalty + novelty bonus.

### ✅ Lỗi 4 (ĐÃ SỬA): Meta-Agent Illusion
**Vấn đề:** Meta-Agent accuracy rất cao (>98%) vì nó train trên dữ liệu Local test của Client (nơi Client đã personalise/overfit). Điều này biến Meta-Agent thành một "lớp vỏ bọc" che giấu sự sụp đổ của Global Model.
**Sửa:** Grounding trên Root Dataset — đổi tập train của Meta-Agent sang `X_root, y_root`. Meta-Agent hiện tại bắt buộc phải học cách phối hợp các client sao cho đúng trên một tập dữ liệu toàn cục, khách quan có sẵn ở Server, từ đó truyền tín hiệu Reward chính xác cho Tier-3 Selector để loại bỏ các client overfit/độc hại.
"""
Neural network architectures for PPO actor-critic.

References:
    Schulman et al., "Proximal Policy Optimization Algorithms", arXiv 2017.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical


def init_weights(m: nn.Module, gain: float = 1.0):
    """Orthogonal initialisation (best practice for PPO)."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class CriticNetwork(nn.Module):
    """PPO Critic — estimates state value V(s)."""

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.net.apply(lambda m: init_weights(m, gain=np.sqrt(2)))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)



# ─── CNN-GRU Hybrid Actor (Tier-1) ─────────────────────────────────────────────

class CNNGRUActor(nn.Module):
    """
    CNN-GRU hybrid for UNSW-NB15: Conv1D feature extraction + GRU temporal learning.

    Architecture:
      Input: [batch, seq_len, feature_dim]
      -> permute(0, 2, 1) -> Conv1D over temporal dimension
      -> Multi-scale Conv1D (k=3, k=5)
      -> CBAM attention (channel + spatial)
      -> GRU temporal learning
      -> Mean pooling -> class logits

    Input:  [batch, seq_len, feature_dim]  (supports 2D [batch, features] as fallback)
    Output: class logits [batch, action_dim]
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        seq_len: int = 5,
        dropout: float = 0.15,
        cnn_channels: list = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.action_dim = action_dim

        if cnn_channels is None:
            cnn_channels = [32, 64]

        # ── Conv1D over temporal dimension (seq_len) ─────────────────────────
        # Input: [batch, seq_len, feature_dim]
        # Conv1D: treat feature_dim as channels, seq_len as length
        # Conv1d(in_channels=feature_dim, out_channels=32, kernel_size=3)
        # This slides a 3-timestep window ACROSS TIME for each feature
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,          # feature_dim as channels
            out_channels=cnn_channels[0],   # 32 temporal feature maps
            kernel_size=3,
            padding=1,                       # same length: seq_len
        )
        # GroupNorm(num_groups=1, num_channels=C) ≡ LayerNorm over channels
        # Works with any batch size and any seq_len (including 1) — ideal for RL inference
        self.bn1 = nn.GroupNorm(1, cnn_channels[0])

        self.conv2 = nn.Conv1d(
            in_channels=cnn_channels[0],
            out_channels=cnn_channels[1],    # 64 temporal feature maps
            kernel_size=5,
            padding=2,                       # same length: seq_len
        )
        self.bn2 = nn.GroupNorm(1, cnn_channels[1])

        # ── CBAM Attention (after conv layers, before GRU) ──────────────────
        # Channel attention: "what is meaningful in the temporal feature maps"
        #  MLP shared across AvgPool and MaxPool paths → sigmoid(MLP(avg) + MLP(max))
        c_attn_hidden = max(1, cnn_channels[1] // 4)
        self.channel_mlp = nn.Sequential(
            nn.Linear(cnn_channels[1], c_attn_hidden),
            nn.ReLU(),
            nn.Linear(c_attn_hidden, cnn_channels[1]),
        )

        # Spatial attention: "where to focus in the temporal dimension"
        #  Conv1d(2→1, k=7, p=3) over concatenated AvgPool/MaxPool along TIME → sigmoid
        self.spatial_conv = nn.Conv1d(2, 1, kernel_size=7, padding=3)

        # ── GRU temporal learning ───────────────────────────────────────────
        # Input to GRU: [batch, seq_len, cnn_channels[1]]
        self.gru = nn.GRU(
            input_size=cnn_channels[1],
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

        # ── Output heads ───────────────────────────────────────────────────
        self.logits_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

        # ── Weight initialization ─────────────────────────────────────────
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param, gain=0.01)
                    elif "bias" in name:
                        nn.init.zeros_(param)

        nn.init.orthogonal_(self.logits_head.weight, gain=0.01)
        nn.init.zeros_(self.logits_head.bias)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"  [CNNGRUActor] Parameters: {total_params:,} | Hidden: {hidden_dim} | "
              f"Layers: {num_layers} | seq_len: {seq_len} | Features: {input_dim} | +CBAM")

    def _apply_cbam(self, x: torch.Tensor) -> torch.Tensor:
        """
        CBAM attention: channel attention then spatial attention.

        Args:
            x: [batch, channels, seq_len] — Conv1D output (after bn2, relu)

        Returns:
            [batch, channels, seq_len] — attention-weighted features
        """
        # ── Channel attention ─────────────────────────────────────────────
        # Pool over spatial (seq) dimension → [batch, channels, 1]
        avg_pool = x.mean(dim=2, keepdim=True)               # [batch, channels, 1]
        max_pool = x.max(dim=2, keepdim=True)[0]             # [batch, channels, 1]
        # Shared MLP applied to each pooled view → [batch, channels, 1]
        avg_attn = self.channel_mlp(avg_pool.squeeze(-1)).unsqueeze(-1)
        max_attn = self.channel_mlp(max_pool.squeeze(-1)).unsqueeze(-1)
        channel_attn = torch.sigmoid(avg_attn + max_attn)     # [batch, channels, 1]
        x = x * channel_attn                                  # scale: [batch, channels, seq_len]

        # ── Spatial attention ─────────────────────────────────────────────
        # Pool over channel dimension → [batch, 1, seq_len]
        avg_sp = x.mean(dim=1, keepdim=True)                   # [batch, 1, seq_len]
        max_sp = x.max(dim=1, keepdim=True)[0]                # [batch, 1, seq_len]
        # Concatenate along channel → [batch, 2, seq_len]
        concat = torch.cat([avg_sp, max_sp], dim=1)
        spatial_attn = torch.sigmoid(self.spatial_conv(concat))  # [batch, 1, seq_len]
        x = x * spatial_attn                                    # scale: [batch, channels, seq_len]

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with FIXED temporal Conv1D + CBAM attention.

        Args:
            x: [batch, seq_len, feature_dim] or [batch, feature_dim]

        Returns:
            logits: [batch, action_dim]
        """
        # Handle 2D input: expand to seq_len
        if x.dim() == 2:
            x = x.unsqueeze(1)                       # [batch, 1, feature_dim]
            x = x.expand(-1, self.seq_len, -1)       # [batch, seq_len, feature_dim]

        batch, seq, feat = x.shape

        # ── Conv1D over temporal dimension ──────────────────────────────
        # Permute: [batch, seq, feat] → [batch, feat, seq]
        # Conv1D(in=feat, out=32, k=3) slides over TIME
        x_conv = x.permute(0, 2, 1)                         # [batch, feat, seq]
        x_conv = self.conv1(x_conv)                          # [batch, 32, seq]
        x_conv = self.bn1(x_conv)
        x_conv = torch.relu(x_conv)

        x_conv = self.conv2(x_conv)                          # [batch, 64, seq]
        x_conv = self.bn2(x_conv)
        x_conv = torch.relu(x_conv)

        # ── CBAM: channel then spatial attention ──────────────────────────
        x_conv = self._apply_cbam(x_conv)                     # [batch, 64, seq]

        # Permute back: [batch, 64, seq] → [batch, seq, 64]
        x_conv = x_conv.permute(0, 2, 1)                     # [batch, seq, 64]

        # ── GRU temporal learning ────────────────────────────────────────
        gru_out, _ = self.gru(x_conv)                         # [batch, seq, hidden_dim]
        gru_out = self.layer_norm(gru_out)

        # ── Mean pooling over temporal dimension ──────────────────────────
        # All timesteps contribute equally (no EMA bias toward recent)
        pooled = gru_out.mean(dim=1)                          # [batch, hidden_dim]

        return self.logits_head(pooled)

    def get_distribution(self, x: torch.Tensor) -> Categorical:
        logits = self.forward(x)
        return Categorical(logits=logits)

    def act(self, x: torch.Tensor, deterministic: bool = False):
        """
        Returns (class_index_tensor, log_prob) for PPO buffer storage.

        class_index_tensor: shape [1, 1] LongTensor — the predicted class index.
        This is what PPO agents must return so their buffer stores int class indices.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        dist = self.get_distribution(x)
        if deterministic:
            action_idx = dist.probs.argmax(dim=-1)  # [1]
        else:
            action_idx = dist.sample()              # [1]
        log_prob = dist.log_prob(action_idx).squeeze(-1)  # scalar
        return action_idx, log_prob  # (LongTensor[1], scalar) — NOT one-hot

    def evaluate(self, x: torch.Tensor, action: torch.Tensor):
        """Evaluate with fixed temporal Conv1D (must match forward())."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
            x = x.expand(-1, self.seq_len, -1)

        batch, seq, feat = x.shape

        # Conv1D over temporal dimension
        x_conv = x.permute(0, 2, 1)
        x_conv = self.conv1(x_conv)
        x_conv = self.bn1(x_conv)
        x_conv = torch.relu(x_conv)
        x_conv = self.conv2(x_conv)
        x_conv = self.bn2(x_conv)
        x_conv = torch.relu(x_conv)

        # CBAM attention (must match forward)
        x_conv = self._apply_cbam(x_conv)

        x_conv = x_conv.permute(0, 2, 1)

        # GRU
        gru_out, _ = self.gru(x_conv)
        gru_out = self.layer_norm(gru_out)

        # Mean pooling (must match forward)
        pooled = gru_out.mean(dim=1)

        # Value
        value = self.value_head(pooled).squeeze(-1)

        # Distribution
        dist = self.get_distribution(x)
        if action.dtype in (torch.float32, torch.float64):
            action_idx = action.argmax(dim=-1)
        else:
            action_idx = action
        log_prob = dist.log_prob(action_idx)
        entropy = dist.entropy()
        return log_prob, entropy, value


# ─── Architecture Registry ──────────────────────────────────────────────────────

ARCHITECTURE_REGISTRY = {
    "cnn_gru": CNNGRUActor,
}

DATASET_DEFAULTS = {
    # All datasets use CNNGRUActor (Conv1D over temporal dimension + GRU)
    "nsl_kdd":     {"backbone": "cnn_gru", "seq_len": 1,  "hidden_dim": 128},
    "iomt_2024":   {"backbone": "cnn_gru", "seq_len": 10, "hidden_dim": 128},
    "edge_iiot":   {"backbone": "cnn_gru", "seq_len": 8,  "hidden_dim": 128},
    "unsw_nb15":   {"backbone": "cnn_gru", "seq_len": 5,  "hidden_dim": 128},
    "unified":     {"backbone": "cnn_gru", "seq_len": 8,  "hidden_dim": 128},
}


def build_actor(
    dataset: str,
    input_dim: int,
    action_dim: int,
    override_backbone: str = None,
    **kwargs,
) -> nn.Module:
    """
    Factory: returns CNNGRUActor for all datasets.

    CNNGRUActor: Conv1D over temporal dimension → multi-scale patterns → GRU → mean pool.
    Handles both 2D input [batch, features] AND 3D [batch, seq, features].
    """
    dataset_key = dataset.lower()
    if dataset_key.startswith("iomt"):
        dataset_key = "iomt_2024"
    elif dataset_key.startswith("edge"):
        dataset_key = "edge_iiot"
    elif dataset_key.startswith("nsl"):
        dataset_key = "nsl_kdd"
    elif dataset_key.startswith("unsw"):
        dataset_key = "unsw_nb15"
    elif dataset_key == "unified":
        pass  # already "unified"

    cfg = DATASET_DEFAULTS.get(dataset_key, DATASET_DEFAULTS["edge_iiot"])

    backbone = override_backbone or cfg.get("backbone", "cnn_gru")
    seq_len = kwargs.pop("seq_len", cfg.get("seq_len", 1))
    hidden_dim = kwargs.pop("hidden_dim", cfg.get("hidden_dim", 128))
    num_layers = kwargs.pop("num_layers", cfg.get("num_layers", 2))
    dropout = kwargs.pop("dropout", 0.15)

    cls = ARCHITECTURE_REGISTRY[backbone]

    return cls(
        input_dim, action_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        seq_len=seq_len,
        dropout=dropout,
    )

"""
Export trained CNNGRU-CBAM model to ONNX for FastAPI deployment.
"""

import argparse
import sys
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class _CNNGRUActorONNX(nn.Module):
    def __init__(self, input_dim=13, action_dim=3, hidden_dim=128,
                 num_layers=2, seq_len=8, cnn_channels=None, dropout=0.15):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [32, 64]

        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=cnn_channels[0],
                               kernel_size=3, padding=1)
        self.bn1 = nn.GroupNorm(1, cnn_channels[0])
        self.conv2 = nn.Conv1d(in_channels=cnn_channels[0], out_channels=cnn_channels[1],
                               kernel_size=5, padding=2)
        self.bn2 = nn.GroupNorm(1, cnn_channels[1])

        c_attn_hidden = max(1, cnn_channels[1] // 4)
        self.channel_mlp = nn.Sequential(
            nn.Linear(cnn_channels[1], c_attn_hidden),
            nn.ReLU(),
            nn.Linear(c_attn_hidden, cnn_channels[1]),
        )
        self.spatial_conv = nn.Conv1d(2, 1, kernel_size=7, padding=3)
        self.gru = nn.GRU(
            input_size=cnn_channels[1],
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.logits_head = nn.Linear(hidden_dim, action_dim)

    def _apply_cbam(self, x):
        avg_pool = x.mean(dim=2, keepdim=True)
        max_pool = x.max(dim=2, keepdim=True)[0]
        avg_attn = self.channel_mlp(avg_pool.squeeze(-1)).unsqueeze(-1)
        max_attn = self.channel_mlp(max_pool.squeeze(-1)).unsqueeze(-1)
        channel_attn = torch.sigmoid(avg_attn + max_attn)
        x = x * channel_attn
        avg_sp = x.mean(dim=1, keepdim=True)
        max_sp = x.max(dim=1, keepdim=True)[0]
        concat = torch.cat([avg_sp, max_sp], dim=1)
        spatial_attn = torch.sigmoid(self.spatial_conv(concat))
        x = x * spatial_attn
        return x

    def forward(self, x):
        x_conv = x.permute(0, 2, 1)
        x_conv = torch.relu(self.bn1(self.conv1(x_conv)))
        x_conv = torch.relu(self.bn2(self.conv2(x_conv)))
        x_conv = self._apply_cbam(x_conv)
        x_conv = x_conv.permute(0, 2, 1)
        gru_out, _ = self.gru(x_conv)
        gru_out = self.layer_norm(gru_out)
        pooled = gru_out.mean(dim=1)
        return self.logits_head(pooled)


def load_actor_weights(pt_path, model):
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "actor" in ckpt:
        state = {k.replace("actor.", ""): v for k, v in ckpt["actor"].items()}
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = {
            k.replace("model_state_dict.", "").replace("actor.", ""): v
            for k, v in ckpt["model_state_dict"].items()
        }
    else:
        state = ckpt
    model.load_state_dict(state, strict=False)
    return model


def export(pt_path, output_path, input_dim=13, action_dim=3,
           seq_len=8, hidden_dim=128):
    model = _CNNGRUActorONNX(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        seq_len=seq_len,
        dropout=0.15,
    )
    model = load_actor_weights(pt_path, model)
    model.eval()

    dummy = torch.randn(1, seq_len, input_dim)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Capture torch.onnx export output to avoid Windows encoding issues
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        torch.onnx.export(
            model,
            dummy,
            output_path,
            input_names=["x"],
            output_names=["logits"],
            opset_version=18,
        )
    finally:
        sys.stdout = old_stdout

    size_kb = Path(output_path).stat().st_size // 1024
    sys.stdout.write(f"[OK] ONNX exported: {output_path} ({size_kb} KB)\n")
    sys.stdout.flush()

    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"x": dummy.numpy()})
    sys.stdout.write(f"[VERIFIED] ONNX output shape: {out[0].shape}\n")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="outputs/outputs_nsl_kdd/best_model.pt")
    parser.add_argument("--output", "-o", default="outputs/outputs_nsl_kdd/model.onnx")
    parser.add_argument("--dataset", default="nsl_kdd")
    args = parser.parse_args()

    seq_map = {"edge_iiot": 8, "nsl_kdd": 1, "iomt": 10, "unsw_nb15": 5}
    key = args.dataset.lower().replace("-", "_").replace("_2024", "")
    seq_len = seq_map.get(key, 8)

    export(
        pt_path=args.model,
        output_path=args.output,
        seq_len=seq_len,
    )


if __name__ == "__main__":
    main()

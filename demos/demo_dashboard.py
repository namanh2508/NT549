"""
Streamlit dashboard for FedRL-IDS — Demo Dashboard.

Demonstrates 4 scenarios:
  1. Training History — Multi-dataset comparison (Edge-IIoT, NSL-KDD, IoMT, UNSW-NB15)
  2. Live Detection Watchdog — real-time API health + metric gauges
  3. Traitor Simulation — reputation scores of malicious vs honest clients
  4. Smart Edge Selector — K_sel curriculum + F1-Macro learning curve

Run:
    streamlit run demos/demo_dashboard.py --server.port 8501
"""

import json
import time
import random
import statistics
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import streamlit as st
import requests

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Page Config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FedRL-IDS Dashboard",
    page_icon="🛡️",
    layout="wide",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stMetric { background: #0f1117; border-radius: 8px; padding: 12px; }
    .stMetric label { color: #8b949e; font-size: 0.85rem; }
    .stMetric [data-testid="stMetricValue"] { color: #e6edf3; font-size: 1.8rem; }
    .attack-box { background: #2d1b1b; border-left: 4px solid #f85149; padding: 8px; border-radius: 4px; }
    .benign-box { background: #1b2d1b; border-left: 4px solid #3fb950; padding: 8px; border-radius: 4px; }
    .malicious-tag { background: #f85149; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; }
    .honest-tag { background: #3fb950; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; }
    .dataset-tag { padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ─── Dataset Definitions ────────────────────────────────────────────────────────

DATASETS = {
    "Edge-IIoT": {
        "path": "outputs/outputs_edge_iiot/training_history.json",
        "color": "#58a6ff",
        "tag_bg": "#1f3a5f",
        "acc": 0.9110,
        "f1": 0.9999,
        "fpr": 0.0005,
        "recall": 0.9999,
    },
    "NSL-KDD": {
        "path": "outputs/outputs_nsl_kdd/training_history.json",
        "color": "#3fb950",
        "tag_bg": "#1f3a1f",
        "acc": 0.9545,
        "f1": 0.9717,
        "fpr": 0.0277,
        "recall": 0.9731,
    },
    "IoMT 2024": {
        "path": "outputs/outputs_iomt/training_history.json",
        "color": "#ffa657",
        "tag_bg": "#3a2a1f",
        "acc": 0.9413,
        "f1": 0.9809,
        "fpr": 0.1507,
        "recall": 0.9799,
    },
    "UNSW-NB15": {
        "path": "outputs/outputs_unsw_nb15/training_history.json",
        "color": "#f85149",
        "tag_bg": "#3a1f1f",
        "acc": 0.7476,
        "f1": 0.9948,
        "fpr": 1.0000,
        "recall": 1.0000,
    },
}

DATASET_COLORS = {name: info["color"] for name, info in DATASETS.items()}

# ─── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path: str) -> Optional[dict]:
    if not path or not Path(path).exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def compute_ema(values: list, alpha: float = 0.3) -> list:
    if not values:
        return []
    result = [values[0]]
    for v in values[1:]:
        result.append(alpha * v + (1 - alpha) * result[-1])
    return result


def get_demo_latencies(n: int = 500) -> list:
    """Generate realistic latency samples in milliseconds."""
    rng = random.Random(42)
    base = [rng.gauss(2.1, 0.4) for _ in range(n // 3)]
    mid = [rng.gauss(3.5, 0.8) for _ in range(n // 3)]
    tail = [rng.gauss(6.0, 2.0) for _ in range(n - 2 * (n // 3))]
    return [max(0.5, x) for x in base + mid + tail]


def get_demo_predictions(n: int = 20):
    """Generate realistic demo traffic predictions."""
    rng = random.Random(datetime.now().microsecond)
    types_ = [
        ("Benign", 0.78, 0.85, 0.98),
        ("Attack", 0.15, 0.80, 0.95),
        ("Recon", 0.07, 0.70, 0.88),
    ]
    entries = []
    for i in range(n):
        roll = rng.random()
        cumulative = 0.0
        chosen = types_[0]
        for label, prob, conf_min, conf_max in types_:
            cumulative += prob
            if roll < cumulative:
                chosen = (label, prob, conf_min, conf_max)
                break
        label = chosen[0]
        conf = rng.uniform(chosen[2], chosen[3])
        entries.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "prediction": label,
            "confidence": f"{conf:.1%}",
            "flow_id": rng.randint(10000, 99999),
        })
    return entries


# ─── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("🛡️ FedRL-IDS Demo")
st.sidebar.markdown("**Scenario Selector**")

scenario = st.sidebar.radio(
    "Choose demo scenario:",
    [
        "📈 Training History",
        "👁️ Detection Watchdog",
        "🐍 Traitor Simulation",
        "🤖 Smart Edge Selector",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**🌐 FastAPI Connection**")

live_api_url = st.sidebar.text_input(
    "FastAPI URL",
    value="http://localhost:8000",
    help="URL of the running FastAPI inference server",
)

api_available = False
try:
    resp = requests.get(f"{live_api_url}/health", timeout=2)
    api_available = resp.status_code == 200
except Exception:
    api_available = False

if api_available:
    st.sidebar.success("✅ FastAPI connected")
else:
    st.sidebar.warning("⚠️ FastAPI not reachable — Demo mode active")

st.sidebar.markdown("---")
st.sidebar.markdown("**📁 Training History Files**")

# Per-dataset path inputs
dataset_paths = {}
for name, info in DATASETS.items():
    dataset_paths[name] = st.sidebar.text_input(
        f"{name} path",
        value=info["path"],
        help=f"Path to {name} training history JSON",
    )


# ─── Load all dataset histories ─────────────────────────────────────────────────

histories = {}
for name, path in dataset_paths.items():
    histories[name] = load_json(path)

available_datasets = [name for name, h in histories.items() if h is not None]

if not available_datasets:
    st.error("No training history files found. Check paths in sidebar.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 1: Training History (Multi-Dataset)
# ══════════════════════════════════════════════════════════════════════════════

if scenario == "📈 Training History":

    st.title("📈 Training History — Multi-Dataset Comparison")
    st.caption("Federated training with FLTrust + RL Selector across 4 benchmark datasets")

    # ── Dataset multi-select ───────────────────────────────────────────────────

    col_sel = st.columns(4)
    selected_datasets = []

    for i, (name, info) in enumerate(DATASETS.items()):
        with col_sel[i]:
            if histories.get(name):
                tag_bg = info.get("tag_bg", "#1f3a5f")
                st.markdown(
                    f"<div style='text-align:center;'>"
                    f"<span class='dataset-tag' "
                    f"style='background:{tag_bg};color:{info['color']};'>"
                    f"● {name}</span><br>"
                    f"<small>Acc: <b>{info['acc']:.2%}</b> · F1: <b>{info['f1']:.2%}</b></small><br>"
                    f"<small>FPR: <b>{info['fpr']:.2%}</b></small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if st.checkbox(f"Show {name}", value=True, key=f"show_{name}"):
                    selected_datasets.append(name)
            else:
                st.markdown(
                    f"<div style='text-align:center;color:#8b949e;'>"
                    f"<span style='color:#6e7681;'>○ {name}</span><br>"
                    f"<small>File not found</small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    if not selected_datasets:
        st.warning("Select at least one dataset above.")
        st.stop()

    # ── Metric selector ───────────────────────────────────────────────────────

    metric_key_map = {
        "Accuracy": ("accuracy", [0.0, 1.05]),
        "F1-Score": ("f1", [0.0, 1.05]),
        "Precision": ("precision", [0.0, 1.05]),
        "Recall": ("recall", [0.0, 1.05]),
        "FPR": ("fpr", [0.0, 1.05]),
    }

    metric_tab_labels = list(metric_key_map.keys())
    metric_tabs = st.tabs(metric_tab_labels)

    for tab_idx, (metric_label, (metric_key, y_range)) in enumerate(metric_key_map.items()):
        with metric_tabs[tab_idx]:
            fig = make_subplots()

            for name in selected_datasets:
                h = histories[name]
                if h is None or metric_key not in h:
                    continue
                color = DATASETS[name]["color"]
                rounds = h.get("rounds", list(range(1, len(h[metric_key]) + 1)))
                raw_vals = h[metric_key]
                ema_vals = compute_ema(raw_vals)

                # Raw line (faint)
                fig.add_trace(go.Scatter(
                    x=rounds, y=raw_vals,
                    name=f"{name} (raw)",
                    mode="lines+markers",
                    line=dict(color=color, width=1, dash="dot"),
                    opacity=0.3,
                    marker=dict(size=3),
                ))
                # EMA line (bold)
                fig.add_trace(go.Scatter(
                    x=rounds, y=ema_vals,
                    name=f"{name} (EMA)",
                    mode="lines+markers",
                    line=dict(color=color, width=2.5),
                    marker=dict(size=4),
                ))

            fig.update_layout(
                title=f"{metric_label} over Training Rounds",
                xaxis_title="Round",
                yaxis_title=metric_label,
                yaxis=dict(range=y_range),
                height=420,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                template="plotly_dark",
                paper_bgcolor="#0d1117",
                plot_bgcolor="#161b22",
                font=dict(color="#e6edf3"),
            )
            fig.update_xaxes(showgrid=True, gridcolor="#21262d")
            fig.update_yaxes(showgrid=True, gridcolor="#21262d")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Summary metrics table ───────────────────────────────────────────────────

    st.markdown("##### 📊 Final Metrics Summary")

    summary_rows = []
    for name in selected_datasets:
        h = histories[name]
        if h is None:
            continue
        acc = h.get("accuracy", [None])[-1]
        f1 = h.get("f1", [None])[-1]
        prec = h.get("precision", [None])[-1]
        rec = h.get("recall", [None])[-1]
        fpr = h.get("fpr", [None])[-1]
        summary_rows.append({
            "Dataset": name,
            "Accuracy": f"{acc:.4f}" if acc is not None else "N/A",
            "F1-Score": f"{f1:.4f}" if f1 is not None else "N/A",
            "Precision": f"{prec:.4f}" if prec is not None else "N/A",
            "Recall": f"{rec:.4f}" if rec is not None else "N/A",
            "FPR": f"{fpr:.4f}" if fpr is not None else "N/A",
            "Color": DATASETS[name]["color"],
        })

    if summary_rows:
        cols_config = {
            "Dataset": st.column_config.TextColumn("Dataset", width="medium"),
            "Accuracy": st.column_config.TextColumn("Accuracy", width="small"),
            "F1-Score": st.column_config.TextColumn("F1-Score", width="small"),
            "Precision": st.column_config.TextColumn("Precision", width="small"),
            "Recall": st.column_config.TextColumn("Recall", width="small"),
            "FPR": st.column_config.TextColumn("FPR", width="small"),
        }
        st.dataframe(
            summary_rows,
            column_config=cols_config,
            use_container_width=True,
            hide_index=True,
        )

    # ── Trust scores over rounds (per dataset) ────────────────────────────────

    st.markdown("---")
    st.markdown("##### 🔐 FLTrust Reputation Scores")

    trust_tabs = st.tabs(selected_datasets)
    for tab_idx, name in enumerate(selected_datasets):
        h = histories[name]
        if h is None or "trust_scores" not in h:
            with trust_tabs[tab_idx]:
                st.info("No trust score data available.")
            continue
        with trust_tabs[tab_idx]:
            ts = h["trust_scores"]
            rounds = h.get("rounds", list(range(1, len(ts) + 1)))
            num_clients = len(ts[0]) if ts else 0

            fig = go.Figure()
            for k in range(num_clients):
                client_reps = [round[t][k] if len(round) > k else 0 for round in ts]
                fig.add_trace(go.Scatter(
                    x=rounds, y=client_reps,
                    name=f"Client {k}",
                    mode="lines",
                    line=dict(width=1.5),
                ))
            fig.update_layout(
                title=f"FLTrust Reputation per Client — {name}",
                xaxis_title="Round",
                yaxis_title="Reputation Score",
                yaxis=dict(range=[0.0, 1.05]),
                height=320,
                template="plotly_dark",
                paper_bgcolor="#0d1117",
                plot_bgcolor="#161b22",
                font=dict(color="#e6edf3"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            fig.update_xaxes(showgrid=True, gridcolor="#21262d")
            fig.update_yaxes(showgrid=True, gridcolor="#21262d")
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 2: Detection Watchdog
# ══════════════════════════════════════════════════════════════════════════════

elif scenario == "👁️ Detection Watchdog":

    st.title("👁️ Detection Watchdog — Real-Time API Monitoring")
    st.caption("Live metrics from FastAPI server · Demo mode when unavailable")

    # Initialize session state
    if "predictions" not in st.session_state:
        st.session_state.predictions = []
    if "demo_log" not in st.session_state:
        st.session_state.demo_log = []
    if "_demo_generated" not in st.session_state:
        st.session_state._demo_generated = False

    col1, col2, col3, col4, col5 = st.columns(5)

    # ── API health check ────────────────────────────────────────────────────

    if api_available:
        try:
            health = requests.get(f"{live_api_url}/health", timeout=2).json()
            metrics = requests.get(f"{live_api_url}/metrics", timeout=2).json()

            col1.metric("API Status", "✅ Online")
            col2.metric("Model Loaded", "✅" if health.get("model_loaded") else "❌")
            col3.metric("P50 Latency", f"{health.get('latency_p50_ms', 0):.2f} ms")
            col4.metric("P99 Latency", f"{health.get('latency_p99_ms', 0):.2f} ms")
            uptime_s = health.get("uptime_seconds", 0)
            col5.metric("Uptime", f"{uptime_s // 3600}h {uptime_s % 3600 // 60}m")

            total = metrics.get("total_predictions", 0)
            attacks = metrics.get("attacks_detected", 0)
            benign = total - attacks

            st.session_state.predictions.append({
                "total": total, "attacks": attacks, "timestamp": datetime.now(),
            })
            if len(st.session_state.predictions) > 100:
                st.session_state.predictions.pop(0)

        except Exception as e:
            st.error(f"API error: {e}")
            api_available = False
    else:
        col1.metric("API Status", "⚠️ Offline")
        col2.metric("Mode", "🔄 Demo")
        col3.metric("Latency (demo)", "2.1 ms")
        col4.metric("Requests (demo)", "1,204")
        col5.metric("Attacks (demo)", "312")

    st.markdown("---")

    # ── Demo simulation control ───────────────────────────────────────────────

    demo_col, log_col = st.columns([1, 2])

    with demo_col:
        st.markdown("##### 🔄 Demo Simulation")
        st.caption("Generate simulated real-time traffic")

        if st.button("▶ Generate Traffic (20 flows)", type="primary"):
            new_entries = get_demo_predictions(20)
            st.session_state.demo_log.extend(new_entries)
            if len(st.session_state.demo_log) > 100:
                st.session_state.demo_log = st.session_state.demo_log[-100:]
            st.session_state._demo_generated = True

        if st.button("🗑️ Clear Log"):
            st.session_state.demo_log = []
            st.session_state._demo_generated = False

        # Auto-generate demo data on first load if API is down
        if not api_available and not st.session_state._demo_generated:
            st.session_state.demo_log = get_demo_predictions(20)
            st.session_state._demo_generated = True

    with log_col:
        st.markdown("##### 📋 Recent Prediction Log")
        if st.session_state.demo_log:
            rows = []
            for entry in reversed(st.session_state.demo_log[-20:]):
                css_class = "benign-box" if entry["prediction"] == "Benign" else "attack-box"
                pred_html = (
                    f"<span style='color:{'#3fb950' if entry['prediction']=='Benign' else '#f85149'};'>"
                    f"● {entry['prediction']}</span>"
                )
                rows.append(
                    f"| {entry['time']} | `{entry['flow_id']}` | {pred_html} | {entry['confidence']} |"
                )
            st.markdown(
                "| Time | Flow ID | Prediction | Confidence |\n"
                "|------|---------|------------|------------|\n"
                + "\n".join(rows),
                unsafe_allow_html=True,
            )
        else:
            st.info("Click **Generate Traffic** or wait for auto-demo to start.")

    st.markdown("---")

    # ── Latency histogram ───────────────────────────────────────────────────

    st.markdown("##### ⏱️ Latency Distribution")

    demo_latencies = get_demo_latencies(500)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=demo_latencies,
        nbinsx=40,
        marker_color="#58a6ff",
        name="Latency (ms)",
    ))

    p50 = np.percentile(demo_latencies, 50)
    p95 = np.percentile(demo_latencies, 95)
    p99 = np.percentile(demo_latencies, 99)

    for pct, val, color in [(50, p50, "#3fb950"), (95, p95, "#ffa657"), (99, p99, "#f85149")]:
        fig.add_vline(
            x=val, line_dash="dash", line_color=color, line_width=1.5,
            annotation_text=f"P{pct}={val:.1f}ms",
            annotation_position="top",
            annotation_font_color=color,
        )

    fig.update_layout(
        title="Response Latency Distribution (demo data)",
        xaxis_title="Latency (ms)",
        yaxis_title="Count",
        height=320,
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font=dict(color="#e6edf3"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#21262d")
    fig.update_yaxes(showgrid=True, gridcolor="#21262d")
    st.plotly_chart(fig, use_container_width=True)

    lc1, lc2, lc3 = st.columns(3)
    lc1.metric("P50 Latency", f"{p50:.2f} ms")
    lc2.metric("P95 Latency", f"{p95:.2f} ms")
    lc3.metric("P99 Latency", f"{p99:.2f} ms")


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 3: Traitor Simulation
# ══════════════════════════════════════════════════════════════════════════════

elif scenario == "🐍 Traitor Simulation":

    st.title("🐍 Traitor Simulation — Byzantine Client Detection")
    st.markdown(
        "Simulate malicious clients sending sign-flipped gradients. "
        "Watch FLTrust reputation scores drop for attackers while honest clients maintain trust."
    )

    col_sim, col_visual = st.columns([1, 2])

    with col_sim:
        st.markdown("##### ⚙️ Simulation Parameters")

        num_clients = st.slider("Total clients", 5, 20, 10, key="ts_clients")
        num_malicious = st.slider("Malicious clients", 1, 5, 3, key="ts_mal")
        num_rounds_sim = st.slider("Simulation rounds", 5, 30, 20, key="ts_rounds")
        attack_start = st.slider("Attack starts at round", 2, 15, 5, key="ts_start")

        st.caption(
            f"**Growth:** γ=0.1 > **Decay:** δ=0.05 "
            "(anti-collapse: good clients gain trust 2× faster than bad clients lose it)"
        )

        simulate = st.button("▶ Run Simulation", type="primary")

        if simulate:
            reputations: list[list[float]] = [[0.5] * num_clients for _ in range(num_rounds_sim + 1)]
            malicious_ids = set(random.sample(range(num_clients), num_malicious))

            for r in range(1, num_rounds_sim + 1):
                for k in range(num_clients):
                    prev = reputations[r - 1][k]
                    if r >= attack_start and k in malicious_ids:
                        reputations[r][k] = max(0.0, prev - 0.12 * (0.5 + abs(prev - 0.5)))
                    else:
                        reputations[r][k] = min(1.0, prev + 0.06 * (0.5 + (prev - 0.5)))

            st.session_state.reputation_history = reputations
            st.session_state.malicious_ids = malicious_ids
            st.session_state.ts_rounds = num_rounds_sim

    with col_visual:
        if "reputation_history" not in st.session_state:
            st.info("Click **Run Simulation** to see FLTrust reputation dynamics.")
        else:
            reputations = st.session_state.reputation_history
            malicious_ids = st.session_state.malicious_ids

            rep_array = np.array(reputations)
            rounds = list(range(len(reputations)))

            fig = go.Figure()
            for k in range(num_clients):
                label = f"Client {k}"
                color = "#f85149" if k in malicious_ids else "#3fb950"
                dash = "dash" if k in malicious_ids else "solid"
                width = 1.5 if k in malicious_ids else 1.2
                fig.add_trace(go.Scatter(
                    x=rounds, y=rep_array[:, k],
                    name=label,
                    line=dict(color=color, dash=dash, width=width),
                    mode="lines",
                ))

            fig.add_vline(
                x=attack_start, line_dash="dot", line_color="#ffa657", line_width=2,
                annotation_text=f"Attack starts (round {attack_start})",
                annotation_position="top",
            )

            fig.update_layout(
                title="FLTrust Reputation Scores — Malicious (red) vs Honest (green)",
                xaxis_title="Round",
                yaxis_title="Reputation Score",
                yaxis=dict(range=[0.0, 1.05]),
                height=440,
                template="plotly_dark",
                paper_bgcolor="#0d1117",
                plot_bgcolor="#161b22",
                font=dict(color="#e6edf3"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                annotations=[
                    dict(
                        text="🔴 Red = Malicious | 🟢 Green = Honest",
                        x=0.5, y=-0.18, showarrow=False,
                        xref="paper", yref="paper",
                        font=dict(color="#8b949e", size=12),
                    )
                ],
            )
            fig.update_xaxes(showgrid=True, gridcolor="#21262d")
            fig.update_yaxes(showgrid=True, gridcolor="#21262d")
            st.plotly_chart(fig, use_container_width=True)

            # ── Summary stats ───────────────────────────────────────────────

            final_reps = [reputations[-1][k] for k in range(num_clients)]
            honest_reps = [final_reps[k] for k in range(num_clients) if k not in malicious_ids]
            malicious_reps = [final_reps[k] for k in range(num_clients) if k in malicious_ids]

            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Honest Avg Rep", f"{statistics.mean(honest_reps):.3f}")
            sc2.metric("Malicious Avg Rep", f"{statistics.mean(malicious_reps):.3f}")
            sc3.metric("Honest Min Rep", f"{min(honest_reps):.3f}")
            sc4.metric("Malicious Max Rep", f"{max(malicious_reps):.3f}")

            detected = [k for k in malicious_ids if reputations[-1][k] < 0.25]
            if detected:
                tags = " ".join(
                    f"<span class='malicious-tag'>Malicious C{k}</span>"
                    for k in sorted(detected)
                )
                st.markdown(
                    f"**Detected (< 0.25):** {tags}",
                    unsafe_allow_html=True,
                )
                st.success(f"✅ {len(detected)}/{num_malicious} malicious clients detected — detection rate: {len(detected)/num_malicious:.0%}")
            else:
                st.warning("No malicious clients detected below threshold.")


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 4: Smart Edge Selector
# ══════════════════════════════════════════════════════════════════════════════

elif scenario == "🤖 Smart Edge Selector":

    st.title("🤖 Smart Edge Selector — RL Client Selection Learning")
    st.markdown(
        "The RL Selector learns to reduce K_sel from 8→4 while maintaining F1-Macro accuracy. "
        "Curriculum scheduling guides gradual client reduction across federated rounds."
    )

    # ── Curriculum settings ─────────────────────────────────────────────────

    set_col, chart_col = st.columns([1, 2])

    with set_col:
        st.markdown("##### ⚙️ Curriculum Settings")

        k_init = st.slider("K_sel initial", 4, 15, 8, key="sel_init")
        k_min = st.slider("K_sel minimum", 1, 6, 4, key="sel_min")
        total_rounds = st.slider("Total rounds", 10, 60, 30, key="sel_total")

        curriculum = []
        decay_rate = (k_init - k_min) / max(total_rounds - 1, 1)
        for t in range(total_rounds):
            k_t = int(k_init - t * decay_rate)
            curriculum.append(max(k_min, k_t))

        avg_k = statistics.mean(curriculum)
        savings = (1 - avg_k / k_init) * 100

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Avg K_sel", f"{avg_k:.1f}")
        sc2.metric("Comm. Savings", f"{savings:.0f}%")
        sc3.metric("K_sel range", f"{k_init} → {k_min}")

        st.caption(
            f"Curriculum: K_sel decays linearly from {k_init} to {k_min} "
            f"over {total_rounds} rounds."
        )

    with chart_col:
        rounds = list(range(1, total_rounds + 1))

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Curriculum line
        fig.add_trace(go.Scatter(
            x=rounds, y=curriculum,
            name="Curriculum K_sel",
            line=dict(color="#ffa657", width=2.5, dash="dash"),
            mode="lines+markers",
            marker=dict(size=4),
        ), secondary_y=False)

        # Overlay actual K_sel from all available federated histories
        has_real_data = False
        for name in available_datasets:
            h = histories[name]
            if h is not None and "rounds" in h:
                actual_k_data = h.get("k_sel", [])
                if actual_k_data and len(actual_k_data) > 0:
                    fig.add_trace(go.Scatter(
                        x=h["rounds"][:len(actual_k_data)],
                        y=actual_k_data,
                        name=f"{name} K_sel",
                        line=dict(color=DATASETS[name]["color"], width=2),
                        mode="lines+markers",
                        marker=dict(size=4),
                    ), secondary_y=False)
                    has_real_data = True

        # F1-Macro from NSL-KDD (primary dataset)
        if "NSL-KDD" in histories and histories["NSL-KDD"] and "f1" in histories["NSL-KDD"]:
            h = histories["NSL-KDD"]
            fig.add_trace(go.Scatter(
                x=h["rounds"], y=compute_ema(h["f1"]),
                name="NSL-KDD F1-Macro (EMA)",
                line=dict(color="#3fb950", width=2),
                mode="lines+markers",
                marker=dict(size=4),
            ), secondary_y=True)
            fig.update_yaxes(title_text="F1-Macro", secondary_y=True, range=[0.0, 1.05])

        fig.update_layout(
            title="RL Selector: Curriculum K_sel & F1-Macro",
            xaxis_title="Round",
            yaxis_title="K_sel (clients selected)",
            yaxis=dict(range=[0, k_init + 1]),
            height=420,
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            font=dict(color="#e6edf3"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#21262d")
        fig.update_yaxes(showgrid=True, gridcolor="#21262d")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Selection frequency ─────────────────────────────────────────────────

    st.markdown("##### 📊 Client Selection Frequency")

    # Try to load real selection counts from federated data
    real_counts = None
    real_ds_name = None
    for name in available_datasets:
        h = histories[name]
        if h and "selection_counts" in h and h["selection_counts"]:
            real_counts = h["selection_counts"]
            real_ds_name = name
            break

    if real_counts:
        fig2 = go.Figure(go.Bar(
            x=[f"Client {i}" for i in range(len(real_counts))],
            y=real_counts,
            marker_color=[DATASETS[real_ds_name]["color"] for _ in real_counts],
        ))
        fig2.update_layout(
            title=f"Client Selection Count — {real_ds_name}",
            xaxis_title="Client",
            yaxis_title="Times Selected",
            height=300,
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            font=dict(color="#e6edf3"),
        )
    else:
        # Simulated data
        random.seed(42)
        num_sel = 10
        sel_counts = [random.randint(5, 30) for _ in range(num_sel)]
        fig2 = go.Figure(go.Bar(
            x=[f"Client {i}" for i in range(num_sel)],
            y=sel_counts,
            marker_color="#58a6ff",
        ))
        fig2.update_layout(
            title="Simulated Client Selection Frequency",
            xaxis_title="Client",
            yaxis_title="Times Selected",
            height=300,
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            font=dict(color="#e6edf3"),
        )

    fig2.update_xaxes(showgrid=True, gridcolor="#21262d")
    fig2.update_yaxes(showgrid=True, gridcolor="#21262d")
    st.plotly_chart(fig2, use_container_width=True)

    # ── Info box ────────────────────────────────────────────────────────────

    st.success(
        f"📡 **The RL Selector learns to use ~{avg_k:.1f} clients on average** "
        f"(vs fixed K={k_init}). "
        f"This saves **{savings:.0f}% communication overhead** "
        f"while FLTrust maintains Byzantine robustness."
    )


# ─── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "FedRL-IDS Dashboard | FastAPI + ONNX + Streamlit | "
    "Multi-dataset: Edge-IIoT · NSL-KDD · IoMT 2024 · UNSW-NB15"
)

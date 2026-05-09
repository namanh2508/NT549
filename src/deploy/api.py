"""
FastAPI server for FedRL-IDS — ONNX inference endpoint.

Usage:
    python -m src.deploy.api
    # or with uvicorn:
    uvicorn src.deploy.api:app --host 0.0.0.0 --port 8000
"""

import time
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


# ─── Global State ────────────────────────────────────────────────────────────────

ort_session: Optional[ort.InferenceSession] = None
model_loaded = False
onnx_path: Optional[str] = None
_seq_len: int = 8  # detected from ONNX model at startup

_metrics_lock = threading.Lock()
_total_predictions = 0
_attacks_detected = 0
_latencies: list = []
_start_time = time.time()

CLASS_NAMES = ["Benign", "Attack", "Recon"]

# Feature scaling — must match training preprocessing
FEATURE_MIN = np.array([1, 0, 0.0001, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
FEATURE_MAX = np.array([800, 50000, 60.0, 65535, 65535, 255, 1500.0, 255, 5000.0, 500, 500, 100, 100], dtype=np.float32)


# ─── Pydantic Models ───────────────────────────────────────────────────────────

class FlowInput(BaseModel):
    packet_count: int = Field(..., ge=1)
    byte_count: int = Field(..., ge=0)
    duration: float = Field(..., gt=0)
    src_port: int = Field(0, ge=0, le=65535)
    dst_port: int = Field(0, ge=0, le=65535)
    tcp_flags: int = Field(0, ge=0, le=255)
    rate: float = Field(..., ge=0)
    ttl: int = Field(64, ge=0, le=255)
    avg_iat: float = Field(0, ge=0)
    syn_count: int = Field(0, ge=0)
    ack_count: int = Field(0, ge=0)
    rst_count: int = Field(0, ge=0)
    fin_count: int = Field(0, ge=0)


class PredictRequest(BaseModel):
    flow: FlowInput


class BatchRequest(BaseModel):
    flows: list[FlowInput]


class PredictResponse(BaseModel):
    label: str
    confidence: float
    is_attack: bool
    latency_ms: float


# ─── Lifespan (startup/shutdown) ────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ONNX model at server startup."""
    global ort_session, model_loaded, onnx_path, _seq_len
    default_paths = [
        "outputs/outputs_nsl_kdd/model.onnx",
        "outputs/federated/model.onnx",
        "outputs/nsl_kdd/model.onnx",
    ]
    for p in default_paths:
        if Path(p).exists():
            try:
                ort_session = ort.InferenceSession(
                    p,
                    providers=["CPUExecutionProvider"],
                )
                model_loaded = True
                onnx_path = p
                # Detect seq_len from the ONNX model's first input shape
                _seq_len = ort_session.get_inputs()[0].shape[1]  # e.g., 1 for NSL-KDD, 8 for Edge-IIoT
                print(f"[OK] ONNX loaded from {p}, seq_len={_seq_len}")
                break
            except Exception as e:
                print(f"[ERROR] Failed to load {p}: {e}")
    if not model_loaded:
        print("[WARN] No ONNX model found at startup.")
    yield
    ort_session = None
    model_loaded = False


# ─── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FedRL-IDS API",
    description="Federated RL-based Intrusion Detection — ONNX inference",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Helpers ───────────────────────────────────────────────────────────────────

def preprocess_flow(flow: FlowInput) -> np.ndarray:
    """Normalize FlowInput and tile to [1, seq_len, feature_dim=13]."""
    raw = np.array([[
        flow.packet_count, flow.byte_count, flow.duration,
        flow.src_port, flow.dst_port, flow.tcp_flags,
        flow.rate, flow.ttl, flow.avg_iat,
        flow.syn_count, flow.ack_count, flow.rst_count, flow.fin_count,
    ]], dtype=np.float32)
    scaled = (raw - FEATURE_MIN) / (FEATURE_MAX - FEATURE_MIN + 1e-8)
    seq = np.tile(scaled, (_seq_len, 1))  # [seq_len, 13]
    return seq[np.newaxis, :, :]    # [1, seq_len, 13]


def run_inference(x: np.ndarray) -> tuple[str, float]:
    """Run ONNX inference, return (label, confidence)."""
    global ort_session
    sess = ort_session
    if sess is None:
        raise RuntimeError("ONNX session not initialized")

    input_name = sess.get_inputs()[0].name
    logits = sess.run(None, {input_name: x})[0]  # [1, 3]
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    pred_idx = int(probs.argmax(axis=1)[0])
    confidence = float(probs[0, pred_idx])
    return CLASS_NAMES[pred_idx], confidence


# ─── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    with _metrics_lock:
        p50 = np.percentile(_latencies[-1000:], 50) if _latencies else 0
        p99 = np.percentile(_latencies[-1000:], 99) if _latencies else 0
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "model_path": onnx_path or "",
        "latency_p50_ms": round(p50 * 1000, 2),
        "latency_p99_ms": round(p99 * 1000, 2),
        "uptime_seconds": round(time.time() - _start_time, 1),
    }


@app.get("/metrics")
def metrics():
    global _total_predictions, _attacks_detected
    with _metrics_lock:
        total = _total_predictions
        attacks = _attacks_detected
    return {
        "total_predictions": total,
        "attacks_detected": attacks,
        "benign_detected": total - attacks,
        "attack_rate": round(attacks / total, 4) if total > 0 else 0.0,
        "model_loaded": model_loaded,
    }


@app.post("/predict", response_model=PredictResponse)
def predict_single(req: PredictRequest):
    global _total_predictions, _attacks_detected, _latencies
    if not model_loaded or ort_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        t0 = time.perf_counter()
        x = preprocess_flow(req.flow)
        label, confidence = run_inference(x)
        latency = time.perf_counter() - t0
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {type(e).__name__}: {e}")

    is_attack = label != "Benign"
    with _metrics_lock:
        _total_predictions += 1
        if is_attack:
            _attacks_detected += 1
        _latencies.append(latency)
        if len(_latencies) > 5000:
            _latencies = _latencies[-2000:]

    return PredictResponse(
        label=label,
        confidence=round(confidence, 4),
        is_attack=is_attack,
        latency_ms=round(latency * 1000, 2),
    )


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    global _total_predictions, _attacks_detected, _latencies
    if not model_loaded or ort_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.flows:
        raise HTTPException(status_code=400, detail="Empty flows list")

    t0 = time.perf_counter()
    results = []
    attack_count = 0
    for flow in req.flows:
        x = preprocess_flow(flow)
        label, confidence = run_inference(x)
        results.append({"label": label, "confidence": round(confidence, 4)})
        if label != "Benign":
            attack_count += 1

    total = len(req.flows)
    elapsed = time.perf_counter() - t0

    with _metrics_lock:
        _total_predictions += total
        _attacks_detected += attack_count
        _latencies.append(elapsed)
        if len(_latencies) > 5000:
            _latencies = _latencies[-2000:]

    return {
        "total": total,
        "attacks": attack_count,
        "benign": total - attack_count,
        "processing_ms": round(elapsed * 1000, 2),
        "throughput_per_sec": round(total / elapsed, 1) if elapsed > 0 else 0,
        "results": results,
    }


if __name__ == "__main__":
    import uvicorn, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    uvicorn.run(
        "src.deploy.api:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )

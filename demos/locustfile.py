"""
Locust load testing script for FedRL-IDS FastAPI server.

Usage:
    locust -f locustfile.py --headless \
        --host http://localhost:8000 \
        -u 1000 -r 100 \
        --run-time 60s \
        --csv results/stress_test

This script simulates realistic network traffic hitting the /predict/batch endpoint.
Attack payload mix:
    60% Benign    - normal web browsing patterns
    15% DDoS     - high packet rate, short duration
    10% Port Scan - many ports, low byte count
     8% SQL Injection - special chars in payload
     5% Brute Force - repeated login attempts
     2% Normal variation
"""

import random
import json
import numpy as np
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner

# ─── Attack Payload Generators ────────────────────────────────────────────────

def _rand():
    return random.uniform(0, 1)

def _norm(mean, std):
    return max(0.0, random.gauss(mean, std))

def _uniform(lo, hi):
    return random.uniform(lo, hi)


class BenignFlow:
    """Normal web browsing / API traffic."""

    @staticmethod
    def generate():
        return {
            "flow": {
                "packet_count":  int(_norm(20, 8)),
                "byte_count":    int(_norm(3000, 1200)),
                "duration":      round(_norm(1.5, 0.6), 3),
                "src_port":      random.choice([80, 443, 8080, 3000]),
                "dst_port":      random.choice([80, 443, 5432, 3306]),
                "tcp_flags":     random.choice([16, 17, 18]),   # ACK / PSH+ACK
                "rate":          round(_norm(13.0, 5.0), 2),
                "ttl":           random.choice([64, 128, 255]),
                "avg_iat":       round(_norm(75.0, 30.0), 2),
                "syn_count":     random.randint(0, 1),
                "ack_count":     random.randint(1, 15),
                "rst_count":     0,
                "fin_count":     random.randint(0, 1),
            }
        }


class DDoSFlow:
    """High-volume, short-duration DDoS burst."""

    @staticmethod
    def generate():
        return {
            "flow": {
                "packet_count":  int(_uniform(200, 800)),
                "byte_count":    int(_uniform(5000, 30000)),
                "duration":      round(_uniform(0.1, 0.8), 3),
                "src_port":      random.randint(1024, 65535),
                "dst_port":      random.choice([80, 443, 8080]),
                "tcp_flags":     random.choice([2, 18]),         # SYN / SYN+ACK
                "rate":          round(_uniform(250.0, 1500.0), 2),
                "ttl":           random.randint(64, 128),
                "avg_iat":       round(_uniform(0.1, 2.0), 3),
                "syn_count":     int(_uniform(100, 500)),
                "ack_count":     random.randint(0, 10),
                "rst_count":     random.randint(0, 5),
                "fin_count":     0,
            }
        }


class PortScanFlow:
    """Reconnaissance: many ports, low byte count per port."""

    @staticmethod
    def generate():
        num_ports = random.randint(30, 150)
        return {
            "flow": {
                "packet_count":  num_ports,
                "byte_count":    int(num_ports * _uniform(40, 80)),
                "duration":      round(_uniform(5.0, 20.0), 3),
                "src_port":      random.randint(40000, 60000),
                "dst_port":      0,    # varies per packet; report 0 here
                "tcp_flags":     2,    # SYN only
                "rate":          round(_uniform(3.0, 30.0), 2),
                "ttl":           random.choice([64, 128]),
                "avg_iat":       round(_uniform(30.0, 300.0), 2),
                "syn_count":     num_ports,
                "ack_count":     0,
                "rst_count":     0,
                "fin_count":     0,
            }
        }


class SQLInjectionFlow:
    """Web attack: payload contains SQL special characters."""

    @staticmethod
    def generate():
        chars = random.choice([
            "' OR '1'='1",
            "'; DROP TABLE--",
            "' UNION SELECT",
            "1' AND 'x'='x",
            "admin'--",
        ])
        return {
            "flow": {
                "packet_count":  random.randint(1, 5),
                "byte_count":    int(_uniform(200, 1500)),
                "duration":      round(_uniform(0.2, 1.5), 3),
                "src_port":      random.randint(1024, 65535),
                "dst_port":      random.choice([80, 443, 8080]),
                "tcp_flags":     24,   # PSH+ACK
                "rate":          round(_norm(5.0, 2.0), 2),
                "ttl":           random.choice([64, 128]),
                "avg_iat":       round(_uniform(100.0, 500.0), 2),
                "syn_count":     1,
                "ack_count":     random.randint(1, 3),
                "rst_count":     0,
                "fin_count":     0,
                "payload_hint":  chars,
            }
        }


class BruteForceFlow:
    """Repeated login attempts from same source."""

    @staticmethod
    def generate():
        return {
            "flow": {
                "packet_count":  random.randint(3, 10),
                "byte_count":    int(_uniform(300, 1200)),
                "duration":      round(_uniform(2.0, 10.0), 3),
                "src_port":      random.randint(1024, 65535),
                "dst_port":      random.choice([22, 23, 3389, 8080]),
                "tcp_flags":     16,   # ACK
                "rate":          round(_uniform(0.5, 5.0), 2),
                "ttl":           random.choice([64, 128]),
                "avg_iat":       round(_uniform(200.0, 2000.0), 2),
                "syn_count":     1,
                "ack_count":     random.randint(1, 8),
                "rst_count":     0,
                "fin_count":     0,
            }
        }


# ─── Flow Distribution ───────────────────────────────────────────────────────

ATTACK_FLOWS = [DDoSFlow, PortScanFlow, SQLInjectionFlow, BruteForceFlow]
ATTACK_WEIGHTS = [0.15, 0.10, 0.08, 0.05]   # cumulative = 0.38


def generate_random_flow():
    """Return a JSON payload sampled from the attack mix distribution."""
    r = random.random()
    if r < 0.62:   # 62% Benign
        return BenignFlow.generate()
    elif r < 0.77:  # 15% DDoS
        return DDoSFlow.generate()
    elif r < 0.87:  # 10% Port Scan
        return PortScanFlow.generate()
    elif r < 0.95:  # 8% SQL Injection
        return SQLInjectionFlow.generate()
    else:            # 5% Brute Force
        return BruteForceFlow.generate()


# ─── Latency Metrics Collector ───────────────────────────────────────────────

_latencies: list = []


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    _latencies.append(response_time)
    if len(_latencies) >= 1000:
        _latencies.clear()


# ─── Locust User ─────────────────────────────────────────────────────────────

class IDSUser(HttpUser):
    """
    Simulates a client sending flow payloads to the FastAPI /predict endpoint.

    Two modes are supported:
      - /predict      → single flow per request (realistic HTTP overhead)
      - /predict/batch → batch of 50 flows per request (higher throughput)
    """

    wait_time = between(0.001, 0.01)   # very short delay → high concurrency
    _use_batch = True

    @task(8)
    def predict_batch(self):
        """Primary task: send a batch of 50 flows at once."""
        batch_size = 50
        payload = {"flows": [generate_random_flow()["flow"] for _ in range(batch_size)]}
        self.client.post("/predict/batch", json=payload, name="/predict/batch")

    @task(2)
    def predict_single(self):
        """Secondary task: send one flow per request."""
        payload = generate_random_flow()
        self.client.post("/predict", json=payload, name="/predict")

    @task(1)
    def health_check(self):
        """Lightweight health check every ~10 requests."""
        self.client.get("/health", name="/health")

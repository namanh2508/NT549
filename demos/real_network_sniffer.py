"""
Real-time network packet sniffer for FedRL-IDS deployment.

Captures live packets from a network interface (Wi-Fi / Ethernet), extracts
flow-level features matching the IDS feature schema, and sends them to the
FastAPI /predict endpoint for real-world intrusion detection.

Usage:
    # Requires root/admin privileges for raw socket access
    sudo python real_network_sniffer.py \
        --interface "Wi-Fi" \
        --duration 60 \
        --api_url http://localhost:8000/predict \
        --output results/live_capture.json \
        --interval 5

Requirements:
    pip install scapy httpx plotly streamlit
    # Also requires: npcap (Windows) or libpcap (Linux)
"""

import argparse
import json
import time
import socket
import struct
import statistics
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# scapy: cross-platform packet capture
try:
    from scapy.all import (
        sniff,
        IP,
        TCP,
        UDP,
        ICMP,
        Raw,
        wrpcap,
        get_if_list,
    )
    from scapy.sessions import TCPSession
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("[WARN] scapy not installed. Run: pip install scapy")

# Async HTTP client for non-blocking API calls
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    print("[WARN] httpx not installed. Run: pip install httpx")

# ─── Flow Aggregation ───────────────────────────────────────────────────────────

# Simple 5-tuple flow key
def flow_key(pkt) -> Optional[str]:
    if not pkt.haslayer(IP):
        return None
    src = pkt["IP"].src
    dst = pkt["IP"].dst
    proto = pkt["IP"].proto
    if pkt.haslayer(TCP):
        sport = pkt["TCP"].sport
        dport = pkt["TCP"].dport
    elif pkt.haslayer(UDP):
        sport = pkt["UDP"].sport
        dport = pkt["UDP"].dport
    else:
        sport = dport = 0
    return f"{src}:{sport}>{dst}:{dport}@{proto}"


class FlowCollector:
    """
    Aggregates raw packets into bi-directional network flows.

    Features extracted per flow (matching the IDS schema):
      - packet_count, byte_count, duration
      - src_port, dst_port, tcp_flags
      - rate (packets/second)
      - ttl, avg_iat (inter-arrival time)
      - syn_count, ack_count, rst_count, fin_count
    """

    def __init__(self, inactivity_timeout: float = 30.0):
        self.inactivity_timeout = inactivity_timeout
        # flow_key -> {pkts: [], timestamps: [], bytes: [], start_time: float}
        self.flows: dict = defaultdict(lambda: {"pkts": [], "ts": [], "bytes": [], "start": None})

    def add_packet(self, pkt, timestamp: float):
        key = flow_key(pkt)
        if key is None:
            return

        f = self.flows[key]
        if f["start"] is None:
            f["start"] = timestamp

        f["pkts"].append(pkt)
        f["ts"].append(timestamp)
        f["bytes"].append(len(pkt))

    def extract_features(self, key: str) -> Optional[dict]:
        """Convert a completed flow into an IDS feature vector."""
        f = self.flows[key]
        if len(f["pkts"]) < 2:
            return None

        pkts = f["pkts"]
        ts = f["ts"]
        bytes_list = f["bytes"]
        n = len(pkts)

        duration = max(ts[-1] - ts[0], 0.001)
        packet_count = n
        byte_count = sum(bytes_list)

        # TCP flags
        syn_count = sum(1 for p in pkts if p.haslayer(TCP) and p["TCP"].flags & 0x02)
        ack_count = sum(1 for p in pkts if p.haslayer(TCP) and p["TCP"].flags & 0x10)
        rst_count = sum(1 for p in pkts if p.haslayer(TCP) and p["TCP"].flags & 0x04)
        fin_count = sum(1 for p in pkts if p.haslayer(TCP) and p["TCP"].flags & 0x01)

        # TCP flags as a single integer (sum of flag values)
        tcp_flags = 0
        if pkts[-1].haslayer(TCP):
            tcp_flags = int(pkts[-1]["TCP"].flags)

        # Inter-arrival times
        iats = [ts[i] - ts[i - 1] for i in range(1, n)]
        avg_iat = statistics.mean(iats) if iats else 0.0

        # Rate (packets/second)
        rate = packet_count / duration

        # Ports (from last packet)
        last = pkts[-1]
        if last.haslayer(TCP):
            src_port = last["TCP"].sport
            dst_port = last["TCP"].dport
        elif last.haslayer(UDP):
            src_port = last["UDP"].sport
            dst_port = last["UDP"].dport
        else:
            src_port = dst_port = 0

        # TTL from IP layer
        ttl = pkts[-1]["IP"].ttl if pkts[-1].haslayer(IP) else 64

        return {
            "packet_count": packet_count,
            "byte_count": byte_count,
            "duration": round(duration, 3),
            "src_port": src_port,
            "dst_port": dst_port,
            "tcp_flags": tcp_flags,
            "rate": round(rate, 2),
            "ttl": ttl,
            "avg_iat": round(avg_iat * 1000, 3),   # ms
            "syn_count": syn_count,
            "ack_count": ack_count,
            "rst_count": rst_count,
            "fin_count": fin_count,
        }

    def prune_inactive(self, now: float):
        """Remove flows inactive for > inactivity_timeout seconds."""
        to_delete = []
        for key, f in self.flows.items():
            if f["ts"] and (now - f["ts"][-1]) > self.inactivity_timeout:
                to_delete.append(key)
        for key in to_delete:
            del self.flows[key]
        return to_delete

    def completed_flows(self) -> list:
        """Return feature dicts for all flows that appear complete."""
        return [self.extract_features(k) for k in list(self.flows.keys()) if self.flows[k]["pkts"]]


# ─── Packet Handler ─────────────────────────────────────────────────────────────

def packet_handler(pkt, collector: FlowCollector, timestamp: float):
    """Called by scapy for each captured packet."""
    collector.add_packet(pkt, timestamp)


# ─── API Client ────────────────────────────────────────────────────────────────

async def send_prediction(client: httpx.AsyncClient, api_url: str, flow: dict) -> dict:
    """Send a single flow to the FastAPI /predict endpoint."""
    try:
        resp = await client.post(api_url, json={"flow": flow}, timeout=5.0)
        return {"status": "ok", "response": resp.json(), "flow": flow}
    except Exception as e:
        return {"status": "error", "error": str(e), "flow": flow}


# ─── Main Sniffer ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Real-time network sniffer → FastAPI IDS prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--interface", "-i",
        default="Wi-Fi",
        help="Network interface name (e.g. 'Wi-Fi', 'eth0', 'en0'). "
             "Use --list-interfaces to see available interfaces.",
    )
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=60,
        help="Capture duration in seconds (0 = infinite).",
    )
    parser.add_argument(
        "--api_url",
        default="http://localhost:8000/predict",
        help="FastAPI predict endpoint URL.",
    )
    parser.add_argument(
        "--output", "-o",
        default="results/live_capture.json",
        help="Output JSON file for captured flows + predictions.",
    )
    parser.add_argument(
        "--interval", "-t",
        type=int,
        default=5,
        help="Interval (seconds) between batch API calls.",
    )
    parser.add_argument(
        "--list-interfaces", "-L",
        action="store_true",
        help="List available network interfaces and exit.",
    )
    parser.add_argument(
        "--pcap",
        default=None,
        help="Optional PCAP file to write captured packets to.",
    )
    parser.add_argument(
        "--filter",
        default="tcp or udp or icmp",
        help="BPF filter expression (e.g. 'tcp port 80').",
    )
    args = parser.parse_args()

    if not SCAPY_AVAILABLE:
        print("[ERROR] scapy is required. Install: pip install scapy")
        return

    # ── List interfaces ─────────────────────────────────────────────────────

    if args.list_interfaces:
        print("Available network interfaces:")
        for iface in get_if_list():
            print(f"  - {iface}")
        return

    # ── Setup ───────────────────────────────────────────────────────────────

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    collector = FlowCollector(inactivity_timeout=30.0)

    captured: list = []
    running = True
    start_time = time.time()

    print(f"[*] Starting sniffer on interface: {args.interface}")
    print(f"[*] Capture duration: {args.duration}s (0 = infinite)")
    print(f"[*] API endpoint: {args.api_url}")
    print(f"[*] Output file: {args.output}")
    print(f"[*] BPF filter: {args.filter}")

    # ── Capture loop ────────────────────────────────────────────────────────

    def capture_thread():
        """Runs in a separate thread; scapy blocks."""
        try:
            sniff(
                iface=args.interface,
                filter=args.filter,
                prn=lambda pkt: packet_handler(pkt, collector, time.time()),
                store=False,
                timeout=args.duration if args.duration > 0 else None,
                stop_filter=lambda _: not running,
            )
        except PermissionError:
            print("[ERROR] Need root/admin privileges. Run with: sudo python ...")
        except OSError as e:
            print(f"[ERROR] Interface '{args.interface}' not found: {e}")
            print("Run with --list-interfaces to see available interfaces.")

    # Start capture in background thread
    cap_thread = threading.Thread(target=capture_thread, daemon=True)
    cap_thread.start()

    # ── Processing loop ─────────────────────────────────────────────────────

    interval = args.interval
    last_batch_time = start_time

    while running:
        time.sleep(1.0)
        elapsed = time.time() - start_time

        if args.duration > 0 and elapsed >= args.duration:
            running = False
            break

        # Prune inactive flows
        collector.prune_inactive(time.time())

        # Send batch to API at interval
        if time.time() - last_batch_time >= interval:
            flows = collector.completed_flows()
            if flows and HTTPX_AVAILABLE:
                import asyncio
                try:
                    async def batch_send():
                        async with httpx.AsyncClient() as client:
                            tasks = [
                                send_prediction(client, args.api_url, f)
                                for f in flows
                            ]
                            results = await asyncio.gather(*tasks, return_exceptions=True)
                            return results
                    api_results = asyncio.run(batch_send())
                    for res in api_results:
                        if isinstance(res, dict):
                            captured.append(res)
                except Exception as e:
                    print(f"[WARN] API call failed: {e}")
                    for f in flows:
                        captured.append({"status": "skipped", "flow": f})
            else:
                for f in flows:
                    captured.append({"status": "skipped_no_httpx", "flow": f})

            collector.flows.clear()
            last_batch_time = time.time()

            # Progress
            total_flows = len(captured)
            attacks = sum(1 for c in captured if isinstance(c, dict) and c.get("response", {}).get("is_attack"))
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"Elapsed: {int(elapsed)}s | "
                f"Flows captured: {total_flows} | "
                f"Attacks detected: {attacks}"
            )

    cap_thread.join(timeout=5.0)

    # ── Final report ────────────────────────────────────────────────────────

    total = len(captured)
    attacks = sum(
        1 for c in captured
        if isinstance(c, dict)
        and c.get("response", {}).get("is_attack", False)
    )
    benign = total - attacks

    summary = {
        "capture_time": datetime.now().isoformat(),
        "duration_seconds": time.time() - start_time,
        "interface": args.interface,
        "total_flows": total,
        "attacks_detected": attacks,
        "benign_flows": benign,
        "attack_rate": round(attacks / total, 4) if total > 0 else 0.0,
        "results": captured,
    }

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[✓] Capture complete. {total} flows, {attacks} attacks detected.")
    print(f"[✓] Results saved to: {args.output}")
    print(f"[✓] Attack rate: {summary['attack_rate']:.2%}")

    # ── Write PCAP ───────────────────────────────────────────────────────────

    if args.pcap:
        print(f"[*] Note: PCAP writing requires storing packets during capture.")
        print(f"    Pass 'store=True' to sniff() to enable --pcap output.")

    return summary


if __name__ == "__main__":
    main()

"""
Standalone Traitor Simulation CLI for FedRL-IDS.

Simulates Byzantine (malicious) clients sending corrupted gradients in a federated
round. Computes FLTrust reputation dynamics and outputs results as JSON.

Usage:
    python demo_traitor_simulation.py \
        --num_clients 10 \
        --malicious_clients 3 \
        --rounds 20 \
        --attack_start 5 \
        --output results/traitor_simulation.json

This script is the CLI counterpart of the "Traitor Simulation" tab in demo_dashboard.py.
Both share the same reputation dynamics model.
"""

import argparse
import json
import random
from pathlib import Path


def simulate_reputations(
    num_clients: int,
    num_malicious: int,
    num_rounds: int,
    attack_start: int,
    growth: float = 0.06,
    decay: float = 0.12,
    seed: int = 42,
) -> dict:
    """
    Simulate FLTrust temporal reputation dynamics under Byzantine attack.

    Args:
        num_clients: Total number of federated clients
        num_malicious: Number of clients that turn malicious at attack_start
        num_rounds: Total number of federated rounds to simulate
        attack_start: Round at which malicious clients begin their attack
        growth: Reputation growth rate for honest clients
        decay: Reputation decay rate for malicious clients
        seed: Random seed for reproducibility

    Returns:
        dict with 'reputations', 'malicious_ids', 'rounds', 'stats'
    """
    random.seed(seed)
    malicious_ids = set(random.sample(range(num_clients), num_malicious))

    # Initialize reputations at neutral 0.5
    reputations = [0.5] * num_clients
    history = [list(reputations)]  # round 0

    for r in range(1, num_rounds + 1):
        new_reps = []
        for k in range(num_clients):
            prev = reputations[k]
            if r >= attack_start and k in malicious_ids:
                # Byzantine attack: negative cosine contribution
                # Decay rate proportional to how far from 0.5
                decay_amount = decay * (0.5 + abs(prev - 0.5))
                new_r = max(0.0, prev - decay_amount)
            else:
                # Honest: proportional growth
                growth_amount = growth * (0.5 + (prev - 0.5))
                new_r = min(1.0, prev + growth_amount)
            new_reps.append(round(new_r, 4))
        reputations = new_reps
        history.append(list(reputations))

    # Compute statistics
    honest_ids = [k for k in range(num_clients) if k not in malicious_ids]
    final_reps = history[-1]
    honest_final = [final_reps[k] for k in honest_ids]
    malicious_final = [final_reps[k] for k in malicious_ids]

    detection_threshold = 0.25
    detected = [k for k in malicious_ids if final_reps[k] < detection_threshold]

    return {
        "config": {
            "num_clients": num_clients,
            "num_malicious": num_malicious,
            "malicious_ids": sorted(malicious_ids),
            "num_rounds": num_rounds,
            "attack_start_round": attack_start,
            "growth_rate": growth,
            "decay_rate": decay,
            "detection_threshold": detection_threshold,
            "seed": seed,
        },
        "rounds": list(range(num_rounds + 1)),
        "reputations": history,
        "final_reputations": final_reps,
        "stats": {
            "honest_mean_final": round(sum(honest_final) / len(honest_final), 4),
            "honest_min_final": round(min(honest_final), 4),
            "honest_max_final": round(max(honest_final), 4),
            "malicious_mean_final": round(sum(malicious_final) / len(malicious_final), 4),
            "malicious_min_final": round(min(malicious_final), 4),
            "malicious_max_final": round(max(malicious_final), 4),
            "detected_clients": sorted(detected),
            "detection_rate": round(len(detected) / num_malicious, 4) if num_malicious > 0 else 0.0,
        },
    }


def print_summary(result: dict):
    cfg = result["config"]
    stats = result["stats"]
    fr = result["final_reputations"]

    print(f"\n{'='*60}")
    print(f"  FedRL-IDS Traitor Simulation Summary")
    print(f"{'='*60}")
    print(f"  Clients:          {cfg['num_clients']} total, "
          f"{cfg['num_malicious']} malicious (IDs: {cfg['malicious_ids']})")
    print(f"  Rounds:           {cfg['num_rounds']} (attack starts at round {cfg['attack_start_round']})")
    print(f"  Growth/Decay:    {cfg['growth_rate']} / {cfg['decay_rate']}")
    print()
    print(f"  Final Reputation Scores:")
    print(f"  {'Client':<10} {'Reputation':<12} {'Status'}")
    print(f"  {'-'*40}")
    for k in range(cfg["num_clients"]):
        status = "MALICIOUS" if k in cfg["malicious_ids"] else "Honest"
        rep = fr[k]
        bar = "█" * int(rep * 20) + "░" * (20 - int(rep * 20))
        print(f"  Client {k:<3} {bar} {rep:.4f}  [{status}]")

    print()
    print(f"  Honest clients  — mean: {stats['honest_mean_final']:.4f}, "
          f"min: {stats['honest_min_final']:.4f}, max: {stats['honest_max_final']:.4f}")
    print(f"  Malicious clients — mean: {stats['malicious_mean_final']:.4f}, "
          f"min: {stats['malicious_min_final']:.4f}, max: {stats['malicious_max_final']:.4f}")
    print()
    print(f"  Detected: {stats['detected_clients']} "
          f"(threshold < {cfg['detection_threshold']}, "
          f"rate: {stats['detection_rate']:.0%})")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Simulate Byzantine (malicious) client detection with FLTrust reputation.",
    )
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--malicious_clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--attack_start", type=int, default=5)
    parser.add_argument("--growth", type=float, default=0.06)
    parser.add_argument("--decay", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", "-o", default="results/traitor_simulation.json")
    args = parser.parse_args()

    result = simulate_reputations(
        num_clients=args.num_clients,
        num_malicious=args.malicious_clients,
        num_rounds=args.rounds,
        attack_start=args.attack_start,
        growth=args.growth,
        decay=args.decay,
        seed=args.seed,
    )

    print_summary(result)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[+] Results saved to: {args.output}")


if __name__ == "__main__":
    main()

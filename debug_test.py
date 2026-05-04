"""
FedRL-IDS Debug Test: 10-Round Sanity Check
============================================
Wraps the existing run_training() with debug-level per-round logging.
"""

import os, sys, time, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.train import run_training


def run_debug(cfg, num_rounds=10):
    """Run with debug logging wrapper."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  FedRL-IDS DEBUG TEST")
    print(f"  Device: {dev}")
    print(f"  Dataset: {cfg.training.dataset}")
    print(f"  Clients: {cfg.training.num_clients}")
    print(f"  Rounds: {num_rounds}")
    print(f"  Selector: {cfg.training.client_selection_enabled}")
    print(f"  Meta-Agent: {cfg.training.meta_agent_enabled}")
    print(f"{'='*60}")

    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    t_start = time.time()
    history, summary = run_training(cfg)
    total_time = time.time() - t_start

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETED in {total_time:.1f}s")
    print(f"{'='*60}")

    # Per-round results
    rounds = history["rounds"]
    accs = history["accuracy"]
    f1s = history["f1"]
    fprs = history["fpr"]
    selected = history.get("selected_clients", [])
    trusts = history.get("trust_scores", [])

    print("\n--- Per-Round Results ---")
    for i, r in enumerate(rounds):
        print(f"  R{r:2d}: Acc={accs[i]:.4f} F1={f1s[i]:.4f} FPR={fprs[i]:.4f}"
              + (f" Sel={selected[i]}" if i < len(selected) else "")
              + (f" Trust={[f'{t:.3f}' for t in trusts[i]]}" if i < len(trusts) else ""))

    # Client selection frequency
    if selected:
        print("\n--- Client Selection Frequency ---")
        sel_cnt = [0] * cfg.training.num_clients
        for s in selected:
            for c in s:
                sel_cnt[c] += 1
        for cid, cnt in enumerate(sel_cnt):
            print(f"  Client {cid}: {cnt}/{len(selected)} rounds selected")

    # Sanity checks
    print(f"\n{'='*60}")
    print(f"  SANITY CHECKS")
    print(f"{'='*60}")

    accs_arr = np.array(accs)
    issues = []

    if len(accs_arr) < num_rounds:
        issues.append(f"Only {len(accs_arr)}/{num_rounds} rounds completed")
    if np.isnan(accs_arr).any():
        issues.append("NaN found in accuracy history")
    if np.isinf(accs_arr).any():
        issues.append("Inf found in accuracy history")
    if max(accs_arr) - min(accs_arr) < 0.01:
        issues.append("Accuracy essentially constant")
    if accs_arr[-1] < 0.30:
        issues.append(f"Final accuracy too low: {accs_arr[-1]:.4f}")

    if selected and len(set(tuple(sorted(s)) for s in selected)) == 1:
        issues.append("Selector always chose the same clients")

    if trusts:
        all_t = [t for ts in trusts for t in ts]
        if max(all_t) - min(all_t) < 0.01:
            issues.append("Trust scores do not differentiate clients")

    # Numerical stability check
    for i, a in enumerate(accs):
        if np.isnan(a) or np.isinf(a):
            issues.append(f"Round {i+1}: NaN/Inf accuracy")
            break

    if issues:
        print("  ISSUES FOUND:")
        for iss in issues:
            print(f"    - {iss}")
    else:
        print("  [PASS] All sanity checks passed")

    print(f"\n  Best Accuracy: {max(accs_arr):.4f}")
    print(f"  Final Accuracy: {accs_arr[-1]:.4f}")
    print(f"  Total Time: {total_time:.1f}s")
    print(f"\n{'='*60}")
    if not issues:
        print("  RESULT: READY for extended training runs")
    else:
        print("  RESULT: Issues found - fix before extended runs")
    print(f"{'='*60}")

    return history, issues


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="iomt", choices=["iomt","nsl_kdd","edge_iiot","unsw_nb15"])
    p.add_argument("--num_rounds", type=int, default=10)
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    cfg = Config()
    cfg.training.dataset = args.dataset
    cfg.training.num_clients = args.num_clients
    cfg.training.num_rounds = args.num_rounds
    cfg.training.device = args.device
    cfg.training.clients_per_round = min(args.num_clients, 8)
    cfg.training.client_selection_enabled = True
    cfg.training.meta_agent_enabled = True
    cfg.training.meta_agent_eval_interval = 5
    cfg.training.seed = 42

    run_debug(cfg, args.num_rounds)

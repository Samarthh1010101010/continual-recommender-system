"""
run_all_experiments.py

Master script that runs all four models and the full lambda sweep
in the correct order. Results land in results/ with consistent naming.

Usage:
    python run_all_experiments.py               # full experiment suite (10 epochs, 3 seeds)
    python run_all_experiments.py --quick       # 3 epochs, 1 seed (debugging)
    python run_all_experiments.py --test        # 5 epochs, 1 seed, lambda=10 only (quick validation)
    python run_all_experiments.py --lambda_only # only hybrid lambda sweep
"""

import argparse
import subprocess
import sys
import os


def run(cmd: list, label: str) -> None:
    print(f"\n{'#'*60}")
    print(f"# {label}")
    print(f"{'#'*60}")
    print("CMD:", " ".join(cmd))
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        print(f"FAILED: {label}")
        sys.exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick",       action="store_true",
                   help="Use 3 epochs and 1 seed for fast debugging")
    p.add_argument("--test",        action="store_true",
                   help="Test mode: 5 epochs, lambda=10 only, baseline+replay+ewc+hybrid")
    p.add_argument("--lambda_only", action="store_true",
                   help="Run only the hybrid lambda sweep")
    p.add_argument("--seed",        type=int, default=42)
    args = p.parse_args()

    if args.test:
        epochs = 5
        seeds = [42]
        test_lambdas = [10.0]
        run_baseline = True
        run_replay = True
    else:
        epochs = 3 if args.quick else 10
        seeds  = [42] if args.quick else [42, 123, 456]
        test_lambdas = [0.0, 0.1, 1.0, 10.0, 100.0]
        run_baseline = not args.lambda_only
        run_replay = not args.lambda_only

    python = sys.executable

    os.makedirs("results", exist_ok=True)

    if run_baseline:
        # 1. Baseline
        for seed in seeds:
            run(
                [python, "training/train_baseline.py",
                 "--seed",       str(seed),
                 "--num_epochs", str(epochs)],
                f"Baseline  seed={seed}"
            )

    if run_replay:
        # 2. Replay only
        for seed in seeds:
            run(
                [python, "training/train_replay.py",
                 "--seed",       str(seed),
                 "--num_epochs", str(epochs)],
                f"Replay  seed={seed}"
            )

    if not args.lambda_only:
        # 3. EWC only - lambda sweep
        for lam in test_lambdas:
            if lam == 0.0:
                continue  # lambda=0 EWC == baseline; skip
            for seed in seeds:
                run(
                    [python, "training/train_ewc.py",
                     "--seed",       str(seed),
                     "--num_epochs", str(epochs),
                     "--lambda_ewc", str(lam)],
                    f"EWC  lambda={lam}  seed={seed}"
                )

    # 4. Hybrid - lambda sweep
    for lam in test_lambdas:
        for seed in seeds:
            run(
                [python, "training/train_hybrid.py",
                 "--seed",       str(seed),
                 "--num_epochs", str(epochs),
                 "--lambda_ewc", str(lam)],
                f"Hybrid  lambda={lam}  seed={seed}"
            )

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print("Results are in results/")
    print("="*60)


if __name__ == "__main__":
    main()

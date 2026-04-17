"""
training/train_baseline.py

Baseline: plain NCF trained sequentially on each task chunk.
No EWC, no replay. This is the "catastrophic forgetting" control condition.

Training data scope:  ONLY current chunk (chunk == t).
Negative sampling:    Pre-sampled into task_t.pt by data pipeline.
Evaluation protocol:  1-positive-vs-99-negatives per user (NCF benchmark).
"""

import argparse
import os
import sys

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.ncf import NCF
from evaluation.metrics import evaluate_model
from training.utils import (
    set_seed,
    verify_label_distribution,
    ForgettingTracker,
    assert_prediction_label_shape,
)


# CLI

def parse_args():
    p = argparse.ArgumentParser(description="Baseline NCF (no continual learning)")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--num_tasks",   type=int,   default=5)
    p.add_argument("--num_epochs",  type=int,   default=10)
    p.add_argument("--batch_size",  type=int,   default=256)
    p.add_argument("--lr",          type=float, default=0.001)
    p.add_argument("--data_dir",    type=str,   default="fixed_tasks")
    p.add_argument("--results_dir", type=str,   default="results")
    return p.parse_args()


# Main

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: seed={args.seed}, epochs={args.num_epochs}, "
          f"batch={args.batch_size}, lr={args.lr}")

    base_dir    = os.path.dirname(os.path.abspath(__file__))
    data_dir    = os.path.join(base_dir, "..", args.data_dir)
    model_dir   = os.path.join(base_dir, "..", "saved_models", "baseline")
    results_dir = os.path.join(base_dir, "..", args.results_dir)
    os.makedirs(model_dir,   exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Load meta
    meta      = torch.load(os.path.join(data_dir, "meta.pt"), weights_only=True)
    num_users = meta["num_users"]
    num_items = meta["num_items"]
    print(f"Dataset: {num_users} users, {num_items} items")

    # Model, optimizer, loss - fresh for the whole experiment
    model     = NCF(num_users, num_items).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    tracker    = ForgettingTracker()
    all_rows   = []          # per-task evaluation rows for CSV

    for t in range(args.num_tasks):
        print(f"\n{'='*50}")
        print(f"TASK {t}  (baseline - current chunk only)")
        print(f"{'='*50}")

        # Load current task's training data
        task_data  = torch.load(os.path.join(data_dir, f"task_{t}.pt"), weights_only=True)
        train_data = task_data["train"]
        verify_label_distribution(train_data, t)

        user  = train_data["user"]   # LongTensor [N]
        item  = train_data["item"]   # LongTensor [N]
        label = train_data["label"]  # FloatTensor [N]

        dataset = TensorDataset(user, item, label)
        loader_gen = torch.Generator().manual_seed(args.seed + t)
        loader  = DataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                     drop_last=False,
                     generator=loader_gen)

        # Train
        model.train()
        for epoch in range(args.num_epochs):
            epoch_loss = 0.0
            for b_u, b_i, b_l in loader:
                b_u = b_u.to(device)
                b_i = b_i.to(device)
                b_l = b_l.to(device)

                preds = model(b_u, b_i).squeeze(-1)
                assert_prediction_label_shape(preds, b_l, context=f"baseline task={t} epoch={epoch+1}")
                loss  = criterion(preds, b_l)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(b_u)

            avg_loss = epoch_loss / len(dataset)
            print(f"  Epoch {epoch+1}/{args.num_epochs}  loss={avg_loss:.4f}")

        # Evaluate on all tasks seen so far
        model.eval()
        current_results = {}

        with torch.no_grad():
            for k in range(t + 1):
                test_data = torch.load(os.path.join(data_dir, f"task_{k}.pt"), weights_only=True)["test"]
                recall, ndcg = evaluate_model(model, test_data, device)
                current_results[k] = (recall, ndcg)
                print(f"  Eval task {k}: Recall@10={recall:.4f}  NDCG@10={ndcg:.4f}")

                # Update tracker AFTER computing forgetting, BEFORE storing new value
                # (tracker.compute_forgetting uses history BEFORE this task's results)
                all_rows.append({
                    "trained_up_to_task": t,
                    "eval_task":          k,
                    "model":              "baseline",
                    "seed":               args.seed,
                    "recall@10":          recall,
                    "ndcg@10":            ndcg,
                })

        # Forgetting
        avg_f_recall, avg_f_ndcg = tracker.compute_forgetting(current_results)
        print(f"  Avg forgetting - Recall: {avg_f_recall:.4f}  NDCG: {avg_f_ndcg:.4f}")

        # Update tracker history with this round's results
        for k, (recall, ndcg) in current_results.items():
            tracker.update(k, recall, ndcg)

        # Tag the aggregate row
        all_rows.append({
            "trained_up_to_task": t,
            "eval_task":          "avg",
            "model":              "baseline",
            "seed":               args.seed,
            "recall@10":          sum(r for r, n in current_results.values()) / len(current_results),
            "ndcg@10":            sum(n for r, n in current_results.values()) / len(current_results),
            "forgetting_recall":  avg_f_recall,
            "forgetting_ndcg":    avg_f_ndcg,
        })

        # Save model checkpoint
        torch.save(model.state_dict(),
                   os.path.join(model_dir, f"baseline_task_{t}.pt"))

    # Save results
    out_path = os.path.join(results_dir, f"baseline_seed{args.seed}.csv")
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print("Baseline training complete.")


if __name__ == "__main__":
    main()
